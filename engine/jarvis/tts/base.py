"""Text-to-speech interface."""

from __future__ import annotations

import re

import numpy as np

# Split on sentence enders, including the Devanagari full stop (danda).
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+|\n+")


class Speaker:
    name: str = "base"
    available: bool = False

    def synthesize(self, text: str, language: str = "en") -> tuple[np.ndarray, int]:
        """Return (mono float32 audio, sample_rate)."""
        raise NotImplementedError

    def speak(self, text: str, language: str = "en") -> bool:
        """Synthesize and play. Returns False if interrupted."""
        raise NotImplementedError

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class NullSpeaker(Speaker):
    """Text-only mode. Used by --text and when TTS is disabled."""

    name = "none"
    available = True

    def synthesize(self, text: str, language: str = "en") -> tuple[np.ndarray, int]:
        return np.zeros(0, dtype=np.float32), 16000

    def speak(self, text: str, language: str = "en") -> bool:
        return True


def chunk_text(text: str, max_chars: int = 220,
               first_min_chars: int = 24) -> list[str]:
    """Split into speakable chunks at sentence boundaries.

    Chunking lets playback of the first sentence start while the rest is still
    being synthesized, which is the difference between a reply that feels
    instant and one that feels laggy.

    **The first chunk ends at the first sentence**, and that asymmetry is the
    whole trick. This used to return the text unsplit whenever it fit in
    `max_chars`, so an ordinary two-sentence reply was a single chunk and
    nothing played until every word of it had been synthesised — the exact
    behaviour the paragraph above says it avoids. Only the *first* chunk has
    to be small: once audio is playing, synthesis of the rest runs ahead of
    the speaking, and larger chunks there mean fewer round trips and prosody
    that isn't chopped up mid-thought.

    `first_min_chars` stops that becoming silly in the other direction. A
    first sentence of "Done." is not worth its own websocket connection, so
    short openers keep absorbing sentences until they are long enough to buy
    back the round trip they cost.
    """
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif not chunks:
            # Still building the opener: take another sentence only while it
            # is too short to be worth a round trip of its own.
            if len(current) < first_min_chars:
                current = f"{current} {sentence}"
            else:
                chunks.append(current)
                current = sentence
        elif len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence

        # A single sentence longer than the cap gets split on commas.
        while len(current) > max_chars:
            cut = current.rfind(",", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            chunks.append(current[:cut].strip())
            current = current[cut:].strip()

    if current:
        chunks.append(current)
    return chunks
