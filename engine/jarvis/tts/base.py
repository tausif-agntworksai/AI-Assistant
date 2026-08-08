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


def chunk_text(text: str, max_chars: int = 220) -> list[str]:
    """Split into speakable chunks at sentence boundaries.

    Chunking lets playback of the first sentence start while the rest is still
    being synthesized, which is the difference between a reply that feels
    instant and one that feels laggy.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if not current:
            current = sentence
        elif len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence

        # A single sentence longer than the limit gets split on commas.
        while len(current) > max_chars:
            cut = current.rfind(",", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            chunks.append(current[:cut].strip())
            current = current[cut:].strip()

    if current:
        chunks.append(current)
    return chunks
