"""Speech-to-text interface.

Everything downstream depends only on this shape, so swapping local Whisper for
a cloud transcriber is a config change rather than a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Transcript:
    text: str
    language: str = "en"
    language_probability: float = 0.0
    audio_duration: float = 0.0
    latency: float = 0.0
    backend: str = ""
    segments: list[dict] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def is_hindi(self) -> bool:
        return self.language.lower().startswith("hi")

    def __str__(self) -> str:
        return f"[{self.language} {self.language_probability:.2f}] {self.text!r}"


class Transcriber:
    name: str = "base"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        raise NotImplementedError

    def warmup(self) -> None:
        """Pay the model-load cost up front instead of on the first command."""

    def close(self) -> None:
        pass


# Whisper reliably emits these on silence or noise. Treating them as speech
# would make the assistant respond to a cough or a door closing.
HALLUCINATION_PHRASES = {
    "thank you.", "thank you", "thanks for watching!", "thanks for watching",
    "you", "bye.", "bye", ".", "...", "subscribe!", "please subscribe",
    "शुक्रिया", "धन्यवाद", "धन्यवाद।", "आप", "जी", "हाँ",
    "मैं", "okay", "ok", "oh", "hmm", "mm", "uh", "ah",
    "Thank you for watching.", "[music]", "[Music]", "(music)",
}


def is_probable_hallucination(text: str) -> bool:
    stripped = text.strip().lower()
    return not stripped or stripped in {p.lower() for p in HALLUCINATION_PHRASES}


def is_repetition_loop(text: str) -> bool:
    """Detect Whisper's degenerate repeat output.

    On unclear audio the decoder can lock into emitting one token forever
    ("ॐ ॐ ॐ ॐ …"). It looks like a confident transcript, so nothing downstream
    would catch it — but no real command repeats one word dozens of times.
    """
    stripped = text.strip()
    if len(stripped) < 12:
        return False

    words = stripped.split()
    if len(words) >= 6 and len(set(words)) <= max(2, len(words) // 8):
        return True

    # Same failure without spaces, e.g. "हाहाहाहाहा…".
    compact = "".join(stripped.split())
    if len(compact) >= 20 and len(set(compact)) <= 3:
        return True

    # A short phrase repeated back to back.
    for size in (2, 3, 4):
        if len(words) >= size * 4:
            chunks = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
            if len(set(chunks)) <= max(1, len(chunks) // 6):
                return True
    return False
