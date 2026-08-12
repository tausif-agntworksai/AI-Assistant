"""Speech-to-text interface.

Everything downstream depends only on this shape, so swapping local Whisper for
a cloud transcriber is a config change rather than a rewrite.
"""

from __future__ import annotations

import math
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
    #: Mean per-token log probability from the decoder. 0.0 when the backend
    #: doesn't report one (the cloud providers don't).
    avg_logprob: float = 0.0
    #: True when the words were dropped by a guard rather than never spoken —
    #: the difference between "you said nothing" and "I couldn't make it out",
    #: which is the difference between staying quiet and asking you to repeat.
    filtered: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    @property
    def is_hindi(self) -> bool:
        return self.language.lower().startswith("hi")

    @property
    def confidence(self) -> float:
        """0..1 estimate of how much to trust this transcription.

        Whisper reports an average log probability per token; −0.2 is a clean
        read and −1.0 is the model guessing. Mapping it through exp() gives a
        number that can be compared against a plain threshold, which is what
        the orchestrator uses to decide whether to re-decode with the larger
        model instead of acting on a probable mishearing.
        """
        if not self.text.strip():
            return 0.0
        if self.avg_logprob == 0.0:
            return 1.0  # backend doesn't report one; don't punish it
        return float(min(1.0, math.exp(self.avg_logprob)))

    def __str__(self) -> str:
        return (
            f"[{self.language} {self.language_probability:.2f} "
            f"conf {self.confidence:.2f}] {self.text!r}"
        )


class Transcriber:
    name: str = "base"

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        *,
        short_answer: bool = False,
        hint: str | None = None,
    ) -> Transcript:
        """Turn audio into text.

        `short_answer` marks a yes/no reply to a confirmation. Those are one
        word long and consist almost entirely of words the hallucination guard
        would otherwise throw away, so the guard is relaxed for them.

        `hint` is extra vocabulary for the decoder — installed app names, for
        instance — which meaningfully improves proper nouns.
        """
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
    "शुक्रिया", "धन्यवाद", "धन्यवाद।",
    "okay", "ok", "oh", "hmm", "mm", "uh", "ah",
    "Thank you for watching.", "[music]", "[Music]", "(music)",
}

# Words that are filler in the middle of a command but are the *entire content*
# of an answer to "shall I shut down?". Discarding them was silently turning
# every spoken "haan" into a refusal, so they are only treated as noise when we
# are not waiting for a yes or no.
ANSWER_WORDS = {
    "आप", "जी", "हाँ", "हां", "मैं", "नहीं", "ना",
    "haan", "han", "ha", "ji", "nahi", "na", "yes", "no", "yeah", "nope",
    "okay", "ok", "sure", "theek", "thik",
}


def is_probable_hallucination(text: str, *, short_answer: bool = False) -> bool:
    stripped = text.strip().lower()
    if not stripped:
        return True
    if short_answer and stripped in {w.lower() for w in ANSWER_WORDS}:
        return False
    if stripped in {p.lower() for p in HALLUCINATION_PHRASES}:
        return True
    return not short_answer and stripped in {w.lower() for w in ANSWER_WORDS}


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
