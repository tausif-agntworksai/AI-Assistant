"""Two-pass recognition: fast first, accurate only when the fast pass wobbles.

The old arrangement forced a single choice for every utterance. `base` is
0.63× realtime here but mishears Hindi; `small` is 2.0× realtime and gets it
right. Picking `base` meant repeating yourself; picking `small` meant waiting
three extra seconds for "chrome kholo", which the rule matcher would have
resolved in microseconds anyway.

Neither is necessary. Most utterances come back from the small-and-fast model
with high decoder confidence, and those are done. The rest — the ones that
would have made you repeat yourself — are re-decoded from the *same audio* by
the larger model. Nobody speaks twice; the machine listens twice.

    fast pass ──confident?──▶ done                        (~90% of commands)
                    │
                    └─no──▶ accurate pass ──▶ best of the two

The orchestrator can also force the second pass after the fact, via
`escalate()`, when the words came back clean but meant nothing — a transcript
that routes to no skill is itself evidence of a mishearing.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from .base import Transcriber, Transcript

log = logging.getLogger(__name__)

# Below this decoder confidence the fast model is guessing. Measured against
# `Transcript.confidence`, which is exp(avg per-token logprob): a clean read of
# a short command sits around 0.75-0.9, a mishearing around 0.4.
DEFAULT_MIN_CONFIDENCE = 0.62

# Clips shorter than this are single words where the fast model is as good as
# the slow one, and the escalation would double the latency of "haan".
MIN_ESCALATE_SEC = 0.45


class TieredTranscriber(Transcriber):
    name = "tiered"

    def __init__(
        self,
        fast: Transcriber,
        accurate: Transcriber,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self.fast = fast
        self.accurate = accurate
        self.min_confidence = min_confidence
        self.name = f"tiered({getattr(fast, 'model_size', '?')}→"\
                    f"{getattr(accurate, 'model_size', '?')})"
        self._warming = False

    # -- warmup -------------------------------------------------------------

    def warmup(self) -> None:
        """Load the fast model now; the accurate one follows in the background.

        Order matters on a first run: the fast model is ~150 MB and the
        accurate one ~500 MB. Blocking on both would leave the assistant deaf
        for minutes on a slow connection, so the second download happens behind
        an assistant that already works.
        """
        self.fast.warmup()
        self.warm_accurate_async()

    def warm_accurate_async(self) -> None:
        if self._warming:
            return
        self._warming = True

        def run() -> None:
            try:
                self.accurate.warmup()
                log.info("Accurate speech model ready for escalations")
            except Exception as exc:  # noqa: BLE001
                log.warning("Accurate speech model unavailable (%s) — "
                            "staying on the fast model only", exc)

        threading.Thread(target=run, name="stt-warm-accurate", daemon=True).start()

    # -- recognition --------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        *,
        short_answer: bool = False,
        hint: str | None = None,
    ) -> Transcript:
        first = self.fast.transcribe(
            audio, sample_rate, short_answer=short_answer, hint=hint
        )

        duration = np.asarray(audio).size / max(1, sample_rate)
        if duration < MIN_ESCALATE_SEC:
            return first
        if first.text.strip() and first.confidence >= self.min_confidence:
            return first

        reason = "nothing intelligible" if first.is_empty else \
                 f"confidence {first.confidence:.2f}"
        log.info("Escalating to the accurate model (%s)", reason)
        return self._best_of(first, self._accurate(audio, sample_rate, short_answer, hint))

    def escalate(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        *,
        previous: Transcript | None = None,
        short_answer: bool = False,
        hint: str | None = None,
    ) -> Transcript:
        """Force the accurate pass. Used when the words parsed to nothing.

        Unlike the automatic path, this does *not* fall back to the more
        confident read. The caller has already established that the fast
        model's confident answer was wrong — it routed to no skill — so
        confidence has stopped being evidence, and the better model wins.
        """
        second = self._accurate(audio, sample_rate, short_answer, hint)
        if not second.is_empty:
            return second
        return previous if previous is not None else second

    def _accurate(
        self,
        audio: np.ndarray,
        sample_rate: int,
        short_answer: bool,
        hint: str | None,
    ) -> Transcript:
        try:
            return self.accurate.transcribe(
                audio, sample_rate, short_answer=short_answer, hint=hint
            )
        except Exception as exc:  # noqa: BLE001 - the fast result still stands
            log.error("Accurate pass failed (%s)", exc)
            return Transcript(text="", backend="accurate:failed")

    @staticmethod
    def _best_of(first: Transcript | None, second: Transcript) -> Transcript:
        """Prefer words over silence, then prefer the more confident read."""
        if first is None or first.is_empty:
            return second if not second.is_empty else (first or second)
        if second.is_empty:
            return first
        return second if second.confidence >= first.confidence else first

    def close(self) -> None:
        self.fast.close()
        self.accurate.close()
