"""How long a turn took, broken down by stage.

The assistant already logged the two slowest pieces — recognition and the
model — but never the whole turn, so "it feels slow" had no number attached to
it and every optimisation was a guess. A stage that isn't measured is a stage
nobody can argue about.

Deliberately a plain timer and a dict. The cost of measuring has to be far
below the resolution of what it measures, or the instrument becomes part of
the problem; this is a `perf_counter` call per stage and one log line per turn.

    timer = TurnTimer()
    with timer.stage("stt"):
        transcript = transcribe(audio)
    with timer.stage("route"):
        intent = route(transcript.text)
    timer.done()        # logs: turn 1.61s  stt 1.38  route 0.01  act 0.22
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

log = logging.getLogger(__name__)

#: The stages a turn can pass through, in the order they happen. Fixed so the
#: log line reads the same every time and can be scanned down a column.
STAGES = ("wake", "record", "stt", "route", "brain", "act", "tts")


class TurnTimer:
    """Stage timings for one utterance."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.stages: dict[str, float] = {}
        self.facts: dict[str, Any] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block. Re-entering a stage adds to it rather than replacing.

        Adding matters for the stages that genuinely happen twice — a
        re-decode is a second `stt`, and the honest number for the turn is
        both passes together, which is what the user waited through.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (
                time.perf_counter() - start
            )

    def note(self, **facts: Any) -> None:
        """Record something about the turn that isn't a duration."""
        self.facts.update(facts)

    @property
    def total(self) -> float:
        return time.perf_counter() - self.started

    def summary(self) -> dict[str, Any]:
        return {
            "total_ms": round(self.total * 1000),
            "stages_ms": {
                name: round(self.stages[name] * 1000)
                for name in STAGES
                if name in self.stages
            },
            **self.facts,
        }

    def done(self, publish: bool = True) -> dict[str, Any]:
        """Log the turn, and put it on the bus for the HUD."""
        summary = self.summary()
        parts = " ".join(
            f"{name} {self.stages[name]:.2f}" for name in STAGES if name in self.stages
        )
        # Anything the caller noted, so "which path did this take" sits beside
        # "how long did it take" rather than in a different log line.
        extra = " ".join(f"{k}={v}" for k, v in self.facts.items() if v != "")
        log.info("turn %.2fs  %s  %s", self.total, parts, extra)

        if publish:
            from .bus import Event, bus

            bus.publish(Event.TIMING, **summary)
        return summary
