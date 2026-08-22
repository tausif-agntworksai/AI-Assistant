"""Measure the wake word against *this* microphone and *this* voice.

The wake-word threshold is the one setting nobody can pick for you. openWakeWord
ships a default of 0.5, and whether that is right depends on your accent, your
microphone's gain, how far away you sit and what your room sounds like — none of
which are visible from here. Set it too high and you say "hey jarvis" three
times; too low and the television wakes the assistant up.

So this measures instead of guessing. You say the wake word a few times, it
records the peak score for each, then it listens to the room saying nothing in
particular and records the highest score that came out of *that*. A threshold
that sits between the two is the answer, and if there is no gap between them the
honest output is to say so rather than to pick a number anyway.

The recommendation deliberately sits nearer the quiet end of your utterances
than the middle: a missed wake word costs you a whole repeated sentence, while a
false wake costs a 140 ms cue and a discarded second of silence. Those are not
symmetric, so the threshold should not be either.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

RATE = 16000

#: How far below your weakest utterance to sit. Multiplicative rather than
#: additive so it scales sensibly whether your scores cluster at 0.9 or 0.4.
SAFETY = 0.75

#: Never recommend anything below this. openWakeWord's scores are not calibrated
#: probabilities and the bottom of the range is where unrelated speech lives.
FLOOR = 0.08

#: Nor above this: 0.5 is the default that is already causing repeats, so a
#: recommendation higher than it would be advice to make things worse.
CEILING = 0.5


@dataclass
class Utterance:
    """One peak in the score stream — a candidate "hey jarvis"."""

    score: float
    at: float
    level: float  # speech RMS, so a quiet microphone can be named as the cause


@dataclass
class TuningResult:
    utterances: list[Utterance] = field(default_factory=list)
    noise_peak: float = 0.0
    expected: int = 0

    @property
    def heard(self) -> int:
        return len(self.utterances)

    @property
    def weakest(self) -> float:
        return min((u.score for u in self.utterances), default=0.0)

    @property
    def strongest(self) -> float:
        return max((u.score for u in self.utterances), default=0.0)

    @property
    def median(self) -> float:
        if not self.utterances:
            return 0.0
        return float(np.median([u.score for u in self.utterances]))

    @property
    def recommended(self) -> float | None:
        """A threshold below every utterance and above the room, or None.

        None is a real answer. If the room scored higher than something you
        said, no threshold separates them and the fix is the microphone or the
        room, not this number.
        """
        if not self.utterances:
            return None
        target = max(self.weakest * SAFETY, FLOOR)
        if target <= self.noise_peak:
            return None
        return round(min(target, CEILING), 2)

    @property
    def margin(self) -> float:
        """How much room is left between the room and your quietest utterance."""
        return self.weakest - self.noise_peak


def _peaks(scores: list[tuple[float, float, float]], threshold: float,
           min_gap: float = 1.0) -> list[Utterance]:
    """Collapse a score stream into one entry per utterance.

    A single "hey jarvis" produces a rising run of frames, not one spike, so the
    stream has to be grouped before it can be counted. Anything within `min_gap`
    seconds of the previous peak is the same utterance, and only its maximum is
    kept — that maximum is what a threshold would actually be compared against.
    """
    peaks: list[Utterance] = []
    for score, at, level in scores:
        if score < threshold:
            continue
        if peaks and at - peaks[-1].at < min_gap:
            if score > peaks[-1].score:
                peaks[-1] = Utterance(score, peaks[-1].at, level)
            continue
        peaks.append(Utterance(score, at, level))
    return peaks


def measure(
    detector,
    capture,
    say_times: int = 5,
    listen_sec: float = 20.0,
    quiet_sec: float = 5.0,
    on_progress=None,
    detection_floor: float = 0.05,
) -> TuningResult:
    """Score live microphone audio, first while speaking, then while silent.

    `detector` is scored directly rather than through `triggered()`, because the
    threshold and cooldown are exactly what is being measured — going through
    them would only show detections that already work.
    """
    from .enhance import speech_rms

    result = TuningResult(expected=say_times)
    stream: list[tuple[float, float, float]] = []
    detector.reset()

    for elapsed, frame in _frames(capture, listen_sec):
        score = detector.detect(frame)
        stream.append((score, elapsed, float(speech_rms(frame, RATE))))
        counted = len(_peaks(stream, detection_floor))
        if on_progress is not None:
            on_progress("speak", elapsed, score, counted)
        if counted >= say_times:
            break

    result.utterances = _peaks(stream, detection_floor)

    # Now the other half: what does this room score when nobody says the wake
    # word? Without it a recommendation is only half-measured.
    detector.reset()
    for elapsed, frame in _frames(capture, quiet_sec):
        score = detector.detect(frame)
        result.noise_peak = max(result.noise_peak, score)
        if on_progress is not None:
            on_progress("quiet", elapsed, score, 0)

    return result


def _frames(capture, seconds: float):
    """Yield `(elapsed, frame)` from the capture stream, timed by the audio.

    Elapsed time is counted in samples consumed rather than read off the wall
    clock, because inference happens between frames and the wall clock includes
    it. On a machine where scoring a frame takes longer than the 80 ms of audio
    it covers, wall-clock timing stretches the gaps apart and splits a single
    "hey jarvis" into two utterances — the measurement would then depend on how
    busy the CPU was. Sample counting cannot drift that way.

    The wall clock is still consulted, purely as a stall guard: if the capture
    stream dries up, audio time stops advancing and nothing else would end the
    loop.
    """
    from .capture import FrameAccumulator
    from .wakeword import OWW_FRAME

    accumulator = FrameAccumulator(OWW_FRAME)
    samples = 0
    stall_deadline = time.monotonic() + seconds * 3 + 5.0
    for block in capture.blocks(timeout=0.5):
        for frame in accumulator.push(np.asarray(block, dtype=np.float32).ravel()):
            samples += frame.shape[0]
            yield samples / RATE, frame
        if samples / RATE >= seconds or time.monotonic() >= stall_deadline:
            return


def explain(result: TuningResult, current: float) -> list[str]:
    """The measurement, in words, with the reason for the recommendation."""
    lines: list[str] = []

    if not result.utterances:
        lines += [
            "  Nothing scored high enough to count as an utterance at all.",
            "",
            "  That is not a threshold problem — the model never got close. In",
            "  order of likelihood: the wrong input device is selected (check",
            "  --list-devices), the microphone is muted or its level is very",
            "  low (check --test-mic), or the phrase was not 'hey jarvis' said",
            "  as two connected words.",
        ]
        return lines

    lines.append(
        "  heard %d of %d — scores %.2f (weakest) to %.2f (strongest), median %.2f"
        % (result.heard, result.expected, result.weakest, result.strongest, result.median)
    )
    lines.append("  the quiet room peaked at %.2f" % result.noise_peak)

    quiet = [u for u in result.utterances if u.level < 0.02]
    if quiet:
        lines.append(
            "  %d of those were quiet at the microphone (below -34 dBFS), which"
            % len(quiet)
        )
        lines.append("  caps how high the scores can get — moving closer helps more")
        lines.append("  than any threshold will")

    recommended = result.recommended
    lines.append("")
    if recommended is None:
        lines += [
            "  No threshold separates your voice from this room: the room scored",
            "  %.2f and your weakest 'hey jarvis' scored %.2f. Lowering the" % (
                result.noise_peak, result.weakest),
            "  threshold far enough to catch you would also let the room wake it.",
            "  Fix the input first — a closer or better microphone, or a quieter",
            "  time of day — then run this again.",
        ]
        return lines

    lines.append("  recommended threshold: %.2f  (currently %.2f)" % (recommended, current))
    lines.append(
        "  — %.0f%% of your weakest utterance, leaving %.2f of margin above the room"
        % (SAFETY * 100, recommended - result.noise_peak)
    )
    if recommended < current:
        lines.append(
            "  Your utterances were landing near the current threshold, which is"
        )
        lines.append("  why some of them needed repeating.")
    elif recommended > current:
        lines.append("  Your voice already clears the current threshold comfortably.")
        lines.append("  If it still needs repeating, the cause is upstream: check")
        lines.append("  --test-mic for the input level.")
    return lines
