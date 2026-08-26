# -*- coding: utf-8 -*-
"""Measuring the wake word, and the decision that came out of the measurement.

The wake word was needing two to four attempts. The cause turned out to be the
threshold rather than the microphone: openWakeWord's default of 0.5 sat above
where accented utterances actually score. These tests hold on to both halves of
that — the recommendation logic, and the numbers that justified moving it.
"""

import numpy as np
import pytest

from jarvis.audio.wake_tune import (
    CEILING,
    FLOOR,
    RATE,
    TuningResult,
    Utterance,
    _peaks,
    explain,
    measure,
)
from jarvis.config import WakeWordConfig


def utterances(*scores: float, level: float = 0.1, gap: float = 2.0):
    return [Utterance(s, i * gap, level) for i, s in enumerate(scores)]


# --- grouping a score stream into utterances -------------------------------


def test_one_word_spoken_once_counts_once():
    """A single "hey jarvis" produces a rising run of frames, not one spike. If
    each frame counted, five utterances would look like fifty."""
    stream = [(s, 0.08 * i, 0.1) for i, s in enumerate([0.0, 0.2, 0.6, 0.9, 0.7, 0.1])]
    peaks = _peaks(stream, threshold=0.05)
    assert len(peaks) == 1
    assert peaks[0].score == pytest.approx(0.9), "the peak is what a threshold sees"


def test_two_utterances_a_breath_apart_count_twice():
    stream = [(0.8, 0.0, 0.1), (0.7, 0.3, 0.1), (0.9, 4.0, 0.1)]
    assert len(_peaks(stream, threshold=0.05)) == 2


def test_silence_produces_nothing():
    assert _peaks([(0.001, i * 0.08, 0.0) for i in range(50)], threshold=0.05) == []


# --- the recommendation ----------------------------------------------------


def test_it_recommends_below_the_weakest_utterance_not_the_average():
    """The weakest utterance is the one that decides whether you repeat
    yourself. A threshold at the median would still miss half of them."""
    result = TuningResult(utterances=utterances(0.9, 0.8, 0.4), noise_peak=0.02)
    assert result.recommended is not None
    assert result.recommended < 0.4


def test_it_leaves_margin_rather_than_sitting_exactly_on_the_weakest():
    """Sitting exactly on the measured minimum means the next slightly-worse
    utterance is a miss. Five samples do not establish the true floor."""
    result = TuningResult(utterances=utterances(0.5, 0.5, 0.5), noise_peak=0.01)
    assert result.recommended is not None and result.recommended < 0.5


def test_it_refuses_to_recommend_when_the_room_scores_higher_than_the_voice():
    """No number separates them, and inventing one would mean either constant
    false wakes or constant misses. Saying so is the useful answer."""
    result = TuningResult(utterances=utterances(0.2, 0.3), noise_peak=0.35)
    assert result.recommended is None
    assert "No threshold separates" in "\n".join(explain(result, 0.5))


def test_it_never_recommends_below_the_floor():
    """openWakeWord's scores are not calibrated probabilities and the bottom of
    the range is where unrelated speech lives."""
    result = TuningResult(utterances=utterances(0.02, 0.03), noise_peak=0.0)
    assert result.recommended is None or result.recommended >= FLOOR


def test_it_never_recommends_raising_the_bar_past_the_current_default():
    """0.5 is the default that was already causing repeats, so anything above it
    would be advice to make the problem worse."""
    result = TuningResult(utterances=utterances(0.99, 0.99, 0.99), noise_peak=0.0)
    assert result.recommended is not None and result.recommended <= CEILING


def test_hearing_nothing_at_all_blames_the_input_not_the_threshold():
    """Zero detections is not a tuning result. The model never got close, so the
    device, the mute switch or the phrase itself is the problem, and a
    recommendation here would send the user to adjust the wrong thing."""
    lines = "\n".join(explain(TuningResult(expected=5), 0.5))
    assert "Nothing scored high enough" in lines
    assert "recommended" not in lines.lower()


def test_a_quiet_microphone_is_named_as_the_cause():
    result = TuningResult(utterances=utterances(0.35, 0.4, 0.45, level=0.004),
                          noise_peak=0.01)
    assert "quiet at the microphone" in "\n".join(explain(result, 0.5))


def test_the_explanation_says_which_way_it_is_moving_and_why():
    result = TuningResult(utterances=utterances(0.55, 0.6), noise_peak=0.02)
    lines = "\n".join(explain(result, 0.5))
    assert "why some of them needed repeating" in lines


# --- the measurement loop --------------------------------------------------


class ScriptedDetector:
    """Returns a fixed score per frame, so the loop can be tested without audio."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0
        self.resets = 0

    def detect(self, frame):
        score = self.scores[self.calls] if self.calls < len(self.scores) else 0.0
        self.calls += 1
        return score

    def reset(self):
        self.resets += 1


class FakeCapture:
    def __init__(self, blocks=400):
        self.count = blocks

    def blocks(self, timeout=0.5):
        for _ in range(self.count):
            yield np.zeros(1280, dtype=np.float32)


def test_it_stops_once_it_has_heard_enough():
    """Otherwise the user stands there saying it into a timer that has already
    got what it needs."""
    detector = ScriptedDetector([0.9, 0.0, 0.0, 0.0, 0.0] * 40)
    result = measure(detector, FakeCapture(), say_times=2, listen_sec=60.0,
                     quiet_sec=0.0)
    assert result.heard == 2


def test_the_quiet_pass_measures_the_room_separately():
    """Without it the recommendation is half-measured: you know what your voice
    scores and nothing about what would set it off on its own."""
    detector = ScriptedDetector([0.0] * 200)
    result = measure(detector, FakeCapture(), say_times=1, listen_sec=0.2,
                     quiet_sec=0.5)
    assert detector.resets >= 2, "the feature buffers must be cleared between passes"


def test_it_scores_frames_directly_rather_than_through_the_threshold():
    """Going through `triggered()` would only ever show detections that already
    work — precisely the ones that are not the problem."""
    detector = ScriptedDetector([0.2] * 200)  # under any sane threshold
    result = measure(detector, FakeCapture(), say_times=1, listen_sec=0.5,
                     quiet_sec=0.0, detection_floor=0.05)
    assert result.heard == 1, "a sub-threshold utterance must still be measured"


# --- the shipped default ---------------------------------------------------


def test_the_shipped_threshold_matches_what_was_measured():
    """This is a documented decision, not a preference: at 0.50 the benchmark
    missed 18 of 80 utterances and at 0.30 it missed 7, while the only
    non-wake-word utterance that scores above 0.30 is the bare name "jarvis".

    If someone raises this back to 0.5, the repeats come back — so the number is
    pinned here with its reason. See docs/HEARING.md for the full table.
    """
    assert WakeWordConfig().threshold == 0.3


def test_the_wake_word_is_acknowledged_by_default():
    """The other half of the same complaint: with no reply, there is no way to
    tell a miss from a slow response, and the natural reaction is to say it
    again — which is itself scored lower."""
    assert WakeWordConfig().acknowledge != "none"
