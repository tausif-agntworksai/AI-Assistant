# -*- coding: utf-8 -*-
"""Frame confirmation on the wake word.

openWakeWord scores each 80 ms frame on its own, so a single spike — a cough,
a consonant off the television — used to be a detection all by itself. Real
speech holds the score up across consecutive frames, which is the difference
these tests pin down.
"""

import numpy as np

from jarvis.audio.wakeword import OpenWakeWord


class _Scripted(OpenWakeWord):
    """An OpenWakeWord whose scores come from a list instead of from ONNX.

    Subclassed rather than mocked so the threshold, cooldown and confirmation
    logic under test are the real ones; only the model is replaced.
    """

    def __init__(self, scores, **kwargs):
        self.scores = list(scores)
        self._index = 0
        params = {"threshold": 0.3, "cooldown_sec": 0.0,
                  "confirm_frames": 2, "confirm_window": 3}
        params.update(kwargs)
        for key, value in params.items():
            setattr(self, key, value)
        from collections import deque
        self._recent = deque(maxlen=self.confirm_window)
        self._last_fire = 0.0

    def detect(self, frame):
        score = self.scores[self._index] if self._index < len(self.scores) else 0.0
        self._index += 1
        return score

    def reset(self):
        self._recent.clear()


FRAME = np.zeros(1280, dtype=np.float32)


def _fire_count(scores, **kwargs):
    detector = _Scripted(scores, **kwargs)
    return sum(1 for _ in scores if detector.triggered(FRAME))


def test_one_loud_frame_is_not_a_wake_word():
    """The case this exists for: a single spike in an otherwise quiet room."""
    assert _fire_count([0.0, 0.0, 0.95, 0.0, 0.0, 0.0]) == 0


def test_the_wake_word_held_across_frames_fires():
    assert _fire_count([0.0, 0.9, 0.95, 0.0]) == 1


def test_two_hits_split_by_a_miss_still_count():
    """They only have to fall inside the window, not be adjacent.

    A quiet frame mid-phrase is ordinary — the gap between "hey" and "jarvis"
    dips — so requiring adjacency would reintroduce the misses that dropping
    the threshold to 0.30 was meant to fix.
    """
    assert _fire_count([0.9, 0.1, 0.9, 0.0]) == 1


def test_hits_further_apart_than_the_window_do_not_accumulate():
    assert _fire_count([0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]) == 0


def test_scores_under_the_threshold_never_count_however_many():
    assert _fire_count([0.29] * 10) == 0


def test_confirm_frames_of_one_restores_single_frame_behaviour():
    assert _fire_count([0.0, 0.95, 0.0], confirm_frames=1) == 1


def test_a_detection_clears_the_window_behind_it():
    """Otherwise the tail of one detection counts toward the next."""
    detector = _Scripted([0.9, 0.9, 0.9, 0.0])
    assert detector.triggered(FRAME) is False
    assert detector.triggered(FRAME) is True
    # Third frame is still over the threshold, but it is now the only one.
    assert detector.triggered(FRAME) is False


def test_the_cooldown_still_applies():
    detector = _Scripted([0.9] * 6, cooldown_sec=60.0)
    fires = sum(1 for _ in range(6) if detector.triggered(FRAME))
    assert fires == 1


def test_the_log_reports_the_window_that_fired(caplog):
    """It read the window *after* clearing it, so every detection logged
    "0/0 frames" -- the one number someone checking the confirmation logic
    would most want, reported as nothing."""
    import logging

    detector = _Scripted([0.9, 0.9])
    with caplog.at_level(logging.INFO, logger="jarvis.audio.wakeword"):
        detector.triggered(FRAME)
        assert detector.triggered(FRAME) is True

    line = [r.getMessage() for r in caplog.records if "Wake word detected" in r.getMessage()]
    assert line and "0/0" not in line[-1], line
    assert "2/2" in line[-1]
