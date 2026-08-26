# -*- coding: utf-8 -*-
"""Speech output shaping.

`trim_silence` exists because Edge pads every clip: the wake-word reply
"Mm-hmm?" came back as 1.78 s of audio containing 0.58 s of speech. That
padding sits between the wake word and the microphone opening, so it is dead
time at the worst possible moment — and it is invisible in a transcript, which
is why it gets a test rather than a comment.
"""

import numpy as np
import pytest

from jarvis.tts.base import chunk_text
from jarvis.tts.edge import trim_silence

RATE = 24000


def _clip(lead_sec: float, speech_sec: float, trail_sec: float) -> np.ndarray:
    """Speech-shaped noise surrounded by digital silence."""
    rng = np.random.default_rng(0)
    speech = rng.normal(0.0, 0.3, int(RATE * speech_sec)).astype(np.float32)
    return np.concatenate([
        np.zeros(int(RATE * lead_sec), dtype=np.float32),
        speech,
        np.zeros(int(RATE * trail_sec), dtype=np.float32),
    ])


def test_padding_is_removed():
    trimmed = trim_silence(_clip(0.2, 0.6, 1.0), RATE)
    duration = trimmed.shape[0] / RATE
    # 0.6 s of speech plus at most 30 ms of padding at each end.
    assert 0.6 <= duration <= 0.72, duration


def test_the_speech_itself_survives():
    """Trimming too eagerly would clip the first consonant."""
    original = _clip(0.3, 0.5, 0.9)
    trimmed = trim_silence(original, RATE)
    assert np.isclose(np.abs(trimmed).max(), np.abs(original).max())


def test_a_clip_with_no_padding_is_left_alone():
    clip = _clip(0.0, 0.5, 0.0)
    assert trim_silence(clip, RATE).shape[0] == clip.shape[0]


def test_internal_pauses_are_kept():
    """Only the ends are trimmed — a gap between sentences must remain."""
    rng = np.random.default_rng(1)
    word = rng.normal(0.0, 0.3, int(RATE * 0.4)).astype(np.float32)
    gap = np.zeros(int(RATE * 0.5), dtype=np.float32)
    clip = np.concatenate([np.zeros(int(RATE * 0.3), np.float32), word, gap, word,
                           np.zeros(int(RATE * 0.8), np.float32)])
    trimmed = trim_silence(clip, RATE)
    assert trimmed.shape[0] / RATE >= 1.3  # 0.4 + 0.5 + 0.4, plus padding


@pytest.mark.parametrize("audio", [
    np.zeros(0, dtype=np.float32),
    np.zeros(1000, dtype=np.float32),
])
def test_degenerate_input_is_returned_unchanged(audio):
    """Silence and empty buffers must not raise or become negative-length."""
    assert trim_silence(audio, RATE).shape[0] == audio.shape[0]


def test_short_acknowledgements_are_one_chunk():
    """The wake-word reply must never be split into two synthesis calls."""
    assert chunk_text("Mm-hmm?") == ["Mm-hmm?"]
    assert chunk_text("जी?") == ["जी?"]
