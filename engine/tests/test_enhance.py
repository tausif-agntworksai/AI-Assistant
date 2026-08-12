# -*- coding: utf-8 -*-
"""Signal conditioning before recognition.

Whisper's answer to a quiet clip is not "I'm not sure", it's a confident
transcription of the wrong words — which is what made people repeat
themselves. These assert the three properties that make the fix safe: quiet
speech gets louder, silence does not, and nothing clips.
"""

import numpy as np

from jarvis.audio.enhance import (
    PAD_SEC,
    TARGET_RMS,
    condition,
    is_too_quiet,
    normalise,
    remove_rumble,
    speech_rms,
)

RATE = 16000


def _tone(seconds=1.0, amplitude=0.01, freq=220.0, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_a_quiet_recording_is_brought_up_to_a_usable_level():
    quiet = _tone(amplitude=0.008)
    boosted, gain = normalise(quiet, RATE)
    assert gain > 1.0
    assert speech_rms(boosted, RATE) > TARGET_RMS * 0.5


def test_silence_is_not_amplified_into_noise():
    """The failure this guards: hiss boosted 26 dB reads as speech."""
    silence = np.zeros(RATE, dtype=np.float32)
    _, gain = normalise(silence, RATE)
    assert gain == 1.0


def test_a_loud_recording_is_left_alone():
    loud = _tone(amplitude=0.3)
    _, gain = normalise(loud, RATE)
    assert gain == 1.0


def test_nothing_ever_clips():
    boosted, _ = normalise(_tone(amplitude=0.02), RATE)
    assert np.abs(boosted).max() <= 1.0


def test_dc_offset_and_rumble_are_removed():
    speech = _tone(amplitude=0.05, freq=300.0)
    with_rumble = speech + 0.2  # a constant offset is rumble at 0 Hz
    cleaned = remove_rumble(with_rumble, RATE)
    assert abs(float(cleaned.mean())) < 0.01
    # The 300 Hz content has to survive — this is a high-pass, not a mute.
    assert speech_rms(cleaned, RATE) > 0.5 * speech_rms(speech, RATE)


def test_the_clip_is_padded_so_no_phoneme_starts_on_sample_zero():
    audio = _tone(seconds=0.5, amplitude=0.05)
    out = condition(audio, RATE)
    assert out.shape[0] == audio.shape[0] + 2 * int(RATE * PAD_SEC)
    assert float(np.abs(out[: int(RATE * PAD_SEC)]).max()) == 0.0


def test_conditioning_survives_an_empty_clip():
    assert condition(np.zeros(0, dtype=np.float32), RATE).size == 0


def test_measuring_loudness_ignores_the_silence_around_the_words():
    """A short command inside a long buffer must not measure as quiet."""
    buffer = np.zeros(RATE * 3, dtype=np.float32)
    buffer[RATE : RATE + RATE // 2] = _tone(seconds=0.5, amplitude=0.2)
    # Plain RMS over the whole buffer would be ~0.06; the speech is at 0.14.
    assert speech_rms(buffer, RATE) > 3 * float(
        np.sqrt(np.mean(np.square(buffer, dtype=np.float64)))
    ) / 3


def test_a_dead_microphone_is_reported_rather_than_transcribed():
    assert is_too_quiet(np.zeros(RATE, dtype=np.float32), RATE)
    assert is_too_quiet(_tone(amplitude=0.0005), RATE)
    assert not is_too_quiet(_tone(amplitude=0.05), RATE)
