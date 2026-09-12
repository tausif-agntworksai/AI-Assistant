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


# --- the first chunk decides how soon speech starts ------------------------
#
# `EdgeSpeaker.speak` synthesises chunk n+1 while chunk n plays, so the wait
# before the first sound is the time to synthesise the *first chunk* — not the
# whole reply. That only helps if the first chunk is actually short, and for a
# long time it wasn't: anything under 220 characters came back unsplit, which
# is most replies.


def test_a_two_sentence_reply_starts_speaking_after_the_first_sentence():
    parts = chunk_text("The meeting has moved to 4 PM. I've let Sana know as well.")
    assert len(parts) == 2
    assert parts[0] == "The meeting has moved to 4 PM."


def test_a_single_sentence_is_left_whole():
    assert chunk_text("Opening Chrome.") == ["Opening Chrome."]


def test_a_short_opener_absorbs_the_next_sentence():
    """"Done." is not worth its own websocket connection."""
    assert chunk_text("Done. Opening Chrome.") == ["Done. Opening Chrome."]


def test_later_chunks_are_allowed_to_be_long():
    """Once audio is playing, synthesis runs ahead — bigger chunks there mean
    fewer round trips and prosody that isn't chopped mid-thought."""
    parts = chunk_text(
        "Battery is at 65 percent and charging. You have about three hours "
        "left. I'd suggest plugging in before the call."
    )
    assert len(parts) == 2
    assert parts[0] == "Battery is at 65 percent and charging."
    assert len(parts[1]) > len(parts[0])


def test_hindi_sentences_split_on_the_danda():
    parts = chunk_text(
        "लैपटॉप को सुला रहा हूँ। शुभ रात्रि। आराम कीजिए। कल मिलते हैं।"
    )
    assert len(parts) >= 2
    assert parts[0].startswith("लैपटॉप")


def test_every_chunk_is_non_empty_and_nothing_is_lost():
    text = ("First sentence here. Second one follows. Third is a little "
            "longer than the others. Fourth ends it.")
    parts = chunk_text(text)
    assert all(p.strip() for p in parts)
    joined = " ".join(parts)
    for word in ("First", "Second", "Third", "Fourth"):
        assert word in joined


def test_a_uniform_limit_is_still_available():
    """Passing the floor as zero restores one-chunk-per-cap behaviour."""
    text = "One. Two. Three."
    assert chunk_text(text, first_min_chars=len(text)) == [text]
