# -*- coding: utf-8 -*-
"""The sound the assistant makes when it starts listening.

Before this existed, saying the wake word produced a log line and a colour
change on an orb that is usually hidden in the tray — so there was no way to
tell whether you had been heard, and the natural response is to say it again.

The properties worth testing are not "does it make a noise" but the ones that
decide whether the noise helps: it has to be instant, it must never be the
reason a turn fails, and the frames it occupies must not reach the recogniser
glued to the front of the command.
"""

import numpy as np
import pytest

from jarvis.audio import earcon
from jarvis.audio.earcon import RATE, Acknowledger, chime


class FakePlayer:
    """Records what it was asked to play, and how."""

    def __init__(self) -> None:
        self.earcons: list[np.ndarray] = []
        self.replies: list[np.ndarray] = []

    def play_earcon(self, audio, sample_rate):
        assert sample_rate == RATE
        self.earcons.append(np.asarray(audio))

    def play_async(self, audio, sample_rate, earcon=False):
        (self.earcons if earcon else self.replies).append(np.asarray(audio))


# --- the chime -------------------------------------------------------------


def test_the_chime_is_short_enough_to_talk_over():
    """Every millisecond of cue is a millisecond of command we discard."""
    clip = chime()
    milliseconds = clip.shape[0] / RATE * 1000
    assert 80 <= milliseconds <= 250, f"{milliseconds:.0f} ms is the wrong length"


def test_the_chime_does_not_click():
    """A bare sine switched on and off clicks at both ends, and a click is the
    least reassuring sound a machine can make when you are checking it heard
    you."""
    clip = chime()
    assert abs(float(clip[0])) < 1e-6
    assert abs(float(clip[-1])) < 1e-6


def test_the_chime_never_clips():
    assert float(np.abs(chime()).max()) < 0.5


def test_the_chime_rises():
    """Rising reads as a question — "go on". Falling reads as completion."""
    clip = chime()
    half = clip.shape[0] // 2
    # Zero crossings are a good enough proxy for pitch on a pure tone.
    first = np.count_nonzero(np.diff(np.signbit(clip[:half])))
    second = np.count_nonzero(np.diff(np.signbit(clip[half:])))
    assert second > first, "the second note should be higher than the first"


# --- the acknowledger ------------------------------------------------------


def test_it_plays_the_chime_before_the_voice_cues_are_ready():
    """A first launch spends minutes downloading models. The wake word has to be
    acknowledged during that, not after it."""
    player = FakePlayer()
    Acknowledger(player, "voice").play("en")
    assert len(player.earcons) == 1
    assert np.allclose(player.earcons[0], chime())


def test_none_really_is_silent():
    player = FakePlayer()
    Acknowledger(player, "none").play("en")
    assert player.earcons == []


def test_it_plays_as_an_earcon_not_as_a_reply():
    """The listen loop treats the two completely differently: a reply can be
    barged in on and cut off, a cue must not be."""
    player = FakePlayer()
    Acknowledger(player, "chime").play("en")
    assert len(player.earcons) == 1 and player.replies == []


def test_a_broken_output_device_never_costs_a_turn():
    """No microphone turn should ever be lost because a chime could not play."""

    class Broken:
        def play_earcon(self, audio, sample_rate):
            raise RuntimeError("no output device")

    Acknowledger(Broken(), "chime").play("en")  # must not raise


def test_the_cue_follows_the_language_of_the_conversation(monkeypatch):
    quiet = np.zeros(1600, dtype=np.float32)
    loud = np.ones(3200, dtype=np.float32) * 0.2
    player = FakePlayer()
    ack = Acknowledger(player, "voice")
    ack._voice = {"en": [quiet], "hi": [loud]}

    ack.play("en")
    ack.play("hi")
    assert player.earcons[0].shape[0] == 1600
    assert player.earcons[1].shape[0] == 3200


def test_variants_rotate_so_it_is_not_identical_every_time():
    player = FakePlayer()
    ack = Acknowledger(player, "voice")
    ack._voice = {"en": [np.zeros(100, dtype=np.float32), np.zeros(200, dtype=np.float32)]}
    for _ in range(4):
        ack.play("en")
    lengths = [clip.shape[0] for clip in player.earcons]
    assert lengths == [100, 200, 100, 200]


def test_a_failed_render_falls_back_rather_than_going_silent(monkeypatch, tmp_path):
    """Offline, or with TTS disabled, there must still be an acknowledgement."""
    monkeypatch.setattr(earcon.paths, "CACHE_DIR", tmp_path)

    class DeadSpeaker:
        def synthesize(self, text, language):
            raise RuntimeError("no network")

        def close(self):
            pass

    ack = Acknowledger(FakePlayer(), "voice")
    monkeypatch.setattr(ack, "_cue_speaker", lambda: DeadSpeaker())
    ack._prepare_now()

    assert ack._voice == {}
    ack.play("en")
    assert np.allclose(ack.player.earcons[0], chime())


def test_rendered_cues_are_cached_so_a_later_offline_launch_still_has_them(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(earcon.paths, "CACHE_DIR", tmp_path)
    rendered: list[str] = []

    class Speaker:
        def synthesize(self, text, language):
            rendered.append(text)
            return np.ones(4000, dtype=np.float32) * 0.3, RATE

        def close(self):
            pass

    first = Acknowledger(FakePlayer(), "voice")
    monkeypatch.setattr(first, "_cue_speaker", lambda: Speaker())
    first._prepare_now()
    assert rendered, "nothing was rendered"
    count = len(rendered)

    # A second instance must read the cache and synthesise nothing at all.
    second = Acknowledger(FakePlayer(), "voice")
    monkeypatch.setattr(
        second, "_cue_speaker", lambda: pytest.fail("should not have rendered again")
    )
    second._prepare_now()
    assert len(rendered) == count
    assert sorted(second._voice) == sorted(first._voice)


def test_leading_and_trailing_silence_is_trimmed():
    """Synthesisers pad their output, and that padding would double the cue's
    apparent latency while lengthening the window where speech is discarded."""
    padded = np.concatenate([
        np.zeros(RATE // 2, dtype=np.float32),
        np.ones(RATE // 10, dtype=np.float32) * 0.3,
        np.zeros(RATE // 2, dtype=np.float32),
    ])
    trimmed = earcon._trim_silence(padded)
    assert trimmed.shape[0] < padded.shape[0] // 2
    assert float(np.abs(trimmed).max()) == pytest.approx(0.3)
