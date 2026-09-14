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


# --- shaping: why the cue stopped sounding abrupt ---------------------------


def test_the_cue_fades_in_rather_than_switching_on():
    """Trimming the synthesiser's padding leaves the clip starting on its first
    loud sample, so playback begins mid-waveform. That onset is heard as a click
    followed by a word, and it was most of why the cue sounded curt even before
    the speech rate came down."""
    blunt = np.ones(RATE // 2, dtype=np.float32) * 0.4
    shaped = earcon._soften(blunt)
    assert abs(float(shaped[0])) < 0.01, "still starts at full level"
    assert abs(float(shaped[-1])) < 0.01, "still stops dead"


def test_the_release_is_longer_than_the_attack():
    """A sound that stops abruptly feels curt, and this one exists to feel like
    the beginning of listening rather than the end of a beep."""
    flat = np.ones(RATE, dtype=np.float32) * 0.4
    shaped = earcon._soften(flat)
    quiet = np.abs(shaped) < 0.4 * 0.99
    leading = int(np.argmin(quiet))
    trailing = len(quiet) - int(np.argmin(quiet[::-1]))
    assert (len(quiet) - trailing) > leading


def test_every_variant_is_levelled_to_the_same_peak():
    """The variants rotate, so one being louder than the next is heard as a
    glitch rather than as variety. Measured, the Hindi cue came back 43% hotter
    than the English one from the same synthesiser at the same volume."""
    loud = np.ones(RATE // 2, dtype=np.float32) * 0.9
    soft = np.ones(RATE // 2, dtype=np.float32) * 0.05
    peaks = [float(np.abs(earcon._soften(clip)).max()) for clip in (loud, soft)]
    assert peaks[0] == pytest.approx(peaks[1], abs=0.01)
    assert peaks[0] == pytest.approx(earcon.CUE_PEAK, abs=0.01)


def test_shaping_an_empty_clip_does_not_explode():
    assert earcon._soften(np.zeros(0, dtype=np.float32)).size == 0


def test_a_silent_clip_is_not_amplified_into_noise():
    """Dividing by a peak of zero would turn a silent clip into whatever the
    floating point gods decided."""
    shaped = earcon._soften(np.zeros(1600, dtype=np.float32))
    assert np.all(np.isfinite(shaped)) and float(np.abs(shaped).max()) == 0.0


def test_the_cue_is_quieter_than_a_reply():
    """It is a nod, not information."""
    assert earcon.CUE_VOLUME.startswith("-")


def test_the_cache_key_changes_when_the_shaping_does(monkeypatch):
    """Otherwise a machine that has run the old version keeps playing the old
    cue forever, and the fix appears not to work.

    Asserted as a property rather than against a literal tag: the literal
    version of this test failed the moment the cue wording legitimately
    changed, which is the opposite of what it is for.
    """
    baseline = earcon._cue_signature()

    monkeypatch.setattr(earcon, "CUES", {"en": ("Something else?",), "hi": ("कुछ और?",)})
    assert earcon._cue_signature() != baseline, "new wording must re-render"

    monkeypatch.setattr(earcon, "CUE_RATE", "-25%")
    assert earcon._cue_signature() != baseline, "new rate must re-render"


def test_the_cache_key_is_stable_across_calls():
    """A key that moved on its own would re-render the cues every launch."""
    assert earcon._cue_signature() == earcon._cue_signature()
    assert earcon._RATE_TAG == earcon._cue_signature()


# Sounds a neural voice has no reliable pronunciation for. It renders them as
# an unintelligible mumble, which is what "not sounding very clearly" was.
# A letter count can't tell these apart from real short words — "Mm" has two
# letters and "जी" has one plus a combining vowel sign — so this is a list.
NON_LEXICAL = {"mm", "mmm", "hm", "hmm", "mm-hmm", "mhm", "uh-huh", "हम्म", "हूँ"}


def test_cues_are_real_words():
    """Every cue must be something the synthesiser can actually pronounce."""
    for language, phrases in earcon.CUES.items():
        for phrase in phrases:
            assert any(c.isalpha() for c in phrase), f"{language}: {phrase!r}"
            bare = phrase.strip().rstrip("?.!").lower()
            assert bare not in NON_LEXICAL, (
                f"{language}: {phrase!r} is a hum, not a word — it will mumble"
            )


def test_the_chime_is_as_loud_as_the_voice_cue_it_replaces():
    """The regression this pins down.

    The chime carried a hardcoded amplitude while the spoken cues were
    normalised to CUE_PEAK, leaving it 11.5 dB quieter. When the default cue
    changed from voice to chime, the acknowledgement did not just get shorter,
    it became hard to hear at all on laptop speakers -- so people said the
    wake word a second time, which is the exact problem the cue exists to
    prevent.
    """
    from jarvis.audio.earcon import CUE_PEAK

    assert float(np.abs(chime()).max()) == pytest.approx(CUE_PEAK, abs=0.01)
