"""The sound the assistant makes when it starts listening.

Without this, saying the wake word produced a log line and a colour change on
an orb that is usually hidden in the tray — so there was no way to know whether
you had been heard. You end up saying "hey jarvis" twice, which is the exact
problem the rest of the hearing work went into removing.

**Latency is the whole design constraint.** An acknowledgement that arrives
400 ms late is worse than none: by then you have already started speaking, and
the cue lands on top of your first word. Edge-TTS needs a network round trip,
so synthesising the cue at wake time is not an option. Everything here is
therefore prepared in advance and played from memory:

  * the **chime** is generated with numpy at startup — no network, no model,
    available before the first launch has finished downloading anything;
  * the **voice** cues are rendered once in the background, cached to disk so a
    later offline launch still has them, and fall back to the chime whenever
    they are not ready.

Whichever it is, playback starts within a block or two of the detection.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from .. import paths

log = logging.getLogger(__name__)

RATE = 16000

# Short single syllables, and deliberately so. A cue overlaps whatever you say
# next, and the frames it occupies are dropped rather than transcribed — so
# every millisecond of cue is a millisecond of your command we cannot hear.
#
# Real words only. The soft-sound versions ("Mm?", "Hm?", "हम्म?") read better
# on paper — a person looking up rather than a machine reporting for duty —
# but they did not survive contact with the synthesiser: reported as "not
# sounding very clearly", and they aren't. Neural voices are trained on
# written language, so a non-lexical sound has no pronunciation they can
# render reliably; what comes out is a mumble that could be anything.
#
# Still one or two syllables, because the frames a cue occupies are dropped
# rather than transcribed — every millisecond of cue is a millisecond of the
# command we cannot hear.
CUES: dict[str, tuple[str, ...]] = {
    "en": ("Yes?", "Yeah?", "I'm here."),
    "hi": ("जी?", "हाँ?", "बोलिए?"),
}

# Just under conversational pace. This was +40% — chosen to keep the cue short,
# because the frames it occupies are dropped and a shorter cue discards less of
# what you say next. It measured well and sounded wrong: a one-syllable
# acknowledgement rushed by 40% is clipped and mechanical, which is the opposite
# of the reassurance it exists to give. 90 ms of extra overlap is worth paying
# for a sound that reads as a person rather than a beep.
CUE_RATE = "-5%"

# The cue should be quieter than a reply. It is not information, it is a nod.
CUE_VOLUME = "-15%"

# Every cue is levelled to this peak. The variants are rotated through, so one
# being noticeably louder than the next is heard as a glitch rather than as
# variety — measured, "हम्म?" came back 43% hotter than "Yes?" from the same
# synthesiser at the same volume setting.
CUE_PEAK = 0.42

# Part of the cache filename, so changing how the cue is rendered re-renders
# rather than silently reusing clips recorded the old way. Bumped when the
# shaping below changes, not just when the rate does.
#
# Derived, not hand-maintained. The cache filename is keyed by index, so
# changing the phrases above while reusing the tag would silently keep
# replaying the old recordings — and the fix would appear not to work. This
# used to be a literal that you had to remember to bump; deriving it from the
# values that decide how a cue sounds removes the chance of forgetting.
#
# `_SHAPING_VERSION` covers what isn't captured below: the attack/release
# envelope in `_soften`, which is defined further down the file.
_SHAPING_VERSION = 1


def _cue_signature() -> str:
    import hashlib

    material = repr((sorted(CUES.items()), CUE_RATE, CUE_VOLUME, CUE_PEAK,
                     _SHAPING_VERSION))
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:10]


_RATE_TAG = _cue_signature()


def chime(rate: int = RATE) -> np.ndarray:
    """A two-note rising blip, about 140 ms.

    Rising rather than falling because a rising pair reads as a question — "go
    on" — where a falling pair reads as completion. Every assistant that does
    this uses the same grammar.

    Shaped with a raised-cosine envelope: a bare sine switched on and off clicks
    audibly at both ends, and a click is the least reassuring sound a machine
    can make when you are trying to tell whether it heard you.

    Levelled to `CUE_PEAK`, the same target the spoken cues are normalised to.
    It used to carry a hardcoded amplitude instead, which left it 11.5 dB below
    the spoken cue it stands in for — quiet enough on laptop speakers to read
    as no acknowledgement at all, so people said the wake word again.
    """
    notes = ((587.3, 0.09), (783.99, 0.13))  # D5 then G5, the same rising fourth
    parts: list[np.ndarray] = []
    for frequency, seconds in notes:
        samples = int(rate * seconds)
        t = np.arange(samples) / rate
        envelope = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(samples) / max(1, samples - 1))
        # A touch of the octave above, at a tenth of the level. A pure sine reads
        # as a test tone; one quiet partial is enough to read as an instrument.
        tone = np.sin(2 * np.pi * frequency * t) + 0.1 * np.sin(4 * np.pi * frequency * t)
        parts.append(tone * envelope)

    clip = np.concatenate(parts).astype(np.float32)
    peak = float(np.abs(clip).max())
    if peak > 1e-6:
        clip *= CUE_PEAK / peak
    return clip


class Acknowledger:
    """Plays the wake-word acknowledgement. Never blocks, never waits on a network."""

    def __init__(self, player, mode: str = "voice") -> None:
        self.player = player
        self.mode = mode
        self._chime = chime()
        #: language -> list of rendered cue clips, rotated through so an
        #: assistant you wake twenty times a day is not word-perfect identical
        #: every single time.
        self._voice: dict[str, list[np.ndarray]] = {}
        self._turn = 0
        self._preparing = False

    # -- preparation --------------------------------------------------------

    @property
    def _cache_dir(self):
        return paths.CACHE_DIR / "cues"

    def prepare(self) -> None:
        """Load or render the voice cues, off the critical path."""
        if self.mode != "voice" or self._preparing:
            return
        self._preparing = True
        threading.Thread(target=self._prepare_now, name="cues", daemon=True).start()

    def _cue_speaker(self):
        """A speaker of our own, at the cue rate.

        Separate from the assistant's voice because the requirements differ: a
        reply should sound unhurried, a cue should be over with.
        """
        from ..config import settings
        from ..tts import create_speaker

        return create_speaker(settings.tts.model_copy(
            update={"rate": CUE_RATE, "volume": CUE_VOLUME}))

    def _prepare_now(self) -> None:
        speaker = None
        for language, phrases in CUES.items():
            clips: list[np.ndarray] = []
            for index, phrase in enumerate(phrases):
                path = self._cache_dir / f"{language}-{index}-{_RATE_TAG}.npy"
                clip = self._load(path)
                if clip is None:
                    # Built only if something actually needs rendering, so a
                    # warm cache costs no synthesiser at all.
                    if speaker is None:
                        speaker = self._cue_speaker()
                    clip = self._render(speaker, phrase, language)
                    if clip is not None:
                        self._save(path, clip)
                if clip is not None and clip.size:
                    clips.append(clip)
            if clips:
                self._voice[language] = clips

        if speaker is not None:
            speaker.close()

        if self._voice:
            log.info("Wake-word cues ready (%s)", ", ".join(sorted(self._voice)))
        else:
            log.info("Wake-word cues unavailable — using the chime instead")

    def _load(self, path) -> np.ndarray | None:
        try:
            return np.load(path).astype(np.float32)
        except Exception:  # noqa: BLE001 - absent or corrupt, both mean "render it"
            return None

    def _save(self, path, clip: np.ndarray) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, clip)
        except OSError as exc:
            # Losing the cache costs one render on the next launch, nothing more.
            log.debug("Could not cache a cue: %s", exc)

    def _render(self, speaker, phrase: str, language: str) -> np.ndarray | None:  # noqa: ANN001
        if speaker is None:
            return None
        try:
            audio, rate = speaker.synthesize(phrase, language)
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not render the cue %r: %s", phrase, exc)
            return None
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if audio.size == 0:
            return None
        if rate != RATE:
            audio = _resample(audio, rate, RATE)
        return _soften(_trim_silence(audio))

    # -- playback ----------------------------------------------------------

    def play(self, language: str = "en") -> None:
        """Acknowledge, right now. Returns as soon as playback has started."""
        if self.mode == "none":
            return

        clip = self._chime
        if self.mode == "voice":
            key = "hi" if (language or "").lower().startswith("hi") else "en"
            options = self._voice.get(key)
            if options:
                clip = options[self._turn % len(options)]
                self._turn += 1

        try:
            self.player.play_earcon(clip, RATE)
        except Exception as exc:  # noqa: BLE001 - never lose a turn to a chime
            log.debug("Could not play the acknowledgement: %s", exc)


def _resample(audio: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target or audio.size == 0:
        return audio
    count = int(round(audio.shape[0] * target / source))
    if count <= 1:
        return np.zeros(0, dtype=np.float32)
    old = np.linspace(0.0, 1.0, audio.shape[0], endpoint=False)
    new = np.linspace(0.0, 1.0, count, endpoint=False)
    return np.interp(new, old, audio).astype(np.float32)


def _soften(audio: np.ndarray, attack_ms: float = 25.0,
            release_ms: float = 90.0) -> np.ndarray:
    """Fade the cue in and out instead of switching it on.

    Trimming the synthesiser's padding leaves the clip starting on the first
    loud sample, so playback begins mid-waveform and the onset reads as a click
    followed by a word — which is most of why the old cue sounded abrupt even
    before the speech rate came down. A short attack and a longer release give
    it the shape of something spoken rather than something triggered.

    The release is the longer of the two on purpose: a sound that stops dead
    feels curt, and this one is meant to feel like the beginning of listening.
    """
    if audio.size == 0:
        return audio
    out = audio.astype(np.float32).copy()

    peak = float(np.abs(out).max())
    if peak > 1e-6:
        out *= CUE_PEAK / peak

    attack = min(int(RATE * attack_ms / 1000), out.shape[0] // 2)
    if attack > 1:
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, attack, dtype=np.float32))
        out[:attack] *= ramp

    release = min(int(RATE * release_ms / 1000), out.shape[0] // 2)
    if release > 1:
        ramp = 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, release, dtype=np.float32))
        out[-release:] *= ramp
    return out


def _trim_silence(audio: np.ndarray, floor: float = 0.01) -> np.ndarray:
    """Cut the leading and trailing silence a synthesiser pads its output with.

    Worth doing: that padding is often 200 ms at each end, which would double
    the apparent latency of the cue and lengthen the window in which the user's
    own speech is being discarded.
    """
    loud = np.flatnonzero(np.abs(audio) > floor)
    if loud.size == 0:
        return audio
    start = max(0, int(loud[0]) - int(RATE * 0.01))
    end = min(audio.shape[0], int(loud[-1]) + int(RATE * 0.03))
    return audio[start:end]
