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
CUES: dict[str, tuple[str, ...]] = {
    "en": ("Yes?", "Mm?"),
    "hi": ("हाँ?", "जी?"),
}

# Faster than the assistant's speaking voice, on purpose. A one-syllable
# acknowledgement delivered at conversational pace is mostly attack and tail:
# measured here, "Yes?" runs 472 ms at the normal rate and 382 ms at this one,
# and the difference is 90 ms less of your command being thrown away.
CUE_RATE = "+40%"
# Part of the cache filename, so changing the rate above re-renders rather
# than silently reusing clips recorded at the old one.
_RATE_TAG = CUE_RATE.strip("+%")


def chime(rate: int = RATE) -> np.ndarray:
    """A two-note rising blip, about 140 ms.

    Rising rather than falling because a rising pair reads as a question — "go
    on" — where a falling pair reads as completion. Every assistant that does
    this uses the same grammar.

    Shaped with a raised-cosine envelope: a bare sine switched on and off clicks
    audibly at both ends, and a click is the least reassuring sound a machine
    can make when you are trying to tell whether it heard you.
    """
    notes = ((880.0, 0.06), (1174.7, 0.08))  # A5 then D6, a rising fourth
    parts: list[np.ndarray] = []
    for frequency, seconds in notes:
        samples = int(rate * seconds)
        t = np.arange(samples) / rate
        envelope = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(samples) / max(1, samples - 1))
        parts.append(np.sin(2 * np.pi * frequency * t) * envelope * 0.18)
    return np.concatenate(parts).astype(np.float32)


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

        return create_speaker(settings.tts.model_copy(update={"rate": CUE_RATE}))

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
        return _trim_silence(audio)

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
