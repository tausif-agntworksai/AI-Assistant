"""Offline TTS via the Windows Speech API. Windows only.

Fallback for when Edge TTS can't reach the network. Note the real limitation on
this machine: only en-US voices (David, Zira) are installed, so Hindi text is
pronounced with an English phoneme set and sounds wrong. Installing the Windows
Hindi language pack adds `Microsoft Hemant` and fixes it offline.

Constructing this on a machine without SAPI raises, which is what lets
`create_speaker` fall through to the next backend — see the comment in
`__init__`.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from .base import Speaker

log = logging.getLogger(__name__)

SVSF_ASYNC = 1
SVSF_PURGE_BEFORE_SPEAK = 2


class SapiSpeaker(Speaker):
    name = "sapi"
    available = True

    def __init__(self, rate: int = 1) -> None:
        # Fail construction if SAPI isn't reachable, rather than at the first
        # utterance. Nothing in this class used to import anything until
        # `_ensure_voice`, so `available = True` was a claim about every
        # platform — and off Windows `create_speaker` would hand back a speaker
        # that accepted every reply and played none of them. Silence reads as a
        # crash; NullSpeaker's honest text-only mode does not.
        #
        # A bare import is deliberate: `_ensure_voice` calls `CoInitialize`,
        # which binds a COM apartment to the calling thread, and speaking
        # happens on a worker thread rather than this one.
        import pythoncom  # noqa: F401
        import win32com.client  # noqa: F401

        self.rate = rate
        self._voice = None
        self._lock = threading.Lock()
        self._voices: list[tuple[str, str]] = []

    def _ensure_voice(self):
        if self._voice is not None:
            return self._voice
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        self._voice = win32com.client.Dispatch("SAPI.SpVoice")
        self._voice.Rate = self.rate

        self._voices = []
        for token in self._voice.GetVoices():
            try:
                self._voices.append((token.GetDescription(), token.GetAttribute("Language")))
            except Exception:  # noqa: BLE001 - malformed voice tokens exist in the wild
                continue
        log.info("SAPI voices: %s", ", ".join(name for name, _ in self._voices) or "none")
        return self._voice

    def _select_voice(self, language: str) -> None:
        """Pick a matching installed voice, if one exists."""
        voice = self._ensure_voice()
        want_hindi = (language or "").lower().startswith("hi")
        # SAPI language IDs are hex LCIDs: 439 = hi-IN, 409/809 = en-US/en-GB.
        wanted = "439" if want_hindi else "409"

        for token in voice.GetVoices():
            try:
                if wanted in str(token.GetAttribute("Language")):
                    voice.Voice = token
                    return
            except Exception:  # noqa: BLE001
                continue

        if want_hindi:
            log.warning(
                "No Hindi SAPI voice installed - speaking Hindi with an English voice. "
                "Install the Windows Hindi language pack for offline Hindi speech."
            )

    def synthesize(self, text: str, language: str = "en") -> tuple[np.ndarray, int]:
        # SAPI plays through its own audio path; we don't expose raw PCM.
        return np.zeros(0, dtype=np.float32), 16000

    def speak(self, text: str, language: str = "en") -> bool:
        text = text.strip()
        if not text:
            return True
        with self._lock:
            try:
                voice = self._ensure_voice()
                self._select_voice(language)
                voice.Speak(text, SVSF_ASYNC)
                voice.WaitUntilDone(120_000)
                return True
            except Exception as exc:  # noqa: BLE001
                log.error("SAPI speech failed: %s", exc)
                return False

    def stop(self) -> None:
        if self._voice is None:
            return
        try:
            self._voice.Speak("", SVSF_ASYNC | SVSF_PURGE_BEFORE_SPEAK)
        except Exception as exc:  # noqa: BLE001
            log.debug("SAPI stop failed: %s", exc)
