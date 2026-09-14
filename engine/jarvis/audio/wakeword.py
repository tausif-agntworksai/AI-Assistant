"""Wake word detection — fully local ONNX inference.

This is the privacy boundary of the whole assistant: while idle, audio is
scored here and nowhere else. Nothing is transcribed, stored, or sent anywhere
until this module reports a detection.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np

log = logging.getLogger(__name__)

# openWakeWord's feature extractor is built around 80 ms windows.
OWW_FRAME = 1280

AVAILABLE_MODELS = ("hey_jarvis", "alexa", "hey_mycroft", "hey_rhasspy")


class WakeWordDetector:
    frame_size: int = OWW_FRAME
    available: bool = False

    def detect(self, frame: np.ndarray) -> float:
        """Return the detection score for this frame (0..1)."""
        return 0.0

    def reset(self) -> None:
        pass


class NullWakeWord(WakeWordDetector):
    """Used when models can't be loaded. Hotkey and HUD click still work."""

    available = False

    def detect(self, frame: np.ndarray) -> float:
        return 0.0


class OpenWakeWord(WakeWordDetector):
    frame_size = OWW_FRAME
    available = True

    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.5,
                 cooldown_sec: float = 2.0, confirm_frames: int = 2,
                 confirm_window: int = 3) -> None:
        from openwakeword.model import Model

        ensure_models(model)
        self.model_name = model
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec
        self.confirm_frames = max(1, int(confirm_frames))
        self.confirm_window = max(self.confirm_frames, int(confirm_window))
        self._last_fire = 0.0
        self._recent: deque[bool] = deque(maxlen=self.confirm_window)

        self._model = Model(wakeword_models=[model], inference_framework="onnx")
        # The key openWakeWord reports back is not always the string we passed
        # (it uses the model filename stem, e.g. "hey_jarvis_v0.1").
        self._key = next(
            (k for k in self._model.models if model in k),
            next(iter(self._model.models), model),
        )
        log.info("Wake word ready: %r (threshold %.2f)", self._key, threshold)

    def detect(self, frame: np.ndarray) -> float:
        # openWakeWord expects int16 PCM.
        pcm = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16)
        scores = self._model.predict(pcm)
        return float(scores.get(self._key, 0.0))

    def triggered(self, frame: np.ndarray) -> bool:
        """Score the frame, then require the score to hold across frames.

        The threshold alone is a per-frame decision, and 80 ms is short enough
        that a cough or a consonant off the television can clear it once. The
        wake word spoken by a person holds the score up over several
        consecutive frames, so asking for `confirm_frames` hits out of the
        last `confirm_window` separates the two at a cost of one or two frames
        of latency.
        """
        score = self.detect(frame)
        self._recent.append(score >= self.threshold)

        if sum(self._recent) < self.confirm_frames:
            return False

        now = time.monotonic()
        if now - self._last_fire < self.cooldown_sec:
            return False
        self._last_fire = now
        # Read the window before resetting it, or the log reports the empty
        # window rather than the one that caused the detection — every line
        # said "0/0 frames", which is exactly the number a person checking
        # whether confirmation works would most like to see.
        hits, window = sum(self._recent), len(self._recent)
        self.reset()  # clear feature buffers so the next detection starts clean
        log.info(
            "Wake word detected (score %.2f, %d/%d frames)", score, hits, window,
        )
        return True

    def reset(self) -> None:
        # The confirmation window has to go too: leaving it full would let the
        # tail of one detection count toward the next.
        self._recent.clear()
        try:
            self._model.reset()
        except Exception:  # noqa: BLE001 - older versions lack reset()
            pass


def ensure_models(model: str = "hey_jarvis") -> None:
    """Download openWakeWord's models on first run.

    Downloads the shared feature extractors plus the requested wake word. This
    is the one network call the assistant makes before it can run offline.
    """
    import openwakeword.utils

    try:
        openwakeword.utils.download_models(model_names=[model])
    except TypeError:
        # Older signature takes no arguments and fetches everything.
        openwakeword.utils.download_models()


def create_wakeword(model: str = "hey_jarvis", threshold: float = 0.5,
                    cooldown_sec: float = 2.0, confirm_frames: int = 2,
                    confirm_window: int = 3) -> WakeWordDetector:
    """Build the detector, degrading to hotkey-only rather than failing."""
    try:
        return OpenWakeWord(model, threshold, cooldown_sec,
                            confirm_frames, confirm_window)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Wake word unavailable (%s: %s) - use the hotkey or HUD to talk",
            type(exc).__name__, exc,
        )
        return NullWakeWord()
