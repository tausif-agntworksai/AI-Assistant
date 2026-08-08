"""Voice activity detection and utterance segmentation.

Silero VAD is run directly through onnxruntime rather than via the `silero-vad`
PyPI package, which would drag in torch (~2 GB) for a 2 MB model. An
energy-based detector is always available as a fallback so a failed model
download degrades the experience instead of breaking it.
"""

from __future__ import annotations

import enum
import logging
from pathlib import Path

import numpy as np

from .. import paths

log = logging.getLogger(__name__)

SILERO_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
SILERO_FRAME = 512  # samples @ 16 kHz — the model accepts nothing else
# v5 wants this much of the preceding frame prepended to each window.
SILERO_CONTEXT_16K = 64
SILERO_CONTEXT_8K = 32


class VadBackend:
    """A detector that scores fixed-size frames for speech probability."""

    frame_size: int = SILERO_FRAME

    def probability(self, frame: np.ndarray) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class EnergyVad(VadBackend):
    """RMS detector with an adaptive noise floor.

    Calibrates against the quietest recent frames, so it adapts to fan noise or
    a noisy room instead of relying on a fixed threshold that would be wrong in
    both directions.
    """

    frame_size = SILERO_FRAME

    def __init__(self, sensitivity: float = 3.0) -> None:
        self.sensitivity = sensitivity
        self._noise_floor = 0.005
        self._warmup = 0

    def probability(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))) + 1e-9)
        if self._warmup < 20:
            self._warmup += 1
            self._noise_floor = max(1e-4, 0.9 * self._noise_floor + 0.1 * rms)
            return 0.0
        if rms < self._noise_floor * self.sensitivity:
            # Track the floor upward slowly during silence only.
            self._noise_floor = 0.995 * self._noise_floor + 0.005 * rms
            return 0.0
        ratio = rms / (self._noise_floor * self.sensitivity)
        return float(min(1.0, 0.5 + 0.5 * min(1.0, (ratio - 1.0))))

    def reset(self) -> None:
        self._warmup = 0


class SileroVad(VadBackend):
    """Silero VAD via onnxruntime. Handles both the v4 and v5 model layouts.

    v5 expects each 512-sample frame to arrive with the previous 64 samples
    glued to its front. Omitting that context is not an error the model can
    report - it just scores every frame near zero, which reads downstream as
    permanent silence rather than as a bug.
    """

    frame_size = SILERO_FRAME

    def __init__(self, model_path: Path | None = None, sample_rate: int = 16000) -> None:
        import onnxruntime as ort

        self.sample_rate = sample_rate
        self.context_size = SILERO_CONTEXT_8K if sample_rate == 8000 else SILERO_CONTEXT_16K
        path = model_path or ensure_silero_model()

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

        names = {i.name for i in self._session.get_inputs()}
        self._is_v5 = "state" in names
        self._batch = 1
        self.reset()
        log.info("Silero VAD loaded (%s layout) from %s", "v5" if self._is_v5 else "v4", path.name)

    def reset(self) -> None:
        if self._is_v5:
            self._state = np.zeros((2, self._batch, 128), dtype=np.float32)
            # Silence in front of the first frame, so the utterance starts clean.
            self._context = np.zeros(self.context_size, dtype=np.float32)
        else:
            self._h = np.zeros((2, self._batch, 64), dtype=np.float32)
            self._c = np.zeros((2, self._batch, 64), dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        frame = frame.astype(np.float32, copy=False).ravel()
        if frame.shape[0] != self.frame_size:
            raise ValueError(f"Silero needs {self.frame_size} samples, got {frame.shape[0]}")

        sr = np.array(self.sample_rate, dtype=np.int64)
        if self._is_v5:
            x = np.concatenate((self._context, frame)).reshape(1, -1)
            out, self._state = self._session.run(None, {"input": x, "state": self._state, "sr": sr})
            self._context = frame[-self.context_size:].copy()
        else:
            x = frame.reshape(1, -1)
            out, self._h, self._c = self._session.run(
                None, {"input": x, "h": self._h, "c": self._c, "sr": sr}
            )
        return float(np.asarray(out).ravel()[0])


def ensure_silero_model() -> Path:
    """Return the local model path, downloading it once if needed."""
    paths.ensure_dirs()
    dest = paths.MODELS_DIR / "silero_vad.onnx"
    if dest.exists() and dest.stat().st_size > 100_000:
        return dest

    import requests

    log.info("Downloading Silero VAD model (~2 MB)...")
    resp = requests.get(SILERO_URL, timeout=60)
    resp.raise_for_status()
    tmp = dest.with_suffix(".onnx.part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)  # atomic, so an interrupted download can't leave a corrupt model
    log.info("Silero VAD model saved to %s", dest)
    return dest


def create_vad(backend: str = "silero") -> VadBackend:
    """Build the configured detector, falling back to energy on any failure."""
    if backend == "energy":
        return EnergyVad()
    try:
        return SileroVad()
    except Exception as exc:  # noqa: BLE001
        log.warning("Silero VAD unavailable (%s) - falling back to energy detection", exc)
        return EnergyVad()


class SegmentResult(enum.Enum):
    WAITING = "waiting"      # no speech yet
    SPEAKING = "speaking"    # mid-utterance
    COMPLETE = "complete"    # trailing silence reached; audio is ready
    TOO_SHORT = "too_short"  # speech ended but was only a blip
    TIMEOUT = "timeout"      # hit max_utterance_sec; audio is ready but truncated
    SILENT = "silent"        # nothing was ever said


class SpeechSegmenter:
    """Turns a frame stream into one utterance, ended by trailing silence.

    Push frames until the result is terminal, then read `.audio`.
    """

    def __init__(
        self,
        vad: VadBackend,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        silence_ms: int = 700,
        min_speech_ms: int = 250,
        max_utterance_sec: int = 15,
        no_speech_timeout_sec: float = 6.0,
    ) -> None:
        self.vad = vad
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.frame_ms = 1000.0 * vad.frame_size / sample_rate
        self.silence_frames = max(1, int(silence_ms / self.frame_ms))
        self.min_speech_frames = max(1, int(min_speech_ms / self.frame_ms))
        self.max_frames = int(max_utterance_sec * 1000 / self.frame_ms)
        self.no_speech_frames = int(no_speech_timeout_sec * 1000 / self.frame_ms)
        self.reset()

    def reset(self, preroll: np.ndarray | None = None) -> None:
        self.vad.reset()
        self._chunks: list[np.ndarray] = []
        if preroll is not None and preroll.size:
            self._chunks.append(preroll.astype(np.float32, copy=False))
        self._speech_frames = 0
        self._silence_run = 0
        self._total_frames = 0
        self._started = False
        self.peak_level = 0.0

    def push(self, frame: np.ndarray) -> SegmentResult:
        self._chunks.append(frame)
        self._total_frames += 1

        prob = self.vad.probability(frame)
        is_speech = prob >= self.threshold
        self.peak_level = max(self.peak_level, float(np.abs(frame).max()))

        if is_speech:
            self._speech_frames += 1
            self._silence_run = 0
            self._started = True
        elif self._started:
            self._silence_run += 1

        if not self._started:
            if self._total_frames >= self.no_speech_frames:
                return SegmentResult.SILENT
            return SegmentResult.WAITING

        if self._total_frames >= self.max_frames:
            return SegmentResult.TIMEOUT

        if self._silence_run >= self.silence_frames:
            if self._speech_frames < self.min_speech_frames:
                return SegmentResult.TOO_SHORT
            return SegmentResult.COMPLETE

        return SegmentResult.SPEAKING

    @property
    def audio(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks)

    @property
    def duration_sec(self) -> float:
        return sum(c.shape[0] for c in self._chunks) / self.sample_rate
