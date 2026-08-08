"""Microphone capture.

The sounddevice callback runs on a realtime audio thread: anything slow in
there causes input overflows and dropped frames. So the callback does exactly
one thing — copy the block into a queue — and all inference (wake word, VAD)
happens on the consumer thread.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)


# Virtual and loopback inputs that Windows will happily nominate as the system
# default. Picking one of these means the assistant hears silence forever and
# looks broken, so an unconfigured install skips past them to a real mic.
_VIRTUAL_INPUTS = (
    "droidcam", "eshare", "stereo mix", "virtual", "vb-audio", "cable output",
    "voicemeeter", "obs", "nvidia broadcast", "sound mapper", "primary sound",
    "wave", "midi", "what u hear", "loopback",
)


def _is_virtual(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _VIRTUAL_INPUTS)


def preferred_input_device() -> int | None:
    """Pick a real microphone when the config doesn't name one.

    Returns None (meaning "use the system default") whenever the default is
    already a physical device — we only override when it is a virtual input.
    """
    import sounddevice as sd

    try:
        devices = sd.query_devices()
        default_index = sd.default.device[0]
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not inspect audio devices: %s", exc)
        return None

    if isinstance(default_index, int) and 0 <= default_index < len(devices):
        default_name = devices[default_index]["name"]
        if not _is_virtual(default_name):
            return None
        log.info("System default input %r looks virtual — choosing a physical mic",
                 default_name)

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0 and not _is_virtual(dev["name"]):
            log.info("Using input device [%d] %s", idx, dev["name"])
            return idx

    log.warning("No physical microphone found; falling back to the system default")
    return None


def resolve_device(spec: int | str | None, kind: str = "input") -> int | None:
    """Resolve a config value to a sounddevice index.

    Accepts an index, a case-insensitive substring of the device name, or None.
    Falls back with a warning rather than failing outright, so a renamed or
    unplugged mic doesn't stop startup.
    """
    import sounddevice as sd

    if isinstance(spec, int):
        return spec
    if spec is None:
        return preferred_input_device() if kind == "input" else None

    needle = str(spec).lower()
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    for idx, dev in enumerate(sd.query_devices()):
        if dev[channel_key] > 0 and needle in dev["name"].lower():
            log.info("Resolved %s device %r -> [%d] %s", kind, spec, idx, dev["name"])
            return idx

    log.warning("No %s device matching %r; falling back", kind, spec)
    return preferred_input_device() if kind == "input" else None


class FrameAccumulator:
    """Re-windows a stream of blocks into fixed-size frames.

    Consumers need different window sizes from the same stream — openWakeWord
    wants 1280 samples, Silero VAD wants exactly 512 — so each keeps its own
    accumulator and the capture block size stays independent of both.
    """

    def __init__(self, frame_size: int) -> None:
        self.frame_size = frame_size
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        self._buf = np.concatenate((self._buf, block.astype(np.float32, copy=False)))
        frames: list[np.ndarray] = []
        while self._buf.shape[0] >= self.frame_size:
            frames.append(self._buf[: self.frame_size].copy())
            self._buf = self._buf[self.frame_size :]
        return frames

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)


class RingBuffer:
    """Fixed-duration rolling audio history.

    This is what makes the assistant feel responsive rather than clipped: when
    the wake word fires we have already-captured audio from *before* the
    trigger, so a command spoken immediately after "hey jarvis" isn't truncated.
    """

    def __init__(self, capacity_samples: int) -> None:
        self.capacity = capacity_samples
        self._buf = np.zeros(capacity_samples, dtype=np.float32)
        self._filled = 0
        self._pos = 0

    def push(self, block: np.ndarray) -> None:
        block = block.astype(np.float32, copy=False)
        n = block.shape[0]
        if n >= self.capacity:
            self._buf[:] = block[-self.capacity :]
            self._pos = 0
            self._filled = self.capacity
            return
        end = self._pos + n
        if end <= self.capacity:
            self._buf[self._pos : end] = block
        else:
            split = self.capacity - self._pos
            self._buf[self._pos :] = block[:split]
            self._buf[: end - self.capacity] = block[split:]
        self._pos = end % self.capacity
        self._filled = min(self.capacity, self._filled + n)

    def read(self, samples: int | None = None) -> np.ndarray:
        """Return the most recent `samples` in chronological order."""
        want = min(samples or self._filled, self._filled)
        if want == 0:
            return np.zeros(0, dtype=np.float32)
        start = (self._pos - want) % self.capacity
        if start + want <= self.capacity:
            return self._buf[start : start + want].copy()
        return np.concatenate((self._buf[start:], self._buf[: (start + want) % self.capacity]))

    def clear(self) -> None:
        self._filled = 0
        self._pos = 0


class AudioCapture:
    """Owns the input stream and hands mono float32 blocks to one consumer."""

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 16000,
        block_size: int = 512,
        queue_max: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._device_spec = device
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_max)
        self._stream = None
        self._running = threading.Event()
        self.overflow_count = 0
        self.dropped_blocks = 0

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            if getattr(status, "input_overflow", False):
                self.overflow_count += 1
            else:
                log.debug("Input stream status: %s", status)
        try:
            self._queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # Dropping the newest block is better than blocking the audio
            # thread, which would cascade into more overflows.
            self.dropped_blocks += 1

    def start(self) -> None:
        import sounddevice as sd

        if self._running.is_set():
            return
        device = resolve_device(self._device_spec, "input")
        self._stream = sd.InputStream(
            device=device,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._running.set()
        name = sd.query_devices(device if device is not None else sd.default.device[0])["name"]
        log.info("Microphone open: %s @ %d Hz, %d-sample blocks",
                 name, self.sample_rate, self.block_size)

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("Error closing stream: %s", exc)
            self._stream = None
        self._queue.put(None)  # unblock any waiting consumer
        if self.overflow_count or self.dropped_blocks:
            log.info("Capture finished with %d overflow(s), %d dropped block(s)",
                     self.overflow_count, self.dropped_blocks)

    @property
    def running(self) -> bool:
        return self._running.is_set()

    def blocks(self, timeout: float = 0.5) -> Iterator[np.ndarray]:
        """Yield captured blocks until stopped. Blocks the calling thread."""
        while self._running.is_set():
            try:
                block = self._queue.get(timeout=timeout)
            except queue.Empty:
                continue
            if block is None:
                break
            yield block

    def drain(self) -> None:
        """Discard buffered audio.

        Called after speaking so the assistant doesn't transcribe its own voice
        echoing back through the microphone.
        """
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def rms_level(block: np.ndarray) -> float:
    """Normalised 0..1 loudness, for the HUD waveform."""
    if block.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    return min(1.0, rms * 8.0)
