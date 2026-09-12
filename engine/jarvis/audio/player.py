"""Audio playback with barge-in.

Playback is non-blocking by design: the orchestrator keeps reading the
microphone while the assistant is talking, so saying the wake word mid-reply
can cut the speech off immediately. A blocking player would make that
impossible.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .capture import resolve_device

log = logging.getLogger(__name__)


class AudioPlayer:
    def __init__(self, device: int | str | None = None) -> None:
        self._device_spec = device
        self._stream = None
        self._stop_flag = threading.Event()
        self._finished = threading.Event()
        self._finished.set()
        self._lock = threading.Lock()
        self.interrupted = False
        self._earcon = False
        #: When the most recent playback actually started, as a monotonic
        #: timestamp. The speaker synthesises before it can play anything, so
        #: this is the only point that answers "how long after we decided to
        #: reply did the user hear a sound?" — which is the latency that is
        #: felt, as opposed to how long the reply then took to say.
        self.last_started_at = 0.0

    @property
    def is_playing(self) -> bool:
        return not self._finished.is_set()

    @property
    def is_earcon(self) -> bool:
        """True while the thing playing is a short cue rather than a reply.

        The listen loop treats these completely differently. A reply that is
        playing can be barged in on and cut off; an acknowledgement is a tenth
        of a second long and exists precisely so the user speaks *next*, so
        cutting it off, or reading it as the assistant talking over itself,
        would both be wrong.
        """
        return self._earcon and self.is_playing

    def play_earcon(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play a short cue. Non-blocking, and not subject to barge-in."""
        self.play_async(audio, sample_rate, earcon=True)

    def play_async(
        self, audio: np.ndarray, sample_rate: int, earcon: bool = False
    ) -> None:
        """Start playback and return immediately.

        `earcon` is set inside the lock, before the stream starts. Setting it
        afterwards left a window — small, but the listen loop runs every 32 ms —
        in which a cue looked like a reply and got cut off as barge-in.
        """
        import sounddevice as sd

        self.stop()
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if audio.size == 0:
            return

        with self._lock:
            self._stop_flag.clear()
            self._finished.clear()
            self.interrupted = False
            self._earcon = earcon
            position = 0

            def callback(outdata, frames, time_info, status):  # noqa: ANN001
                nonlocal position
                if status:
                    log.debug("Output stream status: %s", status)
                if self._stop_flag.is_set():
                    outdata[:] = 0
                    raise sd.CallbackStop

                chunk = audio[position : position + frames]
                n = chunk.shape[0]
                outdata[:n, 0] = chunk
                if n < frames:
                    outdata[n:, 0] = 0.0
                    position += n
                    raise sd.CallbackStop
                position += frames

            def on_finished() -> None:
                self._finished.set()

            self._stream = sd.OutputStream(
                device=resolve_device(self._device_spec, "output"),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                callback=callback,
                finished_callback=on_finished,
            )
            self._stream.start()
            self.last_started_at = time.monotonic()

    def wait(self, timeout: float | None = None) -> bool:
        """Block until playback ends. Returns False if it was interrupted."""
        self._finished.wait(timeout)
        self._close_stream()
        return not self.interrupted

    def play(self, audio: np.ndarray, sample_rate: int) -> bool:
        self.play_async(audio, sample_rate)
        return self.wait()

    def stop(self) -> None:
        """Cut playback off immediately (barge-in)."""
        if self._finished.is_set() and self._stream is None:
            return
        self.interrupted = True
        self._stop_flag.set()
        self._finished.wait(0.5)
        self._close_stream()

    def _close_stream(self) -> None:
        with self._lock:
            if self._stream is None:
                return
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("Error closing output stream: %s", exc)
            finally:
                self._stream = None
                self._finished.set()

    def close(self) -> None:
        self.stop()
