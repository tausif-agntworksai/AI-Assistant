"""Audio decoding — MP3/WAV/anything to mono float32.

Edge TTS returns MP3 and `--test-stt` takes arbitrary audio files, so both
paths need a decoder. PyAV handles everything; the stdlib `wave` module is the
fallback for plain WAV when PyAV is missing.
"""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def decode_audio(source: bytes | str | Path, target_rate: int | None = None
                 ) -> tuple[np.ndarray, int]:
    """Decode to (mono float32 in [-1, 1], sample_rate)."""
    try:
        return _decode_with_av(source, target_rate)
    except Exception as exc:  # noqa: BLE001
        log.debug("PyAV decode failed (%s); trying wave", exc)
        pcm, rate = _decode_wav(source)
        if target_rate and rate != target_rate:
            pcm = resample(pcm, rate, target_rate)
            rate = target_rate
        return pcm, rate


def _decode_with_av(source: bytes | str | Path, target_rate: int | None
                    ) -> tuple[np.ndarray, int]:
    import av

    handle = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    with av.open(handle) as container:
        stream = container.streams.audio[0]
        rate = target_rate or stream.rate or 16000
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)

        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().ravel())
        # Flush whatever the resampler is still holding.
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().ravel())

    if not chunks:
        return np.zeros(0, dtype=np.float32), rate
    return np.concatenate(chunks).astype(np.float32, copy=False), rate


def _decode_wav(source: bytes | str | Path) -> tuple[np.ndarray, int]:
    handle = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    with wave.open(handle, "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes")

    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        data = (data - 128.0) / 128.0
    else:
        data /= float(np.iinfo(dtype).max)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32, copy=False), rate


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio
    n_out = int(round(audio.shape[0] * dst_rate / src_rate))
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    """Write mono float32 audio as 16-bit PCM. Used for debug dumps."""
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
