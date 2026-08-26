"""Neural TTS via Microsoft Edge voices.

This is the default because the machine has no Hindi SAPI voice installed —
Windows literally cannot pronounce Hindi out of the box here. Edge TTS is free,
needs no key, and provides natural hi-IN and en-IN voices.

It does need internet. `SapiSpeaker` covers English offline.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..audio.decode import decode_audio
from ..audio.player import AudioPlayer
from .base import Speaker, chunk_text

log = logging.getLogger(__name__)

# Short acknowledgements are spoken constantly; caching them removes a network
# round-trip from the most latency-sensitive replies.
_CACHE_MAX_CHARS = 60
_CACHE_MAX_ENTRIES = 64


def trim_silence(audio: np.ndarray, rate: int, pad_ms: int = 30) -> np.ndarray:
    """Strip the silence Edge pads onto every clip.

    Measured on this voice: "Mm-hmm?" comes back as 1.78 s of audio holding
    0.58 s of speech — a full second of trailing nothing. For a wake-word
    acknowledgement that silence is dead time before the microphone opens, and
    on ordinary replies it is a pause before the assistant will take the next
    instruction. Only the ends are touched, so pauses between sentences stay.
    """
    if audio.size == 0:
        return audio

    peak = float(np.abs(audio).max())
    if peak <= 0.0:
        return audio

    loud = np.flatnonzero(np.abs(audio) > peak * 0.02)
    if loud.size == 0:
        return audio

    pad = int(rate * pad_ms / 1000)
    start = max(0, int(loud[0]) - pad)
    end = min(audio.shape[0], int(loud[-1]) + pad)
    return audio[start:end]


class EdgeSpeaker(Speaker):
    name = "edge-tts"
    available = True

    def __init__(
        self,
        voice_hi: str = "hi-IN-SwaraNeural",
        voice_en: str = "en-IN-NeerjaExpressiveNeural",
        rate: str = "-4%",
        volume: str = "+0%",
        pitch: str = "+3Hz",
        player: AudioPlayer | None = None,
    ) -> None:
        self.voice_hi = voice_hi
        self.voice_en = voice_en
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.player = player or AudioPlayer()
        self._cache: dict[tuple[str, str], tuple[np.ndarray, int]] = {}
        self._cache_lock = threading.Lock()
        self._cancel = threading.Event()

    def voice_for(self, language: str) -> str:
        return self.voice_hi if (language or "").lower().startswith("hi") else self.voice_en

    # -- synthesis ----------------------------------------------------------

    async def _synth_async(self, text: str, voice: str) -> bytes:
        import edge_tts

        comm = edge_tts.Communicate(
            text, voice, rate=self.rate, volume=self.volume, pitch=self.pitch
        )
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        return bytes(buf)

    def synthesize(self, text: str, language: str = "en") -> tuple[np.ndarray, int]:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32), 24000

        voice = self.voice_for(language)
        key = (text, voice)
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]

        mp3 = asyncio.run(self._synth_async(text, voice))
        audio, rate = decode_audio(mp3)
        audio = trim_silence(audio, rate)

        if len(text) <= _CACHE_MAX_CHARS:
            with self._cache_lock:
                if len(self._cache) >= _CACHE_MAX_ENTRIES:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = (audio, rate)
        return audio, rate

    # -- playback -----------------------------------------------------------

    def speak(self, text: str, language: str = "en") -> bool:
        """Speak `text`, synthesizing the next chunk while the current one plays."""
        chunks = chunk_text(text)
        if not chunks:
            return True

        self._cancel.clear()
        log.info("Speaking (%s, %d chunk(s)): %s", self.voice_for(language), len(chunks),
                 text[:80] + ("..." if len(text) > 80 else ""))

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts") as pool:
            pending = pool.submit(self.synthesize, chunks[0], language)
            for i, _ in enumerate(chunks):
                try:
                    audio, rate = pending.result(timeout=30)
                except Exception as exc:  # noqa: BLE001
                    log.warning("TTS synthesis failed: %s", exc)
                    return False

                if i + 1 < len(chunks):
                    pending = pool.submit(self.synthesize, chunks[i + 1], language)

                if self._cancel.is_set():
                    return False
                if audio.size and not self.player.play(audio, rate):
                    return False  # barge-in
        return True

    def stop(self) -> None:
        self._cancel.set()
        self.player.stop()

    def close(self) -> None:
        self.stop()
        self.player.close()


async def list_voices(language_prefix: str = "") -> list[dict]:
    """Available Edge voices, optionally filtered by locale prefix (e.g. 'hi')."""
    import edge_tts

    voices = await edge_tts.list_voices()
    if language_prefix:
        voices = [v for v in voices if v["Locale"].lower().startswith(language_prefix.lower())]
    return sorted(voices, key=lambda v: (v["Locale"], v["ShortName"]))
