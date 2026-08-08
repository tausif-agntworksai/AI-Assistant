"""Optional cloud speech recognition.

Off by default — the assistant is local-first, and nothing here runs unless a
key is set in `.env`. Worth turning on when Hindi accuracy on longer dictation
matters more than keeping audio on the machine.

Both providers accept a WAV upload, so the audio is encoded in memory rather
than written to a temp file.
"""

from __future__ import annotations

import io
import logging
import time
import wave

import numpy as np

from .base import Transcriber, Transcript, is_probable_hallucination, is_repetition_loop

log = logging.getLogger(__name__)

TIMEOUT_SEC = 30


def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(np.asarray(audio, dtype=np.float32) * 32767.0, -32768, 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buffer.getvalue()


class CloudTranscriber(Transcriber):
    """Dispatches to whichever provider is configured."""

    def __init__(self, provider: str, api_key: str, language: str | None = None) -> None:
        self.provider = (provider or "").strip().lower()
        self.api_key = api_key
        self.language = language
        self.name = f"cloud:{self.provider}"
        if self.provider not in ("openai", "deepgram"):
            raise ValueError(
                f"Unknown CLOUD_STT_PROVIDER {provider!r}; expected 'openai' or 'deepgram'"
            )
        if not api_key:
            raise ValueError("CLOUD_STT_API_KEY is not set")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        audio = np.asarray(audio, dtype=np.float32).ravel()
        duration = audio.shape[0] / sample_rate
        if duration < 0.2:
            return Transcript(text="", audio_duration=duration, backend=self.name)

        t0 = time.perf_counter()
        try:
            if self.provider == "openai":
                text, language = self._openai(audio, sample_rate)
            else:
                text, language = self._deepgram(audio, sample_rate)
        except Exception as exc:  # noqa: BLE001 - never take the engine down
            log.error("Cloud transcription failed (%s): %s", self.provider, exc)
            return Transcript(text="", audio_duration=duration, backend=self.name,
                              latency=time.perf_counter() - t0)

        if is_repetition_loop(text) or is_probable_hallucination(text):
            text = ""

        result = Transcript(
            text=text.strip(),
            language=language or "en",
            language_probability=1.0 if language else 0.0,
            audio_duration=duration,
            latency=time.perf_counter() - t0,
            backend=self.name,
        )
        log.info("Cloud STT %.2fs audio -> %.2fs | %s", duration, result.latency, result)
        return result

    def _openai(self, audio: np.ndarray, sample_rate: int) -> tuple[str, str]:
        import requests

        response = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": ("speech.wav", _to_wav_bytes(audio, sample_rate), "audio/wav")},
            data={
                "model": "whisper-1",
                "response_format": "verbose_json",
                **({"language": self.language} if self.language else {}),
            },
            timeout=TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("text", ""), payload.get("language", "")

    def _deepgram(self, audio: np.ndarray, sample_rate: int) -> tuple[str, str]:
        import requests

        # nova-3 with detect_language handles Hindi/English code-switching,
        # which is the reason to reach for a cloud provider at all.
        params = {"model": "nova-3", "smart_format": "true", "detect_language": "true"}
        if self.language:
            params.pop("detect_language")
            params["language"] = self.language

        response = requests.post(
            "https://api.deepgram.com/v1/listen",
            headers={"Authorization": f"Token {self.api_key}", "Content-Type": "audio/wav"},
            params=params,
            data=_to_wav_bytes(audio, sample_rate),
            timeout=TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()

        channel = payload["results"]["channels"][0]
        alternative = channel["alternatives"][0]
        return alternative.get("transcript", ""), channel.get("detected_language", "")


class FallbackTranscriber(Transcriber):
    """Local recognition, with a cloud retry when the local pass comes back empty.

    This is the "local-first, cloud when it matters" arrangement: nothing
    leaves the machine unless the local model already failed to make sense of
    the utterance.
    """

    name = "local+cloud"

    def __init__(self, local: Transcriber, cloud: Transcriber) -> None:
        self.local = local
        self.cloud = cloud

    def warmup(self) -> None:
        self.local.warmup()

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> Transcript:
        result = self.local.transcribe(audio, sample_rate)
        if result.is_empty and audio.shape[0] / sample_rate >= 0.5:
            log.info("Local recognition came back empty — retrying in the cloud")
            cloud_result = self.cloud.transcribe(audio, sample_rate)
            if not cloud_result.is_empty:
                return cloud_result
        return result

    def close(self) -> None:
        self.local.close()
        self.cloud.close()
