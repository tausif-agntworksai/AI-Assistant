"""Local speech recognition with faster-whisper (CPU, int8).

Language is deliberately left to auto-detection: this assistant is spoken to in
Hindi, English and a mix of both, often within one sentence, and pinning
`language` would force one of them to be transcribed wrongly.

Every clip is conditioned first (see `audio/enhance.py`). Whisper's accuracy on
a quiet, rumbly laptop microphone is not a fixed property of the model — a
normalised clip is a materially easier problem than the raw one, and that
difference is the difference between being understood the first time and being
asked to repeat.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from .. import paths
from ..audio.enhance import condition
from .base import (
    Transcriber,
    Transcript,
    is_probable_hallucination,
    is_repetition_loop,
)

log = logging.getLogger(__name__)

# Nudges Whisper toward command vocabulary and Hinglish spelling. Kept short —
# a long prompt increases the chance of the model echoing it back as output.
DEFAULT_PROMPT = (
    "Chrome, YouTube, WhatsApp, VS Code, Spotify kholo. "
    "Volume badhao. Screenshot lo. Timer laga do. What's the battery?"
)

# A yes/no reply is one or two syllables. Telling the decoder what shape of
# answer to expect stops it inventing a sentence out of "haan".
ANSWER_PROMPT = "Haan. Nahi. Yes. No. Theek hai. Rehne do. Bilkul. Cancel."

# The hint (installed app names) is appended to the prompt. Whisper's prompt
# window is 224 tokens and anything near that starts leaking into the output.
MAX_HINT_CHARS = 220


class LocalWhisper(Transcriber):
    name = "faster-whisper"

    def __init__(
        self,
        model: str = "small",
        compute_type: str = "int8",
        cpu_threads: int = 6,
        language: str | None = None,
        beam_size: int = 5,
        initial_prompt: str | None = DEFAULT_PROMPT,
        best_of: int = 5,
    ) -> None:
        self.model_size = model
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.language = language
        self.beam_size = beam_size
        self.best_of = best_of
        self.initial_prompt = initial_prompt
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from faster_whisper import WhisperModel

            paths.ensure_dirs()
            t0 = time.perf_counter()
            log.info("Loading Whisper %r (%s)... first run downloads the model",
                     self.model_size, self.compute_type)
            self._model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                download_root=str(paths.MODELS_DIR / "whisper"),
            )
            log.info("Whisper %r ready in %.1fs", self.model_size,
                     time.perf_counter() - t0)
        return self._model

    def warmup(self) -> None:
        model = self._ensure_model()
        # Half a second of silence is enough to build the compute graph.
        silence = np.zeros(8000, dtype=np.float32)
        list(model.transcribe(silence, beam_size=1)[0])
        log.debug("Whisper %r warmup complete", self.model_size)

    def _prompt(self, short_answer: bool, hint: str | None) -> str | None:
        if short_answer:
            return ANSWER_PROMPT
        base = self.initial_prompt or ""
        if hint:
            base = f"{base} {hint[:MAX_HINT_CHARS]}".strip()
        return base or None

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        *,
        short_answer: bool = False,
        hint: str | None = None,
    ) -> Transcript:
        model = self._ensure_model()
        audio = np.asarray(audio, dtype=np.float32).ravel()
        duration = audio.shape[0] / sample_rate

        if sample_rate != 16000:
            audio = _resample_to_16k(audio, sample_rate)

        if duration < 0.20:
            return Transcript(text="", audio_duration=duration, backend=self.name)

        prepared = condition(audio, 16000)

        t0 = time.perf_counter()
        segments, info = model.transcribe(
            prepared,
            language=self.language,
            beam_size=self.beam_size,
            best_of=self.best_of,
            vad_filter=False,          # we already segmented the utterance
            condition_on_previous_text=False,  # stops repetition loops on short commands
            initial_prompt=self._prompt(short_answer, hint),
            # A temperature *list* enables fallback decoding: a clean clip is
            # decoded once at 0.0, and only output that trips the
            # compression-ratio check is retried hotter. That's what breaks a
            # decoder stuck repeating one token.
            temperature=[0.0, 0.2, 0.4, 0.6],
            compression_ratio_threshold=2.4,
            # Whisper's own confidence gates. Left at their defaults the model
            # will happily return its best guess at pure noise; with these it
            # says nothing instead, which the orchestrator can then answer with
            # "say that again?" rather than acting on an invented command.
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            repetition_penalty=1.1,
        )

        parts: list[dict] = []
        text_chunks: list[str] = []
        no_speech: list[float] = []
        logprobs: list[float] = []
        for seg in segments:
            text_chunks.append(seg.text)
            no_speech.append(getattr(seg, "no_speech_prob", 0.0))
            seg_logprob = getattr(seg, "avg_logprob", 0.0)
            logprobs.append(seg_logprob)
            parts.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
                "avg_logprob": seg_logprob,
            })

        text = " ".join(c.strip() for c in text_chunks).strip()
        latency = time.perf_counter() - t0
        avg_logprob = float(np.mean(logprobs)) if logprobs else 0.0

        # Whisper fills silence with plausible-sounding filler; drop it rather
        # than letting the assistant act on a phantom command. `filtered` marks
        # the difference between "nothing was said" and "something was said and
        # I threw it away", so the caller can ask the user to repeat.
        filtered = False
        if is_repetition_loop(text):
            log.info("Discarded repetition loop: %r", text[:60])
            text, filtered = "", True
        elif no_speech and min(no_speech) > 0.6 and is_probable_hallucination(
            text, short_answer=short_answer
        ):
            log.debug("Discarded likely hallucination: %r (no_speech=%.2f)",
                      text, min(no_speech))
            text = ""
        elif is_probable_hallucination(text, short_answer=short_answer):
            log.debug("Discarded filler transcript: %r", text)
            text, filtered = "", bool(text.strip())

        result = Transcript(
            text=text,
            language=info.language or "en",
            language_probability=float(info.language_probability or 0.0),
            audio_duration=duration,
            latency=latency,
            backend=f"{self.name}:{self.model_size}",
            segments=parts,
            avg_logprob=avg_logprob,
            filtered=filtered,
        )
        log.info("STT %.2fs audio -> %.2fs compute | %s", duration, latency, result)
        return result


def _resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Linear resample. Whisper only accepts 16 kHz."""
    target = 16000
    if sample_rate == target:
        return audio
    n_out = int(round(audio.shape[0] * target / sample_rate))
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)
