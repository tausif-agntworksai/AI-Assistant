"""Signal conditioning applied to an utterance just before recognition.

Whisper was trained on broadcast-level audio. A laptop microphone two feet
away, in a room with a fan, delivers something much quieter and much more
rumbly than that — and the model's response to a quiet clip is not "I'm not
sure", it's a confident transcription of the wrong words. That is what makes
people repeat themselves.

Three fixes, in the order they have to happen:

  1. **Rumble removal.** Desk thump, fan noise and the microphone's own DC
     offset all sit below ~80 Hz, well under the lowest voiced pitch. Left in,
     they dominate the RMS measurement and make step 2 under-amplify speech.
  2. **Loudness normalisation.** Bring the speech up to roughly the level
     Whisper expects instead of leaving it 25 dB down. Gain is capped so a
     silent clip isn't amplified into pure hiss, and a limiter keeps the loud
     end from clipping.
  3. **Padding.** Whisper regularly drops the first phoneme of a clip that
     starts on speech ("kholo" for "chrome kholo"). A short lead-in of silence
     costs nothing and stops it.

Everything here is O(n) numpy — this runs on the listening thread between the
user finishing a sentence and the assistant answering, so it has a latency
budget of a millisecond or two, not a hundred.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Speech energy starts around 85 Hz (a low male voice); everything below that
# is the room, not the person.
HIGHPASS_HZ = 80.0

# −20 dBFS RMS. Whisper's training data sits around here; measured laptop
# microphone input on this machine is closer to −38 dBFS.
TARGET_RMS = 0.1

# Ceiling on the boost. Beyond ~26 dB there is no speech left to recover, only
# noise to amplify — and amplified hiss is exactly what Whisper hallucinates
# "Thank you for watching" onto.
MAX_GAIN = 20.0

# Below this the clip is silence; amplifying it would manufacture noise.
SILENCE_RMS = 1e-4

PEAK_CEILING = 0.97

# Lead-in / lead-out silence, in seconds.
PAD_SEC = 0.25


def remove_rumble(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """High-pass by subtracting a moving average.

    A one-pole IIR would be the textbook filter, but it is inherently
    sequential — a Python loop over 100k samples. Subtracting a boxcar mean is
    the same idea (whatever survives averaging is the low frequencies) and
    computes in two cumsum passes. The stopband ripple that costs us is
    irrelevant here: nothing downstream measures anything below 80 Hz.
    """
    window = max(3, int(sample_rate / HIGHPASS_HZ))
    if audio.shape[0] <= window:
        return audio - float(audio.mean()) if audio.size else audio

    # Reflect-pad so the filter doesn't manufacture a step at either end, which
    # would land in the output as an audible click.
    half = window // 2
    padded = np.concatenate((audio[half:0:-1], audio, audio[-2 : -half - 2 : -1]))
    padded = padded[: audio.shape[0] + 2 * half]
    if padded.shape[0] < audio.shape[0] + window:
        padded = np.pad(padded, (0, audio.shape[0] + window - padded.shape[0]), mode="edge")

    cumsum = np.cumsum(np.concatenate(([0.0], padded.astype(np.float64))))
    moving = (cumsum[window:] - cumsum[:-window]) / window
    return (audio - moving[: audio.shape[0]]).astype(np.float32)


def speech_rms(audio: np.ndarray, sample_rate: int) -> float:
    """RMS of the loudest ~40% of the clip.

    Plain RMS over the whole utterance is dragged down by the silence around
    the words, so a short command inside a long buffer measures as quiet and
    gets over-amplified. Measuring only the frames that actually carry speech
    gives a level that reflects how loudly the person spoke.
    """
    frame = max(1, sample_rate // 50)  # 20 ms
    usable = (audio.shape[0] // frame) * frame
    if usable < frame:
        return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))) if audio.size else 0.0)

    frames = audio[:usable].reshape(-1, frame).astype(np.float64)
    energies = np.sqrt(np.mean(np.square(frames), axis=1))
    loud = np.sort(energies)[-max(1, int(len(energies) * 0.4)) :]
    return float(loud.mean())


def normalise(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    """Scale the clip toward TARGET_RMS. Returns the audio and the gain used."""
    level = speech_rms(audio, sample_rate)
    if level < SILENCE_RMS:
        return audio, 1.0

    gain = min(MAX_GAIN, TARGET_RMS / level)
    if gain <= 1.02:
        return audio, 1.0  # already loud enough; leave it alone

    boosted = audio * gain

    # Soft limiter rather than a hard clip: hard clipping generates harmonics
    # across the whole spectrum, which reads to the model as a new consonant.
    peak = float(np.abs(boosted).max()) if boosted.size else 0.0
    if peak > PEAK_CEILING:
        boosted = np.tanh(boosted * (1.0 / PEAK_CEILING)) * PEAK_CEILING

    return boosted.astype(np.float32), gain


def pad(audio: np.ndarray, sample_rate: int, seconds: float = PAD_SEC) -> np.ndarray:
    """Add silence either side so no phoneme sits on the very first sample."""
    n = int(sample_rate * seconds)
    if n <= 0:
        return audio
    silence = np.zeros(n, dtype=np.float32)
    return np.concatenate((silence, audio, silence))


def condition(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """The whole chain. Safe on empty input and never raises."""
    audio = np.asarray(audio, dtype=np.float32).ravel()
    if audio.size == 0:
        return audio

    try:
        cleaned = remove_rumble(audio, sample_rate)
        boosted, gain = normalise(cleaned, sample_rate)
        if gain > 1.02:
            log.debug("Conditioned utterance: +%.1f dB", 20 * np.log10(gain))
        return pad(boosted, sample_rate)
    except Exception as exc:  # noqa: BLE001 - never lose an utterance to this
        log.warning("Audio conditioning failed (%s) — using the raw clip", exc)
        return audio


def is_too_quiet(audio: np.ndarray, sample_rate: int = 16000) -> bool:
    """True when the microphone is so quiet that recognition can't be trusted.

    Used to tell the user their microphone is the problem rather than silently
    mis-transcribing and leaving them to guess why.
    """
    audio = np.asarray(audio, dtype=np.float32).ravel()
    if audio.size == 0:
        return True
    return speech_rms(audio, sample_rate) < 0.004  # ≈ −48 dBFS
