# -*- coding: utf-8 -*-
"""Locks the Silero v5 calling convention.

v5 scores each 512-sample frame only if the previous 64 samples are prepended
to it. Get that wrong and the model still loads, still runs and still returns
numbers - they are just always ~0, so every utterance is discarded as silence
and nothing ever reaches Whisper. Nothing raises, so only a test catches it.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.audio.vad import (
    SILERO_CONTEXT_16K,
    SILERO_FRAME,
    SegmentResult,
    SileroVad,
    SpeechSegmenter,
)


class _StubSession:
    """Records what the VAD asks the ONNX runtime to run."""

    def __init__(self, score: float = 0.9) -> None:
        self.inputs: list[np.ndarray] = []
        self._score = score

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in ("input", "state", "sr")]

    def run(self, _outputs, feeds):
        self.inputs.append(np.array(feeds["input"], copy=True))
        return (
            np.array([[self._score]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32),
        )


def _vad(session: _StubSession) -> SileroVad:
    """Builds a SileroVad around a stub, skipping the real model load."""
    vad = object.__new__(SileroVad)
    vad.sample_rate = 16000
    vad.context_size = SILERO_CONTEXT_16K
    vad._session = session
    vad._is_v5 = True
    vad._batch = 1
    vad.reset()
    return vad


def test_v5_frame_is_sent_with_context():
    session = _StubSession()
    vad = _vad(session)
    vad.probability(np.full(SILERO_FRAME, 0.5, dtype=np.float32))

    assert session.inputs[0].shape == (1, SILERO_FRAME + SILERO_CONTEXT_16K)


def test_first_frame_is_padded_with_silence():
    session = _StubSession()
    vad = _vad(session)
    vad.probability(np.full(SILERO_FRAME, 0.5, dtype=np.float32))

    lead = session.inputs[0][0, :SILERO_CONTEXT_16K]
    assert np.all(lead == 0.0)


def test_context_carries_the_previous_frame_tail():
    session = _StubSession()
    vad = _vad(session)
    first = np.linspace(-1.0, 1.0, SILERO_FRAME, dtype=np.float32)
    second = np.full(SILERO_FRAME, 0.25, dtype=np.float32)

    vad.probability(first)
    vad.probability(second)

    lead = session.inputs[1][0, :SILERO_CONTEXT_16K]
    np.testing.assert_allclose(lead, first[-SILERO_CONTEXT_16K:], rtol=0, atol=0)


def test_reset_drops_the_context():
    session = _StubSession()
    vad = _vad(session)
    vad.probability(np.full(SILERO_FRAME, 0.75, dtype=np.float32))
    vad.reset()
    vad.probability(np.full(SILERO_FRAME, 0.75, dtype=np.float32))

    lead = session.inputs[-1][0, :SILERO_CONTEXT_16K]
    assert np.all(lead == 0.0), "a new utterance must not inherit the last one's audio"


def test_wrong_frame_size_is_rejected():
    vad = _vad(_StubSession())
    with pytest.raises(ValueError):
        vad.probability(np.zeros(SILERO_FRAME + 1, dtype=np.float32))


class _ScriptedVad:
    """A VAD whose per-frame verdicts are fixed in advance."""

    frame_size = SILERO_FRAME

    def __init__(self, probs):
        self._probs = list(probs)
        self._i = 0

    def probability(self, _frame):
        p = self._probs[min(self._i, len(self._probs) - 1)]
        self._i += 1
        return p

    def reset(self):
        self._i = 0


def test_segmenter_completes_when_the_vad_reports_speech():
    """The regression in user terms: speech must yield an utterance."""
    frame_ms = 1000.0 * SILERO_FRAME / 16000
    speech_frames = int(600 / frame_ms)
    silence_frames = int(900 / frame_ms)
    vad = _ScriptedVad([0.9] * speech_frames + [0.0] * silence_frames)

    seg = SpeechSegmenter(vad, sample_rate=16000, threshold=0.5, silence_ms=700,
                          min_speech_ms=250, max_utterance_sec=15)
    seg.reset()

    frame = np.full(SILERO_FRAME, 0.1, dtype=np.float32)
    result = None
    for _ in range(speech_frames + silence_frames):
        result = seg.push(frame)
        if result not in (SegmentResult.WAITING, SegmentResult.SPEAKING):
            break

    assert result is SegmentResult.COMPLETE
    assert seg.audio.size > 0


def test_a_pause_mid_command_does_not_end_the_turn():
    """"chrome… kholo" must arrive whole.

    Half a command routes to nothing, and the user's experience of that is
    having to say the whole thing again — which is the complaint this adaptive
    window exists to answer. Barely any speech yet means the pause is probably
    a thought, not a full stop.
    """
    frame_ms = 1000.0 * SILERO_FRAME / 16000
    said_chrome = int(300 / frame_ms)
    pause = int(800 / frame_ms)          # longer than silence_ms, shorter than patience
    said_kholo = int(400 / frame_ms)

    vad = _ScriptedVad(
        [0.9] * said_chrome + [0.0] * pause + [0.9] * said_kholo + [0.0] * 60
    )
    seg = SpeechSegmenter(vad, sample_rate=16000, threshold=0.5, silence_ms=650,
                          patience_silence_ms=1300, min_speech_ms=200,
                          max_utterance_sec=15)
    seg.reset()

    frame = np.full(SILERO_FRAME, 0.1, dtype=np.float32)
    results = []
    for _ in range(said_chrome + pause + said_kholo + 60):
        result = seg.push(frame)
        results.append(result)
        if result not in (SegmentResult.WAITING, SegmentResult.SPEAKING):
            break

    # It did not end during the pause…
    assert results[said_chrome + pause - 1] is SegmentResult.SPEAKING
    # …and the finished utterance contains both halves.
    assert results[-1] is SegmentResult.COMPLETE
    assert seg.speech_ms >= 600


def test_a_settled_utterance_still_ends_promptly():
    """The patience window must not make every command feel laggy."""
    frame_ms = 1000.0 * SILERO_FRAME / 16000
    speech_frames = int(1200 / frame_ms)
    silence_frames = int(700 / frame_ms)
    vad = _ScriptedVad([0.9] * speech_frames + [0.0] * silence_frames)

    seg = SpeechSegmenter(vad, sample_rate=16000, threshold=0.5, silence_ms=650,
                          patience_silence_ms=1300, min_speech_ms=200,
                          max_utterance_sec=15)
    seg.reset()

    frame = np.full(SILERO_FRAME, 0.1, dtype=np.float32)
    result = None
    for _ in range(speech_frames + silence_frames):
        result = seg.push(frame)
        if result not in (SegmentResult.WAITING, SegmentResult.SPEAKING):
            break

    assert result is SegmentResult.COMPLETE


def test_segmenter_reports_silence_when_the_vad_never_fires():
    """A VAD stuck at zero must surface as SILENT, which is what we saw."""
    vad = _ScriptedVad([0.0])
    seg = SpeechSegmenter(vad, sample_rate=16000, threshold=0.5, silence_ms=700,
                          min_speech_ms=250, max_utterance_sec=15,
                          no_speech_timeout_sec=1.0)
    seg.reset()

    frame = np.full(SILERO_FRAME, 0.1, dtype=np.float32)
    result = None
    for _ in range(200):
        result = seg.push(frame)
        if result not in (SegmentResult.WAITING, SegmentResult.SPEAKING):
            break

    assert result is SegmentResult.SILENT
