# -*- coding: utf-8 -*-
"""Two-pass recognition, and the guard that was eating every spoken "yes".

Both behaviours here exist to remove a repetition. The tiered transcriber
re-decodes audio the fast model was unsure about instead of asking the user to
say it again; the `short_answer` mode stops the hallucination filter from
throwing away the one-word replies that answer a confirmation prompt.
"""

import numpy as np

from jarvis.stt.base import Transcriber, Transcript, is_probable_hallucination
from jarvis.stt.tiered import TieredTranscriber

RATE = 16000
CLIP = np.full(RATE, 0.05, dtype=np.float32)  # one second, comfortably long


class _Fake(Transcriber):
    """A transcriber that returns whatever it was handed, and counts calls."""

    def __init__(self, text, confidence=1.0, name="fake"):
        self.name = name
        self.text = text
        # Transcript.confidence is exp(avg_logprob); invert to script a value.
        self.avg_logprob = float(np.log(confidence)) if confidence < 1.0 else 0.0
        self.calls = 0
        self.warmups = 0

    def transcribe(self, audio, sample_rate=RATE, *, short_answer=False, hint=None):
        self.calls += 1
        self.last_hint = hint
        return Transcript(text=self.text, avg_logprob=self.avg_logprob,
                          backend=self.name)

    def warmup(self):
        self.warmups += 1


def _tiered(fast, accurate):
    tiered = TieredTranscriber(fast, accurate, min_confidence=0.62)
    tiered._warming = True  # don't spawn the background warmup thread in tests
    return tiered


def test_a_confident_fast_pass_is_not_escalated():
    """The common case has to stay fast, or the whole design is pointless."""
    fast, accurate = _Fake("chrome kholo", 0.9), _Fake("chrome kholo", 0.95)
    result = _tiered(fast, accurate).transcribe(CLIP, RATE)
    assert result.text == "chrome kholo"
    assert accurate.calls == 0


def test_an_unsure_fast_pass_gets_a_second_opinion():
    fast = _Fake("chrome colo", 0.4, name="fast")
    accurate = _Fake("chrome kholo", 0.9, name="accurate")
    result = _tiered(fast, accurate).transcribe(CLIP, RATE)
    assert result.text == "chrome kholo"
    assert accurate.calls == 1


def test_an_empty_fast_pass_gets_a_second_opinion():
    fast, accurate = _Fake("", 1.0), _Fake("volume badhao", 0.8)
    assert _tiered(fast, accurate).transcribe(CLIP, RATE).text == "volume badhao"


def test_the_fast_result_survives_a_worse_second_opinion():
    fast = _Fake("chrome kholo", 0.5, name="fast")
    accurate = _Fake("chrome kolo", 0.2, name="accurate")
    assert _tiered(fast, accurate).transcribe(CLIP, RATE).text == "chrome kholo"


def test_a_failing_accurate_pass_never_loses_the_utterance():
    class _Broken(Transcriber):
        def transcribe(self, audio, sample_rate=RATE, *, short_answer=False, hint=None):
            raise RuntimeError("model file is corrupt")

    fast = _Fake("chrome kholo", 0.3)
    assert _tiered(fast, _Broken()).transcribe(CLIP, RATE).text == "chrome kholo"


def test_a_one_word_answer_is_not_worth_escalating():
    """"haan" must not cost a second decode — the reply has to feel instant."""
    fast, accurate = _Fake("haan", 0.3), _Fake("haan", 0.9)
    _tiered(fast, accurate).transcribe(np.full(RATE // 4, 0.05, dtype=np.float32), RATE)
    assert accurate.calls == 0


def test_escalate_forces_the_second_pass_even_on_a_confident_read():
    """The orchestrator calls this when the words matched no skill at all."""
    fast = _Fake("crumb hollow", 0.95)
    accurate = _Fake("chrome kholo", 0.9)
    tiered = _tiered(fast, accurate)
    first = tiered.transcribe(CLIP, RATE)
    assert accurate.calls == 0
    better = tiered.escalate(CLIP, RATE, previous=first)
    assert better.text == "chrome kholo"


def test_the_app_vocabulary_hint_reaches_the_decoder():
    fast = _Fake("obsidian kholo", 0.9)
    _tiered(fast, _Fake("", 1.0)).transcribe(CLIP, RATE, hint="Obsidian, Rufus")
    assert fast.last_hint == "Obsidian, Rufus"


# --- the confirmation guard ------------------------------------------------


def test_a_spoken_yes_is_kept_when_a_yes_is_what_we_asked_for():
    """This is the bug that made "haan" read as a refusal: the hallucination
    filter listed every word a one-word answer is made of."""
    for answer in ("haan", "हाँ", "ji", "ok", "yes", "nahi"):
        assert not is_probable_hallucination(answer, short_answer=True), answer


def test_the_same_words_are_still_filler_mid_conversation():
    for noise in ("haan", "ok", "जी"):
        assert is_probable_hallucination(noise, short_answer=False), noise


def test_whispers_favourite_inventions_are_dropped_either_way():
    for noise in ("Thanks for watching!", "[Music]", "..."):
        assert is_probable_hallucination(noise, short_answer=True), noise
        assert is_probable_hallucination(noise, short_answer=False), noise
