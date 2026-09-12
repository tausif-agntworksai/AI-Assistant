# -*- coding: utf-8 -*-
"""Two ways a turn can go wrong without anyone noticing.

The follow-up window is the one place the assistant listens with nobody having
said the wake word, so it is the one place the room can be mistaken for the
user. And a skill that raises is the one place an internal failure can reach
someone as a stack trace instead of a sentence.
"""

import pytest

from jarvis.orchestrator import _FOLLOWUP_BRAIN_CONFIDENCE, _MAX_FOLLOWUP_CHAIN
from jarvis.permissions import Risk
from jarvis.skills import load_all, registry
from jarvis.skills.registry import SkillContext, skill

load_all()


# --- the follow-up window --------------------------------------------------


def test_unrecognised_follow_up_audio_needs_to_be_clearly_heard():
    """The bar exists because nobody said the wake word in this window.

    Anything that matches no skill offline has to come back confidently
    transcribed before it may reach the model. Without the bar, a television
    in the room produced a spoken "sorry, say that again?" — which is exactly
    what "it wakes up on its own" looks like from the outside.
    """
    assert 0.5 < _FOLLOWUP_BRAIN_CONFIDENCE < 1.0


def test_follow_ups_cannot_chain_forever():
    """Each reply arms the next window, so an uncapped chain lets one
    detection hold the microphone open for as long as the room keeps talking."""
    assert 1 <= _MAX_FOLLOWUP_CHAIN <= 5


def test_the_local_only_path_never_reaches_the_model(monkeypatch):
    """`local_only` is what makes the noise check cheap: unrecognised audio is
    dropped before anything is spent on it."""
    from jarvis.orchestrator import orchestrator

    def explode(*args, **kwargs):
        raise AssertionError("the model must not be asked during a follow-up")

    monkeypatch.setattr(orchestrator, "_ask_brain", explode)
    turn = orchestrator._handle("mmm hmm yeah anyway", local_only=True)
    assert not turn.understood


# --- a skill that fails ----------------------------------------------------


@skill(
    name="_exploding_test_skill",
    description="Raises, to prove one bad skill cannot take the engine down",
    risk=Risk.SAFE,
    category="general",
    examples=["explode for the tests"],
)
def _exploding_test_skill() -> object:
    raise RuntimeError("something went wrong deep inside")


def test_a_skill_that_raises_becomes_a_sentence():
    """Not a stack trace, and not silence."""
    result = registry.execute("_exploding_test_skill", {}, SkillContext())
    assert result.ok is False
    spoken = result.text("en")
    assert spoken and "Traceback" not in spoken and "RuntimeError" not in spoken


def test_a_skill_that_raises_does_not_become_a_referent():
    """Memory should carry what worked, not what was attempted."""
    from jarvis.context import TurnMemory

    memory = TurnMemory()
    result = registry.execute("_exploding_test_skill", {}, SkillContext())
    memory.record("_exploding_test_skill", result.data, result.ok)
    assert memory.snapshot() == {}


def test_an_unknown_skill_is_answered_rather_than_raised():
    result = registry.execute("no_such_skill_exists", {}, SkillContext())
    assert result.ok is False and result.text("en")


def test_bad_arguments_are_answered_rather_than_raised():
    """The model can hallucinate a parameter; that must not kill the turn."""
    result = registry.execute("_exploding_test_skill", {"nonsense": 1},
                              SkillContext())
    assert result.ok is False and result.text("en")
