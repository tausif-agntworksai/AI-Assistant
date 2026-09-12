# -*- coding: utf-8 -*-
"""Operating modes.

A mode changes *how* something is done. It must never change *what can be
done* — no mode adds or removes a single skill, because an assistant whose
abilities silently depend on a setting is one you stop trusting.
"""

import pytest

from jarvis.config import AssistantConfig


@pytest.mark.parametrize(
    "mode, speaks, may_use_model",
    [
        ("normal", True, True),
        ("fast", True, False),
        ("deep", True, True),
        ("silent", False, True),
        ("offline", True, False),
    ],
)
def test_each_mode_answers_the_two_questions(mode, speaks, may_use_model):
    config = AssistantConfig(mode=mode)
    assert config.speaks is speaks
    assert config.may_use_model is may_use_model


def test_deep_thinks_harder_and_fast_does_not():
    assert AssistantConfig(mode="deep").effort == "high"
    assert AssistantConfig(mode="fast").effort == "low"
    assert AssistantConfig(mode="normal").effort is None


def test_silent_still_reaches_the_model():
    """Silent is about the speakers, not about capability — the reply still
    appears in the window."""
    assert AssistantConfig(mode="silent").may_use_model is True


def test_do_not_disturb_is_independent_of_mode():
    """Wanting quiet for an hour is not the same as wanting a different kind
    of assistant. Folding them together would force a choice between a fast
    Jarvis and an undisturbed one."""
    config = AssistantConfig(mode="fast", do_not_disturb=True)
    assert config.do_not_disturb is True
    assert config.mode == "fast"
    assert AssistantConfig().do_not_disturb is False


def test_an_unknown_mode_is_refused_rather_than_guessed():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AssistantConfig(mode="turbo")


def test_the_default_is_the_shipped_balance():
    config = AssistantConfig()
    assert config.mode == "normal"
    assert config.speaks and config.may_use_model


def test_no_mode_changes_the_registry():
    """The invariant the whole design rests on."""
    from jarvis.skills import load_all, registry

    load_all()
    baseline = {s.name for s in registry.all()}
    for mode in ("normal", "fast", "deep", "silent", "offline"):
        AssistantConfig(mode=mode)
        assert {s.name for s in registry.all()} == baseline
