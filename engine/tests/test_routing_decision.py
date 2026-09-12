# -*- coding: utf-8 -*-
"""The routing decision as one record.

Risk lived on the skill, confidence on the matcher, and the path was known
only to the orchestrator. Answering "why did this turn go the way it did?"
meant joining three things by hand. This is that join, done once.
"""

import pytest

from jarvis.nlu.rules import route
from jarvis.permissions import Risk
from jarvis.skills import load_all, registry

load_all()


@pytest.mark.parametrize(
    "spoken, skill, risk, path",
    [
        ("go to sleep", "sleep_pc", 0, "fast"),
        ("open chrome", "open_app", 0, "fast"),
        ("take a screenshot", "take_screenshot", 0, "fast"),
        ("shutdown the computer", "shutdown_pc", 2, "fast"),
        ("put my laptop to sleep", "sleep_pc", 0, "local"),
    ],
)
def test_a_decision_carries_intent_risk_and_path(spoken, skill, risk, path):
    intent = route(spoken)
    assert intent is not None, f"{spoken!r} no longer routes offline"
    decision = intent.decision()
    assert decision["intent"] == skill
    assert decision["risk_level"] == risk
    assert decision["execution_path"] == path
    assert decision["requires_reasoning"] is False


def test_confidence_is_reported_as_a_fraction():
    """Section 22 shows 0.99, not 99 — one scale, everywhere it is read."""
    decision = route("go to sleep").decision()
    assert 0.0 <= decision["confidence"] <= 1.0


def test_sending_a_message_is_a_level_two_decision():
    """It reaches another person, so it outranks anything local."""
    assert route("text sana on whatsapp saying hi").decision()["risk_level"] == 2


def test_risk_follows_the_skill_rather_than_a_copy():
    """Stored risk would drift the first time a tier changed."""
    assert registry.get("sleep_pc").risk is Risk.SAFE
    assert route("go to sleep").decision()["risk_level"] == 0


def test_a_config_override_moves_the_reported_risk():
    """The decision has to reflect the gate that will actually run."""
    from jarvis.permissions import gate

    before = dict(getattr(gate.cfg, "risk_overrides", {}) or {})
    gate.cfg.risk_overrides = {"sleep_pc": "critical"}
    try:
        assert route("go to sleep").decision()["risk_level"] == 2
    finally:
        gate.cfg.risk_overrides = before


def test_an_unroutable_utterance_has_no_decision():
    """Anything the model has to be asked about never becomes an Intent."""
    assert route("compare these two flights and tell me which is better") is None
