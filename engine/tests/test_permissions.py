# -*- coding: utf-8 -*-
"""The permission gate — the safety property the whole design rests on."""

import pytest

from jarvis.nlu.confirm import is_affirmative, is_negative
from jarvis.permissions import PermissionGate, Risk
from jarvis.skills import load_all, registry
from jarvis.skills.registry import SkillContext

load_all()


class _Config:
    confirm_tier_enabled = True
    allow_critical_without_confirm = False
    confirm_timeout_sec = 5.0


def test_safe_skills_never_ask():
    gate = PermissionGate(_Config())
    assert gate.check("get_battery", Risk.SAFE, "?", "?") is True


def test_gated_skills_are_denied_when_nobody_can_ask():
    """With no confirmer installed, 'allow' would mean silently shutting down."""
    gate = PermissionGate(_Config())
    assert gate.check("shutdown_pc", Risk.CRITICAL, "?", "?") is False
    assert gate.check("sleep_pc", Risk.CONFIRM, "?", "?") is False


def test_gate_honours_the_confirmer():
    gate = PermissionGate(_Config())
    gate.set_confirmer(lambda en, hi, risk: True)
    assert gate.check("shutdown_pc", Risk.CRITICAL, "?", "?") is True

    gate.set_confirmer(lambda en, hi, risk: False)
    assert gate.check("shutdown_pc", Risk.CRITICAL, "?", "?") is False


def test_a_failing_confirmer_denies():
    def explode(en, hi, risk):
        raise RuntimeError("microphone died")

    gate = PermissionGate(_Config())
    gate.set_confirmer(explode)
    assert gate.check("shutdown_pc", Risk.CRITICAL, "?", "?") is False


def test_every_destructive_skill_is_gated():
    """A new skill that shuts down or messages someone must not default to SAFE."""
    must_be_gated = {
        "shutdown_pc", "restart_pc", "sign_out", "empty_recycle_bin",
        "send_whatsapp", "compose_email", "sleep_pc", "lock_screen",
        "close_app", "close_window", "type_text", "toggle_wifi",
        "clear_reminders",
    }
    for name in must_be_gated:
        spec = registry.get(name)
        assert spec is not None, f"{name} is missing from the registry"
        assert spec.risk is not Risk.SAFE, f"{name} is ungated"


def test_dry_run_never_executes():
    result = registry.execute("shutdown_pc", {}, SkillContext(dry_run=True))
    assert result.ok
    assert "dry run" in result.reply.en.lower()


def test_unknown_skill_fails_cleanly():
    result = registry.execute("make_coffee", {}, SkillContext())
    assert not result.ok


def test_arguments_are_coerced_to_their_annotated_types():
    """Speech and LLMs both hand back strings; skills declare ints."""
    spec = registry.get("set_volume")
    coerced = registry._coerce_args(spec, {"level": "50"})
    assert coerced["level"] == 50
    assert isinstance(coerced["level"], int)


def test_hallucinated_arguments_are_dropped():
    spec = registry.get("get_battery")
    assert registry._coerce_args(spec, {"nonexistent": "x"}) == {}


@pytest.mark.parametrize(
    "text", ["yes", "yeah", "haan", "ha", "ji haan", "theek hai", "kar do",
             "ok", "sure", "go ahead", "bilkul"],
)
def test_affirmatives(text):
    assert is_affirmative(text)


@pytest.mark.parametrize(
    "text", ["no", "nahi", "cancel", "ruko", "rehne do", "mat karo", "stop",
             "nevermind", "", "banana", "haan nahi ruko"],
)
def test_non_affirmatives(text):
    """Anything unclear must read as a refusal, not approval."""
    assert not is_affirmative(text)


def test_explicit_refusals_are_recognised():
    assert is_negative("nahi")
    assert is_negative("cancel karo")
    assert not is_negative("haan")
