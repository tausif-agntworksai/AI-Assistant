# -*- coding: utf-8 -*-
"""The permission gate — the safety property the whole design rests on."""

import pytest

from jarvis.nlu.confirm import is_affirmative, is_negative
from jarvis.permissions import Capability, PermissionGate, Risk, capability_for, consent
from jarvis.skills import load_all, registry
from jarvis.skills.registry import SkillContext

load_all()


@pytest.fixture
def consent_store():
    """Restores the process-wide consent singleton after a test changes it."""
    before = consent.snapshot()
    yield consent
    if before["asked"]:
        consent.save(before["granted"])
    else:
        consent.path.unlink(missing_ok=True)
        consent.load()


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
        "send_message", "compose_email",
        "close_app", "close_window", "type_text", "toggle_wifi",
        "clear_reminders",
    }
    for name in must_be_gated:
        spec = registry.get(name)
        assert spec is not None, f"{name} is missing from the registry"
        assert spec.risk is not Risk.SAFE, f"{name} is ungated"


def test_sleep_and_lock_are_deliberately_instant():
    """These two are SAFE on purpose, and it should take a decision to change.

    Both are fully reversible — a keypress or a password puts the machine back
    exactly where it was — and confirming them out loud costs a spoken prompt,
    a listening window and a second recognition pass on the single most common
    command the assistant gets.
    """
    for name in ("sleep_pc", "lock_screen"):
        assert registry.get(name).risk is Risk.SAFE


def test_config_can_put_a_skill_back_behind_a_prompt():
    """The tier is a preference, so it has to be settable without editing code."""
    cfg = _Config()
    cfg.risk_overrides = {"sleep_pc": "confirm"}
    gate = PermissionGate(cfg)

    asked = []
    gate.set_confirmer(lambda en, hi, risk: asked.append(risk) or True)

    assert gate.check("sleep_pc", Risk.SAFE, "Sleep?", "Sula doon?") is True
    assert asked == [Risk.CONFIRM]


def test_config_can_take_a_prompt_away():
    """And the other direction, which is what makes sleep instant by default."""
    cfg = _Config()
    cfg.risk_overrides = {"close_app": "safe"}
    gate = PermissionGate(cfg)
    gate.set_confirmer(lambda en, hi, risk: pytest.fail("should not have asked"))

    assert gate.check("close_app", Risk.CONFIRM, "Close it?", "Band karun?") is True


def test_an_unparseable_override_is_ignored():
    """A typo must never silently downgrade something destructive."""
    cfg = _Config()
    cfg.risk_overrides = {"shutdown_pc": "safe-ish"}
    gate = PermissionGate(cfg)

    # Still CRITICAL, so with no confirmer installed it is refused outright.
    assert gate.check("shutdown_pc", Risk.CRITICAL, "?", "?") is False


def test_every_skill_declares_a_capability_or_is_purely_local():
    """A skill nobody classified would silently escape the consent screen."""
    local_only = {"productivity", "general"}
    for spec in registry.all():
        if spec.category in local_only:
            continue
        assert spec.capability is not None, (
            f"{spec.name} ({spec.category}) reaches the machine but maps to no "
            "capability, so the permission screen can't cover it"
        )


def test_a_revoked_capability_blocks_the_skill(consent_store):
    """Consent is enforced in the engine, not just drawn in the UI."""
    consent_store.save({c.value: True for c in Capability})
    assert capability_for("shutdown_pc", "system") is Capability.POWER

    consent_store.save({**{c.value: True for c in Capability},
                        Capability.POWER.value: False})
    result = registry.execute("shutdown_pc", {}, SkillContext())
    assert not result.ok
    assert "permission" in result.reply.en.lower()
    # ...and it is refused before anyone is asked to confirm, so revoking the
    # capability removes the ability rather than adding a question.
    assert result.detail == f"capability:{Capability.POWER.value}"


def test_an_unasked_consent_store_allows_everything(consent_store):
    """Running `python -m jarvis` by hand must not be silently crippled."""
    consent_store.path.unlink(missing_ok=True)
    consent_store.load()
    assert consent_store.asked is False
    assert consent_store.allows(Capability.POWER) is True


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
