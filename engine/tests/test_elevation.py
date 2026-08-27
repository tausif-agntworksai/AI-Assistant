# -*- coding: utf-8 -*-
"""Administrator actions.

Windows refuses to switch the Wi-Fi or Bluetooth radio for an unelevated
program, so those commands used to give up and open the Settings app. They can
now ask for Administrator rights — which is exactly the kind of capability
that needs its gate asserted rather than described.

The property that matters: an always-listening assistant steered by a language
model must never be able to raise a UAC prompt on a machine where the user
never allowed elevated actions. A misheard word should not put a consent
dialog on screen.
"""

import pytest

from jarvis.permissions import Capability, consent
from jarvis.skills import device, load_all, registry
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


@pytest.fixture
def never_elevated(monkeypatch):
    """Pretend we are an ordinary process, and record elevation attempts."""
    attempts = []

    def _run_elevated(program, arguments, timeout=30.0):
        attempts.append((program, arguments))
        return False, "cancelled at the Windows prompt"

    monkeypatch.setattr(device.winutil, "is_elevated", lambda: False)
    monkeypatch.setattr(device.winutil, "run_elevated", _run_elevated)
    monkeypatch.setattr(device.winutil, "shell_open", lambda target: True)
    return attempts


# --- the gate --------------------------------------------------------------


def test_without_consent_it_never_even_asks(consent_store, never_elevated, monkeypatch):
    """No consent means no UAC prompt — not a prompt the user then declines."""
    consent_store.save({**{c.value: True for c in Capability},
                        Capability.ADMIN.value: False})
    monkeypatch.setattr(device.winutil, "run", _failing_run)

    result = device.toggle_wifi("off")
    assert not result.ok
    assert never_elevated == [], "asked for elevation without consent"
    assert "administrator access" in result.reply.en.lower()


def test_with_consent_it_asks(consent_store, never_elevated, monkeypatch):
    consent_store.save({c.value: True for c in Capability})
    monkeypatch.setattr(device.winutil, "run", _failing_run)

    device.toggle_wifi("off")
    assert len(never_elevated) == 1
    program, arguments = never_elevated[0]
    assert program == "netsh.exe"
    assert "admin=disabled" in arguments


def test_a_refused_prompt_is_reported_not_claimed(consent_store, never_elevated,
                                                  monkeypatch):
    """Declining the Windows dialog must not read as success."""
    consent_store.save({c.value: True for c in Capability})
    monkeypatch.setattr(device.winutil, "run", _failing_run)

    result = device.toggle_wifi("on")
    assert not result.ok
    assert "cancelled" in result.reply.en.lower()


def test_elevation_is_skipped_when_the_plain_command_works(consent_store,
                                                           never_elevated, monkeypatch):
    """Most machines need no elevation for some of this; don't prompt anyway."""
    consent_store.save({c.value: True for c in Capability})
    monkeypatch.setattr(device.winutil, "run", _succeeding_run)

    result = device.toggle_wifi("on")
    assert result.ok
    assert never_elevated == []


# --- the commands ----------------------------------------------------------


@pytest.mark.parametrize(
    "spoken, expected_on",
    [("on", True), ("chalu", True), ("enable", True), ("yes", True),
     ("off", False), ("band", False), ("disable", False), ("", False)],
)
def test_on_and_off_are_understood_in_both_languages(spoken, expected_on):
    assert device._wants_on(spoken) is expected_on


def test_bluetooth_targets_the_radio_not_every_paired_device(consent_store,
                                                             never_elevated):
    consent_store.save({c.value: True for c in Capability})
    device.toggle_bluetooth("off")

    assert len(never_elevated) == 1
    _, arguments = never_elevated[0]
    assert "Disable-PnpDevice" in arguments
    assert "-Class Bluetooth" in arguments
    # Enumerators are the bus, not the radio; disabling those takes the stack out.
    assert "Enumerator" in arguments


def test_both_toggles_still_confirm_out_loud():
    """Elevation is an addition to the confirmation, not a replacement."""
    from jarvis.permissions import Risk

    for name in ("toggle_wifi", "toggle_bluetooth"):
        assert registry.get(name).risk is not Risk.SAFE, name


def test_the_skill_is_refused_when_settings_consent_is_off(consent_store):
    """The outer gate still applies: no Windows-settings consent, no toggle."""
    consent_store.save({**{c.value: True for c in Capability},
                        Capability.SETTINGS.value: False})
    result = registry.execute("toggle_wifi", {"state": "off"}, SkillContext())
    assert not result.ok
    assert result.detail == f"capability:{Capability.SETTINGS.value}"


# --- helpers ---------------------------------------------------------------


class _Proc:
    def __init__(self, code):
        self.returncode = code
        self.stdout = ""
        self.stderr = "access denied" if code else ""


def _failing_run(*args, **kwargs):
    return _Proc(1)


def _succeeding_run(*args, **kwargs):
    return _Proc(0)
