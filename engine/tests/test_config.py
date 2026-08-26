# -*- coding: utf-8 -*-
"""Configuration resolution.

The port test exists because of a real failure mode: the desktop app picks a
free port before spawning the engine and passes it in JARVIS_PORT, but the
engine used to bind whatever config.yaml said. Whenever 8756 was already taken
— an orphaned engine, anything else on the port — the app then health-checked
a port nothing was listening on and sat there until its timeout expired. To a
user that is indistinguishable from "the engine never starts".
"""

import pytest

from jarvis.config import ServerConfig


@pytest.fixture
def server():
    return ServerConfig()


def test_falls_back_to_config_without_the_env_var(server, monkeypatch):
    monkeypatch.delenv("JARVIS_PORT", raising=False)
    assert server.resolved_port == server.port


def test_env_var_wins(server, monkeypatch):
    monkeypatch.setenv("JARVIS_PORT", "8791")
    assert server.resolved_port == 8791


def test_blank_env_var_is_ignored(server, monkeypatch):
    monkeypatch.setenv("JARVIS_PORT", "   ")
    assert server.resolved_port == server.port


@pytest.mark.parametrize("bad", ["notanumber", "80.5", "-1", "0", "70000", "8756abc"])
def test_unusable_values_fall_back_rather_than_crash(server, monkeypatch, bad):
    """A bad launcher value must not stop the assistant from starting at all."""
    monkeypatch.setenv("JARVIS_PORT", bad)
    assert server.resolved_port == server.port


def test_boundary_ports_are_accepted(server, monkeypatch):
    for value in ("1", "65535"):
        monkeypatch.setenv("JARVIS_PORT", value)
        assert server.resolved_port == int(value)


# --- wake-word acknowledgement ---------------------------------------------


from jarvis.config import WakeWordConfig  # noqa: E402


def test_acknowledgements_default_to_a_pool():
    wake = WakeWordConfig()
    assert len(wake.ack_texts("en")) > 1
    assert len(wake.ack_texts("hi")) > 1


def test_a_bare_string_still_works():
    """The setting used to be one string; an old config.yaml must still load."""
    wake = WakeWordConfig(ack_text_en="Yes?", ack_text_hi="जी?")
    assert wake.ack_texts("en") == ["Yes?"]
    assert wake.ack_texts("hi") == ["जी?"]


def test_blank_entries_are_dropped():
    wake = WakeWordConfig(ack_text_en=["Yes?", "", "   "])
    assert wake.ack_texts("en") == ["Yes?"]


def test_no_acknowledgement_uses_non_lexical_sounds():
    """"Mm-hmm" rendered as an unintelligible mumble.

    Neural voices are trained on written language and have no reliable
    pronunciation for sounds that aren't words, so every phrase must contain
    at least one real letter-bearing word.
    """
    import re

    wake = WakeWordConfig()
    for phrase in wake.ack_texts("en"):
        assert re.search(r"[A-Za-z]{2,}", phrase), phrase
        assert "mm-hmm" not in phrase.lower()


def test_rotation_never_repeats_immediately():
    """Hearing the same phrase twice running is what sounds like a recording."""
    from jarvis.orchestrator import Orchestrator

    o = Orchestrator()  # no start() — this touches no audio device
    picks = [o._pick_acknowledgement("en") for _ in range(40)]
    assert not any(a == b for a, b in zip(picks, picks[1:]))
    # ...and it should actually use the variety it has.
    assert len(set(picks)) > 1


def test_rotation_survives_a_single_phrase():
    """With one phrase configured there is nothing to alternate with."""
    from jarvis.orchestrator import Orchestrator

    o = Orchestrator()
    o.cfg.wake_word.ack_text_en = ["Yes?"]
    try:
        assert [o._pick_acknowledgement("en") for _ in range(3)] == ["Yes?"] * 3
    finally:
        o.cfg.wake_word.ack_text_en = WakeWordConfig().ack_text_en
