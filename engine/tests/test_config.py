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
