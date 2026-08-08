# -*- coding: utf-8 -*-
"""HUD API contract.

These exist because of a specific bug: this module uses
`from __future__ import annotations`, so FastAPI resolves endpoint annotations
against module globals. Names imported inside the factory function were
invisible there, and FastAPI degraded the parameters silently — POST /command
became a 422 and the WebSocket handshake a 403, with nothing in the logs. Both
failures are invisible to a plain import test, so they're asserted here.
"""

import pytest
from fastapi.testclient import TestClient

from jarvis.server import create_app
from jarvis.skills import load_all

load_all()


class _FakeOrchestrator:
    """Stands in for the real engine so no audio or Windows call happens."""

    wake = None
    last_language = "en"
    speaker = None

    def __init__(self):
        self.commands = []
        self.listen_calls = 0

    def handle_text(self, text, language="en", source="text", dry_run=False):
        self.commands.append((text, language, source, dry_run))
        return f"handled: {text}"

    def speak(self, text, language="en"):
        return True

    def trigger_listen(self):
        self.listen_calls += 1


@pytest.fixture
def client():
    orchestrator = _FakeOrchestrator()
    app = create_app(orchestrator)
    with TestClient(app) as test_client:
        test_client.orchestrator = orchestrator
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["skills"] > 0


def test_command_accepts_a_json_body(client):
    """A regression guard: this returned 422 when the model wasn't resolvable."""
    response = client.post("/command", json={"text": "what time is it"})
    assert response.status_code == 200, response.text
    assert response.json()["reply"] == "handled: what time is it"
    assert client.orchestrator.commands[0][0] == "what time is it"


def test_command_passes_dry_run_through(client):
    client.post("/command", json={"text": "shutdown", "dry_run": True})
    assert client.orchestrator.commands[-1][3] is True


def test_command_rejects_a_missing_text_field(client):
    assert client.post("/command", json={}).status_code == 422


def test_listen_triggers_the_orchestrator(client):
    assert client.post("/listen").status_code == 200
    assert client.orchestrator.listen_calls == 1


def test_skills_are_grouped_by_category(client):
    categories = client.get("/skills").json()["categories"]
    assert "apps" in categories and "system" in categories
    names = [s["name"] for s in categories["system"]]
    assert "shutdown_pc" in names
    shutdown = next(s for s in categories["system"] if s["name"] == "shutdown_pc")
    assert shutdown["risk"] == "critical"


def test_audit_endpoint_responds(client):
    assert isinstance(client.get("/audit?limit=5").json()["entries"], list)


def test_websocket_accepts_and_replays_state(client):
    """A regression guard: the handshake was rejected with 403."""
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "state"
        assert "state" in first


def test_websocket_runs_a_command(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # the replayed state frame
        ws.send_json({"type": "command", "text": "battery kitni hai"})
        # The command runs on a worker thread; the assertion below only needs
        # the send to have been accepted without closing the socket.
        ws.send_json({"type": "listen"})
    assert any(c[0] == "battery kitni hai" for c in client.orchestrator.commands)
