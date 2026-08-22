# -*- coding: utf-8 -*-
"""HUD API contract.

Two things are asserted here, and both exist because of real failures.

The first is a FastAPI trap: this module uses `from __future__ import
annotations`, so endpoint annotations are resolved against module globals.
Names imported inside the factory function were invisible there, and FastAPI
degraded the parameters silently — POST /command became a 422 and the
WebSocket handshake a 403, with nothing in the logs.

The second is the security boundary. This API can shut the machine down, and
it listens on a port every browser on this machine can reach, so "does an
unauthenticated request get refused" is a behaviour worth a test rather than a
comment.
"""

import pytest
from fastapi.testclient import TestClient

from jarvis.security import API_TOKEN, session
from jarvis.server import WS_PROTOCOL, create_app
from jarvis.skills import load_all

load_all()

AUTH = {"Authorization": f"Bearer {API_TOKEN}"}


class _FakeOrchestrator:
    """Stands in for the real engine so no audio or Windows call happens."""

    wake = None
    last_language = "en"
    speaker = None
    listening_enabled = True

    def __init__(self):
        self.commands = []
        self.listen_calls = 0
        self.resumed = 0
        self.paused = 0
        self.consent_applied = 0

    def handle_text(self, text, language="en", source="text", dry_run=False):
        self.commands.append((text, language, source, dry_run))
        return f"handled: {text}"

    def speak(self, text, language="en"):
        return True

    def trigger_listen(self):
        self.listen_calls += 1

    def resume(self):
        self.resumed += 1

    def pause(self):
        self.paused += 1

    def apply_consent(self):
        self.consent_applied += 1

    def microphone_health(self):
        return {"open": True, "reason": None}


@pytest.fixture
def client():
    orchestrator = _FakeOrchestrator()
    app = create_app(orchestrator)
    session.unlock("tester@example.com", "uid-1", ttl_sec=600)
    with TestClient(app, headers=AUTH) as test_client:
        test_client.orchestrator = orchestrator
        yield test_client
    session.lock("test teardown")


# --- the security boundary -------------------------------------------------


@pytest.fixture
def anonymous(client):
    """The same app, called by something that has never seen the token."""
    with TestClient(client.app) as bare:
        yield bare


def test_health_needs_no_token(anonymous):
    """The desktop app has to be able to tell the engine came up at all."""
    body = anonymous.get("/health").json()
    assert body["ok"] is True
    # ...and it must not volunteer anything else while unauthenticated.
    assert "skills" not in body


def test_every_other_route_needs_the_token(anonymous):
    for path in ("/status", "/skills", "/audit", "/session", "/state"):
        assert anonymous.get(path).status_code == 401, path
    assert anonymous.post("/command", json={"text": "hi"}).status_code == 401
    assert anonymous.post("/session/unlock", json={}).status_code == 401
    assert anonymous.post("/consent", json={"granted": {}}).status_code == 401


def test_a_wrong_token_is_refused(client):
    bad = {"Authorization": "Bearer not-the-real-token"}
    assert client.get("/status", headers=bad).status_code == 401


def test_a_web_page_origin_is_refused_even_with_the_token(client):
    """The threat this API actually faces: a page doing fetch() at 127.0.0.1."""
    response = client.get("/status", headers={**AUTH, "Origin": "https://evil.example"})
    assert response.status_code == 403


def test_commands_are_refused_while_locked(client, monkeypatch):
    """Launched by the desktop app, an unlocked session is what opens the door."""
    monkeypatch.setenv("JARVIS_REQUIRE_SESSION", "1")
    session.lock("test")
    try:
        assert client.post("/command", json={"text": "shutdown"}).status_code == 423
        assert client.post("/listen").status_code == 423
        assert client.orchestrator.commands == []
    finally:
        session.unlock("tester@example.com", "uid-1", ttl_sec=600)


def test_a_command_line_launch_needs_no_sign_in(client, monkeypatch):
    """`python -m jarvis` has no app to sign in with; the token is the gate.

    Enforcing an unlock nobody could perform would leave the HUD talking to an
    engine that silently ignores it.
    """
    monkeypatch.setenv("JARVIS_REQUIRE_SESSION", "0")
    session.lock("test")
    try:
        assert client.post("/command", json={"text": "what time is it"}).status_code == 200
    finally:
        session.unlock("tester@example.com", "uid-1", ttl_sec=600)


def test_unlock_resumes_the_orchestrator(client):
    response = client.post("/session/unlock",
                           json={"account": "a@b.co", "uid": "u", "ttl_sec": 60})
    assert response.status_code == 200
    assert response.json()["unlocked"] is True
    assert client.orchestrator.resumed == 1


def test_lock_pauses_the_orchestrator(client):
    assert client.post("/session/lock").json()["unlocked"] is False
    assert client.orchestrator.paused == 1
    session.unlock("tester@example.com", "uid-1", ttl_sec=600)


def test_consent_round_trips_and_is_applied(client):
    from jarvis.permissions import consent

    # The store is a process-wide singleton backed by a file, so put it back
    # afterwards — otherwise every later test runs with the microphone as the
    # only granted capability.
    before = consent.snapshot()
    try:
        response = client.post("/consent", json={"granted": {"microphone": True,
                                                             "system_power": False}})
        assert response.status_code == 200
        granted = response.json()["granted"]
        assert granted["microphone"] is True
        assert granted["system_power"] is False
        assert client.orchestrator.consent_applied == 1
    finally:
        if before["asked"]:
            consent.save(before["granted"])
        else:
            consent.path.unlink(missing_ok=True)
            consent.load()


# --- the ordinary contract -------------------------------------------------


def test_status_reports_the_engine(client):
    body = client.get("/status").json()
    assert body["skills"] > 0
    assert body["session"]["unlocked"] is True


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
    assert shutdown["capability"] == "system_power"


def test_audit_endpoint_responds(client):
    assert isinstance(client.get("/audit?limit=5").json()["entries"], list)


# --- bring-your-own-key ----------------------------------------------------


@pytest.fixture
def llm_selection():
    """Restores the process-wide model selection after a test changes it."""
    from jarvis import llm

    before = (llm.selection.provider_id, llm.selection.model, llm.selection._api_key)
    yield llm.selection
    llm.selection.provider_id, llm.selection.model, llm.selection._api_key = before


def test_the_settings_screen_can_read_what_it_needs(client):
    body = client.get("/llm").json()
    assert {p["id"] for p in body["providers"]} == {
        "anthropic", "openai", "gemini", "groq", "deepseek", "mistral"
    }
    assert {p["id"] for p in body["speech_providers"]} == {"deepgram", "openai"}
    assert "has_key" in body["selected"]


def test_a_key_can_be_set_and_is_never_read_back(client, llm_selection):
    secret = "sk-ant-do-not-echo-this-anywhere"
    response = client.post("/llm", json={
        "provider": "anthropic", "model": "claude-sonnet-5", "api_key": secret,
    })
    assert response.status_code == 200
    assert response.json()["has_key"] is True

    # Not in this response, and not in any other endpoint either.
    for route in ("/llm", "/status"):
        assert secret not in client.get(route).text, route


def test_an_unknown_provider_is_refused_rather_than_swapped(client, llm_selection):
    """Falling back to a default would hand this key to a provider the user
    never chose."""
    response = client.post("/llm", json={
        "provider": "not-a-real-provider", "api_key": "gsk_something",
    })
    assert response.status_code == 400
    assert "don't know that AI provider" in response.json()["error"]
    assert llm_selection.provider_id != "not-a-real-provider"


def test_omitting_the_key_keeps_the_one_already_set(client, llm_selection):
    """Switching model shouldn't make the user paste their key again."""
    client.post("/llm", json={"provider": "groq", "api_key": "gsk_kept"})
    client.post("/llm", json={"provider": "groq", "model": "llama-3.1-8b-instant"})
    assert llm_selection.snapshot()["has_key"] is True
    assert llm_selection.active_model == "llama-3.1-8b-instant"


def test_an_empty_key_forgets_it(client, llm_selection, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client.post("/llm", json={"provider": "groq", "api_key": "gsk_temporary"})
    client.post("/llm", json={"provider": "groq", "api_key": ""})
    assert llm_selection.snapshot()["has_key"] is False


def test_key_checks_are_rate_limited(client):
    """An unlimited "is this key good?" endpoint is a free oracle for testing
    stolen keys, and any process on this machine with the token can reach it."""
    from jarvis.server import VALIDATE_LIMIT

    statuses = {
        client.post("/llm/validate",
                    json={"provider": "groq", "api_key": f"gsk_{n:040d}"}).status_code
        for n in range(VALIDATE_LIMIT.max_calls + 4)
    }
    assert 429 in statuses
    VALIDATE_LIMIT._count = 0  # don't leak exhaustion into other tests


def test_the_speech_recogniser_is_off_until_a_key_is_given(client):
    from jarvis.stt.selection import selection as speech

    try:
        assert client.post("/speech", json={"provider": "deepgram",
                                            "api_key": "dg-key"}).json()["has_key"]
        assert client.post("/speech", json={"provider": ""}).json()["has_key"] is False
    finally:
        speech.clear()


def test_an_unknown_speech_provider_is_refused(client):
    assert client.post("/speech", json={"provider": "nonesuch",
                                        "api_key": "k"}).status_code == 400


# --- the websocket ---------------------------------------------------------


def test_websocket_accepts_and_replays_state(client):
    """A regression guard: the handshake was rejected with 403."""
    with client.websocket_connect("/ws", subprotocols=[WS_PROTOCOL, API_TOKEN]) as ws:
        first = ws.receive_json()
        assert first["type"] == "state"
        assert "state" in first


@pytest.mark.parametrize("origin", ["null", "file://"])
def test_the_real_origins_a_local_page_sends_are_accepted(client, origin):
    """Chromium spells a file:// page's origin two different ways.

    An HTTP fetch reports `null`; a WebSocket handshake from the very same page
    reports the literal `file://`. Allowing only one of them refused every
    socket while letting every fetch through — so the HUD reconnected forever
    and, since engine state arrives over that socket, sat on "starting…"
    indefinitely. TestClient sends no Origin by default, which is exactly why
    this went unnoticed.
    """
    assert client.get("/status", headers={**AUTH, "Origin": origin}).status_code == 200

    with client.websocket_connect(
        "/ws", subprotocols=[WS_PROTOCOL, API_TOKEN], headers={"Origin": origin}
    ) as ws:
        assert ws.receive_json()["type"] == "state"


def test_a_website_origin_is_still_refused_on_the_websocket(client):
    """Widening the allow-list must not open it to a real page."""
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(
            "/ws",
            subprotocols=[WS_PROTOCOL, API_TOKEN],
            headers={"Origin": "https://evil.example"},
        ) as ws:
            ws.receive_json()


def test_websocket_without_the_token_is_closed(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()


def test_websocket_runs_a_command(client):
    with client.websocket_connect("/ws", subprotocols=[WS_PROTOCOL, API_TOKEN]) as ws:
        ws.receive_json()  # the replayed state frame
        ws.send_json({"type": "command", "text": "battery kitni hai"})
        # The command runs on a worker thread; the assertion below only needs
        # the send to have been accepted without closing the socket.
        ws.send_json({"type": "listen"})
    assert any(c[0] == "battery kitni hai" for c in client.orchestrator.commands)
