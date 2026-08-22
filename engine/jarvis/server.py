"""Localhost HTTP + WebSocket API for the Electron HUD.

Bound to 127.0.0.1 on purpose: this exposes shutdown and messaging, so it must
never be reachable off the machine. Loopback alone is not enough, though —
every browser on this machine can also reach 127.0.0.1 — so every route beyond
`/health` needs the launch token, and every command needs an unlocked session
as well. See `security.py` for why those two are separate.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .bus import Event, bus
from .config import settings
from .permissions import consent, read_audit
from .security import (
    PUBLIC_PATHS,
    RateLimiter,
    bearer_from_header,
    redact,
    session,
    token_matches,
)

log = logging.getLogger(__name__)

# Everything FastAPI needs to resolve an endpoint annotation must live at
# module level. This file uses `from __future__ import annotations`, so every
# annotation is a string that FastAPI resolves against module globals — a name
# imported or defined inside `create_app` is invisible there. When that
# happens FastAPI doesn't error: it silently reclassifies the parameter as a
# query field, which turned POST /command into a 422 and the WebSocket route
# into a 403 handshake rejection.

# The HUD is an Electron page loaded from disk, and Chromium spells that origin
# two different ways depending on the request: `null` for an HTTP fetch, but the
# literal `file://` for a WebSocket handshake from the very same page. Accepting
# only one of them refused every socket while letting every fetch through — the
# HUD reconnected forever and, since engine state arrives over that socket, sat
# on "starting…" indefinitely.
#
# Both are safe to allow because neither is what guards this API: the launch
# token is. A page on a real website cannot forge either value, and a local HTML
# file still cannot read the token.
LOCAL_ORIGINS = frozenset({"null", "file://"})

# WebSocket handshakes can't carry an Authorization header from a browser, so
# the token rides in the subprotocol list instead of a query string — query
# strings end up in logs and referrers.
WS_PROTOCOL = "jarvis.bearer"

# Generous for a person talking, useless for a script hammering the model key.
COMMAND_LIMIT = RateLimiter(max_calls=60, window_sec=60.0)

# Key checks and model listings each cost a live call to a provider with a
# caller-supplied key. Enough for someone fixing a typo, useless for anyone
# using this machine to test a list of stolen keys.
VALIDATE_LIMIT = RateLimiter(max_calls=12, window_sec=60.0)


class Command(BaseModel):
    """Request body for POST /command."""

    text: str = Field(max_length=4000)
    language: str = Field(default="en", max_length=8)
    dry_run: bool = False


class Unlock(BaseModel):
    """Request body for POST /session/unlock."""

    account: str = Field(default="", max_length=254)
    uid: str = Field(default="", max_length=128)
    #: How long the unlock stands before the desktop app must renew it. Matches
    #: the lifetime of the Firebase ID token that vouched for the user.
    ttl_sec: float = 3600.0


class ConsentPayload(BaseModel):
    """Request body for POST /consent — the permission screen's decision."""

    granted: dict[str, bool] = Field(default_factory=dict)


class LlmConfig(BaseModel):
    """Request body for POST /llm — the user's provider, model and key.

    The key arrives here, is held in memory, and is written nowhere. The
    desktop app is what stores it, encrypted with the OS keychain; this
    process only borrows it for the length of a request. `api_key` omitted
    means "keep whatever you have"; an empty string means "forget it".
    """

    provider: str = Field(max_length=32)
    model: str = Field(default="", max_length=128)
    api_key: str | None = Field(default=None, max_length=512)


class LlmProbe(BaseModel):
    """Request body for POST /llm/validate and /llm/models."""

    provider: str = Field(max_length=32)
    api_key: str = Field(max_length=512)
    model: str = Field(default="", max_length=128)


class SpeechConfig(BaseModel):
    """Request body for POST /speech — the optional cloud recogniser."""

    provider: str = Field(default="", max_length=32)
    api_key: str | None = Field(default=None, max_length=512)


def create_app(orchestrator):  # noqa: ANN001 - avoids a circular import
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Worker threads publish through the bus, which needs the running loop
        # to hand events over to.
        bus.bind_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(title="Jarvis Engine", version="0.1.0",
                  docs_url=None, redoc_url=None, lifespan=lifespan)

    # Only the file:// HUD, and only for the two headers it actually sends.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(LOCAL_ORIGINS),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600,
    )

    @app.middleware("http")
    async def authenticate(request: Request, call_next):  # noqa: ANN001
        path = request.url.path

        # A browser page from a real website must never get through, even in
        # the hypothetical where it has somehow learned the token.
        origin = request.headers.get("origin")
        if origin is not None and origin not in LOCAL_ORIGINS:
            log.warning("Rejected a request from origin %r", origin[:80])
            return JSONResponse({"error": "forbidden origin"}, status_code=403)

        # Preflights carry no Authorization by definition; CORSMiddleware has
        # already answered them by the time a real request arrives.
        if request.method == "OPTIONS" or path in PUBLIC_PATHS:
            return await call_next(request)

        if not token_matches(bearer_from_header(request.headers.get("authorization"))):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)

    def locked() -> bool:
        """Whether commands should be refused for want of a signed-in user.

        Mirrors the microphone's rule rather than inventing a second one: a
        launch that doesn't require sign-in (a bare `python -m jarvis`) is
        already authorised by the token, and demanding an unlock nobody can
        perform would leave the HUD talking to an engine that ignores it.
        """
        return settings.security.session_required and not session.active

    def locked_response() -> JSONResponse:
        return JSONResponse(
            {
                "error": "Sign in to Jarvis before giving it commands.",
                "code": "locked",
            },
            status_code=423,
        )

    # -- status -------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Deliberately says almost nothing: this is the one unauthenticated
        route, and it exists only so the desktop app knows the process is up."""
        return {"ok": True, "state": bus.state.value}

    @app.get("/status")
    async def status() -> dict[str, Any]:
        from . import llm
        from .nlu.llm import brain
        from .skills.registry import registry
        from .stt.selection import selection as speech

        return {
            "ok": True,
            "state": bus.state.value,
            "language": orchestrator.last_language,
            "skills": len(registry.all()),
            "brain": brain.available,
            "llm": llm.selection.snapshot(),
            "speech": speech.snapshot(),
            "wake_word": getattr(orchestrator.wake, "available", False),
            "listening": orchestrator.listening_enabled,
            "microphone": orchestrator.microphone_health(),
            "session": session.snapshot(),
            "consent": consent.snapshot(),
        }

    @app.get("/state")
    async def state() -> dict[str, Any]:
        return {"state": bus.state.value, "language": orchestrator.last_language}

    @app.get("/skills")
    async def skills() -> dict[str, Any]:
        from .skills.registry import registry

        return {
            "categories": {
                category: [
                    {
                        "name": s.name,
                        "description": s.description,
                        "risk": s.risk.value,
                        "capability": s.capability.value if s.capability else None,
                        "examples": s.examples[:4],
                    }
                    for s in specs
                ]
                for category, specs in sorted(registry.by_category().items())
            }
        }

    @app.get("/audit")
    async def audit(limit: int = 100) -> dict[str, Any]:
        return {"entries": read_audit(min(limit, 500))}

    # -- session ------------------------------------------------------------

    @app.get("/session")
    async def get_session() -> dict[str, Any]:
        return session.snapshot()

    @app.post("/session/unlock")
    async def unlock(payload: Unlock) -> dict[str, Any]:
        session.unlock(payload.account, payload.uid, payload.ttl_sec)
        await asyncio.to_thread(orchestrator.resume)
        return session.snapshot()

    @app.post("/session/lock")
    async def lock() -> dict[str, Any]:
        session.lock("desktop app signed out")
        await asyncio.to_thread(orchestrator.pause)
        return session.snapshot()

    # -- consent ------------------------------------------------------------

    @app.get("/consent")
    async def get_consent() -> dict[str, Any]:
        return consent.snapshot()

    @app.post("/consent")
    async def set_consent(payload: ConsentPayload) -> dict[str, Any]:
        consent.save(payload.granted)
        await asyncio.to_thread(orchestrator.apply_consent)
        return consent.snapshot()

    # -- the language model and the speech recogniser -----------------------

    @app.get("/llm")
    async def get_llm() -> dict[str, Any]:
        from . import llm
        from .stt import selection as speech

        return {
            "providers": llm.provider_list(),
            "selected": llm.selection.snapshot(),
            "speech_providers": speech.provider_list(),
            "speech": speech.selection.snapshot(),
        }

    @app.post("/llm")
    async def set_llm(payload: LlmConfig) -> Any:
        from . import llm

        try:
            llm.selection.configure(payload.provider, payload.model, payload.api_key)
        except llm.LlmError as exc:
            return JSONResponse({"error": exc.friendly}, status_code=400)
        return llm.selection.snapshot()

    @app.post("/llm/validate")
    async def validate_llm(payload: LlmProbe) -> Any:
        from . import llm

        # Rate-limited for the reason the sibling calculator spells out: an
        # unlimited "is this key good?" endpoint is a free oracle for testing
        # stolen keys, and this one is reachable from any process on the machine
        # that has the launch token.
        allowed, retry_after = VALIDATE_LIMIT.allow()
        if not allowed:
            return JSONResponse(
                {"error": f"Too many key checks — wait {retry_after}s.",
                 "code": "rate-limited"},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        try:
            await asyncio.to_thread(
                llm.validate_key, payload.provider, payload.api_key, payload.model
            )
        except llm.LlmError as exc:
            log.info("Key check failed for %s: %s", payload.provider, exc)
            return JSONResponse({"ok": False, "error": exc.friendly}, status_code=400)
        return {"ok": True}

    @app.post("/llm/models")
    async def llm_models(payload: LlmProbe) -> Any:
        from . import llm

        allowed, retry_after = VALIDATE_LIMIT.allow()
        if not allowed:
            return JSONResponse(
                {"error": f"Too many requests — wait {retry_after}s."},
                status_code=429,
            )
        try:
            models = await asyncio.to_thread(
                llm.list_models, payload.provider, payload.api_key
            )
        except llm.LlmError as exc:
            return JSONResponse({"error": exc.friendly}, status_code=400)
        return {"models": models}

    @app.post("/speech")
    async def set_speech(payload: SpeechConfig) -> Any:
        from .stt.selection import selection as speech

        try:
            speech.configure(payload.provider, payload.api_key)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return speech.snapshot()

    # -- commands -----------------------------------------------------------

    @app.post("/command")
    async def command(payload: Command) -> Any:
        if locked():
            return locked_response()
        allowed, retry_after = COMMAND_LIMIT.allow()
        if not allowed:
            return JSONResponse(
                {"error": f"Too many commands — try again in {retry_after}s.",
                 "code": "rate-limited"},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )

        # Skills block on Win32 calls and the network, so keep them off the loop.
        reply = await asyncio.to_thread(
            orchestrator.handle_text, payload.text, payload.language,
            "hud", payload.dry_run,
        )
        if reply and not payload.dry_run:
            asyncio.create_task(asyncio.to_thread(
                orchestrator.speak, reply, orchestrator.last_language
            ))
        return {"reply": reply}

    @app.post("/listen")
    async def listen() -> Any:
        if locked():
            return locked_response()
        orchestrator.trigger_listen()
        return {"ok": True}

    # -- events -------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        # `new WebSocket(url, [WS_PROTOCOL, token])` arrives here as a
        # comma-separated header. Nothing else may connect.
        offered = [
            p.strip() for p in (ws.headers.get("sec-websocket-protocol") or "").split(",")
            if p.strip()
        ]
        origin = ws.headers.get("origin")
        if (origin is not None and origin not in LOCAL_ORIGINS) or len(offered) < 2 \
                or offered[0] != WS_PROTOCOL or not token_matches(offered[1]):
            log.warning("Rejected a WebSocket handshake (origin=%r)", (origin or "")[:80])
            await ws.close(code=1008)
            return

        await ws.accept(subprotocol=WS_PROTOCOL)
        log.info("HUD connected")

        async def pump_events() -> None:
            async for event in bus.subscribe():
                await ws.send_json(event)

        pump = asyncio.create_task(pump_events())
        try:
            while True:
                message = await ws.receive_json()
                kind = message.get("type")
                if kind in ("command", "listen") and locked():
                    await ws.send_json({"type": "error", "message":
                                        "Sign in to Jarvis before giving it commands."})
                    continue
                if kind == "command":
                    reply = await asyncio.to_thread(
                        orchestrator.handle_text, str(message.get("text", ""))[:4000],
                        message.get("language", "en"), "hud",
                    )
                    if reply:
                        asyncio.create_task(asyncio.to_thread(
                            orchestrator.speak, reply, orchestrator.last_language
                        ))
                elif kind == "listen":
                    orchestrator.trigger_listen()
                elif kind == "stop_speaking":
                    orchestrator.speaker and orchestrator.speaker.stop()
        except WebSocketDisconnect:
            log.info("HUD disconnected")
        except Exception as exc:  # noqa: BLE001
            log.debug("WebSocket closed: %s", redact(str(exc)))
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    return app


def serve(orchestrator) -> None:  # noqa: ANN001
    """Run the API server. Blocks until interrupted."""
    import uvicorn

    host = settings.server.host
    if host not in ("127.0.0.1", "localhost", "::1"):
        # Widening this puts shutdown and messaging on the network. Refuse
        # rather than obey a config typo that hands the machine away.
        log.error("server.host is %r — refusing to bind anywhere but loopback", host)
        host = "127.0.0.1"

    app = create_app(orchestrator)
    config = uvicorn.Config(
        app,
        host=host,
        port=settings.server.bind_port,
        log_level="warning",
        access_log=False,
    )
    log.info("HUD API on http://%s:%d (token required)", host, settings.server.bind_port)
    uvicorn.Server(config).run()
