"""Localhost HTTP + WebSocket API for the Electron HUD.

Bound to 127.0.0.1 on purpose: this exposes shutdown and messaging, so it must
never be reachable off the machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .bus import Event, bus
from .config import settings
from .permissions import read_audit

log = logging.getLogger(__name__)

# Everything FastAPI needs to resolve an endpoint annotation must live at
# module level. This file uses `from __future__ import annotations`, so every
# annotation is a string that FastAPI resolves against module globals — a name
# imported or defined inside `create_app` is invisible there. When that
# happens FastAPI doesn't error: it silently reclassifies the parameter as a
# query field, which turned POST /command into a 422 and the WebSocket route
# into a 403 handshake rejection.


class Command(BaseModel):
    """Request body for POST /command."""

    text: str
    language: str = "en"
    dry_run: bool = False


def create_app(orchestrator):  # noqa: ANN001 - avoids a circular import
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Worker threads publish through the bus, which needs the running loop
        # to hand events over to.
        bus.bind_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(title="Jarvis Engine", version="0.1.0",
                  docs_url=None, redoc_url=None, lifespan=lifespan)

    # The HUD is an Electron page on a file:// origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        from .nlu.llm import brain
        from .skills.registry import registry

        return {
            "ok": True,
            "state": bus.state.value,
            "skills": len(registry.all()),
            "brain": brain.available,
            "wake_word": getattr(orchestrator.wake, "available", False),
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

    @app.post("/command")
    async def command(payload: Command) -> dict[str, Any]:
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
    async def listen() -> dict[str, Any]:
        orchestrator.trigger_listen()
        return {"ok": True}

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await ws.accept()
        log.info("HUD connected")

        async def pump_events() -> None:
            async for event in bus.subscribe():
                await ws.send_json(event)

        pump = asyncio.create_task(pump_events())
        try:
            while True:
                message = await ws.receive_json()
                kind = message.get("type")
                if kind == "command":
                    reply = await asyncio.to_thread(
                        orchestrator.handle_text, message.get("text", ""),
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
            log.debug("WebSocket closed: %s", exc)
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump

    return app


def serve(orchestrator) -> None:  # noqa: ANN001
    """Run the API server. Blocks until interrupted."""
    import uvicorn

    app = create_app(orchestrator)
    config = uvicorn.Config(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level="warning",
        access_log=False,
    )
    log.info("HUD API on http://%s:%d", settings.server.host, settings.server.port)
    uvicorn.Server(config).run()
