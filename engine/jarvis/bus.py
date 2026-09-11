"""Event bus — the single channel between the engine and the HUD.

The audio pipeline runs on sounddevice's callback thread while the WebSocket
server runs on the asyncio loop, so `publish()` is deliberately thread-safe and
non-blocking: it can be called from anywhere without the caller knowing or
caring whether a loop is running or whether any HUD is connected.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from enum import Enum
from typing import Any, AsyncIterator

log = logging.getLogger(__name__)

# Bounded so a disconnected-but-not-cleaned-up subscriber can't grow forever.
_QUEUE_MAX = 256
_HISTORY_MAX = 100


class State(str, Enum):
    """Engine states. Mirrored one-to-one by the HUD orb."""

    STARTING = "starting"
    IDLE = "idle"          # listening for the wake word only; nothing recorded
    LISTENING = "listening"  # recording an utterance
    THINKING = "thinking"    # transcribing / routing
    ACTING = "acting"        # a skill is executing
    SPEAKING = "speaking"
    CONFIRMING = "confirming"  # waiting for the user to approve a gated action
    ERROR = "error"


class Event(str, Enum):
    STATE = "state"
    TRANSCRIPT = "transcript"
    REPLY = "reply"
    ACTION = "action"
    CONFIRM_REQUEST = "confirm_request"
    CONFIRM_RESULT = "confirm_result"
    ERROR = "error"
    LOG = "log"
    LEVEL = "level"  # live mic amplitude, for the HUD waveform
    TIMING = "timing"  # stage-by-stage latency for one finished turn
    #: Something needed the language model and no key has been set. The HUD
    #: opens settings on this rather than making the user work out why a
    #: question went unanswered.
    NEEDS_KEY = "needs_key"


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=_HISTORY_MAX)
        self._state: State = State.STARTING

    # -- wiring -------------------------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Called once by the server so worker threads know where to hand off."""
        self._loop = loop

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        """The most recent events, oldest first.

        The same buffer `subscribe` replays to a late-connecting HUD, exposed
        so a caller that isn't an async consumer — a test, a diagnostic — can
        see what was published without reaching into the deque.
        """
        return list(self._history)[-limit:]

    @property
    def state(self) -> State:
        return self._state

    # -- producing ----------------------------------------------------------

    def publish(self, event: Event, **data: Any) -> None:
        """Fan out an event. Safe from any thread; never raises, never blocks."""
        payload = {"type": event.value, "ts": time.time(), **data}

        # The waveform level fires ~30x/second — keeping it out of history
        # stops it from evicting everything else.
        if event is not Event.LEVEL:
            self._history.append(payload)

        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._deliver, payload)
        except RuntimeError:
            # Loop shut down between the check and the call. Nothing to do.
            pass

    def set_state(self, state: State, **data: Any) -> None:
        if state is self._state and not data:
            return
        self._state = state
        self.publish(Event.STATE, state=state.value, **data)

    def _deliver(self, payload: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # A slow HUD loses the oldest event rather than stalling audio.
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(payload)

    # -- consuming ----------------------------------------------------------

    async def subscribe(self, replay: bool = True) -> AsyncIterator[dict[str, Any]]:
        """Yield events until the consumer stops iterating.

        `replay` sends recent history first so a HUD that connects late still
        renders the current state instead of a blank screen.
        """
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._subscribers.add(q)
        try:
            if replay:
                yield {"type": Event.STATE.value, "ts": time.time(), "state": self._state.value}
                for item in list(self._history)[-25:]:
                    yield item
            while True:
                yield await q.get()
        finally:
            self._subscribers.discard(q)


bus = EventBus()
