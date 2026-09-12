"""Stopping something that is already happening.

"Actually, stop" is the difference between an assistant and a batch job, and
until now Jarvis could only stop one thing: the sound coming out of the
speakers. `AudioPlayer.stop()` cut playback, and everything behind it — the
model request still in flight, the skill about to run, the next step of a
multi-step task — carried on regardless, because nothing was listening.

This is the thing they can all listen to. One token per turn, checked at the
boundaries where stopping is safe: between steps, before a skill runs, between
sentences of a reply. Deliberately **not** checked in the middle of an action.
A half-sent message or a half-typed line is worse than one that finished, so
cancellation lands on the seams rather than wherever the news happens to
arrive.

The token is also the honest place to record *why* something stopped, because
"you interrupted me" and "that took too long" deserve different replies.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

log = logging.getLogger(__name__)


class Cancelled(Exception):
    """Raised at a checkpoint when the turn has been called off.

    An exception rather than a return value because the call stack it has to
    unwind — orchestrator, brain, agent loop, skill — is deep, and threading a
    boolean back through every one of those returns would mean every caller
    remembering to check it. Missing a check is how a cancelled turn keeps
    talking.
    """

    def __init__(self, reason: str = "cancelled") -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class CancelToken:
    """A turn's stop switch, shared by everyone working on that turn."""

    #: What asked for the stop — "user", "timeout", "shutdown". Recorded
    #: because the reply differs: being interrupted deserves no apology, and
    #: giving up after thirty seconds does.
    reason: str = ""
    _event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "user") -> None:
        """Ask everything working on this turn to stop at its next checkpoint."""
        if self._event.is_set():
            return
        self.reason = reason
        self._event.set()
        log.info("Turn cancelled (%s)", reason)

    def check(self) -> None:
        """Raise if the turn has been called off. Call this at a seam."""
        if self._event.is_set():
            raise Cancelled(self.reason or "cancelled")

    def wait(self, timeout: float) -> bool:
        """Sleep, but wake early if cancelled. True if it was.

        Anything that would otherwise `time.sleep` through a cancellable turn
        should wait here instead — a poll loop that sleeps blind is a poll loop
        that keeps going for a second after being told to stop.
        """
        return self._event.wait(timeout)


#: A token that is never cancelled, for callers outside a turn.
#: Saves every one of them writing `if token is not None` around a check.
NEVER = CancelToken()


def token_or_never(token: "CancelToken | None") -> CancelToken:
    return token if token is not None else NEVER


class TurnControl:
    """The one turn currently in flight, so anything can stop it.

    Process-wide rather than thread-local on purpose: the word "stop" arrives
    on a *different* thread from the turn it is stopping — the server's event
    loop, or the listening thread while the worker is busy — so a token filed
    under the working thread would be exactly the one the canceller cannot
    reach. The orchestrator already serialises turns behind its own lock, so
    there is only ever one slot to fill.
    """

    def __init__(self) -> None:
        self._token: CancelToken | None = None
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._token is not None

    def begin(self) -> CancelToken:
        token = CancelToken()
        with self._lock:
            self._token = token
        return token

    def end(self, token: CancelToken) -> None:
        """Retire a token — but only if it is still the current one.

        The guard matters: a turn that overran and a turn that has just started
        can both reach this, and letting the older one clear the slot would
        leave the newer turn uncancellable for the rest of its life.
        """
        with self._lock:
            if self._token is token:
                self._token = None

    @contextmanager
    def turn(self) -> "Iterator[CancelToken]":
        """Own the cancellable slot for the duration of one turn.

        A context manager because the failure mode of forgetting to release it
        is invisible and lasting: the slot stays filled, and every later turn
        cancels a token nobody is listening to any more.
        """
        token = self.begin()
        try:
            yield token
        finally:
            self.end(token)

    def cancel_current(self, reason: str = "user") -> bool:
        """Stop whatever is running. False when there was nothing to stop."""
        with self._lock:
            token = self._token
        if token is None:
            return False
        token.cancel(reason)
        return True


control = TurnControl()


#: What counts as "stop" while something is running. Deliberately narrow.
#: "band karo" means *close* or *mute* everywhere else in the assistant, and
#: "ruko" is already a refusal at the confirmation gate — so these only mean
#: cancel while a turn is actually in flight, and are never consulted
#: otherwise. A wider list would start eating ordinary commands.
_STOP_PHRASES = frozenset({
    "stop", "stop it", "stop that", "actually stop", "cancel", "cancel that",
    "cancel it", "abort", "never mind", "nevermind", "forget it", "wait",
    "hold on", "that's enough", "thats enough", "enough",
    "ruko", "ruk jao", "rehne do", "rahne do", "chhodo", "chodo",
    "cancel karo", "band karo abhi", "mat karo", "rok do", "roko",
})


def is_stop_request(text: str) -> bool:
    """Is this someone calling off what is already running?

    Only ever asked while a turn is in flight. Matched as a whole utterance
    rather than a substring: "stop the music" is a command to a skill, and
    reading it as a cancellation would make the media controls unusable.
    """
    from .nlu.normalize import normalize

    return normalize(text) in _STOP_PHRASES
