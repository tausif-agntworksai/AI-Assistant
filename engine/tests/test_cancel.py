# -*- coding: utf-8 -*-
"""Stopping something that is already happening.

The property that matters most here is not that cancellation works, but that
it works *while the assistant is busy* — a stop that queues behind the turn it
is meant to kill arrives after that turn has finished, which is the same as
having no stop button at all.
"""

import threading
import time

import pytest

from jarvis.cancel import (
    Cancelled,
    CancelToken,
    TurnControl,
    is_stop_request,
)


# --- the token -------------------------------------------------------------


def test_a_fresh_token_is_not_cancelled():
    assert CancelToken().cancelled is False


def test_check_raises_once_cancelled():
    token = CancelToken()
    token.cancel("user")
    with pytest.raises(Cancelled) as caught:
        token.check()
    assert caught.value.reason == "user"


def test_check_is_silent_while_running():
    CancelToken().check()  # must not raise


def test_cancelling_twice_keeps_the_first_reason():
    """The first reason is the true one; a later cancel is an echo of it."""
    token = CancelToken()
    token.cancel("user")
    token.cancel("timeout")
    assert token.reason == "user"


def test_wait_returns_early_when_cancelled():
    """Anything that would otherwise sleep through a turn waits here instead."""
    token = CancelToken()
    threading.Timer(0.05, lambda: token.cancel("user")).start()

    start = time.perf_counter()
    woke = token.wait(5.0)
    elapsed = time.perf_counter() - start

    assert woke is True
    assert elapsed < 1.0, "wait() slept through a cancellation"


def test_wait_times_out_when_nothing_happens():
    assert CancelToken().wait(0.01) is False


# --- the slot --------------------------------------------------------------


def test_nothing_to_cancel_is_reported_not_raised():
    """The HUD's stop button is pressable when nothing is running."""
    assert TurnControl().cancel_current() is False


def test_a_turn_can_be_cancelled_while_it_holds_the_slot():
    control = TurnControl()
    with control.turn() as token:
        assert control.active
        assert control.cancel_current("user") is True
        assert token.cancelled
    assert control.active is False


def test_the_slot_is_released_even_if_the_turn_raises():
    """A turn that dies must not leave every later turn uncancellable."""
    control = TurnControl()
    with pytest.raises(ValueError):
        with control.turn():
            raise ValueError("skill exploded")
    assert control.active is False


def test_a_stale_turn_cannot_release_a_newer_ones_slot():
    """An overrunning turn ending after a new one began would otherwise leave
    the new turn with nothing listening for its stop."""
    control = TurnControl()
    old = control.begin()
    new = control.begin()

    control.end(old)                       # the straggler finishes late
    assert control.active, "the newer turn lost its slot"
    assert control.cancel_current() is True
    assert new.cancelled and not old.cancelled


# --- what counts as "stop" -------------------------------------------------


@pytest.mark.parametrize(
    "spoken",
    ["stop", "stop it", "actually stop", "cancel", "cancel that", "forget it",
     "ruko", "ruk jao", "rehne do", "chhodo", "rok do"],
)
def test_these_call_off_what_is_running(spoken):
    assert is_stop_request(spoken)


@pytest.mark.parametrize(
    "spoken",
    ["stop the music", "stop the timer", "pause", "band karo",
     "volume band karo", "cancel the shutdown", "stop spotify"],
)
def test_these_are_commands_and_must_reach_their_skill(spoken):
    """Matched as a whole utterance, never a substring — otherwise the media
    controls and "cancel shutdown" stop being usable."""
    assert not is_stop_request(spoken)


# --- the property the whole design rests on --------------------------------


def test_stop_does_not_queue_behind_the_turn_it_is_stopping(monkeypatch):
    """Turns are serialised behind a lock. If "stop" waited for that lock it
    would run after the turn had already finished — a stop button that only
    works once there is nothing left to stop.

    So the cancel check sits *above* the lock, and this proves it: a skill is
    held open, and "stop" must still return promptly.
    """
    from jarvis.cancel import control
    from jarvis.orchestrator import orchestrator
    from jarvis.skills import load_all, registry
    from jarvis.skills.registry import ok

    load_all()
    started, release = threading.Event(), threading.Event()
    real_execute = registry.execute

    def hang(name, args, ctx):
        if name == "get_battery":
            started.set()
            release.wait(5)
            return ok("Battery is at 65 percent.")
        return real_execute(name, args, ctx)

    monkeypatch.setattr(registry, "execute", hang)
    spoken = {}

    def busy_turn():
        spoken["reply"] = orchestrator.handle_text("what's the battery")

    worker = threading.Thread(target=busy_turn, daemon=True)
    worker.start()
    try:
        assert started.wait(3), "the slow skill never started"
        assert control.active, "the turn did not claim the cancellable slot"

        start = time.perf_counter()
        reply = orchestrator.handle_text("stop")
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"'stop' blocked for {elapsed:.2f}s on the turn lock"
        assert reply == "Stopped."
    finally:
        release.set()
        worker.join(timeout=5)


def test_an_ordinary_command_still_waits_its_turn(monkeypatch):
    """The bypass is for cancellation only. Two real commands must still
    serialise, or two skills run at once on one machine."""
    from jarvis.cancel import control
    from jarvis.orchestrator import orchestrator
    from jarvis.skills import load_all, registry
    from jarvis.skills.registry import ok

    load_all()
    started, release = threading.Event(), threading.Event()
    real_execute = registry.execute

    def hang(name, args, ctx):
        if name == "get_battery":
            started.set()
            release.wait(5)
            return ok("Battery is at 65 percent.")
        return real_execute(name, args, ctx)

    monkeypatch.setattr(registry, "execute", hang)
    done = threading.Event()

    def busy_turn():
        orchestrator.handle_text("what's the battery")

    worker = threading.Thread(target=busy_turn, daemon=True)
    worker.start()
    try:
        assert started.wait(3)

        def second():
            orchestrator.handle_text("what time is it")
            done.set()

        threading.Thread(target=second, daemon=True).start()
        assert not done.wait(0.4), "a second command ran while one was in flight"
    finally:
        release.set()
        worker.join(timeout=5)
        done.wait(3)


def test_cancelling_with_nothing_running_is_harmless():
    """The HUD stop button, pressed into silence."""
    from jarvis.orchestrator import orchestrator

    assert orchestrator.cancel_current_turn() is False
