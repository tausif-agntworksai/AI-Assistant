# -*- coding: utf-8 -*-
"""The cost of understanding a command, pinned.

Recognition dominates a spoken turn and varies with the machine, so it is not
what this measures. What it measures is everything the assistant itself
decides to do — routing, entity resolution, the risk gate — because that is
the part a change can quietly make expensive, and the part nobody notices
until commands start feeling sluggish.

The numbers are deliberately loose. This is a tripwire for a regression of an
order of magnitude, not a benchmark; a machine under load should not turn the
suite red.
"""

import time

import pytest

from jarvis import contacts, messaging
from jarvis.app_index import AppIndex, AppEntry
from jarvis.nlu.rules import route
from jarvis.skills import load_all

load_all()

#: Routing one utterance offline. Measured at 0.03-0.14 ms for rule hits and
#: 8.4 ms when it falls through to fuzzy example matching.
ROUTE_BUDGET_MS = 50.0

#: Resolving a name against a few hundred contacts.
RESOLVE_BUDGET_MS = 50.0


def _median_ms(fn, runs: int = 20) -> float:
    """Median rather than mean: one scheduler hiccup should not fail a build."""
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    return timings[len(timings) // 2]


@pytest.mark.parametrize(
    "spoken",
    [
        "go to sleep",
        "lock my laptop",
        "open chrome",
        "take a screenshot",
        "increase volume",
        "chrome kholo",
        "text Sana on WhatsApp saying I'll be there in 10 minutes",
    ],
)
def test_routing_a_common_command_stays_cheap(spoken):
    """These are the commands given most often; none may need the model."""
    assert route(spoken) is not None, f"{spoken!r} no longer routes offline"
    assert _median_ms(lambda: route(spoken)) < ROUTE_BUDGET_MS


def test_an_unroutable_utterance_fails_fast():
    """Falling through to the model must not itself be slow."""
    spoken = "what do you think about the economic situation in argentina"
    assert route(spoken) is None
    assert _median_ms(lambda: route(spoken)) < ROUTE_BUDGET_MS


def test_resolving_a_contact_stays_cheap(tmp_path, monkeypatch):
    monkeypatch.setattr(contacts.store.paths, "STORE_DIR", tmp_path)
    monkeypatch.setattr(contacts.store.paths, "ensure_dirs", lambda: None)
    contacts.import_rows([
        {"name": f"Person {n}", "phone": f"9198765{n:05d}"} for n in range(300)
    ] + [{"name": "Sana Ahmed", "phone": "919876543210"}])

    assert messaging.resolve_recipient("Sana").found
    assert _median_ms(lambda: messaging.resolve_recipient("Sana")) < RESOLVE_BUDGET_MS


def test_resolving_an_app_stays_cheap():
    index = AppIndex()
    index.entries = [
        AppEntry(name=f"Program {n}", launch=f"c:/p{n}.exe", kind="exe")
        for n in range(300)
    ] + [AppEntry(name="Google Chrome", launch="c:/chrome.lnk", kind="shortcut",
                  aliases=["chrome", "browser"])]

    assert index.resolve("chrome").name == "Google Chrome"
    assert _median_ms(lambda: index.resolve("chrome")) < RESOLVE_BUDGET_MS
