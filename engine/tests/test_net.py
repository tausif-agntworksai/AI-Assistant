# -*- coding: utf-8 -*-
"""Working without internet, and saying so when something can't.

Roughly fifty commands here are pure local control and must keep working with
the router unplugged. A handful genuinely need the network. Before this, losing
the connection made those few fail through the generic handler and say "That
didn't work." — which is indistinguishable from a bug, and invites the user to
repeat a command that cannot succeed.

The properties worth holding on to are about *which* way each failure leans:
never block a working feature on a bad connectivity guess, and never blame the
network for an expired API key.
"""

import socket

import pytest
import requests

from jarvis import net
from jarvis.permissions import Risk
from jarvis.skills.registry import SkillContext, SkillSpec, ok, registry


@pytest.fixture(autouse=True)
def _forget_cached_state():
    net.invalidate()
    yield
    net.invalidate()


# --- what counts as a connectivity failure ---------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        socket.gaierror("name resolution failed"),
        socket.timeout("timed out"),
        ConnectionError("reset"),
        ConnectionRefusedError("refused"),
        requests.exceptions.ConnectionError("no route"),
        requests.exceptions.ConnectTimeout("slow"),
        requests.exceptions.ReadTimeout("slow"),
    ],
)
def test_a_transport_failure_is_read_as_being_offline(exc):
    assert net.looks_like_connectivity(exc)


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad json"),
        KeyError("missing"),
        PermissionError("denied"),
        RuntimeError("provider said no"),
    ],
)
def test_an_ordinary_error_is_not_blamed_on_the_network(exc):
    """A 401, a 404 or a rate limit is a working connection delivering bad news.
    Calling those "offline" sends the user to check their wi-fi over an expired
    API key, which is the most annoying possible wrong answer."""
    assert not net.looks_like_connectivity(exc)


def test_a_wrapped_requests_error_is_still_recognised():
    """requests wraps socket errors in its own hierarchy, which `isinstance`
    against the socket classes would miss."""

    class Custom(requests.exceptions.ConnectionError):
        pass

    assert net.looks_like_connectivity(Custom("nested"))


# --- the probe -------------------------------------------------------------


def test_being_unsure_counts_as_online(monkeypatch):
    """A false "you are offline" blocks a feature that works, which is worse
    than a slow failure. So anything unexpected in the check itself leans
    towards letting the skill run and produce its own error."""

    def explode(*_args, **_kwargs):
        raise MemoryError("something very odd")

    monkeypatch.setattr(net.socket, "create_connection", explode)
    assert net.is_online() is True


def test_a_refused_socket_means_offline(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError("unreachable")

    monkeypatch.setattr(net.socket, "create_connection", refuse)
    assert net.is_online() is False


def test_one_provider_being_blocked_is_not_an_outage(monkeypatch):
    """Some networks block 1.1.1.1 specifically. Reading that as "no internet"
    would take the assistant offline on a working connection."""
    seen: list[tuple] = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def connect(address, timeout=None):
        seen.append(address)
        if address[0] == net.PROBES[0][0]:
            raise OSError("blocked")
        return Socket()

    monkeypatch.setattr(net.socket, "create_connection", connect)
    assert net.is_online() is True
    assert len(seen) == 2, "the second probe was never tried"


def test_the_result_is_cached_so_local_commands_pay_nothing(monkeypatch):
    """A burst of commands must not be a burst of socket connections."""
    calls = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(
        net.socket, "create_connection",
        lambda *a, **k: (calls.append(1), Socket())[1],
    )
    for _ in range(20):
        net.is_online()
    assert len(calls) == 1


def test_a_failure_seen_in_the_wild_invalidates_the_cache(monkeypatch):
    """A ConnectionError out of a provider is better evidence about the
    connection than any probe, and it arrived for free."""
    calls = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(
        net.socket, "create_connection",
        lambda *a, **k: (calls.append(1), Socket())[1],
    )
    net.is_online()
    net.note_failure()
    net.is_online()
    assert len(calls) == 2


# --- the message -----------------------------------------------------------


def test_the_message_names_the_feature_not_the_failure():
    english, hindi = net.offline_message("The weather", "Mausam")
    assert "weather" in english.lower()
    assert "error" not in english.lower() and "failed" not in english.lower()
    assert "Mausam" in hindi


def test_the_message_says_the_rest_still_works():
    """The point of the whole exercise: one feature being unavailable must not
    read as the assistant being broken."""
    english, hindi = net.offline_message()
    assert "still works" in english
    assert "chalte rahenge" in hindi


# --- the gate in front of the skills ---------------------------------------


def _register(name: str, **kwargs) -> str:
    registry.add(SkillSpec(
        name=name, func=lambda: ok("ran"), description="test skill",
        risk=Risk.SAFE, category="knowledge", **kwargs,
    ))
    return name


def test_a_network_skill_is_refused_offline_with_a_useful_reason(monkeypatch):
    name = _register("_test_needs_net", needs_network=True,
                     network_label="The weather", network_label_hi="Mausam")
    monkeypatch.setattr(net, "is_online", lambda force=False: False)
    result = registry.execute(name, {}, SkillContext())
    assert not result.ok
    assert "weather" in result.text("en").lower()
    assert "internet" in result.text("en").lower()


def test_a_local_skill_runs_offline(monkeypatch):
    """The whole point. Fifty-odd commands are local control and must not care
    about the connection at all."""
    name = _register("_test_local_only")
    monkeypatch.setattr(net, "is_online", lambda force=False: False)
    result = registry.execute(name, {}, SkillContext())
    assert result.ok


def test_a_local_skill_never_triggers_a_connectivity_check(monkeypatch):
    """Checking would put a socket timeout in front of "volume up"."""
    checked = []
    name = _register("_test_no_check")
    monkeypatch.setattr(
        net, "is_online", lambda force=False: (checked.append(1), True)[1]
    )
    registry.execute(name, {}, SkillContext())
    assert checked == []


def test_a_dry_run_is_not_blocked_by_being_offline(monkeypatch):
    """`--dry-run` resolves a command without performing it, so there is nothing
    for the connection to matter to."""
    name = _register("_test_dry", needs_network=True)
    monkeypatch.setattr(net, "is_online", lambda force=False: False)
    result = registry.execute(name, {}, SkillContext(dry_run=True))
    assert result.ok


def test_the_shipped_network_skills_are_exactly_the_ones_that_reach_out():
    """A new skill that calls out and forgets this flag gets "That didn't work."
    offline, which is the bug this list exists to prevent."""
    from jarvis.skills import load_all

    load_all()
    # The fixtures above register into the real registry, and no shipped skill
    # starts with an underscore.
    flagged = {
        s.name for s in registry.all()
        if s.needs_network and not s.name.startswith("_")
    }
    assert flagged == {
        "answer_question", "translate_text", "summarize_clipboard",
        "get_weather", "get_news", "web_search", "youtube_search",
    }


def test_opening_a_page_is_not_gated_on_the_connection():
    """`open_website` and `open_url` can point at localhost, a router or an
    intranet host, all of which work with no internet at all. Refusing them
    offline would block something that works; they warn instead."""
    from jarvis.skills import load_all

    load_all()
    for name in ("open_website", "open_url"):
        assert registry.get(name).needs_network is False
