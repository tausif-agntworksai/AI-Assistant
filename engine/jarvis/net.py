"""Is there internet, and what should the assistant say when there isn't?

Most of what this assistant does needs no network at all: opening apps, volume,
brightness, windows, power, timers, the clipboard, file search — around fifty
commands that are pure local control. A handful genuinely need to reach out:
the model, weather, news, and Edge's neural voices.

Before this module, losing the connection made those few fail through the
generic handler in `registry.execute()` and say **"That didn't work."** That is
the worst available answer, because it is indistinguishable from a bug: it
invites the user to try again, to check their command, to wonder whether the
assistant is broken. "That needs the internet" ends the matter in four words.

Two rules shape the design:

* **Never make a working command wait on a network check.** The check is only
  consulted for skills that declared they need one, and the result is cached, so
  a local command never pays for it.
* **Assume online.** If the check itself fails or is inconclusive, the skill
  runs and is allowed to produce its own error. A false "you are offline" would
  block a working feature, which is worse than a slow failure.
"""

from __future__ import annotations

import logging
import socket
import threading
import time

log = logging.getLogger(__name__)

#: How long a reachability result stays good for. Long enough that a burst of
#: commands costs one check, short enough that reconnecting is noticed without
#: restarting anything.
TTL_ONLINE = 30.0

#: Rechecked sooner when offline: someone who just lost their connection is
#: more likely to be waiting to get it back than someone who has it.
TTL_OFFLINE = 5.0

#: A TCP connect, not a DNS lookup or an HTTP request. DNS can be answered by a
#: captive portal or a stale cache and HTTP needs a round trip we don't need —
#: opening a socket to a well-known resolver on 53 is the cheapest honest
#: question. Two of them, so one provider being blocked is not read as an
#: outage.
PROBES = (("1.1.1.1", 53), ("8.8.8.8", 53))

#: Short, because this sits between the user and an answer. A connection slow
#: enough to miss this is too slow for the model anyway.
TIMEOUT = 1.5

_lock = threading.Lock()
_state: tuple[bool, float] | None = None  # (online, checked_at)


def is_online(force: bool = False) -> bool:
    """Whether the internet is reachable, cached. Assumes yes when unsure."""
    global _state

    with _lock:
        if _state is not None and not force:
            online, checked_at = _state
            ttl = TTL_ONLINE if online else TTL_OFFLINE
            if time.monotonic() - checked_at < ttl:
                return online

    online = _probe()
    with _lock:
        _state = (online, time.monotonic())
    return online


def _probe() -> bool:
    for host, port in PROBES:
        try:
            with socket.create_connection((host, port), timeout=TIMEOUT):
                return True
        except OSError:
            continue
        except Exception as exc:  # noqa: BLE001
            # Anything unexpected here is a bug in the check, not evidence of
            # an outage, and must not take a working feature offline with it.
            log.debug("Connectivity probe raised %s: %s", type(exc).__name__, exc)
            return True
    return False


def invalidate() -> None:
    """Forget the cached result — the next question re-probes."""
    global _state
    with _lock:
        _state = None


def note_failure() -> None:
    """Record that something network-shaped just failed.

    Called from the exception path rather than guessing: a `ConnectionError` out
    of a provider is better evidence about the connection than any probe, and it
    arrived for free. This only invalidates the cache, so the next skill that
    cares re-checks rather than trusting a stale "you were online 20 seconds
    ago".
    """
    invalidate()


#: Exception types that mean "the network, not the request". Kept here so the
#: LLM providers, the skills and the speech backends all agree on what counts.
def looks_like_connectivity(exc: BaseException) -> bool:
    """True when this exception is about reaching the network at all.

    Deliberately narrow. A 401, a 404 or a rate limit is a working connection
    delivering bad news, and calling those "offline" would send the user to
    check their wi-fi over an expired API key.
    """
    if isinstance(exc, (socket.gaierror, socket.timeout, ConnectionError)):
        return True

    # requests and httpx are optional at import time here, so they are matched
    # by name rather than by class. Both wrap the socket errors above in their
    # own hierarchy, which `isinstance` would otherwise miss.
    for cls in type(exc).__mro__:
        qualified = f"{cls.__module__}.{cls.__name__}"
        if qualified in _CONNECTIVITY_NAMES:
            return True
    return False


_CONNECTIVITY_NAMES = frozenset({
    "requests.exceptions.ConnectionError",
    "requests.exceptions.ConnectTimeout",
    "requests.exceptions.ReadTimeout",
    "requests.exceptions.Timeout",
    "httpx.ConnectError",
    "httpx.ConnectTimeout",
    "httpx.ReadTimeout",
    "httpx.NetworkError",
    "urllib.error.URLError",
    "aiohttp.client_exceptions.ClientConnectorError",
})


def offline_message(what: str = "", what_hi: str = "") -> tuple[str, str]:
    """What to say when a feature cannot run without a connection.

    Names the feature rather than the failure. "Weather needs the internet" is
    actionable; "network error" is a status code read aloud.
    """
    subject = what.strip() or "That"
    subject_hi = what_hi.strip() or "Ye"
    return (
        f"{subject} needs the internet, and there's no connection right now. "
        "Everything on your computer still works.",
        f"{subject_hi} ke liye internet chahiye, aur abhi connection nahi hai. "
        "Computer ke saare kaam waise hi chalte rahenge.",
    )
