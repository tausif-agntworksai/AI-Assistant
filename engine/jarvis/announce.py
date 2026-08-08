"""Unprompted announcements — timers firing, reminders coming due.

Skills can't import the orchestrator (it imports them), so they publish here
and the orchestrator installs the handler at startup. Without a handler,
announcements still reach the log rather than vanishing.
"""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)

# (english, hindi, kind)
Handler = Callable[[str, str, str], None]

_handler: Handler | None = None


def set_handler(fn: Handler | None) -> None:
    global _handler
    _handler = fn


def announce(en: str, hi: str = "", kind: str = "info") -> None:
    """Say something the user didn't ask for right now. Never raises."""
    log.info("Announce [%s]: %s", kind, en)
    if _handler is None:
        return
    try:
        _handler(en, hi or en, kind)
    except Exception as exc:  # noqa: BLE001
        log.error("Announcement handler failed: %s", exc)
