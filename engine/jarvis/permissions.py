"""Permission tiers, the confirmation gate, and the audit log.

This is the "and given permissions" part of the assistant. An always-listening
program that can shut down the machine or message contacts needs one place
where every action is classified and, if consequential, confirmed — not a
judgement call scattered across thirty skill functions.

Every execution attempt is recorded, allowed or not.
"""

from __future__ import annotations

import enum
import json
import logging
import threading
import time
from typing import Any, Callable

from . import paths

log = logging.getLogger(__name__)

_AUDIT_MAX_BYTES = 1_000_000


class Risk(str, enum.Enum):
    """How much damage this action can do if the assistant misheard."""

    SAFE = "safe"          # trivially reversible: open an app, read the battery
    CONFIRM = "confirm"    # disruptive but recoverable: sleep, lock, close a window
    CRITICAL = "critical"  # data loss or reaches other people: shutdown, delete, send

    @property
    def rank(self) -> int:
        return {"safe": 0, "confirm": 1, "critical": 2}[self.value]


class PermissionDenied(Exception):
    def __init__(self, skill: str, reason: str = "not confirmed") -> None:
        super().__init__(f"{skill}: {reason}")
        self.skill = skill
        self.reason = reason


# Returns True if the user approved. Receives (english_prompt, hindi_prompt,
# risk) and is responsible for asking however it can — by voice, or via a HUD
# modal for CRITICAL actions.
Confirmer = Callable[[str, str, Risk], bool]


class PermissionGate:
    def __init__(self, cfg=None) -> None:
        if cfg is None:
            from .config import settings

            cfg = settings.permissions
        self.cfg = cfg
        self._confirmer: Confirmer | None = None
        self._lock = threading.Lock()

    def set_confirmer(self, fn: Confirmer | None) -> None:
        """Install the thing that actually asks the user."""
        self._confirmer = fn

    def check(
        self,
        skill_name: str,
        risk: Risk,
        prompt_en: str,
        prompt_hi: str,
        language: str = "en",
    ) -> bool:
        """Decide whether `skill_name` may run. Blocks while asking the user."""
        if risk is Risk.SAFE:
            return True

        if risk is Risk.CONFIRM and not self.cfg.confirm_tier_enabled:
            log.debug("Confirm tier disabled by config; allowing %s", skill_name)
            return True

        if risk is Risk.CRITICAL and self.cfg.allow_critical_without_confirm:
            log.warning("Critical action %s allowed without confirmation (config)", skill_name)
            return True

        if self._confirmer is None:
            # Refusing is the only safe default: with nothing able to ask the
            # user, "allow" would mean silently shutting down the machine.
            log.warning("No confirmer installed; denying %s (%s)", skill_name, risk.value)
            return False

        with self._lock:
            try:
                approved = bool(self._confirmer(prompt_en, prompt_hi, risk))
            except Exception as exc:  # noqa: BLE001
                log.error("Confirmation failed for %s: %s", skill_name, exc)
                approved = False

        log.info("Confirmation for %s (%s): %s", skill_name, risk.value,
                 "approved" if approved else "denied")
        return approved


def audit(
    action: str,
    *,
    risk: str = "safe",
    args: dict[str, Any] | None = None,
    allowed: bool = True,
    ok: bool | None = None,
    detail: str = "",
    source: str = "voice",
    transcript: str = "",
) -> None:
    """Append one line to the audit log. Never raises."""
    record = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "risk": risk,
        "args": args or {},
        "allowed": allowed,
        "ok": ok,
        "detail": detail[:500],
        "source": source,
        "transcript": transcript[:300],
    }
    try:
        paths.ensure_dirs()
        _rotate_if_needed()
        with paths.AUDIT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 - auditing must never break the action
        log.debug("Could not write audit record: %s", exc)


def _rotate_if_needed() -> None:
    try:
        if paths.AUDIT_LOG.exists() and paths.AUDIT_LOG.stat().st_size > _AUDIT_MAX_BYTES:
            backup = paths.AUDIT_LOG.with_suffix(".jsonl.1")
            backup.unlink(missing_ok=True)
            paths.AUDIT_LOG.rename(backup)
    except OSError:
        pass


def read_audit(limit: int = 100) -> list[dict[str, Any]]:
    """Most recent audit records, newest first. For the HUD."""
    if not paths.AUDIT_LOG.exists():
        return []
    try:
        lines = paths.AUDIT_LOG.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    out: list[dict[str, Any]] = []
    for line in reversed(lines[-limit * 2 :]):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out


gate = PermissionGate()
