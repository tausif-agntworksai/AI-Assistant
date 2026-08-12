"""Permission tiers, capability consent, the confirmation gate, and the audit log.

This is the "and given permissions" part of the assistant. An always-listening
program that can shut down the machine or message contacts needs one place
where every action is classified and, if consequential, confirmed — not a
judgement call scattered across seventy skill functions.

There are two independent gates, and they answer different questions:

  **Consent** — *may this program touch that part of the machine at all?*
  Granted once, up front, per capability (microphone, screen, clipboard,
  shell, …). This is the install-time question, and the answer is a setting.

  **Risk tier** — *given that it may, should it ask before doing this
  particular thing?* Asked every time, out loud, for anything disruptive.

Consent is checked first: refusing at the capability level means the action
never reaches the confirmation prompt, so revoking "system power" removes the
ability to shut the machine down rather than merely adding a question to it.

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


class Capability(str, enum.Enum):
    """A part of the machine the assistant can reach.

    These are the things the user is asked about on first run. They are chosen
    to match how a person thinks about their computer ("can it read my
    clipboard?"), not how the code is organised — a capability that can't be
    explained in one sentence isn't one a user can meaningfully consent to.
    """

    MICROPHONE = "microphone"          # always-on listening for the wake word
    SPEAKER = "speaker"                # spoken replies
    CAMERA = "camera"                  # declared, not used by any skill today
    SCREEN = "screen_capture"          # screenshots
    FILES = "files"                    # open folders, search personal folders
    CLIPBOARD = "clipboard"            # read and write the clipboard
    INPUT = "input_synthesis"          # typing into the focused window
    APPS = "app_control"               # launch and close applications
    WINDOWS = "window_control"         # move, snap, minimise, close windows
    POWER = "system_power"             # shutdown, restart, sleep, lock, sign out
    SETTINGS = "system_settings"       # brightness, wi-fi, bluetooth, night light
    DEVICE_STATUS = "device_status"    # battery, CPU, disk, network readings
    SHELL = "shell"                    # runs PowerShell / command-line tools
    NETWORK = "network"                # Claude, weather, news, neural voices
    MESSAGING = "messaging"            # WhatsApp and email drafts

    @property
    def label(self) -> str:
        return _CAPABILITY_LABELS[self][0]

    @property
    def label_hi(self) -> str:
        return _CAPABILITY_LABELS[self][1]


_CAPABILITY_LABELS: dict[Capability, tuple[str, str]] = {
    Capability.MICROPHONE: ("the microphone", "माइक्रोफ़ोन"),
    Capability.SPEAKER: ("the speakers", "स्पीकर"),
    Capability.CAMERA: ("the camera", "कैमरा"),
    Capability.SCREEN: ("screen capture", "स्क्रीन कैप्चर"),
    Capability.FILES: ("your files", "आपकी फ़ाइलें"),
    Capability.CLIPBOARD: ("the clipboard", "क्लिपबोर्ड"),
    Capability.INPUT: ("typing for you", "आपकी जगह टाइप करना"),
    Capability.APPS: ("opening and closing apps", "ऐप्स खोलना और बंद करना"),
    Capability.WINDOWS: ("managing windows", "विंडो संभालना"),
    Capability.POWER: ("power controls", "पावर कंट्रोल"),
    Capability.SETTINGS: ("Windows settings", "विंडोज़ सेटिंग्स"),
    Capability.DEVICE_STATUS: ("device readings", "डिवाइस की जानकारी"),
    Capability.SHELL: ("running system commands", "सिस्टम कमांड चलाना"),
    Capability.NETWORK: ("the internet", "इंटरनेट"),
    Capability.MESSAGING: ("messaging apps", "मैसेजिंग ऐप्स"),
}


# What each skill category needs by default. One line per category keeps the
# mapping auditable; the whole point of this module is that a person can read
# it in one sitting and know what the assistant can reach.
_CATEGORY_CAPABILITY: dict[str, Capability] = {
    "apps": Capability.APPS,
    "windows": Capability.WINDOWS,
    "audio": Capability.SPEAKER,
    "media": Capability.SPEAKER,
    "display": Capability.SETTINGS,
    "device": Capability.DEVICE_STATUS,
    "files": Capability.FILES,
    "knowledge": Capability.NETWORK,
    "messaging": Capability.MESSAGING,
    "system": Capability.POWER,
    "web": Capability.NETWORK,
}

# Skills whose capability doesn't follow their category. Each of these reaches
# something the rest of its category doesn't.
_SKILL_CAPABILITY: dict[str, Capability] = {
    "take_screenshot": Capability.SCREEN,
    "read_clipboard": Capability.CLIPBOARD,
    "copy_to_clipboard": Capability.CLIPBOARD,
    "summarize_clipboard": Capability.CLIPBOARD,
    "type_text": Capability.INPUT,
    "toggle_wifi": Capability.SETTINGS,
    "toggle_bluetooth": Capability.SETTINGS,
    "open_settings": Capability.SETTINGS,
    "get_network_status": Capability.DEVICE_STATUS,
    "open_website": Capability.NETWORK,
    "open_url": Capability.NETWORK,
    "web_search": Capability.NETWORK,
    "youtube_search": Capability.NETWORK,
    # Productivity is local-only (timers, notes, reminders) — nothing to grant.
}


def capability_for(skill_name: str, category: str) -> Capability | None:
    """Which capability a skill needs, or None when it needs nothing special."""
    override = _SKILL_CAPABILITY.get(skill_name)
    if override is not None:
        return override
    return _CATEGORY_CAPABILITY.get(category)


class PermissionDenied(Exception):
    def __init__(self, skill: str, reason: str = "not confirmed") -> None:
        super().__init__(f"{skill}: {reason}")
        self.skill = skill
        self.reason = reason


# --- consent ---------------------------------------------------------------


class ConsentStore:
    """What the user agreed the assistant may reach.

    Written by the desktop app's first-run permission screen and mirrored here
    so the engine enforces it rather than trusting the UI. Running the engine
    straight from the command line leaves no consent file — and that is treated
    as "not asked", which allows everything. Typing `python -m jarvis` is its
    own consent; silently refusing to work for someone who launched it by hand
    would be a puzzle, not a safeguard.
    """

    FILE = "consent.json"
    VERSION = 1

    def __init__(self) -> None:
        self._granted: dict[str, bool] = {}
        self._asked = False
        self._lock = threading.Lock()
        self.load()

    @property
    def path(self):
        return paths.DATA_DIR / self.FILE

    @property
    def asked(self) -> bool:
        """True once the user has been through the permission screen."""
        return self._asked

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._granted, self._asked = {}, False
            return
        granted = raw.get("granted")
        if not isinstance(granted, dict):
            self._granted, self._asked = {}, False
            return
        self._granted = {str(k): bool(v) for k, v in granted.items()}
        self._asked = True
        log.info("Consent loaded: %d of %d capabilities granted",
                 sum(self._granted.values()), len(Capability))

    def save(self, granted: dict[str, bool]) -> None:
        """Replace the record. Called when the desktop app pushes a decision."""
        with self._lock:
            known = {c.value for c in Capability}
            self._granted = {k: bool(v) for k, v in granted.items() if k in known}
            self._asked = True
            try:
                paths.ensure_dirs()
                self.path.write_text(
                    json.dumps(
                        {
                            "version": self.VERSION,
                            "decided_at": time.time(),
                            "granted": self._granted,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError as exc:
                log.error("Could not persist consent (%s) — it applies to this "
                          "session only", exc)
        log.info("Consent updated: %s",
                 ", ".join(sorted(k for k, v in self._granted.items() if v)) or "(none)")

    def allows(self, capability: Capability | None) -> bool:
        if capability is None:
            return True
        if not self._asked:
            return True  # never asked → command-line use, see the class docstring
        return self._granted.get(capability.value, False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "asked": self._asked,
            "granted": {c.value: self.allows(c) for c in Capability},
        }


consent = ConsentStore()


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

    @staticmethod
    def check_consent(capability: Capability | None) -> bool:
        """Whether the user granted this capability at all."""
        return consent.allows(capability)

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
    account: str = "",
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
        "account": account[:120],
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
