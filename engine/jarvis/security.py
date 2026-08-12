"""Who may talk to the engine, and what gets written down about it.

The engine's HTTP + WebSocket API can open apps, type into the focused window
and shut the machine down. It listens on loopback, which keeps it off the
network — but loopback is not a security boundary on a desktop: any page in any
browser on this machine can issue `fetch("http://127.0.0.1:8756/command")`. The
old CORS policy (`allow_origins=["*"]`) actively invited exactly that.

So three things guard it, and they compose:

  1. **A bearer token**, minted per launch and handed to the desktop app
     through the environment. A web page cannot read it, and cannot attach an
     `Authorization` header cross-origin without a preflight this server
     refuses. This is the load-bearing control.
  2. **Origin refusal.** Any request carrying a browser `Origin` is rejected
     outright, token or not. The HUD is a `file://` page and sends none.
  3. **A session lock.** Even with the token, the engine ignores commands and
     keeps the microphone shut until the signed-in desktop app unlocks it.

Deliberately dependency-free — a security layer nobody can audit isn't one.
"""

from __future__ import annotations

import hmac
import logging
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field

from . import paths

log = logging.getLogger(__name__)

TOKEN_ENV = "JARVIS_API_TOKEN"
TOKEN_FILE = "api-token"
TOKEN_BYTES = 32

# Paths reachable without a token. `/health` has to answer before the desktop
# app has anything to authenticate with — it is how the app learns the engine
# came up at all — so it says nothing beyond "yes, I'm here".
PUBLIC_PATHS = frozenset({"/health"})


# --- the token -------------------------------------------------------------


def _read_or_create_token() -> str:
    """The launch token: from the environment, else from disk, else fresh.

    The desktop app passes one in, which is the normal path. A bare
    `python -m jarvis` gets a generated one persisted under the data directory
    so command-line tools on this machine can read it — the file is only
    reachable by this Windows account, which is the same trust boundary the
    assistant already runs inside.
    """
    from_env = os.environ.get(TOKEN_ENV, "").strip()
    if from_env:
        return from_env

    path = paths.DATA_DIR / TOKEN_FILE
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if len(existing) >= 32:
            return existing
    except OSError:
        pass

    token = secrets.token_urlsafe(TOKEN_BYTES)
    try:
        paths.ensure_dirs()
        path.write_text(token, encoding="utf-8")
        os.chmod(path, 0o600)
    except OSError as exc:
        log.warning("Could not persist the API token (%s) — it is valid for "
                    "this run only", exc)
    return token


API_TOKEN: str = _read_or_create_token()


def token_matches(candidate: str | None) -> bool:
    """Constant-time comparison, so a wrong token leaks nothing by timing."""
    if not candidate:
        return False
    return hmac.compare_digest(candidate, API_TOKEN)


def bearer_from_header(value: str | None) -> str | None:
    if not value:
        return None
    match = re.fullmatch(r"Bearer\s+(\S+)", value.strip(), re.IGNORECASE)
    return match.group(1) if match else None


# --- the session lock ------------------------------------------------------


@dataclass
class Session:
    """Whether a signed-in person is present at the machine.

    The engine boots locked. The desktop app signs the user in against
    Firebase, then unlocks — and until it does, the microphone never opens and
    `/command` refuses. That makes sign-in mean something on a voice assistant:
    not a screen to click past, but the thing that decides whether the machine
    is listening at all.
    """

    unlocked: bool = False
    account: str = ""
    uid: str = ""
    #: Unix time after which the unlock lapses and must be renewed. The desktop
    #: app renews it whenever it refreshes the user's Firebase token, so a
    #: revoked or expired account stops the assistant within one refresh cycle.
    expires_at: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def unlock(self, account: str, uid: str = "", ttl_sec: float = 3600.0) -> None:
        with self._lock:
            self.unlocked = True
            self.account = account
            self.uid = uid
            self.expires_at = time.time() + max(60.0, ttl_sec)
        log.info("Session unlocked for %s", account or "(unknown account)")

    def lock(self, reason: str = "signed out") -> None:
        with self._lock:
            was = self.unlocked
            self.unlocked = False
            self.account = ""
            self.uid = ""
            self.expires_at = 0.0
        if was:
            log.info("Session locked (%s)", reason)

    @property
    def active(self) -> bool:
        if not self.unlocked:
            return False
        if self.expires_at and time.time() > self.expires_at:
            # Don't mutate here: `active` is read from the audio thread and a
            # lock acquisition on that path would be a needless stall.
            return False
        return True

    def snapshot(self) -> dict[str, object]:
        return {
            "unlocked": self.active,
            "account": self.account,
            "expires_in": max(0, int(self.expires_at - time.time())) if self.expires_at else 0,
        }


session = Session()


# --- rate limiting ---------------------------------------------------------


class RateLimiter:
    """Fixed-window limiter. In-memory, because there is one engine per user.

    Not aimed at a remote attacker — the token already handles those — but at a
    misbehaving script on this machine that would otherwise be able to spend the
    Anthropic key as fast as the network allows.
    """

    def __init__(self, max_calls: int, window_sec: float) -> None:
        self.max_calls = max_calls
        self.window_sec = window_sec
        self._count = 0
        self._reset_at = 0.0
        self._lock = threading.Lock()

    def allow(self) -> tuple[bool, int]:
        """Returns (allowed, seconds until the window resets)."""
        now = time.monotonic()
        with self._lock:
            if now >= self._reset_at:
                self._count = 0
                self._reset_at = now + self.window_sec
            self._count += 1
            remaining = int(self._reset_at - now) + 1
            return self._count <= self.max_calls, remaining


# --- redaction -------------------------------------------------------------

# Shapes of the credentials this app handles. Provider errors quote the key
# that was rejected, and those messages reach the HUD and the log file people
# attach to bug reports.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{12,}"),        # Anthropic
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),            # OpenAI, DeepSeek, Mistral
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}"),           # Google
    re.compile(r"\bya29\.[A-Za-z0-9._-]{20,}"),        # Google OAuth
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
    re.compile(r"\b(?:api[-_]?key|authorization|bearer|token)\b\s*[:=]?\s*"
               r"[\"']?[A-Za-z0-9._-]{16,}", re.IGNORECASE),
)


def redact(text: str) -> str:
    """Replace anything credential-shaped with a marker. Never raises."""
    if not text:
        return text
    out = text
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub("[redacted]", out)
    # The launch token isn't key-shaped, so match it literally.
    if API_TOKEN and API_TOKEN in out:
        out = out.replace(API_TOKEN, "[redacted]")
    return out
