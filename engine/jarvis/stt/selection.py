"""Which cloud recogniser to fall back to, if any.

Speech is the second place a user can bring their own key, and the reason is
accuracy on exactly the thing this assistant is for: Deepgram's `nova-3` is
built for Hindi/English code-switching, which is where the local `base` model
struggles most.

The behaviour it buys is deliberately narrow, and the settings screen says so
plainly: **audio only leaves the machine after both local passes have already
failed to make sense of it.** A command the offline rules recognise never
reaches a network at all, and neither does one the local model transcribes
cleanly. Nothing here is on by default.

Held in memory like the language-model key, and for the same reason — the
desktop app owns it, encrypted with the OS keychain, and pushes it down over
the authenticated loopback API. Changing it must not cost a model reload, which
is why this is a selection consulted per-utterance rather than a constructor
argument baked into the transcriber at startup.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

PROVIDERS: dict[str, dict[str, str]] = {
    "deepgram": {
        "label": "Deepgram",
        "help_url": "https://console.deepgram.com/",
        "note": "nova-3 — best for Hindi and English mixed together",
    },
    "openai": {
        "label": "OpenAI Whisper",
        "help_url": "https://platform.openai.com/api-keys",
        "note": "whisper-1 — solid, a little behind Deepgram on Hinglish",
    },
}


def provider_list() -> list[dict[str, str]]:
    return [{"id": pid, **meta} for pid, meta in sorted(PROVIDERS.items())]


@dataclass
class SpeechSelection:
    provider: str = ""
    _api_key: str = ""
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # -- reading ------------------------------------------------------------

    @property
    def api_key(self) -> str:
        with self._lock:
            if self._api_key:
                return self._api_key
        return os.environ.get("CLOUD_STT_API_KEY", "").strip()

    @property
    def active_provider(self) -> str:
        with self._lock:
            if self.provider:
                return self.provider
        return os.environ.get("CLOUD_STT_PROVIDER", "").strip().lower()

    @property
    def available(self) -> bool:
        return bool(self.active_provider in PROVIDERS and self.api_key)

    def snapshot(self) -> dict[str, Any]:
        """Safe for the UI: whether a key exists, never what it is."""
        provider = self.active_provider
        return {
            "provider": provider,
            "label": PROVIDERS.get(provider, {}).get("label", ""),
            "has_key": self.available,
            "from_environment": bool(not self._api_key and self.api_key),
        }

    # -- writing ------------------------------------------------------------

    def configure(self, provider: str, api_key: str | None = None) -> None:
        provider = (provider or "").strip().lower()
        if provider and provider not in PROVIDERS:
            # Same reasoning as the language-model registry: silently falling
            # back to some other provider would hand this key to a service the
            # user never chose.
            raise ValueError(f"unknown speech provider {provider!r}")
        with self._lock:
            self.provider = provider
            if api_key is not None:
                self._api_key = api_key.strip()
        log.info("Cloud speech set to %s, key %s",
                 provider or "(off)", "present" if self.available else "absent")

    def clear(self) -> None:
        with self._lock:
            self.provider = ""
            self._api_key = ""
        log.info("Cloud speech turned off")


selection = SpeechSelection()
