"""The provider registry, and which provider is currently selected.

Six providers behind one interface, plus the runtime choice. The choice lives
**in memory only**: the key itself is stored by the desktop app, encrypted with
the OS keychain, and pushed down here over the authenticated loopback API on
unlock and whenever it changes. Nothing in the engine writes it to disk, and
`GET /llm` reports whether a key is present without ever echoing it back.

Two traps from the sibling AI Calculator, deliberately not repeated:

  * An unrecognised provider id there falls back to Gemini **carrying whatever
    key was supplied** — so a stale id in storage hands an OpenAI key to
    Google. Here an unknown id is rejected.
  * `defaultAvailable` is hardcoded to the literal string `"gemini"` in three
    separate files. Here whether a provider can run without a user key is a
    property of the provider, asked once.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

from .base import (
    Completion,
    LlmError,
    ModelInfo,
    Provider,
    ToolCall,
    check_key_shape,
)

log = logging.getLogger(__name__)

__all__ = [
    "Completion",
    "LlmError",
    "ModelInfo",
    "Provider",
    "ToolCall",
    "PROVIDERS",
    "get_provider",
    "provider_list",
    "selection",
    "validate_key",
    "list_models",
]


def _build_registry() -> dict[str, Provider]:
    from . import anthropic_provider, gemini_provider
    from .openai_compatible import FLAVOURS, make_provider

    providers = [anthropic_provider.PROVIDER, gemini_provider.PROVIDER]
    providers += [make_provider(flavour) for flavour in FLAVOURS]
    return {provider.id: provider for provider in providers}


PROVIDERS: dict[str, Provider] = _build_registry()

#: Offered first in the settings screen. Anthropic because the assistant's
#: prompts and tool-use behaviour were built and tested against it.
DEFAULT_PROVIDER = "anthropic"


def get_provider(provider_id: str) -> Provider:
    """Look up a provider, or refuse.

    Refusing matters more than it looks: the alternative — quietly falling back
    to a default — means handing one provider's key to a different provider's
    API, which leaks the key to a third party and reports the failure as "that
    key was rejected".
    """
    provider = PROVIDERS.get((provider_id or "").strip().lower())
    if provider is None:
        known = ", ".join(sorted(PROVIDERS))
        raise LlmError(
            f"unknown provider {provider_id!r}",
            f"I don't know that AI provider. Choose one of: {known}.",
        )
    return provider


def provider_list() -> list[dict[str, Any]]:
    """What the settings screen renders before any key has been entered."""
    return [PROVIDERS[pid].as_dict() for pid in sorted(PROVIDERS)]


# --- the current choice ----------------------------------------------------


@dataclass
class Selection:
    """Which provider, which model, and the key — held in memory, never written.

    A key set through the environment (`ANTHROPIC_API_KEY`) is honoured so that
    a development checkout and `python -m jarvis --text` keep working exactly as
    they did. A packaged build ships no environment key, which is the whole
    point: it is handed to strangers, and a key inside it would be both
    extractable and billed to whoever built the installer.
    """

    provider_id: str = DEFAULT_PROVIDER
    model: str = ""
    _api_key: str = ""
    _lock: threading.RLock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = threading.RLock()

    # -- reading ------------------------------------------------------------

    @property
    def provider(self) -> Provider:
        return get_provider(self.provider_id)

    @property
    def api_key(self) -> str:
        with self._lock:
            if self._api_key:
                return self._api_key
        return self._key_from_environment()

    def _key_from_environment(self) -> str:
        """The pre-BYOK path: a key in `.env`, for this provider only."""
        variable = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }.get(self.provider_id, "")
        return os.environ.get(variable, "").strip() if variable else ""

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    @property
    def active_model(self) -> str:
        return self.model or self.provider.default_model

    def snapshot(self) -> dict[str, Any]:
        """Safe to send to the UI: says whether a key exists, never what it is."""
        return {
            "provider": self.provider_id,
            "label": self.provider.label,
            "model": self.active_model,
            "has_key": self.available,
            "from_environment": bool(not self._api_key and self.available),
        }

    # -- writing ------------------------------------------------------------

    def configure(self, provider_id: str, model: str = "", api_key: str | None = None) -> None:
        """Apply a choice from the settings screen. Validates the provider id."""
        provider = get_provider(provider_id)
        with self._lock:
            self.provider_id = provider.id
            self.model = (model or "").strip()
            if api_key is not None:
                self._api_key = check_key_shape(api_key) if api_key.strip() else ""
        log.info("Language model set to %s (%s), key %s",
                 provider.label, self.active_model,
                 "present" if self.available else "absent")

    def clear_key(self) -> None:
        with self._lock:
            self._api_key = ""
        log.info("Language-model key cleared")


selection = Selection()


# --- one-off operations the settings screen needs --------------------------


def validate_key(provider_id: str, api_key: str, model: str = "") -> None:
    """Prove a key works before it is saved. Raises LlmError if it doesn't."""
    provider = get_provider(provider_id)
    provider.validate_key(api_key)
    # A key can be valid and still have no access to the chosen model, which is
    # a different problem with a different fix — so check it separately rather
    # than letting the user discover it on their first question.
    if model and model != provider.default_model:
        provider.complete(
            system="Reply with the single word ok.",
            messages=[{"role": "user", "content": "ping"}],
            tools=[],
            api_key=api_key,
            model=model,
            max_tokens=4,
        )


def list_models(provider_id: str, api_key: str) -> list[dict[str, Any]]:
    provider = get_provider(provider_id)
    return [model.as_dict() for model in provider.list_models(api_key)]
