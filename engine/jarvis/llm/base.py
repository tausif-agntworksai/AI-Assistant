"""The shape every language-model provider has to fit.

Jarvis ships with no API key. That is deliberate — the installer is meant to be
handed to strangers, and a key baked into it would be both extractable and
billed to whoever built it. So the ~50 commands the offline rules recognise work
immediately, and conversation, questions and translation ask for a key the user
supplies.

Six providers, one interface. The interesting design constraint is that this is
a *voice* assistant, which makes model choice a latency decision as much as a
quality one: Groq answers in a couple of hundred milliseconds and Opus takes
several seconds, and for "chrome kholo" the difference between those is the
difference between a product and a demo. So the user picks a model, not just a
provider, and every model carries a speed hint.

Deliberately thin on dependencies. Only Anthropic gets an SDK, because its
tool-use-plus-thinking handling is genuinely intricate; everything else speaks
plain HTTP through `requests`, which the engine already depends on. Three more
SDKs would each land in the PyInstaller bundle, and the installer is already
212 MB.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 60
VALIDATE_TIMEOUT = 20

#: Longest key we will even look at. Nothing legitimate is close to this, and
#: it bounds what an accidental paste can send.
MAX_KEY_CHARS = 512


class LlmError(Exception):
    """Anything that went wrong talking to a provider.

    `friendly` is safe to speak out loud or show in the UI; `str(exc)` may
    contain provider detail worth logging but not worth reading aloud.
    """

    def __init__(self, detail: str, friendly: str) -> None:
        super().__init__(detail)
        self.friendly = friendly


class Speed(str):
    """How quick a model feels, as a label the settings screen can show."""

    FAST = "fastest"
    BALANCED = "balanced"
    DEEP = "most capable"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    #: One of Speed's values, or "" when we have no opinion about this model.
    speed: str = ""
    #: False for models that can't be given tools, which rules them out for
    #: anything except plain question answering.
    supports_tools: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "speed": self.speed,
            "supports_tools": self.supports_tools,
        }


@dataclass
class ToolCall:
    skill: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Completion:
    """What one turn produced, whichever provider produced it."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    refused: bool = False


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    help_url: str
    default_model: str
    #: Reads a key and either returns or raises LlmError. One live call, because
    #: the provider is the only authority on whether a key works — guessing from
    #: an `sk-` prefix tells you nothing about whether it has been revoked.
    validate_key: Callable[[str], None]
    #: The provider's own model list, filtered to things worth offering.
    list_models: Callable[[str], list[ModelInfo]]
    #: One turn: system prompt, messages, Anthropic-shaped tool schemas.
    complete: Callable[..., Completion]
    #: Models we know about even before a key is entered, so the settings
    #: screen has something to show while the list loads (or if it fails).
    known_models: tuple[ModelInfo, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "help_url": self.help_url,
            "default_model": self.default_model,
            "known_models": [m.as_dict() for m in self.known_models],
        }


# --- error mapping ---------------------------------------------------------

# Providers quote the key that was rejected inside the very message saying it
# was rejected, and these messages reach the HUD and the log file people attach
# to bug reports. Everything user-facing goes through jarvis.security.redact.
_KEY_REJECTED = "That API key was rejected — check you copied all of it."
_RATE_LIMITED = "That key works, but the account is rate-limited right now."
_NO_ACCESS = "That key is valid but has no access to this model. Pick another one."
_QUOTA = "That account is out of quota. Check its billing, or use another key."

_BAD_KEY_TEXT = re.compile(
    r"api[ _-]?key not valid|invalid[ _-]api[ _-]key|incorrect api key|"
    r"unauthenticated|unauthorized|invalid authentication",
    re.IGNORECASE,
)
_NO_MODEL_TEXT = re.compile(
    r"model.{0,20}(not found|does not exist|not exist|unavailable|"
    r"no access|not allowed|decommissioned)|"
    r"(unknown|invalid|unsupported).{0,10}model",
    re.IGNORECASE,
)
_QUOTA_TEXT = re.compile(
    r"quota|insufficient[ _-]?(quota|balance|credit)|billing|resource[ _-]?exhausted",
    re.IGNORECASE,
)


def friendly_key_error(status: int | None, raw: str) -> str:
    """Turn a provider's rejection into something a person can act on.

    Status first, then the text — but with one important exception the AI
    Calculator gets wrong. It maps HTTP 400 to "that key looks malformed",
    which is right for Gemini and wrong for every OpenAI-compatible provider:
    Groq, DeepSeek and Mistral all answer 400 or 404 for an unknown *model*.
    Telling someone their key is broken when the model name is what's wrong
    sends them to re-copy a key that was fine all along.
    """
    from ..security import redact

    text = redact(raw or "")

    if _NO_MODEL_TEXT.search(text):
        return _NO_ACCESS
    if status in (401, 403) or _BAD_KEY_TEXT.search(text):
        return _KEY_REJECTED
    if status == 429:
        return _QUOTA if _QUOTA_TEXT.search(text) else _RATE_LIMITED
    if _QUOTA_TEXT.search(text):
        return _QUOTA
    if status == 404:
        return _NO_ACCESS
    if status == 400:
        return "That request was rejected. Check the key and the chosen model."
    if status in (500, 502, 503, 529):
        return "That provider is having trouble right now. Try again shortly."

    first_line = text.strip().splitlines()[0][:180] if text.strip() else "unknown error"
    return f"Could not reach the model: {first_line}"


def check_key_shape(api_key: str) -> str:
    """Cheap sanity checks, before spending a network round-trip on it."""
    key = (api_key or "").strip()
    if not key:
        raise LlmError("empty key", "Paste an API key first.")
    if len(key) > MAX_KEY_CHARS:
        raise LlmError("key too long", "That doesn't look like an API key.")
    if "\n" in key or " " in key:
        raise LlmError("key has whitespace",
                       "That key has a space or line break in it — paste it again.")
    return key


# --- speed hints -----------------------------------------------------------

# Matched against a model id in order, first hit wins. Names rather than an
# exhaustive list, because every provider renames its models every few months
# and an unknown model should render without a badge rather than disappear.
_SPEED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"opus|gpt-5|o[13]-|reasoner|large|ultra|pro\b"), Speed.DEEP),
    (re.compile(r"haiku|mini|flash|instant|8b|small|lite|nano|turbo"), Speed.FAST),
    (re.compile(r"sonnet|gpt-4|llama|mixtral|medium|chat"), Speed.BALANCED),
)


def speed_for(model_id: str) -> str:
    lowered = model_id.lower()
    for pattern, speed in _SPEED_PATTERNS:
        if pattern.search(lowered):
            return speed
    return ""


def pretty_model_name(model_id: str) -> str:
    """A readable label from an id like `claude-sonnet-5` or `gpt-4o-mini`."""
    cleaned = re.sub(r"[-_]", " ", model_id)
    cleaned = re.sub(r"\s+(latest|preview)$", "", cleaned)
    return cleaned[:1].upper() + cleaned[1:]
