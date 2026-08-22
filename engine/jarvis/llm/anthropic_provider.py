"""Anthropic (Claude).

The one provider that keeps its SDK. Two reasons: it was here first and its
behaviour is the reference the other adapters are checked against, and its
tool-use handling is genuinely intricate — adaptive thinking, an effort budget
shared between reasoning and the reply, prompt caching on the system block, and
a `refusal` stop reason that arrives on a successful HTTP call.

Thinking stays **on**, deliberately. With it disabled this model can emit a
tool call as plain text in the visible response: the turn succeeds, the call
never runs, and nothing errors. In a voice assistant that failure is completely
invisible — you say "chrome kholo", it says "opening Chrome", and nothing
opens. Latency is managed with a low effort level instead.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    Completion,
    LlmError,
    ModelInfo,
    Provider,
    ToolCall,
    check_key_shape,
    friendly_key_error,
    pretty_model_name,
    speed_for,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"

KNOWN_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "fastest"),
    ModelInfo("claude-sonnet-5", "Claude Sonnet 5", "balanced"),
    ModelInfo("claude-opus-5", "Claude Opus 5", "most capable"),
)


def _client(api_key: str):
    import anthropic

    return anthropic.Anthropic(api_key=check_key_shape(api_key))


def _wrap(exc: Exception) -> LlmError:
    status = getattr(exc, "status_code", None)
    body = getattr(exc, "message", None) or str(exc)
    return LlmError(f"anthropic: {type(exc).__name__}: {body}",
                    friendly_key_error(status, str(body)))


def validate_key(api_key: str) -> None:
    try:
        _client(api_key).messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
    except LlmError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


def list_models(api_key: str) -> list[ModelInfo]:
    try:
        page = _client(api_key).models.list(limit=50)
    except Exception as exc:  # noqa: BLE001
        # The settings screen falls back to KNOWN_MODELS, which is enough to
        # choose from — this is a nicety, not a requirement.
        log.info("Could not list Anthropic models (%s)", exc)
        return list(KNOWN_MODELS)

    found = [
        ModelInfo(
            id=model.id,
            label=getattr(model, "display_name", None) or pretty_model_name(model.id),
            speed=speed_for(model.id),
        )
        for model in getattr(page, "data", [])
        if getattr(model, "id", "")
    ]
    return found or list(KNOWN_MODELS)


def complete(
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    api_key: str,
    model: str,
    max_tokens: int = 2048,
    effort: str = "low",
) -> Completion:
    try:
        response = _client(api_key).messages.create(
            model=model or DEFAULT_MODEL,
            max_tokens=max_tokens,
            # Marked for caching: the system prompt carries the installed-app
            # list and is identical between turns, so it is worth the marker.
            # Nothing volatile may go in here or the prefix changes every
            # request and the cache never hits.
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
            tools=tools,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
        )
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc

    # A safety classifier can decline the request while the HTTP call still
    # succeeds, so this has to be checked before reading content.
    if response.stop_reason == "refusal":
        category = getattr(getattr(response, "stop_details", None), "category", None)
        log.warning("Claude declined the request (category=%s)", category)
        return Completion(refused=True)

    calls: list[ToolCall] = []
    text_parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            calls.append(ToolCall(skill=block.name, args=dict(block.input or {})))

    return Completion(
        tool_calls=calls,
        text=" ".join(p.strip() for p in text_parts if p.strip()).strip(),
    )


PROVIDER = Provider(
    id="anthropic",
    label="Anthropic (Claude)",
    help_url="https://console.anthropic.com/settings/keys",
    default_model=DEFAULT_MODEL,
    validate_key=validate_key,
    list_models=list_models,
    complete=complete,
    known_models=KNOWN_MODELS,
)
