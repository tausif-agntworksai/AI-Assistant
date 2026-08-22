"""Google Gemini, over the REST API.

No SDK, for the same reason the OpenAI-compatible adapter has none: `requests`
is already a dependency and every extra SDK lands in the frozen bundle.

Gemini's function calling differs from everyone else's in three ways that all
have to be handled here rather than leaked upward:

  * the API key goes in a header, not a bearer token;
  * the system prompt is `systemInstruction`, a separate top-level field;
  * roles are `user` and `model` (not `assistant`), and message content is a
    list of `parts` rather than a string.

It also rejects JSON Schema keywords it doesn't recognise, which is why the
tool schemas get filtered on the way through.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    REQUEST_TIMEOUT,
    VALIDATE_TIMEOUT,
    Completion,
    LlmError,
    ModelInfo,
    Provider,
    ToolCall,
    check_key_shape,
    classify,
    pretty_model_name,
    speed_for,
)

log = logging.getLogger(__name__)

BASE = "https://generativelanguage.googleapis.com/v1beta"

# Models that advertise generateContent but are not conversational: image,
# music, robotics and retrieval endpoints. Offering them in a voice assistant's
# model picker would be offering a choice that cannot work.
_NOT_CHAT = (
    "embedding", "aqa", "imagen", "veo", "tts", "image", "lyria",
    "nano-banana", "robotics", "vision-", "learnlm",
)
# An alias rather than a pinned version, deliberately. `gemini-2.5-flash` was
# hardcoded here and Google has since stopped serving it to new keys — "This
# model is no longer available to new users" — so a perfectly good key was
# rejected with what looked like a key problem. Pinned model ids rot; the
# `-latest` aliases are Google's answer to that and cost nothing to prefer.
DEFAULT_MODEL = "gemini-flash-latest"

KNOWN_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("gemini-flash-lite-latest", "Gemini Flash-Lite (latest)", "fastest"),
    ModelInfo("gemini-flash-latest", "Gemini Flash (latest)", "balanced"),
    ModelInfo("gemini-pro-latest", "Gemini Pro (latest)", "most capable"),
)

# Keywords Gemini's schema dialect accepts. Anything else — `additionalProperties`,
# `$schema`, `default`, `examples` — makes it reject the whole request with a 400
# that reads like a key problem.
_ALLOWED_SCHEMA_KEYS = frozenset(
    {"type", "description", "enum", "items", "properties", "required", "nullable", "format"}
)


def _headers(api_key: str) -> dict[str, str]:
    return {"x-goog-api-key": api_key, "Content-Type": "application/json"}


def _request(method: str, path: str, api_key: str, body: Any, timeout: int) -> dict[str, Any]:
    import requests

    try:
        response = requests.request(
            method, f"{BASE}{path}", headers=_headers(check_key_shape(api_key)),
            json=body, timeout=timeout,
        )
    except requests.RequestException as exc:
        raise LlmError(str(exc), "Couldn't reach Gemini.", "network") from exc

    if response.status_code >= 400:
        kind, friendly = classify(response.status_code, response.text)
        raise LlmError(
            f"gemini HTTP {response.status_code}: {response.text[:400]}",
            friendly, kind,
        )
    try:
        return response.json()
    except ValueError as exc:
        raise LlmError("gemini sent non-JSON", "Gemini sent a bad reply.") from exc


def _clean_schema(schema: Any) -> Any:
    """Strip JSON Schema keywords Gemini refuses, recursively."""
    if not isinstance(schema, dict):
        return schema
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _ALLOWED_SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {k: _clean_schema(v) for k, v in value.items()}
        elif key == "items":
            cleaned[key] = _clean_schema(value)
        else:
            cleaned[key] = value
    # Gemini rejects an object with no properties at all, which is exactly the
    # shape of every no-argument skill (get_battery, take_screenshot, …).
    if cleaned.get("type") == "object" and not cleaned.get("properties"):
        cleaned["properties"] = {}
    return cleaned


def _to_gemini_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "function_declarations": [
            {
                "name": tool["name"],
                "description": tool.get("description", "")[:1000],
                "parameters": _clean_schema(
                    tool.get("input_schema") or {"type": "object", "properties": {}}
                ),
            }
            for tool in tools
        ]
    }]


def _to_gemini_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`assistant` becomes `model`, and content becomes a parts list."""
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        text = content if isinstance(content, str) else str(content)
        out.append({
            "role": "model" if message.get("role") == "assistant" else "user",
            "parts": [{"text": text}],
        })
    return out


def validate_key(api_key: str) -> None:
    """Prove the key works, without depending on any one model existing."""
    _request("GET", "/models?pageSize=1", api_key, None, VALIDATE_TIMEOUT)


def list_models(api_key: str) -> list[ModelInfo]:
    payload = _request("GET", "/models?pageSize=200", api_key, None, VALIDATE_TIMEOUT)

    found: list[ModelInfo] = []
    for item in payload.get("models") or []:
        # `name` arrives as "models/gemini-2.5-flash".
        model_id = str(item.get("name") or "").removeprefix("models/").strip()
        methods = item.get("supportedGenerationMethods") or []
        if not model_id or "generateContent" not in methods:
            continue
        if any(bad in model_id for bad in _NOT_CHAT):
            continue
        found.append(ModelInfo(
            id=model_id,
            label=str(item.get("displayName") or pretty_model_name(model_id)),
            speed=speed_for(model_id),
        ))
    found.sort(key=lambda m: m.id)
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
    body: dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": _to_gemini_messages(messages),
        "generationConfig": {"maxOutputTokens": max_tokens},
    }
    if tools:
        body["tools"] = _to_gemini_tools(tools)

    payload = _request(
        "POST", f"/models/{model or DEFAULT_MODEL}:generateContent",
        api_key, body, REQUEST_TIMEOUT,
    )

    candidates = payload.get("candidates") or []
    if not candidates:
        # No candidate at all means the prompt itself was blocked.
        blocked = (payload.get("promptFeedback") or {}).get("blockReason")
        return Completion(refused=bool(blocked))

    candidate = candidates[0]
    if candidate.get("finishReason") == "SAFETY":
        return Completion(refused=True)

    calls: list[ToolCall] = []
    text_parts: list[str] = []
    for part in (candidate.get("content") or {}).get("parts") or []:
        if "functionCall" in part:
            call = part["functionCall"]
            name = str(call.get("name") or "")
            if name:
                calls.append(ToolCall(skill=name, args=dict(call.get("args") or {})))
        elif part.get("text"):
            text_parts.append(str(part["text"]))

    return Completion(
        tool_calls=calls,
        text=" ".join(p.strip() for p in text_parts if p.strip()).strip(),
    )


PROVIDER = Provider(
    id="gemini",
    label="Google Gemini",
    help_url="https://aistudio.google.com/apikey",
    default_model=DEFAULT_MODEL,
    validate_key=validate_key,
    list_models=list_models,
    complete=complete,
    known_models=KNOWN_MODELS,
)
