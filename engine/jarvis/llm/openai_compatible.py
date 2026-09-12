"""OpenAI, Groq, DeepSeek and Mistral — one adapter, four base URLs.

All four speak the `/chat/completions` shape, including its function-calling
form, so the only differences are the host, the model list and which of them
bother to implement `tool_choice`. Written against plain HTTP rather than the
`openai` SDK: the engine already depends on `requests`, and each additional SDK
would be another few megabytes inside a PyInstaller bundle that is already
212 MB.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Flavour:
    id: str
    label: str
    base_url: str
    help_url: str
    default_model: str
    known_models: tuple[ModelInfo, ...] = ()
    #: Substrings that mark a model as not worth offering for chat — embeddings,
    #: speech, moderation and image models all appear in `/models`.
    exclude: tuple[str, ...] = (
        "embed", "whisper", "tts", "dall-e", "moderation", "audio",
        "image", "rerank", "guard", "vision-preview", "davinci", "babbage",
    )


def _post(flavour: Flavour, api_key: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    import requests

    try:
        response = requests.post(
            f"{flavour.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise LlmError(str(exc), f"Couldn't reach {flavour.label}.", "network") from exc

    if response.status_code >= 400:
        kind, friendly = classify(response.status_code, response.text)
        raise LlmError(
            f"{flavour.id} HTTP {response.status_code}: {response.text[:400]}",
            friendly, kind,
        )
    try:
        return response.json()
    except ValueError as exc:
        raise LlmError(f"{flavour.id} sent non-JSON", f"{flavour.label} sent a bad reply.") from exc


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic tool schemas → OpenAI function schemas.

    The registry emits one shape (Anthropic's, since that was the only provider
    when it was written) and every adapter translates from it. Keeping the
    registry as the single source of truth matters more than any adapter's
    convenience — a skill must not have to know which model will be asked to
    call it.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for tool in tools
    ]


def _parse_arguments(raw: Any, name: str) -> dict[str, Any]:
    """Tool arguments arrive as a JSON *string*, and sometimes a broken one.

    A smaller model will occasionally emit trailing prose or an empty string.
    Dropping the whole call for that would lose a command the user did give, so
    a bad payload becomes an argument-less call — which the skill's own
    defaults, or the registry's TypeError handler, will deal with sensibly.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        log.warning("Tool %s sent unparseable arguments: %r", name, str(raw)[:120])
        return {}
    return parsed if isinstance(parsed, dict) else {}


def make_provider(flavour: Flavour) -> Provider:
    def validate_key(api_key: str) -> None:
        """Prove the key works, without depending on any one model existing.

        This used to send a one-token completion against the *default* model,
        which conflated two failures: a bad key and a default that the provider
        has since retired. Listing models proves the credential and nothing
        else, so a stale default can be corrected rather than blamed on the key.
        """
        list_models(api_key)

    def list_models(api_key: str) -> list[ModelInfo]:
        import requests

        key = check_key_shape(api_key)
        try:
            response = requests.get(
                f"{flavour.base_url}/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=VALIDATE_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise LlmError(str(exc), f"Couldn't reach {flavour.label}.", "network") from exc

        if response.status_code >= 400:
            kind, friendly = classify(response.status_code, response.text)
            raise LlmError(
                f"{flavour.id} HTTP {response.status_code}: {response.text[:300]}",
                friendly, kind,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise LlmError(f"{flavour.id} sent non-JSON",
                           f"{flavour.label} sent a bad reply.") from exc

        found: list[ModelInfo] = []
        for item in payload.get("data") or []:
            model_id = str(item.get("id") or "").strip()
            if not model_id or any(bad in model_id.lower() for bad in flavour.exclude):
                continue
            found.append(
                ModelInfo(id=model_id, label=pretty_model_name(model_id),
                          speed=speed_for(model_id))
            )
        found.sort(key=lambda m: m.id)
        return found or list(flavour.known_models)

    def complete(
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        api_key: str,
        model: str,
        max_tokens: int = 2048,
        effort: str = "low",
        **_: Any,
    ) -> Completion:
        """One turn against an OpenAI-shaped chat completions API.

        `effort` is accepted and ignored. It is a real setting on Anthropic and
        Gemini, and `Brain.interpret` passes it to whichever provider is
        selected — so omitting it here did not mean "this provider has no
        effort setting", it meant `TypeError: complete() got an unexpected
        keyword argument 'effort'` on every single request. The broad handler
        in `Brain` caught it and the user heard "I couldn't reach my brain just
        now", which made all four OpenAI-shaped providers look like a network
        problem. A flavour whose models support a reasoning effort can start
        honouring this; the rest ignore it deliberately.

        `**_` is the same guard made general: this signature is called by name
        from one place that cannot see it, so a parameter added there must
        never again take four providers offline.
        """
        body: dict[str, Any] = {
            "model": model or flavour.default_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if tools:
            body["tools"] = _to_openai_tools(tools)
            body["tool_choice"] = "auto"

        payload = _post(flavour, check_key_shape(api_key), body, REQUEST_TIMEOUT)
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        calls = [
            ToolCall(
                skill=str((call.get("function") or {}).get("name") or ""),
                args=_parse_arguments(
                    (call.get("function") or {}).get("arguments"),
                    str((call.get("function") or {}).get("name") or "?"),
                ),
            )
            for call in message.get("tool_calls") or []
        ]
        return Completion(
            tool_calls=[c for c in calls if c.skill],
            text=(message.get("content") or "").strip(),
            # `content_filter` is the OpenAI-compatible spelling of a refusal.
            refused=choice.get("finish_reason") == "content_filter",
        )

    return Provider(
        id=flavour.id,
        label=flavour.label,
        help_url=flavour.help_url,
        default_model=flavour.default_model,
        validate_key=validate_key,
        list_models=list_models,
        complete=complete,
        known_models=flavour.known_models,
    )


FLAVOURS: tuple[Flavour, ...] = (
    Flavour(
        id="openai",
        label="OpenAI",
        base_url="https://api.openai.com/v1",
        help_url="https://platform.openai.com/api-keys",
        default_model="gpt-4o-mini",
        known_models=(
            ModelInfo("gpt-4o-mini", "GPT-4o mini", "fastest"),
            ModelInfo("gpt-4o", "GPT-4o", "balanced"),
        ),
    ),
    Flavour(
        id="groq",
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        help_url="https://console.groq.com/keys",
        # The reason Groq is worth having in a voice assistant: replies land in
        # a couple of hundred milliseconds, which is inside the window where a
        # spoken answer still feels like a conversation.
        default_model="llama-3.3-70b-versatile",
        known_models=(
            ModelInfo("llama-3.3-70b-versatile", "Llama 3.3 70B", "fastest"),
            ModelInfo("llama-3.1-8b-instant", "Llama 3.1 8B", "fastest"),
        ),
    ),
    Flavour(
        id="deepseek",
        label="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        help_url="https://platform.deepseek.com/api_keys",
        default_model="deepseek-chat",
        known_models=(
            ModelInfo("deepseek-chat", "DeepSeek Chat", "balanced"),
            ModelInfo("deepseek-reasoner", "DeepSeek Reasoner", "most capable"),
        ),
    ),
    Flavour(
        id="mistral",
        label="Mistral",
        base_url="https://api.mistral.ai/v1",
        help_url="https://console.mistral.ai/api-keys",
        default_model="mistral-large-latest",
        known_models=(
            ModelInfo("mistral-small-latest", "Mistral Small", "fastest"),
            ModelInfo("mistral-large-latest", "Mistral Large", "most capable"),
        ),
    ),
)
