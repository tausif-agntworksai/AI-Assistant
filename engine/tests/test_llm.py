# -*- coding: utf-8 -*-
"""The provider registry, and the traps it is built to avoid.

Six providers behind one interface, so the interesting tests are not "does
Anthropic work" — that needs a live key — but the seams: does an unknown
provider get rejected, does a Groq key ever reach Google, does a bad model name
get reported as a bad key, does a key ever appear in something we log.

Several of these guard against patterns confirmed in the sibling AI Calculator.
Copying its design was the plan; copying its bugs was not, and a comment saying
"don't do that" is weaker than a test.
"""

import json

import pytest

from jarvis import llm
from jarvis.llm import base
from jarvis.llm.base import LlmError, ModelInfo


# --- the registry ----------------------------------------------------------


def test_all_six_providers_are_registered():
    assert set(llm.PROVIDERS) == {
        "anthropic", "openai", "gemini", "groq", "deepseek", "mistral"
    }


def test_every_provider_is_fully_wired():
    """A half-declared provider would fail at the moment someone selected it."""
    for provider in llm.PROVIDERS.values():
        assert provider.label and provider.help_url, provider.id
        assert provider.default_model, provider.id
        assert callable(provider.validate_key) and callable(provider.list_models)
        assert callable(provider.complete)
        # Something to show before a key is entered, so the model dropdown is
        # never empty while the real list loads.
        assert provider.known_models, provider.id


def test_an_unknown_provider_is_rejected_rather_than_silently_swapped():
    """The AI Calculator falls back to Gemini here — *carrying the key it was
    given*. A stale id in storage therefore hands an OpenAI key to Google and
    reports the result as "that key was rejected"."""
    with pytest.raises(LlmError) as caught:
        llm.get_provider("gpt5-turbo-ultra")
    assert "don't know that AI provider" in caught.value.friendly


def test_provider_metadata_never_carries_a_callable_to_the_ui():
    for entry in llm.provider_list():
        assert set(entry) == {"id", "label", "help_url", "default_model", "known_models"}


# --- the selection ---------------------------------------------------------


@pytest.fixture
def selection():
    """Restores the process-wide selection after a test changes it."""
    before = (llm.selection.provider_id, llm.selection.model, llm.selection._api_key)
    yield llm.selection
    llm.selection.provider_id, llm.selection.model, llm.selection._api_key = before


def test_a_key_is_never_included_in_what_the_ui_is_told(selection):
    selection.configure("groq", "llama-3.3-70b-versatile", "gsk_pretend_this_is_real_key")
    snapshot = selection.snapshot()
    assert snapshot["has_key"] is True
    assert "gsk_pretend_this_is_real_key" not in json.dumps(snapshot)
    assert snapshot["provider"] == "groq"
    assert snapshot["model"] == "llama-3.3-70b-versatile"


def test_an_empty_model_means_the_providers_own_default(selection):
    selection.configure("mistral", "", "key-here")
    assert selection.active_model == "mistral-large-latest"


def test_switching_provider_switches_the_model_too(selection):
    """Otherwise a Claude model id would be sent to Groq and 404."""
    selection.configure("anthropic", "claude-opus-5", "k")
    selection.configure("groq", "", "k")
    assert selection.active_model == "llama-3.3-70b-versatile"


def test_clearing_the_key_makes_the_brain_unavailable(selection, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    selection.configure("anthropic", "", "a-key")
    assert selection.available
    selection.clear_key()
    assert not selection.available


def test_a_key_in_the_environment_still_works(selection, monkeypatch):
    """The pre-BYOK path: a checkout with ANTHROPIC_API_KEY in .env must keep
    behaving exactly as it did, or every existing setup breaks on upgrade."""
    selection.configure("anthropic", "", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-dotenv")
    assert selection.available
    assert selection.snapshot()["from_environment"] is True


def test_an_environment_key_is_not_read_across_providers(selection, monkeypatch):
    """An Anthropic key in .env must not be presented to Groq."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-something")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    selection.configure("groq", "", "")
    assert not selection.available


def test_configuring_an_unknown_provider_leaves_the_old_one_alone(selection):
    selection.configure("anthropic", "claude-opus-5", "k")
    with pytest.raises(LlmError):
        selection.configure("not-a-provider", "", "k")
    assert selection.provider_id == "anthropic"


# --- key hygiene -----------------------------------------------------------


@pytest.mark.parametrize("bad,expected", [
    ("", "Paste an API key first."),
    ("   ", "Paste an API key first."),
    ("x" * 600, "That doesn't look like an API key."),
])
def test_obviously_bad_keys_are_refused_without_a_network_call(bad, expected):
    with pytest.raises(LlmError) as caught:
        base.check_key_shape(bad)
    assert caught.value.friendly == expected


def test_a_pasted_key_with_a_line_break_is_caught_early():
    """Copying from a web page picks up a newline surprisingly often, and the
    provider's own error for it is unhelpful."""
    with pytest.raises(LlmError) as caught:
        base.check_key_shape("sk-ant-abc\ndef")
    assert "space or line break" in caught.value.friendly


# --- error mapping ---------------------------------------------------------


def test_a_bad_model_name_is_not_reported_as_a_bad_key():
    """The Calculator maps HTTP 400 to "that key looks malformed". Right for
    Gemini, wrong for every OpenAI-compatible provider — Groq, DeepSeek and
    Mistral all answer 400/404 for an unknown *model*, which sends the user off
    to re-copy a key that was fine."""
    message = base.friendly_key_error(
        400, '{"error":{"message":"The model `llama-9` does not exist"}}'
    )
    # It has to say two things: the key is fine, and the model is the problem.
    assert "valid" in message and "no access to this model" in message
    assert "rejected" not in message and "malformed" not in message


@pytest.mark.parametrize("status,body,fragment", [
    (401, "unauthorized", "rejected"),
    (403, "", "rejected"),
    (429, "rate limit exceeded", "rate-limited"),
    (429, "insufficient_quota", "out of quota"),
    (404, "not found", "no access"),
    (503, "overloaded", "having trouble"),
])
def test_provider_failures_map_to_something_actionable(status, body, fragment):
    assert fragment in base.friendly_key_error(status, body)


def test_a_rejected_key_is_never_echoed_back_in_the_error():
    """Providers quote the rejected key in the message saying it was rejected,
    and that message reaches the HUD and the log people attach to bug reports."""
    leaked = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz012345"
    message = base.friendly_key_error(500, f"Invalid key: {leaked}")
    assert leaked not in message


# --- model metadata --------------------------------------------------------


@pytest.mark.parametrize("model_id,speed", [
    ("claude-haiku-4-5-20251001", "fastest"),
    ("gpt-4o-mini", "fastest"),
    ("llama-3.1-8b-instant", "fastest"),
    ("gemini-2.5-flash", "fastest"),
    ("claude-sonnet-5", "balanced"),
    ("claude-opus-5", "most capable"),
    ("deepseek-reasoner", "most capable"),
])
def test_models_carry_a_latency_hint(model_id, speed):
    """On a voice assistant the difference between 200 ms and 6 s is the
    difference between a conversation and a progress bar, so the settings
    screen has to say which is which."""
    assert base.speed_for(model_id) == speed


def test_an_unknown_model_renders_without_a_badge_rather_than_vanishing():
    assert base.speed_for("some-future-model-2031") == ""
    assert ModelInfo("x", "X").speed == ""


# --- tool schema translation ----------------------------------------------


def _anthropic_tool():
    return {
        "name": "set_volume",
        "description": "Set the system volume",
        "input_schema": {
            "type": "object",
            "properties": {"level": {"type": "integer", "description": "0-100"}},
            "required": ["level"],
        },
    }


def test_tool_schemas_survive_translation_to_openai_shape():
    from jarvis.llm.openai_compatible import _to_openai_tools

    converted = _to_openai_tools([_anthropic_tool()])[0]
    assert converted["type"] == "function"
    assert converted["function"]["name"] == "set_volume"
    assert converted["function"]["parameters"]["required"] == ["level"]


def test_gemini_gets_a_schema_it_will_actually_accept():
    """Gemini rejects JSON Schema keywords it doesn't know, and answers with a
    400 that reads exactly like a key problem."""
    from jarvis.llm.gemini_provider import _clean_schema, _to_gemini_tools

    tool = _anthropic_tool()
    tool["input_schema"]["additionalProperties"] = False
    tool["input_schema"]["properties"]["level"]["default"] = 50

    declaration = _to_gemini_tools([tool])[0]["function_declarations"][0]
    params = declaration["parameters"]
    assert "additionalProperties" not in params
    assert "default" not in params["properties"]["level"]
    assert params["required"] == ["level"]

    # A no-argument skill still needs a properties key, or Gemini 400s.
    assert _clean_schema({"type": "object", "properties": {}})["properties"] == {}


def test_a_no_argument_skill_translates_for_every_provider():
    from jarvis.llm.gemini_provider import _to_gemini_tools
    from jarvis.llm.openai_compatible import _to_openai_tools

    bare = {"name": "get_battery", "description": "Report the battery",
            "input_schema": {"type": "object", "properties": {}, "required": []}}
    assert _to_openai_tools([bare])[0]["function"]["name"] == "get_battery"
    assert _to_gemini_tools([bare])[0]["function_declarations"][0]["parameters"][
        "properties"] == {}


def test_malformed_tool_arguments_become_an_argumentless_call():
    """Smaller models emit trailing prose after the JSON. Dropping the call
    would lose a command the user actually gave; the skill's own defaults and
    the registry's TypeError handler cope with missing arguments."""
    from jarvis.llm.openai_compatible import _parse_arguments

    assert _parse_arguments('{"level": 50}', "set_volume") == {"level": 50}
    assert _parse_arguments("", "set_volume") == {}
    assert _parse_arguments("{oh dear", "set_volume") == {}
    assert _parse_arguments("[1,2,3]", "set_volume") == {}
    assert _parse_arguments({"level": 20}, "set_volume") == {"level": 20}


def test_gemini_message_roles_are_translated():
    """Gemini says `model`, everyone else says `assistant`."""
    from jarvis.llm.gemini_provider import _to_gemini_messages

    converted = _to_gemini_messages([
        {"role": "user", "content": "chrome kholo"},
        {"role": "assistant", "content": "Chrome khol raha hoon."},
    ])
    assert [m["role"] for m in converted] == ["user", "model"]
    assert converted[0]["parts"] == [{"text": "chrome kholo"}]
