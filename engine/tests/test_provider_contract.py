# -*- coding: utf-8 -*-
"""Every provider must accept the call the Brain actually makes.

This exists because of a bug that took four of the six providers offline
without anyone noticing. `Brain.interpret` passes `effort=` to whichever
provider is selected; `openai_compatible.complete` did not declare it, so
OpenAI, Groq, DeepSeek and Mistral raised

    TypeError: complete() got an unexpected keyword argument 'effort'

on every single request. `Brain` catches broad exceptions so the assistant
stays up, so the user heard "I couldn't reach my brain just now" — which reads
as a network problem and sent anyone debugging it to the wrong place entirely.

Key validation never caught it because the probe in `llm/__init__.py` calls
`complete` with a different, shorter argument list than the Brain does.
"""

import inspect

import pytest

from jarvis import llm

#: Exactly what `Brain.interpret` passes. Keep this in step with `nlu/llm.py`;
#: that is the whole point of the file.
BRAIN_CALL = dict(
    system="you are a test",
    messages=[{"role": "user", "content": "hello"}],
    tools=[],
    api_key="test-key",
    model="test-model",
    max_tokens=2048,
    effort="low",
)


def _every_provider():
    from jarvis.llm import anthropic_provider, gemini_provider
    from jarvis.llm.openai_compatible import FLAVOURS, make_provider

    flavours = list(FLAVOURS) if not isinstance(FLAVOURS, dict) else list(FLAVOURS.values())
    providers = [(make_provider(f).id, make_provider(f).complete) for f in flavours]
    providers.append(("anthropic", anthropic_provider.complete))
    providers.append(("gemini", gemini_provider.complete))
    return providers


@pytest.mark.parametrize("name, complete", _every_provider(),
                         ids=lambda v: v if isinstance(v, str) else "")
def test_complete_accepts_the_brains_arguments(name, complete):
    """A signature mismatch here is invisible until someone switches provider."""
    try:
        inspect.signature(complete).bind(**BRAIN_CALL)
    except TypeError as exc:
        pytest.fail(f"{name}.complete cannot accept the Brain's call: {exc}")


def test_the_registry_lists_every_provider_we_checked():
    """A new provider must not slip past this file by not being enumerated."""
    listed = {p["id"] for p in llm.provider_list()}
    checked = {name for name, _ in _every_provider()}
    assert listed == checked, f"unchecked providers: {listed - checked}"


def test_the_brain_call_matches_what_the_brain_sends():
    """If `interpret` grows an argument, this test is the one that fails."""
    from pathlib import Path

    source = Path(llm.__file__).parent.parent / "nlu" / "llm.py"
    text = source.read_text(encoding="utf-8")
    start = text.index("completion = provider.complete(")
    call = text[start:text.index(")", start)]
    passed = {line.split("=")[0].strip() for line in call.splitlines()[1:] if "=" in line}
    missing = passed - set(BRAIN_CALL)
    assert not missing, f"Brain now passes {missing}; add it to BRAIN_CALL and to every provider"
