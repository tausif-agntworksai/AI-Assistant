"""The language model — understands anything the local rules didn't.

Two jobs in one call: pick a skill (via tool use) when the user wants an
action, or just answer when they want a conversation. The skill registry
generates the tool schemas, so a new skill is reachable by the model the moment
it's written.

Which model is a user setting, not a constant. Everything provider-specific
lives in `jarvis.llm`; this module owns the prompt, the conversation history and
the translation between a `Completion` and a `BrainResult`. That split is what
lets someone paste a Groq key and have every one of the 72 skills work through
it without a line changing here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .. import llm

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Jarvis, a voice assistant running on the user's computer.

You are spoken to and you answer out loud, so your replies are read by a
text-to-speech voice. Keep them short — one or two sentences for anything
conversational, and a brief confirmation for actions. No markdown, no bullet
points, no code blocks, no emoji: none of it survives being spoken.

The user speaks Hindi, English, and a mix of the two. Answer in the language
they used — English question, English answer; Hindi question, Hindi answer.
Never switch languages on your own, and never answer in both. A tag at the top
of each message tells you which one was detected; that tag is the decision, so
follow it even when the words in front of you look like the other language
(romanised Hindi reads like English, and that is exactly the case the tag is
there to settle).

When replying in Hindi, write in Devanagari script — the Hindi voice
pronounces Devanagari correctly and mangles romanised Hindi. Keep proper nouns
(app names, brands, numbers) in Latin script.

When the user wants something done on the computer, call the matching tool.
You can call several tools in one turn if they asked for several things.
When they're asking a question, making conversation, or want something
translated or explained, just answer — don't reach for a tool.

If a request is ambiguous about which app or file it means, ask a short
clarifying question rather than guessing.

Some tools are gated: the user is asked to confirm before they run. That
happens automatically — call the tool as normal, don't ask for permission
yourself."""


@dataclass
class Action:
    skill: str
    args: dict[str, Any] = field(default_factory=dict)


def _describe(actions: list[Action]) -> str:
    """A sentence standing in for what a tool-use turn did.

    The model is told what it just did in the same form it would have said it,
    so the next turn can refer back to it. "open_app(app=chrome)" would work
    as well for a machine and read as noise to the model, which has to
    continue a conversation from it.
    """
    done = []
    for action in actions:
        target = next((str(v) for v in action.args.values() if v), "")
        verb = action.skill.replace("_", " ")
        done.append(f"{verb} {target}".strip())
    return "Done: " + ", ".join(done) + "."


@dataclass
class BrainResult:
    actions: list[Action] = field(default_factory=list)
    speech: str = ""
    language: str = "en"
    latency: float = 0.0
    refused: bool = False
    error: str = ""
    #: Set when the failure is "nobody has given me a key yet", which the
    #: orchestrator answers by opening settings rather than by apologising.
    needs_key: bool = False
    #: Which kind of failure, from `llm.base.ErrorKind`. The orchestrator needs
    #: to tell "your wi-fi is down" from "your key is wrong" without reading
    #: the English sentence in `error`, and a connection failure is also the
    #: best evidence available about connectivity.
    error_kind: str = ""

    @property
    def has_actions(self) -> bool:
        return bool(self.actions)

    @property
    def is_empty(self) -> bool:
        return not self.actions and not self.speech.strip()


class Brain:
    def __init__(self, cfg=None) -> None:
        if cfg is None:
            from ..config import settings

            cfg = settings.brain
        self.cfg = cfg
        self._history: list[dict[str, Any]] = []
        # Seed the runtime selection from config.yaml. The desktop app
        # overwrites this the moment it pushes the user's own choice down.
        try:
            llm.selection.configure(cfg.provider, cfg.model)
        except llm.LlmError as exc:
            log.error("brain.provider in config.yaml is not a known provider: %s", exc)

    @property
    def available(self) -> bool:
        """True when there is a key to use — the user's, or one from `.env`."""
        return bool(self.cfg.enabled) and llm.selection.available

    @property
    def describe(self) -> str:
        snapshot = llm.selection.snapshot()
        return f"{snapshot['label']} ({snapshot['model']})"

    # -- prompt assembly ----------------------------------------------------

    def _system_prompt(self) -> str:
        """The stable prompt. Marked for caching by whichever provider supports it.

        Nothing volatile goes in here — a timestamp or battery level would
        change the prefix on every request and defeat the cache. Live context
        rides in the user turn instead.
        """
        from ..app_index import app_index

        names = [e.name for e in app_index.entries[:120]]
        installed = ", ".join(names) if names else "(app index not built yet)"
        return f"{SYSTEM_PROMPT}\n\nApps available on this machine: {installed}"

    @staticmethod
    def _language_note(language: str) -> str:
        """The one instruction that must not be left to inference.

        The model is looking at romanised Hindi half the time, which is
        indistinguishable from English at a glance. The engine has already
        decided — from the script, the vocabulary and Whisper's own label —
        so it states the answer rather than hoping.
        """
        if (language or "").lower().startswith("hi"):
            return "[reply in Hindi, Devanagari script]\n"
        return "[reply in English]\n"

    @staticmethod
    def _context_note(ctx: dict[str, Any] | None) -> str:
        if not ctx:
            return ""
        parts = []
        if ctx.get("time"):
            parts.append(f"time is {ctx['time']}")
        if ctx.get("foreground"):
            parts.append(f"the focused window is \"{ctx['foreground']}\"")
        if ctx.get("battery") is not None:
            parts.append(f"battery is at {ctx['battery']}%")
        if ctx.get("recent"):
            # What was just acted on, so "close it" and "tell her" have a
            # referent. Volatile by nature, which is why it belongs here
            # rather than in the cached system prompt.
            parts.append(f"just now: {ctx['recent']}")
        if not parts:
            return ""
        return "[context: " + ", ".join(parts) + "]\n"

    # -- the main entry point ----------------------------------------------

    def interpret(
        self,
        text: str,
        language: str = "en",
        context: dict[str, Any] | None = None,
    ) -> BrainResult:
        """Route an utterance: pick tools, answer, or both."""
        if not self.available:
            return BrainResult(error="no API key configured", needs_key=True)

        from ..skills.registry import registry

        provider = llm.selection.provider
        user_turn = self._language_note(language) + self._context_note(context) + text
        messages = [*self._history, {"role": "user", "content": user_turn}]

        t0 = time.perf_counter()
        try:
            completion = provider.complete(
                system=self._system_prompt(),
                messages=messages,
                tools=registry.tool_schemas(),
                api_key=llm.selection.api_key,
                model=llm.selection.active_model,
                max_tokens=self.cfg.max_tokens,
                effort=self.cfg.effort,
            )
        except llm.LlmError as exc:
            log.error("%s request failed: %s", provider.id, exc)
            return BrainResult(error=exc.friendly, error_kind=exc.kind,
                               latency=time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - the assistant must stay up
            log.exception("%s request failed unexpectedly", provider.id)
            from .. import net

            return BrainResult(
                error=f"{type(exc).__name__}: {exc}",
                error_kind="network" if net.looks_like_connectivity(exc) else "other",
                latency=time.perf_counter() - t0,
            )

        latency = time.perf_counter() - t0

        if completion.refused:
            return BrainResult(
                speech="I can't help with that one.",
                language=language, latency=latency, refused=True,
            )

        actions = [Action(skill=call.skill, args=call.args) for call in completion.tool_calls]
        speech = completion.text

        # Tool-use turns are remembered as a plain sentence rather than as
        # tool_use blocks, which would need their tool_result counterparts
        # echoed back to stay valid. Dropping them entirely was worse: the
        # model would open Chrome and, one sentence later, have no idea what
        # "it" referred to, because the only turn that mattered was the one
        # turn never written down.
        if actions:
            self._remember(user_turn, speech or _describe(actions))
        elif speech:
            self._remember(user_turn, speech)

        log.info("Brain %.2fs via %s -> %d action(s)%s", latency, provider.id,
                 len(actions), f", speech: {speech[:60]!r}" if speech else "")
        return BrainResult(actions=actions, speech=speech, language=language,
                           latency=latency)

    def ask(self, question: str, language: str = "en") -> str:
        """Plain question answering, with no tools offered."""
        if not self.available:
            return ""

        provider = llm.selection.provider
        asked = self._language_note(language) + question
        try:
            completion = provider.complete(
                system=self._system_prompt(),
                messages=[*self._history, {"role": "user", "content": asked}],
                tools=[],
                api_key=llm.selection.api_key,
                model=llm.selection.active_model,
                max_tokens=self.cfg.max_tokens,
                effort=self.cfg.effort,
            )
        except llm.LlmError as exc:
            log.error("%s request failed: %s", provider.id, exc)
            return ""
        except Exception:  # noqa: BLE001
            log.exception("%s request failed unexpectedly", provider.id)
            return ""

        if completion.refused:
            return "I can't help with that one."
        if completion.text:
            self._remember(asked, completion.text)
        return completion.text

    # -- conversation memory ------------------------------------------------

    def _remember(self, user_text: str, reply: str) -> None:
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": reply})
        limit = max(2, self.cfg.history_turns * 2)
        if len(self._history) > limit:
            self._history = self._history[-limit:]

    def reset_history(self) -> None:
        self._history.clear()


brain = Brain()
