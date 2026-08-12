"""Claude fallback — understands anything the local rules didn't.

Two jobs in one call: pick a skill (via tool use) when the user wants an
action, or just answer when they want a conversation. The skill registry
generates the tool schemas, so a new skill is reachable by Claude the moment
it's written.

Thinking is deliberately left ON. Disabling it on this model can make a tool
call arrive as plain text in the visible response — the turn succeeds, the
call never runs, and nothing errors. That failure would be invisible in a
voice assistant, so latency is managed with a low effort level instead.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Jarvis, a voice assistant running on the user's Windows laptop.

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


@dataclass
class BrainResult:
    actions: list[Action] = field(default_factory=list)
    speech: str = ""
    language: str = "en"
    latency: float = 0.0
    refused: bool = False
    error: str = ""

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
        self._client = None
        self._history: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return self.cfg.available

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        import anthropic

        self._client = anthropic.Anthropic(api_key=self.cfg.api_key)
        log.info("Claude brain ready (%s, effort=%s)", self.cfg.model, self.cfg.effort)
        return self._client

    # -- prompt assembly ----------------------------------------------------

    def _system_blocks(self) -> list[dict[str, Any]]:
        """Stable system prompt, marked for prompt caching.

        Nothing volatile goes in here — a timestamp or battery level would
        change the prefix on every request and defeat the cache. Live context
        rides in the user turn instead.
        """
        from ..app_index import app_index

        names = [e.name for e in app_index.entries[:120]]
        installed = ", ".join(names) if names else "(app index not built yet)"

        return [{
            "type": "text",
            "text": f"{SYSTEM_PROMPT}\n\nApps available on this machine: {installed}",
            "cache_control": {"type": "ephemeral"},
        }]

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
            return BrainResult(error="no API key configured")

        from ..skills.registry import registry

        client = self._ensure_client()
        tools = registry.tool_schemas()

        user_turn = self._language_note(language) + self._context_note(context) + text
        messages = [*self._history, {"role": "user", "content": user_turn}]

        t0 = time.perf_counter()
        try:
            response = client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                system=self._system_blocks(),
                messages=messages,
                tools=tools,
                thinking={"type": "adaptive"},
                output_config={"effort": self.cfg.effort},
            )
        except Exception as exc:  # noqa: BLE001 - the assistant must stay up
            log.error("Claude request failed: %s: %s", type(exc).__name__, exc)
            return BrainResult(error=f"{type(exc).__name__}: {exc}",
                               latency=time.perf_counter() - t0)

        latency = time.perf_counter() - t0

        # Safety classifiers can decline a request; the HTTP call still
        # succeeds, so this has to be checked before reading content.
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            log.warning("Claude declined the request (category=%s)", category)
            return BrainResult(
                speech="I can't help with that one.",
                language=language, latency=latency, refused=True,
            )

        actions: list[Action] = []
        speech_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                speech_parts.append(block.text)
            elif block.type == "tool_use":
                actions.append(Action(skill=block.name, args=dict(block.input or {})))

        speech = " ".join(p.strip() for p in speech_parts if p.strip()).strip()

        # Only conversational turns are worth remembering. Tool-use turns would
        # need their tool_result blocks echoed back to stay valid, and the
        # assistant speaks the skill's own reply rather than Claude's.
        if not actions and speech:
            self._remember(user_turn, speech)

        result = BrainResult(actions=actions, speech=speech, language=language,
                             latency=latency)
        log.info("Brain %.2fs -> %d action(s)%s", latency, len(actions),
                 f", speech: {speech[:60]!r}" if speech else "")
        return result

    def ask(self, question: str, language: str = "en") -> str:
        """Plain question answering, with no tools offered."""
        if not self.available:
            return ""

        client = self._ensure_client()
        asked = self._language_note(language) + question
        try:
            response = client.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                system=self._system_blocks(),
                messages=[*self._history, {"role": "user", "content": asked}],
                thinking={"type": "adaptive"},
                output_config={"effort": self.cfg.effort},
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Claude request failed: %s", exc)
            return ""

        if response.stop_reason == "refusal":
            return "I can't help with that one."

        answer = " ".join(b.text for b in response.content if b.type == "text").strip()
        if answer:
            self._remember(asked, answer)
        return answer

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
