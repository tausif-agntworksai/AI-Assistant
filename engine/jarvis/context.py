"""What we were just talking about.

Every turn until now has been an island. `Brain._history` keeps the last few
*conversational* exchanges, but `nlu/llm.py` deliberately drops any turn that
ran a skill — so the assistant could open Chrome and, one sentence later, have
no idea what "it" meant. Rule-routed commands, which are most of them, left no
trace at all.

The missing piece was never data collection. Skills already report what they
resolved — `open_app` hands back `app=Google Chrome`, `send_message` hands back
the contact it matched — and `registry.execute` already funnels every one of
them through a single point. That structured data was being dropped on the
floor one line after it arrived. This module is where it lands instead.

**Short-lived on purpose.** A referent is a half-finished sentence, not a
setting: "tell her I'll be late" means nothing an hour later, and a stale
answer is worse than no answer because the user cannot see that it is stale.
So entries expire, and nothing here is written to disk.

**Only people are substituted into the text.** Pronouns for *things* are a
trap: "turn it up" is a volume command that exists today, and rewriting "it"
to the last app would break it. Person pronouns are safe because no existing
rule matches on them. Things are still remembered — the model receives them as
context, and a skill can ask — they are simply never spliced into an utterance
before the rules see it.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

#: How long a referent stays usable. Long enough to finish a thought, short
#: enough that a command after lunch never picks up this morning's contact.
DEFAULT_TTL_SEC = 300.0

#: Which key in a skill's returned data names which kind of thing. Skills
#: already emit these; nothing had been reading them.
_KINDS: dict[str, str] = {
    "contact": "contact",
    "app": "app",
}

# Pronouns that can only mean a person, so substituting them can never
# collide with an existing command. Hindi oblique forms included because
# "usko bol do ki ..." is the ordinary way to say it.
_PERSON_PRONOUN = re.compile(
    r"\b(her|him|them|usko|unko|unhe|unhein|usse|uske|unke|unka|unki)\b",
    re.IGNORECASE,
)

#: The case particle each Hindi pronoun carries, which the name has to keep.
_PARTICLES: dict[str, str] = {
    "usko": "ko", "unko": "ko", "unhe": "ko", "unhein": "ko",
    "usse": "se", "uske": "ke", "unke": "ke", "unka": "ka", "unki": "ki",
}

#: "close it" and its Hindi equivalents — the only place a thing-pronoun
#: is unambiguous enough to substitute.
_CLOSE_THAT = re.compile(
    r"^(?P<verb>close|quit|exit|shut|band\s+karo|band\s+kar\s+do)"
    r"\s+(?:it|this|that|ise|isko|usko)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Referent:
    """One thing the assistant resolved, and when."""

    kind: str
    name: str
    at: float

    def fresh(self, ttl: float = DEFAULT_TTL_SEC) -> bool:
        return (time.monotonic() - self.at) <= ttl


class TurnMemory:
    """The last thing of each kind that the assistant actually acted on.

    One slot per kind rather than a history: "her" means the last person, not
    a person from three turns ago, and a list would only invite guessing.
    """

    def __init__(self, ttl: float = DEFAULT_TTL_SEC) -> None:
        self.ttl = ttl
        self._slots: dict[str, Referent] = {}
        self._lock = threading.Lock()

    def record(self, skill: str, data: dict[str, Any] | None,
               ok: bool = True) -> None:
        """Note what a skill resolved. Failures teach nothing, so they are skipped.

        Reads the data the skill already returns, so adding a new skill to this
        costs a keyword argument on its `ok(...)` rather than a change here.
        """
        if not ok or not data:
            return
        now = time.monotonic()
        with self._lock:
            for key, kind in _KINDS.items():
                value = str(data.get(key) or "").strip()
                if value:
                    self._slots[kind] = Referent(kind=kind, name=value, at=now)
                    log.debug("Context: %s is now %r (via %s)", kind, value, skill)

    def recall(self, kind: str) -> Referent | None:
        """The last referent of this kind, if it hasn't gone stale."""
        with self._lock:
            found = self._slots.get(kind)
        if found is None or not found.fresh(self.ttl):
            return None
        return found

    def describe(self) -> str:
        """A one-line summary for the model's volatile context block.

        Deliberately prose rather than JSON: it is appended to a note the model
        already receives in English, and a second serialisation format there
        would buy nothing.
        """
        parts = [
            f"{kind} {found.name}"
            for kind, found in sorted(self._slots.items())
            if found.fresh(self.ttl)
        ]
        return "; ".join(parts)

    def snapshot(self) -> dict[str, str]:
        """Every fresh referent as {kind: name}, for `SkillContext.recent`."""
        return {
            kind: found.name
            for kind, found in self._slots.items()
            if found.fresh(self.ttl)
        }

    def clear(self) -> None:
        with self._lock:
            self._slots.clear()


def resolve_pronouns(text: str, memory: TurnMemory) -> str:
    """Replace a pronoun with whoever it refers to, or leave the text alone.

    Runs on the raw utterance, before normalisation: `nlu/normalize.py` strips
    fillers and `nlu/rules.py` discards pronouns as stopwords, so by the time
    the matcher sees the words the referent is already gone.

    Leaving the text untouched is always an acceptable answer. With no
    referent, "tell her I'll be late" goes to the model exactly as it does
    today — which is a worse answer than substituting, and a much better one
    than substituting the wrong person.
    """
    if not text or not _PERSON_PRONOUN.search(text):
        return text

    who = memory.recall("contact")
    if who is None:
        return text
    # Already named in this sentence — "tell Sana that her order arrived"
    # needs no help and would read oddly if we rewrote the second half.
    if who.name.lower() in text.lower():
        return text

    def substitute(match: re.Match[str]) -> str:
        # Hindi pronouns carry the case particle inside the word: "usko" is
        # "us" + "ko". Replacing the whole token with a name drops the
        # particle and leaves "Sana Ahmed bol do", which parses as nothing.
        particle = _PARTICLES.get(match.group(1).lower(), "")
        return f"{who.name} {particle}".strip()

    rewritten = _PERSON_PRONOUN.sub(substitute, text, count=1)
    log.info("Resolved a pronoun: %r -> %r", text, rewritten)
    return rewritten


def repair_referent_args(args: dict[str, Any], original: str,
                         memory: TurnMemory) -> None:
    """Put the exact contact into a recipient slot, in place.

    Substitution exists to make the *rule* match; it is not a reliable way to
    fill in the recipient. "tell Sana Ahmed I'll be late" splits on the first
    space after the verb, so the rule reads the name as "sana" and the message
    as "ahmed I'll be late" — a two-word name is enough to break it, and every
    substituted name is at least two words.

    So once a rule has matched, the recipient it guessed is replaced by the
    contact we already knew. Only ever narrows a guess to a known name, and
    only when the utterance actually contained a pronoun.
    """
    if "to" not in args or not _PERSON_PRONOUN.search(original or ""):
        return
    who = memory.recall("contact")
    if who is None:
        return
    guessed = str(args.get("to") or "").strip().lower()
    # The rule's guess should be part of the name it came from. Anything else
    # means this turn is about somebody new and memory has no business here.
    if guessed and guessed not in who.name.lower():
        return
    args["to"] = who.name

    # The same split that truncated the name spilled its tail into the
    # message: "tell Sana Ahmed I'll be late" is read as a recipient of "sana"
    # and a message of "Ahmed I'll be late". Whatever part of the name the
    # recipient slot did not take has to come back off the front of the body,
    # or the surname gets sent to the person it names.
    body = str(args.get("message") or "").strip()
    if not body:
        return
    leftover = [w for w in who.name.split() if w.lower() not in guessed.split()]
    while leftover and body.lower().startswith(leftover[0].lower()):
        body = body[len(leftover[0]):].lstrip(" ,")
        leftover.pop(0)
    args["message"] = body


def resolve_object_pronoun(text: str, memory: TurnMemory) -> str:
    """The one thing-pronoun worth substituting: "close it".

    Kept apart from the general case and deliberately narrow. "turn it up" is
    a volume command that works today, and a rule rewriting every "it" to the
    last application would break it. This fires only when a closing verb makes
    the object unambiguous — without it, "close it" reaches `close_app` with
    the literal word "it" as the application name.
    """
    match = _CLOSE_THAT.match(text.strip())
    if not match:
        return text
    what = memory.recall("app")
    if what is None:
        return text
    rewritten = f"{match.group('verb')} {what.name}"
    log.info("Resolved an object pronoun: %r -> %r", text, rewritten)
    return rewritten
