"""How Jarvis says things, and nothing else.

The brief asks for a particular register — calm, brief, confident, occasionally
dry — and warns against the one every assistant drifts into: "Sure! I'd be
happy to help you with that." The difference between those is entirely in the
wording, which is exactly why this module exists **downstream of every
decision**.

It is handed a finished result and returns different words for it. It cannot
see the tool, cannot change the arguments, cannot suppress an action and cannot
invent one. That separation is the whole point: personality is the part of an
assistant most likely to be tuned on a whim, and it must not be possible to
change how Jarvis talks and accidentally change what Jarvis does.

Two things it deliberately does not do. It does not rewrite a skill's own
reply into something vaguer — "Sent to Sana Ahmed on WhatsApp" names the person
the message actually reached, and trimming that to "Done" would remove the one
detail worth hearing. And it does not touch Hindi. The trimming below is a list
of English hedges; applying English rules to Devanagari would mangle it.
"""

from __future__ import annotations

import logging
import random
import re

log = logging.getLogger(__name__)

#: Discourse markers: openers that can be removed and still leave a complete
#: sentence behind. Only ever stripped from the front — "actually" in the
#: middle of a sentence is doing real work.
#:
#: Verb-phrase openers like "I'd be happy to" are deliberately NOT in this
#: list. Removing one leaves a bare infinitive — "Sure! I'd be happy to help
#: you with that." became "Help you with that.", which is worse than either
#: the original or silence. Those are handled as whole-reply filler below.
_MARKERS = (
    "sure, ", "sure! ", "sure thing, ", "of course, ", "of course! ",
    "certainly, ", "absolutely, ", "no problem, ", "alright, ", "okay, ",
    "ok, ", "well, ", "so, ", "actually, ", "just to confirm, ",
    "as an ai assistant, ", "as an ai, ",
)

#: Whole replies that are pure offering. Matched as a prefix, because the tail
#: varies ("...help you with that" / "...assist with this"). A reply that only
#: offers to do something has said nothing, and for a voice assistant that is
#: better dropped than read aloud.
_OFFERS = (
    "i'd be happy to help", "i would be happy to help", "i'd be glad to help",
    "i'm happy to help", "happy to help", "how can i help",
    "how may i assist", "what can i do for you",
)

#: Whole replies that are pure filler.
_EMPTY = frozenset({
    "sure", "of course", "no problem", "you're welcome", "certainly",
    "is there anything else i can help you with?",
    "let me know if you need anything else.",
})

#: Confirmations for an action that produced nothing worth reporting. Rotated
#: so a run of commands doesn't sound like one recording played four times.
_ACKS_EN = ("Done.", "Certainly.", "Of course.", "Right away.")
_ACKS_HI = ("हो गया।", "ज़रूर।", "ठीक है।", "अभी करता हूँ।")

_TRAILING_OFFER = re.compile(
    r"\s*(?:is there anything else[^.?!]*[.?!]"
    r"|let me know if[^.?!]*[.?!]"
    r"|anything else[^.?!]*[.?!])\s*$",
    re.IGNORECASE,
)


def _is_devanagari(text: str) -> bool:
    """Hindi is left alone — the trimming below is a list of English hedges."""
    return any("ऀ" <= ch <= "ॿ" for ch in text)


def trim(text: str) -> str:
    """Remove the padding around a reply, leaving what it actually said.

    Only ever removes; never rephrases. A reply that is already terse comes
    back untouched, which is the common case — skills write their own short
    strings and it is the model's prose this is really aimed at.
    """
    if not text or _is_devanagari(text):
        return text

    cleaned = text.strip()
    if cleaned.lower().rstrip(".!") in _EMPTY:
        return ""

    # Markers stack: "Well, actually, the battery is at 65 percent."
    changed = True
    while changed:
        changed = False
        for marker in _MARKERS:
            if cleaned.lower().startswith(marker):
                cleaned = cleaned[len(marker):].lstrip()
                changed = True
                break

    # Only after the markers are off, because "Sure! I'd be happy to help"
    # hides the offer behind one.
    if cleaned.lower().startswith(_OFFERS):
        return ""

    cleaned = _TRAILING_OFFER.sub("", cleaned).strip()
    if not cleaned or cleaned.lower().rstrip(".!") in _EMPTY:
        return ""

    # Stripping a leading hedge leaves the next word lowercased mid-sentence.
    if cleaned and cleaned[0].islower() and not cleaned.startswith(("http", "www")):
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


class Voice:
    """The register Jarvis speaks in.

    Stateful only to the extent of not repeating itself: an acknowledgement
    picked at random twice running sounds like a bug, so the last one is
    remembered and avoided.
    """

    def __init__(self) -> None:
        self._last_ack = ""

    def acknowledge(self, language: str = "en") -> str:
        """A short confirmation for an action that reported nothing itself."""
        options = _ACKS_HI if language.startswith("hi") else _ACKS_EN
        choices = [o for o in options if o != self._last_ack] or list(options)
        self._last_ack = random.choice(choices)
        return self._last_ack

    def compose(self, reply: str, language: str = "en",
                acted: bool = False) -> str:
        """The words to say for a finished turn.

        `acted` distinguishes "the skill ran and had nothing to add" from
        "nothing happened at all". The first deserves a word of confirmation;
        the second deserves silence, because inventing "Done." for a turn that
        did nothing is how an assistant becomes untrustworthy.
        """
        spoken = trim(reply or "")
        if spoken:
            return spoken
        if acted:
            return self.acknowledge(language)
        return ""


voice = Voice()
