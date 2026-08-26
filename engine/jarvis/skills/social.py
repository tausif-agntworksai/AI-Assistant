"""Greetings, acknowledgements and dismissals — answered without a model.

These exist for one reason: they were reaching the language model, and they
have no business doing so. Saying "hey jarvis" on its own, or "thanks", or
"never mind" would fall through the rule table, get handed to Claude, and — on
a fresh install with no key — come back as "I need an AI key for that one."
Asking a paid model to answer "hello" is the wrong shape of solution twice
over: it costs money and it takes two seconds.

Nothing here needs a capability. It reads no files, touches no windows and
makes no network call; it is the assistant saying a sentence back.
"""

from __future__ import annotations

import random
import time

from ..permissions import Risk
from .registry import SkillContext, ok, skill

# A little variety, so an assistant you greet every morning doesn't answer with
# the same four words for the rest of its life.
_GREETINGS = (
    ("Hello.", "नमस्ते।"),
    ("Hi there.", "नमस्ते जी।"),
    ("Hey.", "हाँ जी।"),
)

_ACKS = (
    ("I'm listening.", "सुन रहा हूँ।"),
    ("Yes?", "जी?"),
    ("Go ahead.", "बोलिए।"),
)


def _pick(options: tuple[tuple[str, str], ...]) -> tuple[str, str]:
    return random.choice(options)


@skill(
    name="acknowledge",
    description=(
        "Acknowledge being addressed by name with nothing after it. Use when "
        "the user has said only the assistant's name."
    ),
    risk=Risk.SAFE,
    category="general",
    examples=["jarvis", "hey jarvis", "jarvis?", "o jarvis", "sun jarvis"],
)
def acknowledge() -> object:
    english, hindi = _pick(_ACKS)
    return ok(english, hindi)


@skill(
    name="greet",
    description="Greet the user back",
    risk=Risk.SAFE,
    category="general",
    examples=[
        "hello", "hi", "namaste", "good morning", "good evening",
        "kaise ho", "how are you",
    ],
)
def greet() -> object:
    hour = time.localtime().tm_hour
    if 4 <= hour < 12:
        return ok("Good morning.", "सुप्रभात।")
    if 17 <= hour < 22:
        return ok("Good evening.", "शुभ संध्या।")
    english, hindi = _pick(_GREETINGS)
    return ok(english, hindi)


@skill(
    name="acknowledge_thanks",
    description="Respond to thanks",
    risk=Risk.SAFE,
    category="general",
    examples=["thanks", "thank you", "shukriya", "dhanyavad", "thanks jarvis"],
)
def acknowledge_thanks() -> object:
    return ok("Any time.", "कोई बात नहीं।")


@skill(
    name="never_mind",
    description="Drop whatever was being asked for and stand down",
    risk=Risk.SAFE,
    category="general",
    examples=[
        "never mind", "forget it", "nothing", "kuch nahi", "rehne do",
        "chhodo", "cancel that",
    ],
)
def never_mind(ctx: SkillContext = None) -> object:
    return ok("Okay.", "ठीक है।")


@skill(
    name="who_are_you",
    description="Explain what the assistant is and what it can do",
    risk=Risk.SAFE,
    category="general",
    examples=[
        "who are you", "what can you do", "tum kaun ho", "kya kar sakte ho",
        "what are you",
    ],
)
def who_are_you() -> object:
    # Answered locally and on purpose: this is the first thing a new user asks,
    # and it would be a poor introduction to reply that an API key is needed
    # before the assistant can say what it is.
    from .registry import registry

    count = len(registry.all())
    return ok(
        f"I'm Jarvis. I can do about {count} things on this computer — open apps, "
        "control volume and brightness, manage windows, take screenshots, set "
        "timers, and answer questions. Say “what can you do” in the app to see "
        "the full list.",
        f"मैं Jarvis हूँ। इस कंप्यूटर पर लगभग {count} काम कर सकता हूँ — ऐप्स खोलना, "
        "वॉल्यूम और ब्राइटनेस, विंडो संभालना, स्क्रीनशॉट, टाइमर, और सवालों के जवाब।",
    )
