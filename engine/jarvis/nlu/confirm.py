"""Yes/no detection for the confirmation gate.

Deliberately conservative: anything that isn't clearly a yes counts as a no.
The gate protects shutdowns and outbound messages, so an ambiguous mumble must
never be read as approval.
"""

from __future__ import annotations

from .normalize import normalize

_YES = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "affirmative", "confirm",
    "confirmed", "go", "go ahead", "do it", "please do", "correct", "right",
    "haan", "han", "haa", "ha", "ji", "ji haan", "bilkul", "theek", "theek hai",
    "thik hai", "sahi", "kar do", "karo", "kar dijiye", "chalo", "acha",
}

_NO = {
    "no", "nope", "nah", "cancel", "stop", "don't", "dont", "do not", "abort",
    "never mind", "nevermind", "forget it", "wait", "negative",
    "nahi", "nai", "na", "mat", "mat karo", "rehne do", "rehne de", "ruko",
    "chhodo", "chodo", "cancel karo", "nahi karo", "bilkul nahi",
}


def is_affirmative(text: str) -> bool:
    """True only for a clear yes."""
    cleaned = normalize(text)
    if not cleaned:
        return False

    if cleaned in _NO:
        return False
    if cleaned in _YES:
        return True

    words = set(cleaned.split())

    # An explicit refusal anywhere wins — "haan nahi ruko" is a no.
    if words & {w for w in _NO if " " not in w}:
        return False
    if any(phrase in cleaned for phrase in _NO if " " in phrase):
        return False

    if words & {w for w in _YES if " " not in w}:
        return True
    return any(phrase in cleaned for phrase in _YES if " " in phrase)


def is_negative(text: str) -> bool:
    """True for a clear no (distinct from 'unclear')."""
    cleaned = normalize(text)
    if not cleaned:
        return False
    if cleaned in _NO:
        return True
    words = set(cleaned.split())
    if words & {w for w in _NO if " " not in w}:
        return True
    return any(phrase in cleaned for phrase in _NO if " " in phrase)
