"""Yes/no detection for the confirmation gate.

Deliberately conservative: anything that isn't clearly a yes counts as a no.
The gate protects shutdowns and outbound messages, so an ambiguous mumble must
never be read as approval.
"""

from __future__ import annotations

from .normalize import normalize

_YES = {
    "yes", "yeah", "yep", "yup", "ya", "yah", "sure", "ok", "okay", "affirmative",
    "confirm", "confirmed", "go", "go ahead", "do it", "please do", "correct",
    "right", "alright", "fine", "of course", "definitely", "absolutely",
    "yes please", "carry on", "proceed", "continue",
    "haan", "han", "haa", "ha", "hanji", "han ji", "ji", "ji haan", "ji han",
    "bilkul", "bilkul karo", "theek", "theek hai", "thik hai", "thik", "sahi",
    "sahi hai", "kar do", "kardo", "karo", "kar dijiye", "kar dijie", "chalo",
    "acha", "achha", "haan karo", "haan kar do", "zaroor", "jaroor",
}

_NO = {
    "no", "nope", "nah", "cancel", "stop", "don't", "dont", "do not", "abort",
    "never mind", "nevermind", "forget it", "wait", "negative", "not now",
    "no thanks", "no thank you", "leave it", "hold on",
    "nahi", "nahin", "nahim", "nai", "na", "mat", "mat karo", "mat kar",
    "rehne do", "rehne de", "rahne do", "ruko", "ruk jao",
    "chhodo", "chodo", "cancel karo", "nahi karo", "bilkul nahi", "abhi nahi",
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
