"""Which app sends a message, and to which number.

Split out of `skills/messaging.py` because three separate things needed it — the
skill, the rule matcher and the tests — and because the interesting decisions
here are worth stating in one place rather than three.

**The app.** "Say hi to sana" names no app, and the answer has to be WhatsApp:
it is what the phrase means in practice for this user. Naming one ("tell sana on
telegram") overrides it. So the default is configuration, not a hardcoded
constant, and the override is a lookup rather than a branch.

**The number.** Every app here addresses people by phone number, and the user
says names. That means a fuzzy lookup over saved contacts, and fuzzy lookups on
*names* are where this could go badly wrong: sending "I'm running late" to the
wrong person is not something the user can take back. So the matching here is
deliberately stricter than elsewhere in the codebase, and it reports *why* it
failed — "I don't have a number for Sana" and "did you mean Sana or Sanjay?" need
different answers, and both are better than sending to the wrong Sana.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from dataclasses import dataclass

from . import store

log = logging.getLogger(__name__)

#: A confident single match. Higher than the 80 used elsewhere for skill names,
#: because the cost of being wrong is different in kind: a wrong skill wastes a
#: turn, a wrong contact reaches a person.
MATCH_STRONG = 86

#: Below this, it is not a candidate at all.
MATCH_FLOOR = 70

#: Two candidates within this many points of each other are a tie worth asking
#: about rather than guessing between. "Sana" against a contact list holding both
#: Sana and Sanjay is the case this exists for.
TIE_WINDOW = 8

#: Bare local numbers are assumed Indian, matching the rest of the assistant.
DEFAULT_COUNTRY_CODE = "91"


@dataclass(frozen=True)
class App:
    """A messaging app the assistant can hand a pre-filled message to."""

    id: str
    label: str
    #: Words a user might say for it, normalised. The rule matcher and the LLM
    #: both resolve through these.
    aliases: tuple[str, ...]
    #: URI template taking `phone` and `text`. None when the app has no way to
    #: address a specific person from outside it.
    deep_link: str | None
    #: Opened when there is no number to address — the app itself, so the user
    #: can pick the chat by hand.
    fallback: str | None = None
    #: Process name, used to confirm the right window has focus before any
    #: keystroke is sent to it.
    process: str = ""
    needs_number: bool = True


APPS: tuple[App, ...] = (
    App(
        id="whatsapp",
        label="WhatsApp",
        aliases=("whatsapp", "whats app", "wa", "watsapp", "whatsap"),
        # wa.me hands off to WhatsApp Desktop when it is installed and to the
        # browser when it is not, which is the behaviour we want in both cases.
        deep_link="https://wa.me/{phone}?text={text}",
        fallback="whatsapp://",
        process="WhatsApp.exe",
    ),
    App(
        id="sms",
        label="Messages",
        aliases=("sms", "text message", "message app", "text"),
        deep_link="sms:{phone}?body={text}",
        process="",
    ),
    App(
        id="telegram",
        label="Telegram",
        aliases=("telegram", "tg"),
        # Telegram addresses by @username, not by number, so a phone number
        # cannot open a specific chat from outside. Opening the app with the
        # text staged is the most that is honestly possible.
        deep_link=None,
        fallback="tg://msg?text={text}",
        process="Telegram.exe",
        needs_number=False,
    ),
    App(
        id="signal",
        label="Signal",
        aliases=("signal",),
        deep_link=None,
        fallback="sgnl://",
        process="Signal.exe",
        needs_number=False,
    ),
    App(
        id="slack",
        label="Slack",
        aliases=("slack",),
        deep_link=None,
        fallback="slack://open",
        process="slack.exe",
        needs_number=False,
    ),
)

_BY_ID = {app.id: app for app in APPS}


def app_names() -> str:
    """Alternation of every alias, for embedding in a rule pattern."""
    aliases = sorted(
        (a for app in APPS for a in app.aliases), key=len, reverse=True
    )
    return "|".join(re.escape(a) for a in aliases)


def resolve_app(name: str = "", default: str = "whatsapp") -> App:
    """Find the app the user named, falling back to the configured default.

    An unrecognised name falls back rather than failing. "Send hi to sana on
    hike" should still send it — on WhatsApp, with the reply saying so — because
    the recipient and the message were both understood and refusing over the
    transport helps nobody.
    """
    needle = _normalise(name)
    if needle:
        for app in APPS:
            if needle in {_normalise(a) for a in app.aliases}:
                return app
        log.info("Unknown messaging app %r — using %s", name, default)
    return _BY_ID.get(default, _BY_ID["whatsapp"])


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (text or "").strip().lower())


@dataclass(frozen=True)
class Recipient:
    """The outcome of looking a name up in the contact list."""

    phone: str = ""
    name: str = ""
    #: Set when two contacts were too close to choose between.
    ambiguous: tuple[str, ...] = ()

    @property
    def found(self) -> bool:
        return bool(self.phone)


def normalise_phone(raw: str) -> str:
    """Digits only, with a country code. Empty when it isn't a phone number."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) < 10:
        return ""
    if len(digits) == 10:
        return DEFAULT_COUNTRY_CODE + digits
    return digits


def resolve_recipient(target: str) -> Recipient:
    """Turn what the user said into a number, a tie, or nothing.

    Three outcomes rather than two, because "I don't have a number for Sana" and
    "did you mean Sana or Sanjay?" call for different replies, and guessing
    between two close contacts is the one failure that reaches a stranger.
    """
    spoken = (target or "").strip()
    if not spoken:
        return Recipient()

    direct = normalise_phone(spoken)
    if direct:
        return Recipient(phone=direct, name=spoken)

    from rapidfuzz import fuzz

    needle = spoken.lower()
    scored: list[tuple[float, str, str]] = []
    for contact in store.load("contacts", []):
        name = str(contact.get("name", "")).strip()
        phone = normalise_phone(str(contact.get("phone", "")))
        if not name or not phone:
            continue
        # token_set_ratio so "sana" matches a contact saved as "Sana Khan",
        # which WRatio scores down for being shorter than the stored name.
        score = max(
            fuzz.token_set_ratio(needle, name.lower()),
            fuzz.ratio(needle, name.lower()),
        )
        scored.append((score, name, phone))

    if not scored:
        return Recipient()

    scored.sort(key=lambda row: row[0], reverse=True)
    best_score, best_name, best_phone = scored[0]
    if best_score < MATCH_FLOOR:
        return Recipient()

    rivals = [
        name for score, name, _ in scored[1:]
        if best_score - score <= TIE_WINDOW and score >= MATCH_FLOOR
    ]
    if rivals:
        return Recipient(ambiguous=tuple([best_name, *rivals[:2]]))

    if best_score < MATCH_STRONG:
        # Close enough to be the intended person, not close enough to message
        # without saying whose name was matched. The skill puts the resolved
        # name in its reply, so the user hears it before pressing send.
        log.info("Contact %r matched %r at %.0f", spoken, best_name, best_score)
    return Recipient(phone=best_phone, name=best_name)


def build_link(app: App, phone: str, message: str) -> str:
    """The URI that opens this app with the message staged."""
    text = urllib.parse.quote(message.strip())
    if phone and app.deep_link:
        return app.deep_link.format(phone=phone, text=text)
    template = app.fallback or ""
    return template.format(phone=phone, text=text) if template else ""
