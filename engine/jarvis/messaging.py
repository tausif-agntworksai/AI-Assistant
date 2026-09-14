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

#: Charged per word the user said that a stored name does not contain. Larger
#: than TIE_WINDOW on purpose: one missing word has to be enough to settle a
#: tie outright, or saying a name in full still ends in a question.
_MISSING_WORD_PENALTY = 12.0

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
    #: What this app addresses people by. WhatsApp and SMS want a phone
    #: number; Gmail wants an email address. The contact book holds both, so
    #: this decides which column a lookup reads rather than forcing a second
    #: resolver per platform.
    address_kind: str = "phone"
    #: The keys that focus this app's own search box. Used when we have a name
    #: but no number: the app already knows the contact even when we do not,
    #: so putting the name into its search is the difference between landing
    #: on the right chat and being dropped at a list of everyone.
    search_keys: tuple[str, ...] = ()


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
        search_keys=("ctrl", "f"),
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
        id="gmail",
        label="Gmail",
        aliases=("gmail", "email", "mail", "e mail", "google mail"),
        # Gmail's compose URL, which opens a pre-filled draft in the browser
        # already signed in as the user. `mailto:` was the old route and went
        # to whatever Windows had registered — usually nothing at all.
        deep_link=("https://mail.google.com/mail/?view=cm&fs=1"
                   "&to={phone}&su={subject}&body={text}"),
        fallback="https://mail.google.com/mail/?view=cm&fs=1&body={text}",
        process="",
        address_kind="email",
    ),
    App(
        id="googlechat",
        label="Google Chat",
        aliases=("google chat", "gchat", "g chat", "chat", "hangouts"),
        # Chat has no documented URL that opens a conversation with a named
        # person from outside it — the per-chat URLs are opaque internal ids.
        # Opening Chat itself is the most that can be done honestly, so the
        # message is staged and the user picks the conversation.
        deep_link=None,
        fallback="https://mail.google.com/chat/u/0/",
        process="",
        needs_number=False,
        address_kind="email",
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
    #: Filled for apps that address people by email rather than by number.
    email: str = ""

    @property
    def found(self) -> bool:
        return bool(self.phone or self.email)

    @property
    def address(self) -> str:
        """Whichever of the two this recipient was resolved for."""
        return self.phone or self.email


def normalise_phone(raw: str) -> str:
    """Digits only, with a country code. Empty when it isn't a phone number."""
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) < 10:
        return ""
    if len(digits) == 10:
        return DEFAULT_COUNTRY_CODE + digits
    return digits


def match_key(name: str) -> str:
    """A contact name reduced to the words someone would actually say.

    Phone books are not written to be spoken. The same person is "Sana Ahmed"
    to one app, "sana ❤️" to another, "Sana(Office)" at work and "सना" in
    Hindi — and a heart is not a word anybody pronounces. Scoring against the
    stored spelling alone made "sana" a 50 against "sana(office)", far under
    the floor, so a contact the user could see in their phone simply could not
    be found.

    `normalize` already does most of this for the command language — it
    transliterates Devanagari, drops emoji, and splits on dots and brackets —
    so this reuses it rather than growing a second, differently-wrong copy.
    What it leaves behind are the joiners that only ever appear in names:
    underscores and hyphens.
    """
    from .nlu.normalize import normalize

    text = normalize(name or "")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", (value or "").strip()))


def resolve_recipient(target: str, address_kind: str = "phone") -> Recipient:
    """Turn what the user said into a number, a tie, or nothing.

    Three outcomes rather than two, because "I don't have a number for Sana" and
    "did you mean Sana or Sanjay?" call for different replies, and guessing
    between two close contacts is the one failure that reaches a stranger.
    """
    spoken = (target or "").strip()
    if not spoken:
        return Recipient()

    if address_kind == "email":
        if _looks_like_email(spoken):
            return Recipient(email=spoken, name=spoken)
    else:
        direct = normalise_phone(spoken)
        if direct:
            return Recipient(phone=direct, name=spoken)

    from rapidfuzz import fuzz

    from . import contacts as book

    # What this name meant the last time we had to ask. Checked before any
    # scoring: the user already answered this question once, and asking again
    # is the thing that made them stop using the feature.
    preferred = book.learned(spoken)

    from .resolve import decide, rank

    wants_email = address_kind == "email"

    reachable: list[tuple[str, str]] = []
    for contact in _contact_rows():
        name = str(contact.get("name", "")).strip()
        if wants_email:
            address = str(contact.get("email", "")).strip()
            if not _looks_like_email(address):
                address = ""
        else:
            address = normalise_phone(str(contact.get("phone", "")))
        # A contact with no address of the kind this app needs is not a
        # candidate for it. Someone can be perfectly findable for WhatsApp and
        # unreachable by email, and scoring them here would produce a confident
        # match that then has nowhere to send anything.
        if not name or not address:
            continue
        if preferred and match_key(name) == match_key(preferred):
            log.info("Contact %r resolved to %r from a remembered choice",
                     spoken, name)
            return _recipient(name, address, wants_email)
        reachable.append((name, address))

    def score(needle: str, term: str) -> float:
        """How well a spoken name matches one spelling of a stored one.

        `token_set_ratio` so "sana" matches a contact saved as "Sana Khan",
        which `WRatio` scores down for being shorter than the stored name. But
        it treats a subset as a perfect match in both directions, which makes
        saying *more* useless: "sana ahmed" scored 100 against both Sana Ahmed
        and a contact saved as plain "sana", so naming someone in full asked
        which one you meant instead of answering.

        The asymmetry it is missing is that the two directions mean different
        things. A stored name with extra words is ordinary — people shorten
        names constantly — but a stored name *missing* a word the user said is
        evidence against it being the one they meant.
        """
        base = max(fuzz.token_set_ratio(needle, term), fuzz.ratio(needle, term))
        unmatched = set(needle.split()) - set(term.split())
        return base - _MISSING_WORD_PENALTY * len(unmatched)

    candidates = rank(
        match_key(spoken) or spoken, reachable,
        # Both spellings, because `rank` takes the best term and neither is
        # reliably the right one: the stored name is what the user chose, the
        # match key is what they will actually say out loud.
        terms=lambda row: [row[0], match_key(row[0])],
        name=lambda row: row[0],
        score=score,
        floor=MATCH_FLOOR,
        limit=4,
    )
    # A confident score must not silence a tie here: two people called Sana
    # both score 100 against "Sana", and that is the case this exists for.
    outcome = decide(candidates, strong=MATCH_STRONG, tie_window=TIE_WINDOW,
                     strong_breaks_ties=False)

    if outcome.best is None:
        return Recipient()

    if outcome.ambiguous:
        # Hold on to what was asked and who it was between. If the next attempt
        # names one of these, that answer is worth keeping — otherwise the user
        # is asked "which Sana?" every single time, which is the complaint this
        # whole path exists to avoid.
        global _pending
        choices = (outcome.best.name, *outcome.rivals[:2])
        _pending = (spoken, choices)
        return Recipient(ambiguous=choices)

    best_name, best_address = outcome.best.item
    if outcome.best.score < MATCH_STRONG:
        # Close enough to be the intended person, not close enough to message
        # without saying whose name was matched. The skill puts the resolved
        # name in its reply, so the user hears it before pressing send.
        log.info("Contact %r matched %r at %.0f", spoken, best_name,
                 outcome.best.score)

    _learn_from_answer(best_name)
    return _recipient(best_name, best_address, wants_email)


def _recipient(name: str, address: str, wants_email: bool) -> Recipient:
    return (Recipient(email=address, name=name) if wants_email
            else Recipient(phone=address, name=name))


#: The last name we had to ask about, and who it was between. Process-local on
#: purpose: it is a half-finished sentence, not a setting, and it should not
#: survive a restart.
_pending: tuple[str, tuple[str, ...]] | None = None


def _learn_from_answer(resolved: str) -> None:
    """If this answers an earlier "which one did you mean?", write it down."""
    global _pending
    if _pending is None:
        return
    spoken, candidates = _pending
    _pending = None
    if resolved not in candidates:
        return  # a different request entirely; the question went unanswered
    from . import contacts as book

    book.remember(spoken, resolved)


def _contact_rows() -> list[dict]:
    """Every contact as a plain row.

    Reads through `contacts.entries()` so an imported address book is visible
    here, while still going through `store` for the manual ones — which is what
    keeps this substitutable in tests.
    """
    from . import contacts as book

    return [{"name": c.name, "phone": c.phone, "email": c.email}
            for c in book.entries()]


def build_link(app: App, phone: str, message: str, subject: str = "") -> str:
    """The URI that opens this app with the message staged.

    `phone` is whatever this app addresses people by — a number for WhatsApp,
    an email address for Gmail. The parameter keeps its name because every
    caller and every template already uses it, and giving the same slot two
    names would be worse than one slightly wrong one.
    """
    text = urllib.parse.quote(message.strip())
    fields = {
        "phone": urllib.parse.quote(phone) if app.address_kind == "email" else phone,
        "text": text,
        "subject": urllib.parse.quote(subject.strip()),
    }
    template = app.deep_link if (phone and app.deep_link) else (app.fallback or "")
    if not template:
        return ""
    # Templates only use the fields they need, so format with all of them and
    # let the unused ones fall away.
    return template.format(**fields)
