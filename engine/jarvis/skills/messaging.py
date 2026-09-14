"""Outbound messages — WhatsApp, SMS, other chat apps, and email.

Everything here is CRITICAL: it reaches other people, and a misheard command is
the one kind of mistake the user cannot take back. So every one of these is read
back and confirmed before anything opens.

`send_message` will press send once the draft is up, which is what "say hi to
sana" asks for. Two things make that safe enough to be the default:

  * the command is confirmed out loud first, *with the resolved contact name in
    it* — so a fuzzy match onto the wrong Sana is caught by the user before
    anything is typed, not after;
  * the keystroke is only sent once the messaging app is confirmed to be the
    focused window. If it is slow to start, or the user clicks away, the draft
    is left sitting there and the reply says so. A stray Enter into whatever
    happened to be focused is the failure this prevents.

Set `messaging.auto_send: false` to always stop at the draft.

`compose_email` does not do this: mail clients vary far too much for a blind
keystroke to mean "send" in all of them.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse

from .. import messaging, store, winutil
from ..config import settings
from ..permissions import Risk
from .registry import fail, ok, skill

log = logging.getLogger(__name__)


#: How much of a spoken message becomes the subject line when the user never
#: said one. Long enough to be recognisable in an inbox, short enough not to
#: repeat the whole message in the preview.
_SUBJECT_WORDS = 8


def _subject_from(body: str) -> str:
    """A subject line for a message that was dictated, not composed.

    Nobody says "subject colon" out loud, and asking for one turns a
    one-sentence errand into an interview. The first clause of the message is
    what a person would have written anyway, and it is honest — it promises
    the reader exactly what the mail says.
    """
    first = re.split(r"(?<=[.!?])\s|\n", (body or "").strip(), maxsplit=1)[0]
    words = first.split()
    if not words:
        return "(no subject)"
    if len(words) <= _SUBJECT_WORDS:
        return first.rstrip(".")
    return " ".join(words[:_SUBJECT_WORDS]) + "…"


def _send_by_api(address: str, body: str) -> tuple[bool, str]:
    """Try to send through the Gmail API. False means use the browser instead.

    Never starts an authorisation flow: being redirected to a consent page
    because you said "email Sana" would be startling, and the middle of a
    spoken turn is the worst moment to read a permissions dialog. Unauthorised
    simply means the compose window, which has always worked.

    A dry run never reaches here — `registry.execute` reports what it would do
    and returns before the skill is called at all.
    """
    from .. import gmail_api

    if not gmail_api.authorised():
        return False, "not authorised"

    sent, detail = gmail_api.send(address, _subject_from(body), body)
    if not sent:
        log.info("Gmail API send failed (%s) — opening the compose window", detail)
    return sent, detail


def _same_program(current: str, wanted: str) -> bool:
    """Whether two executable names are the same program.

    Exact equality was wrong in the one case that matters most. The Store
    build of WhatsApp runs as `WhatsApp.Root.exe`, not `WhatsApp.exe`, so the
    focus check never matched, Enter was never pressed, and every message
    stopped one keystroke short with "it didn't come to the front in time" —
    a message that was not even true.

    Compared on the first dot-separated component, which is the program's own
    name; what follows is the vendor's business (`.Root`, a channel, a
    version) and never distinguishes one application from another.
    """
    def stem(name: str) -> str:
        return name.lower().removesuffix(".exe").split(".")[0].strip()

    left, right = stem(current), stem(wanted)
    return bool(left) and left == right


def _search_in_app(app, name: str, timeout: float) -> bool:
    """Put `name` into the app's own search box. True if it got there.

    The app knows the contact even when we do not. WhatsApp has "Sana ❤️" in
    its list whether or not anything was ever imported into Jarvis, so the
    useful thing to do with a name and no number is hand the name to the
    program that can already find it — the difference between landing on the
    right chat and being dropped at a list of everyone.

    Deliberately stops at the search results. Pressing Enter would open
    whichever chat happened to rank first, and "first result" is not a
    standard worth messaging a stranger over.
    """
    if not app.search_keys or not name.strip():
        return False
    if not _await_focus(app.process, timeout):
        return False
    if not winutil.send_keys(*app.search_keys):
        return False

    # A beat for the search field to take focus before anything is typed, for
    # the same reason the send path waits: the keystrokes land wherever focus
    # is at that instant, not where it is about to be.
    time.sleep(0.4)
    if not _same_program(str(winutil.foreground_window().get("process", "")),
                         app.process):
        return False

    try:
        import pyautogui

        pyautogui.write(name, interval=0.01)
    except Exception as exc:  # noqa: BLE001 - a failed search is not a failed turn
        log.info("Could not type the search term (%s)", exc)
        return False
    return True


def _await_focus(process: str, timeout: float) -> bool:
    """Wait until `process` owns the focused window. False if it never does.

    This is the whole safety story for pressing send. Without it the keystroke
    goes wherever focus happens to be when the timer expires, which could be a
    different chat, a different app, or a terminal.
    """
    if not process:
        return False
    deadline = time.monotonic() + timeout
    wanted = process
    while time.monotonic() < deadline:
        current = str(winutil.foreground_window().get("process", ""))
        if _same_program(current, wanted):
            # Focus has landed, but the chat pane may still be painting and the
            # pre-filled text is placed by the app itself. A beat here is the
            # difference between sending the message and sending an empty one.
            time.sleep(0.6)
            return _same_program(
                str(winutil.foreground_window().get("process", "")), wanted
            )
        time.sleep(0.15)
    log.info("%s never took focus — leaving the draft unsent", process)
    return False


@skill(
    name="send_message",
    description=(
        "Send a message to a contact through WhatsApp or another chat app. "
        "Use this for 'say hi to X', 'tell X that ...', and 'message X'."
    ),
    risk=Risk.CRITICAL,
    category="messaging",
    params={
        "to": "Contact name, or a phone number with country code",
        "message": "The message text",
        "app": "Which app to use. Leave blank for the default (WhatsApp).",
    },
    examples=[
        "say hi to sana",
        "tell amit i'm running late",
        "send a whatsapp to amit saying i'm on my way",
        "whatsapp karo mummy ko ki main aa raha hoon",
        "sana ko hi bol do",
        "message rahul on telegram",
        "papa ko message bhejo ki khana kha liya",
    ],
    # The resolved contact name is interpolated here on purpose. It is the one
    # chance the user gets to catch a fuzzy match onto the wrong person, and it
    # happens before anything opens.
    confirm_en="Send \u201c{message}\u201d to {to}?",
    confirm_hi="{to} ko \u201c{message}\u201d bhej doon?",
)
def send_message(to: str, message: str = "", app: str = "") -> object:
    target = (to or "").strip()
    body = (message or "").strip()

    if not target:
        return fail("Who should I message?", "Kise message karun?")
    if not body:
        return fail(f"What should I say to {target}?",
                    f"{target} ko kya bolna hai?")

    chosen = messaging.resolve_app(app, settings.messaging.default_app)
    who = messaging.resolve_recipient(target, chosen.address_kind)

    # A tie between two contacts is the one case worth stopping for. Guessing
    # here sends a private message to the wrong person, which no confirmation
    # after the fact can undo.
    if who.ambiguous:
        names = " or ".join(who.ambiguous)
        return fail(
            f"I have more than one contact like that — did you mean {names}?",
            f"Ek se zyada contact hain — {names} mein se kaun?",
            detail="ambiguous contact",
        )

    if not who.found and chosen.needs_number:
        link = messaging.build_link(chosen, "", body)
        if link:
            winutil.shell_open(link)

        # We have a name and no number, but the app has both. Handing the name
        # to its search leaves the user one click from the right chat instead
        # of staring at a list.
        if _search_in_app(chosen, target, settings.messaging.focus_timeout_sec):
            return ok(
                f"I don't have a number saved for {target}, so I've opened "
                f"{chosen.label} and searched for them \u2014 pick the chat and "
                "I'll remember the number if you tell me.",
                f"{target} ka number save nahi hai, isliye {chosen.label} mein "
                f"unhe search kar diya hai \u2014 chat chun lijiye.",
                detail=f"searched {chosen.id} for {target!r}",
                searched=target, platform=chosen.id,
            )

        # Gmail addresses people by email, WhatsApp by number, and telling
        # someone to save a "number" for Gmail sends them looking for the
        # wrong thing.
        missing = ("an email address" if chosen.address_kind == "email"
                   else "a number")
        return fail(
            f"I don't have {missing} saved for {target}, so I've opened "
            f"{chosen.label} for you. Say \u201csave {target}\u2019s number as\u201d "
            "and the number to fix that.",
            f"{target} ka number save nahi hai, isliye {chosen.label} khol diya "
            f"hai. \u201c{target} ka number save karo\u201d bol kar number bata "
            "dijiye.",
            detail="no contact match",
        )

    named = who.name or target

    # Gmail can send outright once the user has authorised it, which is the
    # one platform here where the whole errand can finish without a window
    # opening at all. Everything else — and Gmail before it is authorised —
    # goes through the deep link below and is typed into the real app.
    if chosen.address_kind == "email":
        sent, detail = _send_by_api(who.address, body)
        if sent:
            return ok(
                f"Sent to {named}.",
                f"{named} ko bhej diya.",
                detail=detail, to=who.address, contact=named, platform=chosen.id,
            )

    link = messaging.build_link(chosen, who.address, body)
    if not link:
        return fail(
            f"I can't start a {chosen.label} chat from outside the app.",
            f"{chosen.label} ka chat bahar se khol nahi sakta.",
        )

    if not winutil.shell_open(link):
        return fail(f"I couldn't open {chosen.label}.",
                    f"{chosen.label} nahi khul paya.")

    if not settings.messaging.auto_send:
        return ok(
            f"{chosen.label} is open with your message to {named} \u2014 "
            "press send when you're happy with it.",
            f"{chosen.label} khul gaya hai, {named} ko message taiyaar hai \u2014 "
            "aap send daba dijiye.",
            detail=f"draft to={who.phone} app={chosen.id}",
            to=who.phone, contact=named, platform=chosen.id,
        )

    if not _await_focus(chosen.process, settings.messaging.focus_timeout_sec):
        return ok(
            f"Your message to {named} is ready in {chosen.label}, but it didn't "
            "come to the front in time so I've left it for you to send.",
            f"{named} ko message {chosen.label} mein taiyaar hai, par window "
            "samay par saamne nahi aayi \u2014 aap send kar dijiye.",
            detail=f"unsent (no focus) to={who.phone} app={chosen.id}",
            to=who.phone, contact=named, platform=chosen.id,
        )

    if not winutil.send_keys("enter"):
        return ok(
            f"Your message to {named} is ready in {chosen.label} \u2014 press "
            "send when you're ready.",
            f"{named} ko message taiyaar hai \u2014 send daba dijiye.",
            detail=f"unsent (no keystroke) to={who.phone} app={chosen.id}",
            to=who.phone, contact=named, platform=chosen.id,
        )

    return ok(
        f"Sent to {named} on {chosen.label}.",
        f"{named} ko {chosen.label} par bhej diya.",
        detail=f"sent to={who.phone} app={chosen.id}",
        to=who.phone, contact=named, platform=chosen.id,
    )



@skill(
    name="compose_email",
    description="Open a new email in the default mail client, pre-filled",
    risk=Risk.CRITICAL,
    category="messaging",
    params={"to": "Recipient email address", "subject": "Subject line",
            "body": "Message body"},
    examples=[
        "email tausif@example.com about the meeting",
        "compose an email to my manager", "email bhejo boss ko",
    ],
    confirm_en="Open a draft email to {to}?",
    confirm_hi="{to} ko email ka draft kholun?",
)
def compose_email(to: str = "", subject: str = "", body: str = "") -> object:
    url = "mailto:" + urllib.parse.quote(to.strip())
    params = {}
    if subject.strip():
        params["subject"] = subject.strip()
    if body.strip():
        params["body"] = body.strip()
    if params:
        url += "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)

    if not winutil.shell_open(url):
        return fail("I couldn't open your mail app.", "Mail app nahi khul paya.")
    return ok(
        "Draft is open — review it and hit send.",
        "Draft khol diya — dekh lijiye aur send kar dijiye.",
        detail=f"to={to}",
    )


@skill(
    name="add_contact",
    description="Save a contact's phone number for messaging",
    risk=Risk.SAFE,
    category="messaging",
    params={"name": "Contact name", "phone": "Phone number with country code"},
    examples=["save amit's number as 919876543210", "add contact mummy 9876543210"],
)
def add_contact(name: str, phone: str) -> object:
    clean_name = (name or "").strip()
    digits = re.sub(r"\D", "", phone or "")
    if not clean_name or len(digits) < 10:
        return fail("I need a name and a full phone number.",
                    "Naam aur pura phone number chahiye.")

    contacts = store.load("contacts", [])
    contacts = [c for c in contacts if str(c.get("name", "")).lower() != clean_name.lower()]
    contacts.append({"name": clean_name, "phone": digits})
    store.save("contacts", contacts)
    return ok(f"Saved {clean_name}.", f"{clean_name} ka number save kar liya.")
