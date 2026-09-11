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


def _await_focus(process: str, timeout: float) -> bool:
    """Wait until `process` owns the focused window. False if it never does.

    This is the whole safety story for pressing send. Without it the keystroke
    goes wherever focus happens to be when the timer expires, which could be a
    different chat, a different app, or a terminal.
    """
    if not process:
        return False
    deadline = time.monotonic() + timeout
    wanted = process.lower()
    while time.monotonic() < deadline:
        current = str(winutil.foreground_window().get("process", "")).lower()
        if current == wanted:
            # Focus has landed, but the chat pane may still be painting and the
            # pre-filled text is placed by the app itself. A beat here is the
            # difference between sending the message and sending an empty one.
            time.sleep(0.6)
            return str(
                winutil.foreground_window().get("process", "")
            ).lower() == wanted
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

    link = messaging.build_link(chosen, who.address, body)
    if not link:
        return fail(
            f"I can't start a {chosen.label} chat from outside the app.",
            f"{chosen.label} ka chat bahar se khol nahi sakta.",
        )

    if not winutil.shell_open(link):
        return fail(f"I couldn't open {chosen.label}.",
                    f"{chosen.label} nahi khul paya.")

    named = who.name or target
    if not settings.messaging.auto_send:
        return ok(
            f"{chosen.label} is open with your message to {named} \u2014 "
            "press send when you're happy with it.",
            f"{chosen.label} khul gaya hai, {named} ko message taiyaar hai \u2014 "
            "aap send daba dijiye.",
            detail=f"draft to={who.phone} app={chosen.id}",
            to=who.phone,
        )

    if not _await_focus(chosen.process, settings.messaging.focus_timeout_sec):
        return ok(
            f"Your message to {named} is ready in {chosen.label}, but it didn't "
            "come to the front in time so I've left it for you to send.",
            f"{named} ko message {chosen.label} mein taiyaar hai, par window "
            "samay par saamne nahi aayi \u2014 aap send kar dijiye.",
            detail=f"unsent (no focus) to={who.phone} app={chosen.id}",
            to=who.phone,
        )

    if not winutil.send_keys("enter"):
        return ok(
            f"Your message to {named} is ready in {chosen.label} \u2014 press "
            "send when you're ready.",
            f"{named} ko message taiyaar hai \u2014 send daba dijiye.",
            detail=f"unsent (no keystroke) to={who.phone} app={chosen.id}",
            to=who.phone,
        )

    return ok(
        f"Sent to {named} on {chosen.label}.",
        f"{named} ko {chosen.label} par bhej diya.",
        detail=f"sent to={who.phone} app={chosen.id}",
        to=who.phone,
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
