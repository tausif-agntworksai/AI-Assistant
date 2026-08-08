"""Outbound messages — WhatsApp and email.

Both are CRITICAL: they reach other people, and a misheard command here is the
one kind of mistake the user cannot undo. Neither skill sends anything by
itself. They open the message pre-filled and leave the send button to the user.
"""

from __future__ import annotations

import logging
import re
import urllib.parse

from .. import store, winutil
from ..permissions import Risk
from .registry import fail, ok, skill

log = logging.getLogger(__name__)


def _lookup_contact(name: str) -> str | None:
    """Resolve a saved contact name to a phone number.

    Contacts live in %LOCALAPPDATA%\\Jarvis\\store\\contacts.json as
    [{"name": "Amit", "phone": "919876543210"}].
    """
    from rapidfuzz import fuzz

    needle = (name or "").strip().lower()
    best, best_score = None, 0.0
    for contact in store.load("contacts", []):
        score = fuzz.WRatio(needle, str(contact.get("name", "")).lower())
        if score > best_score:
            best, best_score = contact, score
    if best and best_score >= 80:
        return re.sub(r"\D", "", str(best.get("phone", ""))) or None
    return None


@skill(
    name="send_whatsapp",
    description="Open WhatsApp with a message ready to send to a contact or number",
    risk=Risk.CRITICAL,
    category="messaging",
    params={"to": "Contact name or phone number with country code",
            "message": "The message text"},
    examples=[
        "send a whatsapp to amit saying i'm running late",
        "whatsapp karo mummy ko ki main aa raha hoon",
        "message rahul on whatsapp",
    ],
    confirm_en="Open WhatsApp to {to} with that message?",
    confirm_hi="{to} ko WhatsApp par ye message bhejne ke liye kholun?",
)
def send_whatsapp(to: str, message: str = "") -> object:
    target = (to or "").strip()
    if not target:
        return fail("Who should I message?", "Kise message karun?")

    digits = re.sub(r"\D", "", target)
    phone = digits if len(digits) >= 10 else _lookup_contact(target)

    if not phone:
        # No number: open WhatsApp itself so the user can pick the chat.
        winutil.spawn(["explorer.exe", "shell:AppsFolder\\5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"])
        return ok(
            f"I don't have a number for {target}, so I've opened WhatsApp for you.",
            f"{target} ka number nahi hai, isliye WhatsApp khol diya hai.",
            detail="no contact match",
        )

    if len(phone) == 10:
        phone = "91" + phone  # bare 10-digit numbers here are Indian

    url = f"https://wa.me/{phone}"
    if message.strip():
        url += "?text=" + urllib.parse.quote(message.strip())

    winutil.shell_open(url)
    return ok(
        "WhatsApp is open with the message ready — press send when you're happy with it.",
        "WhatsApp khol diya hai, message taiyaar hai — aap send daba dijiye.",
        detail=f"to={phone}", to=phone,
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
