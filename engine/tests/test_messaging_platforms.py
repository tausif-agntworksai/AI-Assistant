# -*- coding: utf-8 -*-
"""Naming the platform, and addressing people on it.

WhatsApp addresses people by phone number and Gmail by email address, so
"which app" and "who" stopped being independent questions. These tests pin the
three phrasings from the brief and the failures that used to sit behind them:
the platform being swallowed into the recipient's name, and a message body
that begins with "the" being mistaken for a question.
"""

import pytest

from jarvis import contacts, messaging
from jarvis.nlu.rules import route
from jarvis.skills import load_all

load_all()


@pytest.fixture(autouse=True)
def book(tmp_path, monkeypatch):
    monkeypatch.setattr(contacts.store.paths, "STORE_DIR", tmp_path)
    monkeypatch.setattr(contacts.store.paths, "ensure_dirs", lambda: None)
    monkeypatch.setattr(messaging, "_pending", None, raising=False)
    contacts.import_rows([
        {"name": "Sana Ahmed", "phone": "919876543210", "email": "sana@example.com"},
        {"name": "Rohit Sharma", "phone": "919000000001"},
    ])
    return contacts


# --- routing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "spoken, app, message",
    [
        ("text Sana on WhatsApp saying I'll be there in 10 minutes",
         "whatsapp", "I'll be there in 10 minutes"),
        ("email Sana saying I'll send the report tomorrow",
         "gmail", "I'll send the report tomorrow"),
        ("send a message to Sana on Google Chat saying I'm joining in 5 minutes",
         "google chat", "I'm joining in 5 minutes"),
        ("send an email to Sana saying the report is done",
         "gmail", "the report is done"),
    ],
)
def test_the_brief_phrasings_route_offline(spoken, app, message):
    """None of these should need the model to be understood."""
    intent = route(spoken)
    assert intent is not None and intent.skill == "send_message"
    assert intent.args["to"] == "sana"
    assert intent.args["app"] == app
    assert intent.args["message"] == message


def test_the_platform_is_not_swallowed_into_the_name():
    """"text Sana on WhatsApp ..." used to resolve a contact called
    "sana on whatsapp", which of course matched nobody."""
    intent = route("text Sana on WhatsApp saying hello")
    assert intent.args["to"] == "sana"


def test_a_body_may_begin_with_an_article():
    """"the deck is ready" is a sentence, not a question."""
    intent = route("email Ahmed saying the deck is ready")
    assert intent is not None
    assert intent.args["message"] == "the deck is ready"


@pytest.mark.parametrize("spoken", ["tell me a joke", "what is the time"])
def test_questions_are_still_not_messages(spoken):
    intent = route(spoken)
    assert intent is None or intent.skill != "send_message"


# --- addressing ------------------------------------------------------------


def test_gmail_resolves_a_contact_by_email():
    app = messaging.resolve_app("gmail")
    assert app.address_kind == "email"
    who = messaging.resolve_recipient("Sana", app.address_kind)
    assert who.found and who.email == "sana@example.com"


def test_whatsapp_still_resolves_by_number():
    app = messaging.resolve_app("whatsapp")
    who = messaging.resolve_recipient("Rohit", app.address_kind)
    assert who.phone == "919000000001"


def test_someone_with_no_email_is_not_an_email_candidate():
    """Matching them would produce a confident answer with nowhere to send."""
    assert messaging.resolve_recipient("Rohit", "email").found is False
    assert messaging.resolve_recipient("Rohit", "phone").found is True


def test_a_spoken_email_address_is_used_directly():
    who = messaging.resolve_recipient("someone@example.com", "email")
    assert who.email == "someone@example.com"


def test_a_gmail_link_carries_the_recipient_and_the_body():
    app = messaging.resolve_app("email")
    link = messaging.build_link(app, "sana@example.com", "running late")
    assert link.startswith("https://mail.google.com/mail/")
    assert "sana%40example.com" in link
    assert "running%20late" in link


def test_google_chat_opens_without_pretending_to_address_anyone():
    """Chat has no public per-person URL, so the honest thing is to open it."""
    app = messaging.resolve_app("google chat")
    assert app.deep_link is None
    assert messaging.build_link(app, "", "hello") == "https://mail.google.com/chat/u/0/"


def test_an_unknown_platform_still_sends():
    """Refusing over the transport helps nobody — who and what were understood."""
    assert messaging.resolve_app("hike").id == "whatsapp"
