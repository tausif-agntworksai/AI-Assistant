# -*- coding: utf-8 -*-
"""Sending a message: "say hi to sana".

Two failure modes here are different in kind from the rest of the assistant,
because they reach another person and cannot be taken back:

  * **the wrong recipient** — a fuzzy match on a name, sending something private
    to a stranger. This is why a tie between two contacts stops and asks rather
    than picking the higher score, and why the resolved name goes into the
    confirmation the user hears.
  * **a stray keystroke** — pressing send into whatever window happens to be
    focused. This is why the Enter is gated on the messaging app actually
    holding focus, and why failing that check leaves the draft alone.

Everything below is one of those two, the app/recipient parsing that feeds them,
or the rules that let the whole thing work offline.
"""

import pytest

from jarvis import messaging, store
from jarvis.nlu.rules import route
from jarvis.skills import load_all

load_all()


@pytest.fixture
def contacts(monkeypatch):
    """A contact list, without touching the user's real one."""

    def use(rows):
        monkeypatch.setattr(
            store, "load",
            lambda name, default=None: rows if name == "contacts" else default,
        )
        monkeypatch.setattr(
            messaging.store, "load",
            lambda name, default=None: rows if name == "contacts" else default,
        )

    return use


# --- which app ------------------------------------------------------------


def test_naming_no_app_uses_the_default():
    """"Say hi to sana" names no app, and in practice means WhatsApp."""
    assert messaging.resolve_app("").id == "whatsapp"


@pytest.mark.parametrize(
    ("spoken", "expected"),
    [
        ("telegram", "telegram"), ("Telegram", "telegram"), ("tg", "telegram"),
        ("whatsapp", "whatsapp"), ("whats app", "whatsapp"),
        ("watsapp", "whatsapp"), ("sms", "sms"), ("signal", "signal"),
    ],
)
def test_a_named_app_wins(spoken, expected):
    assert messaging.resolve_app(spoken).id == expected


def test_an_unknown_app_falls_back_rather_than_failing():
    """"Send hi to sana on hike" understood both the person and the message.
    Refusing over the transport helps nobody — it goes out on the default and
    the reply says which app it used."""
    assert messaging.resolve_app("hike").id == "whatsapp"


def test_the_default_is_configurable_not_hardcoded():
    assert messaging.resolve_app("", default="sms").id == "sms"


def test_the_rules_and_the_skill_agree_on_the_app_names():
    """The rule pattern is built from this same table, so a new app cannot be
    understood by one and not the other."""
    names = messaging.app_names()
    for app in messaging.APPS:
        for alias in app.aliases:
            assert alias.replace(" ", r"\ ") in names or alias in names


# --- which number ---------------------------------------------------------


def test_a_spoken_number_is_used_directly(contacts):
    contacts([])
    who = messaging.resolve_recipient("9876543210")
    assert who.phone == "919876543210", "a bare 10-digit number gets +91"


def test_a_number_that_already_has_a_country_code_is_left_alone(contacts):
    contacts([])
    assert messaging.resolve_recipient("919876543210").phone == "919876543210"


def test_a_saved_contact_is_found_by_first_name(contacts):
    contacts([{"name": "Sana Khan", "phone": "919876543210"}])
    who = messaging.resolve_recipient("sana")
    assert who.found and who.name == "Sana Khan"


def test_two_close_contacts_stop_and_ask(contacts):
    """The one failure that reaches a stranger. Sana and Sanjay both score
    against "sana", and picking the higher one sends a private message to
    whichever the fuzzy matcher happened to prefer."""
    contacts([
        {"name": "Sana", "phone": "919000000001"},
        {"name": "Sanaya", "phone": "919000000002"},
    ])
    who = messaging.resolve_recipient("sanaa")
    assert not who.found
    assert len(who.ambiguous) >= 2


def test_a_clear_winner_is_not_treated_as_a_tie(contacts):
    """Stopping to ask on every message would make the feature useless."""
    contacts([
        {"name": "Sana", "phone": "919000000001"},
        {"name": "Rajesh Kumar", "phone": "919000000002"},
    ])
    who = messaging.resolve_recipient("sana")
    assert who.found and not who.ambiguous


def test_an_unknown_name_is_not_forced_onto_the_nearest_contact(contacts):
    """The nearest contact to a name you don't have saved is still the wrong
    person."""
    contacts([{"name": "Rajesh", "phone": "919000000001"}])
    assert not messaging.resolve_recipient("priyanka").found


def test_a_contact_with_no_usable_number_is_skipped(contacts):
    contacts([{"name": "Sana", "phone": "123"}])
    assert not messaging.resolve_recipient("sana").found


def test_an_empty_contact_list_is_not_an_error(contacts):
    contacts([])
    who = messaging.resolve_recipient("sana")
    assert not who.found and not who.ambiguous


# --- the link -------------------------------------------------------------


def test_the_whatsapp_link_carries_the_number_and_the_text():
    app = messaging.resolve_app("whatsapp")
    link = messaging.build_link(app, "919876543210", "I'm running late")
    assert "919876543210" in link
    assert "running" in link and "%20" in link, "the text must be encoded"


def test_a_message_with_a_hash_survives_encoding():
    """Unencoded, everything after a # is dropped as a URL fragment — the
    message would arrive truncated."""
    app = messaging.resolve_app("whatsapp")
    link = messaging.build_link(app, "919876543210", "meeting #2 at 5")
    assert "#" not in link.split("?", 1)[1]


def test_an_app_that_cannot_address_a_person_falls_back_to_opening_itself():
    """Telegram addresses by @username, so a phone number cannot open a
    specific chat from outside. Staging the text is the honest maximum."""
    link = messaging.build_link(messaging.resolve_app("telegram"),
                                "919876543210", "hi")
    assert link.startswith("tg://")


# --- the rules ------------------------------------------------------------


@pytest.mark.parametrize(
    ("utterance", "to", "message"),
    [
        ("say hi to sana", "sana", "hi"),
        ("Say hi to Sana", "sana", "hi"),
        ("say happy birthday to amit", "amit", "happy birthday"),
        ("sana ko hi bol do", "sana", "hi"),
        ("mummy ko bol do ki main aa raha hoon", "mummy", "main aa raha hoon"),
        ("papa ko message bhejo ki khana kha liya", "papa", "khana kha liya"),
        ("whatsapp karo mummy ko ki main aa raha hoon", "mummy",
         "main aa raha hoon"),
    ],
)
def test_a_message_routes_offline(utterance, to, message):
    """No model round trip. Sending a message is exactly what someone does while
    their connection is down, and four words do not need a language model."""
    intent = route(utterance)
    assert intent is not None and intent.skill == "send_message"
    assert intent.args.get("to") == to
    assert intent.args.get("message") == message


def test_the_message_keeps_the_words_the_user_actually_said():
    """The body is quoted to another person, so it must survive normalisation:
    apostrophes closed up, filler stripped and everything lowercased is fine for
    matching a skill name and wrong for a sentence someone else will read."""
    intent = route("tell amit I'm running late")
    assert intent.args["message"] == "I'm running late"


def test_trailing_politeness_is_not_part_of_the_message():
    intent = route("Tell Sana I cannot make it, please")
    assert intent.args["message"] == "I cannot make it"


def test_a_named_app_reaches_the_skill():
    intent = route("send hi to sana on telegram")
    assert intent.args.get("app") == "telegram"


def test_no_message_still_routes_so_the_chat_can_be_opened():
    intent = route("message rahul")
    assert intent.skill == "send_message"
    assert intent.args.get("to") == "rahul" and "message" not in intent.args


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        # App control, not messaging. "whatsapp chalu karo" means open it.
        ("whatsapp chalu karo", "open_app"),
        ("whatsapp kholo", "open_app"),
        ("whatsapp band karo", "close_app"),
        ("chrome ko band kar do", "close_app"),
        # Questions, not messages.
        ("tell me the time", "get_time"),
        ("batao kitna baja hai", "get_time"),
        ("what is the battery", "get_battery"),
    ],
)
def test_what_must_not_become_a_message(utterance, expected):
    """Every one of these was caught by a messaging rule at some point during
    development. "whatsapp chalu karo" became a message to a contact called
    "chalu karo"."""
    intent = route(utterance)
    assert intent is not None and intent.skill == expected


def test_the_assistant_is_not_a_recipient():
    """"tell me a joke" is a question. Sending a joke to a contact called Me is
    the failure this guards."""
    intent = route("tell me a joke")
    assert intent is None or intent.skill != "send_message"


# --- the send keystroke ---------------------------------------------------


def test_send_is_only_pressed_once_the_app_holds_focus(monkeypatch):
    """The whole safety story. Without this the Enter goes wherever focus
    happens to be — another chat, another app, a terminal."""
    from jarvis.skills import messaging as skill

    monkeypatch.setattr(skill.winutil, "foreground_window",
                        lambda: {"process": "Notepad.exe"})
    assert skill._await_focus("WhatsApp.exe", timeout=0.4) is False


def test_focus_is_confirmed_twice(monkeypatch):
    """Focus can land and then be stolen while the chat pane is still painting,
    which is exactly when the pre-filled text is not there yet."""
    from jarvis.skills import messaging as skill

    seen = iter(["whatsapp.exe", "Notepad.exe"])
    monkeypatch.setattr(
        skill.winutil, "foreground_window",
        lambda: {"process": next(seen, "Notepad.exe")},
    )
    assert skill._await_focus("WhatsApp.exe", timeout=2.0) is False


def test_an_app_with_no_known_process_never_gets_a_keystroke(monkeypatch):
    """SMS opens the Messages app, whose process name is not pinned down here.
    Not knowing what to check for means not pressing anything."""
    from jarvis.skills import messaging as skill

    assert skill._await_focus("", timeout=0.2) is False


def test_a_tie_sends_nothing_at_all(contacts, monkeypatch):
    from jarvis.skills.messaging import send_message

    contacts([
        {"name": "Sana", "phone": "919000000001"},
        {"name": "Sanaya", "phone": "919000000002"},
    ])
    opened: list[str] = []
    monkeypatch.setattr(
        "jarvis.skills.messaging.winutil.shell_open",
        lambda target: (opened.append(target), True)[1],
    )
    result = send_message("sanaa", "hi")
    assert not result.ok
    assert opened == [], "nothing should have been opened"
    assert "Sana" in result.text("en")


def test_an_empty_message_is_refused_before_anything_opens(contacts):
    from jarvis.skills.messaging import send_message

    contacts([{"name": "Sana", "phone": "919000000001"}])
    result = send_message("sana", "")
    assert not result.ok and "what should i say" in result.text("en").lower()
