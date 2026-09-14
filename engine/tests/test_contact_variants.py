# -*- coding: utf-8 -*-
"""Finding a person whatever their name is saved as.

Phone books are not written to be spoken. The same person is "Sana Ahmed" in
one app, "sana [heart]" in another, "Sana(Office)" at work and Devanagari on a
Hindi handset -- and nobody pronounces a heart. Scoring against the stored
spelling alone put "sana" at 50 against "sana(office)", far under the floor,
so a contact visible in the user's own phone could not be found at all.
"""

import pytest

from jarvis import contacts, messaging

HEART = "\u2764\ufe0f"


@pytest.fixture(autouse=True)
def book(tmp_path, monkeypatch):
    monkeypatch.setattr(contacts.store.paths, "STORE_DIR", tmp_path)
    monkeypatch.setattr(contacts.store.paths, "ensure_dirs", lambda: None)
    monkeypatch.setattr(messaging, "_pending", None, raising=False)
    return contacts


def only(name: str):
    """A book holding exactly one person, saved as `name`."""
    contacts.import_rows([{"name": name, "phone": "919876543210"}])
    return messaging.resolve_recipient("sana")


@pytest.mark.parametrize("stored", [
    "Sana",
    "Sana Ahmed",
    f"sana {HEART}",
    f"{HEART} Sana {HEART}",
    f"Sana{HEART}Ahmed",
    "Sana(Office)",
    "Sana_Ahmed",
    "Sana-Work",
    "Sana.Ahmed",
    "SANA AHMED OFFICE",
    "Sana ji",
    "\u0938\u0928\u093e",            # सना, the same name in Devanagari
])
def test_one_sana_is_found_however_she_is_saved(stored):
    """The user should never have to know how a contact was spelled."""
    who = only(stored)
    assert who.found, f"{stored!r} was not reachable by saying 'sana'"
    assert who.name == stored


def test_the_full_name_settles_a_genuine_tie():
    """Saying more must narrow, not widen.

    A subset scores 100 under token_set_ratio in both directions, so "sana
    ahmed" tied with a contact saved as plain "sana" and asked which was
    meant -- punishing the user for being precise.
    """
    contacts.import_rows([
        {"name": "Sana Ahmed", "phone": "919876500001"},
        {"name": f"sana {HEART}", "phone": "919876500002"},
    ])
    assert messaging.resolve_recipient("sana").ambiguous      # two real Sanas
    assert messaging.resolve_recipient("sana ahmed").name == "Sana Ahmed"


def test_a_surname_alone_still_finds_her():
    contacts.import_rows([{"name": "Sana Ahmed", "phone": "919876500001"}])
    assert messaging.resolve_recipient("ahmed").name == "Sana Ahmed"


@pytest.mark.parametrize("spoken, expected", [
    ("sanjay", "Sanjay Gupta"),
    ("rohit", "Rohit Sharma"),
    ("rohan", "Rohan Mehta"),
])
def test_similar_names_are_still_told_apart(spoken, expected):
    """Normalising must not blur people into each other."""
    contacts.import_rows([
        {"name": "Sana Ahmed", "phone": "919876500001"},
        {"name": "Sanjay Gupta", "phone": "919876500002"},
        {"name": "Rohit Sharma", "phone": "919876500003"},
        {"name": "Rohan Mehta", "phone": "919876500004"},
    ])
    assert messaging.resolve_recipient(spoken).name == expected


def test_a_stranger_is_still_not_forced_onto_the_nearest_contact():
    contacts.import_rows([{"name": "Sana Ahmed", "phone": "919876500001"}])
    assert messaging.resolve_recipient("nobody").found is False


def test_a_remembered_choice_survives_decoration():
    """The remembered name is matched on the same key it was stored under."""
    contacts.import_rows([
        {"name": f"sana {HEART}", "phone": "919876500002"},
        {"name": "Sana Khan", "phone": "919876500003"},
    ])
    contacts.remember("sana", f"sana {HEART}")
    who = messaging.resolve_recipient("sana")
    assert who.found and who.name == f"sana {HEART}"


@pytest.mark.parametrize("stored, key", [
    ("Sana(Office)", "sana office"),
    ("Sana_Ahmed", "sana ahmed"),
    (f"sana {HEART}", "sana"),
    ("\u0938\u0928\u093e", "sana"),
    ("Dr. Patel", "dr patel"),
])
def test_the_match_key_is_what_a_person_would_say(stored, key):
    assert messaging.match_key(stored) == key


# --- repairing an empty contact book, offline ------------------------------
#
# This is the command that gets the first number in, so it is the one command
# that must not need a working model key: an assistant that can only learn a
# contact when it has cloud access cannot be set up offline at all.


@pytest.mark.parametrize("spoken, name, phone", [
    ("save sana's number as 9876543210", "sana", "9876543210"),
    ("save sana ka number 9876543210", "sana", "9876543210"),
    ("sana ka number save karo 9876543210", "sana", "9876543210"),
    ("add contact sana 9876543210", "sana", "9876543210"),
    ("store rohit number 9876500001", "rohit", "9876500001"),
    ("save mom's number as +91 98765 43210", "mom", "919876543210"),
])
def test_saving_a_number_routes_offline(spoken, name, phone):
    from jarvis.nlu.rules import route

    intent = route(spoken)
    assert intent is not None and intent.skill == "add_contact"
    assert intent.args == {"name": name, "phone": phone}


def test_a_name_that_really_ends_in_s_is_not_truncated():
    """The possessive is recovered from the raw utterance, not guessed at by
    chopping a trailing letter."""
    from jarvis.nlu.rules import route

    assert route("add contact charles 9876543210").args["name"] == "charles"


def test_saving_a_number_is_not_read_as_a_message():
    """Without ordering, "save sana's number as 9876543210" parses as a
    message to Sana whose body is her own phone number."""
    from jarvis.nlu.rules import route

    assert route("save sana's number as 9876543210").skill == "add_contact"


@pytest.mark.parametrize("spoken", [
    "save sana's number", "add contact sana", "save sana ka number 12345",
])
def test_an_incomplete_number_is_not_saved(spoken):
    """Half a phone number stored under a name is worse than no contact."""
    from jarvis.nlu.rules import route

    intent = route(spoken)
    assert intent is None or intent.skill != "add_contact"


def test_saving_then_messaging_closes_the_loop(tmp_path, monkeypatch):
    """The whole point: one spoken command makes the next one work."""
    from jarvis.skills import load_all, registry
    from jarvis.skills.registry import SkillContext

    load_all()
    monkeypatch.setattr(contacts.store.paths, "STORE_DIR", tmp_path)
    monkeypatch.setattr(contacts.store.paths, "ensure_dirs", lambda: None)

    assert messaging.resolve_recipient("sana").found is False
    registry.execute("add_contact", {"name": "sana", "phone": "919876543210"},
                     SkillContext())
    who = messaging.resolve_recipient("sana")
    assert who.found and who.phone == "919876543210"
