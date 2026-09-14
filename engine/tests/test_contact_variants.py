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
