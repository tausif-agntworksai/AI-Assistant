# -*- coding: utf-8 -*-
"""The contact book: importing it, merging it, and remembering the answers.

The matching was never the weak part — the book was empty. These tests cover
getting names into it from the formats people actually have, and the one piece
of memory the assistant keeps: who "Sana" turned out to mean.
"""

import pytest

from jarvis import contacts, messaging


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the JSON store at a temp directory for every test in this file."""
    monkeypatch.setattr(contacts.store.paths, "STORE_DIR", tmp_path)
    monkeypatch.setattr(contacts.store.paths, "ensure_dirs", lambda: None)
    monkeypatch.setattr(messaging, "_pending", None, raising=False)
    tmp_path.mkdir(parents=True, exist_ok=True)
    return tmp_path


# --- CSV ------------------------------------------------------------------

GOOGLE_CSV = (
    "Name,Given Name,Family Name,Phone 1 - Value,E-mail 1 - Value\n"
    "Sana Ahmed,Sana,Ahmed,+91 98765 43210,sana@example.com\n"
    "Rohit Sharma,Rohit,Sharma,+91 91234 56780,rohit@example.com\n"
)

OUTLOOK_CSV = (
    "First Name,Last Name,Mobile Phone,E-mail Address\n"
    "Amit,Khan,9876500001,amit@example.com\n"
)


def test_a_google_export_is_read():
    rows = contacts.parse_csv(GOOGLE_CSV)
    assert [r["name"] for r in rows] == ["Sana Ahmed", "Rohit Sharma"]
    assert rows[0]["phone"] == "+91 98765 43210"
    assert rows[0]["email"] == "sana@example.com"


def test_an_outlook_export_builds_the_name_from_two_columns():
    """Outlook has no single Name column, which is the whole difficulty."""
    rows = contacts.parse_csv(OUTLOOK_CSV)
    assert rows[0]["name"] == "Amit Khan"
    assert rows[0]["phone"] == "9876500001"


def test_a_vcard_export_is_read():
    vcf = (
        "BEGIN:VCARD\nVERSION:3.0\nFN:Sana Ahmed\n"
        "TEL;TYPE=CELL:+919876543210\nEMAIL:sana@example.com\nEND:VCARD\n"
        "BEGIN:VCARD\nVERSION:3.0\nFN:Mom\nTEL:+919000000002\nEND:VCARD\n"
    )
    rows = contacts.parse_vcard(vcf)
    assert [r["name"] for r in rows] == ["Sana Ahmed", "Mom"]
    assert rows[1]["phone"] == "+919000000002"


def test_a_contact_with_no_way_to_reach_them_is_dropped():
    """A name on its own reaches nobody, so it is not a contact."""
    assert contacts.import_rows([{"name": "Nobody"}]) == 0


def test_importing_replaces_rather_than_accumulates():
    """A contact deleted on the phone must not survive in here."""
    contacts.import_rows([{"name": "Old", "phone": "919000000001"}])
    contacts.import_rows([{"name": "New", "phone": "919000000002"}])
    assert [c.name for c in contacts.entries()] == ["New"]


def test_a_manually_saved_number_outranks_an_imported_one():
    """Someone who spelled a number out loud meant that number."""
    contacts.store.save(contacts.MANUAL,
                        [{"name": "Sana Ahmed", "phone": "919999999999"}])
    contacts.import_rows([{"name": "Sana Ahmed", "phone": "911111111111"}])
    found = [c for c in contacts.entries() if c.name == "Sana Ahmed"]
    assert len(found) == 1
    assert found[0].phone == "919999999999"


# --- Windows People -------------------------------------------------------


def test_the_windows_address_book_is_read(tmp_path):
    """One namespaced XML per person. The namespace URI varies by version."""
    folder = tmp_path / "Contacts"
    folder.mkdir()
    (folder / "Sana.contact").write_text(
        '<?xml version="1.0"?>'
        '<c:contact xmlns:c="http://schemas.microsoft.com/Contact">'
        "<c:NameCollection><c:Name><c:FormattedName>Sana Ahmed"
        "</c:FormattedName></c:Name></c:NameCollection>"
        "<c:PhoneNumberCollection><c:PhoneNumber><c:Number>+91 98765 43210"
        "</c:Number></c:PhoneNumber></c:PhoneNumberCollection>"
        "</c:contact>",
        encoding="utf-8",
    )
    rows = contacts.parse_windows_contacts(folder)
    assert rows == [{"name": "Sana Ahmed", "phone": "+91 98765 43210", "email": ""}]


def test_a_missing_address_book_is_not_an_error(tmp_path):
    assert contacts.parse_windows_contacts(tmp_path / "nope") == []


def test_an_unreadable_contact_does_not_lose_the_rest(tmp_path):
    folder = tmp_path / "Contacts"
    folder.mkdir()
    (folder / "broken.contact").write_text("not xml at all", encoding="utf-8")
    (folder / "ok.contact").write_text(
        '<?xml version="1.0"?><contact><FormattedName>Mom</FormattedName>'
        "<Number>919000000002</Number></contact>",
        encoding="utf-8",
    )
    assert [r["name"] for r in contacts.parse_windows_contacts(folder)] == ["Mom"]


# --- remembering ----------------------------------------------------------


def test_a_remembered_choice_is_used_next_time():
    contacts.import_rows([
        {"name": "Sana Ahmed", "phone": "919876543210"},
        {"name": "Sana Khan", "phone": "919876543211"},
    ])
    # Without a memory these two are a tie, and the user is asked.
    assert messaging.resolve_recipient("Sana").ambiguous

    contacts.remember("Sana", "Sana Ahmed")
    who = messaging.resolve_recipient("Sana")
    assert who.found and who.name == "Sana Ahmed"


def test_answering_the_question_teaches_the_book():
    """The point of section 18: asked once, never again."""
    contacts.import_rows([
        {"name": "Sana Ahmed", "phone": "919876543210"},
        {"name": "Sana Khan", "phone": "919876543211"},
    ])
    assert messaging.resolve_recipient("Sana").ambiguous     # "which Sana?"
    assert messaging.resolve_recipient("Sana Ahmed").found   # "Sana Ahmed"

    assert contacts.learned("Sana") == "Sana Ahmed"
    assert messaging.resolve_recipient("Sana").name == "Sana Ahmed"


def test_an_unrelated_next_request_teaches_nothing():
    """Walking away from the question must not record a wrong answer."""
    contacts.import_rows([
        {"name": "Sana Ahmed", "phone": "919876543210"},
        {"name": "Sana Khan", "phone": "919876543211"},
        {"name": "Rohit Sharma", "phone": "919876543212"},
    ])
    assert messaging.resolve_recipient("Sana").ambiguous
    assert messaging.resolve_recipient("Rohit").found
    assert contacts.learned("Sana") == ""


def test_forgetting_a_choice():
    contacts.remember("Sana", "Sana Ahmed")
    contacts.forget("Sana")
    assert contacts.learned("Sana") == ""


def test_remembering_a_name_as_itself_is_not_recorded():
    """It would be a no-op entry that only grows the file."""
    contacts.remember("Sana Ahmed", "Sana Ahmed")
    assert contacts.learned("Sana Ahmed") == ""
