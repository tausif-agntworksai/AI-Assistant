"""Where the names the user says come from.

The matching in `messaging.py` was always sound; what it lacked was anything to
match against. Contacts could only arrive one at a time, by voice, through the
`save_contact` skill — so on a fresh install every "text Sana" failed, and the
fuzzy scoring, the tie window and the confidence floor never got to do their
job. This module is the other half: the book itself.

**Imports are explicit and persisted, not scanned live.** Reading the address
book on every utterance would put a disk walk in the latency budget of a
command that is supposed to feel instant, and would make what the assistant
knows depend on what a scan happened to find that second. So an import is a
thing the user does once; resolution afterwards reads one merged list.

Sources are kept apart in storage and merged on read, so re-importing a phone
export can never overwrite a number the user saved by voice — manual entries
win every collision, because a person who spelled a number out loud meant it.

**The book also remembers.** The first time "Sana" has to be disambiguated,
the answer is written down; after that "Sana" resolves to Sana Ahmed without
asking again. That is the whole of the learning here, and it is deliberately
this small — a mapping from what someone says to who they meant is a
preference, and preferences belong in a file you can read and delete.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from . import store

log = logging.getLogger(__name__)

#: Manually saved contacts. Written by the `save_contact` skill, one at a time.
MANUAL = "contacts"
#: Everything that arrived from an address book. Replaced wholesale on import.
IMPORTED = "contacts_imported"
#: What the user meant, the last time we had to ask. {spoken: canonical name}
ALIASES = "contact_aliases"


@dataclass(frozen=True)
class Contact:
    name: str
    phone: str = ""
    email: str = ""
    source: str = ""

    @property
    def usable(self) -> bool:
        """A contact we can actually address. A name alone reaches nobody."""
        return bool(self.name and (self.phone or self.email))


def _clean(value: object) -> str:
    return str(value or "").strip()


def entries() -> list[Contact]:
    """Every contact, manual ones first so they win collisions.

    De-duplicated on name, not on number: the same person often appears in an
    export twice with the number written two different ways, and the first
    spelling is as good as the second.
    """
    out: list[Contact] = []
    seen: set[str] = set()
    for origin, rows in ((MANUAL, store.load(MANUAL, [])),
                         (IMPORTED, store.load(IMPORTED, []))):
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            contact = Contact(
                name=_clean(row.get("name")),
                phone=_clean(row.get("phone")),
                email=_clean(row.get("email")),
                source=_clean(row.get("source")) or origin,
            )
            key = contact.name.lower()
            if not contact.name or key in seen:
                continue
            seen.add(key)
            out.append(contact)
    return out


# --- what the user meant last time -----------------------------------------


def learned(spoken: str) -> str:
    """The contact this spoken name was resolved to before, if any."""
    table = store.load(ALIASES, {})
    if not isinstance(table, dict):
        return ""
    return _clean(table.get(_clean(spoken).lower()))


def remember(spoken: str, name: str) -> None:
    """Record that "Sana" meant Sana Ahmed, so we stop asking."""
    spoken, name = _clean(spoken).lower(), _clean(name)
    if not spoken or not name or spoken == name.lower():
        return
    table = store.load(ALIASES, {})
    if not isinstance(table, dict):
        table = {}
    if table.get(spoken) == name:
        return
    table[spoken] = name
    store.save(ALIASES, table)
    log.info("Remembered %r -> %r", spoken, name)


def forget(spoken: str = "") -> None:
    """Drop one learned mapping, or all of them."""
    if not spoken:
        store.save(ALIASES, {})
        return
    table = store.load(ALIASES, {})
    if isinstance(table, dict) and table.pop(_clean(spoken).lower(), None) is not None:
        store.save(ALIASES, table)


# --- importing --------------------------------------------------------------


def import_rows(rows: list[dict], source: str = "import") -> int:
    """Replace the imported book. Returns how many usable contacts landed.

    Replace rather than merge: an address book is a snapshot of the truth, and
    a contact deleted on the phone should not survive here because an older
    import still remembers it. Manually saved numbers live in a different file
    and are untouched by this.
    """
    keep: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        contact = Contact(
            name=_clean(row.get("name")),
            phone=_clean(row.get("phone")),
            email=_clean(row.get("email")),
            source=source,
        )
        key = contact.name.lower()
        if not contact.usable or key in seen:
            continue
        seen.add(key)
        keep.append({"name": contact.name, "phone": contact.phone,
                     "email": contact.email, "source": source})
    store.save(IMPORTED, keep)
    log.info("Imported %d contacts from %s", len(keep), source)
    return len(keep)


# Column names the common exports use. Google Contacts, Outlook and WhatsApp
# all write CSV, and all three spell every one of these differently.
_NAME_COLUMNS = ("name", "full name", "display name")
_FIRST_COLUMNS = ("first name", "given name")
_LAST_COLUMNS = ("last name", "family name", "surname")
_PHONE_COLUMNS = ("phone", "mobile", "telephone")
_EMAIL_COLUMNS = ("email", "e-mail")


def _pick(row: dict, exact: tuple, fuzzy: tuple | None = None) -> str:
    """First non-empty column whose header matches, headers normalised.

    Exact names are tried before substrings so that "Phone 1 - Value" doesn't
    beat a plain "Phone" column, and so an export with both "Email" and
    "Email 2" answers with the first one.

    `fuzzy` defaults to the exact names; pass `()` to switch the substring pass
    off entirely. That matters for the name column, where a loose "name" would
    match Outlook's "First Name" and return half of it — the two halves have to
    be found separately and joined.
    """
    lowered = {_clean(k).lower(): _clean(v) for k, v in row.items() if k}
    for candidate in exact:
        if lowered.get(candidate):
            return lowered[candidate]
    for word in (exact if fuzzy is None else fuzzy):
        for key, value in lowered.items():
            if value and word in key:
                return value
    return ""


def parse_csv(text: str) -> list[dict]:
    """Google Contacts, Outlook and WhatsApp exports are all CSV."""
    rows: list[dict] = []
    for raw in csv.DictReader(text.splitlines()):
        first = _pick(raw, _FIRST_COLUMNS)
        last = _pick(raw, _LAST_COLUMNS)
        name = _pick(raw, _NAME_COLUMNS, ()) or " ".join(p for p in (first, last) if p)
        rows.append({
            "name": name,
            "phone": _pick(raw, _PHONE_COLUMNS, _PHONE_COLUMNS),
            "email": _pick(raw, _EMAIL_COLUMNS, _EMAIL_COLUMNS),
        })
    return rows


_VCARD_NAME = re.compile(r"^FN[;:]", re.IGNORECASE)
_VCARD_TEL = re.compile(r"^TEL[;:]", re.IGNORECASE)
_VCARD_EMAIL = re.compile(r"^EMAIL[;:]", re.IGNORECASE)


def parse_vcard(text: str) -> list[dict]:
    """iPhone, Android and Google all export .vcf as well."""
    rows: list[dict] = []
    current: dict = {}
    for line in text.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("BEGIN:VCARD"):
            current = {"name": "", "phone": "", "email": ""}
        elif upper.startswith("END:VCARD"):
            if current.get("name"):
                rows.append(current)
            current = {}
        elif not current:
            continue
        elif _VCARD_NAME.match(line):
            current["name"] = line.split(":", 1)[-1].strip()
        elif _VCARD_TEL.match(line) and not current.get("phone"):
            current["phone"] = line.split(":", 1)[-1].strip()
        elif _VCARD_EMAIL.match(line) and not current.get("email"):
            current["email"] = line.split(":", 1)[-1].strip()
    return rows


def parse_windows_contacts(folder: Path) -> list[dict]:
    """The Windows address book: one XML file per person under ~/Contacts.

    This is what the People app and Outlook write to on a desktop, and it needs
    no authentication and no network — which is why it is the one source that
    can be imported without the user exporting anything first.
    """
    import xml.etree.ElementTree as ET

    rows: list[dict] = []
    if not folder.exists():
        return rows

    for path in sorted(folder.glob("*.contact")):
        try:
            root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as exc:
            log.debug("Skipping unreadable contact %s: %s", path.name, exc)
            continue

        # The schema namespaces every tag, and the namespace URI has changed
        # between Windows versions — matching on the local name is what keeps
        # this working across them.
        def first(tag: str, _root=root) -> str:
            for node in _root.iter():
                if node.tag.rsplit("}", 1)[-1] == tag and _clean(node.text):
                    return _clean(node.text)
            return ""

        address = first("Address")
        rows.append({
            "name": first("FormattedName") or path.stem,
            "phone": first("Number"),
            "email": address if "@" in address else "",
        })
    return rows


def import_file(path: str | Path) -> int:
    """Import a .csv or .vcf export. Returns how many contacts landed."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    rows = (parse_vcard(text) if path.suffix.lower() in (".vcf", ".vcard")
            else parse_csv(text))
    return import_rows(rows, source=path.name)


def import_windows_people(folder: Path | None = None) -> int:
    """Import the local Windows address book. No export, no sign-in."""
    folder = folder or (Path.home() / "Contacts")
    return import_rows(parse_windows_contacts(folder), source="windows-people")
