"""Timers, alarms, reminders and notes."""

from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timedelta

from ..announce import announce
from ..nlu.normalize import normalize, parse_number
from ..permissions import Risk
from .. import store
from .registry import fail, ok, skill

log = logging.getLogger(__name__)

_timers: dict[str, threading.Timer] = {}
_timers_lock = threading.Lock()


def _schedule(record_id: str, delay_sec: float, label: str) -> None:
    def fire() -> None:
        with _timers_lock:
            _timers.pop(record_id, None)
        store.update("timers", record_id, fired=True)
        if label:
            announce(f"Time's up — {label}.", f"Samay pura hua — {label}.", kind="timer")
        else:
            announce("Time's up.", "Samay pura ho gaya.", kind="timer")

    timer = threading.Timer(delay_sec, fire)
    timer.daemon = True
    with _timers_lock:
        _timers[record_id] = timer
    timer.start()


def restore_timers() -> int:
    """Re-arm timers that were still pending when the engine last stopped."""
    pending = [t for t in store.load("timers", []) if not t.get("fired")]
    restored = 0
    now = time.time()
    for record in pending:
        remaining = record.get("due", 0) - now
        if remaining <= 0:
            store.update("timers", record["id"], fired=True)
            continue
        _schedule(record["id"], remaining, record.get("label", ""))
        restored += 1
    if restored:
        log.info("Restored %d pending timer(s)", restored)
    return restored


@skill(
    name="set_timer",
    description="Set a countdown timer in minutes",
    risk=Risk.SAFE,
    category="productivity",
    params={"minutes": "How many minutes to count down",
            "label": "Optional description, e.g. 'tea'"},
    examples=[
        "set a timer for 5 minutes", "paanch minute ka timer laga do",
        "10 minute ka timer", "timer for 20 minutes", "das minute ka timer lagao",
    ],
)
def set_timer(minutes: int = 5, label: str = "") -> object:
    minutes = max(1, min(600, int(minutes)))
    due = time.time() + minutes * 60
    record = store.append("timers", {
        "minutes": minutes, "label": label, "due": due, "fired": False,
    })
    _schedule(record["id"], minutes * 60, label)

    suffix_en = f" for {label}" if label else ""
    suffix_hi = f" {label} ke liye" if label else ""
    return ok(
        f"Timer set{suffix_en} — {minutes} minutes.",
        f"{minutes} minute ka timer laga diya{suffix_hi}.",
        minutes=minutes, id=record["id"],
    )


@skill(
    name="set_alarm",
    description="Set an alarm for a specific clock time today or tomorrow",
    risk=Risk.SAFE,
    category="productivity",
    params={"when": "Clock time, e.g. '7:30 am', '18:00', 'saat baje'",
            "label": "Optional description"},
    examples=["set an alarm for 7 am", "alarm laga do 6 baje", "wake me at 5:30"],
)
def set_alarm(when: str, label: str = "") -> object:
    target = _parse_clock_time(when)
    if target is None:
        return fail(f"I couldn't work out what time {when} is.",
                    f"{when} ka matlab samajh nahi aaya.")

    delay = (target - datetime.now()).total_seconds()
    record = store.append("timers", {
        "minutes": round(delay / 60), "label": label or "alarm",
        "due": time.time() + delay, "fired": False,
    })
    _schedule(record["id"], delay, label or "alarm")

    spoken = target.strftime("%I:%M %p").lstrip("0")
    return ok(f"Alarm set for {spoken}.", f"{spoken} ka alarm laga diya.",
              at=spoken, id=record["id"])


def _parse_clock_time(text: str) -> datetime | None:
    """Parse '7:30 am', '18:00', '7 pm' or Hindi 'saat baje'."""
    raw = (text or "").strip().lower()
    now = datetime.now()

    m = re.search(r"(\d{1,2})[:.](\d{2})\s*(am|pm)?", raw)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        meridiem = m.group(3)
    else:
        m = re.search(r"(\d{1,2})\s*(am|pm)", raw)
        if m:
            hour, minute, meridiem = int(m.group(1)), 0, m.group(2)
        else:
            hour = parse_number(normalize(raw))
            if hour is None or not 1 <= hour <= 24:
                return None
            minute, meridiem = 0, None

    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None

    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)  # a time that already passed means tomorrow
    return target


@skill(
    name="list_timers",
    description="List timers and alarms that are still counting down",
    risk=Risk.SAFE,
    category="productivity",
    examples=["what timers are running", "kitne timer chal rahe hain", "list my alarms"],
)
def list_timers() -> object:
    now = time.time()
    pending = [t for t in store.load("timers", []) if not t.get("fired") and t.get("due", 0) > now]
    if not pending:
        return ok("No timers running.", "Koi timer nahi chal raha.")

    parts = []
    for t in pending[:5]:
        left = max(0, int((t["due"] - now) / 60))
        label = f" ({t['label']})" if t.get("label") else ""
        parts.append(f"{left} minutes{label}")
    listed = ", ".join(parts)
    return ok(f"{len(pending)} running: {listed}.", f"{len(pending)} chal rahe hain: {listed}.",
              count=len(pending))


@skill(
    name="cancel_timers",
    description="Cancel all running timers and alarms",
    risk=Risk.SAFE,
    category="productivity",
    examples=["cancel all timers", "timer cancel karo", "stop the alarm"],
)
def cancel_timers() -> object:
    with _timers_lock:
        count = len(_timers)
        for timer in _timers.values():
            timer.cancel()
        _timers.clear()

    items = store.load("timers", [])
    for item in items:
        item["fired"] = True
    store.save("timers", items)

    if not count:
        return ok("Nothing was running.", "Kuch chal hi nahi raha tha.")
    return ok(f"Cancelled {count} timer(s).", f"{count} timer cancel kar diye.")


@skill(
    name="add_reminder",
    description="Save a reminder, optionally due after a number of minutes",
    risk=Risk.SAFE,
    category="productivity",
    params={"text": "What to be reminded about",
            "minutes": "Optional: remind after this many minutes"},
    examples=[
        "remind me to call mom", "remind me to take a break in 30 minutes",
        "yaad dila do ki paani peena hai", "reminder set karo meeting ka",
    ],
)
def add_reminder(text: str, minutes: int = 0) -> object:
    content = (text or "").strip()
    if not content:
        return fail("What should I remind you about?", "Kis cheez ki yaad dilaun?")

    record = store.append("reminders", {"text": content, "done": False})

    if minutes and minutes > 0:
        due = time.time() + minutes * 60
        timer_record = store.append("timers", {
            "minutes": int(minutes), "label": content, "due": due, "fired": False,
        })
        _schedule(timer_record["id"], minutes * 60, content)
        return ok(f"I'll remind you in {minutes} minutes: {content}",
                  f"{minutes} minute baad yaad dila dunga: {content}",
                  id=record["id"])

    return ok(f"Saved: {content}", f"Yaad rakh liya: {content}", id=record["id"])


@skill(
    name="list_reminders",
    description="Read back saved reminders",
    risk=Risk.SAFE,
    category="productivity",
    examples=["what are my reminders", "reminders batao", "list my reminders"],
)
def list_reminders() -> object:
    items = [r for r in store.load("reminders", []) if not r.get("done")]
    if not items:
        return ok("You have no reminders.", "Koi reminder nahi hai.")
    listed = "; ".join(r["text"] for r in items[:6])
    return ok(f"You have {len(items)}: {listed}", f"{len(items)} reminder hain: {listed}",
              count=len(items))


@skill(
    name="clear_reminders",
    description="Clear all saved reminders",
    risk=Risk.CONFIRM,
    category="productivity",
    examples=["clear my reminders", "sab reminder hata do"],
    confirm_en="Clear all reminders?",
    confirm_hi="Saare reminders hata doon?",
)
def clear_reminders() -> object:
    count = len(store.load("reminders", []))
    store.save("reminders", [])
    return ok(f"Cleared {count} reminder(s).", f"{count} reminder hata diye.")


@skill(
    name="add_note",
    description="Save a quick note",
    risk=Risk.SAFE,
    category="productivity",
    params={"text": "The note to save"},
    examples=["note that the wifi password is abc123", "likh lo meeting kal 4 baje hai",
              "make a note about the project deadline"],
)
def add_note(text: str) -> object:
    content = (text or "").strip()
    if not content:
        return fail("What should I note down?", "Kya likhun?")
    store.append("notes", {"text": content})
    return ok("Noted.", "Likh liya.", detail=content[:80])


@skill(
    name="list_notes",
    description="Read back saved notes",
    risk=Risk.SAFE,
    category="productivity",
    examples=["read my notes", "notes batao", "what notes do i have"],
)
def list_notes() -> object:
    items = store.load("notes", [])
    if not items:
        return ok("You have no notes.", "Koi note nahi hai.")
    recent = items[-5:]
    listed = "; ".join(n["text"] for n in recent)
    return ok(f"Your last {len(recent)} notes: {listed}",
              f"Aapke aakhri {len(recent)} notes: {listed}", count=len(items))
