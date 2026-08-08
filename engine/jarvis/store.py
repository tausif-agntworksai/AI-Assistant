"""Small JSON-file store for reminders, notes and preferences.

Deliberately not a database: these are tens of records, they need to survive a
restart, and being hand-editable is a feature.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from . import paths

log = logging.getLogger(__name__)

_lock = threading.Lock()


def _file(name: str):
    paths.ensure_dirs()
    return paths.STORE_DIR / f"{name}.json"


def load(name: str, default: Any = None) -> Any:
    path = _file(name)
    if not path.exists():
        return [] if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.warning("Store %s unreadable (%s); starting fresh", name, exc)
        return [] if default is None else default


def save(name: str, data: Any) -> None:
    path = _file(name)
    with _lock:
        try:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(path)  # atomic: a crash mid-write can't corrupt the store
        except OSError as exc:
            log.error("Could not write store %s: %s", name, exc)


def append(name: str, record: dict[str, Any]) -> dict[str, Any]:
    items = load(name, [])
    record.setdefault("id", f"{int(time.time() * 1000):x}")
    record.setdefault("created", time.time())
    items.append(record)
    save(name, items)
    return record


def remove(name: str, record_id: str) -> bool:
    items = load(name, [])
    remaining = [i for i in items if i.get("id") != record_id]
    if len(remaining) == len(items):
        return False
    save(name, remaining)
    return True


def update(name: str, record_id: str, **changes: Any) -> bool:
    items = load(name, [])
    found = False
    for item in items:
        if item.get("id") == record_id:
            item.update(changes)
            found = True
    if found:
        save(name, items)
    return found
