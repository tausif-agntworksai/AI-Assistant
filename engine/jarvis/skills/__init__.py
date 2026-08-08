"""Skills — everything the assistant can actually do.

Importing this package registers every skill. Each module runs its `@skill`
decorators on import, so `load_all()` is the single place that decides which
capabilities exist; nothing else maintains a list.
"""

from __future__ import annotations

import importlib
import logging

from .registry import (
    Reply,
    SkillContext,
    SkillResult,
    SkillSpec,
    fail,
    ok,
    registry,
    skill,
)

log = logging.getLogger(__name__)

__all__ = [
    "Reply", "SkillContext", "SkillResult", "SkillSpec",
    "fail", "ok", "registry", "skill", "load_all",
]

_MODULES = (
    "apps", "web", "system", "windows_mgr",
    "media", "volume", "display", "device",
    "files", "productivity", "messaging", "knowledge",
)

_loaded = False


def load_all(force: bool = False) -> int:
    """Import every skill module. Returns the number of registered skills."""
    global _loaded
    if _loaded and not force:
        return len(registry.all())

    for name in _MODULES:
        try:
            importlib.import_module(f".{name}", __package__)
        except Exception as exc:  # noqa: BLE001 - one bad module shouldn't
            # cost us every other capability
            log.error("Skill module %r failed to load: %s: %s",
                      name, type(exc).__name__, exc)

    _loaded = True
    count = len(registry.all())
    log.info("Loaded %d skills across %d categories",
             count, len(registry.by_category()))
    return count
