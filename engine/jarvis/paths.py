"""Filesystem locations.

Everything the engine writes at runtime lives under one root so that
uninstalling is a single directory delete, and so a packaged (PyInstaller)
build never tries to write next to its own frozen executable.
"""

from __future__ import annotations

import os
import sys
import sys as _sys
from pathlib import Path


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


# Repo root when running from source; alongside the exe when frozen.
ENGINE_DIR: Path = (
    Path(sys.executable).resolve().parent
    if _is_frozen()
    else Path(__file__).resolve().parent.parent
)

PROJECT_DIR: Path = ENGINE_DIR.parent


def _default_data_dir() -> Path:
    """Where this OS expects an application to keep its own data.

    Falling back to `Path.home() / "Jarvis"` when `LOCALAPPDATA` is unset —
    which is every non-Windows machine — would have dropped a bare `~/Jarvis`
    into the user's home directory and left the models there.
    """
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Jarvis"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Jarvis"
    # Linux and the rest: the XDG base directory spec.
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "jarvis"


# Writable runtime root. Overridable for tests via JARVIS_DATA_DIR.
DATA_DIR: Path = Path(os.environ.get("JARVIS_DATA_DIR") or _default_data_dir())

MODELS_DIR: Path = DATA_DIR / "models"
LOGS_DIR: Path = DATA_DIR / "logs"
CACHE_DIR: Path = DATA_DIR / "cache"
STORE_DIR: Path = DATA_DIR / "store"  # reminders, notes, preferences

CONFIG_EXAMPLE: Path = ENGINE_DIR / "config.example.yaml"


def _user_file(name: str) -> Path:
    """Where a user-editable file lives.

    Running from source, that's the repo (next to config.example.yaml) so the
    checked-out tree stays self-contained. In an installed build it's the data
    directory instead — nobody should have to dig into the install folder to
    paste an API key, and that folder may not even be writable.
    """
    if _is_frozen():
        in_data = DATA_DIR / name
        if in_data.exists():
            return in_data
        beside_exe = ENGINE_DIR / name
        return beside_exe if beside_exe.exists() else in_data
    return ENGINE_DIR / name


CONFIG_FILE: Path = _user_file("config.yaml")
ENV_FILE: Path = _user_file(".env")

APP_INDEX_CACHE: Path = CACHE_DIR / "app_index.json"
AUDIT_LOG: Path = LOGS_DIR / "audit.jsonl"
ENGINE_LOG: Path = LOGS_DIR / "engine.log"


def pictures_dir() -> Path:
    """The user's Pictures folder, by whatever name this OS gives it.

    Not under DATA_DIR: a photo the user asked for is theirs, and belongs
    somewhere they would look for it rather than in an application cache they do
    not know exists.
    """
    home = Path.home()
    if _sys.platform == "win32":
        # The folder can be redirected (OneDrive does this by default), and the
        # registry is the only place that knows where it actually went.
        try:
            import winreg

            key = (r"Software\Microsoft\Windows\CurrentVersion"
                   r"\Explorer\Shell Folders")
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as handle:
                value, _ = winreg.QueryValueEx(handle, "My Pictures")
            if value:
                return Path(value)
        except Exception:  # noqa: BLE001 - fall through to the usual place
            pass
    return home / "Pictures"


def ensure_dirs() -> None:
    """Create every writable directory. Safe to call repeatedly."""
    for d in (DATA_DIR, MODELS_DIR, LOGS_DIR, CACHE_DIR, STORE_DIR):
        d.mkdir(parents=True, exist_ok=True)
