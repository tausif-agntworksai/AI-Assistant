# -*- coding: utf-8 -*-
"""Things that only break on the platform you aren't developing on.

Every test here corresponds to a row in docs/BUGS.md, and every one of them
would have passed on Windows while failing on macOS — which is exactly the
class of bug that gets rediscovered instead of fixed. They run on Windows by
pretending not to be on Windows: `sys.platform` is monkeypatched and the
Windows-only attributes are deleted, so the CI you actually have exercises the
CI you don't.

Named after symptoms, not fixes. When one of these fails, the failure message
should tell you what the user would have experienced.
"""

import builtins
import os
import sys
from pathlib import Path

import pytest

from jarvis import winutil


@pytest.fixture
def as_macos(monkeypatch):
    """Make the process look like macOS: no os.startfile, darwin platform."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delattr(os, "startfile", raising=False)
    return monkeypatch


@pytest.fixture
def spawned(monkeypatch):
    """Capture what winutil.spawn was asked to launch instead of launching it."""
    calls: list[list[str]] = []
    monkeypatch.setattr(winutil, "spawn",
                        lambda args, detached=True: calls.append(list(args)) or True)
    return calls


# --- winutil.shell_open ----------------------------------------------------


def test_opening_a_url_works_when_startfile_is_missing(as_macos, spawned):
    """The single busiest chokepoint in the engine.

    `os.startfile` is Windows-only and raises AttributeError, which is not an
    OSError — so it escaped the handler and every website, file, folder and
    settings page answered "That didn't work."
    """
    assert winutil.shell_open("https://example.com") is True
    assert spawned == [["open", "https://example.com"]]


def test_opening_a_folder_uses_xdg_open_on_linux(monkeypatch, spawned):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delattr(os, "startfile", raising=False)
    assert winutil.shell_open("/home/someone/Downloads") is True
    assert spawned == [["xdg-open", "/home/someone/Downloads"]]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows behaviour")
def test_windows_still_uses_startfile():
    """The fix must not change the platform it was already right on."""
    called: list[str] = []
    original = os.startfile  # type: ignore[attr-defined]
    os.startfile = called.append  # type: ignore[attr-defined]
    try:
        assert winutil.shell_open("C:\\Windows") is True
    finally:
        os.startfile = original  # type: ignore[attr-defined]
    assert called == ["C:\\Windows"]


# --- the TTS fallback chain ------------------------------------------------


def test_tts_falls_back_to_silence_not_to_a_speaker_that_cannot_speak(monkeypatch):
    """SAPI used to claim `available = True` on every platform.

    Nothing in that class imported anything until the first utterance, so off
    Windows `create_speaker` handed back a speaker that accepted every reply
    and played none of them. A user hears nothing and has no idea why;
    NullSpeaker at least reports itself as text-only.
    """
    from jarvis.tts import base, create_speaker

    real_import = builtins.__import__

    def no_pywin32(name, *args, **kwargs):
        if name in ("pythoncom", "win32com.client", "win32com"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pywin32)

    class Config:
        backend = "sapi"
        voice_hi = "hi-IN-MadhurNeural"
        voice_en = "en-IN-NeerjaNeural"
        rate = "+0%"
        volume = "+0%"
        barge_in = True

    speaker = create_speaker(Config())
    assert isinstance(speaker, base.NullSpeaker), (
        f"got {type(speaker).__name__}, which would swallow every reply silently"
    )


# --- filesystem locations --------------------------------------------------


def test_the_data_directory_is_not_a_bare_folder_in_the_home_directory():
    """`LOCALAPPDATA` is unset off Windows, and the old fallback dropped
    ~/Jarvis into the user's home directory along with 650 MB of models."""
    import importlib

    from jarvis import paths

    for platform, expected in (
        ("darwin", ("Library", "Application Support", "Jarvis")),
        ("linux", (".local", "share", "jarvis")),
    ):
        real_platform = sys.platform
        saved_env = os.environ.pop("JARVIS_DATA_DIR", None)
        try:
            sys.platform = platform  # type: ignore[misc]
            reloaded = importlib.reload(paths)
            parts = reloaded.DATA_DIR.parts
            assert parts[-len(expected):] == expected, (
                f"{platform}: got {reloaded.DATA_DIR}"
            )
        finally:
            sys.platform = real_platform  # type: ignore[misc]
            if saved_env is not None:
                os.environ["JARVIS_DATA_DIR"] = saved_env
            importlib.reload(paths)

    # And the test data dir is still honoured, or the whole suite would be
    # writing into the real one.
    assert "jarvis-tests" in str(paths.DATA_DIR)


def test_the_disk_report_does_not_assume_a_c_drive():
    """`psutil.disk_usage("C:\\\\")` raises off Windows, so `get_system_status`
    answered "That didn't work" rather than reporting anything."""
    from jarvis.skills import load_all, registry
    from jarvis.skills.registry import SkillContext

    load_all()
    result = registry.execute("get_system_status", {}, SkillContext())
    assert result.ok, result.reply.en
    assert "disk_free_gb" in result.data


def test_the_video_folder_is_called_movies_on_macos(monkeypatch):
    import importlib

    from jarvis.skills import files

    monkeypatch.setattr(sys, "platform", "darwin")
    reloaded = importlib.reload(files)
    try:
        assert reloaded.KNOWN_FOLDERS["videos"].name == "Movies"
        # Either word has to find it — people say both.
        assert reloaded.KNOWN_FOLDERS["movies"] == reloaded.KNOWN_FOLDERS["videos"]
        # And the trash is a real directory, not an Explorer-only shell: path.
        assert reloaded.TRASH == Path.home() / ".Trash"
    finally:
        monkeypatch.undo()
        importlib.reload(files)


def test_the_start_menu_scan_does_not_walk_the_working_directory(monkeypatch):
    """`Path(os.environ.get("APPDATA", ""))` is `Path(".")` off Windows, so the
    scan silently globbed the current directory instead of a Start Menu."""
    from jarvis.app_index import AppIndex

    monkeypatch.delenv("ProgramData", raising=False)
    monkeypatch.delenv("APPDATA", raising=False)
    assert AppIndex()._scan_start_menu() == []


# --- the doctor ------------------------------------------------------------


def test_the_doctor_can_pass_without_pywin32(monkeypatch, capsys):
    """`--doctor` listed win32api as required unconditionally, so it could
    never pass on macOS — and the build script gates on its exit code, which
    would have blocked every macOS build."""
    from jarvis.__main__ import cmd_doctor

    monkeypatch.setattr(sys, "platform", "darwin")
    cmd_doctor()
    report = capsys.readouterr().out
    assert "win32api" not in report
    assert "Platform darwin" in report


# --- the port the desktop app actually chose --------------------------------


def test_the_engine_binds_the_port_the_app_asked_for(monkeypatch):
    """The desktop app scans for a free port and passes it in JARVIS_PORT.

    Nothing read it. So on any machine where 8756 was already taken, the app
    would poll the port it picked while the engine sat on the one from
    config.yaml — and give up fifteen minutes later with "The engine did not
    respond in time." Nothing in that message points at a port conflict.
    """
    from jarvis.config import ServerConfig

    server = ServerConfig()
    monkeypatch.delenv("JARVIS_PORT", raising=False)
    assert server.bind_port == server.port

    monkeypatch.setenv("JARVIS_PORT", "8761")
    assert server.bind_port == 8761


@pytest.mark.parametrize("junk", ["", "  ", "not-a-port", "0", "70000", "-1", "80.5"])
def test_a_nonsense_port_falls_back_to_the_configured_one(monkeypatch, junk):
    """Better to listen somewhere predictable than to crash on a bad value."""
    from jarvis.config import ServerConfig

    monkeypatch.setenv("JARVIS_PORT", junk)
    assert ServerConfig().bind_port == 8756
