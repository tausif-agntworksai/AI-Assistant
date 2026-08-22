"""Shared Windows helpers.

Every subprocess launched here uses CREATE_NO_WINDOW. Without it, each
PowerShell call would flash a console window on screen — dozens of times a
session for an assistant that shells out this often.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from typing import Any

log = logging.getLogger(__name__)

CREATE_NO_WINDOW = 0x08000000
DETACHED_PROCESS = 0x00000008


def _startupinfo() -> subprocess.STARTUPINFO | None:
    if os.name != "nt":
        return None
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return si


def run(
    args: list[str],
    timeout: float = 20.0,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command with no visible window.

    This is the single place the assistant reaches the command line, which is
    what makes "may it run system commands?" a question the user can be asked
    once and have enforced everywhere. Brightness, wi-fi, battery details,
    window snapping and the Store-app index all arrive through here.
    """
    from .permissions import Capability, PermissionDenied, consent

    if not consent.allows(Capability.SHELL):
        raise PermissionDenied(
            args[0] if args else "command",
            "running system commands is turned off",
        )

    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
        creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
        startupinfo=_startupinfo(),
    )


def powershell(script: str, timeout: float = 20.0) -> str:
    """Run a PowerShell snippet and return stdout ('' on failure)."""
    from .permissions import PermissionDenied

    try:
        proc = run(
            ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-Command", script],
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        log.warning("PowerShell timed out: %s", script[:80])
        return ""
    except PermissionDenied:
        # Callers here are read-only lookups (the app index, a battery detail).
        # Degrading to "I don't know" is the right answer when the user has
        # turned command-line access off; failing the whole skill is not.
        log.debug("PowerShell skipped — shell permission not granted")
        return ""
    if proc.returncode != 0:
        log.debug("PowerShell exit %d: %s", proc.returncode, (proc.stderr or "").strip()[:200])
    return (proc.stdout or "").strip()


def powershell_json(script: str, timeout: float = 20.0) -> Any:
    """Run PowerShell that emits JSON and parse it.

    Always returns a list for list-shaped results — PowerShell collapses a
    single-element array to a bare object, which would otherwise make callers
    branch on the result shape.
    """
    out = powershell(f"{script} | ConvertTo-Json -Compress -Depth 4", timeout=timeout)
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:
        log.debug("Could not parse PowerShell JSON: %s", exc)
        return []
    return data


def spawn(args: list[str], detached: bool = True) -> bool:
    """Launch a process without waiting for it. True if it started."""
    try:
        flags = CREATE_NO_WINDOW
        if detached and os.name == "nt":
            flags |= DETACHED_PROCESS
        subprocess.Popen(
            args,
            creationflags=flags if os.name == "nt" else 0,
            startupinfo=_startupinfo(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to spawn %s: %s", args[:1], exc)
        return False


def shell_open(target: str) -> bool:
    """Open a file, folder, URL or shell: path with its default handler.

    The single busiest chokepoint in the engine — every website, file, folder,
    settings page and WhatsApp chat goes through here, roughly fifteen of the
    seventy-two skills.

    `os.startfile` exists **only on Windows**, so off Windows this raised
    `AttributeError`, which is not an `OSError` and so escaped the handler
    below. Every one of those skills reported "That didn't work" from this one
    line. Hence `getattr` rather than a bare call: the absence of the function
    is a platform fact to branch on, not an error to catch.
    """
    startfile = getattr(os, "startfile", None)
    if startfile is not None:
        try:
            startfile(target)
            return True
        except OSError as exc:
            log.debug("os.startfile(%r) failed: %s; trying explorer", target, exc)
            return spawn(["explorer.exe", target])

    opener = "open" if sys.platform == "darwin" else "xdg-open"
    return spawn([opener, target], detached=False)


def foreground_window() -> dict[str, Any]:
    """Title, process name and handle of the focused window.

    Used as LLM context so commands like "isko band karo" ("close this") can be
    resolved to a concrete window.
    """
    try:
        import win32gui
        import win32process
        import psutil

        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return {}
        title = win32gui.GetWindowText(hwnd)
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            process = psutil.Process(pid).name()
        except Exception:  # noqa: BLE001
            process = ""
        return {"hwnd": hwnd, "title": title, "process": process, "pid": pid}
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not read foreground window: %s", exc)
        return {}


def list_windows(visible_only: bool = True) -> list[dict[str, Any]]:
    """Every top-level window with a title, plus its owning process."""
    try:
        import psutil
        import win32con
        import win32gui
        import win32process
    except ImportError:
        return []

    windows: list[dict[str, Any]] = []
    process_names: dict[int, str] = {}

    def callback(hwnd: int, _: Any) -> bool:
        if visible_only and not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title.strip():
            return True
        # Tool windows are palettes and popups, never what "switch to X" means.
        if win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE) & win32con.WS_EX_TOOLWINDOW:
            return True
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
        except Exception:  # noqa: BLE001
            pid = 0
        if pid and pid not in process_names:
            try:
                process_names[pid] = psutil.Process(pid).name()
            except Exception:  # noqa: BLE001
                process_names[pid] = ""
        windows.append({
            "hwnd": hwnd,
            "title": title,
            "pid": pid,
            "process": process_names.get(pid, ""),
            "minimized": bool(win32gui.IsIconic(hwnd)),
        })
        return True

    try:
        win32gui.EnumWindows(callback, None)
    except Exception as exc:  # noqa: BLE001
        log.debug("EnumWindows failed: %s", exc)
    return windows


def find_windows(query: str, min_score: int = 65) -> list[dict[str, Any]]:
    """Windows whose title or process name fuzzily matches `query`."""
    from rapidfuzz import fuzz

    needle = (query or "").strip().lower()
    if not needle:
        return []

    scored: list[tuple[float, dict[str, Any]]] = []
    for win in list_windows():
        title = win["title"].lower()
        process = win["process"].lower().removesuffix(".exe")
        score = max(
            fuzz.partial_ratio(needle, title),
            fuzz.WRatio(needle, process) if process else 0,
        )
        if score >= min_score:
            scored.append((score, win))

    scored.sort(key=lambda pair: -pair[0])
    return [win for _, win in scored]


def focus_window(hwnd: int) -> bool:
    """Bring a window to the front, restoring it if minimised."""
    try:
        import win32con
        import win32gui

        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        # Windows blocks foreground changes from background processes unless
        # the calling thread is attached to the current foreground thread.
        try:
            import win32process
            from ctypes import windll

            current = windll.kernel32.GetCurrentThreadId()
            target = win32process.GetWindowThreadProcessId(
                win32gui.GetForegroundWindow())[0]
            if current != target:
                windll.user32.AttachThreadInput(target, current, True)
                win32gui.SetForegroundWindow(hwnd)
                windll.user32.AttachThreadInput(target, current, False)
                return True
        except Exception:  # noqa: BLE001 - fall through to the plain call
            pass
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not focus window %s: %s", hwnd, exc)
        return False


def close_window(hwnd: int) -> bool:
    """Ask a window to close (WM_CLOSE), letting it save and prompt as usual."""
    try:
        import win32con
        import win32gui

        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not close window %s: %s", hwnd, exc)
        return False


def show_window(hwnd: int, command: str) -> bool:
    try:
        import win32con
        import win32gui

        flag = {
            "minimize": win32con.SW_MINIMIZE,
            "maximize": win32con.SW_MAXIMIZE,
            "restore": win32con.SW_RESTORE,
        }[command]
        win32gui.ShowWindow(hwnd, flag)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("ShowWindow(%s) failed: %s", command, exc)
        return False


def send_keys(*keys: str) -> bool:
    """Send a key combination (e.g. send_keys('win', 'd'))."""
    try:
        import pyautogui

        pyautogui.hotkey(*keys)
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("send_keys%s failed: %s", keys, exc)
        return False


def resolve_shortcut(lnk_path: str) -> tuple[str, str]:
    """Return (target_path, arguments) for a .lnk file."""
    try:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()
        shell = win32com.client.Dispatch("WScript.Shell")
        link = shell.CreateShortCut(lnk_path)
        return str(link.TargetPath or ""), str(link.Arguments or "")
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not resolve %s: %s", lnk_path, exc)
        return "", ""
