"""Window management: minimise, maximise, close, snap, switch desktops."""

from __future__ import annotations

from .. import winutil
from ..permissions import Risk
from .registry import fail, ok, skill


def _foreground():
    info = winutil.foreground_window()
    return info if info.get("hwnd") else None


@skill(
    name="minimize_window",
    description="Minimise the window that currently has focus",
    risk=Risk.SAFE,
    category="windows",
    examples=["minimize this", "minimise the window", "window chhota karo", "niche karo"],
)
def minimize_window() -> object:
    win = _foreground()
    if not win:
        return fail("Nothing is focused.", "Koi window select nahi hai.")
    winutil.show_window(win["hwnd"], "minimize")
    return ok("Minimised.", "Chhota kar diya.", detail=win["title"][:60])


@skill(
    name="maximize_window",
    description="Maximise the window that currently has focus",
    risk=Risk.SAFE,
    category="windows",
    examples=["maximize this", "full screen", "window bada karo", "pura karo"],
)
def maximize_window() -> object:
    win = _foreground()
    if not win:
        return fail("Nothing is focused.", "Koi window select nahi hai.")
    winutil.show_window(win["hwnd"], "maximize")
    return ok("Maximised.", "Bada kar diya.", detail=win["title"][:60])


@skill(
    name="close_window",
    description="Close the window that currently has focus",
    risk=Risk.CONFIRM,
    category="windows",
    examples=["close this window", "close this", "ye window band karo", "isko band karo"],
    confirm_en="Close this window?",
    confirm_hi="Ye window band kar doon?",
)
def close_window() -> object:
    win = _foreground()
    if not win:
        return fail("Nothing is focused.", "Koi window select nahi hai.")
    title = win["title"][:60]
    if not winutil.close_window(win["hwnd"]):
        return fail("I couldn't close it.", "Band nahi kar paya.")
    return ok(f"Closed {title}.", f"{title} band kar diya.", detail=title)


@skill(
    name="minimize_all",
    description="Minimise every window and show the desktop",
    risk=Risk.SAFE,
    category="windows",
    examples=["show the desktop", "minimize everything", "sab minimize karo", "desktop dikhao"],
)
def minimize_all() -> object:
    winutil.send_keys("win", "d")
    return ok("Showing the desktop.", "Desktop dikha raha hoon.")


@skill(
    name="snap_window",
    description="Snap the focused window to one side of the screen",
    risk=Risk.SAFE,
    category="windows",
    params={"side": "Which side: left or right"},
    examples=["snap this left", "move window to the right", "window left me lagao"],
)
def snap_window(side: str = "left") -> object:
    direction = "left" if "l" in (side or "").lower()[:1] else "right"
    winutil.send_keys("win", direction)
    return ok(f"Snapped to the {direction}.", f"{direction} taraf laga diya.")


@skill(
    name="switch_desktop",
    description="Switch to the next or previous virtual desktop",
    risk=Risk.SAFE,
    category="windows",
    params={"direction": "next or previous"},
    examples=["next desktop", "switch desktop", "agla desktop", "previous desktop"],
)
def switch_desktop(direction: str = "next") -> object:
    key = "right" if "n" in (direction or "next").lower()[:1] else "left"
    winutil.send_keys("ctrl", "win", key)
    return ok("Switched desktop.", "Desktop badal diya.")


@skill(
    name="list_windows",
    description="List the windows that are currently open",
    risk=Risk.SAFE,
    category="windows",
    examples=["what windows are open", "list my windows", "kaunse window khule hain"],
)
def list_open_windows() -> object:
    windows = winutil.list_windows()
    if not windows:
        return ok("No windows are open.", "Koi window khuli nahi hai.")
    titles = [w["title"][:50] for w in windows[:8]]
    listed = "; ".join(titles)
    return ok(f"{len(windows)} windows open: {listed}",
              f"{len(windows)} window khuli hain: {listed}",
              windows=titles)
