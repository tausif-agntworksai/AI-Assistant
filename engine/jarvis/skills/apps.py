"""Launching, closing and switching applications."""

from __future__ import annotations

import logging

from .. import winutil
from ..app_index import app_index
from ..permissions import Risk
from .registry import SkillContext, fail, ok, skill

log = logging.getLogger(__name__)


def _launch(entry) -> bool:
    """Launch an index entry according to its kind.

    UWP apps are the reason this isn't just `os.startfile`: Store apps have no
    executable path, only an AppUserModelID that must go through Explorer.
    """
    if entry.kind == "uwp":
        return winutil.spawn(["explorer.exe", entry.launch])
    return winutil.shell_open(entry.launch)


@skill(
    name="open_app",
    description="Open or launch an application, program or website by name",
    risk=Risk.SAFE,
    category="apps",
    params={"app": "Name of the app or site, e.g. chrome, whatsapp, youtube, vs code"},
    examples=[
        "open chrome", "chrome kholo", "launch whatsapp", "whatsapp chalu karo",
        "open youtube", "youtube kholo", "start spotify", "vs code kholo",
        "calculator kholo", "open file explorer", "notepad chalu karo",
    ],
)
def open_app(app: str) -> object:
    query = (app or "").strip()
    if not query:
        return fail("Which app should I open?", "Kaunsa app kholun?")

    entry = app_index.resolve(query)
    if entry is None:
        app_index.build(force=True)  # maybe it was installed since we last looked
        entry = app_index.resolve(query)

    if entry is None:
        return fail(
            f"I couldn't find an app called {query}.",
            f"{query} naam ka koi app nahi mila.",
            detail=f"no index match for {query!r}",
        )

    candidates = app_index.ambiguous(query)
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates[:3])
        return fail(
            f"Did you mean {names}?",
            f"Aapka matlab {names} me se kaunsa hai?",
            detail=f"ambiguous: {names}",
        )

    if not _launch(entry):
        return fail(f"I couldn't start {entry.name}.", f"{entry.name} shuru nahi ho paya.")

    verb = "Opening" if entry.kind != "web" else "Opening"
    return ok(
        f"{verb} {entry.name}.",
        f"{entry.name} khol raha hoon.",
        detail=f"{entry.kind}: {entry.launch}",
        app=entry.name,
        kind=entry.kind,
    )


@skill(
    name="close_app",
    description="Close a running application",
    risk=Risk.CONFIRM,
    category="apps",
    params={"app": "Name of the app to close"},
    examples=[
        "close chrome", "chrome band karo", "quit spotify", "whatsapp band kar do",
        "close notepad", "exit vs code",
    ],
    confirm_en="Close {app}?",
    confirm_hi="{app} band kar doon?",
)
def close_app(app: str) -> object:
    query = (app or "").strip()
    if not query:
        return fail("Which app should I close?", "Kaunsa app band karun?")

    windows = winutil.find_windows(query)
    if windows:
        closed = sum(1 for w in windows if winutil.close_window(w["hwnd"]))
        if closed:
            name = windows[0]["process"].removesuffix(".exe") or query
            return ok(
                f"Closed {name}.",
                f"{name} band kar diya.",
                detail=f"closed {closed} window(s)",
                closed=closed,
            )

    # No window matched — fall back to matching the process name, which covers
    # background or tray-only apps.
    terminated = _terminate_processes(query)
    if terminated:
        return ok(f"Closed {query}.", f"{query} band kar diya.",
                  detail=f"terminated {terminated} process(es)")

    return fail(f"{query} doesn't seem to be running.", f"{query} chal hi nahi raha.")


def _terminate_processes(query: str) -> int:
    from rapidfuzz import fuzz

    import psutil

    needle = query.lower().replace(" ", "")
    count = 0
    for proc in psutil.process_iter(["name", "pid"]):
        name = (proc.info.get("name") or "").lower().removesuffix(".exe")
        if not name:
            continue
        if fuzz.WRatio(needle, name) < 88:
            continue
        try:
            proc.terminate()  # graceful; never kill()
            count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            log.debug("Could not terminate %s: %s", name, exc)
    return count


@skill(
    name="switch_to_app",
    description="Switch focus to an already-running application window",
    risk=Risk.SAFE,
    category="apps",
    params={"app": "Name of the app to bring to the front"},
    examples=[
        "switch to chrome", "go to whatsapp", "chrome pe jao", "show me vs code",
        "switch to spotify",
    ],
)
def switch_to_app(app: str) -> object:
    query = (app or "").strip()
    if not query:
        return fail("Switch to what?", "Kispe jaun?")

    windows = winutil.find_windows(query)
    if not windows:
        return fail(f"{query} isn't open.", f"{query} khula hi nahi hai.")

    win = windows[0]
    if not winutil.focus_window(win["hwnd"]):
        return fail(f"I couldn't bring {query} to the front.",
                    f"{query} ko saamne nahi la paya.")

    title = win["title"][:60]
    return ok(f"Switched to {title}.", f"{title} pe aa gaya.", detail=title)


@skill(
    name="list_running_apps",
    description="List the applications that currently have open windows",
    risk=Risk.SAFE,
    category="apps",
    examples=[
        "what's running", "list open apps", "kya kya khula hai",
        "what apps are running", "which apps are open", "kaunse app khule hain",
    ],
)
def list_running_apps(ctx: SkillContext = None) -> object:  # noqa: ARG001
    seen: dict[str, str] = {}
    for win in winutil.list_windows():
        process = win["process"].removesuffix(".exe")
        if process and process not in seen:
            seen[process] = win["title"]

    if not seen:
        return ok("Nothing seems to be open.", "Kuch bhi khula nahi hai.")

    names = sorted(seen)[:10]
    listed = ", ".join(names)
    return ok(f"Open right now: {listed}.", f"Abhi khula hai: {listed}.",
              apps=names)


@skill(
    name="refresh_app_index",
    description="Rescan the system for installed applications",
    risk=Risk.SAFE,
    category="apps",
    examples=["refresh app list", "rescan apps", "app list update karo"],
)
def refresh_app_index() -> object:
    entries = app_index.build(force=True)
    return ok(f"Found {len(entries)} apps.", f"{len(entries)} apps mile.",
              count=len(entries))
