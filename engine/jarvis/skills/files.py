"""Screenshots, folders, file search, clipboard and dictation."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from .. import winutil
from ..permissions import Risk
from .registry import fail, ok, skill

log = logging.getLogger(__name__)

HOME = Path.home()

KNOWN_FOLDERS: dict[str, Path] = {
    "downloads": HOME / "Downloads",
    "download": HOME / "Downloads",
    "documents": HOME / "Documents",
    "document": HOME / "Documents",
    "desktop": HOME / "Desktop",
    "pictures": HOME / "Pictures",
    "photos": HOME / "Pictures",
    "music": HOME / "Music",
    "videos": HOME / "Videos",
    "video": HOME / "Videos",
    "home": HOME,
    "recycle bin": Path("shell:RecycleBinFolder"),
}

SEARCH_ROOTS = [HOME / "Downloads", HOME / "Documents", HOME / "Desktop",
                HOME / "Pictures", HOME / "Videos"]
SEARCH_TIME_BUDGET = 6.0  # seconds — a voice reply that takes longer feels broken


@skill(
    name="take_screenshot",
    description="Take a screenshot of the whole screen and save it",
    risk=Risk.SAFE,
    category="files",
    examples=[
        "take a screenshot", "screenshot lo", "capture the screen",
        "screenshot le lo", "screen capture karo",
    ],
)
def take_screenshot() -> object:
    import mss

    folder = HOME / "Pictures" / "Screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"jarvis-{time.strftime('%Y%m%d-%H%M%S')}.png"

    try:
        with mss.mss() as sct:
            sct.shot(mon=-1, output=str(target))  # -1 = all monitors combined
    except Exception as exc:  # noqa: BLE001
        log.error("Screenshot failed: %s", exc)
        return fail("I couldn't take the screenshot.", "Screenshot nahi le paya.")

    return ok(
        "Screenshot saved to your Pictures folder.",
        "Screenshot Pictures folder me save kar diya.",
        detail=str(target), path=str(target),
    )


@skill(
    name="open_folder",
    description="Open a folder in File Explorer",
    risk=Risk.SAFE,
    category="files",
    params={"name": "Folder name (downloads, documents, desktop, pictures) or a full path"},
    examples=[
        "open downloads", "downloads folder kholo", "open my documents",
        "show me the desktop folder", "open pictures",
    ],
)
def open_folder(name: str = "downloads") -> object:
    key = (name or "").strip().lower().removesuffix(" folder")
    target = KNOWN_FOLDERS.get(key)

    if target is None:
        candidate = Path(os.path.expandvars(name)).expanduser()
        if candidate.exists():
            target = candidate
        else:
            from rapidfuzz import process

            hit = process.extractOne(key, list(KNOWN_FOLDERS), score_cutoff=72)
            if hit:
                target = KNOWN_FOLDERS[hit[0]]

    if target is None:
        return fail(f"I couldn't find a folder called {name}.",
                    f"{name} naam ka folder nahi mila.")

    winutil.shell_open(str(target))
    return ok(f"Opening {target.name or name}.", f"{target.name or name} khol raha hoon.",
              detail=str(target))


@skill(
    name="search_files",
    description="Search your personal folders for files matching a name",
    risk=Risk.SAFE,
    category="files",
    params={"query": "Part of the file name to look for"},
    examples=[
        "find files called invoice", "search for resume", "resume file dhundo",
        "find my presentation", "budget file dhoondo",
    ],
)
def search_files(query: str) -> object:
    needle = (query or "").strip().lower()
    if len(needle) < 2:
        return fail("What should I look for?", "Kya dhoondhun?")

    deadline = time.monotonic() + SEARCH_TIME_BUDGET
    matches: list[Path] = []

    for root in SEARCH_ROOTS:
        if not root.exists() or time.monotonic() > deadline:
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if time.monotonic() > deadline:
                break
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "$"))]
            for filename in filenames:
                if needle in filename.lower():
                    matches.append(Path(dirpath) / filename)
                    if len(matches) >= 20:
                        break
            if len(matches) >= 20:
                break

    if not matches:
        return fail(f"I couldn't find anything matching {query}.",
                    f"{query} se milta koi file nahi mila.")

    names = [m.name for m in matches[:5]]
    listed = ", ".join(names)
    winutil.shell_open(str(matches[0].parent))
    return ok(
        f"Found {len(matches)}. Top matches: {listed}. I've opened the first one's folder.",
        f"{len(matches)} mile. Sabse upar: {listed}. Pehle wale ka folder khol diya.",
        detail=str(matches[0]), matches=[str(m) for m in matches[:10]],
    )


@skill(
    name="read_clipboard",
    description="Read out what is currently on the clipboard",
    risk=Risk.SAFE,
    category="files",
    examples=["what's on my clipboard", "read the clipboard", "clipboard me kya hai"],
)
def read_clipboard() -> object:
    import pyperclip

    try:
        content = (pyperclip.paste() or "").strip()
    except Exception as exc:  # noqa: BLE001
        log.error("Clipboard read failed: %s", exc)
        return fail("I couldn't read the clipboard.", "Clipboard nahi padh paya.")

    if not content:
        return ok("The clipboard is empty.", "Clipboard khali hai.")

    spoken = content if len(content) <= 400 else content[:400] + "... and there's more"
    return ok(spoken, spoken, detail=f"{len(content)} chars", content=content)


@skill(
    name="copy_to_clipboard",
    description="Put some text on the clipboard",
    risk=Risk.SAFE,
    category="files",
    params={"text": "The text to copy"},
    examples=["copy this to the clipboard", "clipboard me copy karo"],
)
def copy_to_clipboard(text: str) -> object:
    import pyperclip

    try:
        pyperclip.copy(text or "")
    except Exception as exc:  # noqa: BLE001
        log.error("Clipboard write failed: %s", exc)
        return fail("I couldn't copy that.", "Copy nahi kar paya.")
    return ok("Copied.", "Copy kar diya.")


@skill(
    name="type_text",
    description="Type text into whatever window currently has focus (dictation)",
    risk=Risk.CONFIRM,
    category="files",
    params={"text": "The text to type"},
    examples=["type hello world", "likho namaste", "dictate this", "type karo"],
    confirm_en="Type that into the focused window?",
    confirm_hi="Ye type kar doon?",
)
def type_text(text: str) -> object:
    import pyautogui

    content = (text or "").strip()
    if not content:
        return fail("What should I type?", "Kya likhun?")

    try:
        # write() only handles ASCII reliably; anything else goes via the
        # clipboard so Devanagari and emoji survive.
        if content.isascii():
            pyautogui.write(content, interval=0.01)
        else:
            import pyperclip

            previous = pyperclip.paste()
            pyperclip.copy(content)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.15)
            pyperclip.copy(previous)
    except Exception as exc:  # noqa: BLE001
        log.error("Typing failed: %s", exc)
        return fail("I couldn't type that.", "Type nahi kar paya.")

    preview = content[:40] + ("..." if len(content) > 40 else "")
    return ok(f"Typed: {preview}", f"Likh diya: {preview}", detail=preview)


@skill(
    name="open_file",
    description="Open a specific file by its full path",
    risk=Risk.SAFE,
    category="files",
    params={"path": "Full path to the file"},
    examples=["open the file at C:/notes.txt"],
    hidden=True,
)
def open_file(path: str) -> object:
    target = Path(os.path.expandvars(path or "")).expanduser()
    if not target.exists():
        return fail("That file doesn't exist.", "Ye file nahi mili.")
    winutil.shell_open(str(target))
    return ok(f"Opening {target.name}.", f"{target.name} khol raha hoon.")
