"""Answering and declining calls that ring on this computer.

On a laptop the calls that arrive are WhatsApp Desktop calls and Google
Chat/Meet calls in a browser tab. Neither has an API, a URL scheme or a
documented global hotkey for "answer" — so the only route available is the one a
person uses: find the window that is ringing, bring it to the front, and press
the key that accepts.

**Say plainly what that means.** This is the least reliable thing in the engine,
and pretending otherwise would waste the user's time at exactly the wrong
moment. Two consequences are designed for rather than hoped away:

* **The ringing window has to be found first.** Nothing is sent if it is not,
  because a stray Enter into whatever happens to be focused could do anything.
  Not finding it produces "I can't see a call ringing", which is true and
  useful, rather than a keystroke into the void.
* **Google Meet needs the tab to be the active one.** A background tab receives
  no keystrokes at all, so the browser window is focused and the reply says to
  check the tab. Meet's own shortcut (Ctrl+D toggles the mic, not answer) does
  not help here: the join button takes Enter when focused, and that is that.

Phone calls over the cellular network are not here and cannot be. A Windows
machine has no cellular radio and no access to the phone's call stack — that
needs the mobile client, which is not built. `answer_call` says so rather than
failing silently, so the limitation is heard once instead of discovered
repeatedly.
"""

from __future__ import annotations

import logging
import sys
import time

from .. import winutil
from ..permissions import Risk
from .registry import fail, ok, skill

log = logging.getLogger(__name__)

#: How long to wait for the window to actually come forward before giving up.
#: A ringing call is a few seconds of opportunity, so this cannot be generous.
FOCUS_TIMEOUT = 2.5


class CallApp:
    """An app whose ringing window can be recognised and acted on."""

    def __init__(self, name: str, processes: tuple[str, ...],
                 title_hints: tuple[str, ...], accept: tuple[str, ...],
                 decline: tuple[str, ...], note: str = "", note_hi: str = ""):
        self.name = name
        self.processes = processes
        self.title_hints = title_hints
        self.accept = accept
        self.decline = decline
        self.note = note
        self.note_hi = note_hi


# Ordered by how confidently the ringing window can be identified. WhatsApp
# opens a separate window whose title carries the caller's name, which is a
# strong signal; a browser tab is a weak one, so it is tried last.
CALL_APPS: tuple[CallApp, ...] = (
    CallApp(
        name="WhatsApp",
        processes=("whatsapp.exe",),
        # WhatsApp Desktop titles the incoming-call window with the caller and
        # a word for what it is. Matching on the word rather than the caller,
        # since the caller is the unknown.
        title_hints=("incoming voice call", "incoming video call",
                     "incoming call", "calling", "whatsapp call"),
        accept=("enter",),
        decline=("escape",),
    ),
    CallApp(
        name="Microsoft Teams",
        processes=("ms-teams.exe", "teams.exe"),
        title_hints=("incoming call", "is calling"),
        # Teams documents these, and they are global while Teams has focus.
        accept=("ctrl", "shift", "a"),
        decline=("ctrl", "shift", "d"),
    ),
    CallApp(
        name="Zoom",
        processes=("zoom.exe",),
        title_hints=("incoming", "is inviting you"),
        accept=("enter",),
        decline=("escape",),
    ),
    CallApp(
        name="Google Chat",
        processes=("chrome.exe", "msedge.exe", "firefox.exe", "brave.exe"),
        title_hints=("google chat", "meet -", "google meet", "is calling"),
        accept=("enter",),
        decline=("escape",),
        note=(" It's a browser call, so make sure that tab is the one you're "
              "looking at — a background tab won't take the keypress."),
        note_hi=(" Ye browser call hai, to dekh lijiye ki wahi tab saamne ho — "
                 "background tab keypress nahi leta."),
    ),
)


def _ringing() -> tuple[CallApp, dict] | tuple[None, None]:
    """Find the window that looks like it is ringing.

    Both halves have to agree — the process and something in the title — because
    either alone is far too loose. "chrome.exe" is always running, and a window
    called "Incoming" could be anything.
    """
    if sys.platform != "win32":
        return None, None

    windows = winutil.list_windows(visible_only=True)
    for app in CALL_APPS:
        for window in windows:
            process = str(window.get("process", "")).lower()
            title = str(window.get("title", "")).lower()
            if process not in app.processes:
                continue
            if any(hint in title for hint in app.title_hints):
                log.info("Ringing window: %s / %r", process, window.get("title"))
                return app, window
    return None, None


def _act(app: CallApp, window: dict, keys: tuple[str, ...]) -> bool:
    """Focus the ringing window, confirm it took focus, then send the keys.

    The confirmation is the point. Without it these keystrokes go wherever focus
    happens to be, and "escape" or "enter" into an arbitrary window is not
    something to do on a guess.
    """
    winutil.focus_window(int(window.get("hwnd", 0)))

    deadline = time.monotonic() + FOCUS_TIMEOUT
    while time.monotonic() < deadline:
        current = str(winutil.foreground_window().get("process", "")).lower()
        if current in app.processes:
            return winutil.send_keys(*keys)
        time.sleep(0.1)

    log.info("%s never came to the front — sending nothing", app.name)
    return False


@skill(
    name="answer_call",
    description="Answer an incoming call ringing on this computer",
    risk=Risk.CONFIRM,
    category="calls",
    examples=[
        "answer the call", "pick up", "call uthao", "answer",
        "phone uthao", "call receive karo", "pick up the call",
    ],
    # No confirmation prompt in practice: a ringing call is a few seconds long
    # and asking "shall I answer it?" spends them. CONFIRM covers the typed and
    # model-driven paths, where there is no ringing phone to be late for.
    confirm_en="Answer the call?",
    confirm_hi="Call utha loon?",
)
def answer_call() -> object:
    app, window = _ringing()
    if app is None:
        return fail(
            "I can't see a call ringing. I can answer WhatsApp, Teams, Zoom and "
            "Google Chat calls on this computer — a call to your phone number "
            "rings on your phone, and I can't reach that from here.",
            "Mujhe koi call bajti nahi dikh rahi. Is computer par WhatsApp, "
            "Teams, Zoom aur Google Chat ki call utha sakta hoon — phone number "
            "par aayi call phone par bajti hai, wahan tak main nahi pahunch "
            "sakta.",
            detail="no ringing window",
        )

    if not _act(app, window, app.accept):
        return fail(
            f"I found the {app.name} call but couldn't bring it forward in time "
            "— answer it by hand, it's still ringing.",
            f"{app.name} ki call mili par samay par saamne nahi aa payi — "
            "haath se utha lijiye, abhi baj rahi hai.",
            detail=f"{app.name}: focus failed",
        )

    return ok(
        f"Answered on {app.name}.{app.note}",
        f"{app.name} par call utha li.{app.note_hi}",
        detail=f"answered via {app.name}",
    )


@skill(
    name="decline_call",
    description="Decline or hang up a call ringing on this computer",
    risk=Risk.CONFIRM,
    category="calls",
    examples=[
        "decline the call", "reject the call", "hang up", "cut the call",
        "call kaat do", "call reject karo", "phone kaat do",
    ],
    confirm_en="Decline the call?",
    confirm_hi="Call kaat doon?",
)
def decline_call() -> object:
    app, window = _ringing()
    if app is None:
        return fail(
            "I can't see a call ringing.",
            "Mujhe koi call bajti nahi dikh rahi.",
            detail="no ringing window",
        )

    if not _act(app, window, app.decline):
        return fail(
            f"I found the {app.name} call but couldn't bring it forward in time.",
            f"{app.name} ki call mili par samay par saamne nahi aa payi.",
            detail=f"{app.name}: focus failed",
        )
    return ok(f"Declined on {app.name}.", f"{app.name} par call kaat di.",
              detail=f"declined via {app.name}")
