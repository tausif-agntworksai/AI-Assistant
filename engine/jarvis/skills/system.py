"""Power and session control: lock, sleep, restart, shut down, sign out.

Every skill here is gated. Shutdown and restart additionally run on a delay so
there is a real window to say "cancel shutdown" — a misheard command should
never be unrecoverable.
"""

from __future__ import annotations

import ctypes
import logging

from .. import winutil
from ..permissions import Risk
from .registry import fail, ok, skill

log = logging.getLogger(__name__)

SHUTDOWN_DELAY_SEC = 15


@skill(
    name="lock_screen",
    description="Lock the computer screen",
    # Instant and reversible: the machine is one password away from where it
    # was, and nothing is lost. Asking "lock the screen?" out loud costs a
    # spoken prompt, a listening window and a second recognition pass -- about
    # four seconds -- to guard against an outcome you undo by typing.
    risk=Risk.SAFE,
    category="system",
    examples=["lock the screen", "lock my pc", "computer lock karo", "screen lock kar do"],
    confirm_en="Lock the screen?",
    confirm_hi="Screen lock kar doon?",
)
def lock_screen() -> object:
    try:
        ctypes.windll.user32.LockWorkStation()
        return ok("Locking up.", "Lock kar raha hoon.")
    except Exception as exc:  # noqa: BLE001
        log.error("LockWorkStation failed: %s", exc)
        return fail("I couldn't lock the screen.", "Screen lock nahi ho payi.")


@skill(
    name="sleep_pc",
    description="Put the computer to sleep",
    # Same reasoning as lock_screen: sleep loses nothing -- Windows keeps
    # everything in memory and a keypress brings it all back. Set
    # `permissions.risk_overrides: {sleep_pc: confirm}` to be asked again.
    risk=Risk.SAFE,
    category="system",
    examples=[
        "go to sleep", "sleep the laptop", "laptop sula do", "computer ko sula do",
        "put the pc to sleep", "so jao",
    ],
    confirm_en="Put the laptop to sleep?",
    confirm_hi="Laptop ko sula doon?",
)
def sleep_pc() -> object:
    # SetSuspendState(hibernate=False, force=False, disableWakeEvents=False).
    # This is what System.Windows.Forms.Application.SetSuspendState calls.
    try:
        if ctypes.windll.powrprof.SetSuspendState(0, 0, 0):
            return ok("Going to sleep. Good night.", "Sula raha hoon. Shubh ratri.")
    except Exception as exc:  # noqa: BLE001
        log.debug("powrprof.SetSuspendState failed: %s", exc)

    out = winutil.powershell(
        "Add-Type -AssemblyName System.Windows.Forms; "
        "[System.Windows.Forms.Application]::SetSuspendState('Suspend', $false, $false)"
    )
    if out is not None:
        return ok("Going to sleep. Good night.", "Sula raha hoon. Shubh ratri.")
    return fail("I couldn't put it to sleep.", "Sula nahi paya.")


@skill(
    name="shutdown_pc",
    description="Shut down the computer",
    risk=Risk.CRITICAL,
    category="system",
    examples=[
        "shut down the computer", "shutdown", "power off the pc",
        "computer band kar do", "laptop shutdown karo",
    ],
    confirm_en="Shut down the computer?",
    confirm_hi="Computer band kar doon?",
)
def shutdown_pc() -> object:
    proc = winutil.run(["shutdown", "/s", "/t", str(SHUTDOWN_DELAY_SEC), "/c",
                        "Shutdown requested by Jarvis"])
    if proc.returncode != 0:
        return fail("I couldn't shut it down.", "Shutdown nahi ho paya.",
                    detail=proc.stderr.strip()[:200])
    return ok(
        f"Shutting down in {SHUTDOWN_DELAY_SEC} seconds. Say 'cancel shutdown' to stop me.",
        f"{SHUTDOWN_DELAY_SEC} second me band ho jayega. Rokna ho toh 'cancel shutdown' boliye.",
    )


@skill(
    name="restart_pc",
    description="Restart the computer",
    risk=Risk.CRITICAL,
    category="system",
    examples=["restart the computer", "reboot", "computer restart karo", "reboot karo"],
    confirm_en="Restart the computer?",
    confirm_hi="Computer restart kar doon?",
)
def restart_pc() -> object:
    proc = winutil.run(["shutdown", "/r", "/t", str(SHUTDOWN_DELAY_SEC), "/c",
                        "Restart requested by Jarvis"])
    if proc.returncode != 0:
        return fail("I couldn't restart it.", "Restart nahi ho paya.",
                    detail=proc.stderr.strip()[:200])
    return ok(
        f"Restarting in {SHUTDOWN_DELAY_SEC} seconds. Say 'cancel shutdown' to stop me.",
        f"{SHUTDOWN_DELAY_SEC} second me restart hoga. Rokna ho toh 'cancel shutdown' boliye.",
    )


@skill(
    name="cancel_shutdown",
    description="Cancel a pending shutdown or restart",
    risk=Risk.SAFE,
    category="system",
    examples=["cancel shutdown", "stop the shutdown", "shutdown cancel karo", "ruko mat karo"],
)
def cancel_shutdown() -> object:
    proc = winutil.run(["shutdown", "/a"])
    if proc.returncode != 0:
        return ok("Nothing was scheduled.", "Kuch scheduled nahi tha.")
    return ok("Cancelled. Staying on.", "Cancel kar diya. Chalu rahega.")


@skill(
    name="sign_out",
    description="Sign out of the current Windows user session",
    risk=Risk.CRITICAL,
    category="system",
    examples=["sign out", "log out", "log off", "logout karo"],
    confirm_en="Sign out of Windows?",
    confirm_hi="Sign out kar doon?",
)
def sign_out() -> object:
    proc = winutil.run(["shutdown", "/l"])
    if proc.returncode != 0:
        return fail("I couldn't sign out.", "Sign out nahi ho paya.")
    return ok("Signing out.", "Sign out kar raha hoon.")


@skill(
    name="empty_recycle_bin",
    description="Permanently delete everything in the Recycle Bin",
    risk=Risk.CRITICAL,
    category="system",
    examples=["empty the recycle bin", "clear recycle bin", "recycle bin khali karo"],
    confirm_en="Permanently delete everything in the Recycle Bin?",
    confirm_hi="Recycle Bin ka saara data hamesha ke liye delete kar doon?",
)
def empty_recycle_bin() -> object:
    winutil.powershell("Clear-RecycleBin -Force -ErrorAction SilentlyContinue")
    return ok("Recycle Bin emptied.", "Recycle Bin khali kar diya.")
