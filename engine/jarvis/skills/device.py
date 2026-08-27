"""Device status and radios: battery, system health, time, Wi-Fi, Bluetooth."""

from __future__ import annotations

import logging
import time

from .. import winutil
from ..permissions import Capability, Risk, consent
from .registry import fail, ok, skill

log = logging.getLogger(__name__)


@skill(
    name="get_battery",
    description="Report the battery percentage and whether it is charging",
    risk=Risk.SAFE,
    category="device",
    examples=[
        "what's the battery", "how much battery is left", "battery kitni hai",
        "kitna battery bacha hai", "battery status", "battery percentage",
    ],
)
def get_battery() -> object:
    import psutil

    battery = psutil.sensors_battery()
    if battery is None:
        return fail("I can't find a battery on this machine.",
                    "Is machine par battery nahi mili.")

    percent = round(battery.percent)
    if battery.power_plugged:
        return ok(
            f"Battery is at {percent} percent and charging.",
            f"Battery {percent} percent hai aur charge ho rahi hai.",
            percent=percent, charging=True,
        )

    tail_en, tail_hi = "", ""
    if battery.secsleft and battery.secsleft > 0:
        hours, minutes = divmod(int(battery.secsleft) // 60, 60)
        if hours:
            tail_en = f", about {hours} hours {minutes} minutes left"
            tail_hi = f", lagbhag {hours} ghante {minutes} minute bache hain"
        else:
            tail_en = f", about {minutes} minutes left"
            tail_hi = f", lagbhag {minutes} minute bache hain"

    warn_en = " You should plug in." if percent <= 20 else ""
    warn_hi = " Charger laga lijiye." if percent <= 20 else ""

    return ok(
        f"Battery is at {percent} percent{tail_en}.{warn_en}",
        f"Battery {percent} percent hai{tail_hi}.{warn_hi}",
        percent=percent, charging=False,
    )


@skill(
    name="get_system_status",
    description="Report CPU, memory and disk usage",
    risk=Risk.SAFE,
    category="device",
    examples=[
        "system status", "how's the cpu", "memory usage", "pc ka haal batao",
        "computer status", "how much ram is free", "how much ram do i have",
        "kitni ram hai", "disk space", "cpu usage kitna hai",
    ],
)
def get_system_status() -> object:
    from pathlib import Path

    import psutil

    cpu = psutil.cpu_percent(interval=0.4)
    memory = psutil.virtual_memory()
    # The volume the user's own files are on, whatever it's called. `C:\\` was
    # hardcoded, which raises off Windows — and is wrong even on Windows for
    # anyone whose profile lives on another drive. `.anchor` gives "C:\\" here
    # and "/" elsewhere.
    disk = psutil.disk_usage(Path.home().anchor or "/")
    free_gb = disk.free / (1024 ** 3)
    # Spoken aloud, so drop the colon — a voice reading "C colon" is worse
    # than one reading "C". Off Windows the anchor is just "/", which names
    # nothing useful, so say "disk".
    where = Path.home().anchor.rstrip("\\/:") or "disk"

    return ok(
        f"CPU at {cpu:.0f} percent, memory at {memory.percent:.0f} percent, "
        f"and {free_gb:.0f} gigabytes free on {where}.",
        f"CPU {cpu:.0f} percent, memory {memory.percent:.0f} percent, "
        f"aur {where} me {free_gb:.0f} GB khali hai.",
        cpu=cpu, memory=memory.percent, disk_free_gb=round(free_gb, 1),
    )


@skill(
    name="get_time",
    description="Report the current time",
    risk=Risk.SAFE,
    category="device",
    examples=["what time is it", "kya time hua hai", "time batao", "kitne baje hain"],
)
def get_time() -> object:
    now = time.localtime()
    spoken = time.strftime("%I:%M %p", now).lstrip("0")
    return ok(f"It's {spoken}.", f"{spoken} hue hain.", time=spoken)


@skill(
    name="get_date",
    description="Report today's date",
    risk=Risk.SAFE,
    category="device",
    examples=["what's the date", "aaj ki tareekh", "what day is it", "aaj kya din hai"],
)
def get_date() -> object:
    now = time.localtime()
    spoken = time.strftime("%A, %d %B %Y", now)
    return ok(f"Today is {spoken}.", f"Aaj {spoken} hai.", date=spoken)


@skill(
    name="toggle_wifi",
    description="Turn Wi-Fi on or off",
    risk=Risk.CONFIRM,
    category="device",
    params={"state": "on or off"},
    examples=["turn off wifi", "wifi band karo", "turn wifi on", "wifi chalu karo"],
    confirm_en="Turn Wi-Fi {state}?",
    confirm_hi="Wi-Fi {state} kar doon?",
)
def toggle_wifi(state: str = "off") -> object:
    enable = _wants_on(state)
    state = "on" if enable else "off"
    action = "enable" if enable else "disable"

    proc = winutil.run(["netsh", "interface", "set", "interface",
                        "name=Wi-Fi", f"admin={action}d"])
    if proc.returncode == 0:
        return ok(f"Wi-Fi turned {state}.", f"Wi-Fi {state} kar diya.")

    # Windows refuses this to an unelevated process. With the administrator
    # capability granted we may ask for rights; Windows still shows its own
    # prompt, so the user gets a second chance to refuse.
    granted, why = _elevate(
        "netsh.exe", f'interface set interface name="Wi-Fi" admin={action}d'
    )
    if granted:
        return ok(f"Wi-Fi turned {state}.", f"Wi-Fi {state} kar diya.",
                  detail="via administrator elevation")

    winutil.shell_open("ms-settings:network-wifi")
    return fail(
        f"I couldn't switch Wi-Fi — {why}. I've opened the settings instead.",
        f"Wi-Fi nahi badal paya — {why}. Settings khol di hai.",
        detail=(proc.stderr or proc.stdout).strip()[:200],
    )


# The Bluetooth radio, as Windows names it in Device Manager. Matching on the
# class rather than a fixed name because the adapter's name differs per vendor.
_BT_RADIO_QUERY = (
    "Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue | "
    "Where-Object { $_.FriendlyName -notmatch 'Enumerator' }"
)


@skill(
    name="toggle_bluetooth",
    description="Turn Bluetooth on or off",
    risk=Risk.CONFIRM,
    category="device",
    params={"state": "on or off"},
    examples=["turn on bluetooth", "bluetooth band karo", "bluetooth chalu karo",
              "turn off bluetooth"],
    confirm_en="Turn Bluetooth {state}?",
    confirm_hi="Bluetooth {state} kar doon?",
)
def toggle_bluetooth(state: str = "off") -> object:
    enable = _wants_on(state)
    state = "on" if enable else "off"
    verb = "Enable" if enable else "Disable"

    # There is no unelevated way to flip the radio without the WinRT Radio
    # API, which needs a package we don't ship — so this one goes straight to
    # elevation rather than pretending to try first.
    granted, why = _elevate(
        "powershell.exe",
        "-NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "
        f'"{_BT_RADIO_QUERY} | {verb}-PnpDevice -Confirm:$false"',
    )
    if granted:
        return ok(f"Bluetooth turned {state}.", f"Bluetooth {state} kar diya.",
                  detail="via administrator elevation")

    winutil.shell_open("ms-settings:bluetooth")
    return fail(
        f"I couldn't switch Bluetooth — {why}. I've opened the settings instead.",
        f"Bluetooth nahi badal paya — {why}. Settings khol di hai.",
    )


def _wants_on(state: str) -> bool:
    return (state or "").strip().lower() in (
        "on", "enable", "enabled", "start", "true", "yes",
        "chalu", "chaalu", "on karo", "chalu karo",
    )


def _elevate(program: str, arguments: str) -> tuple[bool, str]:
    """Run something as Administrator, if the user has allowed that.

    Two gates, and both matter. Ours decides whether the assistant may even
    ask — a misheard command should not be able to raise a UAC prompt on a
    machine where the user never wanted elevated actions at all. Windows' own
    prompt then decides whether it happens.
    """
    if winutil.is_elevated():
        proc = winutil.run(_split_command(program, arguments))
        if proc.returncode == 0:
            return True, ""
        return False, "the command failed"

    if not consent.allows(Capability.ADMIN):
        return False, (
            "administrator access is switched off in Settings → Permissions"
        )
    return winutil.run_elevated(program, arguments)


def _split_command(program: str, arguments: str) -> list[str]:
    import shlex

    return [program, *shlex.split(arguments, posix=False)]


@skill(
    name="get_network_status",
    description="Report whether the machine is online and which network it is on",
    risk=Risk.SAFE,
    category="device",
    examples=["am i online", "internet chal raha hai", "which wifi am i on",
              "network status"],
)
def get_network_status() -> object:
    ssid = ""
    out = winutil.powershell("netsh wlan show interfaces")
    for line in out.splitlines():
        if line.strip().lower().startswith("ssid") and ":" in line and "bssid" not in line.lower():
            ssid = line.split(":", 1)[1].strip()
            break

    import socket

    try:
        socket.create_connection(("1.1.1.1", 53), timeout=3).close()
        online = True
    except OSError:
        online = False

    if online and ssid:
        return ok(f"Online, connected to {ssid}.", f"Online hoon, {ssid} se juda hoon.",
                  online=True, ssid=ssid)
    if online:
        return ok("Online.", "Internet chal raha hai.", online=True)
    return ok("No internet connection right now.", "Abhi internet nahi chal raha.",
              online=False)


@skill(
    name="open_settings",
    description="Open a Windows settings page",
    risk=Risk.SAFE,
    category="device",
    params={"page": "Settings page: display, sound, bluetooth, wifi, battery, apps, privacy"},
    examples=["open sound settings", "settings kholo", "open bluetooth settings"],
)
def open_settings(page: str = "") -> object:
    pages = {
        "display": "ms-settings:display", "sound": "ms-settings:sound",
        "bluetooth": "ms-settings:bluetooth", "wifi": "ms-settings:network-wifi",
        "network": "ms-settings:network", "battery": "ms-settings:batterysaver",
        "apps": "ms-settings:appsfeatures", "privacy": "ms-settings:privacy",
        "update": "ms-settings:windowsupdate", "storage": "ms-settings:storagesense",
        "": "ms-settings:",
    }
    target = pages.get((page or "").strip().lower(), "ms-settings:")
    winutil.shell_open(target)
    label = page or "Windows"
    return ok(f"Opening {label} settings.", f"{label} settings khol raha hoon.")
