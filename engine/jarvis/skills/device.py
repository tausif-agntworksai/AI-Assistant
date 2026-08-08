"""Device status and radios: battery, system health, time, Wi-Fi, Bluetooth."""

from __future__ import annotations

import logging
import time

from .. import winutil
from ..permissions import Risk
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
    import psutil

    cpu = psutil.cpu_percent(interval=0.4)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("C:\\")
    free_gb = disk.free / (1024 ** 3)

    return ok(
        f"CPU at {cpu:.0f} percent, memory at {memory.percent:.0f} percent, "
        f"and {free_gb:.0f} gigabytes free on C.",
        f"CPU {cpu:.0f} percent, memory {memory.percent:.0f} percent, "
        f"aur C drive me {free_gb:.0f} GB khali hai.",
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
    enable = (state or "").strip().lower() in ("on", "enable", "chalu", "start", "true")
    action = "enable" if enable else "disable"

    proc = winutil.run(["netsh", "interface", "set", "interface",
                        "name=Wi-Fi", f"admin={action}d"])
    if proc.returncode == 0:
        return ok(f"Wi-Fi turned {state}.", f"Wi-Fi {state} kar diya.")

    # netsh needs elevation for this; say so instead of failing opaquely.
    winutil.shell_open("ms-settings:network-wifi")
    return fail(
        "I need administrator rights to switch Wi-Fi, so I opened the settings instead.",
        "Wi-Fi badalne ke liye admin rights chahiye, isliye settings khol di.",
        detail=(proc.stderr or proc.stdout).strip()[:200],
    )


@skill(
    name="toggle_bluetooth",
    description="Open Bluetooth settings to turn Bluetooth on or off",
    risk=Risk.SAFE,
    category="device",
    examples=["turn on bluetooth", "bluetooth band karo", "bluetooth chalu karo"],
)
def toggle_bluetooth() -> object:
    # The supported way to flip the radio is the WinRT Radio API, which needs a
    # package we don't ship. Settings is one tap and always works.
    winutil.shell_open("ms-settings:bluetooth")
    return ok("Opened Bluetooth settings.", "Bluetooth settings khol di.")


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
