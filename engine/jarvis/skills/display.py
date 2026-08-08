"""Screen brightness and display settings."""

from __future__ import annotations

import logging

from .. import winutil
from ..permissions import Risk
from .registry import fail, ok, skill

log = logging.getLogger(__name__)

STEP = 15


def _get_brightness() -> int | None:
    """Current brightness, or None if the panel doesn't expose WMI control.

    External monitors generally don't; the built-in laptop panel does.
    """
    try:
        import pythoncom
        import wmi

        pythoncom.CoInitialize()
        monitors = wmi.WMI(namespace="wmi").WmiMonitorBrightness()
        if monitors:
            return int(monitors[0].CurrentBrightness)
    except Exception as exc:  # noqa: BLE001
        log.debug("WMI brightness read failed: %s", exc)

    out = winutil.powershell(
        "(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness)"
        ".CurrentBrightness"
    )
    try:
        return int(out.splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


def _set_brightness(percent: int) -> bool:
    percent = max(0, min(100, int(percent)))
    try:
        import pythoncom
        import wmi

        pythoncom.CoInitialize()
        methods = wmi.WMI(namespace="wmi").WmiMonitorBrightnessMethods()
        if methods:
            methods[0].WmiSetBrightness(Brightness=percent, Timeout=0)
            return True
    except Exception as exc:  # noqa: BLE001
        log.debug("WMI brightness write failed: %s", exc)

    out = winutil.powershell(
        f"(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods)"
        f".WmiSetBrightness(0,{percent}); 'ok'"
    )
    return "ok" in out


@skill(
    name="set_brightness",
    description="Set the screen brightness to a percentage (0-100)",
    risk=Risk.SAFE,
    category="display",
    params={"level": "Brightness percentage from 0 to 100"},
    examples=[
        "set brightness to 60", "brightness 40 kar do", "make the screen brighter to 80",
        "screen brightness 50 percent",
    ],
)
def set_brightness(level: int = 60) -> object:
    level = max(0, min(100, int(level)))
    if not _set_brightness(level):
        return fail(
            "This display doesn't let me change brightness.",
            "Ye display brightness change karne nahi deti.",
            detail="WMI brightness control unavailable (common on external monitors)",
        )
    return ok(f"Brightness set to {level} percent.", f"Brightness {level} percent kar di.",
              level=level)


@skill(
    name="brightness_up",
    description="Increase the screen brightness",
    risk=Risk.SAFE,
    category="display",
    examples=["brighter", "increase brightness", "brightness badhao", "screen tez karo"],
)
def brightness_up() -> object:
    current = _get_brightness()
    if current is None:
        return fail("I can't read this display's brightness.",
                    "Is display ki brightness nahi padh pa raha.")
    return set_brightness(current + STEP)


@skill(
    name="brightness_down",
    description="Decrease the screen brightness",
    risk=Risk.SAFE,
    category="display",
    examples=["dimmer", "decrease brightness", "brightness kam karo", "screen dhimi karo"],
)
def brightness_down() -> object:
    current = _get_brightness()
    if current is None:
        return fail("I can't read this display's brightness.",
                    "Is display ki brightness nahi padh pa raha.")
    return set_brightness(current - STEP)


@skill(
    name="get_brightness",
    description="Report the current screen brightness",
    risk=Risk.SAFE,
    category="display",
    examples=["what's the brightness", "brightness kitni hai"],
)
def get_brightness() -> object:
    current = _get_brightness()
    if current is None:
        return fail("I can't read this display's brightness.",
                    "Is display ki brightness nahi padh pa raha.")
    return ok(f"Brightness is at {current} percent.",
              f"Brightness {current} percent par hai.", level=current)


@skill(
    name="toggle_night_light",
    description="Open the Night Light settings so it can be turned on or off",
    risk=Risk.SAFE,
    category="display",
    examples=["turn on night light", "night light band karo", "toggle night light",
              "blue light filter"],
)
def toggle_night_light() -> object:
    # Windows exposes no supported API for this; the state lives in an opaque
    # CloudStore registry blob whose format changes between builds. Opening the
    # settings page is the honest option.
    winutil.shell_open("ms-settings:nightlight")
    return ok("Opened Night Light settings — toggle it there.",
              "Night Light settings khol di — wahan se on/off kar lijiye.")


@skill(
    name="open_display_settings",
    description="Open Windows display settings",
    risk=Risk.SAFE,
    category="display",
    examples=["open display settings", "display settings kholo", "screen settings"],
)
def open_display_settings() -> object:
    winutil.shell_open("ms-settings:display")
    return ok("Opening display settings.", "Display settings khol raha hoon.")
