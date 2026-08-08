"""System and per-application volume, via the Windows Core Audio API."""

from __future__ import annotations

import logging

from ..permissions import Risk
from .registry import fail, ok, skill

log = logging.getLogger(__name__)

STEP = 10  # percentage points per "volume up"


def _endpoint():
    """The master audio endpoint.

    COM has to be initialised on whichever thread touches it, and the skill
    runs on the orchestrator's worker thread rather than the main one.

    Recent pycaw wraps the raw IMMDevice in an `AudioDevice` that exposes
    `EndpointVolume` directly; older releases returned the COM object and
    required `.Activate()`. Both are supported so the version doesn't matter.
    """
    import pythoncom
    from pycaw.pycaw import AudioUtilities

    pythoncom.CoInitialize()
    speakers = AudioUtilities.GetSpeakers()

    endpoint = getattr(speakers, "EndpointVolume", None)
    if endpoint is not None:
        return endpoint

    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import IAudioEndpointVolume

    interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def _get_level() -> int:
    return round(_endpoint().GetMasterVolumeLevelScalar() * 100)


def _set_level(percent: int) -> int:
    percent = max(0, min(100, int(percent)))
    endpoint = _endpoint()
    endpoint.SetMasterVolumeLevelScalar(percent / 100.0, None)
    if percent > 0 and endpoint.GetMute():
        endpoint.SetMute(0, None)  # setting a level implies you want to hear it
    return percent


@skill(
    name="set_volume",
    description="Set the system volume to a specific percentage (0-100)",
    risk=Risk.SAFE,
    category="audio",
    params={"level": "Volume percentage from 0 to 100"},
    examples=[
        "set volume to 50", "volume 30 kar do", "volume pachas kar do",
        "make the volume 70", "set the volume to 20 percent",
    ],
)
def set_volume(level: int = 50) -> object:
    try:
        applied = _set_level(level)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not set volume: %s", exc)
        return fail("I couldn't change the volume.", "Volume badal nahi paya.")
    return ok(f"Volume set to {applied} percent.", f"Volume {applied} par set kar diya.",
              level=applied)


@skill(
    name="volume_up",
    description="Increase the system volume",
    risk=Risk.SAFE,
    category="audio",
    examples=["turn up the volume", "volume badhao", "louder", "awaz tez karo",
              "increase volume", "volume badha do"],
)
def volume_up() -> object:
    try:
        applied = _set_level(_get_level() + STEP)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not raise volume: %s", exc)
        return fail("I couldn't change the volume.", "Volume badal nahi paya.")
    return ok(f"Volume {applied} percent.", f"Volume {applied} percent.", level=applied)


@skill(
    name="volume_down",
    description="Decrease the system volume",
    risk=Risk.SAFE,
    category="audio",
    examples=["turn down the volume", "volume kam karo", "quieter", "awaz kam karo",
              "decrease volume", "volume ghata do"],
)
def volume_down() -> object:
    try:
        applied = _set_level(_get_level() - STEP)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not lower volume: %s", exc)
        return fail("I couldn't change the volume.", "Volume badal nahi paya.")
    return ok(f"Volume {applied} percent.", f"Volume {applied} percent.", level=applied)


@skill(
    name="mute_audio",
    description="Mute the system audio",
    risk=Risk.SAFE,
    category="audio",
    examples=["mute", "mute the sound", "awaz band karo", "chup karo", "silence"],
)
def mute_audio() -> object:
    try:
        _endpoint().SetMute(1, None)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not mute: %s", exc)
        return fail("I couldn't mute it.", "Mute nahi kar paya.")
    return ok("Muted.", "Mute kar diya.")


@skill(
    name="unmute_audio",
    description="Unmute the system audio",
    risk=Risk.SAFE,
    category="audio",
    examples=["unmute", "turn the sound back on", "awaz chalu karo", "unmute karo"],
)
def unmute_audio() -> object:
    try:
        endpoint = _endpoint()
        endpoint.SetMute(0, None)
        level = round(endpoint.GetMasterVolumeLevelScalar() * 100)
        if level == 0:
            level = _set_level(30)  # unmuting at zero would still be silent
    except Exception as exc:  # noqa: BLE001
        log.error("Could not unmute: %s", exc)
        return fail("I couldn't unmute it.", "Unmute nahi kar paya.")
    return ok(f"Unmuted, volume {level} percent.", f"Unmute kar diya, volume {level} percent.")


@skill(
    name="get_volume",
    description="Report the current system volume",
    risk=Risk.SAFE,
    category="audio",
    examples=["what's the volume", "volume kitna hai", "current volume"],
)
def get_volume() -> object:
    try:
        endpoint = _endpoint()
        level = round(endpoint.GetMasterVolumeLevelScalar() * 100)
        muted = bool(endpoint.GetMute())
    except Exception as exc:  # noqa: BLE001
        log.error("Could not read volume: %s", exc)
        return fail("I couldn't read the volume.", "Volume pata nahi chala.")
    if muted:
        return ok(f"Muted, but set to {level} percent.",
                  f"Mute hai, lekin {level} percent par set hai.", level=level, muted=True)
    return ok(f"Volume is at {level} percent.", f"Volume {level} percent par hai.", level=level)


@skill(
    name="set_app_volume",
    description="Set the volume of one application without changing the system volume",
    risk=Risk.SAFE,
    category="audio",
    params={"app": "Application name, e.g. chrome, spotify",
            "level": "Volume percentage from 0 to 100"},
    examples=["set chrome volume to 30", "spotify ka volume 50 kar do",
              "make chrome quieter"],
)
def set_app_volume(app: str, level: int = 50) -> object:
    from rapidfuzz import fuzz

    import pythoncom
    from pycaw.pycaw import AudioUtilities

    needle = (app or "").strip().lower()
    if not needle:
        return fail("Which app?", "Kaunsa app?")

    level = max(0, min(100, int(level)))
    try:
        pythoncom.CoInitialize()
        sessions = AudioUtilities.GetAllSessions()
    except Exception as exc:  # noqa: BLE001
        log.error("Could not enumerate audio sessions: %s", exc)
        return fail("I couldn't reach the audio sessions.", "Audio sessions nahi mile.")

    for session in sessions:
        if not session.Process:
            continue
        name = session.Process.name().lower().removesuffix(".exe")
        if fuzz.WRatio(needle, name) >= 80:
            session.SimpleAudioVolume.SetMasterVolume(level / 100.0, None)
            return ok(f"{name} volume set to {level} percent.",
                      f"{name} ka volume {level} percent kar diya.", app=name, level=level)

    return fail(f"{app} isn't playing any audio.", f"{app} se koi awaz nahi aa rahi.")
