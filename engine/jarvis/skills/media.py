"""Media transport control.

Sent as virtual media keys rather than by talking to any particular player, so
the same command works in Spotify, YouTube in a browser, VLC or anything else
that registers for media keys.
"""

from __future__ import annotations

import ctypes
import logging
import time

from ..permissions import Risk
from .registry import fail, ok, skill

log = logging.getLogger(__name__)

VK_MEDIA_NEXT = 0xB0
VK_MEDIA_PREV = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3

KEYEVENTF_KEYUP = 0x0002


def _tap(vk: int) -> bool:
    try:
        ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.03)
        ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Media key %#x failed: %s", vk, exc)
        return False


@skill(
    name="media_play_pause",
    description="Play or pause whatever media is currently active",
    risk=Risk.SAFE,
    category="media",
    examples=[
        "play", "pause", "pause the music", "gana chalao", "music band karo",
        "song play karo", "rok do gana", "resume",
    ],
)
def media_play_pause() -> object:
    if not _tap(VK_MEDIA_PLAY_PAUSE):
        return fail("I couldn't reach the media controls.", "Media control nahi mila.")
    return ok("Done.", "Ho gaya.")


@skill(
    name="media_next",
    description="Skip to the next track",
    risk=Risk.SAFE,
    category="media",
    examples=["next song", "skip this", "agla gana", "next track", "aage badhao"],
)
def media_next() -> object:
    if not _tap(VK_MEDIA_NEXT):
        return fail("I couldn't skip.", "Skip nahi kar paya.")
    return ok("Next track.", "Agla gana.")


@skill(
    name="media_previous",
    description="Go back to the previous track",
    risk=Risk.SAFE,
    category="media",
    examples=["previous song", "go back", "pichla gana", "last track", "peeche jao"],
)
def media_previous() -> object:
    if not _tap(VK_MEDIA_PREV):
        return fail("I couldn't go back.", "Peeche nahi ja paya.")
    return ok("Previous track.", "Pichla gana.")


@skill(
    name="media_stop",
    description="Stop media playback",
    risk=Risk.SAFE,
    category="media",
    examples=["stop the music", "stop playback", "gana band karo", "music stop karo"],
)
def media_stop() -> object:
    if not _tap(VK_MEDIA_STOP):
        return fail("I couldn't stop it.", "Band nahi kar paya.")
    return ok("Stopped.", "Band kar diya.")
