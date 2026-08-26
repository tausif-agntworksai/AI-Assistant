# -*- coding: utf-8 -*-
"""The camera, and answering calls that ring on this computer.

Two features whose failure modes are physical rather than logical, so that is
what these test.

**The camera light.** A webcam left open is a webcam with its light on, and an
assistant that leaves the light on after taking one photo is one nobody keeps
installed. The device must be released on every path out of the capture,
including every failure — so the tests drive it through each failure and check
the handle was closed.

**A stray keystroke.** Answering a call means synthesising a keypress, and a
keypress goes wherever focus is. "Escape" or "Enter" into an arbitrary window
could do anything, so nothing is sent unless the ringing window was found *and*
confirmed to have come forward.
"""

import numpy as np
import pytest

from jarvis.nlu.rules import route
from jarvis.skills import calls, camera, load_all

load_all()


# --- the camera -----------------------------------------------------------


class FakeCamera:
    """Stands in for cv2.VideoCapture, recording whether it was released."""

    def __init__(self, opened=True, frames=None, raise_on_read=False):
        self._opened = opened
        self._frames = list(frames if frames is not None else [])
        self._raise = raise_on_read
        self.released = False
        self.reads = 0

    def isOpened(self):  # noqa: N802 - matching the cv2 API
        return self._opened

    def read(self):
        self.reads += 1
        if self._raise:
            raise RuntimeError("device fell over")
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        self.released = True


class FakeCv2:
    CAP_DSHOW = 700

    def __init__(self, device):
        self._device = device
        self.written: list[str] = []

    def VideoCapture(self, index, backend=None):  # noqa: N802
        return self._device

    def imwrite(self, path, frame):  # noqa: D401
        self.written.append(path)
        return True


def frame():
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_a_photo_is_taken_from_a_settled_frame():
    """Nearly every webcam returns two or three badly exposed frames while
    auto-exposure converges, so the first one is a picture of a dark rectangle."""
    device = FakeCamera(frames=[frame() for _ in range(camera.WARMUP_FRAMES)])
    got, error = camera._capture(FakeCv2(device))
    assert error is None and got is not None
    assert device.reads == camera.WARMUP_FRAMES, "warm-up frames were skipped"


def test_the_device_is_released_after_a_successful_photo():
    device = FakeCamera(frames=[frame() for _ in range(camera.WARMUP_FRAMES)])
    camera._capture(FakeCv2(device))
    assert device.released, "the camera light would still be on"


def test_the_device_is_released_when_no_frame_ever_arrives():
    device = FakeCamera(frames=[])
    got, error = camera._capture(FakeCv2(device))
    assert got is None and error is not None
    assert device.released


def test_the_device_is_released_when_reading_raises():
    device = FakeCamera(raise_on_read=True)
    got, error = camera._capture(FakeCv2(device))
    assert got is None and error is not None
    assert device.released


def test_a_camera_held_by_another_app_gives_up_rather_than_hanging(monkeypatch):
    """Waiting does not make a busy camera available, and going quiet is the
    worst response — the reply should say what happened."""
    monkeypatch.setattr(camera, "OPEN_TIMEOUT", 0.2)
    device = FakeCamera(opened=False)
    got, error = camera._capture(FakeCv2(device))
    assert got is None
    assert "another app" in error[0].lower()
    assert device.released


def test_an_opened_camera_that_sends_nothing_blames_the_right_thing():
    """A covered lens or a Windows privacy block both look like this, and both
    are things the user can act on."""
    _, error = camera._capture(FakeCv2(FakeCamera(frames=[])))
    assert "privacy" in error[0].lower() or "covered" in error[0].lower()


def test_taking_a_photo_without_opencv_says_so(monkeypatch):
    """A missing optional dependency costs one skill and names it, rather than
    failing at startup."""
    import builtins

    real_import = builtins.__import__

    def no_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("no cv2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cv2)
    result = camera.take_photo()
    assert not result.ok and "camera library" in result.text("en")


def test_photos_go_somewhere_the_user_would_look():
    """Not into an application cache directory they have never heard of."""
    assert camera.photo_dir().parent.name == "Pictures"


def test_taking_a_photo_asks_first():
    """Turning the webcam on is not a trivially reversible action."""
    from jarvis.skills.registry import registry

    assert registry.get("take_photo").risk.value != "safe"


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("take a photo", "take_photo"),
        ("click a photo", "take_photo"),
        ("take a picture", "take_photo"),
        ("photo le lo", "take_photo"),
        ("photo khinch do", "take_photo"),
        ("selfie le lo", "take_photo"),
        ("open the camera", "open_camera"),
        ("camera kholo", "open_camera"),
        ("camera chalu karo", "open_camera"),
        ("show my photos", "open_photos"),
        # Not the camera.
        ("take a screenshot", "take_screenshot"),
    ],
)
def test_camera_commands_route_offline(utterance, expected):
    intent = route(utterance)
    assert intent is not None and intent.skill == expected


def test_clicking_a_photo_does_not_merely_open_the_camera_app():
    """"Click a photo" asked out loud wants a photo, not an app to click a photo
    in. Conflating the two is the obvious way to get this wrong."""
    assert route("click a photo").skill == "take_photo"


# --- calls ----------------------------------------------------------------


def _windows(*rows):
    return [{"hwnd": 100 + i, "process": p, "title": t}
            for i, (p, t) in enumerate(rows)]


def test_a_ringing_whatsapp_window_is_found(monkeypatch):
    monkeypatch.setattr(calls.sys, "platform", "win32")
    monkeypatch.setattr(calls.winutil, "list_windows",
                        lambda visible_only=True: _windows(
                            ("whatsapp.exe", "Sana - Incoming voice call")))
    app, window = calls._ringing()
    assert app is not None and app.name == "WhatsApp"


def test_the_process_alone_is_not_enough(monkeypatch):
    """chrome.exe is always running. Treating that as a ringing call would send
    Enter into a browser at random."""
    monkeypatch.setattr(calls.sys, "platform", "win32")
    monkeypatch.setattr(calls.winutil, "list_windows",
                        lambda visible_only=True: _windows(
                            ("chrome.exe", "Gmail - Inbox (3)")))
    app, _ = calls._ringing()
    assert app is None


def test_the_title_alone_is_not_enough(monkeypatch):
    """A window called "Incoming" could be anything — a download manager, a
    spreadsheet someone named badly."""
    monkeypatch.setattr(calls.sys, "platform", "win32")
    monkeypatch.setattr(calls.winutil, "list_windows",
                        lambda visible_only=True: _windows(
                            ("notepad.exe", "incoming call notes.txt")))
    app, _ = calls._ringing()
    assert app is None


def test_nothing_ringing_says_what_it_can_and_cannot_answer(monkeypatch):
    """This is the reply that has to carry the limitation: a call to the user's
    phone number rings on the phone, and no Windows app can reach that."""
    monkeypatch.setattr(calls, "_ringing", lambda: (None, None))
    result = calls.answer_call()
    assert not result.ok
    text = result.text("en").lower()
    assert "whatsapp" in text
    assert "phone" in text, "the phone-number limitation must be stated"


def test_no_keystroke_is_sent_if_the_window_never_comes_forward(monkeypatch):
    """The whole safety story for calls."""
    keys: list[tuple] = []
    monkeypatch.setattr(calls, "FOCUS_TIMEOUT", 0.3)
    monkeypatch.setattr(calls.winutil, "focus_window", lambda hwnd: True)
    monkeypatch.setattr(calls.winutil, "foreground_window",
                        lambda: {"process": "notepad.exe"})
    monkeypatch.setattr(calls.winutil, "send_keys",
                        lambda *k: (keys.append(k), True)[1])

    app = calls.CALL_APPS[0]
    assert calls._act(app, {"hwnd": 1}, app.accept) is False
    assert keys == [], "a keystroke was sent into an unrelated window"


def test_the_keystroke_is_sent_once_the_app_is_in_front(monkeypatch):
    keys: list[tuple] = []
    monkeypatch.setattr(calls.winutil, "focus_window", lambda hwnd: True)
    monkeypatch.setattr(calls.winutil, "foreground_window",
                        lambda: {"process": "whatsapp.exe"})
    monkeypatch.setattr(calls.winutil, "send_keys",
                        lambda *k: (keys.append(k), True)[1])

    app = calls.CALL_APPS[0]
    assert calls._act(app, {"hwnd": 1}, app.accept) is True
    assert keys == [("enter",)]


def test_answering_and_declining_use_different_keys():
    """A copy-paste that made both do the same thing would answer every call the
    user tried to reject."""
    for app in calls.CALL_APPS:
        assert app.accept != app.decline, app.name


def test_a_browser_call_says_the_tab_has_to_be_in_front():
    """A background tab receives no keystrokes at all, and there is no way to
    fix that from outside the browser — so it has to be said."""
    chat = next(a for a in calls.CALL_APPS if a.name == "Google Chat")
    assert "tab" in chat.note.lower()


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("answer the call", "answer_call"),
        ("pick up", "answer_call"),
        ("call uthao", "answer_call"),
        ("phone uthao", "answer_call"),
        ("decline the call", "decline_call"),
        ("hang up", "decline_call"),
        ("call kaat do", "decline_call"),
    ],
)
def test_call_commands_route_offline(utterance, expected):
    """These are said while something is ringing. A model round trip is time the
    call does not have."""
    intent = route(utterance)
    assert intent is not None and intent.skill == expected
