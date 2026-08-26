"""The camera: open it, and take a photo without opening it.

Two skills that look similar and are not. `open_camera` hands the user the
Windows Camera app and gets out of the way. `take_photo` never shows a window at
all — it opens the device, waits for the sensor to settle, grabs one frame,
releases the device and saves the file. That is the one that matters: "click a
photo" asked out loud wants a photo, not an app to click a photo in.

**The device is released before the reply is spoken.** A webcam left open is a
webcam with its light on, and an assistant that leaves the light on after taking
one picture is one nobody will keep installed. Every path out of `_capture`
closes the handle, including the failures.

**Warm-up is not optional.** Nearly every webcam returns two or three black or
badly exposed frames while auto-exposure and auto-white-balance converge. Taking
the first frame is how you get a photo of a dark grey rectangle, so a few are
grabbed and thrown away first. That is most of the second this takes.
"""

from __future__ import annotations

import datetime as dt
import logging
import time

from .. import paths, winutil
from ..permissions import Risk
from .registry import fail, ok, skill

log = logging.getLogger(__name__)

#: Frames discarded before the one that gets saved. Measured against the usual
#: symptom: at 3 the first photo of a session was still visibly dark, at 8 it
#: was not, and 8 frames is about half a second on a 15 fps webcam.
WARMUP_FRAMES = 8

#: Giving up beats hanging. A camera held open by another app never becomes
#: available by being waited on, and the reply should say so rather than the
#: assistant going quiet.
OPEN_TIMEOUT = 6.0


def photo_dir():
    """Where photos go: Pictures\\Jarvis, created on demand."""
    directory = paths.pictures_dir() / "Jarvis"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@skill(
    name="open_camera",
    description="Open the camera app so the user can see the preview",
    risk=Risk.SAFE,
    category="camera",
    examples=[
        "open the camera", "camera kholo", "start the webcam",
        "camera chalu karo", "open webcam",
    ],
)
def open_camera() -> object:
    if not winutil.shell_open("microsoft.windows.camera:"):
        return fail("I couldn't open the camera app.",
                    "Camera app nahi khul paya.")
    return ok("Camera's open.", "Camera khol diya hai.")


@skill(
    name="take_photo",
    description=(
        "Take a photo with the webcam and save it. Does not open the camera app."
    ),
    risk=Risk.CONFIRM,
    category="camera",
    examples=[
        "take a photo", "click a photo", "photo le lo", "photo khinch do",
        "take a picture", "selfie le lo", "meri photo lo",
    ],
    confirm_en="Take a photo with the webcam?",
    confirm_hi="Camera se photo le loon?",
)
def take_photo() -> object:
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        # Consistent with the rest of the engine: a missing optional dependency
        # takes out one skill and says which, rather than failing at startup.
        log.info("take_photo unavailable: opencv is not installed")
        return fail(
            "Taking photos needs the camera library, which isn't installed.",
            "Photo lene ke liye camera library chahiye, jo installed nahi hai.",
        )

    frame, error = _capture(cv2)
    if frame is None:
        return fail(*error)

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = photo_dir() / f"photo_{stamp}.jpg"
    try:
        written = cv2.imwrite(str(path), frame)
    except Exception as exc:  # noqa: BLE001
        log.error("Could not write %s: %s", path, exc)
        written = False
    if not written:
        return fail("I took the photo but couldn't save it.",
                    "Photo li thi par save nahi ho payi.")

    log.info("Photo saved: %s", path)
    return ok(
        f"Photo taken and saved as {path.name}.",
        f"Photo le li, {path.name} naam se save ho gayi.",
        detail=str(path), path=str(path),
    )


def _capture(cv2):  # noqa: ANN001
    """Grab one settled frame, then release the device.

    Returns `(frame, None)` or `(None, (english, hindi))`. The device is closed
    on every path out of here — a webcam light left on after one photo is the
    thing that would make this feature unwelcome.
    """
    camera = None
    try:
        # CAP_DSHOW rather than the default backend: on Windows the MSMF backend
        # takes several seconds to open a device and sometimes never does, which
        # turns "take a photo" into a hang. DirectShow opens immediately.
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        deadline = time.monotonic() + OPEN_TIMEOUT
        while not camera.isOpened() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not camera.isOpened():
            return None, (
                "I couldn't get to the camera — another app may be using it, "
                "or there isn't one connected.",
                "Camera nahi mila — ho sakta hai koi doosri app use kar rahi ho, "
                "ya camera juda hi na ho.",
            )

        frame = None
        for _ in range(WARMUP_FRAMES):
            read, candidate = camera.read()
            if read and candidate is not None:
                frame = candidate

        if frame is None:
            return None, (
                "The camera opened but sent no picture. Check that it isn't "
                "covered or blocked in Windows privacy settings.",
                "Camera khula par tasveer nahi aayi. Dekh lijiye ki wo dhaka na "
                "ho, aur Windows privacy settings mein blocked na ho.",
            )
        return frame, None
    except Exception as exc:  # noqa: BLE001
        log.exception("Camera capture failed")
        return None, (
            "Something went wrong with the camera.",
            "Camera mein kuch dikkat aa gayi.",
        )
    finally:
        if camera is not None:
            try:
                camera.release()
            except Exception:  # noqa: BLE001 - nothing useful to do about it
                log.debug("Camera release failed", exc_info=True)


@skill(
    name="open_photos",
    description="Open the folder where Jarvis saves photos",
    risk=Risk.SAFE,
    category="camera",
    # "photos dikhao" is deliberately absent: the existing `open_folder` rule
    # claims it and opens the whole Pictures folder, which contains this one.
    # Listing it here would only teach the model a phrasing the rules answer
    # differently.
    examples=["show my photos", "open the photos folder",
              "open the jarvis photos folder"],
)
def open_photos() -> object:
    directory = photo_dir()
    if not winutil.shell_open(str(directory)):
        return fail("I couldn't open that folder.", "Folder nahi khul paya.")
    return ok("Here are your photos.", "Ye hain aapki photos.",
              detail=str(directory))
