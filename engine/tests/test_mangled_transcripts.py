# -*- coding: utf-8 -*-
"""Commands that a real recogniser got wrong, and must still be understood.

Every transcript in this file is verbatim output from `faster-whisper` on
synthesised Hinglish, collected while benchmarking tiny/base/small/large-v3-turbo
(see docs/HEARING.md). None of them are invented: they are what the microphone
path actually delivers, apostrophes and French guesses included.

The point of testing at this layer is that these were **not** model failures to
be fixed with a bigger model — every size produced the same mangling, and the
recovery is in normalisation and matching. A test here is cheap; another 1.5 GB
of model weights is not.

The second half is the more important half. Recovering more commands means
letting weaker matches through, and a weak match on a *direction* can land on
the opposite skill. "Brightness come caro" scored 72 against `brightness_up`
when the user said down. Turning the screen up when asked to turn it down is
worse than admitting you did not understand, so those cases assert silence.
"""

import pytest

from jarvis.nlu.rules import route
from jarvis.skills import load_all

load_all()


# --- transcripts that used to be discarded ---------------------------------

# (what Whisper returned, the skill the user actually wanted)
RECOVERED = [
    ("Laptop s'olado.", "sleep_pc"),
    ("Laptop sous l'adot.", "sleep_pc"),
    ("laptop sulado", "sleep_pc"),
    ("laptop soulado", "sleep_pc"),
    ("Chrome kolo.", "open_app"),
    ("crome kholo", "open_app"),
    ("you tube kholo", "open_app"),
    ("panch minakt ka tamar laga do", "set_timer"),
    ("paj minute ka timer laga do", "set_timer"),
    ("skreenshot lo", "take_screenshot"),
    ("wolume kam karo", "volume_down"),
    ("brightnes kam caro", "brightness_down"),
]


@pytest.mark.parametrize(("transcript", "expected"), RECOVERED)
def test_a_mangled_command_still_reaches_the_right_skill(transcript, expected):
    intent = route(transcript)
    assert intent is not None, f"{transcript!r} was discarded entirely"
    assert intent.skill == expected


def test_the_number_survives_the_mangling():
    """Recovering the intent is only half of it — "panch" has to stay five."""
    intent = route("panch minakt ka tamar laga do")
    assert intent is not None and intent.args.get("minutes") == 5


# --- weak matches that must NOT be allowed to guess ------------------------


@pytest.mark.parametrize(
    "transcript",
    [
        "brightness calm",          # scored 76 against brightness_up
        "Brightness come caro.",    # scored 72 against brightness_up
        "volume something",
        "brightness thoda",
    ],
)
def test_an_ambiguous_direction_is_refused_rather_than_guessed(transcript):
    """No word here says which way. Silence sends this to the model, which can
    ask; a fuzzy match would just pick one."""
    intent = route(transcript, threshold=60)
    assert intent is None or intent.matched_by == "rule", (
        f"{transcript!r} was fuzzy-matched to {intent.skill} on no evidence"
    )


def test_a_fuzzy_match_never_supplies_a_magnitude():
    """`set_brightness(level: int = 60)` has a default, so nothing stopped a
    fuzzy match from silently setting the screen to 60%. A number is a decision
    the utterance has to contain."""
    intent = route("brightness calm", threshold=50)
    assert intent is None or "level" in intent.args


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("brightness kam karo", "brightness_down"),
        ("brightness badhao", "brightness_up"),
        ("volume kam", "volume_down"),
        ("volume zyada karo", "volume_up"),
        ("turn the brightness down", "brightness_down"),
        ("turn the brightness up", "brightness_up"),
    ],
)
def test_a_clear_direction_is_never_inverted(transcript, expected):
    intent = route(transcript)
    assert intent is not None and intent.skill == expected
