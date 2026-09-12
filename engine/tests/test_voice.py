# -*- coding: utf-8 -*-
"""How Jarvis says things.

The point of a separate voice layer is that it sits downstream of every
decision. So the tests that matter most are the ones proving it *cannot* reach
the decision: wording changes, actions do not.
"""

import pytest
from pathlib import Path

from jarvis.voice import Voice, trim


# --- what gets trimmed -----------------------------------------------------


@pytest.mark.parametrize(
    "padded, plain",
    [
        ("Of course, the battery is at 65 percent.",
         "The battery is at 65 percent."),
        ("Well, actually, the meeting is at four.", "The meeting is at four."),
        ("Sure, opening Chrome now.", "Opening Chrome now."),
        ("As an AI assistant, I can open that for you.",
         "I can open that for you."),
        ("The meeting has moved to 4 PM. Let me know if you need anything else.",
         "The meeting has moved to 4 PM."),
    ],
)
def test_padding_is_removed(padded, plain):
    assert trim(padded) == plain


@pytest.mark.parametrize(
    "filler",
    ["Sure", "Of course.", "No problem", "You're welcome",
     "I'd be happy to help you with that.",
     "Sure! I'd be happy to help with that."],
)
def test_pure_filler_becomes_silence(filler):
    """Said aloud these are noise; the action already spoke for itself."""
    assert trim(filler) == ""


def test_a_fragment_is_never_produced():
    """Stripping "I'd be happy to" from the front leaves a bare infinitive.
    "Help you with that." is worse than either the original or silence, so
    verb-phrase openers are dropped whole rather than trimmed."""
    assert trim("Sure! I'd be happy to help you with that.") == ""


@pytest.mark.parametrize(
    "already_terse",
    [
        "Opening Chrome.",
        "Sent to Sana Ahmed on WhatsApp.",
        "I couldn't find Photoshop installed on this laptop.",
        "Battery is at 65 percent and charging.",
    ],
)
def test_a_terse_reply_is_untouched(already_terse):
    """Skills write their own short strings; this layer is aimed at prose."""
    assert trim(already_terse) == already_terse


def test_the_resolved_detail_survives():
    """"Sent to Sana Ahmed" names who actually got the message. Shortening
    that to "Done" would remove the one detail worth hearing."""
    assert "Sana Ahmed" in trim("Of course, sent to Sana Ahmed on WhatsApp.")


def test_hindi_is_left_alone():
    """The trimming list is English hedges; applying it to Devanagari mangles
    it, and Hindi replies come from the same skills."""
    hindi = "लैपटॉप को सुला रहा हूँ।"
    assert trim(hindi) == hindi


def test_capitalisation_is_restored_after_a_trim():
    assert trim("Of course, opening Chrome.") == "Opening Chrome."


def test_empty_input_is_handled():
    assert trim("") == ""
    assert trim("   ") == ""


# --- acknowledgements ------------------------------------------------------


def test_an_acknowledgement_never_repeats_itself_twice_running():
    """Randomness that repeats sounds like a bug, not like a person."""
    voice = Voice()
    said = [voice.acknowledge() for _ in range(12)]
    assert all(a != b for a, b in zip(said, said[1:]))


def test_hindi_gets_hindi_acknowledgements():
    assert any("\u0900" <= c <= "\u097f" for c in Voice().acknowledge("hi"))


# --- composing a turn ------------------------------------------------------


def test_an_action_with_nothing_to_say_still_gets_a_word():
    """Silence after "next track" is indistinguishable from not being heard."""
    assert Voice().compose("", acted=True) != ""


def test_a_turn_that_did_nothing_stays_silent():
    """Inventing "Done." for a turn that did nothing is how an assistant
    becomes untrustworthy."""
    assert Voice().compose("", acted=False) == ""


def test_filler_from_a_real_action_becomes_an_acknowledgement():
    assert Voice().compose("Sure", acted=True) != ""


def test_a_real_reply_is_preferred_over_an_acknowledgement():
    assert Voice().compose("Opening Chrome.", acted=True) == "Opening Chrome."


# --- the separation that justifies the module ------------------------------


def test_the_voice_layer_imports_nothing_that_can_act():
    """Personality is the part most likely to be tuned on a whim, so it must
    not be possible to change how Jarvis talks and change what Jarvis does.

    Checked by reading the imports rather than scanning the text — the prose
    in that module discusses skills at length, and a substring search would
    fail on its own documentation.
    """
    import ast

    from jarvis import voice as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
            imported.update(a.name for a in node.names)

    forbidden = {"registry", "skills", "orchestrator", "rules", "permissions",
                 "winutil", "registry_execute"}
    assert not (imported & forbidden), (
        f"voice.py imports {imported & forbidden} — it must only shape wording"
    )


def test_the_voice_layer_takes_no_decision_arguments():
    """Its inputs are a finished reply and a language. Not a skill, not an
    intent, not arguments it could alter."""
    import inspect

    params = set(inspect.signature(Voice.compose).parameters) - {"self"}
    assert params == {"reply", "language", "acted"}
