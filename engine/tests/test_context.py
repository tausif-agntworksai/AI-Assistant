# -*- coding: utf-8 -*-
"""Carrying a referent from one turn to the next.

"Message Sana" then "tell her I'll be late" is the whole feature. The risk it
carries is the opposite one — substituting a pronoun that was never a
reference, and turning a working command into a broken one — so most of what
is pinned here is where the substitution must *not* happen.
"""

import time

import pytest

from jarvis.context import (
    DEFAULT_TTL_SEC,
    TurnMemory,
    repair_referent_args,
    resolve_object_pronoun,
    resolve_pronouns,
)
from jarvis.nlu.rules import route
from jarvis.skills import load_all

load_all()


@pytest.fixture
def memory():
    """A memory holding exactly what the real skills hand back."""
    mem = TurnMemory()
    mem.record("open_app", {"app": "WhatsApp", "kind": "uwp"}, ok=True)
    mem.record("send_message",
               {"to": "919876543210", "contact": "Sana Ahmed",
                "platform": "whatsapp"},
               ok=True)
    return mem


# --- recording -------------------------------------------------------------


def test_a_skill_reports_what_it_resolved(memory):
    assert memory.recall("contact").name == "Sana Ahmed"
    assert memory.recall("app").name == "WhatsApp"


def test_a_messaging_platform_is_not_the_last_app():
    """Sending on WhatsApp should not make "close it" mean the chat service
    rather than the window the user actually opened."""
    mem = TurnMemory()
    mem.record("open_app", {"app": "Google Chrome"}, ok=True)
    mem.record("send_message", {"contact": "Sana Ahmed", "platform": "whatsapp"},
               ok=True)
    assert mem.recall("app").name == "Google Chrome"


def test_a_failed_skill_teaches_nothing():
    """Opening an app that isn't installed must not become the referent."""
    mem = TurnMemory()
    mem.record("open_app", {"app": "Photoshop"}, ok=False)
    assert mem.recall("app") is None


def test_a_skill_that_resolved_nothing_is_ignored():
    mem = TurnMemory()
    mem.record("get_battery", {"percent": 65}, ok=True)
    assert mem.snapshot() == {}


def test_the_newest_referent_wins():
    mem = TurnMemory()
    mem.record("open_app", {"app": "Chrome"}, ok=True)
    mem.record("open_app", {"app": "Spotify"}, ok=True)
    assert mem.recall("app").name == "Spotify"


def test_a_stale_referent_is_not_offered():
    """An hour later, "her" is a guess rather than a reference."""
    mem = TurnMemory(ttl=0.01)
    mem.record("send_message", {"contact": "Sana Ahmed"}, ok=True)
    time.sleep(0.02)
    assert mem.recall("contact") is None
    assert mem.snapshot() == {}


def test_the_default_window_is_minutes_not_hours():
    assert 60 <= DEFAULT_TTL_SEC <= 900


# --- substituting ----------------------------------------------------------


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("tell her I'll be late", "tell Sana Ahmed I'll be late"),
        ("tell them I'm on my way", "tell Sana Ahmed I'm on my way"),
        ("message him the address", "message Sana Ahmed the address"),
    ],
)
def test_an_english_pronoun_becomes_the_name(memory, spoken, expected):
    assert resolve_pronouns(spoken, memory) == expected


@pytest.mark.parametrize(
    "spoken, expected",
    [
        ("usko bol do ki main late hoon", "Sana Ahmed ko bol do ki main late hoon"),
        ("unko message bhejo", "Sana Ahmed ko message bhejo"),
        ("usse pooch lo", "Sana Ahmed se pooch lo"),
    ],
)
def test_a_hindi_pronoun_keeps_its_case_particle(memory, spoken, expected):
    """"usko" is "us" + "ko"; dropping the particle leaves a sentence that
    parses as nothing at all."""
    assert resolve_pronouns(spoken, memory) == expected


def test_the_section_28_scenario_routes(memory):
    """The brief's example, end to end through the matcher and the repair."""
    spoken = "tell her I'll be late"
    intent = route(resolve_pronouns(spoken, memory))
    assert intent is not None and intent.skill == "send_message"

    # The rule splits on the first space after the verb, so a two-word name
    # leaves it holding "sana" and a message of "ahmed I'll be late".
    repair_referent_args(intent.args, spoken, memory)
    assert intent.args["to"] == "Sana Ahmed"


def test_the_repair_leaves_a_named_recipient_alone(memory):
    """A turn about somebody else must not be redirected to the last contact."""
    args = {"to": "rohit", "message": "hello"}
    repair_referent_args(args, "tell her hello", memory)
    assert args["to"] == "rohit"


def test_the_repair_does_nothing_without_a_pronoun(memory):
    args = {"to": "sana", "message": "hi"}
    repair_referent_args(args, "message sana saying hi", memory)
    assert args["to"] == "sana"


def test_a_pronoun_with_no_referent_is_left_alone():
    """Reaching the model unchanged is a worse answer than substituting, and a
    much better one than substituting the wrong person."""
    spoken = "tell her I'll be late"
    assert resolve_pronouns(spoken, TurnMemory()) == spoken


def test_a_sentence_that_already_names_the_person_is_untouched(memory):
    spoken = "tell Sana Ahmed her order arrived"
    assert resolve_pronouns(spoken, memory) == spoken


def test_only_the_first_pronoun_is_replaced(memory):
    """The second one is almost always possessive — "tell her her order
    arrived" wants a name once, not twice."""
    out = resolve_pronouns("tell her her order arrived", memory)
    assert out == "tell Sana Ahmed her order arrived"


def test_an_utterance_with_no_pronoun_is_untouched(memory):
    spoken = "open chrome"
    assert resolve_pronouns(spoken, memory) == spoken


# --- what must NOT be substituted -----------------------------------------


@pytest.mark.parametrize("spoken", ["turn it up", "increase it", "make it louder"])
def test_a_thing_pronoun_never_becomes_an_app(memory, spoken):
    """The regression this guards: "turn it up" is a working volume command,
    and rewriting "it" to the last app would break it."""
    assert resolve_pronouns(spoken, memory) == spoken
    assert resolve_object_pronoun(spoken, memory) == spoken


def test_close_it_is_the_one_object_pronoun_worth_resolving(memory):
    """Without this it reaches close_app with "it" as the application name."""
    assert resolve_object_pronoun("close it", memory) == "close WhatsApp"


def test_close_it_with_nothing_opened_is_left_alone():
    assert resolve_object_pronoun("close it", TurnMemory()) == "close it"


def test_a_named_close_is_untouched(memory):
    assert resolve_object_pronoun("close chrome", memory) == "close chrome"


# --- what the rest of the system sees -------------------------------------


def test_the_snapshot_is_what_a_skill_receives(memory):
    assert memory.snapshot() == {"app": "WhatsApp", "contact": "Sana Ahmed"}


def test_the_description_reads_as_a_sentence_for_the_model(memory):
    described = memory.describe()
    assert "Sana Ahmed" in described and "WhatsApp" in described


def test_clearing_forgets_everything(memory):
    memory.clear()
    assert memory.snapshot() == {}
    assert resolve_pronouns("tell her hello", memory) == "tell her hello"


# --- the split a two-word name causes -------------------------------------


def test_the_surname_does_not_leak_into_the_message(memory):
    """"tell Sana Ahmed I'll be late" splits on the first space after the
    verb, so the rule reads a recipient of "sana" and a message beginning
    "Ahmed". Sending that would put the surname in the message."""
    spoken = "tell her I'll be late"
    intent = route(resolve_pronouns(spoken, memory))
    repair_referent_args(intent.args, spoken, memory)
    assert intent.args["to"] == "Sana Ahmed"
    assert intent.args["message"] == "I'll be late"


def test_a_hindi_phrasing_keeps_its_whole_message(memory):
    spoken = "usko bol do ki main late hoon"
    intent = route(resolve_pronouns(spoken, memory))
    repair_referent_args(intent.args, spoken, memory)
    assert intent.args["to"] == "Sana Ahmed"
    assert intent.args["message"] == "main late hoon"


def test_a_name_inside_the_message_survives(memory):
    """Only the copy the substitution put there is removed.

    "tell her Ahmed is coming too" becomes "tell Sana Ahmed Ahmed is coming
    too", and the rule hands back a message starting with two Ahmeds. Exactly
    one of them belongs to the surname; the other is a person being talked
    about, and stripping it too would change what was said.
    """
    spoken = "tell her Ahmed is coming too"
    intent = route(resolve_pronouns(spoken, memory))
    repair_referent_args(intent.args, spoken, memory)
    assert intent.args["to"] == "Sana Ahmed"
    assert intent.args["message"] == "Ahmed is coming too"


def test_a_recipient_the_memory_does_not_explain_is_left_alone(memory):
    """An over-broad rule match must not be rewritten into a real contact."""
    args = {"to": "sana ahmed the address"}
    repair_referent_args(args, "message him the address", memory)
    assert args["to"] == "sana ahmed the address"
