# -*- coding: utf-8 -*-
"""Which language the assistant answers in.

Answering a Hindi question in English is the most jarring thing a bilingual
assistant can do, and it was easy to trigger: Whisper's language label on a
two-word romanised command is close to a coin flip. Every case below is one
that used to be decided by that coin.
"""

import pytest

from jarvis.nlu.normalize import detect_language


@pytest.mark.parametrize("text", [
    "क्रोम खोलो",
    "बैटरी कितनी बची है",
    "लैपटॉप सुला दो",
])
def test_devanagari_is_always_hindi(text):
    """Script beats every other signal, including a confident wrong label."""
    assert detect_language(text, whisper_language="pl", whisper_probability=0.99) == "hi"


@pytest.mark.parametrize("text", [
    "chrome kholo",
    "volume badhao",
    "battery kitni bachi hai",
    "laptop sula do",
    "gana chalao",
])
def test_romanised_hindi_is_hindi_whatever_whisper_says(text):
    """The observed failure: Whisper labelled "Chrome kholo" Polish."""
    assert detect_language(text, whisper_language="pl", whisper_probability=0.9) == "hi"


@pytest.mark.parametrize("text", [
    "open chrome",
    "what is the battery level",
    "turn the volume up",
    "show me my reminders",
])
def test_plain_english_is_english(text):
    assert detect_language(text, whisper_language="en", whisper_probability=0.9) == "en"


def test_english_wins_over_an_unconfident_hindi_label():
    """A low-confidence label is not evidence; the words are."""
    assert detect_language("what is the time",
                           whisper_language="hi", whisper_probability=0.3) == "en"


def test_a_confident_hindi_label_carries_a_neutral_utterance():
    """No Hindi markers, no English markers — the label is all we have."""
    assert detect_language("spotify", whisper_language="hi",
                           whisper_probability=0.92) == "hi"


@pytest.mark.parametrize("label", ["ur", "mr", "ne", "bn"])
def test_languages_hindi_gets_mistaken_for_count_as_hindi(label):
    assert detect_language("theek", whisper_language=label,
                           whisper_probability=0.8) == "hi"


def test_an_ambiguous_followup_inherits_the_conversation():
    """"aur?" after a Hindi turn is Hindi, not a reset to English."""
    assert detect_language("aur", previous="hi") == "hi"
    assert detect_language("aur", previous="en") == "en"


def test_hinglish_with_an_english_verb_stays_hindi():
    """Mixed sentences are the normal case, not the exception."""
    assert detect_language("chrome band karo", whisper_language="en",
                           whisper_probability=0.7) == "hi"
