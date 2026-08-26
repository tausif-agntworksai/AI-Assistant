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


def test_a_genuinely_ambiguous_followup_inherits_the_conversation():
    """A bare app name is the same word in both languages, so the only signal
    left is what the conversation was already in."""
    assert detect_language("spotify", previous="hi") == "hi"
    assert detect_language("spotify", previous="en") == "en"


def test_a_whole_english_sentence_does_not_inherit_hindi():
    """The bug this guards: asking "explain quantum computing to me" straight
    after "chrome kholo" was answered in Hindi, because inheritance applied to
    an utterance far too long to be ambiguous."""
    for text in (
        "explain quantum computing to me",
        "give me three ideas for dinner",
        "who invented the telephone",
    ):
        assert detect_language(text, previous="hi") == "en", text


def test_a_whole_hindi_sentence_is_still_hindi_after_an_english_turn():
    for text in ("mujhe quantum computing samjhao", "aaj ka mausam kaisa hai"):
        assert detect_language(text, previous="en") == "hi", text


def test_a_technology_loanword_alone_does_not_flip_to_english():
    """"battery", "volume" and "file" live inside Hindi sentences constantly,
    so they are not evidence of anything."""
    for text in ("battery", "volume", "aur battery"):
        assert detect_language(text, previous="hi") == "hi", text


def test_hinglish_with_an_english_verb_stays_hindi():
    """Mixed sentences are the normal case, not the exception."""
    assert detect_language("chrome band karo", whisper_language="en",
                           whisper_probability=0.7) == "hi"
