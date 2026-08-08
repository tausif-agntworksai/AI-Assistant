# -*- coding: utf-8 -*-
"""Normalisation: Devanagari transliteration, numbers, language detection."""

import pytest

from jarvis.nlu.normalize import (
    has_devanagari,
    looks_hindi,
    normalize,
    parse_number,
    parse_percentage,
)


@pytest.mark.parametrize(
    "devanagari, expected",
    [
        ("क्रोम खोलो", "chrome kholo"),
        ("व्हाट्सएप्प चालू करो", "whatsapp chalu karo"),
        ("बंद करो", "band karo"),
        ("लैपटॉप सुला दो", "laptop sula do"),
        ("यूट्यूब खोलो", "youtube kholo"),
        ("कंप्यूटर बंद करो", "computer band karo"),
        ("स्क्रीनशॉट लो", "screenshot lo"),
        ("आवाज़ बढ़ाओ", "avaz badhao"),
    ],
)
def test_devanagari_transliterates_to_matchable_roman(devanagari, expected):
    assert normalize(devanagari) == expected


def test_long_aa_survives_schwa_deletion():
    """"gaanaa" must not lose its final long vowel the way "banda" loses its schwa."""
    assert normalize("गाना") == "song"      # via the spelling-fix table
    assert normalize("लगा") == "laga"
    assert normalize("कैसा") == "kaisa"
    assert normalize("बंद") == "band"       # schwa correctly dropped


def test_filler_is_stripped_but_never_everything():
    assert normalize("Jarvis, please open chrome") == "open chrome"
    assert normalize("jarvis") == "jarvis"  # would otherwise normalise to nothing


def test_devanagari_detection():
    assert has_devanagari("बैटरी")
    assert not has_devanagari("battery")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("volume fifty kar do", 50),
        ("volume pachaas kar do", 50),
        ("set volume to 70", 70),
        ("paanch minute ka timer", 5),
        ("five minute timer", 5),
        ("das minute baad", 10),
        ("brightness thirty percent", 30),
        ("पचास कर दो", 50),
        ("पांच मिनट", 5),
        ("do minute ka timer", 2),
        ("timer for twenty five minutes", 25),
        ("volume badhao", None),
    ],
)
def test_numbers_in_both_languages(text, expected):
    assert parse_number(text) == expected


def test_percentage_is_clamped():
    assert parse_percentage("volume 500 kar do") == 100
    assert parse_percentage("no number here") is None


@pytest.mark.parametrize(
    "text",
    ["Chrome kholo", "laptop sulado", "battery kitni hai", "gana chalao",
     "बैटरी कितनी बची है", "5 min timer laga do"],
)
def test_hindi_utterances_are_detected(text):
    assert looks_hindi(text)


@pytest.mark.parametrize(
    "text",
    ["open chrome", "what is the battery", "set the volume to 50",
     "play some music", "take a screenshot"],
)
def test_english_utterances_are_not_flagged_hindi(text):
    assert not looks_hindi(text)
