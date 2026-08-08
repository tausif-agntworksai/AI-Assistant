# -*- coding: utf-8 -*-
"""Guards against Whisper's two characteristic failure modes."""

import pytest

from jarvis.stt.base import Transcript, is_probable_hallucination, is_repetition_loop


@pytest.mark.parametrize(
    "text",
    [
        "ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ ॐ",
        "yes yes yes yes yes yes yes yes yes yes",
        "हाहाहाहाहाहाहाहाहाहाहा",
        "thank you thank you thank you thank you thank you thank you",
    ],
)
def test_repetition_loops_are_caught(text):
    assert is_repetition_loop(text)


@pytest.mark.parametrize(
    "text",
    [
        "open chrome",
        "chrome kholo",
        "set a timer for five minutes and then open youtube",
        "बैटरी कितनी बची है",
        "no no",                      # short, legitimately repetitive
        "",
    ],
)
def test_real_commands_are_not_flagged(text):
    assert not is_repetition_loop(text)


@pytest.mark.parametrize("text", ["Thank you.", "you", "...", "शुक्रिया", " "])
def test_silence_filler_is_caught(text):
    assert is_probable_hallucination(text)


def test_real_command_is_not_filler():
    assert not is_probable_hallucination("open chrome")


def test_transcript_reports_language():
    assert Transcript(text="नमस्ते", language="hi").is_hindi
    assert not Transcript(text="hello", language="en").is_hindi
    assert Transcript(text="  ").is_empty
