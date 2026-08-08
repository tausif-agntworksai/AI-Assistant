# -*- coding: utf-8 -*-
"""Intent routing across Hindi, English and Hinglish."""

import pytest

from jarvis.nlu.rules import route
from jarvis.skills import load_all

load_all()


# (utterance, expected skill). None means: no local rule should claim it, so
# it falls through to the LLM.
CASES = [
    # apps
    ("chrome kholo", "open_app"),
    ("open chrome", "open_app"),
    ("क्रोम खोलो", "open_app"),
    ("whatsapp chalu karo", "open_app"),
    ("youtube kholo", "open_app"),
    ("chrome band karo", "close_app"),
    ("close notepad", "close_app"),
    # volume — the up/down vs set-level distinction is easy to get wrong
    ("volume badhao", "volume_up"),
    ("आवाज़ बढ़ाओ", "volume_up"),
    ("volume kam karo", "volume_down"),
    ("turn up the volume", "volume_up"),
    ("volume 50 kar do", "set_volume"),
    ("volume pachaas kar do", "set_volume"),
    ("set volume to 30", "set_volume"),
    ("volume 50 do", "set_volume"),   # as Whisper actually renders it
    ("mute", "mute_audio"),
    ("awaz band karo", "mute_audio"),
    # display
    ("brightness badhao", "brightness_up"),
    ("set brightness to 60", "set_brightness"),
    # media
    ("next song", "media_next"),
    ("agla gana", "media_next"),
    ("pause the music", "media_play_pause"),
    ("gana chalao", "media_play_pause"),
    # power — the riskiest to misroute
    ("laptop sula do", "sleep_pc"),
    ("लैपटॉप सुला दो", "sleep_pc"),
    ("go to sleep", "sleep_pc"),
    ("lock the screen", "lock_screen"),
    ("computer band kar do", "shutdown_pc"),
    ("shutdown", "shutdown_pc"),
    ("restart karo", "restart_pc"),
    ("cancel shutdown", "cancel_shutdown"),
    # misc
    ("screenshot lo", "take_screenshot"),
    ("battery kitni hai", "get_battery"),
    ("kitna battery bacha hai", "get_battery"),
    ("what time is it", "get_time"),
    ("paanch minute ka timer laga do", "set_timer"),
    ("set a timer for 10 minutes", "set_timer"),
    ("youtube pe lofi chalao", "youtube_search"),
    ("search for python tutorials", "web_search"),
    ("open downloads", "open_folder"),
    ("show the desktop", "minimize_all"),
    ("close this window", "close_window"),
    ("aaj mausam kaisa hai", "get_weather"),
    ("weather in delhi", "get_weather"),
    ("mumbai ka mausam", "get_weather"),
    # Questions must not be read as "<thing> open" / "<thing> close" commands.
    ("what windows are open", "list_windows"),
    ("what's running", "list_running_apps"),
    ("what apps are running", "list_running_apps"),
    ("which apps are open", "list_running_apps"),
    ("how much ram do i have", "get_system_status"),
    ("kitni ram hai", "get_system_status"),
    # Nouns that look like app names but aren't. "gana" normalises to "song",
    # so these used to resolve to closing an app called Song / Screen.
    ("gana band karo", "media_stop"),
    ("screen band karo", "lock_screen"),
    # Read-back forms must not be stored as new records.
    ("reminder batao", "list_reminders"),
    ("notes padho", "list_notes"),
    ("read my notes", "list_notes"),
    # Settings pages are not applications.
    ("open sound settings", "open_settings"),
    ("open bluetooth settings", "open_settings"),
    # Hindi puts the verb last.
    ("chrome pe jao", "switch_to_app"),
    ("news", "get_news"),
]

REJECTS = [
    "why is the sky blue",
    "mujhe ek kahani sunao",
    "what did i do yesterday",
    "explain how photosynthesis works",
]


@pytest.mark.parametrize("text, expected", CASES)
def test_utterance_routes_to_expected_skill(text, expected):
    intent = route(text)
    assert intent is not None, f"{text!r} matched no rule"
    assert intent.skill == expected


@pytest.mark.parametrize("text", REJECTS)
def test_conversational_input_falls_through_to_the_llm(text):
    """Over-matching here would send a question to a skill instead of Claude."""
    assert route(text) is None


@pytest.mark.parametrize(
    "text, key, value",
    [
        ("chrome kholo", "app", "chrome"),
        ("open whatsapp", "app", "whatsapp"),
        ("volume pachaas kar do", "level", 50),
        ("set brightness to 60", "level", 60),
        ("paanch minute ka timer laga do", "minutes", 5),
        ("youtube pe lofi beats chalao", "query", "lofi beats"),
        # Dropping the city silently reports the weather wherever the IP says
        # you are, which looks like the skill working.
        ("weather in delhi", "city", "delhi"),
        ("what's the weather in new york", "city", "new york"),
        ("mumbai ka mausam", "city", "mumbai"),
        # "hello" used to be stripped as filler everywhere, which deleted the
        # very word being translated.
        ("translate hello to hindi", "text", "hello"),
        ("translate hello to hindi", "target_language", "hindi"),
        ("find files called resume", "query", "resume"),
        ("open sound settings", "page", "sound"),
        ("chrome pe jao", "app", "chrome"),
    ],
)
def test_arguments_are_extracted(text, key, value):
    intent = route(text)
    assert intent is not None
    assert intent.args.get(key) == value


def test_level_rules_do_not_invent_a_number():
    """"volume kam karo" must turn it down, not set it to a default level."""
    intent = route("volume kam karo")
    assert intent.skill == "volume_down"
    assert "level" not in intent.args
