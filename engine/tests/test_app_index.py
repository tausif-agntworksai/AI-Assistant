# -*- coding: utf-8 -*-
"""Fuzzy application resolution."""

import pytest

from jarvis.app_index import AppEntry, AppIndex


@pytest.fixture
def index():
    idx = AppIndex()
    idx.entries = [
        AppEntry(name="Google Chrome", launch=r"C:\chrome.lnk", kind="shortcut",
                 aliases=["chrome", "browser"]),
        AppEntry(name="WhatsApp", launch="shell:AppsFolder\\WA!App", kind="uwp",
                 aliases=["whatsapp", "whats app"]),
        AppEntry(name="Visual Studio Code", launch=r"C:\code.exe", kind="exe",
                 aliases=["vscode", "vs code", "code"]),
        AppEntry(name="Notepad", launch=r"C:\notepad.exe", kind="exe"),
        AppEntry(name="youtube", launch="https://www.youtube.com", kind="web"),
        AppEntry(name="Calculator", launch="shell:AppsFolder\\Calc!App", kind="uwp"),
    ]
    return idx


@pytest.mark.parametrize(
    "query, expected",
    [
        ("chrome", "Google Chrome"),
        ("google chrome", "Google Chrome"),
        ("whatsapp", "WhatsApp"),
        ("whats app", "WhatsApp"),
        ("vs code", "Visual Studio Code"),
        ("vscode", "Visual Studio Code"),
        ("notepad", "Notepad"),
        ("youtube", "youtube"),
        ("calculator", "Calculator"),
    ],
)
def test_resolves_spoken_names(index, query, expected):
    entry = index.resolve(query)
    assert entry is not None, f"{query!r} resolved to nothing"
    assert entry.name == expected


@pytest.mark.parametrize("query", ["spotify", "telegram", "photoshop", "zzzz"])
def test_uninstalled_apps_resolve_to_nothing(index, query):
    """A wrong launch is worse than admitting the app isn't there."""
    assert index.resolve(query) is None


def test_last_word_of_a_display_name_matches(index):
    assert index.resolve("code").name == "Visual Studio Code"


def test_scores_are_bounded(index):
    for _, score in index.search("whatsapp"):
        assert 0 <= score <= 100


def test_installed_apps_outrank_web_fallbacks():
    idx = AppIndex()
    idx.entries = [
        AppEntry(name="youtube", launch="https://www.youtube.com", kind="web"),
        AppEntry(name="YouTube", launch="shell:AppsFolder\\YT!App", kind="uwp"),
    ]
    assert idx.resolve("youtube").kind == "uwp"


def test_search_terms_include_aliases_and_last_word(index):
    terms = index.entries[0].search_terms
    assert "google chrome" in terms
    assert "chrome" in terms


# --- a near-miss must not become a wrong launch ----------------------------


@pytest.fixture
def photos_index():
    """The real shape of the bug: Photos installed, Photoshop not."""
    idx = AppIndex()
    idx.entries = [
        AppEntry(name="Photos", launch=r"shell:AppsFolder\Photos!App", kind="uwp"),
        AppEntry(name="Google Chrome", launch=r"C:\chrome.lnk", kind="shortcut",
                 aliases=["chrome", "browser"]),
        AppEntry(name="Visual Studio Code", launch=r"C:\code.exe", kind="exe"),
    ]
    return idx


def test_a_shared_prefix_is_not_a_match(photos_index):
    """"photoshop" scored 92 against "photos" and silently opened it."""
    assert photos_index.resolve("photoshop") is None


def test_a_whole_word_inside_a_longer_name_still_matches(photos_index):
    """The containment that should count: "code" names Visual Studio Code."""
    assert photos_index.resolve("code").name == "Visual Studio Code"
    assert photos_index.resolve("chrome").name == "Google Chrome"


@pytest.mark.parametrize(
    "spoken, expected",
    [("whats app", "WhatsApp"), ("note pad", "Notepad"), ("calculater", "Calculator")],
)
def test_mishearings_are_still_recovered(index, spoken, expected):
    """The stricter scorer must not undo what normalisation is there to catch."""
    assert index.resolve(spoken).name == expected


def test_a_confident_match_is_not_called_ambiguous(index):
    """An exact alias hit is the answer, not a question."""
    assert index.ambiguous("chrome") == []
    assert index.ambiguous("vs code") == []


def test_genuinely_close_candidates_are_still_raised():
    idx = AppIndex()
    idx.entries = [
        AppEntry(name="Office Home", launch="a", kind="exe"),
        AppEntry(name="Office Tools", launch="b", kind="exe"),
    ]
    assert len(idx.ambiguous("office")) > 1
