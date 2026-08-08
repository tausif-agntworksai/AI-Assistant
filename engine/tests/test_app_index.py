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
