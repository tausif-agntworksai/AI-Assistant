# -*- coding: utf-8 -*-
"""The shared candidate/score/confidence pipeline.

Applications and people go through the same mechanism with different numbers.
What is pinned here is the mechanism — and, most of all, the one place the two
domains deliberately disagree about what a confident score means.
"""

import pytest

from jarvis.resolve import Candidate, decide, rank


def _simple(query, names, floor=70, limit=5):
    from rapidfuzz import fuzz

    return rank(
        query, names,
        terms=lambda n: [n],
        name=lambda n: n,
        score=lambda needle, term: fuzz.WRatio(needle, term),
        floor=floor,
        limit=limit,
    )


def test_an_exact_term_scores_full_marks():
    hits = _simple("chrome", ["Chrome", "Chrome Remote Desktop"])
    assert hits[0].name == "Chrome"
    assert hits[0].score == 100.0


def test_nothing_below_the_floor_survives():
    assert _simple("zzzzzz", ["Chrome", "Spotify"]) == []


def test_an_empty_query_matches_nothing():
    assert _simple("", ["Chrome"]) == []


def test_the_shorter_name_wins_an_equal_score():
    """"Chrome" is a better answer to "chrome" than "Chrome Remote Desktop"."""
    hits = _simple("chrome", ["Chrome Remote Desktop", "Chrome"])
    assert hits[0].name == "Chrome"


def test_the_limit_is_respected():
    assert len(_simple("app", [f"App {n}" for n in range(20)], limit=3)) == 3


def test_an_adjuster_can_demote_a_candidate():
    """This is how an installed program outranks a web fallback."""
    from rapidfuzz import fuzz

    hits = rank(
        "youtube", [("YouTube", "web"), ("YouTube", "uwp")],
        terms=lambda row: [row[0]],
        name=lambda row: row[0],
        score=lambda needle, term: fuzz.WRatio(needle, term),
        floor=70,
        adjust=lambda row, score: score - 40 if row[1] == "web" else score,
    )
    assert hits[0].item[1] == "uwp"


# --- deciding --------------------------------------------------------------


def _candidates(*pairs):
    return [Candidate(item=name, name=name, score=score) for name, score in pairs]


def test_nothing_to_choose_from_resolves_to_nothing():
    outcome = decide([], strong=90, tie_window=6)
    assert outcome.best is None
    assert not outcome.found and not outcome.ambiguous


def test_a_clear_winner_is_returned():
    outcome = decide(_candidates(("Chrome", 95), ("Chromium", 71)),
                     strong=90, tie_window=6)
    assert outcome.found and outcome.best.name == "Chrome"


def test_two_close_candidates_are_a_question():
    outcome = decide(_candidates(("Office Home", 80), ("Office Tools", 78)),
                     strong=90, tie_window=6)
    assert outcome.ambiguous
    assert outcome.rivals == ("Office Tools",)


def test_a_strong_match_ends_the_question_for_things():
    """Both Visual Studio Codes score high; that says the names are alike."""
    outcome = decide(_candidates(("Visual Studio Code", 97),
                                 ("Visual Studio Code Insiders", 95)),
                     strong=92, tie_window=6)
    assert outcome.found and not outcome.ambiguous


def test_a_strong_match_does_not_end_the_question_for_people():
    """The parameter that separates the two domains.

    "Sana" scores 100 against both Sana Ahmed and Sana Khan. Treating that as
    certainty is how a private message reaches a stranger.
    """
    outcome = decide(_candidates(("Sana Ahmed", 100), ("Sana Khan", 100)),
                     strong=86, tie_window=8, strong_breaks_ties=False)
    assert outcome.ambiguous
    assert not outcome.found
    assert outcome.rivals == ("Sana Khan",)


def test_a_distant_runner_up_is_not_a_rival():
    outcome = decide(_candidates(("Sana Ahmed", 100), ("Sanjay", 74)),
                     strong=86, tie_window=8, strong_breaks_ties=False)
    assert outcome.found and outcome.best.name == "Sana Ahmed"


def test_everything_considered_is_reported():
    """The caller needs the losers to name them in a clarifying question."""
    outcome = decide(_candidates(("A", 80), ("B", 79), ("C", 78)),
                     strong=95, tie_window=6)
    assert [c.name for c in outcome.considered] == ["A", "B", "C"]
