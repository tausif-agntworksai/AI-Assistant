"""Turning what someone said into the thing they meant.

Two good implementations of the same pipeline had grown up independently —
`app_index.search` for applications and `messaging.resolve_recipient` for
people. Both generate candidates, score them, apply a floor, and decide
whether the top two are close enough to be worth asking about. Only the
numbers and the scorer differ.

This is that shape, once. It is deliberately *not* a single resolver with a
mode flag: the thresholds are the interesting part of each domain and they are
not interchangeable. Contacts are stricter than applications on purpose,
because opening the wrong program wastes a turn and messaging the wrong person
cannot be taken back. What is shared is the mechanism; what stays local is the
judgement.

    candidates ──score──▶ floor ──▶ sort ──▶ confident?  ──yes──▶ resolved
                                               │
                                               └──no──▶ tie? ──▶ ask which
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """One possible answer, and how well it matched."""

    item: Any
    name: str
    score: float


@dataclass(frozen=True)
class Resolution:
    """What the ranking came to — an answer, a question, or nothing.

    Three outcomes rather than two, because "I don't know who that is" and
    "which of these two did you mean?" need different replies, and collapsing
    them loses the only information that would let the caller ask a useful
    question.
    """

    best: Candidate | None = None
    rivals: tuple[str, ...] = ()
    considered: tuple[Candidate, ...] = field(default=())

    @property
    def found(self) -> bool:
        """A single answer good enough to act on."""
        return self.best is not None and not self.rivals

    @property
    def ambiguous(self) -> bool:
        return bool(self.rivals)


#: Scores a query against one of an item's searchable terms, 0-100.
Scorer = Callable[[str, str], float]
#: Nudges a raw score using the item itself — an installed app outranking a
#: web fallback, for instance. Applied after scoring, before the floor.
Adjuster = Callable[[Any, float], float]


def rank(
    query: str,
    items: Iterable[Any],
    *,
    terms: Callable[[Any], Sequence[str]],
    name: Callable[[Any], str],
    score: Scorer,
    floor: float,
    adjust: Adjuster | None = None,
    limit: int = 5,
) -> list[Candidate]:
    """Score every item against `query`, best first, dropping the hopeless.

    An exact match on any term short-circuits at 100 — there is nothing a
    fuzzy scorer can add once the user has said the name exactly, and letting
    it run only risks an adjustment pushing the perfect answer below a
    near-miss.
    """
    needle = (query or "").strip().lower()
    if not needle:
        return []

    scored: list[Candidate] = []
    for item in items:
        best = 0.0
        for entry in terms(item):
            # A term may carry a penalty, for names the index inferred rather
            # than was told. "Git CMD" ends in the word "cmd" by coincidence;
            # "Command Prompt" answers to it because somebody said so, and the
            # second claim should win.
            term, penalty = entry if isinstance(entry, tuple) else (entry, 0.0)
            term = (term or "").lower()
            if not term:
                continue
            if term == needle:
                best = max(best, 100.0 - penalty)
                if not penalty:
                    break
                continue
            best = max(best, float(score(needle, term)) - penalty)

        if adjust is not None:
            best = adjust(item, best)
        if best >= floor:
            scored.append(Candidate(item=item, name=name(item), score=min(100.0, best)))

    # Shorter names win a tie: "Chrome" is a better answer to "chrome" than
    # "Chrome Remote Desktop", and both score the same on a prefix.
    scored.sort(key=lambda c: (-c.score, len(c.name)))
    return scored[:limit]


def decide(
    candidates: Sequence[Candidate],
    *,
    strong: float,
    tie_window: float,
    strong_breaks_ties: bool = True,
) -> Resolution:
    """Turn a ranking into an answer, a question, or nothing.

    `strong` is the score at or above which the top candidate wins outright.
    Without it, a machine holding both "Visual Studio Code" and "Visual Studio
    Code Insiders" could not open either without being asked which — the
    near-duplicate says the *names* are similar, not that the user was
    unclear.

    `strong_breaks_ties` is where the two domains part company, and it is the
    single most consequential parameter here. For applications a confident top
    match ends the question. For people it must not: "Sana" against a list
    holding Sana Ahmed and Sana Khan scores 100 on both, and a high score
    there is evidence of a genuine ambiguity rather than of certainty.
    Guessing wastes a turn in one case and messages a stranger in the other.
    """
    if not candidates:
        return Resolution()

    best = candidates[0]
    if strong_breaks_ties and best.score >= strong:
        return Resolution(best=best, considered=tuple(candidates))

    rivals = tuple(
        c.name for c in candidates[1:] if best.score - c.score <= tie_window
    )
    if rivals:
        log.debug("Ambiguous %r: %s", best.name, ", ".join((best.name, *rivals)))
        return Resolution(best=best, rivals=rivals, considered=tuple(candidates))

    return Resolution(best=best, considered=tuple(candidates))
