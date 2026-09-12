# -*- coding: utf-8 -*-
"""Turn timing. The instrument has to be cheaper than what it measures."""

import time

import pytest

from jarvis.metrics import STAGES, TurnTimer


def test_a_stage_records_its_duration():
    timer = TurnTimer()
    with timer.stage("stt"):
        time.sleep(0.01)
    assert timer.stages["stt"] >= 0.01


def test_re_entering_a_stage_adds_rather_than_replaces():
    """A re-decode is a second STT pass, and the user waited through both."""
    timer = TurnTimer()
    for _ in range(2):
        with timer.stage("stt"):
            time.sleep(0.01)
    assert timer.stages["stt"] >= 0.02


def test_a_failing_stage_is_still_timed():
    """Otherwise the slowest turns — the ones that broke — are the unmeasured ones."""
    timer = TurnTimer()
    try:
        with timer.stage("brain"):
            raise RuntimeError("provider down")
    except RuntimeError:
        pass
    assert "brain" in timer.stages


def test_the_summary_is_ordered_and_rounded():
    timer = TurnTimer()
    with timer.stage("act"):
        pass
    with timer.stage("stt"):
        pass
    summary = timer.summary()
    # Reported in pipeline order, not in the order the stages happened to run.
    assert list(summary["stages_ms"]) == ["stt", "act"]
    assert all(isinstance(v, int) for v in summary["stages_ms"].values())
    assert isinstance(summary["total_ms"], int)


def test_notes_travel_with_the_timings():
    timer = TurnTimer()
    timer.note(via="rule", skill="sleep_pc")
    assert timer.summary()["via"] == "rule"
    assert timer.summary()["skill"] == "sleep_pc"


def test_every_stage_name_used_is_one_the_summary_knows():
    """A stage missing from STAGES would be timed and then silently dropped."""
    timer = TurnTimer()
    for name in STAGES:
        with timer.stage(name):
            pass
    assert list(timer.summary()["stages_ms"]) == list(STAGES)


def test_timing_is_published_for_the_hud():
    """The HUD reads the bus, so a turn nobody can see is a turn nobody tunes."""
    from jarvis.bus import Event, bus

    timer = TurnTimer()
    with timer.stage("stt"):
        pass
    timer.done()

    published = [e for e in bus.history() if e.get("type") == Event.TIMING.value]
    assert published, "no timing event reached the bus"
    assert "stt" in published[-1]["stages_ms"]


def test_measuring_costs_almost_nothing():
    """The instrument must sit far below the resolution of what it measures."""
    timer = TurnTimer()
    start = time.perf_counter()
    for _ in range(1000):
        with timer.stage("route"):
            pass
    elapsed = time.perf_counter() - start
    assert elapsed < 0.05, f"1000 stages took {elapsed:.3f}s"


# --- durations measured outside the timer ----------------------------------


def test_mark_records_a_duration_inside_the_window():
    """Synthesis runs on the speaking thread; the window already covers it."""
    timer = TurnTimer()
    time.sleep(0.01)
    before = timer.total
    timer.mark("tts", 0.005)
    assert timer.stages["tts"] == 0.005
    # Marking must not stretch the total for something already inside it.
    assert timer.total == pytest.approx(before, abs=0.01)


def test_mark_earlier_extends_the_window_backwards():
    """Waking and recording finish before the timer exists.

    Without this the turn total omits the endpointing tail, which is most of
    a second the user waits through on every turn.
    """
    timer = TurnTimer()
    timer.mark_earlier("record", 1.5)
    assert timer.stages["record"] == 1.5
    assert timer.total >= 1.5


def test_the_two_kinds_of_mark_do_not_double_count():
    timer = TurnTimer()
    timer.mark_earlier("wake", 0.10)
    timer.mark_earlier("record", 0.90)
    timer.mark("tts", 0.30)
    # 1.0s of pre-timer work is in the total; the 0.3s of TTS was already there.
    assert timer.total >= 1.0
    assert timer.total < 1.5


def test_a_negative_duration_is_ignored():
    """A monotonic clock read out of order must not corrupt the budget."""
    timer = TurnTimer()
    timer.mark("stt", -5.0)
    timer.mark_earlier("record", -5.0)
    assert "stt" not in timer.stages and "record" not in timer.stages
