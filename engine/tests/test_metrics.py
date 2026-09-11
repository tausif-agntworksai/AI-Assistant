# -*- coding: utf-8 -*-
"""Turn timing. The instrument has to be cheaper than what it measures."""

import time

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
