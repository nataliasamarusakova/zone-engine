from __future__ import annotations

import logging

import pytest

from event_engine import tracker


def test_timed_tracker_call_preserves_return_value_and_aggregates_time(monkeypatch) -> None:
    monkeypatch.setattr(tracker.request_metrics, "_ENABLED", True)
    timings: dict[str, dict[str, float]] = {}

    result = tracker._timed_tracker_call(timings, "sample", lambda value: value * 2, 21)

    assert result == 42
    assert timings["sample"]["calls"] == 1
    assert timings["sample"]["seconds"] >= 0.0
    assert timings["sample"]["max_seconds"] == timings["sample"]["seconds"]


def test_timed_tracker_call_records_and_reraises_original_exception(monkeypatch) -> None:
    monkeypatch.setattr(tracker.request_metrics, "_ENABLED", True)
    timings: dict[str, dict[str, float]] = {}

    def fail() -> None:
        raise RuntimeError("sentinel")

    with pytest.raises(RuntimeError, match="sentinel"):
        tracker._timed_tracker_call(timings, "sample_failure", fail)

    assert timings["sample_failure"]["calls"] == 1
    assert timings["sample_failure"]["seconds"] >= 0.0


def test_tracker_timing_log_is_compact_and_diagnostic_only(caplog: pytest.LogCaptureFixture, monkeypatch) -> None:
    monkeypatch.setattr(tracker.request_metrics, "_ENABLED", True)
    timings = {
        "kline_1m": {"calls": 2.0, "seconds": 1.25, "max_seconds": 0.90},
        "active_state_save": {"calls": 1.0, "seconds": 0.03, "max_seconds": 0.03},
    }

    with caplog.at_level(logging.INFO, logger="event_engine.tracker"):
        tracker._log_tracker_timings(timings, 1.50, 7)

    message = next(record.message for record in caplog.records if "[TRACKER_PERF_DETAIL]" in record.message)
    assert "total_s=1.500" in message
    assert "active_trades=7" in message
    assert "kline_1m_calls=2" in message
    assert "kline_1m_s=1.250" in message
    assert "active_state_save_calls=1" in message
