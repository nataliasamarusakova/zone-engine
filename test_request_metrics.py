from __future__ import annotations

import logging
import threading

import pytest

from event_engine import request_metrics


class _Response:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code


_DEFAULT_ENABLED = request_metrics.enabled()


def setup_function() -> None:
    request_metrics._ENABLED = True  # Unit tests opt in; production default is disabled.
    request_metrics.reset()


def teardown_function() -> None:
    request_metrics.reset()
    request_metrics._ENABLED = _DEFAULT_ENABLED


def test_timed_request_preserves_response_and_groups_by_phase() -> None:
    expected = _Response(200)
    with request_metrics.symbol_context("AAAUSDT"):
        with request_metrics.phase("scan_symbol"):
            actual = request_metrics.timed_request(
                "binance", "/api/v3/klines", "GET", lambda: expected,
            )

    assert actual is expected
    rows = request_metrics.snapshot()["requests"]
    assert len(rows) == 1
    assert rows[0]["phase"] == "scan_symbol"
    assert rows[0]["service"] == "binance"
    assert rows[0]["endpoint"] == "/api/v3/klines"
    assert rows[0]["count"] == 1
    assert rows[0]["errors"] == 0
    assert rows[0]["outcomes"] == {"200": 1}
    assert request_metrics.snapshot()["http_details"][0]["symbol"] == "AAAUSDT"


def test_timing_diagnostics_are_silent_and_do_not_measure_when_disabled(monkeypatch, caplog):
    monkeypatch.setattr(request_metrics, "_ENABLED", False)
    request_metrics.reset()
    expected = _Response(200)
    assert request_metrics.timed_request("bingx", "/openApi/swap/v2/user/positions", "GET", lambda: expected) is expected
    request_metrics.record_operation("scan_symbol", "AAAUSDT", 1.0)
    with caplog.at_level(logging.INFO):
        request_metrics.log_summary(logging.getLogger("request_metrics_disabled_test"))
    snapshot = request_metrics.snapshot()
    assert snapshot["requests"] == []
    assert snapshot["operations"] == {}
    assert "HTTP_TIMING" not in caplog.text
    assert "OP_TIMING" not in caplog.text
    request_metrics.reset()


def test_timed_request_counts_http_errors_and_transport_exceptions() -> None:
    request_metrics.timed_request(
        "bingx", "/private/positions", "GET", lambda: _Response(503),
    )

    def fail() -> None:
        raise TimeoutError("simulated timeout")

    with pytest.raises(TimeoutError, match="simulated timeout"):
        request_metrics.timed_request("bingx", "/private/positions", "GET", fail)

    rows = request_metrics.snapshot()["requests"]
    assert sum(row["count"] for row in rows) == 2
    assert sum(row["errors"] for row in rows) == 2
    outcomes = {outcome for row in rows for outcome in row["outcomes"]}
    assert "503" in outcomes
    assert "EXC:TimeoutError" in outcomes


def test_phase_is_thread_local_and_operation_summary_is_recorded(caplog: pytest.LogCaptureFixture) -> None:
    observed: list[str] = []

    def worker() -> None:
        observed.append(request_metrics.current_phase())
        with request_metrics.phase("worker_scan"):
            observed.append(request_metrics.current_phase())
        observed.append(request_metrics.current_phase())

    with request_metrics.phase("main_tracker"):
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert request_metrics.current_phase() == "main_tracker"

    assert observed == ["main", "worker_scan", "main"]
    request_metrics.record_operation("scan_symbol", "AAAUSDT", 0.25, outcome="ok")
    request_metrics.record_operation("scan_symbol", "BBBUSDT", 0.75, outcome="error")

    with caplog.at_level(logging.INFO):
        request_metrics.log_summary(logging.getLogger("request_metrics_test"))

    assert "[OP_TIMING] operation=scan_symbol count=2" in caplog.text
    assert "[OP_SLOWEST] operation=scan_symbol label=BBBUSDT seconds=0.750 outcome=error" in caplog.text


def test_metrics_keep_exact_operation_average_and_late_slowest_items(monkeypatch, caplog):
    monkeypatch.setattr(request_metrics, "_MAX_SAMPLES_PER_KEY", 2)
    request_metrics.reset()
    for label, seconds in (("fast-1", 0.01), ("fast-2", 0.02), ("very-slow-later", 99.0)):
        request_metrics.record_operation("scan_symbol", label, seconds)
        request_metrics._record_request(
            phase="scan_symbol", service="bingx", endpoint="/test", method="GET",
            seconds=seconds, outcome="200", failed=False,
        )

    snap = request_metrics.snapshot()
    operation = snap["operations"]["scan_symbol"]
    request = next(row for row in snap["requests"] if row["endpoint"] == "/test")
    assert operation["count"] == 3
    assert operation["total_seconds"] == pytest.approx(99.03)
    assert len(operation["samples"]) == 2
    assert [item["label"] for item in operation["items"]][0] == "very-slow-later"
    assert request["count"] == 3
    assert request["total_seconds"] == pytest.approx(99.03)
    assert request["max_seconds"] == pytest.approx(99.0)
    assert len(request["samples"]) == 2
    assert snap["http_details"][0]["seconds"] == pytest.approx(99.0)

    with caplog.at_level(logging.INFO):
        request_metrics.log_summary(logging.getLogger("request_metrics_bounded_test"))
    assert "avg_s=33.010" in caplog.text
    assert "label=very-slow-later seconds=99.000" in caplog.text
