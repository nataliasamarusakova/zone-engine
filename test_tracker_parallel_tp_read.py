from __future__ import annotations

import pytest


def test_shared_telemetry_reconciliation_journal_is_isolated_from_repo_data():
    from event_engine import telemetry

    production_path = telemetry.PROJECT_ROOT / "data" / "position_reconciliation.jsonl"
    isolated_path = telemetry.POSITION_RECONCILIATION_PATH
    assert isolated_path.resolve() != production_path.resolve()

    telemetry.record_position_reconciliation(
        event_id="EVT_TEST_ISOLATED_TELEMETRY",
        attempt_id="ATT_TEST_ISOLATED_TELEMETRY",
        position_id="POS_TEST_ISOLATED_TELEMETRY",
        symbol="AAA-USDT",
        direction="LONG",
        status="FOUND",
        internal_remaining_qty=1.0,
        exchange_position_qty=1.0,
    )
    assert isolated_path.exists()
    assert not isolated_path.resolve().is_relative_to(production_path.parent.resolve())


def test_thread_local_retry_session_uses_isolated_session_and_keeps_default(monkeypatch):
    from event_engine import bingx

    class FakeResponse:
        status_code = 200
        headers = {}

        @staticmethod
        def json():
            return {"code": 0, "data": {}}

    class FakeSession:
        def __init__(self):
            self.calls = []

        def request(self, **kwargs):
            self.calls.append(dict(kwargs))
            return FakeResponse()

    shared_session = FakeSession()
    worker_session = FakeSession()
    monkeypatch.setattr(bingx, "SESSION", shared_session)
    monkeypatch.setattr(bingx, "_get_retry_session", lambda: worker_session)
    monkeypatch.setattr(bingx, "_base_urls", lambda: ["https://bingx.test"])
    monkeypatch.setattr(bingx, "get_credentials", lambda: ("test-key", "test-secret"))

    normal = bingx._request("GET", "/test", {}, signed=True)

    from concurrent.futures import ThreadPoolExecutor

    def isolated_request():
        with bingx.thread_local_retry_session():
            return bingx._request("GET", "/test", {}, signed=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        isolated = executor.submit(isolated_request).result(timeout=2)

    assert normal["code"] == 0
    assert isolated["code"] == 0
    assert len(shared_session.calls) == 1
    assert len(worker_session.calls) == 1
    assert shared_session.calls[0]["url"].startswith("https://bingx.test/test?")
    assert worker_session.calls[0]["url"].startswith("https://bingx.test/test?")


def test_tracker_tp_worker_returns_value_and_preserves_exception(monkeypatch):
    from event_engine import tracker

    expected = {"status": "ok", "order_status": "NEW", "executed_qty": 0.0}
    monkeypatch.setattr(
        tracker,
        "_get_order_execution_evidence",
        lambda symbol, order_id, *, entry_ts=None: expected,
    )
    value, elapsed, error = tracker._fetch_tp_evidence_timed("AAA-USDT", "TP1", 123, "pre_scan_tracker")
    assert value is expected
    assert elapsed >= 0.0
    assert error is None

    original = RuntimeError("synthetic read failure")

    def failing_read(symbol, order_id, *, entry_ts=None):
        raise original

    monkeypatch.setattr(tracker, "_get_order_execution_evidence", failing_read)
    value, elapsed, error = tracker._fetch_tp_evidence_timed("AAA-USDT", "TP1", 123, "pre_scan_tracker")
    assert value is None
    assert elapsed >= 0.0
    assert error is original


def test_parallel_first_tp_read_can_be_disabled_without_code_changes(monkeypatch):
    from event_engine import tracker

    monkeypatch.delenv("TRACKER_PARALLEL_FIRST_TP_READ", raising=False)
    assert tracker._tracker_parallel_first_tp_read_enabled() is True

    for value in ("0", "false", "no", "off"):
        monkeypatch.setenv("TRACKER_PARALLEL_FIRST_TP_READ", value)
        assert tracker._tracker_parallel_first_tp_read_enabled() is False

    monkeypatch.setenv("TRACKER_PARALLEL_FIRST_TP_READ", "true")
    assert tracker._tracker_parallel_first_tp_read_enabled() is True


def test_first_tp_read_overlaps_klines_but_state_processing_remains_safe(monkeypatch, tmp_path):
    import json
    import threading

    from event_engine import tracker

    monkeypatch.setenv("TRACKER_PARALLEL_FIRST_TP_READ", "true")
    active = tmp_path / "active_trades.json"
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active)
    kline_started = threading.Event()
    order_started = threading.Event()
    worker_names = []
    saved = {}

    trade = {
        "event_id": "EVT_PARALLEL_READ_TEST",
        "symbol": "AAA-USDT",
        "direction": "LONG",
        "name": "AAA",
        "entry_price": 100.0,
        "initial_qty": 1.0,
        "remaining_qty": 1.0,
        "entry_ts": 1_000,
        "closed": False,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1", "qty": 0.5}],
        "sl_order": {"order_id": "SL1", "stop_price": 90.0},
        "hit_legs": [],
        "tp_filled_qty": {},
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "peak_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "planned_risk_pct": 10.0,
        "planned_weighted_rr": 0.45,
        "effective_weighted_rr": 0.45,
        "setup": {},
        "tp_levels": [],
        "effective_tp_levels": [],
    }
    active.write_text(json.dumps({trade["event_id"]: trade}), encoding="utf-8")

    def fake_fetch_klines(symbol, interval, limit=250, **kwargs):
        assert interval == "1m"
        kline_started.set()
        assert order_started.wait(timeout=2), "TP read did not start alongside klines"
        return []

    def fake_order_evidence(symbol, order_id, *, entry_ts=None):
        worker_names.append(threading.current_thread().name)
        order_started.set()
        assert kline_started.wait(timeout=2), "TP read was not overlapped with klines"
        return {"status": "ok", "order_status": "NEW", "executed_qty": 0.0, "avg_price": 0.0}

    monkeypatch.setattr(tracker, "fetch_klines", fake_fetch_klines)
    monkeypatch.setattr(tracker, "_get_order_execution_evidence", fake_order_evidence)
    monkeypatch.setattr(
        tracker,
        "get_position_directional",
        lambda symbol, direction: {
            "status": "found", "symbol": symbol, "positionSide": direction,
            "positionAmt": 1.0, "avgPrice": 100.0,
        },
    )
    monkeypatch.setattr(tracker.telemetry, "record_position_reconciliation", lambda **kwargs: None)
    monkeypatch.setattr(tracker, "_save_active_trades_after_reconciliation", lambda original, updated: saved.update(updated))

    tracker.update_active_trades()

    assert worker_names and worker_names[0].startswith("tracker-tp-read")
    assert kline_started.is_set() and order_started.is_set()
    updated = saved[trade["event_id"]]
    assert updated["remaining_qty"] == pytest.approx(1.0)
    assert updated["hit_legs"] == []
    assert updated["closed"] is False


def test_tracker_close_pnl_log_value_handles_unconfirmed_pnl():
    from event_engine import tracker

    assert tracker._pnl_log_value(None) == "unconfirmed"
    assert tracker._pnl_log_value(1.234) == "+1.23%"
    assert tracker._pnl_log_value(-0.125) == "-0.12%"


def test_tracker_clears_unverified_absence_only_after_positive_exchange_position():
    from event_engine import tracker

    trade = {
        "position_absence_unverified": True,
        "position_absence_unverified_ts": 1000,
        "position_absence_unverified_last_seen_ts": 2000,
        "position_absence_unverified_qty": 0.5,
        "position_absence_unverified_reason": "no_confirmed_residual_exit_order",
        "position_absence_unverified_alerted": True,
    }
    assert not tracker._clear_position_absence_unverified_if_position_found(
        trade, exchange_qty=0.0, now_ms=3000, qty_tolerance=1e-8,
    )
    assert trade["position_absence_unverified"] is True

    assert tracker._clear_position_absence_unverified_if_position_found(
        trade, exchange_qty=0.4, now_ms=4000, qty_tolerance=1e-8,
    )
    assert "position_absence_unverified" not in trade
    assert "position_absence_unverified_last_seen_ts" not in trade
    assert trade["position_absence_unverified_first_seen_ts"] == 1000
    assert trade["position_absence_unverified_resolved_ts"] == 4000
    assert trade["position_absence_unverified_resolution_reason"] == "exchange_position_reappeared"
