from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _bar(ts: pd.Timestamp, timeframe: str = "5m") -> dict:
    step = pd.Timedelta(hours=1) if timeframe == "1h" else pd.Timedelta(minutes=5)
    return {
        "timestamp": ts.isoformat(),
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 10.0,
        "close_time": int((ts + step).timestamp() * 1000),
    }


def _isolate_research_paths(tmp_path: Path, monkeypatch) -> None:
    from event_engine import research

    for name in [
        "ZONE_OBSERVATIONS_PATH", "MARKET_BARS_1H_PATH", "MARKET_BARS_5M_PATH",
        "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH",
    ]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    monkeypatch.setattr(research, "_OBSERVATION_SEEN_IDS", None)
    monkeypatch.setattr(research, "_OBSERVATION_SEEN_IDS_PATH", None)
    monkeypatch.setattr(research, "_PERSISTENCE_BATCH", None)


def test_pending_forward_without_new_boundary_keeps_both_timeframes():
    import run_once

    bars_1h = [{"timestamp": "hourly-source-bar"}]
    bars_5m = [{"timestamp": "five-minute-source-bar"}]
    kept_1h, kept_5m = run_once._research_bar_payload(
        research_enabled=True,
        has_research_boundary=False,
        persist_bars_for_forward=True,
        bars_1h=bars_1h,
        bars_5m=bars_5m,
    )
    assert kept_1h == bars_1h
    assert kept_5m == bars_5m

    dropped_1h, dropped_5m = run_once._research_bar_payload(
        research_enabled=True,
        has_research_boundary=False,
        persist_bars_for_forward=False,
        bars_1h=bars_1h,
        bars_5m=bars_5m,
    )
    assert dropped_1h == []
    assert dropped_5m == []


def test_effective_tp_levels_are_promoted_to_registration_setup():
    import run_once
    from event_engine.tracker import _extract_setup_metrics

    effective = [
        {"leg": "TP1", "pnl_pct": 1.2, "close_fraction": 0.35},
        {"leg": "TP2", "pnl_pct": 2.4, "close_fraction": 0.65},
    ]
    prepared = run_once._setup_with_protection_result(
        {"tp_levels": [{"leg": "TP1", "pnl_pct": 1.0}]},
        {"status": "PROTECTED", "effective_tp_levels": effective},
    )
    assert prepared["effective_tp_levels"] == effective
    assert prepared["protection_result"]["effective_tp_levels"] == effective
    assert _extract_setup_metrics(prepared)["effective_tp_levels"] == effective


def test_batch_cursor_failure_retry_does_not_duplicate_market_bars(tmp_path, monkeypatch):
    from event_engine import research

    _isolate_research_paths(tmp_path, monkeypatch)
    ts = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=3)
    bar = _bar(ts, "1h")
    real_atomic = research._write_json_atomic_unlocked
    did_fail = False

    def fail_cursor_once(path, payload):
        nonlocal did_fail
        if path == research.RESEARCH_BAR_CURSORS_PATH and not did_fail:
            did_fail = True
            raise OSError("simulated cursor commit failure after durable bar append")
        return real_atomic(path, payload)

    monkeypatch.setattr(research, "_write_json_atomic_unlocked", fail_cursor_once)
    with research.batch_persistence() as first:
        assert research.persist_market_bars(
            "AAA-USDT", "1h", [bar], provider="binance", source="test", scan_id="S1"
        ) == 1
    assert first["flush_result"]["persistence_errors"] >= 1
    bars_path = research.MARKET_BARS_1H_PATH
    assert bars_path.exists()
    assert len(bars_path.read_text(encoding="utf-8").splitlines()) == 1
    assert not research.RESEARCH_BAR_CURSORS_PATH.exists()

    # The cursor is still old, so the same source bar is offered to persistence.
    # Idempotent append recognizes it and repairs the cursor without duplicating it.
    with research.batch_persistence() as second:
        assert research.persist_market_bars(
            "AAA-USDT", "1h", [bar], provider="binance", source="test", scan_id="S2"
        ) == 1
    assert second["flush_result"]["persistence_errors"] == 0
    assert len(bars_path.read_text(encoding="utf-8").splitlines()) == 1
    cursor = json.loads(research.RESEARCH_BAR_CURSORS_PATH.read_text(encoding="utf-8"))
    assert cursor["AAA-USDT|1h|binance"]["last_timestamp"] == ts.isoformat()
    manifest = json.loads(research.RESEARCH_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["market_bars_1h_written"] == 1


def test_non_batch_cursor_failure_retry_does_not_duplicate_market_bars(tmp_path, monkeypatch):
    from event_engine import research

    _isolate_research_paths(tmp_path, monkeypatch)
    ts = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=4)
    bar = _bar(ts, "1h")
    real_update_cursor = research._update_cursor
    did_fail = False

    def fail_cursor_once(key, value):
        nonlocal did_fail
        if not did_fail:
            did_fail = True
            raise OSError("simulated non-batch cursor commit failure")
        return real_update_cursor(key, value)

    monkeypatch.setattr(research, "_update_cursor", fail_cursor_once)
    assert research.persist_market_bars(
        "AAA-USDT", "1h", [bar], provider="binance", source="test", scan_id="S1"
    ) == 0
    bars_path = research.MARKET_BARS_1H_PATH
    assert len(bars_path.read_text(encoding="utf-8").splitlines()) == 1
    assert not research.RESEARCH_BAR_CURSORS_PATH.exists()

    assert research.persist_market_bars(
        "AAA-USDT", "1h", [bar], provider="binance", source="test", scan_id="S2"
    ) == 0
    assert len(bars_path.read_text(encoding="utf-8").splitlines()) == 1
    cursor = json.loads(research.RESEARCH_BAR_CURSORS_PATH.read_text(encoding="utf-8"))
    assert cursor["AAA-USDT|1h|binance"]["last_timestamp"] == ts.isoformat()


def test_get_all_orders_recursively_splits_saturated_time_window(monkeypatch):
    from event_engine import bingx

    rows = [{"orderId": str(ts), "time": ts} for ts in range(100, 104)]
    calls = []

    def fake_request(method, path, params, signed=True):
        calls.append(dict(params))
        start, end, limit = params["startTime"], params["endTime"], params["limit"]
        matching = [row for row in rows if start <= row["time"] <= end]
        return {"code": 0, "data": matching[:limit]}

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda value: value)
    monkeypatch.setattr(bingx, "_request", fake_request)
    result = bingx.get_all_orders("AAA-USDT", 100, 103, limit=2)

    assert {row["orderId"] for row in result} == {"100", "101", "102", "103"}
    assert len(result) == 4
    assert len(calls) > 1
    assert all(call["limit"] == 2 for call in calls)


def test_get_all_orders_splits_ranges_longer_than_seven_days(monkeypatch):
    from event_engine import bingx

    max_window = 7 * 24 * 60 * 60 * 1000 - 1
    start = 1_000
    end = start + max_window + 10
    calls = []

    def fake_request(method, path, params, signed=True):
        calls.append(dict(params))
        return {"code": 0, "data": [{"orderId": str(params["startTime"]), "time": params["startTime"]}]}

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda value: value)
    monkeypatch.setattr(bingx, "_request", fake_request)
    result = bingx.get_all_orders("AAA-USDT", start, end, limit=1000)

    assert len(calls) == 2
    assert all(call["endTime"] - call["startTime"] <= max_window for call in calls)
    assert len(result) == 2


def test_get_all_orders_fails_closed_if_saturated_window_cannot_be_split(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda value: value)
    monkeypatch.setattr(
        bingx,
        "_request",
        lambda *args, **kwargs: {"code": 0, "data": [{"orderId": str(i)} for i in range(2)]},
    )
    with pytest.raises(RuntimeError, match="cannot be split safely"):
        bingx.get_all_orders("AAA-USDT", 100, 100, limit=2)


def test_unrequested_market_context_is_not_mislabeled_as_persistence_failure(tmp_path, monkeypatch):
    from event_engine import research

    _isolate_research_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(research, "MARKET_CONTEXT_PATH", tmp_path / "market_context.jsonl")

    index = pd.date_range("2026-01-01T00:00:00Z", periods=48, freq="1h")
    df_1h = pd.DataFrame({
        "timestamp": index,
        "open": [100.0] * len(index),
        "high": [101.0] * len(index),
        "low": [99.0] * len(index),
        "close": [100.0] * len(index),
        "volume": [1000.0] * len(index),
    })
    bar = {
        "timestamp": "2026-01-03T00:05:00Z",
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 200.0,
    }
    zone = {
        "zone_id": "Z_NO_CONTEXT", "top": 105.0, "btm": 95.0, "poi": 100.0,
        "origin_ts_ms": int(df_1h["timestamp"].iloc[0].timestamp() * 1000), "age_bars": 10,
    }
    signal = {
        "event_id": "E_NO_CONTEXT", "symbol": "TEST-USDT", "type": "LONG",
        "trigger_bar_time": bar["timestamp"], "decision_boundary_ts": "2026-01-03T00:10:00Z",
        "entry": 100.5, "sl": 90.45, "tp1": 103.515, "tp2": 106.53,
        "zone": zone, "zone_visit": {"visit_id": "V_NO_CONTEXT", "touch_count_before_trigger": 0},
        "entry_bar": bar, "previous_bar": {}, "target": {"obstacle_price": 120.0},
        "trigger": {"zone_trigger_mode": "zone"},
    }

    result = research.record_scan_symbol(
        scan_id="S_NO_CONTEXT", symbol="TEST-USDT", strategy_version="v1", code_commit_sha="abc",
        provider="binance", source="binance_spot", bars_1h=df_1h.to_dict("records"), bars_5m=[bar],
        df_1h=df_1h, demand=[zone], supply=[],
        diagnostics={"touch_events": [], "rearm_events": [], "zones": {}},
        symbol_state={}, signals=[signal], decision_ts="2026-01-03T00:10:00Z",
        market_context=None, account_context=None,
    )

    assert result["observations"] == 1
    observations = [json.loads(line) for line in (tmp_path / "zone_observations.jsonl").read_text().splitlines() if line.strip()]
    assert len(observations) == 1
    assert observations[0]["market_context_id"] is None
    assert observations[0]["market_context_persisted"] is None
    assert observations[0]["market_context_status"] == "NOT_COLLECTED"
    assert not (tmp_path / "market_context.jsonl").exists()
