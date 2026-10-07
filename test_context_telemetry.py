from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from event_engine import bingx, data_retention, research


def test_depth_metrics_sort_levels_before_top_n():
    payload = {
        "bids": [[99, 1], [101, 2], [100, 3], [98, 4], [97, 5], [96, 100]],
        "asks": [[103, 1], [101, 2], [102, 3], [104, 4], [105, 5], [106, 100]],
        "T": 1234567890000,
    }
    out = bingx._research_depth_metrics(payload, mid_price=101.5)
    assert out["best_bid"] == 101
    assert out["best_ask"] == 101
    assert out["bids_top10"][0] == [101.0, 2.0]
    assert out["asks_top10"][0] == [101.0, 2.0]
    assert out["book_mid_price"] == pytest.approx(101.0)
    assert out["depth_reference_type"] == "MARK_PRICE"
    assert out["depth_reference_price"] == pytest.approx(101.5)
    assert out["timestamp_ms"] == 1234567890000


def test_research_context_funding_units_and_provenance(monkeypatch):
    def fake_get(endpoint_key, path, params, *, timeout_sec=3.0):
        if endpoint_key == "premiumIndex":
            return {"code": 0, "data": [{"symbol": "TEST-USDT", "markPrice": "101", "indexPrice": "100", "lastFundingRate": "0.001", "nextFundingTime": 123, "time": 456}]}
        if endpoint_key == "openInterest":
            return {"code": 0, "data": {"symbol": "TEST-USDT", "openInterest": "777", "time": 789}}
        if endpoint_key == "depth":
            return {"code": 0, "data": {"bids": [[100, 2]], "asks": [[101, 3]], "T": 999}}
        if endpoint_key == "trades":
            return {"code": 0, "data": [{"price": "100.5", "qty": "1", "quoteQty": "100.5", "buyerMaker": False, "time": 1000}]}
        raise AssertionError(endpoint_key)
    monkeypatch.setattr(bingx, "_research_public_get", fake_get)
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda x: str(x).upper())
    out = bingx.fetch_research_market_context("TEST-USDT", depth_limit=5, trades_limit=5, timeout_sec=1.0)
    assert out["context_schema_version"] == 2
    assert out["funding_rate"] == pytest.approx(0.001)
    assert out["funding_rate_pct"] == pytest.approx(0.1)
    assert out["funding_rate_unit"] == "DECIMAL_RATE"
    assert out["open_interest_raw"] == "777"
    assert out["open_interest_unit"] == "PROVIDER_NATIVE_UNSPECIFIED"
    assert out["capture_started_at_ms"] is not None
    assert out["capture_completed_at_ms"] >= out["capture_started_at_ms"]


def test_record_market_context_persists_decision_provenance_before_write(tmp_path: Path, monkeypatch):
    for name in ["MARKET_CONTEXT_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH"]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    row = {
        "scan_id": "S1", "event_id": "E1", "symbol": "TEST-USDT", "provider": "binance",
        "source": "pre_execution_entry_context", "captured_at_ms": 1000,
        "captured_at": "1970-01-01T00:00:01Z", "decision_ts": "1970-01-01T00:00:02Z",
        "context_age_ms_at_decision": 1000, "capture_phase": "PRE_EXECUTION",
    }
    assert research.record_market_context(row) is True
    stored = json.loads((tmp_path / "market_context.jsonl").read_text().splitlines()[0])
    assert stored["decision_ts"] == row["decision_ts"]
    assert stored["context_age_ms_at_decision"] == 1000
    assert stored["context_capture_phase"] == "PRE_EXECUTION"
    assert stored["provider"] == "bingx"
    assert stored["analysis_provider"] == "binance"
    assert stored["context_provider"] == "bingx"
    assert stored["context_source"] == "bingx_swap_public"
    assert stored["recorded_at"]


def test_pre_execution_context_stays_out_of_causal_flattened_features():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=40, freq="1h", tz="UTC"),
        "open": [100.0] * 40, "high": [101.0] * 40, "low": [99.0] * 40, "close": [100.0] * 40, "volume": [1000.0] * 40,
    })
    bar = {"timestamp": "2026-01-02T00:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 200.0}
    signal = {"event_id":"E1","symbol":"TEST-USDT","type":"LONG","trigger_bar_time":bar["timestamp"],"decision_boundary_ts":"2026-01-02T00:05:00Z","entry":100.5,"zone":{},"entry_bar":bar,"trigger":{},"zone_visit":{}}
    context = {"context_id":"MC_1","status":"ok","capture_phase":"PRE_EXECUTION","feature_time_semantics":"CAPTURED_PRE_ORDER_NOT_TRIGGER_BOUNDARY","captured_at_ms":1767312300000,"capture_completed_at_ms":1767312300200,"decision_boundary_ts":"2026-01-02T00:05:00Z","funding_rate":0.001,"funding_rate_pct":0.1,"order_book":{"book_imbalance_5":0.3},"recent_trades":{"aggressor_delta_quote":123.0}}
    out = research._observation_from_signal(signal,scan_id="S1",strategy_version="v1",code_commit_sha="abc",df_1h=df,df_5m=pd.DataFrame(),decision_ts=signal["decision_boundary_ts"],provider="binance",source="test",market_context=context)
    assert out["market_context_id"] == "MC_1"
    assert out["features"]["pre_execution_context"]["funding_rate_pct"] == pytest.approx(0.1)
    assert "funding_rate" not in out["features"]
    assert out["features"]["pre_execution_context"]["order_book"]["book_imbalance_5"] == pytest.approx(0.3)
    assert "context_age_ms_at_execution_call_start" in out["features"]["pre_execution_context"]


def test_retention_has_market_context_policy():
    policies = data_retention._size_guard_plan(Path("market_context.jsonl"), now=data_retention._utc_now(), pending_ids=set(), pending_bar_cutoffs={}, protected_observation_ids=set(), protected_refs=set())
    assert policies
    assert any("market_context" in name for name, _ in policies)


def test_signal_boundary_helper_uses_closed_5m_bar_close():
    import run_once
    signal = {"trigger_bar_time": "2026-01-01T00:00:00Z"}
    value = run_once._signal_decision_boundary_ts_ms(signal)
    assert pd.Timestamp(value, unit="ms", tz="UTC") == pd.Timestamp("2026-01-01T00:05:00Z")
    signal["decision_boundary_ts"] = "2026-01-01T00:06:00Z"
    value = run_once._signal_decision_boundary_ts_ms(signal)
    assert pd.Timestamp(value, unit="ms", tz="UTC") == pd.Timestamp("2026-01-01T00:05:00Z")


def test_record_market_context_round_trips_context_id_to_observation(tmp_path: Path, monkeypatch):
    from event_engine import research
    for name in ["MARKET_CONTEXT_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH"]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    row = {
        "scan_id": "S_CTX", "event_id": "E_CTX", "attempt_id": "A_CTX",
        "symbol": "TEST-USDT", "provider": "bingx", "captured_at_ms": 1791307900000,
        "capture_phase": "PRE_EXECUTION",
    }
    assert research.record_market_context(row) is True
    assert row["context_id"].startswith("MC_")
    assert row["persisted"] is True
    assert row["persisted_at_ms"] >= row["persistence_started_at_ms"]

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=40, freq="1h", tz="UTC"),
        "open": [100.0] * 40, "high": [101.0] * 40, "low": [99.0] * 40,
        "close": [100.0] * 40, "volume": [1000.0] * 40,
    })
    bar = {"timestamp": "2026-01-02T00:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 200.0}
    signal = {
        "event_id": "E_CTX", "symbol": "TEST-USDT", "type": "LONG",
        "trigger_bar_time": bar["timestamp"], "decision_boundary_ts": "2026-01-02T00:05:00Z",
        "entry": 100.5, "zone": {}, "entry_bar": bar, "trigger": {}, "zone_visit": {},
    }
    obs = research._observation_from_signal(
        signal, scan_id="S_CTX", strategy_version="v1", code_commit_sha="abc",
        df_1h=df, df_5m=pd.DataFrame(), decision_ts=signal["decision_boundary_ts"],
        provider="binance", source="test", market_context=row,
    )
    assert obs["market_context_id"] == row["context_id"]
    assert obs["market_context_persisted"] is True
    assert obs["market_context_status"] == "PERSISTED"
    assert obs["market_context_persisted_at_ms"] == row["persisted_at_ms"]
    assert obs["market_context_persist_write_started_at_ms"] == row["persist_write_started_at_ms"]


def test_build_5m_signal_uses_canonical_epoch_ms_trigger_close_and_causal_1h_prefix(monkeypatch):
    import run_once
    trigger = pd.Timestamp("2026-10-06T17:25:00Z")
    h1_open = pd.date_range("2026-10-06T00:00:00Z", periods=20, freq="1h", tz="UTC")
    df1 = pd.DataFrame({
        "timestamp": h1_open,
        "close_time": h1_open + pd.Timedelta(hours=1),
        "open": [100.0] * 20, "high": [101.0] * 20,
        "low": [99.0] * 20, "close": [100.0] * 20,
        "volume": [1000.0] * 20, "atr50": [2.0] * 20,
    })
    bar = pd.Series({
        "timestamp": trigger,
        "close_time": int(pd.Timestamp("2026-10-06T17:29:59.999Z").timestamp() * 1000),
        "open": 99.0, "high": 106.0, "low": 94.0, "close": 100.0, "volume": 100.0,
    })
    zone = {"zone_id": "Z_CAUSAL_SIGNAL", "top": 101.0, "btm": 99.0, "poi": 100.0, "start": 0}
    monkeypatch.setattr(run_once, "_nearest_opposing_level", lambda *a, **k: {"price": 120.0, "source": "test"})
    old_obstacle = run_once.REQUIRE_STRUCTURE_OBSTACLE
    old_min_room = run_once.MIN_STRUCTURE_ROOM_R
    monkeypatch.setattr(run_once, "REQUIRE_STRUCTURE_OBSTACLE", False)
    monkeypatch.setattr(run_once, "MIN_STRUCTURE_ROOM_R", 0.0)
    try:
        signal = run_once._build_5m_zone_signal(
            "TEST-USDT", "LONG", zone, bar, None, df1, [zone], [], {"visit_id": "V1"},
            df_5m=pd.DataFrame(),
        )
    finally:
        run_once.REQUIRE_STRUCTURE_OBSTACLE = old_obstacle
        run_once.MIN_STRUCTURE_ROOM_R = old_min_room
    assert signal["trigger_bar_close_time"] == "2026-10-06T17:29:59.999000+00:00"
    assert signal["decision_boundary_kind"] == "CLOSED_5M_TRIGGER_CLOSE"
    assert pd.Timestamp(signal["decision_boundary_ts"]) == pd.Timestamp("2026-10-06T17:29:59.999Z")


def test_build_5m_signal_passes_only_causal_1h_frame_to_obstacle_lookup(monkeypatch):
    import run_once
    trigger = pd.Timestamp("2026-10-06T17:25:00Z")
    h1_open = pd.date_range("2026-10-06T00:00:00Z", periods=24, freq="1h", tz="UTC")
    df1 = pd.DataFrame({
        "timestamp": h1_open,
        "close_time": h1_open + pd.Timedelta(hours=1),
        "open": [100.0] * 24, "high": [101.0] * 24,
        "low": [99.0] * 24, "close": [100.0] * 24,
        "volume": [1000.0] * 24, "atr50": [2.0] * 24,
    })
    bar = pd.Series({
        "timestamp": trigger,
        "close_time": int(pd.Timestamp("2026-10-06T17:30:00Z").timestamp() * 1000),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100.0,
    })
    zone = {"zone_id": "Z_CASUAL_FRAME", "top": 101.0, "btm": 99.0, "poi": 100.0, "start": 0}
    seen = {}
    def fake_nearest(direction, entry, demand, supply, frame, current_idx):
        seen["last_close"] = frame.iloc[-1]["close_time"]
        seen["current_idx"] = current_idx
        seen["rows"] = len(frame)
        return None
    monkeypatch.setattr(run_once, "_nearest_opposing_level", fake_nearest)
    old_req = run_once.REQUIRE_STRUCTURE_OBSTACLE
    monkeypatch.setattr(run_once, "REQUIRE_STRUCTURE_OBSTACLE", False)
    try:
        run_once._build_5m_zone_signal(
            "TEST-USDT", "LONG", zone, bar, None, df1, [zone], [], {"visit_id": "V2"}, df_5m=pd.DataFrame()
        )
    finally:
        run_once.REQUIRE_STRUCTURE_OBSTACLE = old_req
    assert seen["last_close"] == pd.Timestamp("2026-10-06T17:00:00Z")
    assert seen["rows"] == 17
    assert seen["current_idx"] == 16
