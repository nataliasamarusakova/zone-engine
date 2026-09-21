from __future__ import annotations

import json
from pathlib import Path

import pytest

import pandas as pd

from event_engine import research
from research_forward import calculate_forward_outcome


def _df_1h() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=60, freq="1h", tz="UTC")
    close = [100 + i * 0.5 for i in range(len(ts))]
    return pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": [x + 1 for x in close],
        "low": [x - 1 for x in close],
        "close": close,
        "volume": [1000.0] * len(ts),
    })


def test_sanitize_nonfinite_values():
    payload = {"a": float("nan"), "b": float("inf"), "c": [1.0, float("-inf")]}
    assert research.sanitize(payload) == {"a": None, "b": None, "c": [1.0, None]}


def test_zone_observation_id_is_stable():
    a = research.observation_id("SIGNAL_CREATED", "BTC-USDT", "LONG", "ZID_1", "2026-01-01T00:05:00+00:00", visit_id="VISIT1")
    b = research.observation_id("SIGNAL_CREATED", "BTC-USDT", "LONG", "ZID_1", "2026-01-01T00:05:00+00:00", visit_id="VISIT1")
    assert a == b


def test_build_features_shadow_filters_do_not_depend_on_runtime_gate(monkeypatch):
    monkeypatch.setenv("REQUIRE_DIRECTIONAL_CANDLE", "false")
    zone = {"zone_id": "ZID_X", "top": 105.0, "btm": 95.0, "poi": 100.0, "origin_ts_ms": 1767225600000, "age_bars": 10}
    bar = {"timestamp": "2026-01-02T00:00:00+00:00", "open": 101.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 2000.0}
    f = research.build_research_features(symbol="TEST-USDT", direction="LONG", zone=zone, bar=bar, df_1h=_df_1h(), touch_count_before_trigger=0, structure_room_r=1.5)
    assert f["shadow_directional_candle_ok"] is not None
    assert f["shadow_penetration_le_20pct"] in {True, False}
    assert f["shadow_structure_ge_1_5R"] is True
    assert f["shadow_age_le_24h"] in {True, False}


def test_record_entry_decision_uses_requested_path(tmp_path: Path):
    path = tmp_path / "entry_decisions.jsonl"
    row = {"scan_id": "S1", "event_id": "E1", "stage": "CYCLE_CAP", "reason": "cap", "symbol": "TEST-USDT", "direction": "SHORT"}
    assert research.record_entry_decision(row, path=path) is True
    stored = json.loads(path.read_text(encoding="utf-8").strip())
    assert stored["decision_id"].startswith("DEC_")
    assert stored["schema_version"] == research.RESEARCH_SCHEMA_VERSION


def test_persist_market_bars_bootstrap_and_cursor(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(research, "DATA_DIR", tmp_path)
    monkeypatch.setattr(research, "MARKET_BARS_1H_PATH", tmp_path / "market_bars_1h.jsonl")
    monkeypatch.setattr(research, "MARKET_BARS_5M_PATH", tmp_path / "market_bars_5m.jsonl")
    monkeypatch.setattr(research, "RESEARCH_BAR_CURSORS_PATH", tmp_path / "research_bar_cursors.json")
    monkeypatch.setattr(research, "RESEARCH_MANIFEST_PATH", tmp_path / "research_manifest.json")
    monkeypatch.setattr(research, "RESEARCH_ERRORS_PATH", tmp_path / "research_persistence_errors.jsonl")
    rows = []
    for i in range(5):
        rows.append({"timestamp": f"2026-01-01T0{i}:00:00+00:00", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10})
    n1 = research.persist_market_bars("TEST-USDT", "1h", rows, provider="binance", source="binance_spot", scan_id="S1")
    n2 = research.persist_market_bars("TEST-USDT", "1h", rows, provider="binance", source="binance_spot", scan_id="S2")
    assert n1 == 5
    assert n2 == 0
    assert len((tmp_path / "market_bars_1h.jsonl").read_text().splitlines()) == 5


def test_forward_outcome_uses_only_bars_after_observation():
    ts = pd.date_range("2026-01-01 00:00", periods=6, freq="5min", tz="UTC")
    closes = [100.0, 99.0, 98.0, 97.0, 96.0, 95.0]
    bars = pd.DataFrame({"timestamp": ts, "open": closes, "high": [x + 0.5 for x in closes], "low": [x - 0.5 for x in closes], "close": closes, "volume": [1.0] * 6})
    obs = {"observation_id": "OBS_X", "event_id": "E_X", "scan_id": "S_X", "event_type": "TOUCH_REJECTED", "symbol": "TEST-USDT", "direction": "SHORT", "observation_ts": ts[1].isoformat(), "reference_price": 99.0}
    outcome = calculate_forward_outcome(obs, bars)
    assert outcome is not None
    assert outcome["forward_return_5m_pct"] == pytest.approx(0.0)
    assert outcome["time_to_plus_3pct_min"] == pytest.approx(20.0, abs=1e-6)
    assert outcome["time_to_minus_3pct_min"] is None


def test_record_scan_symbol_writes_signal_and_rejected_observations(tmp_path: Path, monkeypatch):
    for name in [
        "ZONE_OBSERVATIONS_PATH", "MARKET_BARS_1H_PATH", "MARKET_BARS_5M_PATH",
        "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH",
    ]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    df = _df_1h()
    zone = {"zone_id": "ZID_X", "top": 105.0, "btm": 95.0, "poi": 100.0, "origin_ts_ms": int(df["timestamp"].iloc[0].timestamp() * 1000), "age_bars": 10}
    bar = {"timestamp": "2026-01-02T00:00:00+00:00", "open": 101.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 2000.0}
    signal = {
        "event_id": "ZONE_EVT", "symbol": "TEST-USDT", "type": "LONG", "trigger_bar_time": bar["timestamp"],
        "entry": 104.0, "sl": 93.6, "tp1": 107.12, "tp2": 110.24, "tp1_rr": 0.3, "tp2_rr": 0.6, "score": 70.0,
        "zone": zone, "zone_visit": {"visit_id": "VISIT_X", "touch_count_before_trigger": 0},
        "entry_bar": bar, "previous_bar": {}, "target": {"obstacle_price": 120.0}, "trigger": {"zone_trigger_mode": "zone"},
    }
    diagnostics = {
        "touch_events": [{"timestamp": bar["timestamp"], "zone_key": "DEMAND:ZID_X", "direction": "LONG", "midpoint": 100.0, "reason": "directional_candle_required", "visit_id": "VISIT_X", "bar": bar}],
        "rearm_events": [],
        "zones": {"DEMAND:ZID_X": {"direction": "LONG", "midpoint": 100.0, "closest_midpoint_bar": {**bar, "distance_pct": 0.5}}},
    }
    state = {"zones": {"DEMAND:ZID_X": {"visit_id": "VISIT_X"}}}
    result = research.record_scan_symbol(
        scan_id="SCAN_X", symbol="TEST-USDT", strategy_version="v1", code_commit_sha="abc",
        provider="binance", source="binance_spot", bars_1h=_df_1h().to_dict("records"), bars_5m=[bar],
        df_1h=df, demand=[zone], supply=[], diagnostics=diagnostics, symbol_state=state, signals=[signal],
    )
    assert result["observations"] >= 2
    rows = [json.loads(line) for line in (tmp_path / "zone_observations.jsonl").read_text().splitlines()]
    assert {r["event_type"] for r in rows} >= {"SIGNAL_CREATED", "TOUCH_REJECTED", "NEAREST_APPROACH"}
    assert all("features" in r for r in rows)


def test_cursor_updates_preserve_keys_for_multiple_symbols(tmp_path: Path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    for name in ["MARKET_BARS_1H_PATH", "MARKET_BARS_5M_PATH", "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH"]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    def write(symbol: str):
        return research.persist_market_bars(symbol, "1h", [{"timestamp": "2026-01-01T00:00:00+00:00", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}], provider="binance", source="binance_spot", scan_id=symbol)
    with ThreadPoolExecutor(max_workers=4) as ex:
        counts = list(ex.map(write, ["A-USDT", "B-USDT", "C-USDT", "D-USDT"]))
    assert counts == [1, 1, 1, 1]
    cursors = json.loads((tmp_path / "research_bar_cursors.json").read_text())
    assert len(cursors) == 4


def test_persist_market_bars_normalizes_epoch_milliseconds_and_rejects_open_bar(tmp_path: Path, monkeypatch):
    for name in ["MARKET_BARS_1H_PATH", "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH"]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    now = pd.Timestamp.now(tz="UTC")
    closed_ms = int((now - pd.Timedelta(hours=2)).timestamp() * 1000)
    open_ms = int((now - pd.Timedelta(minutes=10)).timestamp() * 1000)
    rows = [
        {"timestamp": closed_ms, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
        {"timestamp": open_ms, "open": 100, "high": 102, "low": 99, "close": 101, "volume": 10},
    ]
    n = research.persist_market_bars("TEST-USDT", "1h", rows, provider="binance", source="binance_spot", scan_id="S1")
    assert n == 1
    stored = json.loads((tmp_path / "market_bars_1h.jsonl").read_text().strip())
    assert stored["timestamp"].startswith("2026-")
    assert stored["close_time"] > stored["timestamp"]


def test_shadow_volume_ratio_uses_prior_5m_bars_only():
    ts = pd.date_range("2026-01-01 00:00", periods=21, freq="5min", tz="UTC")
    vols = [100.0] * 20 + [200.0]
    df5 = pd.DataFrame({"timestamp": ts, "volume": vols, "open": [1]*21, "high": [2]*21, "low": [0.5]*21, "close": [1.5]*21})
    zone = {"zone_id": "Z", "top": 2.0, "btm": 1.0, "poi": 1.5, "origin_ts_ms": int(ts[0].timestamp()*1000)}
    bar = {"timestamp": ts[-1].isoformat(), "open": 1.5, "high": 1.6, "low": 1.4, "close": 1.55, "volume": 200.0}
    f = research.build_research_features(symbol="TEST-USDT", direction="LONG", zone=zone, bar=bar, df_5m=df5)
    assert f["volume_ratio_5m20"] == pytest.approx(2.0)


def test_1h_features_are_anchored_to_decision_time():
    ts = pd.date_range("2026-01-01 00:00", periods=220, freq="1h", tz="UTC")
    close = [100.0 + i for i in range(220)]
    df = pd.DataFrame({"timestamp": ts, "open": close, "high": [x+1 for x in close], "low": [x-1 for x in close], "close": close, "volume": [1000.0]*220})
    zone = {"zone_id": "Z", "top": 250.0, "btm": 240.0, "poi": 245.0, "origin_ts_ms": int(ts[0].timestamp()*1000)}
    bar = {"timestamp": ts[-1].isoformat(), "decision_ts": ts[210].isoformat(), "open": 200, "high": 201, "low": 199, "close": 200, "volume": 100}
    before = research.build_research_features(symbol="TEST-USDT", direction="LONG", zone=zone, bar=bar, df_1h=df)
    df2 = df.copy(); df2.loc[211:, "close"] = 100000.0
    after = research.build_research_features(symbol="TEST-USDT", direction="LONG", zone=zone, bar=bar, df_1h=df2)
    assert before["return_1h_pct"] == after["return_1h_pct"]
    assert before["ema200_distance_pct"] == after["ema200_distance_pct"]


def test_forward_uses_closed_bars_for_mfe_mae():
    from research_forward import calculate_forward_outcome
    obs_ts = pd.Timestamp("2026-01-01T00:05:01Z")
    rows = []
    for i in range(4):
        ts = pd.Timestamp("2026-01-01T00:05:00Z") + pd.Timedelta(minutes=5*i)
        rows.append({"timestamp": ts, "close_time": ts + pd.Timedelta(minutes=5), "open": 100, "high": 101 if i == 0 else 105 if i == 2 else 100, "low": 99, "close": 100, "volume": 1})
    bars = pd.DataFrame(rows)
    obs = {"observation_id":"O1","event_id":"E1","scan_id":"S1","event_type":"TOUCH_REJECTED","symbol":"TEST-USDT","direction":"LONG","observation_ts":obs_ts.isoformat(),"reference_price":100}
    out = calculate_forward_outcome(obs, bars)
    assert out is not None
    assert out["forward_mfe_15m_pct"] == pytest.approx(5.0)


def test_update_outcomes_is_idempotent_against_existing_outcome_and_duplicate_observation(tmp_path: Path, monkeypatch):
    import research_forward
    monkeypatch.setattr(research_forward.research, "ZONE_OBSERVATIONS_PATH", tmp_path / "zone_observations.jsonl")
    monkeypatch.setattr(research_forward.research, "MARKET_BARS_5M_PATH", tmp_path / "market_bars_5m.jsonl")
    monkeypatch.setattr(research_forward.research, "RESEARCH_OUTCOMES_PATH", tmp_path / "research_outcomes.jsonl")
    monkeypatch.setattr(research_forward.research, "RESEARCH_OUTCOME_STATE_PATH", tmp_path / "research_outcome_state.json")
    monkeypatch.setattr(research_forward.research, "RESEARCH_MANIFEST_PATH", tmp_path / "research_manifest.json")
    obs = {
        "observation_id": "OBS_IDEMPOTENT", "event_id": "E1", "scan_id": "S1",
        "event_type": "SIGNAL_CREATED", "symbol": "TEST-USDT", "provider": "binance",
        "direction": "LONG", "observation_ts": "2026-01-01T00:00:00Z", "reference_price": 100.0,
    }
    (tmp_path / "zone_observations.jsonl").write_text(json.dumps(obs) + "\n" + json.dumps(obs) + "\n", encoding="utf-8")
    bars = []
    base = pd.Timestamp("2026-01-01T00:05:00Z")
    for i in range(289):
        ts = base + pd.Timedelta(minutes=5 * i)
        bars.append({"symbol": "TEST-USDT", "provider": "binance", "timestamp": ts.isoformat(), "close_time": (ts + pd.Timedelta(minutes=5)).isoformat(), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0})
    (tmp_path / "market_bars_5m.jsonl").write_text("\n".join(json.dumps(x) for x in bars) + "\n", encoding="utf-8")
    n1, total1 = research_forward.update_outcomes(write=True)
    n2, total2 = research_forward.update_outcomes(write=True)
    assert n1 == 1 and total1 == 1
    assert n2 == 0 and total2 == 1
    assert len((tmp_path / "research_outcomes.jsonl").read_text().splitlines()) == 1


def test_forward_excludes_partial_bar_opened_before_observation():
    obs_ts = pd.Timestamp("2026-01-01T00:04:00Z")
    rows = [
        {"timestamp": pd.Timestamp("2026-01-01T00:00:00Z"), "close_time": pd.Timestamp("2026-01-01T00:05:00Z"), "open": 100, "high": 999, "low": 1, "close": 100, "volume": 1},
        {"timestamp": pd.Timestamp("2026-01-01T00:05:00Z"), "close_time": pd.Timestamp("2026-01-01T00:10:00Z"), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
    ]
    bars = pd.DataFrame(rows)
    obs = {"observation_id":"O_LEAK", "symbol":"TEST-USDT", "direction":"LONG", "event_type":"SIGNAL_CREATED", "observation_ts":obs_ts.isoformat(), "reference_price":100.0}
    out = calculate_forward_outcome(obs, bars)
    assert out is not None
    assert out["forward_mfe_15m_pct"] is None  # incomplete 15m horizon, but 999 must never be used
    assert out["bars_available_after_observation"] == 1


def test_observation_id_canonicalizes_epoch_and_iso_timestamp():
    iso = research.observation_id("SIGNAL_CREATED", "TEST-USDT", "LONG", "Z", "2026-01-01T00:05:00Z", visit_id="V")
    epoch = research.observation_id("SIGNAL_CREATED", "TEST-USDT", "LONG", "Z", int(pd.Timestamp("2026-01-01T00:05:00Z").timestamp() * 1000), visit_id="V")
    assert iso == epoch


def test_persist_market_bars_normalizes_epoch_ms_5m(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(research, "MARKET_BARS_5M_PATH", tmp_path / "market_bars_5m.jsonl")
    monkeypatch.setattr(research, "RESEARCH_BAR_CURSORS_PATH", tmp_path / "research_bar_cursors.json")
    monkeypatch.setattr(research, "RESEARCH_MANIFEST_PATH", tmp_path / "research_manifest.json")
    monkeypatch.setattr(research, "RESEARCH_ERRORS_PATH", tmp_path / "research_persistence_errors.jsonl")
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    raw5 = []
    for i in range(25):
        ts = base + pd.Timedelta(minutes=5*i)
        raw5.append({"timestamp": int(ts.timestamp()*1000), "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10, "close_time": int((ts + pd.Timedelta(minutes=5)).timestamp()*1000)})
    result = research.persist_market_bars(raw5[0]["symbol"] if "symbol" in raw5[0] else "TEST-USDT", "5m", raw5, provider="binance", source="binance_spot", scan_id="S_EPOCH", code_commit_sha="abc")
    assert result == 24
    rows = [json.loads(x) for x in (tmp_path / "market_bars_5m.jsonl").read_text().splitlines()]
    assert rows
    assert rows[0]["timestamp"].startswith("2026-")
    assert all(not str(r["timestamp"]).startswith("1970-") for r in rows)


def test_forward_loader_separates_multiple_providers(tmp_path: Path, monkeypatch):
    import research_forward
    monkeypatch.setattr(research_forward.research, "MARKET_BARS_5M_PATH", tmp_path / "market_bars_5m.jsonl")
    ts0 = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for provider, delta in (("binance", 1.0), ("bingx", -1.0)):
        for i in range(3):
            ts = ts0 + pd.Timedelta(minutes=5 * i)
            close = 100.0 + delta * i
            rows.append({"symbol": "TEST-USDT", "provider": provider, "timestamp": ts.isoformat(), "close_time": (ts + pd.Timedelta(minutes=5)).isoformat(), "open": 100.0, "high": max(100.0, close), "low": min(100.0, close), "close": close, "volume": 1.0})
    (tmp_path / "market_bars_5m.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    loaded = research_forward._load_bars()
    assert "TEST-USDT|binance" in loaded
    assert "TEST-USDT|bingx" in loaded
    assert loaded["TEST-USDT|binance"]["close"].iloc[-1] == pytest.approx(102.0)
    assert loaded["TEST-USDT|bingx"]["close"].iloc[-1] == pytest.approx(98.0)


def test_forward_outcome_marks_incomplete_5m_path_as_censored():
    obs_ts = pd.Timestamp("2026-01-01T00:00:01Z")
    rows = []
    # Deliberately omit the 00:10 close, creating a 10-minute gap.
    for ts in [pd.Timestamp("2026-01-01T00:05:00Z"), pd.Timestamp("2026-01-01T00:15:00Z"), pd.Timestamp("2026-01-01T00:20:00Z")]:
        rows.append({"timestamp": ts, "close_time": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0})
    bars = pd.DataFrame(rows)
    obs = {"observation_id": "O_GAP", "symbol": "TEST-USDT", "direction": "LONG", "event_type": "SIGNAL_CREATED", "observation_ts": obs_ts.isoformat(), "reference_price": 100.0}
    out = calculate_forward_outcome(obs, bars)
    assert out is not None
    assert out["forward_gap_count_15m"] == 1
    assert out["censored_path_15m"] is True
    assert out["forward_mfe_15m_pct"] is None
    assert out["forward_mae_15m_pct"] is None
    assert out["forward_path_complete_24h"] is False


def test_touch_event_structural_distance_is_converted_to_R(monkeypatch):
    zone = {"zone_id": "ZID_STRUCT", "top": 105.0, "btm": 95.0, "poi": 100.0, "origin_ts_ms": 1767225600000}
    event = {
        "timestamp": "2026-01-02T00:00:00Z", "reason": "directional_candle_required",
        "entry_ref": 100.0, "structural_distance": 15.0,
        "bar": {"timestamp": "2026-01-02T00:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
        "previous_bar": {},
    }
    out = research.build_observation_from_touch_event(
        event=event, symbol="TEST-USDT", zone=zone, direction="LONG", df_1h=_df_1h(),
        zone_visit_id="V1", scan_id="S1", decision_ts="2026-01-02T00:05:00Z",
        strategy_version="v1", code_commit_sha="abc", provider="binance", source="binance_spot",
    )
    assert out["features"]["shadow_structure_ge_1_5R"] is True
    assert out["features"]["structure_room_R"] == pytest.approx(1.5)


def test_forward_return_5m_uses_next_completed_close_for_non_aligned_observation():
    obs_ts = pd.Timestamp("2026-01-01T00:05:01Z")
    rows = []
    for ts, close in [
        (pd.Timestamp("2026-01-01T00:05:00Z"), 99.0),
        (pd.Timestamp("2026-01-01T00:10:00Z"), 97.0),
    ]:
        rows.append({"timestamp": ts, "close_time": ts + pd.Timedelta(minutes=5), "open": 99.0, "high": 100.0, "low": 96.0, "close": close, "volume": 1.0})
    out = calculate_forward_outcome(
        {"observation_id": "O_RET", "symbol": "TEST-USDT", "direction": "LONG", "event_type": "SIGNAL_CREATED", "observation_ts": obs_ts.isoformat(), "reference_price": 99.0},
        pd.DataFrame(rows),
    )
    assert out is not None
    assert out["forward_return_5m_pct"] == pytest.approx((99.0 / 99.0 - 1.0) * 100.0)
    assert out["forward_return_5m_sample_ts"].startswith("2026-01-01T00:10:00")


def test_threshold_order_same_bar_is_ambiguous():
    obs_ts = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(5):
        ts = obs_ts + pd.Timedelta(minutes=5 * (i + 1))
        rows.append({"timestamp": ts, "close_time": ts + pd.Timedelta(minutes=5), "open": 100.0, "high": 105.5 if i == 0 else 100.0, "low": 89.5 if i == 0 else 100.0, "close": 100.0, "volume": 1.0})
    out = calculate_forward_outcome(
        {"observation_id": "O_SAME", "symbol": "TEST-USDT", "direction": "LONG", "event_type": "SIGNAL_CREATED", "observation_ts": obs_ts.isoformat(), "reference_price": 100.0},
        pd.DataFrame(rows),
    )
    assert out is not None
    # +5% and -10% happen within the same candle; 5m bars cannot order them.
    assert out["plus_5_before_minus_10"] is None
    assert out["minus_10_before_plus_5"] is None
    assert out["threshold_order_ambiguous_same_bar"] is True


def test_analytics_writes_nonfinite_as_json_null(tmp_path: Path, monkeypatch):
    from event_engine import analytics
    monkeypatch.setattr(analytics, "DATA_DIR", tmp_path)
    path = tmp_path / "test.jsonl"
    analytics._append_jsonl(path, {"nan": float("nan"), "inf": float("inf")})
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row == {"nan": None, "inf": None}


def test_market_context_parser_collects_microstructure_without_private_credentials(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    responses = {
        "premiumIndex": {"code": 0, "data": {"symbol": "TEST-USDT", "markPrice": "101", "indexPrice": "100", "lastFundingRate": "0.001", "nextFundingTime": 1767225600000, "time": 1767225500000}},
        "openInterest": {"code": 0, "data": {"symbol": "TEST-USDT", "openInterest": "12345", "time": 1767225500000}},
        "depth": {"code": 0, "data": {"bids": [["100", "5"], ["99.9", "4"]], "asks": [["101", "3"], ["101.1", "2"]], "T": 1767225500000}},
        "trades": {"code": 0, "data": [
            {"id": 1, "price": "100.5", "qty": "2", "quoteQty": "201", "time": 1767225499000, "buyerMaker": False},
            {"id": 2, "price": "100.4", "qty": "1", "quoteQty": "100.4", "time": 1767225498000, "buyerMaker": True},
        ]},
    }
    monkeypatch.setattr(bingx, "_research_public_get", lambda endpoint_key, path, params, timeout_sec=3.0: responses[endpoint_key])
    snap = bingx.fetch_research_market_context("TEST-USDT", depth_limit=20, trades_limit=100)
    assert snap["status"] == "ok"
    assert snap["funding_rate"] == pytest.approx(0.001)
    assert snap["open_interest"] == pytest.approx(12345.0)
    assert snap["order_book"]["best_bid"] == pytest.approx(100.0)
    assert snap["order_book"]["best_ask"] == pytest.approx(101.0)
    assert snap["order_book"]["book_imbalance_5"] is not None
    assert snap["recent_trades"]["aggressor_delta_quote"] == pytest.approx(100.6)
    assert snap["recent_trades"]["buy_aggressor_ratio"] == pytest.approx(201.0 / 301.4)


def test_market_context_and_extra_kline_flow_fields_are_persisted(tmp_path: Path, monkeypatch):
    names = [
        "ZONE_OBSERVATIONS_PATH", "MARKET_BARS_1H_PATH", "MARKET_BARS_5M_PATH",
        "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH", "MARKET_CONTEXT_PATH",
    ]
    for name in names:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    df = _df_1h()
    ts = pd.Timestamp("2026-01-02T00:00:00Z")
    bar = {
        "timestamp": ts, "open": 101.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 2000.0,
        "quote_volume": 210000.0, "taker_buy_base": 1200.0, "taker_buy_quote": 126000.0,
        "taker_flow_valid": True, "bar_delta_usdt": 42000.0, "trade_count": 123,
    }
    zone = {"zone_id": "ZID_CTX", "top": 105.0, "btm": 95.0, "poi": 100.0, "origin_ts_ms": int(df["timestamp"].iloc[0].timestamp()*1000), "age_bars": 10}
    signal = {"event_id": "E_CTX", "symbol": "TEST-USDT", "type": "LONG", "trigger_bar_time": ts.isoformat(), "entry": 104.0, "sl": 93.6, "tp1": 107.12, "tp2": 110.24, "tp1_rr": 0.3, "tp2_rr": 0.6, "score": 80.0, "zone": zone, "zone_visit": {"visit_id": "V_CTX", "touch_count_before_trigger": 0}, "entry_bar": {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k,v in bar.items() if k in {"timestamp","open","high","low","close","volume"}}, "previous_bar": {}, "target": {"obstacle_price": 120.0}, "trigger": {"zone_trigger_mode": "zone"}}
    result = research.record_scan_symbol(
        scan_id="SCAN_CTX", symbol="TEST-USDT", strategy_version="v1", code_commit_sha="abc", provider="binance", source="binance_spot",
        bars_1h=df.to_dict("records"), bars_5m=[bar], df_1h=df, demand=[zone], supply=[], diagnostics={"touch_events": [], "rearm_events": [], "zones": {}},
        symbol_state={}, signals=[signal], decision_ts="2026-01-02T00:10:00Z",
        market_context={"context_id": "MC_CTX", "funding_rate": 0.001, "open_interest": 12345.0, "order_book": {"best_bid":100.0,"best_ask":101.0,"spread_pct":1.0,"book_imbalance_5":0.2}, "recent_trades": {"valid_trade_count":2,"aggressor_delta_quote":100.0}},
    )
    assert result["market_context"] == 1
    context_rows = [json.loads(x) for x in (tmp_path / "market_context.jsonl").read_text().splitlines()]
    assert context_rows[0]["context_id"] == "MC_CTX"
    market_rows = [json.loads(x) for x in (tmp_path / "market_bars_5m.jsonl").read_text().splitlines()]
    assert market_rows[0]["bar_delta_usdt"] == pytest.approx(42000.0)
    assert market_rows[0]["taker_buy_quote"] == pytest.approx(126000.0)
    obs_rows = [json.loads(x) for x in (tmp_path / "zone_observations.jsonl").read_text().splitlines()]
    sig_row = next(x for x in obs_rows if x["event_type"] == "SIGNAL_CREATED")
    assert sig_row["market_context_id"] == "MC_CTX"
    assert sig_row["features"]["funding_rate"] == pytest.approx(0.001)
    assert sig_row["features"]["book_imbalance_5"] == pytest.approx(0.2)


def test_shadow_context_features_include_session_sweep_fvg_vwap_and_btc():
    ts = pd.date_range("2026-01-01 13:00", periods=30, freq="5min", tz="UTC")
    rows = []
    for i, t in enumerate(ts):
        base = 100.0
        high = base + 1.0
        low = base - 1.0
        close = 100.0
        if i == 28:
            high = 106.0; low = 99.0; close = 103.0
        if i == 29:
            high = 103.0; low = 101.5; close = 102.8
        rows.append({"timestamp": t, "close_time": t + pd.Timedelta(minutes=5), "open": 100.0, "high": high, "low": low, "close": close, "volume": 100.0 + i, "bar_delta_usdt": 10.0})
    x5 = pd.DataFrame(rows)
    btc = x5.assign(close=x5["close"] * 1.01)
    zone = {"zone_id":"Z", "top":104.0, "btm":96.0, "poi":100.0, "origin_ts_ms":int(ts[0].timestamp()*1000), "age_bars":29, "start":0}
    bar = rows[-1]
    features = research.build_research_features(symbol="TEST-USDT", direction="SHORT", zone=zone, bar=bar, df_5m=x5, df_1h=_df_1h(), decision_ts=(ts[-1] + pd.Timedelta(minutes=5)).isoformat(), market_context={}, btc_df_5m=btc, btc_df_1h=_df_1h())
    assert features["session_utc"] == "LONDON_NY_OVERLAP"
    assert features["session_vwap_approx"] is not None
    assert features["session_cvd_proxy_quote"] is not None
    assert features["5m_atr14"] is not None
    assert "liquidity_sweep_bearish_20" in features
    assert "fvg_bullish_5m" in features
    assert features["btc_return_1h_pct"] is not None or features["btc_return_5m_pct"] is not None


def test_recent_trade_boolean_parser_handles_string_flags(monkeypatch):
    from event_engine import bingx
    assert bingx._research_trade_metrics([
        {"price": "100", "qty": "1", "quoteQty": "100", "time": 1000, "buyerMaker": "false"},
        {"price": "100", "qty": "1", "quoteQty": "100", "time": 2000, "buyerMaker": "true"},
    ])[
        "aggressor_delta_quote"
    ] == pytest.approx(0.0)


def test_recent_trade_missing_buyer_maker_is_unknown_not_buy():
    from event_engine import bingx

    metrics = bingx._research_trade_metrics([
        {"price": "100", "qty": "1", "quoteQty": "100", "time": 1000},
        {"price": "101", "qty": "1", "quoteQty": "101", "time": 2000, "buyerMaker": "true"},
    ])
    assert metrics["aggressor_validity"] == "partial"
    assert metrics["buyer_maker_field_coverage"] == pytest.approx(0.5)
    assert metrics["buy_aggressor_ratio"] == pytest.approx(0.0)


def test_binance_kline_parser_preserves_taker_flow(monkeypatch):
    from event_engine import binance
    row = [
        1767225600000, "100", "102", "99", "101", "10",
        1767225659999, "1010", 25, "6", "606", "0",
    ]
    monkeypatch.setattr(binance, "_get", lambda *args, **kwargs: [row])
    rows = binance.fetch_klines("BTCUSDT", interval="5m", limit=1)
    assert rows[0]["taker_buy_base"] == pytest.approx(6.0)
    assert rows[0]["taker_buy_quote"] == pytest.approx(606.0)
    assert rows[0]["bar_delta_usdt"] == pytest.approx(202.0)
    assert rows[0]["taker_flow_valid"] is True


def test_approach_features_use_only_bars_before_trigger():
    ts = pd.date_range("2026-01-01 00:00", periods=40, freq="5min", tz="UTC")
    closes = [100 + i * 0.1 for i in range(40)]
    rows = []
    for i, t in enumerate(ts):
        rows.append({"timestamp": t, "close_time": t + pd.Timedelta(minutes=5), "open": closes[i], "high": closes[i] + 0.5, "low": closes[i] - 0.5, "close": closes[i], "volume": 100.0})
    df5 = pd.DataFrame(rows)
    zone = {"zone_id": "Z", "top": 105.0, "btm": 95.0, "poi": 100.0, "origin_ts_ms": int(ts[0].timestamp() * 1000), "age_bars": 39, "start": 0}
    trigger = rows[-1].copy()
    trigger["close"] = 103.9
    altered = df5.copy()
    altered.loc[altered.index[-1], "close"] = 9999.0
    f1 = research.build_research_features(symbol="TEST-USDT", direction="LONG", zone=zone, bar=trigger, df_5m=df5, decision_ts=(ts[-1] + pd.Timedelta(minutes=5)).isoformat())
    f2 = research.build_research_features(symbol="TEST-USDT", direction="LONG", zone=zone, bar=trigger, df_5m=altered, decision_ts=(ts[-1] + pd.Timedelta(minutes=5)).isoformat())
    assert f1["approach_return_15m_pct"] == pytest.approx(f2["approach_return_15m_pct"])
    assert f1["approach_directional_streak"] == f2["approach_directional_streak"]


def test_4h_features_use_complete_four_hour_buckets_only():
    ts = pd.date_range("2026-01-01", periods=14, freq="1h", tz="UTC")
    close = [100 + i for i in range(len(ts))]
    df = pd.DataFrame({"timestamp": ts, "open": close, "high": [x + 1 for x in close], "low": [x - 1 for x in close], "close": close, "volume": [1.0] * len(ts), "close_time": ts + pd.Timedelta(hours=1)})
    f = research.build_research_features(
        symbol="TEST-USDT", direction="LONG",
        zone={"zone_id":"Z", "top":110, "btm":90, "poi":100, "origin_ts_ms":int(ts[0].timestamp()*1000), "start":0},
        bar={"timestamp": ts[-1].isoformat(), "open": 113, "high":114, "low":112, "close":113, "volume":1},
        df_1h=df, decision_ts="2026-01-01T13:01:00Z",
    )
    # 12:00-13:00 is inside an incomplete 12:00-16:00 4H bucket and must not become a 4H candle.
    assert f["htf4h_return_4h_pct"] is not None


def test_market_context_tracks_latency_and_endpoint_status(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    responses = {
        "premiumIndex": {"code": 0, "data": {"symbol": "TEST-USDT", "markPrice": "101", "indexPrice": "100", "lastFundingRate": "0.001", "nextFundingTime": 1767225600000, "time": 1767225500000}},
        "openInterest": {"code": 0, "data": {"symbol": "TEST-USDT", "openInterest": "12345", "time": 1767225500000}},
        "depth": {"code": 0, "data": {"bids": [["100", "5"]], "asks": [["101", "3"]], "T": 1767225500000}},
        "trades": {"code": 0, "data": [{"price": "100.5", "qty": "2", "quoteQty": "201", "time": 1767225499000, "buyerMaker": "false"}]},
    }
    monkeypatch.setattr(bingx, "_research_public_get", lambda endpoint_key, path, params, timeout_sec=3.0: responses[endpoint_key])
    snap = bingx.fetch_research_market_context("TEST-USDT")
    assert set(snap["endpoint_status"]) == {"premiumIndex", "openInterest", "depth", "trades"}
    assert all(v >= 0 for v in snap["endpoint_latency_ms"].values())
    assert snap["collection_latency_ms"] >= 0
    assert snap["mark_index_basis_pct"] == pytest.approx(1.0)


def test_market_context_depth_adds_quote_imbalance_and_microprice(monkeypatch):
    from event_engine import bingx
    data = {"bids": [[100, 5], [99, 1]], "asks": [[101, 2], [102, 1]], "T": 123}
    out = bingx._research_depth_metrics(data, mid_price=100.5)
    assert out["book_quote_imbalance_5"] is not None
    assert out["microprice"] is not None
    assert out["bid_quote_5"] > 0 and out["ask_quote_5"] > 0


def test_build_features_has_extended_returns_and_approach_features():
    ts = pd.date_range("2026-01-01", periods=310, freq="5min", tz="UTC")
    close = [100.0 + i * 0.05 for i in range(len(ts))]
    df5 = pd.DataFrame({"timestamp": ts, "close_time": ts + pd.Timedelta(minutes=5), "open": close, "high": [x+1 for x in close], "low": [x-1 for x in close], "close": close, "volume": [100.0+i for i in range(len(ts))]})
    zone = {"zone_id":"Z", "top":115, "btm":95, "poi":105, "origin_ts_ms":int(ts[0].timestamp()*1000), "start":0}
    f = research.build_research_features(symbol="TEST-USDT", direction="LONG", zone=zone, bar={"timestamp":ts[-1].isoformat(), "open":115, "high":116, "low":114, "close":115, "volume":400}, df_5m=df5, decision_ts=(ts[-1]+pd.Timedelta(minutes=5)).isoformat())
    assert f["5m_return_3h_pct"] is not None
    assert f["5m_return_12h_pct"] is not None
    assert f["approach_return_15m_pct"] is not None
    assert f["approach_directional_streak"] > 0
    assert f["market_regime"] in {"TREND_UP", "TREND_DOWN", "RANGE_NEAR_EMAS", "TRANSITION", None}


def test_market_context_persistence_failure_is_explicit(tmp_path: Path, monkeypatch):
    names = ["ZONE_OBSERVATIONS_PATH", "MARKET_BARS_1H_PATH", "MARKET_BARS_5M_PATH", "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH", "MARKET_CONTEXT_PATH"]
    for name in names:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    monkeypatch.setattr(research, "record_market_context", lambda row: False)
    df = _df_1h()
    zone = {"zone_id":"Z", "top":105.0, "btm":95.0, "poi":100.0, "origin_ts_ms":int(df["timestamp"].iloc[0].timestamp()*1000), "age_bars":1}
    out = research.record_scan_symbol(
        scan_id="S_FAILCTX", symbol="TEST-USDT", strategy_version="v1", code_commit_sha="abc", provider="binance", source="binance_spot",
        bars_1h=df.to_dict("records"), bars_5m=[], df_1h=df, demand=[zone], supply=[], diagnostics={"touch_events": [], "rearm_events": [], "zones": {}},
        symbol_state={}, signals=[], decision_ts="2026-01-02T00:00:00Z", market_context={"context_id":"MC_FAIL"},
    )
    assert out["market_context"] == 0


def test_fetch_research_account_snapshot_collects_read_only_fields_without_logging_secrets(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "get_credentials", lambda: ("TEST_PUBLIC_KEY", "TEST_SECRET_SENTINEL"))
    calls = []

    def fake_request(method, path, params=None, signed=True, **kwargs):
        calls.append((method, path, params, signed))
        if path == bingx.BALANCE_PATH:
            return {
                "code": 0,
                "data": [{
                    "asset": "USDT",
                    "balance": "123.4",
                    "equity": "120.1",
                    "unrealizedProfit": "-1.2",
                    "realisedProfit": "4.5",
                    "availableMargin": "110.0",
                    "usedMargin": "10.1",
                    "freezedMargin": "0.5",
                }],
            }
        if path == bingx.ALL_FILL_ORDERS_PATH:
            return {"code": 0, "data": [{"tradeId": "T1", "orderId": "O1", "symbol": "BTC-USDT", "side": "SELL", "positionSide": "SHORT", "price": "100", "qty": "0.1", "realizedPnl": "1.5", "fee": "-0.02", "time": 1767225500000}]}
        if path == bingx.INCOME_PATH:
            return {"code": 0, "data": [
                {"symbol": "BTC-USDT", "incomeType": "FUNDING_FEE", "income": "-0.03", "asset": "USDT", "info": "Funding Fee", "time": 1767225501000, "tranId": "I1", "tradeId": "T1"},
                {"symbol": "BTC-USDT", "incomeType": "TRADING_FEE", "income": "-0.02", "asset": "USDT", "info": "Trading Fee", "time": 1767225502000, "tranId": "I2", "tradeId": "T1"},
            ]}
        if path == bingx.FORCE_ORDERS_PATH:
            return {"code": 0, "data": [{"symbol": "BTC-USDT", "side": "BUY", "positionSide": "SHORT", "autoCloseType": "LIQUIDATION", "orderId": "O_FORCE", "time": 1767225505000, "price": "95", "origQty": "0.1", "avgPrice": "95"}]}
        if path == bingx.POSITION_PATH:
            return {"code": 0, "data": []}
        if path == bingx.COMMISSION_RATE_PATH:
            return {"code": 0, "data": {"commission": {"takerCommissionRate": "0.0005", "makerCommissionRate": "0.0002"}}}
        raise AssertionError(path)

    monkeypatch.setattr(bingx, "_request", fake_request)
    snap = bingx.fetch_research_account_snapshot()
    assert snap["status"] == "ok"
    assert snap["asset"] == "USDT"
    assert snap["equity"] == pytest.approx(120.1)
    assert snap["available_margin"] == pytest.approx(110.0)
    assert snap["open_positions_count"] == 0
    assert snap["recent_fill_count"] == 1
    assert snap["recent_fill_fee_total"] == pytest.approx(-0.02)
    assert snap["recent_fill_realized_pnl_total"] == pytest.approx(1.5)
    assert snap["recent_income_count"] == 2
    assert snap["recent_income_totals"]["FUNDING_FEE"] == pytest.approx(-0.03)
    assert snap["recent_income_totals"]["TRADING_FEE"] == pytest.approx(-0.02)
    assert snap["recent_liquidation_count"] == 1
    assert snap["recent_adl_count"] == 0
    assert snap["taker_commission_rate"] == pytest.approx(0.0005)
    assert snap["maker_commission_rate"] == pytest.approx(0.0002)
    assert all(call[3] is True for call in calls)
    rendered = json.dumps(snap, ensure_ascii=False)
    assert "TEST_SECRET_SENTINEL" not in rendered


def test_record_scan_symbol_persists_account_context_once_and_links_observation(tmp_path: Path, monkeypatch):
    for name in [
        "ZONE_OBSERVATIONS_PATH", "MARKET_BARS_1H_PATH", "MARKET_BARS_5M_PATH",
        "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH",
        "ACCOUNT_CONTEXT_PATH",
    ]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    df = _df_1h()
    ts = pd.Timestamp("2026-01-03T00:05:00Z")
    bar = {"timestamp": ts, "open": 101.0, "high": 105.0, "low": 99.0, "close": 104.0, "volume": 2000.0}
    zone = {"zone_id": "Z_AC", "top": 105.0, "btm": 95.0, "poi": 100.0, "origin_ts_ms": int(df["timestamp"].iloc[0].timestamp() * 1000), "age_bars": 10}
    signal = {
        "event_id": "E_AC", "symbol": "TEST-USDT", "type": "LONG", "trigger_bar_time": ts.isoformat(),
        "entry": 104.0, "sl": 93.6, "tp1": 107.12, "tp2": 110.24, "tp1_rr": 0.3, "tp2_rr": 0.6, "score": 80.0,
        "zone": zone, "zone_visit": {"visit_id": "V_AC", "touch_count_before_trigger": 0}, "entry_bar": bar,
        "previous_bar": {}, "target": {"obstacle_price": 120.0}, "trigger": {"zone_trigger_mode": "zone"},
    }
    account = {"account_context_id": "AC_TEST", "captured_at_ms": int(ts.timestamp() * 1000), "equity": 120.0, "available_margin": 100.0}
    diagnostics = {"touch_events": [], "rearm_events": [], "zones": {}}

    out = research.record_scan_symbol(
        scan_id="SCAN_AC", symbol="TEST-USDT", strategy_version="v1", code_commit_sha="abc",
        provider="binance", source="binance_spot", bars_1h=df.to_dict("records"), bars_5m=[bar], df_1h=df,
        demand=[zone], supply=[], diagnostics=diagnostics, symbol_state={}, signals=[signal],
        decision_ts="2026-01-03T00:10:00Z", account_context=account,
    )
    assert out["observations"] == 1
    account_rows = [json.loads(x) for x in (tmp_path / "account_context.jsonl").read_text().splitlines()]
    assert len(account_rows) == 1
    assert account_rows[0]["account_context_id"] == "AC_TEST"
    assert account_rows[0]["scan_id"] == "SCAN_AC"
    obs_rows = [json.loads(x) for x in (tmp_path / "zone_observations.jsonl").read_text().splitlines()]
    assert obs_rows[0]["account_context_id"] == "AC_TEST"
    assert obs_rows[0]["account_context_persisted"] is True

    # A scan symbol call receiving an already-persisted context must not append it again.
    account_persisted = dict(account, persisted=True)
    research.record_scan_symbol(
        scan_id="SCAN_AC", symbol="TEST2-USDT", strategy_version="v1", code_commit_sha="abc",
        provider="binance", source="binance_spot", bars_1h=df.to_dict("records"), bars_5m=[], df_1h=df,
        demand=[], supply=[], diagnostics=diagnostics, symbol_state={}, signals=[],
        decision_ts="2026-01-03T00:10:00Z", account_context=account_persisted,
    )
    account_rows2 = (tmp_path / "account_context.jsonl").read_text().splitlines()
    assert len(account_rows2) == 1


def test_record_scan_symbol_marks_account_context_persistence_failure(tmp_path: Path, monkeypatch):
    for name in [
        "ZONE_OBSERVATIONS_PATH", "MARKET_BARS_1H_PATH", "MARKET_BARS_5M_PATH",
        "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH",
        "ACCOUNT_CONTEXT_PATH",
    ]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    monkeypatch.setattr(research, "record_account_context", lambda row: False)
    df = _df_1h()
    zone = {"zone_id": "Z_FAILAC", "top": 105.0, "btm": 95.0, "poi": 100.0, "origin_ts_ms": int(df["timestamp"].iloc[0].timestamp() * 1000), "age_bars": 1}
    signal = {
        "event_id": "E_FAILAC", "symbol": "TEST-USDT", "type": "LONG", "trigger_bar_time": "2026-01-03T00:00:00Z",
        "entry": 104.0, "sl": 93.6, "tp1": 107.12, "tp2": 110.24, "zone": zone,
        "zone_visit": {"visit_id": "V_FAILAC", "touch_count_before_trigger": 0},
        "entry_bar": {"timestamp": "2026-01-03T00:00:00Z", "open": 101, "high": 105, "low": 99, "close": 104, "volume": 2000},
    }
    out = research.record_scan_symbol(
        scan_id="SCAN_FAILAC", symbol="TEST-USDT", strategy_version="v1", code_commit_sha="abc",
        provider="binance", source="binance_spot", bars_1h=df.to_dict("records"), bars_5m=[], df_1h=df,
        demand=[zone], supply=[], diagnostics={"touch_events": [], "rearm_events": [], "zones": {}},
        symbol_state={}, signals=[signal], decision_ts="2026-01-03T00:00:00Z",
        account_context={"account_context_id": "AC_FAIL", "captured_at_ms": 1, "equity": 100.0},
    )
    assert out["observations"] == 1
    rows = [json.loads(x) for x in (tmp_path / "zone_observations.jsonl").read_text().splitlines()]
    assert rows[0]["account_context_id"] == "AC_FAIL"
    assert rows[0]["account_context_persisted"] is False


def test_observation_from_signal_accepts_account_context_and_propagates_id():
    df = _df_1h()
    ts = "2026-01-02T00:00:00Z"
    zone = {"zone_id": "Z_OBS_AC", "top": 105.0, "btm": 95.0, "poi": 100.0, "origin_ts_ms": int(df["timestamp"].iloc[0].timestamp() * 1000), "age_bars": 10}
    signal = {
        "event_id": "E_OBS_AC", "symbol": "TEST-USDT", "type": "LONG", "trigger_bar_time": ts,
        "entry": 104.0, "sl": 93.6, "tp1": 107.12, "tp2": 110.24,
        "zone": zone, "zone_visit": {"visit_id": "V_OBS_AC", "touch_count_before_trigger": 0},
        "entry_bar": {"timestamp": ts, "open": 101, "high": 105, "low": 99, "close": 104, "volume": 2000},
    }
    out = research._observation_from_signal(
        signal, scan_id="S_OBS_AC", strategy_version="v1", code_commit_sha="abc", df_1h=df,
        decision_ts="2026-01-02T00:01:00Z", account_context={"account_context_id": "AC_OBS"},
    )
    assert out["account_context_id"] == "AC_OBS"
    assert out["features"]["account_context_id"] == "AC_OBS"


def test_fetch_research_account_snapshot_preserves_open_position_risk_context(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "get_credentials", lambda: ("TEST_PUBLIC_KEY", "TEST_SECRET_SENTINEL"))

    def fake_request(method, path, params=None, signed=True, **kwargs):
        if path == bingx.BALANCE_PATH:
            return {"code": 0, "data": [{"asset": "USDT", "balance": "100", "equity": "99", "unrealizedProfit": "-1", "realizedProfit": "2", "availableMargin": "90", "usedMargin": "10", "freezedMargin": "0"}]}
        if path == bingx.ALL_FILL_ORDERS_PATH:
            return {"code": 0, "data": []}
        if path == bingx.FORCE_ORDERS_PATH:
            return {"code": 0, "data": []}
        if path == bingx.POSITION_PATH:
            return {"code": 0, "data": [{"symbol": "BTC-USDT", "positionSide": "SHORT", "positionAmt": "0.25", "avgPrice": "100", "markPrice": "98", "liquidationPrice": "150", "leverage": "10", "unrealizedProfit": "0.5", "initialMargin": "2.45"}, {"symbol": "ETH-USDT", "positionSide": "LONG", "positionAmt": "0", "avgPrice": "0", "markPrice": "1", "liquidationPrice": "0", "leverage": "10", "unrealizedProfit": "0", "initialMargin": "0"}]}
        if path == bingx.COMMISSION_RATE_PATH:
            return {"code": 0, "data": {"commission": {"takerCommissionRate": "0.0005", "makerCommissionRate": "0.0002"}}}
        raise AssertionError(path)

    monkeypatch.setattr(bingx, "_request", fake_request)
    snap = bingx.fetch_research_account_snapshot()
    assert snap["open_positions_count"] == 1
    assert snap["short_positions_count"] == 1
    assert snap["long_positions_count"] == 0
    assert snap["open_positions_notional_usdt"] == pytest.approx(24.5)
    pos = snap["positions"][0]
    assert pos["liquidation_price"] == pytest.approx(150.0)
    assert pos["notional_usdt"] == pytest.approx(24.5)


def test_no_research_boundary_does_not_persist_irrelevant_market_bars(tmp_path: Path, monkeypatch):
    for name in [
        "ZONE_OBSERVATIONS_PATH", "MARKET_BARS_1H_PATH", "MARKET_BARS_5M_PATH",
        "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH",
    ]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    df = _df_1h()
    ts = pd.Timestamp("2026-01-04T00:05:00Z")
    bar = {"timestamp": ts, "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 100.0}
    out = research.record_scan_symbol(
        scan_id="SCAN_NO_BOUNDARY", symbol="TEST-USDT", strategy_version="v1", code_commit_sha="abc",
        provider="binance", source="binance_spot", bars_1h=df.to_dict("records"), bars_5m=[bar], df_1h=df,
        demand=[], supply=[], diagnostics={"touch_events": [], "rearm_events": [], "zones": {}},
        symbol_state={}, signals=[], decision_ts=ts.isoformat(),
        market_context=None,
    )
    assert out["observations"] == 0
    assert out["bars_1h"] == 0
    assert out["bars_5m"] == 0


def test_recent_trade_metrics_reports_buyer_maker_field_coverage():
    from event_engine import bingx
    out = bingx._research_trade_metrics([
        {"price": "100", "qty": "1", "quoteQty": "100", "time": 1000, "buyerMaker": False},
        {"price": "101", "qty": "1", "quoteQty": "101", "time": 2000, "buyerMaker": True},
        {"price": "102", "qty": "1", "quoteQty": "102", "time": 3000},
    ])
    assert out["buyer_maker_field_present_count"] == 2
    assert out["buyer_maker_field_missing_count"] == 1
    assert out["buyer_maker_field_coverage"] == pytest.approx(2 / 3)


def test_research_context_provenance_preserves_analysis_and_context_sources(tmp_path: Path, monkeypatch):
    for name in ["ZONE_OBSERVATIONS_PATH", "MARKET_BARS_1H_PATH", "MARKET_BARS_5M_PATH", "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH", "MARKET_CONTEXT_PATH"]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    df = _df_1h()
    zone = {"zone_id": "Z_PROV", "top": 105.0, "btm": 95.0, "poi": 100.0, "origin_ts_ms": int(df["timestamp"].iloc[0].timestamp() * 1000), "start": 0}
    bar = {"timestamp": "2026-01-02T00:00:00Z", "open": 101.0, "high": 104.0, "low": 99.0, "close": 103.0, "volume": 100.0}
    signal = {
        "event_id": "E_PROV", "symbol": "TEST-USDT", "type": "LONG", "trigger_bar_time": bar["timestamp"], "entry": 103.0,
        "sl": 92.7, "tp1": 106.09, "tp2": 109.18, "tp1_rr": 0.3, "tp2_rr": 0.6, "score": 60.0,
        "zone": zone, "zone_visit": {"visit_id": "V_PROV", "touch_count_before_trigger": 0}, "entry_bar": bar,
        "previous_bar": {}, "target": {"obstacle_price": 120.0}, "trigger": {"zone_trigger_mode": "zone"},
    }
    out = research.record_scan_symbol(
        scan_id="S_PROV", symbol="TEST-USDT", strategy_version="v1", code_commit_sha="abc",
        provider="binance", source="binance_spot", bars_1h=df.to_dict("records"), bars_5m=[bar], df_1h=df,
        demand=[zone], supply=[], diagnostics={"touch_events": [], "rearm_events": [], "zones": {}}, symbol_state={},
        signals=[signal], decision_ts="2026-01-02T00:10:00Z",
        market_context={"context_id": "MC_PROV", "analysis_provider": "binance", "analysis_source": "binance_spot", "context_provider": "bingx", "context_source": "bingx_swap_public", "cross_venue_metric": "binance_bingx_last_price_deviation"},
    )
    assert out["market_context"] == 1
    context = json.loads((tmp_path / "market_context.jsonl").read_text().splitlines()[0])
    assert context["analysis_provider"] == "binance"
    assert context["analysis_source"] == "binance_spot"
    assert context["context_provider"] == "bingx"
    assert context["context_source"] == "bingx_swap_public"
    obs = next(json.loads(x) for x in (tmp_path / "zone_observations.jsonl").read_text().splitlines() if json.loads(x)["event_type"] == "SIGNAL_CREATED")
    assert obs["features"]["context_provider"] == "bingx"
    assert obs["features"]["context_source"] == "bingx_swap_public"


def test_elapsed_minutes_is_monotonic_and_utc_safe():
    import run_once
    assert run_once._elapsed_minutes("2026-01-01T00:00:00Z", "2026-01-01T00:05:30Z") == pytest.approx(5.5)
    assert run_once._elapsed_seconds("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z") == pytest.approx(1.0)


def test_shadow_context_exposes_buyer_maker_coverage():
    from event_engine import bingx
    ctx = {
        "recent_trades": {
            "valid_trade_count": 3,
            "buyer_maker_field_present_count": 2,
            "buyer_maker_field_missing_count": 1,
            "buyer_maker_field_coverage": 2 / 3,
        }
    }
    # Build from minimal context and verify the exact feature names used by research.
    df = _df_1h()
    features = research.build_research_features(
        symbol="TEST-USDT", direction="LONG",
        zone={"zone_id": "Z", "top": 105, "btm": 95, "poi": 100, "origin_ts_ms": int(df["timestamp"].iloc[0].timestamp() * 1000), "start": 0},
        bar={"timestamp": "2026-01-02T00:00:00Z", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10},
        df_1h=df, market_context=ctx, decision_ts="2026-01-02T00:10:00Z",
    )
    assert features["recent_buyer_maker_present_count"] == 2
    assert features["recent_buyer_maker_missing_count"] == 1
    assert features["recent_buyer_maker_coverage"] == pytest.approx(2 / 3)


def test_quote_snapshot_provenance_fields_are_optional_and_non_ambiguous():
    from event_engine import research
    df = _df_1h()
    zone = {"zone_id": "Z_Q", "top": 105, "btm": 95, "poi": 100, "origin_ts_ms": int(df["timestamp"].iloc[0].timestamp() * 1000), "start": 0}
    f = research.build_research_features(
        symbol="TEST-USDT", direction="SHORT", zone=zone,
        bar={"timestamp": "2026-01-02T00:00:00Z", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10},
        df_1h=df, decision_ts="2026-01-02T00:05:00Z",
        market_context={"quote_source":"ticker","quote_sources_attempted":["bookTicker","ticker"],"quote_fallback_reason":"bookTicker invalid"},
    )
    assert f["quote_source"] == "ticker"
    assert f["quote_sources_attempted"] == ["bookTicker", "ticker"]
    assert f["quote_fallback_reason"] == "bookTicker invalid"


def test_observation_journal_deduplicates_across_calls_and_counts_skips(tmp_path, monkeypatch):
    import event_engine.research as research
    monkeypatch.setattr(research, "ZONE_OBSERVATIONS_PATH", tmp_path / "zone_observations.jsonl")
    monkeypatch.setattr(research, "RESEARCH_MANIFEST_PATH", tmp_path / "research_manifest.json")
    monkeypatch.setattr(research, "RESEARCH_ERRORS_PATH", tmp_path / "research_persistence_errors.jsonl")
    monkeypatch.setattr(research, "_OBSERVATION_SEEN_IDS", None)
    row = {"observation_id": "OBS_DUP", "event_type": "SIGNAL_CREATED", "symbol": "TEST-USDT", "direction": "LONG", "zone_id": "Z"}
    assert research.record_zone_observations([row]) == 1
    assert research.record_zone_observations([row]) == 0
    lines = (tmp_path / "zone_observations.jsonl").read_text().splitlines()
    assert len(lines) == 1
    manifest = json.loads((tmp_path / "research_manifest.json").read_text())
    assert manifest["observation_duplicates_skipped"] == 1


def test_observation_generation_does_not_claim_execution_age(tmp_path, monkeypatch):
    import event_engine.research as research
    for name in ["ZONE_OBSERVATIONS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH"]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    monkeypatch.setattr(research, "_OBSERVATION_SEEN_IDS", None)
    df = _df_1h()
    ts = "2026-01-02T00:00:00Z"
    zone = {"zone_id":"Z_AGE", "top":105.0, "btm":95.0, "poi":100.0, "origin_ts_ms":int(df["timestamp"].iloc[0].timestamp()*1000), "start":0}
    signal = {"event_id":"E_AGE", "symbol":"TEST-USDT", "type":"LONG", "trigger_bar_time":ts, "time":ts,
              "entry":100.0, "sl":90.0, "tp1":103.0, "tp2":106.0, "zone":zone,
              "zone_visit":{"visit_id":"V_AGE", "touch_count_before_trigger":0},
              "entry_bar":{"timestamp":ts,"open":99.0,"high":101.0,"low":98.0,"close":100.0,"volume":100.0}}
    out = research._observation_from_signal(signal, scan_id="S", strategy_version="v1", code_commit_sha="abc", df_1h=df,
                                            decision_ts="2026-01-02T00:05:00Z")
    assert out["features"]["execution_age_minutes"] is None
    assert out["features"]["execution_age_semantics"] == "not_available_at_observation_generation"
    assert out["features"]["trigger_to_observation_minutes"] == pytest.approx(5.0)


def test_pending_forward_symbols_identifies_recent_observations(tmp_path, monkeypatch):
    import event_engine.research as research
    monkeypatch.setattr(research, "ZONE_OBSERVATIONS_PATH", tmp_path / "zone_observations.jsonl")
    recent = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=10)).isoformat()
    old = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=30)).isoformat()
    rows = [
        {"observation_id":"O1", "symbol":"AAA-USDT", "observation_ts":recent},
        {"observation_id":"O2", "symbol":"BBB-USDT", "observation_ts":old},
    ]
    (tmp_path / "zone_observations.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert research.pending_forward_symbols(horizon_hours=24) == {"AAA-USDT"}


def test_record_scan_symbol_persists_pending_forward_bars_without_new_observation(tmp_path, monkeypatch):
    import event_engine.research as research
    for name in ["ZONE_OBSERVATIONS_PATH", "MARKET_BARS_1H_PATH", "MARKET_BARS_5M_PATH", "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH"]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    monkeypatch.setattr(research, "_OBSERVATION_SEEN_IDS", None)
    ts = pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(minutes=5)
    bar5 = {"timestamp": ts.isoformat(), "open":100.0,"high":101.0,"low":99.0,"close":100.5,"volume":10.0}
    bar1 = {"timestamp": (ts.floor("h") - pd.Timedelta(hours=1)).isoformat(), "open":100.0,"high":101.0,"low":99.0,"close":100.5,"volume":100.0}
    out = research.record_scan_symbol(
        scan_id="S_PENDING", symbol="TEST-USDT", strategy_version="v1", code_commit_sha="abc",
        provider="binance", source="binance_spot", bars_1h=[bar1], bars_5m=[bar5],
        df_1h=pd.DataFrame([bar1]), demand=[], supply=[], diagnostics={"touch_events":[],"rearm_events":[],"zones":{}},
        symbol_state={}, signals=[], decision_ts=ts.isoformat(), persist_bars_for_forward=True,
    )
    assert out["observations"] == 0
    assert out["bars_5m"] == 1
    assert out["bars_1h"] == 1


def test_observation_id_remains_stable_across_scans_for_same_signal_event():
    import event_engine.research as research
    ts = "2026-01-02T00:05:00Z"
    a = research.observation_id("SIGNAL_CREATED", "TEST-USDT", "SHORT", "ZONE_1", ts, visit_id="VISIT_1")
    b = research.observation_id("SIGNAL_CREATED", "TEST-USDT", "SHORT", "ZONE_1", int(pd.Timestamp(ts).timestamp() * 1000), visit_id="VISIT_1")
    assert a == b


def test_research_record_scan_symbol_accepts_pending_forward_flag_without_observations(tmp_path, monkeypatch):
    import event_engine.research as research
    for name in ["ZONE_OBSERVATIONS_PATH", "MARKET_BARS_1H_PATH", "MARKET_BARS_5M_PATH", "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH"]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    monkeypatch.setattr(research, "_OBSERVATION_SEEN_IDS", None)
    ts = pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(minutes=5)
    bar5 = {"timestamp": ts.isoformat(), "open":100.0, "high":101.0, "low":99.0, "close":100.5, "volume":10.0}
    bar1 = {"timestamp": (ts.floor("h") - pd.Timedelta(hours=1)).isoformat(), "open":100.0, "high":101.0, "low":99.0, "close":100.5, "volume":100.0}
    out = research.record_scan_symbol(
        scan_id="S_PENDING_FLAG", symbol="TEST-USDT", strategy_version="v1", code_commit_sha="abc",
        provider="binance", source="binance_spot", bars_1h=[bar1], bars_5m=[bar5], df_1h=pd.DataFrame([bar1]),
        demand=[], supply=[], diagnostics={"touch_events":[], "rearm_events":[], "zones":{}}, symbol_state={}, signals=[],
        decision_ts=ts.isoformat(), persist_bars_for_forward=True,
    )
    assert out["observations"] == 0
    assert out["bars_5m"] == 1
    assert out["bars_1h"] == 1


def test_observation_dedup_loads_existing_journal_on_new_process_state(tmp_path, monkeypatch):
    import event_engine.research as research
    monkeypatch.setattr(research, "ZONE_OBSERVATIONS_PATH", tmp_path / "zone_observations.jsonl")
    monkeypatch.setattr(research, "RESEARCH_MANIFEST_PATH", tmp_path / "research_manifest.json")
    monkeypatch.setattr(research, "RESEARCH_ERRORS_PATH", tmp_path / "research_persistence_errors.jsonl")
    (tmp_path / "zone_observations.jsonl").write_text(json.dumps({"observation_id":"OBS_EXISTING", "event_type":"SIGNAL_CREATED"}) + "\n")
    monkeypatch.setattr(research, "_OBSERVATION_SEEN_IDS", None)
    assert research.record_zone_observations([{"observation_id":"OBS_EXISTING", "event_type":"SIGNAL_CREATED"}]) == 0
    assert len((tmp_path / "zone_observations.jsonl").read_text().splitlines()) == 1


def test_observation_dedup_is_thread_safe(tmp_path, monkeypatch):
    import concurrent.futures
    import event_engine.research as research
    monkeypatch.setattr(research, "ZONE_OBSERVATIONS_PATH", tmp_path / "zone_observations.jsonl")
    monkeypatch.setattr(research, "RESEARCH_MANIFEST_PATH", tmp_path / "research_manifest.json")
    monkeypatch.setattr(research, "RESEARCH_ERRORS_PATH", tmp_path / "research_persistence_errors.jsonl")
    monkeypatch.setattr(research, "_OBSERVATION_SEEN_IDS", None)
    row = {"observation_id":"OBS_THREAD", "event_type":"SIGNAL_CREATED", "symbol":"TEST-USDT", "direction":"LONG"}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _: research.record_zone_observations([row]), range(16)))
    assert sum(results) == 1
    assert len((tmp_path / "zone_observations.jsonl").read_text().splitlines()) == 1


def test_short_shadow_filter_candidates_are_observational_only():
    ts5 = pd.date_range("2026-01-04 00:00", periods=30, freq="5min", tz="UTC")
    df5 = pd.DataFrame({
        "timestamp": ts5,
        "close_time": ts5 + pd.Timedelta(minutes=5),
        "open": [100.0] * 30,
        "high": [100.01] * 30,
        "low": [98.0] * 30,
        "close": [98.1] * 30,
        "volume": [100.0] * 30,
    })
    ts1 = pd.date_range("2026-01-01", periods=60, freq="1h", tz="UTC")
    btc_close = [100.0] * 59 + [101.0]
    btc = pd.DataFrame({
        "timestamp": ts1,
        "close_time": ts1 + pd.Timedelta(hours=1),
        "open": btc_close,
        "high": [x + 0.1 for x in btc_close],
        "low": [x - 0.1 for x in btc_close],
        "close": btc_close,
        "volume": [1000.0] * 60,
    })
    zone = {
        "zone_id": "Z_SHORT",
        "top": 105.0,
        "btm": 95.0,
        "poi": 100.0,
        "origin_ts_ms": int(ts1[0].timestamp() * 1000),
        "age_bars": 20,
    }
    bar = {
        "timestamp": ts5[-1],
        "open": 100.0,
        "high": 100.01,
        "low": 98.0,
        "close": 98.1,
        "volume": 100.0,
    }
    features = research.build_research_features(
        symbol="TEST-USDT",
        direction="SHORT",
        zone=zone,
        bar=bar,
        df_5m=df5,
        df_1h=btc,
        decision_ts=(ts5[-1] + pd.Timedelta(minutes=5)).isoformat(),
        btc_df_1h=btc,
    )
    assert features["shadow_short_filter_experiment"] == "short_entry_filters_v1"
    assert features["body_to_range"] > research.SHADOW_SHORT_BODY_TO_RANGE_GT
    assert features["lower_wick_ratio"] < research.SHADOW_SHORT_LOWER_WICK_LT
    assert features["shadow_short_geometry_bad"] is True
    assert features["btc_ema50_distance_pct"] > research.SHADOW_SHORT_BTC_EMA50_GT_050
    assert features["shadow_short_btc_ema50_gt_025"] is True
    assert features["shadow_short_btc_ema50_gt_050"] is True
    assert features["shadow_short_combined_gt_025"] is True
    assert features["shadow_short_combined_gt_050"] is True
