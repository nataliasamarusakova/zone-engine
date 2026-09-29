from pathlib import Path

import pandas as pd

from event_engine import shadow


def _bars_5m(values, start="2026-09-28T08:20:00Z"):
    ts = pd.date_range(start=start, periods=len(values), freq="5min", tz="UTC")
    rows = []
    for i, vol in enumerate(values):
        close = 100.0 + i * 0.1
        rows.append({
            "timestamp": ts[i],
            "close_time": ts[i] + pd.Timedelta(minutes=5),
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": float(vol),
            "bar_delta_usdt": 10.0 if i % 2 == 0 else -10.0,
        })
    return pd.DataFrame(rows)


def _bars_1h(n=220, start="2026-09-19T00:00:00Z"):
    ts = pd.date_range(start=start, periods=n, freq="1h", tz="UTC")
    rows = []
    for i, t in enumerate(ts):
        close = 100.0 + i * 0.1
        rows.append({
            "timestamp": t,
            "close_time": t + pd.Timedelta(hours=1),
            "open": close - 0.05,
            "high": close + 0.10,
            "low": close - 0.10,
            "close": close,
            "volume": 1000.0,
        })
    return pd.DataFrame(rows)


def _signal(direction="LONG"):
    return {
        "event_id": "EVT_SHADOW",
        "symbol": "TEST-USDT",
        "type": direction,
        "entry": 100.0,
        "trigger_bar_time": "2026-09-28T10:00:00Z",
        "zone": {"top": 101.0, "btm": 99.0},
        "atr": 1.0,
    }


def test_shadow_is_observational_only():
    snap = shadow.build_entry_snapshot(_signal(), df_5m=_bars_5m([10] * 21 + [30]), df_1h=_bars_1h())
    assert snap["applied"] is False
    assert snap["experiments"]["CF_STRUCTURAL_SL"]["atr_multiplier"] == 0.5


def test_volume_5m_baseline_excludes_trigger_bar():
    snap = shadow.build_entry_snapshot(_signal(), df_5m=_bars_5m([10] * 20 + [30]), df_1h=_bars_1h())
    exp = snap["experiments"]["CF_VOLUME_5M"]
    assert exp["baseline_sma20_previous_bars"] == 10.0
    assert exp["trigger_volume"] == 30.0
    assert exp["ratio"] == 3.0
    assert exp["threshold_grid"]["1.5"] is True


def test_structural_stop_and_rr_are_computed_without_affecting_signal():
    snap = shadow.build_entry_snapshot(_signal("LONG"), df_5m=_bars_5m([10] * 21), df_1h=_bars_1h())
    sl = snap["experiments"]["CF_STRUCTURAL_SL"]
    rr = snap["experiments"]["CF_TP_RR_15_30"]
    assert round(sl["hypothetical_sl"], 6) == 98.925
    assert round(rr["hypothetical_tp1"], 6) == 101.6125
    assert round(rr["hypothetical_tp2"], 6) == 103.225


def test_delta_and_macro_alignment_are_directional():
    long_snap = shadow.build_entry_snapshot(_signal("LONG"), df_5m=_bars_5m([10] * 21), df_1h=_bars_1h())
    short_snap = shadow.build_entry_snapshot(_signal("SHORT"), df_5m=_bars_5m([10] * 21), df_1h=_bars_1h())
    assert long_snap["experiments"]["CF_DELTA_CONFIRM"]["would_pass"] is True
    assert short_snap["experiments"]["CF_DELTA_CONFIRM"]["would_pass"] is False


def test_atr_grid_records_multiple_thresholds():
    snap = shadow.build_entry_snapshot(_signal(), df_5m=_bars_5m([10] * 21), df_1h=_bars_1h())
    grid = snap["experiments"]["CF_ATR_MIN_5M"]["threshold_grid_pct"]
    assert set(grid) == {"0.1", "0.2", "0.3", "0.4", "0.5", "0.75", "1.0"}


def test_counterfactual_path_detects_target():
    snap = shadow.build_entry_snapshot(_signal("LONG"), df_5m=_bars_5m([10] * 21), df_1h=_bars_1h())
    obs = {
        "event_id": "EVT_SHADOW",
        "symbol": "TEST-USDT",
        "direction": "LONG",
        "reference_price": 100.0,
        "source_event_ts": "2026-09-28T10:00:00Z",
        "features": {"shadow_experiments": snap["experiments"]},
    }
    future = pd.DataFrame([
        {"timestamp": "2026-09-28T10:05:00Z", "close_time": "2026-09-28T10:10:00Z", "open": 100, "high": 102.5, "low": 99.5, "close": 102},
    ])
    out = shadow.calculate_counterfactual_outcomes(obs, future)
    assert out["experiments"]["CF_STRUCTURAL_SL"]["exit"] in {"TARGET", "AMBIGUOUS_SAME_BAR"}


def test_no_open_peers_has_explicit_correlation_status():
    snap = shadow.build_entry_snapshot(_signal(), df_5m=_bars_5m([10] * 21), df_1h=_bars_1h(), account_context={"positions": []})
    corr = snap["experiments"]["CF_CORRELATION"]
    assert corr["status"] == "NO_OPEN_PEERS"
    assert corr["would_pass"] is None


def test_session_shadow_uses_exact_00_to_06_utc_window():
    base = _bars_5m([10] * 21)
    for hour, expected in [(5, True), (6, False), (21, False)]:
        sig = _signal()
        sig["trigger_bar_time"] = f"2026-09-28T{hour:02d}:00:00Z"
        snap = shadow.build_entry_snapshot(sig, df_5m=base, df_1h=_bars_1h())
        exp = snap["experiments"]["CF_SESSION"]
        assert exp["asia_off_hours"] is expected
        assert exp["would_pass_if_asia_blocked"] is (not expected)
        assert exp["asia_block_window"] == "00:00<=UTC<06:00"


def test_management_only_isolated_from_structural_experiments():
    sig = _signal("LONG")
    sig.update({"sl": 90.0, "tp1": 103.0, "tp2": 106.0})
    snap = shadow.build_entry_snapshot(sig, df_5m=_bars_5m([10] * 21), df_1h=_bars_1h())
    management = snap["experiments"]["CF_DELAYED_BE"]
    assert management["standalone_management_only"] is True
    assert management["production_sl"] == 90.0
    assert management["production_tp1"] == 103.0
    assert management["production_tp2"] == 106.0
    assert management["version"] == 2


def test_management_only_counts_tp1_and_residual_half_weighted():
    sig = _signal("LONG")
    sig.update({"sl": 90.0, "tp1": 103.0, "tp2": 106.0})
    snap = shadow.build_entry_snapshot(sig, df_5m=_bars_5m([10] * 21), df_1h=_bars_1h())
    obs = {
        "event_id": "EVT_MGMT",
        "symbol": "TEST-USDT",
        "direction": "LONG",
        "reference_price": 100.0,
        "source_event_ts": "2026-09-28T10:00:00Z",
        "features": {"shadow_experiments": snap["experiments"]},
    }
    future = pd.DataFrame([
        {"timestamp": "2026-09-28T10:05:00Z", "close_time": "2026-09-28T10:10:00Z", "open": 100, "high": 103.2, "low": 99.5, "close": 103.0},
        {"timestamp": "2026-09-28T10:10:00Z", "close_time": "2026-09-28T10:15:00Z", "open": 103.0, "high": 103.1, "low": 101.0, "close": 102.0},
        {"timestamp": "2026-09-28T10:15:00Z", "close_time": "2026-09-28T10:20:00Z", "open": 102.0, "high": 102.1, "low": 99.5, "close": 100.0},
    ])
    out = shadow.calculate_counterfactual_outcomes(obs, future)
    cf = out["experiments"]["CF_DELAYED_BE_CURRENT_1BAR"]
    assert cf["standalone_management_only"] is True
    assert abs(cf["tp1_pnl_pct"] - 3.0) < 1e-12
    assert cf["residual_exit_pnl_pct"] == 0.0
    assert abs(cf["pnl_pct"] - 1.5) < 1e-12


def test_shadow_snapshot_contains_production_management_baseline():
    sig = _signal("SHORT")
    snap = shadow.build_entry_snapshot(sig, df_5m=_bars_5m([10] * 21), df_1h=_bars_1h())
    prod = snap["experiments"]["CF_PRODUCTION_MANAGEMENT"]
    assert prod["stop"] == 110.0
    assert prod["tp1"] == 97.0
    assert prod["tp2"] == 94.0
    assert prod["tp1_fraction"] == 0.5


def test_trailing_does_not_use_current_bar_high_for_same_bar_stop():
    sig = _signal("LONG")
    sig.update({"sl": 80.0, "tp1": 103.0, "tp2": 106.0})
    snap = shadow.build_entry_snapshot(sig, df_5m=_bars_5m([10] * 21), df_1h=_bars_1h())
    obs = {
        "event_id": "EVT_TRAIL",
        "symbol": "TEST-USDT",
        "direction": "LONG",
        "reference_price": 100.0,
        "source_event_ts": "2026-09-28T10:00:00Z",
        "features": {"shadow_experiments": snap["experiments"]},
    }
    hist = _bars_5m([10] * 30).tail(20).copy()
    future = pd.DataFrame([
        {"timestamp": "2026-09-28T10:05:00Z", "close_time": "2026-09-28T10:10:00Z", "open": 100, "high": 103.2, "low": 99.5, "close": 103.0},
        {"timestamp": "2026-09-28T10:10:00Z", "close_time": "2026-09-28T10:15:00Z", "open": 103, "high": 120.0, "low": 101.0, "close": 110.0},
        {"timestamp": "2026-09-28T10:15:00Z", "close_time": "2026-09-28T10:20:00Z", "open": 110, "high": 111.0, "low": 100.0, "close": 105.0},
    ])
    all_bars = pd.concat([hist, future], ignore_index=True)
    out = shadow.calculate_counterfactual_outcomes(obs, all_bars)
    cf = out["experiments"]["CF_TRAIL_CURRENT_1_0ATR"]
    # The 120 high on the current bar must not retroactively raise the stop before its low is checked.
    assert cf["exit_ts"] in {"2026-09-28T10:20:00+00:00", None}


def test_parse_ts_numeric_epoch_is_milliseconds():
    ts = shadow._parse_ts(1790648100000)
    assert ts is not None
    assert ts.isoformat() == "2026-09-29T02:15:00+00:00"


def test_missing_account_context_is_not_reported_as_no_open_peers():
    snap = shadow.build_entry_snapshot(
        _signal(), df_5m=_bars_5m([10] * 21), df_1h=_bars_1h(), account_context=None
    )
    corr = snap["experiments"]["CF_CORRELATION"]
    assert corr["status"] == "CONTEXT_UNAVAILABLE"
    assert corr["open_position_symbols"] == []
    assert corr["would_pass"] is None


def test_session_experiment_uses_decision_time_and_preserves_trigger_session():
    sig = _signal()
    sig["trigger_bar_time"] = "2026-09-28T05:55:00Z"
    snap = shadow.build_entry_snapshot(
        sig,
        df_5m=_bars_5m([10] * 21),
        df_1h=_bars_1h(),
        decision_ts="2026-09-28T06:03:52Z",
    )
    exp = snap["experiments"]["CF_SESSION"]
    assert exp["trigger_session_utc"] == "ASIA_OFF_HOURS"
    assert exp["decision_session_utc"] == "LONDON"
    assert exp["asia_off_hours"] is False


def test_counterfactual_excludes_bar_that_started_before_decision_boundary():
    snap = shadow.build_entry_snapshot(_signal("LONG"), df_5m=_bars_5m([10] * 21), df_1h=_bars_1h())
    obs = {
        "event_id": "EVT_BOUNDARY",
        "symbol": "TEST-USDT",
        "direction": "LONG",
        "reference_price": 100.0,
        "source_event_ts": "2026-09-28T10:00:00Z",
        "observation_ts": "2026-09-28T10:07:00Z",
        "counterfactual_path_start_ts": "2026-09-28T10:07:00Z",
        "features": {"shadow_experiments": snap["experiments"]},
    }
    future = pd.DataFrame([
        {"timestamp": "2026-09-28T10:05:00Z", "close_time": "2026-09-28T10:10:00Z", "open": 100, "high": 105, "low": 99, "close": 104},
        {"timestamp": "2026-09-28T10:10:00Z", "close_time": "2026-09-28T10:15:00Z", "open": 100.1, "high": 100.5, "low": 99.8, "close": 100.0},
    ])
    out = shadow.calculate_counterfactual_outcomes(obs, future)
    cf = out["experiments"]["CF_STRUCTURAL_SL"]
    assert cf["exit"] in {None, "PATH_NOT_EXITED"}
    assert cf.get("bars_to_exit") is None
