from __future__ import annotations

import pytest

def test_main_completes_empty_universe_without_private_or_market_api_calls(tmp_path, monkeypatch):
    import run_once

    data_dir = tmp_path / "data"
    scan_summaries = []
    monkeypatch.setattr(run_once, "DATA", data_dir)
    monkeypatch.setattr(run_once, "_write_runtime_version_metadata", lambda: None)
    monkeypatch.setattr(run_once, "_init_diagnostic_log", lambda: None)
    monkeypatch.setattr(run_once, "_private_layer_ready", lambda: False)
    monkeypatch.setattr(run_once.research, "RESEARCH_ENABLED", False)
    monkeypatch.setattr(run_once, "TREND_FILTER_ENABLED", False)
    monkeypatch.setattr(run_once, "get_scan_symbols", lambda: [])
    monkeypatch.setattr(run_once, "contracts", lambda: {})
    monkeypatch.setattr(run_once, "analysis_symbols_for_bingx", lambda _symbols: [])
    monkeypatch.setattr(run_once, "get_positions", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("private API must not be called")))
    monkeypatch.setattr(run_once, "fetch_binance_klines", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("market API must not be called")))
    monkeypatch.setattr(run_once, "fetch_bingx_klines", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("market API must not be called")))
    monkeypatch.setattr(run_once, "_load_successful_trade_ids", lambda: set())
    monkeypatch.setattr(run_once, "_load_terminal_event_ids", lambda: set())
    monkeypatch.setattr(run_once, "_load_failed_signal_ids", lambda: {})
    monkeypatch.setattr(run_once, "_load_zone_visit_state", lambda: {
        "version": run_once.ZONE_VISIT_STATE_VERSION,
        "trigger_mode": run_once.ZONE_TRIGGER_MODE,
        "symbols": {},
    })
    monkeypatch.setattr(run_once, "_save_zone_visit_state", lambda _state: None)
    monkeypatch.setattr(run_once, "save_scan", lambda *args, **kwargs: scan_summaries.append((args, kwargs)))
    monkeypatch.setattr(run_once.research, "update_manifest_run", lambda **_kwargs: None)
    monkeypatch.setattr(run_once.request_metrics, "log_summary", lambda _log: None)

    run_once.main()

    assert len(scan_summaries) == 1
    assert data_dir.exists()
    # No entry or private account methods can run in this public-only empty scan.


@pytest.mark.parametrize(
    ("trigger_age_minutes", "existing_positions", "expected_dispatch"),
    [
        (5, [{"symbol": "BBB-USDT", "positionSide": "BOTH", "positionAmt": "0.5", "markPrice": "20", "avgPrice": "19", "unrealizedProfit": "0.5"}], True),
        (60, [], False),
        (5, [{"symbol": "AAA-USDT", "positionSide": "BOTH", "positionAmt": "0.5"}], False),
    ],
)
def test_main_scans_one_symbol_and_applies_existing_safety_gates_before_dispatch(
    tmp_path, monkeypatch, trigger_age_minutes, existing_positions, expected_dispatch
):
    """Exercise main() from public market scan through the execution dispatch boundary."""
    import pandas as pd
    import run_once

    now = pd.Timestamp.now(tz="UTC")
    latest_closed = now - pd.Timedelta(minutes=5)
    timestamps = pd.date_range(
        end=latest_closed - pd.Timedelta(hours=1), periods=100, freq="1h", tz="UTC"
    )
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "close_time": timestamps + pd.Timedelta(hours=1),
        "open": [100.0] * len(timestamps),
        "high": [101.0] * len(timestamps),
        "low": [99.0] * len(timestamps),
        "close": [100.0] * len(timestamps),
        "volume": [1000.0] * len(timestamps),
    })
    raw_bars = frame.to_dict("records")
    five_min_timestamps = pd.date_range(
        end=latest_closed, periods=100, freq="5min", tz="UTC"
    )
    five_min_frame = pd.DataFrame({
        "timestamp": five_min_timestamps,
        "close_time": five_min_timestamps + pd.Timedelta(minutes=5),
        "open": [100.0] * len(five_min_timestamps),
        "high": [101.0] * len(five_min_timestamps),
        "low": [99.0] * len(five_min_timestamps),
        "close": [100.0] * len(five_min_timestamps),
        "volume": [100.0] * len(five_min_timestamps),
    })
    raw_five_min_bars = five_min_frame.to_dict("records")
    contract = {"symbol": "AAA-USDT", "lastPrice": "100.0"}
    signal = {
        "event_id": "EVT_MAIN_INTEGRATION_1",
        "symbol": "AAA-USDT",
        "type": "LONG",
        "idx": 99,
        "entry": 100.0,
        "sl": 90.0,
        "tp1": 103.0,
        "tp2": 106.0,
        "trigger_timeframe": "5m",
        "trigger_bar_time": (now - pd.Timedelta(minutes=trigger_age_minutes + 5)).isoformat(),
        "trigger_bar_close_time": (now - pd.Timedelta(minutes=trigger_age_minutes)).isoformat(),
        "decision_boundary_ts": (now - pd.Timedelta(minutes=trigger_age_minutes)).isoformat(),
        "zone": {"kind": "demand", "top": 101.0, "btm": 99.0},
        "trigger": {"type": "ZONE_MIDPOINT_TOUCH", "zone_entry_rule": "MIDPOINT"},
        "target": {"tp1_pct": 3.0, "tp2_pct": 6.0, "be_rule": "after_tp1_filled"},
        "risk_model": {"fixed_stop_pct": 10.0},
        "score": 90.0,
    }
    dispatched = []
    recorded_decisions = []
    saved_scans = []
    sent_signals = []
    appended_trade_rows = []
    appended_ledger_rows = []

    data_dir = tmp_path / "data"
    monkeypatch.setattr(run_once, "DATA", data_dir)
    monkeypatch.setattr(run_once, "_write_runtime_version_metadata", lambda: None)
    monkeypatch.setattr(run_once, "_init_diagnostic_log", lambda: None)
    monkeypatch.setattr(run_once, "_private_layer_ready", lambda: True)
    monkeypatch.setattr(run_once, "_load_active_trades_file", lambda **_kwargs: None)
    monkeypatch.setattr(run_once, "get_position_mode", lambda **_kwargs: "ONE_WAY")
    monkeypatch.setattr(run_once, "backfill_active_trade_provenance", lambda: 0)
    monkeypatch.setattr(run_once, "_update_active_trades_with_snapshot", lambda: None)
    monkeypatch.setattr(run_once, "reconcile_all_open_positions", lambda: True)
    monkeypatch.setattr(run_once, "get_positions", lambda **_kwargs: list(existing_positions))
    monkeypatch.setattr(run_once, "get_scan_symbols", lambda: ["AAA-USDT"])
    monkeypatch.setattr(run_once, "contracts", lambda: {"AAA-USDT": contract})
    monkeypatch.setattr(run_once, "get_contract", lambda _symbol: contract)
    monkeypatch.setattr(run_once, "classify_bingx_contract", lambda _contract: "CRYPTO")
    monkeypatch.setattr(
        run_once,
        "analysis_symbols_for_bingx",
        lambda _symbols: [{
            "bingx_symbol": "AAA-USDT", "binance_symbol": "AAAUSDT",
            "binance_available": True, "asset_class": "CRYPTO", "market_provider": "binance",
        }],
    )
    monkeypatch.setattr(
        run_once, "fetch_binance_klines",
        lambda _symbol, interval="1h", **_kwargs: raw_five_min_bars if interval == "5m" else raw_bars,
    )
    monkeypatch.setattr(run_once, "fetch_24h_ticker", lambda _symbol: {"lastPrice": "100.0"})
    monkeypatch.setattr(
        run_once,
        "generate_zone_signals",
        lambda *_args, **_kwargs: (frame.copy(), [], [{"top": 101.0, "btm": 99.0}], []),
    )
    monkeypatch.setattr(run_once, "_process_5m_zone_visits", lambda **_kwargs: (
        [dict(signal)],
        {"version": run_once.ZONE_VISIT_STATE_VERSION, "zones": {}},
        "LONG @ 100.0",
    ))
    monkeypatch.setattr(run_once, "score_zone_signal", lambda _signal: 90.0)
    monkeypatch.setattr(run_once, "add_3h_zone_confluence", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_once.shadow, "build_entry_snapshot", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run_once, "_load_successful_trade_ids", lambda: set())
    monkeypatch.setattr(run_once, "_load_terminal_event_ids", lambda: set())
    monkeypatch.setattr(run_once, "_load_failed_signal_ids", lambda: {})
    monkeypatch.setattr(run_once, "_load_zone_visit_state", lambda: {
        "version": run_once.ZONE_VISIT_STATE_VERSION,
        "trigger_mode": run_once.ZONE_TRIGGER_MODE,
        "symbols": {},
    })
    monkeypatch.setattr(run_once, "_save_zone_visit_state", lambda _state: None)
    monkeypatch.setattr(run_once, "_record_entry_decision", lambda *a, **kw: recorded_decisions.append((a, kw)))
    monkeypatch.setattr(run_once, "_claim_event_for_execution", lambda _event_id: (True, "claimed", "ATT_MAIN_INTEGRATION_1"))
    monkeypatch.setattr(run_once, "_finalize_event_claim", lambda *a, **kw: None)
    monkeypatch.setattr(run_once, "execute_new_position", lambda candidate: (
        dispatched.append(dict(candidate)) or {
            "status": "opened_protected",
            "order": {"order_id": "ENTRY-ORDER-1"},
            "position": {"symbol": "AAA-USDT", "avgPrice": "100.0"},
        }
    ))
    monkeypatch.setattr(run_once, "_append_jsonl", lambda _path, row: appended_trade_rows.append(dict(row)))
    monkeypatch.setattr(run_once, "_append_execution_ledger", lambda *a, **kw: appended_ledger_rows.append((a, kw)))
    monkeypatch.setattr(run_once, "_send_signal", lambda candidate, result: sent_signals.append((dict(candidate), dict(result))))
    monkeypatch.setattr(run_once, "save_scan", lambda *a, **kw: saved_scans.append((a, kw)))
    monkeypatch.setattr(run_once.research, "RESEARCH_ENABLED", False)
    monkeypatch.setattr(run_once.research, "update_manifest_run", lambda **_kwargs: None)
    monkeypatch.setattr(run_once, "TREND_FILTER_ENABLED", False)
    monkeypatch.setattr(run_once, "EXECUTION_ENABLED", True)
    monkeypatch.setattr(run_once, "MAX_5M_TRIGGER_AGE_MINUTES", 30.0)
    monkeypatch.setattr(run_once, "MAX_MARKET_SPREAD_PCT", 10.0)
    monkeypatch.setattr(run_once.request_metrics, "log_summary", lambda _log: None)

    run_once.main()

    assert len(saved_scans) == 1
    assert len(dispatched) == (1 if expected_dispatch else 0)
    if expected_dispatch:
        assert dispatched[0]["event_id"] == signal["event_id"]
        assert dispatched[0]["attempt_id"] == "ATT_MAIN_INTEGRATION_1"
        assert dispatched[0]["strategy_version"] == run_once._effective_strategy_version()
        # An unrelated existing position is recorded for forensics only and does
        # not impose a portfolio-size admission cap.
        assert dispatched[0]["portfolio_context"]["open_positions_count"] == 1
        assert dispatched[0]["portfolio_context"]["production_gate_applied"] is False
        assert dispatched[0]["open_positions_snapshot"]["positions"][0]["symbol"] == "BBB-USDT"
        assert dispatched[0]["vwap_context"]["status"] == "PRESENT"
        assert dispatched[0]["vwap_context"]["kind"] == "PARTIAL_UTC_DAY_TP_VWAP"
        assert dispatched[0]["market_snapshot"]["market_spread_metric"] == "cross_venue_price_deviation"
        assert len(sent_signals) == 1
        assert sent_signals[0][1]["status"] == "opened_protected"
        assert any(call[0][2] == "EXECUTION_SELECTED" for call in recorded_decisions)
        assert appended_trade_rows and appended_trade_rows[-1]["record_type"] == "TRADE_OPEN"
    else:
        assert not sent_signals
        assert not appended_trade_rows
        assert any(call[0][2] == "EXECUTION_GATE_REJECT" for call in recorded_decisions)
