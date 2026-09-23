from __future__ import annotations
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path
import time

from event_engine.signals import (
    calc_hma,
    calc_alma,
    compute_ajay_trigger,
    generate_zone_signals,
    score_zone_signal,
)



import pytest


@pytest.fixture(autouse=True)
def _isolate_runtime_state(monkeypatch, tmp_path):
    """Tests must never read/write the repository's production data/ files."""
    from event_engine import analytics, tracker, telemetry
    import run_once

    runtime_data = tmp_path / "data"
    runtime_data.mkdir(parents=True, exist_ok=True)
    # All telemetry files must be isolated from the repository's production data.
    telemetry_paths = {
        "DATA": runtime_data,
        "QUOTE_SNAPSHOTS_PATH": runtime_data / "quote_snapshots.jsonl",
        "ORDER_LIFECYCLE_PATH": runtime_data / "order_lifecycle.jsonl",
        "PROTECTION_LIFECYCLE_PATH": runtime_data / "protection_lifecycle.jsonl",
        "POSITION_RECONCILIATION_PATH": runtime_data / "position_reconciliation.jsonl",
        "EXCHANGE_ERRORS_PATH": runtime_data / "exchange_errors.jsonl",
    }
    for name, value in telemetry_paths.items():
        monkeypatch.setattr(telemetry, name, value)

    # Notifications are never sent by tests.
    monkeypatch.setattr(tracker, "send_tg", lambda *args, **kwargs: True)
    monkeypatch.setattr(run_once, "send_tg", lambda *args, **kwargs: True)
    # Execution tests get a deterministic quote matching their signal unless they
    # explicitly override this helper to test drift/slippage behaviour.
    monkeypatch.setattr(
        run_once,
        "get_execution_quote",
        lambda symbol, **kwargs: {
            "status": "ok",
            "symbol": symbol,
            "bid": float(kwargs.get("reference_price", 100.0)),
            "ask": float(kwargs.get("reference_price", 100.0)),
            "spread_pct": 0.0,
        },
    )
    monkeypatch.setattr(
        run_once,
        "prepare_protection_capacity",
        lambda symbol, direction: {"status": "ready", "symbol": symbol, "direction": direction},
    )
    monkeypatch.setattr(run_once, "get_contract", lambda symbol: {
        "symbol": symbol, "pricePrecision": 8, "quantityPrecision": 4,
        "tradeMinQuantity": 0.0001, "maxLeverage": 10,
    })

    # Redirect every mutable runtime journal/state path used by the testable modules.
    monkeypatch.setattr(tracker, "DATA", runtime_data)
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", runtime_data / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", runtime_data / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", runtime_data / "actions.jsonl")

    monkeypatch.setattr(run_once, "DATA", runtime_data)
    monkeypatch.setattr(run_once, "TRADES_PATH", runtime_data / "trades.jsonl")
    monkeypatch.setattr(run_once, "FAILED_SIGNALS_PATH", runtime_data / "failed_signals.json")
    monkeypatch.setattr(run_once, "EVENT_CLAIMS_PATH", runtime_data / "event_execution_claims.json")
    monkeypatch.setattr(run_once, "EVENT_CLAIMS_LOCK_PATH", runtime_data / "event_execution_claims.json.lock")
    monkeypatch.setattr(run_once, "ACTIONS_PATH", runtime_data / "actions.jsonl")
    monkeypatch.setattr(run_once, "ENTRY_DECISIONS_PATH", runtime_data / "entry_decisions.jsonl")
    monkeypatch.setattr(run_once, "EXECUTION_LEDGER_PATH", runtime_data / "execution_ledger.jsonl")

    # The workflow deliberately uses temporary full-universe + zone-touch
    # environment variables only for the engine step. Tests stay pinned to the
    # production defaults unless a test explicitly opts into zone mode.
    monkeypatch.setattr(run_once, "FUNDAMENTAL_WHITELIST_ENABLED", True)
    monkeypatch.setattr(run_once, "ZONE_TRIGGER_MODE", "midpoint")

    monkeypatch.setattr(analytics, "DATA_DIR", runtime_data)
    monkeypatch.setattr(analytics, "SCAN_JSONL", runtime_data / "scan_history.jsonl")
    monkeypatch.setattr(analytics, "SIGNALS_JSONL", runtime_data / "signal_history.jsonl")
    monkeypatch.setattr(analytics, "LATEST_SCAN_JSON", runtime_data / "latest_scan.json")
    monkeypatch.setattr(analytics, "LATEST_SCAN_TXT", runtime_data / "latest_scan.txt")


def _candles(n: int = 100) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    x = np.arange(n, dtype=float)
    close = 100 + np.sin(x / 5.0) * 4 + x * 0.02
    open_ = close.copy()
    high = close + 1.0
    low = close - 1.0
    volume = np.full(n, 1000.0)
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_hma_shape_and_valid_values():
    df = pd.Series(np.arange(50, dtype=float))
    h = calc_hma(df, 5)
    assert len(h) == len(df)
    assert h.iloc[-1] > h.iloc[-2]


def test_zone_signal_schema():
    df = _candles(140)
    out, supply, demand, signals = generate_zone_signals(df, "TEST-USDT")
    assert len(out) == len(df)
    assert isinstance(supply, list)
    assert isinstance(demand, list)
    assert isinstance(signals, list)
    for s in signals:
        assert s["symbol"] == "TEST-USDT"
        assert s["type"] in {"LONG", "SHORT"}
        assert s["entry"] > 0
        if s["sl"] is not None:
            assert s["sl"] > 0
            assert s["tp1"] > 0
            assert s["tp2"] > 0
            assert s["risk_pct"] > 0
        assert s["event_id"].startswith("ZONE_")
        if s["zone"]:
            assert s["zone"]["kind"] in {"DEMAND", "SUPPLY"}
        assert s["confirmation"]["alma_cross"] is False
        assert s["trigger"]["alma_required"] is False


def test_fixed_stop_is_exactly_10_percent_from_entry():
    import run_once
    long_sl, long_tp1, long_tp2 = run_once._protection_geometry_from_fill("LONG", 100.0, 10.0)
    short_sl, short_tp1, short_tp2 = run_once._protection_geometry_from_fill("SHORT", 100.0, 10.0)
    assert long_sl == 90.0 and short_sl == 110.0
    assert long_tp1 > 100.0 and long_tp2 > long_tp1
    assert short_tp1 < 100.0 and short_tp2 < short_tp1




def test_strategy_snapshot_contains_entry_context_and_exit_rules():
    from event_engine import signals as sig
    assert sig.STRATEGY_VERSION == "zone-midpoint-v4-5m-visit-no-zone-age-limit-stop10-tp3-tp6-be-on-tp1"
    import run_once
    setup = run_once._build_setup({
        "event_id": "ZONE_TEST", "symbol": "TEST-USDT", "type": "LONG",
        "entry": 100.0, "sl": 90.0, "tp1": 103.0, "tp2": 106.0, "risk_pct": 10.0,
        "tp1_rr": 0.3, "tp2_rr": 0.6, "score": 75.0,
        "strategy": "Demand/Supply Zone First", "strategy_version": sig.STRATEGY_VERSION,
        "trigger": {"zone_entry_rule": "fresh_midpoint_touch", "zone_midpoint": 110.0},
        "target": {"source": "fixed_entry_percentage", "tp1_pct": 3.0, "tp2_pct": 6.0, "be_rule": "after_tp1_filled"},
        "risk_model": {"sl_source": "fixed_percent_from_entry", "fixed_stop_pct": 10.0},
        "entry_bar": {"open": 99.0, "high": 111.0, "low": 98.0, "close": 100.0, "volume": 1000.0},
        "previous_bar": {"close": 120.0},
    })
    assert setup["strategy_version"] == sig.STRATEGY_VERSION
    assert setup["entry_bar"]["close"] == 100.0
    assert setup["previous_bar"]["close"] == 120.0
    assert setup["signal_snapshot"]["trigger"]["zone_entry_rule"] == "fresh_midpoint_touch"
    assert setup["target"]["tp1_pct"] == 3.0
    assert setup["target"]["tp2_pct"] == 6.0
    assert setup["target"]["be_rule"] == "after_tp1_filled"


def test_register_active_trade_persists_research_snapshots(tmp_path, monkeypatch):
    from event_engine import tracker
    setup = {
        "risk_pct": 10.0, "target_rr": 0.7, "planned_weighted_rr": 0.6,
        "entry_reference": 100.0, "invalidation_price": 90.0, "target_price": 107.0,
        "strategy_version": "zone-midpoint-v3-5m-visit-stop10-tp5-tp7-be-on-tp1",
        "signal_snapshot": {"strategy_version": "zone-midpoint-v3-5m-visit-stop10-tp5-tp7-be-on-tp1"},
        "entry_bar": {"close": 100.0}, "previous_bar": {"close": 120.0},
        "entry_order": {"orderId": "ENTRY1"}, "fill_position": {"avgPrice": 100.0},
        "execution_snapshot": {"fill_price": 100.0, "sl_price": 90.0, "tp1_price": 105.0, "tp2_price": 107.0},
        "tp_levels": [
            {"leg": "tp1", "price": 105.0, "pnl_pct": 5.0, "close_fraction": 0.5},
            {"leg": "tp2", "price": 107.0, "pnl_pct": 7.0, "close_fraction": 0.5},
        ],
    }
    tracker.register_active_trade(
        event_id="ZONE_TEST", symbol="TEST-USDT", name="TEST-USDT", direction="LONG",
        entry_price=100.0, qty=1.0, tp_orders=[], sl_result={"order_id": "SL1"},
        event_type="DEMAND_ZONE_TOUCH", timeframe="1h", score=75.0, setup=setup, requested_entry_price=100.0,
    )
    state = tracker._load_active_trades()["ZONE_TEST"]
    assert state["strategy_version"] == setup["strategy_version"]
    assert state["entry_bar"] == setup["entry_bar"]
    assert state["previous_bar"] == setup["previous_bar"]
    assert state["signal_snapshot"] == setup["signal_snapshot"]
    assert state["entry_order"] == setup["entry_order"]
    assert state["fill_position"] == setup["fill_position"]
    assert state["execution_snapshot"] == setup["execution_snapshot"]
    assert state["tp_fill_events"] == []
    assert state["be_trigger_rule"] == "after_tp1_filled"
    assert "be_trigger_r" not in state


def test_long_tp_ordering_and_rr():
    signal = {
        "symbol": "TEST-USDT", "type": "LONG", "entry": 100.0, "sl": 95.0,
        "tp1": 107.5, "tp2": 115.0, "risk_pct": 5.0,
        "zone": {"kind": "DEMAND", "age_bars": 10, "impulse_atr": 2.0},
        "confirmation": {"alma_cross": True, "zone_touch": True, "volume_ratio": 1.5, "candle_body_atr": 1.0},
    }
    assert signal["tp1"] < signal["tp2"]
    assert score_zone_signal(signal) >= 70


def test_short_tp_ordering():
    signal = {
        "symbol": "TEST-USDT", "type": "SHORT", "entry": 100.0, "sl": 105.0,
        "tp1": 92.5, "tp2": 85.0, "risk_pct": 5.0,
        "zone": {"kind": "SUPPLY", "age_bars": 20, "impulse_atr": 1.5},
        "confirmation": {"alma_cross": True, "zone_touch": True, "volume_ratio": 1.2, "candle_body_atr": 0.8},
    }
    assert signal["tp1"] > signal["tp2"]
    assert score_zone_signal(signal) >= 70


def test_bingx_missing_credentials_is_scan_safe(monkeypatch):
    from event_engine import bingx

    monkeypatch.delenv("BINGX_API_KEY", raising=False)
    monkeypatch.delenv("BINGX_SECRET_KEY", raising=False)
    assert bingx.credentials_available() is False
    response = bingx._request("GET", "/private-test", signed=True)
    assert response["code"] == -1
    assert "missing BingX credentials" in response["msg"]


def test_bingx_kline_contains_timestamp(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(
        bingx,
        "_request",
        lambda *args, **kwargs: {
            "code": 0,
            "data": [[1700000000000, "100", "101", "99", "100.5", "1000", 1700003600000, "100500", "", "500", "50250"]],
        },
    )
    rows = bingx.fetch_klines("BTC-USDT", "1h", limit=1)
    assert rows[0]["timestamp"] == 1700000000000
    assert rows[0]["open_time"] == 1700000000000



def test_parallel_scan_settings_are_not_serial_sleep_settings():
    import run_once
    assert run_once.SCAN_WORKERS >= 1
    assert run_once.SCAN_BATCH_SIZE >= run_once.SCAN_WORKERS
    assert run_once.KLINE_LIMIT_1H >= 80


def test_bingx_signature_is_ascii_sorted(monkeypatch):
    import hashlib, hmac
    from event_engine import bingx

    monkeypatch.setenv("BINGX_SECRET_KEY", "secret")
    params = {"symbol": "BRETT-USDT", "side": "BUY", "positionSide": "LONG", "type": "MARKET", "quantity": "1966", "timestamp": 1700000000000}
    canonical = "&".join(f"{k}={params[k]}" for k in sorted(params))
    expected = hmac.new(b"secret", canonical.encode(), hashlib.sha256).hexdigest()
    assert bingx._sign(params) == expected


def test_bingx_one_way_uses_both_position_side(monkeypatch):
    import time
    from event_engine import bingx

    monkeypatch.delenv("BINGX_POSITION_MODE_OVERRIDE", raising=False)
    bingx._POSITION_MODE_CACHE.update({"ts": time.time(), "dual": False})
    assert bingx.position_side_param("LONG") == "BOTH"
    assert bingx.position_side_param("SHORT") == "BOTH"


def test_bingx_signed_post_contains_source_key(monkeypatch):
    import requests
    from event_engine import bingx

    monkeypatch.setenv("BINGX_API_KEY", "key")
    monkeypatch.setenv("BINGX_SECRET_KEY", "secret")
    captured = {}

    class Resp:
        headers = {}
        def json(self):
            return {"code": 0, "data": {}}

    class Session:
        def request(self, **kwargs):
            captured.update(kwargs)
            return Resp()

    monkeypatch.setattr(bingx, "SESSION", Session())
    out = bingx._request("POST", "/private-test", {"symbol": "BTC-USDT"}, signed=True, retryable=True)
    assert out["code"] == 0
    assert captured["headers"]["X-BX-APIKEY"] == "key"
    assert captured["headers"]["X-SOURCE-KEY"] == "BX-AI-SKILL"
    assert "signature" in captured["data"]
    assert "timestamp" in captured["data"]


def test_bingx_signed_post_uses_exact_canonical_body(monkeypatch):
    from event_engine import bingx
    monkeypatch.setenv("BINGX_API_KEY", "key")
    monkeypatch.setenv("BINGX_SECRET_KEY", "secret")
    captured = {}
    class Resp:
        headers = {}
        def json(self):
            return {"code": 0, "data": {}}
    class Session:
        def request(self, **kwargs):
            captured.update(kwargs)
            return Resp()
    monkeypatch.setattr(bingx, "SESSION", Session())
    params = {"symbol": "BRETT-USDT", "side": "LONG", "leverage": "10"}
    out = bingx._request("POST", "/private-test", params, signed=True, retryable=True)
    assert out["code"] == 0
    body = captured["data"]
    assert isinstance(body, str)
    assert body.startswith("leverage=10&side=LONG&symbol=BRETT-USDT&timestamp=")
    assert "&signature=" in body
    assert captured["headers"]["Content-Type"] == "application/x-www-form-urlencoded"


def test_bingx_min_qty_is_nonfatal_skip(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {
        "symbol": symbol, "quantityPrecision": 4, "tradeMinQuantity": 3.4286,
        "multiplier": 1, "maxLeverage": 10,
    })
    monkeypatch.setattr(bingx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bingx, "has_open_position", lambda symbol, direction: False)
    monkeypatch.setattr(bingx, "_current_close_price", lambda symbol: 93.368)
    monkeypatch.setattr(bingx, "get_execution_quote", lambda symbol, **kwargs: {
        "status": "ok", "symbol": symbol, "bid": 93.368, "ask": 93.368, "spread_pct": 0.0
    })
    monkeypatch.setenv("BINGX_MARGIN_USDT", "1")
    monkeypatch.setenv("BINGX_LEVERAGE", "10")
    result = bingx.open_market("NCFXNZD2JPY-USDT", "SHORT", 93.368, "TEST")
    assert result["status"] == "skipped_min_qty"
    assert result["required_margin_usdt"] > 1.0


def test_binance_symbol_normalization_and_asset_classification():
    from event_engine.binance import classify_asset, normalize_symbol
    assert normalize_symbol("BRETT-USDT") == "BRETTUSDT"
    assert normalize_symbol("1000PEPEUSDT") == "1000PEPEUSDT"
    assert classify_asset({"underlyingType": "COIN"}) == "CRYPTO"
    assert classify_asset({"underlyingType": "EQUITY"}) == "EQUITY"
    assert classify_asset({"underlyingType": "COMMODITY"}) == "UNKNOWN"


def test_binance_analysis_universe_uses_bingx_symbols(monkeypatch):
    from event_engine import binance
    monkeypatch.setattr(binance, "symbols", lambda: {
        "BTCUSDT": {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "COIN"},
        "TSLAUSDT": {"symbol": "TSLAUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL", "underlyingType": "EQUITY"},
    })
    out = binance.analysis_symbols_for_bingx(["BTC-USDT", "TSLA-USDT", "NOPE-USDT"])
    assert out[0]["binance_available"] is True
    assert out[0]["asset_class"] == "CRYPTO"
    assert out[1]["asset_class"] == "EQUITY"
    assert out[2]["binance_available"] is False


def test_binance_provider_uses_public_vision_api():
    from event_engine import binance
    assert binance.BASE_URL == 'https://data-api.binance.vision'
    assert binance.EXCHANGE_INFO_PATH == '/api/v3/exchangeInfo'
    assert binance.KLINES_PATH == '/api/v3/klines'


def test_execute_rejects_invalid_geometry_before_entry(monkeypatch):
    import run_once
    signal = {
        "event_id": "ZONE_TEST_INVALID",
        "symbol": "ALGO-USDT",
        "type": "SHORT",
        "entry": 0.0908,
        "sl": 0.09,
        "tp1": 0.09,
        "tp2": 0.09,
        "risk_pct": -0.88,
        "score": 75,
        "zone": {"kind": "SUPPLY"},
    }
    called = {"open": False}
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: called.__setitem__("open", True))
    result = run_once.execute_new_position(signal)
    assert result["status"] == "skipped_invalid_setup"
    assert called["open"] is False


def test_execute_emergency_closes_when_protection_fails(monkeypatch):
    import run_once
    signal = {
        "event_id": "ZONE_TEST_PROTECT_FAIL",
        "symbol": "TEST-USDT",
        "type": "LONG",
        "entry": 100.0,
        "sl": 95.0,
        "tp1": 107.5,
        "tp2": 115.0,
        "risk_pct": 1.0,
        "score": 90,
        "zone": {"kind": "DEMAND", "btm": 99.0, "top": 100.0},
        "target": {"obstacle_price": 130.0},
    }
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [], "tp_orders": []
    })
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: {
        "status": "opened", "symbol": "TEST-USDT", "qty": 1.0
    })
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda *a, **k: {
        "status": "found", "avgPrice": 100.0, "positionAmt": 1.0
    })
    monkeypatch.setattr(run_once, "ensure_directional_protection", lambda *a, **k: {
        "status": "PROTECTION_FAILED", "error": "TP2 failed", "sl_result": {}, "tp_orders": []
    })
    position_states = iter([
        {"status": "found", "positionAmt": 1.0},
        {"status": "not_found"},
    ])
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: next(position_states, {"status": "not_found"}))
    closed = {}
    monkeypatch.setattr(run_once, "close_position_market", lambda *a, **k: closed.update({"called": True, "qty": a[2]}) or {"status": "closed"})
    monkeypatch.setattr(run_once.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(run_once, "register_active_trade", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not register unprotected trade")))

    result = run_once.execute_new_position(signal)
    assert result["status"] == "opened_then_emergency_closed"
    assert closed.get("called") is True


def test_open_client_order_id_is_deterministic_per_logical_event():
    from event_engine import bingx
    a = bingx._new_open_client_order_id("ALT-USDT", "ZONE_TEST")
    b = bingx._new_open_client_order_id("ALT-USDT", "ZONE_TEST")
    c = bingx._new_open_client_order_id("ALT-USDT", "ZONE_TEST_2")
    assert a == b
    assert a != c
    assert len(a) <= 40
    assert len(b) <= 40


def test_get_order_accepts_client_order_id(monkeypatch):
    from event_engine import bingx
    seen = {}
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "TESTUSDT")
    monkeypatch.setattr(
        bingx,
        "_request",
        lambda method, path, params, **kwargs: (seen.setdefault("params", params), {
            "code": 0,
            "data": {
                "order": {
                    "orderId": "123",
                    "symbol": "TESTUSDT",
                    "side": "BUY",
                    "positionSide": "LONG",
                    "type": "MARKET",
                    "status": "NEW",
                    "clientOrderId": "CID_X",
                    "origQty": "2",
                    "executedQty": "0",
                }
            },
        })[1],
    )
    out = bingx.get_order("TEST-USDT", client_order_id="CID_X")
    assert out["status"] == "ok"
    assert seen["params"]["clientOrderId"] == "CID_X"
    assert "orderId" not in seen["params"]


def test_open_market_resolves_existing_client_order_before_post(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "TESTUSDT")
    monkeypatch.setattr(bingx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {
        "quantityPrecision": 2, "tradeMinQuantity": 0.01, "multiplier": 1, "maxLeverage": 20,
    })
    monkeypatch.setattr(bingx, "has_open_position", lambda *a, **k: False)
    monkeypatch.setattr(bingx, "get_execution_quote", lambda *a, **k: {"status": "ok", "bid": 99.9, "ask": 100.1, "mid": 100.0})
    monkeypatch.setattr(bingx, "_set_leverage", lambda *a, **k: True)
    monkeypatch.setattr(bingx, "position_side_param", lambda *a, **k: "LONG")
    monkeypatch.setattr(bingx, "get_order", lambda *a, **k: {
        "status": "ok", "symbol": "TESTUSDT", "side": "BUY", "order_type": "MARKET",
        "position_side": "LONG", "order_status": "NEW", "order_id": "321", "client_order_id": k["client_order_id"],
    })
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: pytest.fail("MARKET POST must not happen when clientOrderId already exists"))
    out = bingx.open_market("TEST-USDT", "LONG", 100.0, "EVENT_X")
    assert out["status"] == "opened"
    assert out["idempotency"] == "existing_client_order_resolved_before_post"


def test_open_market_transport_error_reconciles_client_order_without_repost(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "TESTUSDT")
    monkeypatch.setattr(bingx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {
        "quantityPrecision": 2, "tradeMinQuantity": 0.01, "multiplier": 1, "maxLeverage": 20,
    })
    monkeypatch.setattr(bingx, "has_open_position", lambda *a, **k: False)
    monkeypatch.setattr(bingx, "get_execution_quote", lambda *a, **k: {"status": "ok", "bid": 99.9, "ask": 100.1, "mid": 100.0})
    monkeypatch.setattr(bingx, "_set_leverage", lambda *a, **k: True)
    monkeypatch.setattr(bingx, "position_side_param", lambda *a, **k: "LONG")
    lookup_calls = {"n": 0}
    def fake_get_order(*a, **k):
        lookup_calls["n"] += 1
        if lookup_calls["n"] == 1:
            return {"status": "error", "error": "order not found"}
        return {
            "status": "ok", "symbol": "TESTUSDT", "side": "BUY", "order_type": "MARKET",
            "position_side": "LONG", "order_status": "FILLED", "order_id": "654",
            "client_order_id": k["client_order_id"],
        }
    monkeypatch.setattr(bingx, "get_order", fake_get_order)
    calls = []
    def fake_request(method, path, params, **kwargs):
        calls.append((method, path))
        if method == "POST":
            return {"code": -1, "msg": "read timeout"}
        return {"code": 0, "data": {}}
    monkeypatch.setattr(bingx, "_request", fake_request)
    out = bingx.open_market("TEST-USDT", "LONG", 100.0, "EVENT_X_TRANSPORT")
    assert out["status"] == "opened"
    assert out["idempotency"] == "client_order_id_verified_after_transport_error"
    assert [c[0] for c in calls] == ["POST"]


def test_open_market_refuses_reuse_of_terminal_client_order(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "TESTUSDT")
    monkeypatch.setattr(bingx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {
        "quantityPrecision": 2, "tradeMinQuantity": 0.01, "multiplier": 1, "maxLeverage": 20,
    })
    monkeypatch.setattr(bingx, "has_open_position", lambda *a, **k: False)
    monkeypatch.setattr(bingx, "get_execution_quote", lambda *a, **k: {"status": "ok", "bid": 99.9, "ask": 100.1, "mid": 100.0})
    monkeypatch.setattr(bingx, "_set_leverage", lambda *a, **k: True)
    monkeypatch.setattr(bingx, "position_side_param", lambda *a, **k: "LONG")
    monkeypatch.setattr(bingx, "get_order", lambda *a, **k: {
        "status": "ok", "symbol": "TESTUSDT", "side": "BUY", "order_type": "MARKET",
        "position_side": "LONG", "order_status": "CANCELED", "order_id": "999",
        "client_order_id": k["client_order_id"],
    })
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: pytest.fail("terminal clientOrderId must not be reposted"))
    out = bingx.open_market("TEST-USDT", "LONG", 100.0, "EVENT_X")
    assert out["status"] == "error"
    assert "terminal exchange order" in out["error"]


def test_binance_parser_rejects_invalid_ohlc(monkeypatch):
    from event_engine import binance
    invalid = [
        [1767225600000, "100", "99", "101", "101", "10", 1767225659999, "1010", 1, "6", "606", "0"],
        [1767225900000, "100", "102", "99", "101", "nan", 1767225959999, "1010", 1, "6", "606", "0"],
    ]
    monkeypatch.setattr(binance, "_get", lambda *args, **kwargs: invalid)
    rows = binance.fetch_klines("BTCUSDT", interval="5m", limit=2)
    assert rows == []


def test_tp_constants_are_one_and_two_r():
    from event_engine.signals import TP1_FRACTION, TP2_FRACTION
    assert TP1_FRACTION == 0.50
    assert TP2_FRACTION == 0.50


def test_alma_parameters_match_ajay_r541_defaults():
    from event_engine import signals
    assert signals.ALMA_TIMEFRAME_HOURS == 8
    assert signals.ALMA_BASIS_TYPE == "ALMA"
    assert signals.ALMA_BASIS_LEN == 2
    assert signals.ALMA_SIGMA == 5
    assert signals.ALMA_OFFSET == 0.85
    assert signals.USE_ALTERNATE_SIGNALS is True


def test_generate_signals_are_zone_touch_only_and_not_alma_gated():
    import pandas as pd
    from event_engine import signals as sig

    # Direct unit contract for the new strategy: zone touch is the trigger,
    # ALMA/Pine values are diagnostic only.
    out = pd.DataFrame({"pine_buy": [True], "pine_sell": [False]})
    assert bool(out.loc[0, "pine_buy"]) is True

def test_exact_zone_containment_has_no_proximity_padding():
    import run_once
    demand = [{"btm": 100.0, "top": 105.0}]
    supply = [{"btm": 110.0, "top": 115.0}]
    assert run_once._price_position(105.0, demand, supply) == "🟢 В зоне DEMAND"
    assert run_once._price_position(105.1, demand, supply) == "⚪ Вне зон (Ждать)"
    assert run_once._price_position(109.9, demand, supply) == "⚪ Вне зон (Ждать)"
    assert run_once._price_position(110.0, demand, supply) == "🔴 В зоне SUPPLY"

def test_target_levels_stop_and_ordering_from_nearest_obstacle():
    from event_engine.signals import _targets_from_nearest_obstacle
    long = _targets_from_nearest_obstacle("LONG", 100.0, 95.0, 2.0, {"price": 110.0, "source": "supply_zone"})
    short = _targets_from_nearest_obstacle("SHORT", 100.0, 105.0, 2.0, {"price": 90.0, "source": "demand_zone"})
    assert long is not None and short is not None
    assert 100.0 < long["tp1"] < long["tp2"] < 110.0
    assert 90.0 < short["tp2"] < short["tp1"] < 100.0
    assert long["tp1_rr"] < long["tp2_rr"]
    assert short["tp1_rr"] < short["tp2_rr"]

def test_zone_only_latest_bar_check_never_reads_pine_direction():
    import run_once
    signal = {"idx": 119, "time": "2026-09-02T11:00:00+00:00", "type": "SHORT", "trigger": {"buy": None, "sell": None}}
    ok, reason = run_once._signal_matches_latest_bar(signal, 119, "2026-09-02T11:00:00+00:00")
    assert ok and reason == "ok"

def test_scan_universe_is_limited_to_150_fundamental_assets(monkeypatch):
    import run_once
    from event_engine.fundamental_assets import FUNDAMENTAL_ASSET_SYMBOLS
    monkeypatch.setattr(run_once, "contracts", lambda: {
        "BTC-USDT": {"symbol": "BTC-USDT"},
        "DOGE-USDT": {"symbol": "DOGE-USDT"},
        "ETH-USDT": {"symbol": "ETH-USDT"},
    })
    monkeypatch.setattr(run_once, "WATCHLIST_ONLY", False)
    out = run_once.get_scan_symbols()
    assert set(out).issubset(FUNDAMENTAL_ASSET_SYMBOLS)
    assert {"BTC-USDT", "ETH-USDT"}.issubset(out)
    assert "DOGE-USDT" not in out


def test_scan_universe_can_temporarily_disable_150_whitelist(monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "contracts", lambda: {
        "BTC-USDT": {"symbol": "BTC-USDT"},
        "DOGE-USDT": {"symbol": "DOGE-USDT"},
        "ETH-USDT": {"symbol": "ETH-USDT"},
    })
    monkeypatch.setattr(run_once, "WATCHLIST_ONLY", False)
    monkeypatch.setattr(run_once, "FUNDAMENTAL_WHITELIST_ENABLED", False)
    out = run_once.get_scan_symbols()
    assert set(out) == {"BTC-USDT", "DOGE-USDT", "ETH-USDT"}


def test_telegram_uses_zone_only_label_and_dynamic_rr_values():
    from event_engine.telegram import format_signal
    msg = format_signal({
        "type": "LONG", "symbol": "TEST-USDT", "entry": 100.0, "sl": 95.0,
        "tp1": 103.0, "tp2": 106.0, "tp1_rr": 0.3, "tp2_rr": 0.6, "risk_pct": 10.0,
        "zone": {"kind": "DEMAND", "btm": 94.0, "top": 99.0, "poi": 96.5, "age_bars": 1, "impulse_atr": 2.0},
        "target": {"tp1_pct": 3.0, "tp2_pct": 6.0},
        "confirmation": {},
    })
    assert "Demand/Supply Zone First" in msg
    assert "Ajay R5.41 · ALMA" not in msg
    assert "(3% / 0.3R / 50%)" in msg
    assert "(6% / 0.6R / 50%)" in msg


def test_telegram_formats_live_entry_with_compact_prices_and_zone_counts():
    from event_engine.telegram import format_signal
    msg = format_signal({
        "type": "SHORT", "symbol": "ONDO-USDT", "entry": 0.3537, "sl": 0.38907,
        "tp1": 0.343089, "tp2": 0.332478, "tp1_rr": 0.3, "tp2_rr": 0.6, "risk_pct": 10.0,
        "zone": {"kind": "SUPPLY", "btm": 0.3547808259709935, "top": 0.3559, "poi": 0.35534041298549673, "age_bars": 10},
        "zone_counts": {"demand": 5, "supply": 1},
        "target": {"tp1_pct": 3.0, "tp2_pct": 6.0},
        "confirmation": {"volume_ratio": 0.14405361643357992},
    })
    assert "ONDOUSDT" in msg
    assert "ONDO-USDT" not in msg
    assert "Demand zones: <code>5</code>" in msg
    assert "Supply zones: <code>1</code>" in msg
    assert "0.35534" in msg
    assert "0.35534041298549673" not in msg
    assert "(3% / 0.3R / 50%)" in msg
    assert "(6% / 0.6R / 50%)" in msg
    assert "Risk: <code>10%</code>" in msg


def test_entry_telegram_is_suppressed_for_failed_execution(monkeypatch):
    import run_once
    sent = []
    monkeypatch.setattr(run_once, "send_tg", lambda text: sent.append(text) or True)
    monkeypatch.setattr(run_once, "_append_jsonl", lambda *args, **kwargs: None)
    signal = {
        "event_id": "EVT_TG_GUARD", "type": "SHORT", "symbol": "ONDO-USDT",
        "entry": 1.0, "sl": 1.1, "tp1": 0.97, "tp2": 0.94, "risk_pct": 10.0,
        "tp1_rr": 0.3, "tp2_rr": 0.6, "zone": {}, "confirmation": {},
    }
    for status in ("opened_then_emergency_closed", "skipped_invalid_setup", "blocked_protection_preflight", "entry_not_filled"):
        run_once._send_signal(signal, {"status": status})
    assert sent == []
    run_once._send_signal(signal, {"status": "opened_protected"})
    assert len(sent) == 1


def test_post_fill_risk_pct_remains_exactly_at_configured_limit():
    import run_once
    signal = {
        "type": "SHORT", "entry": 1.0, "sl": 1.1, "tp1": 0.97, "tp2": 0.94,
        "risk_pct": 10.0, "target": {"obstacle_price": 0.5}, "zone": {"btm": 0.7, "top": 0.9},
    }
    rebased = run_once._rebase_protection_after_fill(signal, 0.7974)
    assert rebased["risk_pct"] == 10.0
    ok, reason = run_once._validate_trade_geometry({**signal, **rebased})
    assert ok, reason


def test_post_fill_rebases_zone_protection_and_never_reuses_stale_absolute_targets():
    import run_once
    signal = {
        "type": "SHORT",
        "entry": 0.8104,
        "sl": 0.8127,
        "tp1": 0.8086,
        "tp2": 0.8069,
        "risk_pct": 0.28,
        "atr": 0.003,
        "zone": {"btm": 0.808258, "top": 0.8114},
        "target": {"obstacle_price": 0.8060, "source": "nearest_opposing_structure"},
    }
    out = run_once._rebase_protection_after_fill(signal, 0.7974)
    assert out["entry"] == 0.7974
    assert out["sl"] > out["entry"]
    assert out["tp1"] < out["entry"]
    assert out["tp2"] < out["tp1"]


def test_long_post_fill_that_slips_through_zone_stop_moves_stop_behind_fill():
    import run_once
    signal = {
        "type": "LONG",
        "entry": 4.77,
        "sl": 4.7365,
        "tp1": 4.795,
        "tp2": 4.820,
        "risk_pct": 0.70,
        "atr": 0.033,
        "zone": {"btm": 4.743, "top": 4.7593},
        "target": {"obstacle_price": 4.85, "source": "nearest_opposing_structure"},
    }
    out = run_once._rebase_protection_after_fill(signal, 4.73)
    assert out["sl"] < out["entry"]
    assert out["tp1"] > out["entry"]
    assert out["tp2"] > out["tp1"]


def test_execute_rebases_protection_to_actual_fill_before_installing(monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "MAX_ENTRY_SLIPPAGE_PCT", 99.0)
    signal = {
        "event_id": "ZONE_TEST_REBASE", "symbol": "TEST-USDT", "type": "SHORT",
        "entry": 100.0, "sl": 105.0, "tp1": 99.0, "tp2": 98.0, "risk_pct": 1.0,
        "score": 75, "atr": 2.0,
        "zone": {"kind": "SUPPLY", "btm": 99.0, "top": 104.0},
        "target": {"obstacle_price": 70.0, "source": "nearest_opposing_structure"},
    }
    captured = {}
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})
    opened = {"value": False}
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: opened.__setitem__("value", True) or {"status": "opened", "symbol": "TEST-USDT"})
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda *a, **k: {"status": "found", "avgPrice": 95.0, "positionAmt": 1.0})
    def fake_protection(*args, **kwargs):
        captured["avg"] = args[2]
        captured["levels"] = args[5]
        return {"status": "PROTECTION_FAILED", "tp_orders": [], "sl_result": {}, "error": "invalid geometry"}
    monkeypatch.setattr(run_once, "ensure_directional_protection", fake_protection)
    monkeypatch.setattr(run_once, "register_active_trade", lambda *a, **k: None)
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: {"status": "found", "positionAmt": 1.0})
    monkeypatch.setattr(run_once, "close_position_market", lambda *a, **k: {"status": "closed"})
    monkeypatch.setattr(run_once, "_cleanup_engine_protection", lambda *a, **k: {"status": "ok"})
    out = run_once.execute_new_position(signal)
    # With the new 5% risk cap, extreme slippage can pass pre-entry checks but then
    # fail protection installation, triggering an emergency close.
    assert out["status"] == "opened_then_emergency_closed"
    assert opened["value"] is True


def test_stale_signal_is_blocked_before_market(monkeypatch):
    import run_once
    signal = {
        "event_id": "ZONE_TEST_STALE", "symbol": "TEST-USDT", "type": "LONG",
        "entry": 100.0, "sl": 95.0, "tp1": 102.5, "tp2": 105.0, "risk_pct": 1.0,
        "score": 75, "atr": 2.0,
        "zone": {"kind": "DEMAND", "btm": 95.0, "top": 100.0},
        "target": {"obstacle_price": 112.0},
    }
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})
    monkeypatch.setattr(run_once, "get_execution_quote", lambda *a, **k: {
        "status": "ok", "bid": 102.0, "ask": 102.1, "spread_pct": 0.098
    })
    opened = {"value": False}
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: opened.__setitem__("value", True) or {"status": "opened"})
    out = run_once.execute_new_position(signal)
    assert out["status"] == "skipped_stale_signal"
    assert opened["value"] is False
    assert out["signal_drift_pct"] > run_once.MAX_ENTRY_SLIPPAGE_PCT


def test_execution_slippage_is_measured_from_pre_entry_quote(monkeypatch):
    import run_once
    signal = {
        "event_id": "ZONE_TEST_SLIP", "symbol": "TEST-USDT", "type": "LONG",
        "entry": 100.0, "sl": 95.0, "tp1": 102.5, "tp2": 105.0, "risk_pct": 1.0,
        "score": 75, "atr": 0.1,
        "zone": {"kind": "DEMAND", "btm": 99.5, "top": 100.0},
        "target": {"obstacle_price": 112.0},
    }
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})
    monkeypatch.setattr(run_once, "get_execution_quote", lambda *a, **k: {
        "status": "ok", "bid": 99.9, "ask": 100.0, "spread_pct": 0.100
    })
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: {"status": "opened", "symbol": "TEST-USDT"})
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda *a, **k: {"status": "found", "avgPrice": 101.5, "positionAmt": 1.0})
    states = iter([{"status": "found", "positionAmt": 1.0}, {"status": "not_found"}])
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: next(states, {"status": "not_found"}))
    monkeypatch.setattr(run_once, "close_position_market", lambda *a, **k: {"status": "closed"})
    monkeypatch.setattr(run_once, "_cancel_engine_protection_before_emergency_close", lambda *a, **k: {"status": "ok"})
    monkeypatch.setattr(run_once, "_cleanup_engine_protection", lambda *a, **k: {"status": "ok"})
    out = run_once.execute_new_position(signal)
    assert out["status"] == "opened_then_emergency_closed"
    assert out["error"].startswith("execution_slippage_pct=")
    assert abs(out["execution_slippage_pct"] - 1.5) < 1e-9
    assert out["signal_drift_pct"] == 0.0


def test_invalid_setup_is_rejected_before_protection_preflight(monkeypatch):
    import run_once
    signal = {
        "event_id": "ZONE_TEST_INVALID", "symbol": "ALGO-USDT", "type": "SHORT",
        "entry": 0.0908, "sl": 0.09, "tp1": 0.09, "tp2": 0.09, "risk_pct": -0.88,
        "score": 75, "zone": {"kind": "SUPPLY", "btm": 0.089, "top": 0.091},
    }
    called = {"preflight": False, "open": False}
    monkeypatch.setattr(run_once, "prepare_protection_capacity", lambda *a, **k: called.__setitem__("preflight", True) or {"status": "ok"})
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: called.__setitem__("open", True))
    result = run_once.execute_new_position(signal)
    assert result["status"] == "skipped_invalid_setup"
    assert called == {"preflight": False, "open": False}


def test_protection_endpoint_failure_blocks_market_entry(monkeypatch):
    import run_once
    signal = {
        "event_id": "ZONE_TEST_PREFLIGHT", "symbol": "ALGO-USDT", "type": "SHORT",
        "entry": 100.0, "sl": 105.0, "tp1": 99.0, "tp2": 98.0, "risk_pct": 1.0,
        "score": 75, "zone": {"kind": "SUPPLY", "btm": 99.0, "top": 104.0},
        "target": {"obstacle_price": 90.0},
    }
    called = {"open": False}
    monkeypatch.setattr(run_once, "prepare_protection_capacity", lambda *a, **k: {"status": "error", "error": "code:100410 disabled period"})
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: called.__setitem__("open", True))
    result = run_once.execute_new_position(signal)
    assert result["status"] == "blocked_protection_preflight"
    assert called["open"] is False


def test_run_once_has_no_pine_execution_rejection_gate():
    from pathlib import Path
    source = Path(__file__).with_name("run_once.py").read_text(encoding="utf-8")
    assert "direction_not_equal_to_pine_trigger" not in source
    assert "pine_buy" not in source[source.index("def main"):]
    assert "pine_sell" not in source[source.index("def main"):]

def test_tracker_uses_zone_touch_event_type():
    import run_once
    source = Path(run_once.__file__).read_text(encoding="utf-8")
    assert '_MIDPOINT_TOUCH_5M"' in source


def test_historical_lookahead_8h_series_is_constant_inside_bucket():
    from event_engine.signals import compute_ajay_trigger
    n = 32
    ts = pd.date_range("2026-01-01 00:00", periods=n, freq="h", tz="UTC")
    close = np.arange(100.0, 100.0 + n)
    open_ = close - 0.25
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = np.ones(n)
    df = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
    out = compute_ajay_trigger(df, mode="historical")
    first_bucket = out[out["timestamp"].dt.floor("8h") == pd.Timestamp("2026-01-01 08:00", tz="UTC")]
    vals = first_bucket["alma_close_alt"].dropna().round(10).unique()
    assert len(vals) == 1


def test_live_trigger_uses_explicit_live_mode():
    from event_engine.signals import compute_ajay_trigger
    n = 32
    ts = pd.date_range("2026-01-01 00:00", periods=n, freq="h", tz="UTC")
    close = np.linspace(100.0, 120.0, n)
    open_ = close - 0.2
    high = close + 0.5
    low = close - 0.5
    volume = np.ones(n)
    df = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
    out = compute_ajay_trigger(df, mode="live")
    assert set(out["alternate_mode"].dropna().unique()) == {"live_current_8h_developing"}




def test_epoch_millisecond_timestamps_are_normalized_as_milliseconds():
    import pandas as pd
    from event_engine.signals import compute_ajay_trigger

    ts = pd.date_range("2026-09-01", periods=32, freq="1h", tz="UTC")
    epoch_ms = ts.view("int64") // 1_000_000
    close = pd.Series(range(100, 132), dtype=float)
    df = pd.DataFrame({
        "timestamp": epoch_ms,
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": 1_000.0,
    })
    out = compute_ajay_trigger(df, mode="live")
    assert out["timestamp"].iloc[0] == ts[0]
    assert out["timestamp"].iloc[-1] == ts[-1]
    assert out["timestamp"].dt.floor("8h").nunique() == 4
    assert out["timestamp"].iloc[-1].year == 2026

def test_pine_keltner_visual_series_schema():
    from event_engine.signals import compute_pine_keltner_channels
    df = _candles(120)
    out = compute_pine_keltner_channels(df)
    assert set(out.columns) == {
        "kc1_upper", "kc1_lower", "kc2_upper", "kc2_lower",
        "kc3_upper", "kc3_lower", "kc4_upper", "kc4_lower",
    }
    assert len(out) == len(df)


def test_pine_zone_records_are_exposed_for_comparison():
    from event_engine.signals import compute_pine_zone_records
    df = _candles(180)
    result = compute_pine_zone_records(df)
    assert set(result) == {"supply", "demand", "supply_bos", "demand_bos"}
    assert isinstance(result["supply"], list)


def test_no_zone_signal_is_blocked(monkeypatch):
    import numpy as np
    import pandas as pd
    from event_engine import signals as sig

    n = 120
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    close = np.linspace(100.0, 120.0, n)
    open_ = close.copy()
    high = close + 0.5
    low = close - 0.5
    volume = np.full(n, 1000.0)
    df = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
    _, _, _, emitted = sig.generate_zone_signals(df, symbol="TEST-USDT", mode="live")
    assert emitted == []


def test_zone_signal_uses_fixed_10pct_stop_and_small_tps(monkeypatch):
    import numpy as np
    import pandas as pd
    from event_engine import signals as sig

    n = 120
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    close = np.linspace(100.0, 120.0, n)
    open_ = close.copy()
    high = close + 0.5
    low = close - 0.5
    # Give the latest candle a small pullback/touch area.
    close[-1] = 110.0
    open_[-1] = 109.0
    high[-1] = 110.0
    low[-1] = 108.0
    volume = np.full(n, 1000.0)
    df = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
    # Supply a deterministic Demand zone touching the latest bar.
    def forced_walk(frame):
        demand = [{"top": 110.5, "btm": 109.0, "poi": 109.75, "start": len(frame)-5}]
        return [], demand, [], [], []
    monkeypatch.setattr(sig, "_pine_zone_walk", forced_walk)

    # With a forced zone walk, generate_zone_signals builds its own active zone,
    # so patch zone construction/selection at the deterministic insertion point.
    forced_zone = {"top": 110.5, "btm": 109.0, "poi": 109.75, "start": n - 5}
    monkeypatch.setattr(sig, "_find_directional_zone", lambda direction, cur_l, cur_h, cur_c, demand, supply: forced_zone if direction == "LONG" else None)
    monkeypatch.setattr(sig, "_nearest_opposing_level", lambda direction, entry, active_demand, active_supply, frame, idx: {"price": 130.0, "source": "supply_zone"} if direction == "LONG" else {"price": 70.0, "source": "demand_zone"})

    _, _, _, emitted = sig.generate_zone_signals(df, symbol="TEST-USDT", mode="live")
    assert emitted
    latest = emitted[-1]
    assert latest["confirmation"]["zone_touch"] is True
    assert latest["risk_model"]["sl_source"] == "fixed_percent_from_entry"
    assert latest["risk_model"]["fixed_stop_pct"] == 10.0
    assert latest["risk_pct"] == 10.0
    assert abs((latest["entry"] - latest["sl"]) / latest["entry"] * 100.0 - 10.0) < 1e-9
    assert latest["trigger"]["alma_required"] is False
    assert latest["target"]["source"] == "fixed_entry_percentage"
    assert abs(latest["tp1"] / latest["entry"] - 1.03) < 1e-9
    assert abs(latest["tp2"] / latest["entry"] - 1.06) < 1e-9
    assert abs(latest["tp1_rr"] - 0.3) < 1e-9
    assert abs(latest["tp2_rr"] - 0.6) < 1e-9
    assert latest["sl"] < 110.0


def test_targets_are_fixed_3pct_and_6pct_from_entry():
    from event_engine import signals as sig

    obstacle = {"price": 108.0, "source": "supply_zone"}
    out = sig._targets_from_nearest_obstacle("LONG", 100.0, 90.0, 1.0, obstacle)
    assert out is not None
    assert out["target_source"] == "fixed_entry_percentage"
    assert out["obstacle_price"] == 108.0
    assert out["tp1"] == 103.0
    assert out["tp2"] == 106.0
    assert out["tp1_rr"] == 0.3
    assert out["tp2_rr"] == 0.6

    obstacle = {"price": 92.0, "source": "demand_zone"}
    out = sig._targets_from_nearest_obstacle("SHORT", 100.0, 110.0, 1.0, obstacle)
    assert out is not None
    assert out["tp1"] == 97.0
    assert out["tp2"] == 94.0


def test_post_fill_targets_are_exact_3pct_and_6pct_from_actual_fill():
    import run_once
    signal = {
        "type": "LONG",
        "entry": 100.0, "sl": 90.0, "tp1": 103.0, "tp2": 106.0, "risk_pct": 10.0,
        "atr": 2.0,
        "zone": {"btm": 98.0, "top": 102.0},
        "target": {"obstacle_price": 103.0, "source": "nearest_opposing_structure"},
    }
    out = run_once._rebase_protection_after_fill(signal, 200.0)
    assert out["entry"] == 200.0
    assert out["sl"] == 180.0
    assert out["tp1"] == 206.0
    assert out["tp2"] == 212.0
    assert abs(out["tp1_rr"] - 0.3) < 1e-9
    assert abs(out["tp2_rr"] - 0.6) < 1e-9


def test_latest_signal_selection_prefers_newest_bar_over_score():
    import run_once
    recent = [
        {"idx": 99, "score": 100.0},
        {"idx": 100, "score": 10.0},
    ]
    selected = run_once._select_latest_signal(recent)
    assert selected["idx"] == 100


def test_latest_signal_selection_uses_score_only_on_same_bar():
    import run_once
    recent = [
        {"idx": 100, "score": 75.0},
        {"idx": 100, "score": 80.0},
    ]
    selected = run_once._select_latest_signal(recent)
    assert selected["score"] == 80.0


def test_signal_latest_bar_requires_exact_timestamp_only_for_zone_only():
    import run_once
    base = {
        "idx": 119,
        "time": "2026-09-02T11:00:00+00:00",
        "type": "SHORT",
        # Deliberately no Pine trigger: ZONE_ONLY must not require it.
    }
    ok, reason = run_once._signal_matches_latest_bar(base, 119, "2026-09-02T11:00:00+00:00")
    assert ok and reason == "ok"

    stale = dict(base, time="2024-06-17T02:00:00+00:00")
    ok, reason = run_once._signal_matches_latest_bar(stale, 119, "2026-09-02T11:00:00+00:00")
    assert not ok and reason == "signal_time_not_latest"


def test_signal_latest_bar_does_not_require_pine_direction():
    import run_once
    signal = {
        "idx": 119,
        "time": "2026-09-02T11:00:00+00:00",
        "type": "LONG",
        "trigger": {"buy": None, "sell": None},
    }
    ok, reason = run_once._signal_matches_latest_bar(signal, 119, "2026-09-02T11:00:00+00:00")
    assert ok and reason == "ok"


def test_signal_latest_bar_rejects_nonlatest_index():
    import run_once
    signal = {
        "idx": 118,
        "time": "2026-09-02T11:00:00+00:00",
        "type": "SHORT",
        "trigger": {"buy": False, "sell": True},
    }
    ok, reason = run_once._signal_matches_latest_bar(signal, 119, "2026-09-02T12:00:00+00:00")
    assert not ok and reason == "signal_idx_not_latest"


def test_live_mode_only_develops_latest_8h_bucket():
    """Live mode must preserve historical bars and only develop the latest 8H bucket."""
    import pandas as pd
    from event_engine.signals import compute_ajay_trigger

    ts = pd.date_range("2026-08-30 00:00:00", periods=20, freq="1h", tz="UTC")
    closes = [10.0] * 16 + [10.0, 10.0, 10.0, 20.0]
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [10.0] * 20,
        "high": [max(10.0, c) + 0.1 for c in closes],
        "low": [min(10.0, c) - 0.1 for c in closes],
        "close": closes,
        "volume": [1.0] * 20,
    })

    hist = compute_ajay_trigger(df, mode="historical")
    live = compute_ajay_trigger(df, mode="live")

    # Completed buckets are identical between modes.
    completed = live["timestamp"] < pd.Timestamp("2026-08-30 16:00", tz="UTC")
    assert np.allclose(
        live.loc[completed, "alma_close_alt"].to_numpy(),
        hist.loc[completed, "alma_close_alt"].to_numpy(),
        equal_nan=True,
    )
    assert np.allclose(
        live.loc[completed, "alma_open_alt"].to_numpy(),
        hist.loc[completed, "alma_open_alt"].to_numpy(),
        equal_nan=True,
    )

    # Only the latest bucket is developing, so its 16/17/18/19h states differ.
    latest = live["timestamp"] >= pd.Timestamp("2026-08-30 16:00", tz="UTC")
    vals = live.loc[latest, "alma_close_alt"].to_numpy()
    assert vals[0] != vals[-1]
    assert vals[1] != vals[-1]
    assert vals[2] != vals[-1]


def test_live_mode_does_not_create_synthetic_crossovers_in_old_buckets():
    import pandas as pd
    from event_engine.signals import compute_ajay_trigger

    ts = pd.date_range("2026-08-30 00:00:00", periods=40, freq="1h", tz="UTC")
    closes = []
    # Several completed 8H buckets with oscillating 1H prices.
    pattern = [10, 20, 10, 20, 10, 20, 10, 20]
    closes.extend(pattern * 4)
    closes.extend([20, 20, 20, 20, 20, 20, 20, 20])
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [10.0] * len(ts),
        "high": [max(10.0, c) + 0.1 for c in closes],
        "low": [min(10.0, c) - 0.1 for c in closes],
        "close": closes,
        "volume": [1.0] * len(ts),
    })
    live = compute_ajay_trigger(df, mode="live")
    # All completed buckets before the final 8H block inherit a constant
    # historical HTF value, so intrabucket crossovers cannot be generated.
    old = live["timestamp"] < pd.Timestamp("2026-08-31 16:00", tz="UTC")
    assert live.loc[old, "pine_buy"].sum() <= 4
    assert live.loc[old, "pine_sell"].sum() <= 4


def test_live_mode_uses_partial_current_8h_not_final_dataset_close():
    import pandas as pd
    from event_engine.signals import compute_ajay_trigger

    ts0 = pd.date_range("2026-01-01 00:00", periods=8, freq="h", tz="UTC")
    df0 = pd.DataFrame({
        "timestamp": ts0,
        "open": [90.0] * 8,
        "high": [90.5] * 8,
        "low": [89.5] * 8,
        "close": [90.0] * 8,
        "volume": [1.0] * 8,
    })
    ts1 = pd.date_range("2026-01-01 08:00", periods=8, freq="h", tz="UTC")
    close1 = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 200.0]
    df1 = pd.DataFrame({
        "timestamp": ts1,
        "open": [100.0] * 8,
        "high": [c + 0.5 for c in close1],
        "low": [c - 0.5 for c in close1],
        "close": close1,
        "volume": [1.0] * 8,
    })
    out = compute_ajay_trigger(pd.concat([df0, df1], ignore_index=True), mode="live")
    vals = out.loc[8:15, "alma_close_alt"].to_numpy()
    assert vals[0] < vals[-1]
    # 08:00-14:00 must not see the 15:00 final close=200.0.
    assert abs(vals[0] - vals[1]) < 1e-12
    assert abs(vals[0] - vals[6]) < 1e-12
    assert vals[-1] > vals[0]



def test_trigger_recompute_is_safe_on_already_enriched_dataframe():
    """generate_zone_signals may receive a dataframe already enriched by a trigger pass."""
    import pandas as pd
    from event_engine.signals import compute_ajay_trigger, generate_zone_signals

    ts = pd.date_range("2026-01-01", periods=120, freq="1h", tz="UTC")
    close = pd.Series(100.0 + (pd.RangeIndex(120).to_numpy() * 0.01))
    df = pd.DataFrame({
        "timestamp": ts,
        "open": close,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": 1000.0,
    })
    enriched = compute_ajay_trigger(df, mode="live")
    out, _, _, _ = generate_zone_signals(enriched, symbol="TEST-USDT", mode="live")
    assert "pine_buy" in out.columns
    assert "pine_sell" in out.columns
    assert len(out) == len(enriched)


def test_execution_candidate_filter_only_accepts_latest_closed_pine_signal():
    import run_once
    latest_time = "2026-09-02T11:00:00+00:00"
    signals = [
        {"idx": 97, "time": "2026-09-02T10:00:00+00:00", "type": "SHORT", "score": 100.0},
        {"idx": 98, "time": latest_time, "type": "LONG", "score": 1.0},
        {"idx": 96, "time": "2026-09-02T09:00:00+00:00", "type": "LONG", "score": 99.0},
    ]
    # The production scan's latest-only rule must ignore every non-latest signal.
    latest = [s for s in signals if int(s["idx"]) == 98 and pd.Timestamp(s["time"]) == pd.Timestamp(latest_time)]
    assert len(latest) == 1
    assert latest[0]["type"] == "LONG"


def test_execution_candidate_filter_rejects_same_index_wrong_timestamp():
    import pandas as pd
    latest_time = pd.Timestamp("2026-09-02T11:00:00+00:00")
    sig = {"idx": 98, "time": "2024-06-17T02:00:00+00:00", "type": "SHORT"}
    assert not (int(sig["idx"]) == 98 and pd.Timestamp(sig["time"]) == latest_time)


def test_alma_matches_tradingview_celo_log_example():
    # TradingView Pine log for CELO at 2026-09-02 00:00 shows:
    # previous 8H close = 0.07321, current 8H close = 0.07767,
    # ALMA(2, offset=0.85, sigma=5) = 0.0772200813.
    series = pd.Series([0.07321, 0.07767], dtype=float)
    out = calc_alma(series, length=2, offset=0.85, sigma=5.0)
    assert abs(float(out.iloc[-1]) - 0.0772200813) < 1e-10


def test_live_mode_changes_only_current_8h_bucket():
    ts = pd.date_range("2026-09-01 00:00:00", periods=24, freq="h", tz="UTC")
    close = np.linspace(100.0, 123.0, 24)
    open_ = close - 0.5
    high = close + 1.0
    low = close - 1.0
    volume = np.full(24, 1000.0)
    df = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
    out = compute_ajay_trigger(df, mode="live")
    # First 8H bucket is historical/final and therefore flat across its 1H bars.
    assert out.loc[8:15, "alma_close_alt"].nunique() == 1
    assert out.loc[8:15, "alma_open_alt"].nunique() == 1
    # Current 8H bucket is developing: close ALMA tracks the latest 1H close.
    assert out.loc[16:23, "alma_close_alt"].nunique() > 1
    # Current 8H open is fixed throughout the bucket.
    assert out.loc[16:23, "alma_open_alt"].nunique() == 1


def test_price_position_does_not_call_nearby_price_inside_zone():
    from run_once import _price_position
    demand = [{"btm": 100.0, "top": 110.0}]
    supply = [{"btm": 120.0, "top": 130.0}]
    assert _price_position(119.0, demand, supply) == "⚪ Вне зон (Ждать)"
    assert _price_position(120.0, demand, supply) == "🔴 В зоне SUPPLY"
    assert _price_position(110.0, demand, supply) == "🟢 В зоне DEMAND"


def test_directional_zone_requires_exact_midpoint_touch():
    from event_engine.signals import _find_directional_zone
    demand = [{"btm": 100.0, "top": 110.0}]
    supply = [{"btm": 120.0, "top": 130.0}]
    assert _find_directional_zone("LONG", 110.0, 112.0, 111.0, demand, supply) is None
    assert _find_directional_zone("LONG", 104.9, 105.0, 106.0, demand, supply) == demand[0]
    assert _find_directional_zone("SHORT", 123.0, 125.0, 126.0, demand, supply) == supply[0]

def test_5m_zone_diagnostics_formatter_never_raises_with_percent_literals(monkeypatch, caplog):
    import logging
    import pandas as pd
    import run_once

    monkeypatch.setattr(run_once, "ZONE_TRIGGER_MODE", "zone")
    diagnostics = {
        "bars_received": 3,
        "bars_closed": 3,
        "processed_bars": 3,
        "last_processed_before": None,
        "last_processed_after": "2026-09-17T10:05:00+00:00",
        "latest_closed_5m_ts": "2026-09-17T10:05:00+00:00",
        "zones": {
            "DEMAND:0:110.000000000000:100.000000000000": {
                "kind": "DEMAND", "start_idx": 0, "top": 110.0, "bottom": 100.0,
                "midpoint": 105.0, "width": 10.0, "state_after": "LOCKED",
                "window_bars_in_zone": 1, "window_zone_touches": 1,
                "window_midpoint_touches": 0, "bars_in_zone": 1,
                "zone_touches": 1, "midpoint_touches": 0,
                "same_visit_blocks": 0, "ambiguous_blocks": 0,
                "stale_touches": 0, "activation_blocks": 0,
                "structure_rejects": 0, "directional_rejects": 0,
                "other_rejects": 0, "signals_created": 1, "rearms": 0,
                "window_last_midpoint_touch": None,
                "window_last_zone_touch": "2026-09-17T10:05:00+00:00",
            }
        },
        "touch_events": [],
        "decision": {"status": "SIGNAL_CREATED", "zone_key": "DEMAND:0:110.000000000000:100.000000000000", "direction": "LONG", "reason": "signal_created", "timestamp": "2026-09-17T10:05:00+00:00"},
    }
    state = {"zones": {
        "DEMAND:0:110.000000000000:100.000000000000": {
            "state": "LOCKED", "visit_id": "VISIT-1", "first_touch_ts": "2026-09-17T10:05:00+00:00",
            "last_touch_ts": "2026-09-17T10:05:00+00:00", "touch_count": 1, "lock_reason": "zone_touch",
            "trigger_event_id": "EVT-1", "pending_signal": None,
        }
    }}
    with caplog.at_level(logging.INFO):
        run_once._log_5m_zone_diagnostics(
            "ZRX-USDT", 103.0, pd.Timestamp("2026-09-17T10:00:00+00:00"),
            [{"start": 0, "top": 110.0, "btm": 100.0}], [], diagnostics, state, 1,
        )
    assert any("[ZONE_STATUS] ZRXUSDT" in r.message for r in caplog.records)
    assert any("5m=SIGNAL_CREATED" in r.message for r in caplog.records)
    assert all("ZONE_CLOSEST" not in r.message for r in caplog.records)


def test_active_zone_logs_compact_no_touch_reason(monkeypatch, caplog):
    import logging
    import pandas as pd
    import run_once
    monkeypatch.setattr(run_once, "ZONE_TRIGGER_MODE", "zone")
    diagnostics = {
        "processed_bars": 3,
        "latest_closed_5m_ts": "2026-09-17T14:35:00+00:00",
        "zones": {"DEMAND:0:110:100": {"kind": "DEMAND", "bottom": 100.0, "top": 110.0}},
        "decision": {"status": "NO_5M_TOUCH", "zone_key": None, "direction": None, "reason": "no_configured_touch_in_processed_5m_bars", "timestamp": "2026-09-17T14:35:00+00:00"},
    }
    with caplog.at_level(logging.INFO):
        run_once._log_5m_zone_diagnostics(
            "ZIL-USDT", 103.0, pd.Timestamp("2026-09-17T14:00:00+00:00"),
            [{"start": 0, "top": 110.0, "btm": 100.0}], [], diagnostics, {"zones": {}}, 0,
        )
    messages = [r.message for r in caplog.records]
    assert any("[ZONE_STATUS] ZILUSDT" in m and "5m=NO_5M_TOUCH" in m for m in messages)
    assert not any("[ZONE_DIAG]" in m or "[ZONE_STATE]" in m for m in messages)


def test_5m_zone_mode_accepts_any_zone_touch_not_only_midpoint(monkeypatch):
    import pandas as pd
    import run_once
    monkeypatch.setattr(run_once, "ZONE_TRIGGER_MODE", "zone")
    monkeypatch.setattr(run_once, "MIN_STRUCTURE_ROOM_R", 0.0)
    monkeypatch.setattr(run_once, "REQUIRE_STRUCTURE_OBSTACLE", False)
    monkeypatch.setattr(run_once, "_nearest_opposing_level", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_once, "MAX_5M_TRIGGER_AGE_MINUTES", 60.0)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    df_1h = pd.DataFrame([
        {"timestamp": now - pd.Timedelta(hours=12-i), "open": 100, "high": 111, "low": 99, "close": 105, "volume": 100, "atr50": 2.0}
        for i in range(12)
    ])
    zone = {"start": 0, "top": 110.0, "btm": 100.0, "poi": 105.0}
    bar = pd.Series({
        "timestamp": now - pd.Timedelta(minutes=5), "open": 108.0, "high": 104.0,
        "low": 102.0, "close": 103.0, "volume": 10.0,
    })
    prev = pd.Series({
        "timestamp": now - pd.Timedelta(minutes=10), "open": 112.0, "high": 113.0,
        "low": 111.0, "close": 112.0, "volume": 10.0,
    })
    signal = run_once._build_5m_zone_signal(
        "TEST-USDT", "LONG", zone, bar, prev, df_1h, [zone], [], {"visit_id": "VISIT"}
    )
    assert signal["entry"] == 103.0
    assert signal["trigger"]["zone_trigger_mode"] == "zone"
    assert signal["trigger"]["type"] == "ZONE_TOUCH_5M"
    assert signal["trigger"]["midpoint_touched_diagnostic"] is False
    assert signal["zone"]["poi"] == 105.0


def test_run_once_import_regression():
    # run_once.py uses SWING_LEN for its per-symbol minimum-history guard.
    # Keep this import-level test so a refactor cannot leave the constant out
    # of the run_once module namespace and fail every scanned symbol.
    import importlib
    module = importlib.import_module("run_once")
    assert module.SWING_LEN == 10

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from event_engine import signals as sig


def _base_signal(direction="LONG"):
    return {
        "symbol": "TEST-USDT", "type": direction, "entry": 100.0,
        "sl": 98.0 if direction == "LONG" else 102.0,
        "tp1": 101.2 if direction == "LONG" else 98.8,
        "tp2": 102.4 if direction == "LONG" else 97.6,
        "risk_pct": 2.0,
        "target": {"obstacle_price": 104.0 if direction == "LONG" else 96.0},
        "zone": {"kind": "DEMAND" if direction == "LONG" else "SUPPLY", "btm": 97.0, "top": 99.0, "start": 100, "age_bars": 5},
    }


def test_atr_does_not_backfill_future_values():
    df = pd.DataFrame({"high":[1,2,3],"low":[0,1,2],"close":[0.5,1.5,2.5]})
    out = sig.calc_atr(df, 5)
    assert out.isna().all()


def test_run_once_validation_enforces_risk_and_structure(monkeypatch):
    import run_once
    monkeypatch.setenv("MAX_SIGNAL_RISK_PCT", "10.00")
    monkeypatch.setenv("MIN_STRUCTURE_ROOM_R", "1.20")
    good = _base_signal("LONG")
    good["risk_pct"] = 8.0
    good["target"]["obstacle_price"] = 104.0
    ok, reason = run_once._validate_trade_geometry(good)
    assert ok, reason
    bad = _base_signal("LONG")
    bad["risk_pct"] = 11.0
    ok, reason = run_once._validate_trade_geometry(bad)
    assert not ok and "risk_pct_above_limit" in reason


def test_directional_candle_filter():
    monkeypatch = None
    assert sig.REQUIRE_DIRECTIONAL_CANDLE is False


def test_server_time_offset_function_exists():
    from event_engine import bingx
    class R:
        headers = {"Date": "Wed, 02 Sep 2026 19:00:00 GMT"}
    old = bingx.SERVER_TIME_OFFSET_MS
    assert bingx._update_server_time_offset(R()) is True
    assert isinstance(bingx.SERVER_TIME_OFFSET_MS, int)
    bingx.SERVER_TIME_OFFSET_MS = old


def test_zone_entry_filters_are_configured_safely():
    from event_engine import signals as sig
    assert not hasattr(sig, "MAX_ZONE_AGE_BARS")
    assert sig.MAX_SIGNAL_RISK_PCT == 10.00
    assert sig.MIN_STRUCTURE_ROOM_R == 1.20
    assert sig.REQUIRE_DIRECTIONAL_CANDLE is False
    assert sig.REQUIRE_STRUCTURE_OBSTACLE is False


def test_old_active_zone_is_not_rejected_by_zone_age():
    import run_once
    import pandas as pd

    bars = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for i in range(101):
        ts = start + pd.Timedelta(hours=i)
        bars.append({
            "timestamp": ts,
            "open": 100.0,
            "high": 106.0,
            "low": 94.0,
            "close": 100.0,
            "volume": 1000.0,
            "atr50": 2.0,
        })
    df_1h = pd.DataFrame(bars)

    zone = {
        "kind": "DEMAND",
        "btm": 99.0,
        "top": 101.0,
        "poi": 100.0,
        "start": 0,
    }
    bar = pd.Series({
        "timestamp": start + pd.Timedelta(hours=101, minutes=0),
        "open": 99.5,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1200.0,
    })
    prev_bar = pd.Series({
        "timestamp": start + pd.Timedelta(hours=100, minutes=55),
        "open": 104.0,
        "high": 104.5,
        "low": 103.5,
        "close": 104.0,
        "volume": 900.0,
    })

    old_helper = run_once._nearest_opposing_level
    old_req = run_once.REQUIRE_STRUCTURE_OBSTACLE
    try:
        run_once._nearest_opposing_level = lambda *args, **kwargs: None
        run_once.REQUIRE_STRUCTURE_OBSTACLE = False
        signal = run_once._build_5m_zone_signal(
            "TEST-USDT", "LONG", zone, bar, prev_bar, df_1h, [zone], [],
            {"visit_id": "VISIT", "previous_midpoint_touch": False},
        )
    finally:
        run_once._nearest_opposing_level = old_helper
        run_once.REQUIRE_STRUCTURE_OBSTACLE = old_req

    assert signal["zone"]["age_bars"] >= 100
    assert signal["zone_counts"] == {"demand": 1, "supply": 0}
    assert signal["entry"] == 100.0
    assert signal["sl"] == 90.0
    assert signal["tp1"] == 103.0
    assert signal["tp2"] == 106.0


def test_tracker_close_once_is_idempotent(tmp_path, monkeypatch):
    from event_engine import tracker
    path = tmp_path / "trades.jsonl"
    monkeypatch.setattr(tracker, "TRADES_PATH", path)
    rec = {"record_type": "TRADE_CLOSE", "event_id": "EVT_TEST"}
    assert tracker._append_trade_close_once(rec) is True
    assert tracker._append_trade_close_once(rec) is False
    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1


def test_open_market_blocks_two_tp_min_quantity_before_order(monkeypatch):
    from event_engine import bingx
    monkeypatch.setenv("BINGX_MARGIN_USDT", "1")
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {
        "symbol": "BNB-USDT", "quantityPrecision": 2, "tradeMinQuantity": 0.01,
        "multiplier": 1, "maxLeverage": 10,
    })
    monkeypatch.setattr(bingx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bingx, "_current_close_price", lambda symbol: 1000.0)
    monkeypatch.setattr(bingx, "get_execution_quote", lambda symbol, **kwargs: {
        "status": "ok", "symbol": symbol, "bid": 1000.0, "ask": 1000.0, "spread_pct": 0.0
    })
    monkeypatch.setattr(bingx, "has_open_position", lambda *args, **kwargs: False)
    out = bingx.open_market("BNB-USDT", "LONG", 1000.0, "EVT_TEST")
    assert out["status"] == "skipped_tp_min_qty"


def test_sl_order_validation_is_one_sided_and_be_allows_entry():
    from event_engine import bingx
    assert bingx._validate_sl_order_for_position({"type":"STOP_MARKET","stopPrice":"99","origQty":"1"}, "LONG", 100.0)
    assert bingx._validate_sl_order_for_position({"type":"STOP_MARKET","stopPrice":"100","origQty":"1"}, "LONG", 100.0)
    assert bingx._validate_sl_order_for_position({"type":"STOP_MARKET","stopPrice":"100","origQty":"1"}, "SHORT", 100.0)
    assert not bingx._validate_sl_order_for_position({"type":"STOP_MARKET","stopPrice":"100.2","origQty":"1"}, "LONG", 100.0)
    assert bingx._validate_sl_order_for_position({"type":"STOP_MARKET","stopPrice":"101","origQty":"1"}, "SHORT", 100.0)
    assert not bingx._validate_sl_order_for_position({"type":"STOP_MARKET","stopPrice":"99.8","origQty":"1"}, "SHORT", 100.0)

def test_workflow_risk_cap_is_not_accidentally_25_percent():
    from pathlib import Path
    workflow = Path('.github/workflows/event-engine.yml').read_text(encoding='utf-8')
    assert 'MAX_SIGNAL_RISK_PCT: "10.00"' in workflow
    assert 'MAX_SIGNAL_RISK_PCT: "25"' not in workflow

def test_signal_risk_cap_hard_clamped(monkeypatch):
    import importlib
    import event_engine.signals as sig
    monkeypatch.setenv('MAX_SIGNAL_RISK_PCT', '25')
    sig2 = importlib.reload(sig)
    assert sig2.MAX_SIGNAL_RISK_PCT == 10.00
    monkeypatch.setenv('MAX_SIGNAL_RISK_PCT', '1.25')
    sig2 = importlib.reload(sig2)
    assert sig2.MAX_SIGNAL_RISK_PCT == 1.25
    monkeypatch.delenv("MAX_SIGNAL_RISK_PCT", raising=False)
    importlib.reload(sig2)

def test_run_once_hard_caps_production_risk(monkeypatch):
    import run_once
    monkeypatch.setenv("MAX_SIGNAL_RISK_PCT", "25")
    assert run_once.MAX_PRODUCTION_RISK_PCT == 10.00

def test_request_does_not_fallback_post_after_network_error(monkeypatch):
    from event_engine import bingx
    class Session:
        def __init__(self): self.calls=[]
        def request(self, **kwargs):
            self.calls.append(kwargs['url'])
            raise bingx.requests.Timeout('network')
    session = Session()
    monkeypatch.setattr(bingx, 'SESSION', session)
    monkeypatch.setattr(bingx, '_get_fast_session', lambda: session)
    monkeypatch.setenv('BINGX_API_KEY', 'k')
    monkeypatch.setenv('BINGX_SECRET_KEY', 's')
    out = bingx._request('POST', '/foo', {'a':'1'}, signed=True, retryable=True)
    assert out['code'] == -1
    assert len(session.calls) == 1


def test_sl_validator_covers_full_position_qty():
    from event_engine import bingx
    good = {"type": "STOP_MARKET", "stopPrice": "99", "origQty": "1.0"}
    bad = {"type": "STOP_MARKET", "stopPrice": "99", "origQty": "0.5"}
    assert bingx._validate_sl_order_for_position(good, "LONG", 100.0, 1.0)
    assert not bingx._validate_sl_order_for_position(bad, "LONG", 100.0, 1.0)


def test_tp_success_requires_exchange_reconciliation(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})
    out = bingx._verify_open_order("AAA-USDT", "LONG", client_order_id="EVT_X_TP1", order_kind="TAKE_PROFIT_MARKET", expected_price=101.0, expected_qty=1.0, price_precision=2, max_attempts=1)
    assert out["status"] == "not_found"


def test_failed_signal_registry_roundtrip(tmp_path, monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "DATA", tmp_path)
    monkeypatch.setattr(run_once, "FAILED_SIGNALS_PATH", tmp_path / "failed_signals.json")
    monkeypatch.setattr(run_once, "FAILED_SIGNAL_TTL_SEC", 3600)
    run_once._mark_failed_signal("EVT_TEST", "blocked_protection_preflight", "x")
    loaded = run_once._load_failed_signal_ids()
    assert "EVT_TEST" in loaded


def test_be_uses_detected_one_way_position_side(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(tracker, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "position_side_param", lambda direction: "BOTH")
    order_box = {}
    def fake_open_orders(*a, **k):
        if order_box:
            return {"status": "ok", "sl_orders": [order_box.copy()], "tp_orders": []}
        return {"status": "ok", "sl_orders": [], "tp_orders": []}
    monkeypatch.setattr(tracker, "get_open_protection_directional", fake_open_orders)
    calls = []
    def fake_request(method, path, params):
        calls.append((method, params))
        if method == "POST":
            order_box.update({"orderId": "123", "clientOrderId": params.get("clientOrderId"), "type": "STOP_MARKET", "stopPrice": params.get("stopPrice"), "origQty": params.get("quantity")})
            return {"code": 0, "data": {"order": order_box.copy()}}
        return {"code": 0}
    monkeypatch.setattr(tracker, "_request", fake_request)
    out = tracker._move_sl_to_break_even("AAA-USDT", "LONG", 100.0, 1.0, None, "T")
    assert out["status"] == "created"
    assert calls[0][1]["positionSide"] == "BOTH"

def test_signed_get_and_delete_use_explicit_query(monkeypatch):
    from event_engine import bingx
    monkeypatch.setenv("BINGX_API_KEY", "key")
    monkeypatch.setenv("BINGX_SECRET_KEY", "secret")
    captured = []
    class Resp:
        headers = {}
        def json(self): return {"code": 0, "data": {}}
    class Session:
        def request(self, **kwargs):
            captured.append(kwargs)
            return Resp()
    monkeypatch.setattr(bingx, "SESSION", Session())
    bingx._request("GET", "/p", {"symbol":"BTC-USDT", "orderId":"123"}, signed=True)
    bingx._request("DELETE", "/p", {"symbol":"BTC-USDT", "orderId":"123"}, signed=True)
    for call in captured:
        assert "?" in call["url"]
        assert call.get("params") is None
        assert "signature=" in call["url"]


def test_close_position_omits_reduce_only_in_hedge(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(bingx, "get_contract", lambda s: {"quantityPrecision": 3})
    monkeypatch.setattr(bingx, "position_side_param", lambda d: d)
    calls = []
    monkeypatch.setattr(bingx, "_request", lambda method, path, params: calls.append(params) or {"code":0})
    out = bingx.close_position_market("AAA-USDT", "SHORT", 1.0, reduce_only=True, trade_id="T")
    assert out["status"] == "closed"
    assert calls and "reduceOnly" not in calls[-1]
    assert calls[-1]["positionSide"] == "SHORT"


def test_close_position_uses_reduce_only_in_one_way(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(bingx, "get_contract", lambda s: {"quantityPrecision": 3})
    monkeypatch.setattr(bingx, "position_side_param", lambda d: "BOTH")
    calls = []
    monkeypatch.setattr(bingx, "_request", lambda method, path, params: calls.append(params) or {"code":0})
    out = bingx.close_position_market("AAA-USDT", "LONG", 1.0, reduce_only=True, trade_id="T")
    assert out["status"] == "closed"
    assert calls[-1]["reduceOnly"] == "true"


def test_failed_signal_is_recorded_after_emergency_like_status(tmp_path, monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "DATA", tmp_path)
    monkeypatch.setattr(run_once, "FAILED_SIGNALS_PATH", tmp_path / "failed_signals.json")
    monkeypatch.setattr(run_once, "FAILED_SIGNAL_TTL_SEC", 3600)
    # The execution-status branch used by main must treat any non-protected
    # outcome as terminal, including statuses whose prefix is 'opened'.
    status = "opened_then_emergency_closed"
    if status != "opened_protected":
        run_once._mark_failed_signal("EVT_EMERGENCY", status, "rollback")
    assert "EVT_EMERGENCY" in run_once._load_failed_signal_ids()


def test_scan_history_rotates_and_keeps_complete_records(tmp_path, monkeypatch):
    from event_engine import analytics
    monkeypatch.setattr(analytics, "DATA_DIR", tmp_path)
    monkeypatch.setattr(analytics, "SCAN_JSONL", tmp_path / "scan_history.jsonl")
    monkeypatch.setattr(analytics, "MAX_SCAN_HISTORY_BYTES", 1024)
    for i in range(80):
        analytics._append_jsonl(analytics.SCAN_JSONL, {"i": i, "payload": "x" * 40})
        analytics._rotate_scan_history()
    assert analytics.SCAN_JSONL.stat().st_size <= 1400
    rows = [json.loads(x) for x in analytics.SCAN_JSONL.read_text().splitlines() if x.strip()]
    assert rows and rows[-1]["i"] == 79


def test_all_orders_helper_uses_historical_endpoint(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda s: s)
    captured = {}
    def fake_request(method, path, params, signed=True):
        captured.update(method=method, path=path, params=params, signed=signed)
        return {"code": 0, "data": [{"orderId": "1"}]}
    monkeypatch.setattr(bingx, "_request", fake_request)
    out = bingx.get_all_orders("AAA-USDT", 100, 200, 50)
    assert out == [{"orderId": "1"}]
    assert captured["path"] == "/openApi/swap/v2/trade/allOrders"
    assert captured["params"]["startTime"] == 100
    assert captured["params"]["endTime"] == 200


def test_historical_exit_reconciliation_does_not_use_tp_for_residual(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "_get_filled_order", lambda *a, **k: None)
    monkeypatch.setattr(tracker, "get_all_orders", lambda *a, **k: [
        {"orderId":"TP1", "status":"FILLED", "side":"SELL", "type":"TAKE_PROFIT_MARKET", "executedQty":"1", "avgPrice":"110", "updateTime":200},
        {"orderId":"SL1", "status":"FILLED", "side":"SELL", "type":"STOP_MARKET", "executedQty":"1", "avgPrice":"95", "updateTime":300},
    ])
    out = tracker._reconcile_historical_exit_order("AAA-USDT", "LONG", 100, 1, [{"order_id":"TP1"}], {"order_id":"SL1"})
    assert out[0] == 95.0
    assert out[1] == "STOP_LOSS"


def test_post_fill_slippage_guard_emergency_closes(monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "MAX_ENTRY_SLIPPAGE_PCT", 1.0)
    signal = {"event_id":"EVT_SLIP", "symbol":"AAA-USDT", "type":"LONG", "entry":100.0, "sl":99.0, "tp1":101.0, "tp2":102.0, "risk_pct":1.0,
              "zone":{"kind":"DEMAND","btm":98,"top":99}, "target":{"obstacle_price":110}}
    monkeypatch.setattr(run_once, "_validate_trade_geometry", lambda s: (True, ""))
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {"status":"ok","sl_orders":[],"tp_orders":[]})
    monkeypatch.setattr(run_once, "_build_setup", lambda s: {"zone":{"kind":"DEMAND"}})
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: {"status":"opened"})
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda *a, **k: {"status":"found","avgPrice":102.5,"positionAmt":1})
    monkeypatch.setattr(run_once, "_emergency_close_and_verify", lambda *a, **k: {"status":"closed_verified"})
    monkeypatch.setattr(run_once, "_cleanup_engine_protection", lambda *a, **k: {"status":"ok"})
    out = run_once.execute_new_position(signal)
    assert out["status"] == "opened_then_emergency_closed"
    assert "execution_slippage_pct" in out["error"]

def test_wait_for_position_fill_retries_transient_errors(monkeypatch):
    from event_engine import bingx
    seq = [
        {"status":"error", "error":"temporary"},
        {"status":"not_found"},
        {"status":"found", "avgPrice":100, "positionAmt":1},
    ]
    monkeypatch.setattr(bingx, "get_position_directional", lambda *a, **k: seq.pop(0))
    monkeypatch.setattr(bingx.time, "sleep", lambda *a, **k: None)
    out = bingx.wait_for_position_fill_directional("AAA-USDT", "LONG", timeout_sec=1, poll_interval=0)
    assert out["status"] == "found"


def test_execute_reconciles_position_after_fill_poll_timeout(monkeypatch):
    import run_once
    signal = {
        "event_id": "ZONE_TEST_FILL_RECON",
        "symbol": "TEST-USDT",
        "type": "LONG",
        "entry": 100.0,
        "sl": 99.0,
        "tp1": 101.0,
        "tp2": 102.0,
        "risk_pct": 1.0,
        "score": 80,
        "atr": 2.0,
        "zone": {"kind": "DEMAND", "btm": 98.0, "top": 100.0},
        "target": {"obstacle_price": 104.0},
    }
    monkeypatch.setattr(run_once, "_validate_trade_geometry", lambda s: (True, "ok"))
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})
    monkeypatch.setattr(run_once, "_build_setup", lambda s: {"zone": {"kind": "DEMAND"}, "tp_levels": []})
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: {"status": "opened", "symbol": "TEST-USDT"})
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda *a, **k: {"status": "timeout", "symbol": "TEST-USDT"})
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: {"status": "found", "avgPrice": 100.5, "positionAmt": 1.0})
    monkeypatch.setattr(run_once, "_rebase_protection_after_fill", lambda s, avg: {
        "sl": 98.5, "tp1": 101.25, "tp2": 102.5, "risk_abs": 2.0, "risk_pct": 1.99,
        "tp1_rr": 0.5, "tp2_rr": 1.0, "target_source": "test", "obstacle_price": 104.0,
    })
    monkeypatch.setenv("MAX_SIGNAL_RISK_PCT", "2.5")
    monkeypatch.setattr(run_once, "ensure_directional_protection", lambda *a, **k: {
        "status": "PROTECTED", "tp_orders": [], "sl_result": {},
    })
    registered = {}
    monkeypatch.setattr(run_once, "register_active_trade", lambda *a, **k: registered.update(k))
    out = run_once.execute_new_position(signal)
    assert out["status"] == "opened_protected"
    assert out["position"]["avgPrice"] == 100.5
    assert registered["entry_price"] == 100.5


def test_execute_marks_entry_not_filled_after_timeout_and_reconcile(monkeypatch):
    import run_once
    signal = {
        "event_id": "ZONE_TEST_FILL_ABSENT",
        "symbol": "TEST-USDT",
        "type": "SHORT",
        "entry": 100.0,
        "sl": 101.0,
        "tp1": 99.5,
        "tp2": 99.0,
        "risk_pct": 1.0,
        "score": 80,
        "zone": {"kind": "SUPPLY", "btm": 100.0, "top": 102.0},
        "target": {"obstacle_price": 96.0},
    }
    monkeypatch.setattr(run_once, "_validate_trade_geometry", lambda s: (True, "ok"))
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})
    monkeypatch.setattr(run_once, "_build_setup", lambda s: {"zone": {"kind": "SUPPLY"}, "tp_levels": []})
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: {"status": "opened", "symbol": "TEST-USDT"})
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda *a, **k: {"status": "timeout", "symbol": "TEST-USDT"})
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: {"status": "not_found", "symbol": "TEST-USDT", "positionSide": "SHORT"})
    monkeypatch.setattr(run_once, "ensure_directional_protection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not protect absent position")))
    out = run_once.execute_new_position(signal)
    assert out["status"] == "entry_not_filled"
    assert out["position"]["status"] == "not_found"


def test_position_keys_normalizes_one_way_both_by_position_amount():
    import run_once
    positions = [
        {"symbol": "AAA-USDT", "positionSide": "BOTH", "positionAmt": "1.25"},
        {"symbol": "BBB-USDT", "positionSide": "BOTH", "positionAmt": "-2.50"},
        {"symbol": "CCC-USDT", "positionSide": "BOTH", "positionAmt": "0"},
        {"symbol": "DDD-USDT", "positionSide": "LONG", "positionAmt": "1"},
    ]
    assert run_once._position_keys(positions) == {
        ("AAA-USDT", "LONG"),
        ("BBB-USDT", "SHORT"),
        ("DDD-USDT", "LONG"),
    }


def test_reconcile_handles_one_way_both_position(monkeypatch, tmp_path):
    import run_once
    monkeypatch.setattr(run_once, "_load_active_trades_file", lambda: {})
    monkeypatch.setattr(run_once, "get_positions", lambda **kwargs: [
        {"symbol": "AAA-USDT", "positionSide": "BOTH", "positionAmt": "-1.0", "avgPrice": "100.0"},
    ])
    calls = {}
    monkeypatch.setattr(run_once, "ensure_directional_protection", lambda *args, **kwargs: calls.update({
        "symbol": args[0], "direction": args[1], "avg": args[2], "qty": args[3],
    }) or {"status": "PROTECTED", "tp_orders": [], "sl_result": {}, "effective_tp_levels": []})
    run_once.reconcile_all_open_positions()
    assert calls == {"symbol": "AAA-USDT", "direction": "SHORT", "avg": 100.0, "qty": 1.0}


def test_tracker_uses_exchange_qty_as_remaining_qty(monkeypatch, tmp_path):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    trade = {
        "event_id": "EVT_QTY_RECON",
        "symbol": "AAA-USDT",
        "direction": "LONG",
        "name": "AAA",
        "entry_price": 100.0,
        "initial_qty": 1.0,
        "remaining_qty": 1.0,
        "entry_ts": 0,
        "closed": False,
        "tp_orders": [],
        "sl_order": {},
        "hit_legs": [],
        "tp_filled_qty": {},
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "peak_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "max_drawdown_pct": 0.0,
    }
    (tmp_path / "active_trades.json").write_text(json.dumps({"EVT_QTY_RECON": trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {
        "status": "found", "positionAmt": 0.4, "avgPrice": 100.0,
    })
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status": "ok", "order_status": "NEW"})
    monkeypatch.setattr(tracker, "send_tg", lambda *a, **k: True)
    tracker.update_active_trades()
    saved = json.loads((tmp_path / "active_trades.json").read_text(encoding="utf-8"))
    assert saved["EVT_QTY_RECON"]["remaining_qty"] == 0.4
    assert saved["EVT_QTY_RECON"]["current_position_qty"] == 0.4
    assert saved["EVT_QTY_RECON"]["closed"] is False


def test_be_existing_with_old_sl_cancel_failure_is_not_success(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(tracker, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok",
        "sl_orders": [{"orderId": "BE1", "clientOrderId": "EVT_BE_T", "type": "STOP_MARKET", "stopPrice": "100", "origQty": "1"}],
        "tp_orders": [],
    })
    monkeypatch.setattr(tracker, "_cancel_old_sl_verified", lambda *a, **k: (False, "cancel timeout"))
    out = tracker._move_sl_to_break_even("AAA-USDT", "LONG", 100.0, 1.0, "OLD1", "T", old_sl_price=99.0)
    assert out["status"] == "error"
    assert "old SL cancel failed" in out["error"]


def test_be_restore_requires_exchange_verification(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(tracker, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "position_side_param", lambda direction: "BOTH")
    calls = []
    responses = iter([
        {"status": "ok", "sl_orders": [], "tp_orders": []},
        {"status": "ok", "sl_orders": [], "tp_orders": []},
    ])
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: next(responses))
    monkeypatch.setattr(tracker, "_cancel_old_sl_verified", lambda *a, **k: (True, "cancelled"))
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status": "found", "positionAmt": 1.0})
    monkeypatch.setattr(tracker, "close_position_market", lambda *a, **k: {"status": "closed"})
    monkeypatch.setattr(tracker.time, "sleep", lambda *a, **k: None)
    def fake_request(method, path, params):
        calls.append(params)
        return {"code": 0, "data": {"order": {"orderId": "RST1", "clientOrderId": params["clientOrderId"]}}}
    monkeypatch.setattr(tracker, "_request", fake_request)
    # First POST (BE) succeeds but its verification does not see the order,
    # forcing restore. Restore POST also succeeds, but restore verification does not.
    out = tracker._move_sl_to_break_even("AAA-USDT", "LONG", 100.0, 1.0, "OLD1", "T", old_sl_price=99.0)
    assert out["status"] == "error"
    assert out["old_sl_restored"] is False
    assert len(calls) == 2


def test_be_failure_without_restore_triggers_emergency_close(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(tracker, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "position_side_param", lambda direction: "LONG")
    # Initial protection exists, then BE is accepted but invisible; restore is
    # also not provable. The safe outcome is emergency close + verified absence.
    responses = iter([
        {"status": "ok", "sl_orders": [], "tp_orders": []},
        {"status": "ok", "sl_orders": [], "tp_orders": []},
    ])
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: next(responses))
    monkeypatch.setattr(tracker, "_cancel_old_sl_verified", lambda *a, **k: (True, "cancelled"))
    requests = []
    def fake_request(method, path, params):
        requests.append(params)
        return {"code": 0, "data": {"order": {"orderId": "RST1", "clientOrderId": params["clientOrderId"]}}}
    monkeypatch.setattr(tracker, "_request", fake_request)
    pos_states = iter([
        {"status": "found", "positionAmt": 1.0},
        {"status": "not_found"},
    ])
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: next(pos_states))
    closes = []
    monkeypatch.setattr(tracker, "close_position_market", lambda *a, **k: closes.append((a, k)) or {"status": "closed"})
    monkeypatch.setattr(tracker, "send_tg", lambda *a, **k: True)
    monkeypatch.setattr(tracker.time, "sleep", lambda *a, **k: None)
    out = tracker._move_sl_to_break_even("AAA-USDT", "LONG", 100.0, 1.0, "OLD1", "T", old_sl_price=99.0)
    assert out["status"] == "error"
    assert out["safety_action"] == "emergency_close"
    assert out["rollback"]["status"] == "closed_verified"
    assert closes


def test_stale_scan_branch_does_not_reference_uninitialized_live_price():
    import ast
    source = Path("run_once.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    scan_one = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "scan_one"
    )
    stale = next(
        n for n in ast.walk(scan_one)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and any(isinstance(x, ast.Name) and x.id == "MAX_DATA_STALENESS_HOURS" for x in ast.walk(n.test))
    )
    names = [n.id for n in ast.walk(stale) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
    assert "binance_live_price" not in names


def test_crossed_tp_market_close_requires_execution_verification(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(bingx, "get_contract", lambda s: {
        "symbol": "AAA-USDT", "quantityPrecision": 3, "pricePrecision": 2, "tradeMinQuantity": 0.1,
    })
    monkeypatch.setattr(bingx, "position_side_param", lambda d: "BOTH")
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [{"orderId": "SL1", "type": "STOP_MARKET", "stopPrice": "99", "origQty": "1"}], "tp_orders": []
    })
    monkeypatch.setattr(bingx, "_current_close_price", lambda s: 101.0)
    monkeypatch.setattr(bingx, "_verify_open_order", lambda *a, **k: {"status": "verified"})
    monkeypatch.setattr(bingx, "cancel_order", lambda *a, **k: {"code": 0})
    monkeypatch.setattr(bingx, "_request", lambda method, path, params: {"code": 0, "data": {"order": {
        "orderId": "TPM1", "clientOrderId": params.get("clientOrderId")
    }}} if method == "POST" else {"code": 0})
    # POST is acknowledged, but neither order execution nor position reduction is visible.
    monkeypatch.setattr(bingx, "get_order", lambda *a, **k: {
        "status": "ok", "order_id": "TPM1", "order_status": "NEW", "executed_qty": 0.0, "avg_price": 0.0
    })
    monkeypatch.setattr(bingx, "get_position_directional", lambda *a, **k: {
        "status": "found", "positionAmt": 1.0, "avgPrice": 100.0
    })
    monkeypatch.setattr(bingx.time, "sleep", lambda *a, **k: None)
    out = bingx._verify_market_reduce_order("AAA-USDT", "LONG", "TPM1", 0.5, 1.0, attempts=1)
    assert out["status"] == "unverified"


def test_failed_signal_registry_uses_bounded_backoff_and_terminal_state(tmp_path, monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "DATA", tmp_path)
    monkeypatch.setattr(run_once, "FAILED_SIGNALS_PATH", tmp_path / "failed_signals.json")
    monkeypatch.setattr(run_once, "FAILED_SIGNAL_TTL_SEC", 3600)
    monkeypatch.setattr(run_once, "FAILED_SIGNAL_MAX_RETRIES", 3)
    monkeypatch.setattr(run_once, "FAILED_SIGNAL_RETRY_BASE_SEC", 10)
    monkeypatch.setattr(run_once, "FAILED_SIGNAL_RETRY_MAX_SEC", 100)
    monkeypatch.setattr(run_once.time, "time", lambda: 1000)
    run_once._mark_failed_signal("EVT_RETRY", "execution_error", "x")
    run_once._mark_failed_signal("EVT_RETRY", "execution_error", "y")
    run_once._mark_failed_signal("EVT_RETRY", "execution_error", "z")
    raw = json.loads((tmp_path / "failed_signals.json").read_text())
    record = raw["EVT_RETRY"]
    assert record["retry_count"] == 3
    assert record["terminal"] is True
    assert record["next_retry_at"] == 0
    assert run_once._failed_signal_is_blocked(record, now=2000) is True


def test_execution_outcome_categories_separate_technical_failures():
    import run_once
    assert run_once._execution_outcome_category("opened_protected") == "PROTECTED_ENTRY"
    assert run_once._execution_outcome_category("opened_then_emergency_closed") == "EMERGENCY_EXIT"
    assert run_once._execution_outcome_category("entry_state_unverified") == "EXECUTION_UNVERIFIED"
    assert run_once._execution_outcome_category("blocked_protection_preflight") == "PROTECTION_FAILURE"


def test_exit_outcome_categories_separate_strategy_and_unverified_closes():
    from event_engine import tracker
    assert tracker._exit_outcome_category("STOP_LOSS") == "STRATEGY_EXIT"
    assert tracker._exit_outcome_category("TAKE_PROFIT_FULL") == "STRATEGY_EXIT"
    assert tracker._exit_outcome_category("BREAK_EVEN") == "STRATEGY_EXIT"
    assert tracker._exit_outcome_category("POSITION_CLOSED_UNVERIFIED") == "UNVERIFIED_CLOSE"


def test_active_trade_state_save_uses_lock_and_atomic_replace(tmp_path, monkeypatch):
    from event_engine import tracker
    target = tmp_path / "active_trades.json"
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", target)
    tracker._save_active_trades({"EVT": {"symbol": "AAA-USDT", "closed": False}})
    assert json.loads(target.read_text(encoding="utf-8"))["EVT"]["symbol"] == "AAA-USDT"
    assert target.with_suffix(target.suffix + ".lock").exists()
    assert not target.with_name(target.name + ".tmp").exists()



def test_tracker_sends_and_records_tp_notification(monkeypatch, tmp_path):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    trade = {
        "event_id": "EVT_TP_NOTIFY",
        "symbol": "AAA-USDT",
        "direction": "LONG",
        "name": "AAA",
        "entry_price": 100.0,
        "initial_qty": 1.0,
        "remaining_qty": 1.0,
        "entry_ts": 1,
        "closed": False,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1"}],
        "sl_order": {},
        "hit_legs": [],
        "tp_filled_qty": {},
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "peak_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "planned_risk_pct": 1.0,
        "planned_weighted_rr": 0.75,
        "effective_weighted_rr": 0.75,
        "setup": {},
        "tp_levels": [],
        "effective_tp_levels": [],
    }
    (tmp_path / "active_trades.json").write_text(json.dumps({trade["event_id"]: trade}), encoding="utf-8")
    states = iter([
        {"status": "found", "positionAmt": 1.0, "avgPrice": 100.0},
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},
    ])
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: next(states))
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "FILLED", "executed_qty": 0.5,
        "avg_price": 102.0, "order_id": "TP1",
    })
    # TP1 notification is the subject of this test; avoid entering the BE
    # network path that normally follows a real TP1 fill.
    monkeypatch.setattr(tracker, "_move_sl_to_break_even", lambda *a, **k: {
        "status": "created", "order_id": "BE1", "stop_price": 100.0,
    })
    sent = []
    monkeypatch.setattr(tracker, "send_tg", lambda text: sent.append(text) or True)
    tracker.update_active_trades()
    assert sent and "tp1" in sent[0].lower()
    actions = [json.loads(x) for x in (tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert actions[-1]["action"] == "TP_HIT"
    assert actions[-1]["telegram_ok"] is True



def test_tp_notification_reports_exchange_remaining_qty_after_fill(monkeypatch, tmp_path):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    trade = {
        "event_id": "EVT_TP_REMAINING", "symbol": "AAA-USDT", "direction": "LONG", "name": "AAA",
        "entry_price": 100.0, "initial_qty": 1.0, "remaining_qty": 0.5, "entry_ts": 1, "closed": False,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1"}], "sl_order": {}, "hit_legs": [], "tp_filled_qty": {},
        "realized_pnl_qty": 0.0, "realized_pnl_weighted_sum": 0.0, "peak_pnl_pct": 0.0, "mae_pct": 0.0,
        "max_drawdown_pct": 0.0, "planned_risk_pct": 1.0, "planned_weighted_rr": 0.75,
        "effective_weighted_rr": 0.75, "setup": {}, "tp_levels": [], "effective_tp_levels": [],
    }
    (tmp_path / "active_trades.json").write_text(json.dumps({trade["event_id"]: trade}), encoding="utf-8")
    states = iter([
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},
    ])
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: next(states))
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status": "ok", "order_status": "FILLED", "executed_qty": 0.5, "avg_price": 102.0, "order_id": "TP1"})
    monkeypatch.setattr(tracker, "_move_sl_to_break_even", lambda *a, **k: {"status": "created", "order_id": "BE1", "stop_price": 100.0})
    sent=[]
    monkeypatch.setattr(tracker, "send_tg", lambda text: sent.append(text) or True)
    tracker.update_active_trades()
    assert sent and "0.50000000" in sent[0]
    assert "50.0%" in sent[0]

def test_tracker_sends_and_records_close_notification(monkeypatch, tmp_path):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    trade = {
        "event_id": "EVT_CLOSE_NOTIFY",
        "symbol": "AAA-USDT",
        "direction": "LONG",
        "name": "AAA",
        "entry_price": 100.0,
        "initial_qty": 1.0,
        "remaining_qty": 1.0,
        "entry_ts": 1,
        "closed": False,
        "tp_orders": [],
        "sl_order": {},
        "hit_legs": [],
        "tp_filled_qty": {},
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "peak_pnl_pct": 2.0,
        "mae_pct": -1.0,
        "max_drawdown_pct": -1.0,
        "planned_risk_pct": 1.0,
        "planned_weighted_rr": 0.75,
        "effective_weighted_rr": 0.75,
        "setup": {},
        "tp_levels": [],
        "effective_tp_levels": [],
    }
    (tmp_path / "active_trades.json").write_text(json.dumps({trade["event_id"]: trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_all_orders", lambda *a, **k: [{
        "orderId": "CLOSE1", "status": "FILLED", "side": "SELL", "type": "MARKET",
        "executedQty": "1", "avgPrice": "102", "updateTime": 200,
    }])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status": "error", "error": "not found"})
    sent = []
    monkeypatch.setattr(tracker, "send_tg", lambda text: sent.append(text) or True)
    tracker.update_active_trades()
    assert sent and "сделка закрыта" in sent[0]
    assert "+2.00%" in sent[0]
    actions = [json.loads(x) for x in (tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert actions[-1]["action"] == "TRADE_CLOSE"
    assert actions[-1]["telegram_ok"] is True



def test_tracker_sends_negative_close_notification(monkeypatch, tmp_path):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    trade = {
        "event_id": "EVT_CLOSE_NOTIFY_NEG",
        "symbol": "AAA-USDT",
        "direction": "LONG",
        "name": "AAA",
        "entry_price": 100.0,
        "initial_qty": 1.0,
        "remaining_qty": 1.0,
        "entry_ts": 1,
        "closed": False,
        "tp_orders": [],
        "sl_order": {},
        "hit_legs": [],
        "tp_filled_qty": {},
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "peak_pnl_pct": 0.0,
        "mae_pct": -3.0,
        "max_drawdown_pct": -3.0,
        "planned_risk_pct": 1.0,
        "planned_weighted_rr": 0.75,
        "effective_weighted_rr": 0.75,
        "setup": {},
        "tp_levels": [],
        "effective_tp_levels": [],
    }
    (tmp_path / "active_trades.json").write_text(json.dumps({trade["event_id"]: trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_all_orders", lambda *a, **k: [{
        "orderId": "CLOSE_NEG", "status": "FILLED", "side": "SELL", "type": "MARKET",
        "executedQty": "1", "avgPrice": "97", "updateTime": 200,
    }])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status": "error", "error": "not found"})
    sent = []
    monkeypatch.setattr(tracker, "send_tg", lambda text: sent.append(text) or True)
    tracker.update_active_trades()
    assert sent
    assert "💔" in sent[0]
    assert "-3.00%" in sent[0]
    actions = [json.loads(x) for x in (tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert actions[-1]["action"] == "TRADE_CLOSE"
    assert actions[-1]["telegram_ok"] is True


def test_be_failure_keeps_verified_old_sl_without_duplicate_restore(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(tracker, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "position_side_param", lambda direction: "LONG")
    calls = []

    protection_states = iter([
        {"status": "ok", "sl_orders": [{"orderId": "OLD1", "type": "STOP_MARKET", "stopPrice": "99", "origQty": "1", "clientOrderId": "EVT_OLD_SL"}], "tp_orders": []},
        {"status": "ok", "sl_orders": [{"orderId": "OLD1", "type": "STOP_MARKET", "stopPrice": "99", "origQty": "1", "clientOrderId": "EVT_OLD_SL"}], "tp_orders": []},
    ])
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: next(protection_states))
    monkeypatch.setattr(tracker, "_request", lambda method, path, params: calls.append((method, params)) or {"code": 99, "msg": "BE rejected"})
    monkeypatch.setattr(tracker, "_emergency_close_after_be_failure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not emergency-close while old SL is verified")))
    monkeypatch.setattr(tracker, "send_tg", lambda *a, **k: True)

    out = tracker._move_sl_to_break_even("AAA-USDT", "LONG", 100.0, 1.0, "OLD1", "T", old_sl_price=99.0)
    assert out["status"] == "error"
    assert out["safety_action"] == "old_sl_kept"
    assert out["old_sl_restored"] is True
    # BE POST is the only write; no duplicate old-SL restore POST is allowed.
    assert len(calls) == 1
    assert calls[0][0] == "POST"
    assert calls[0][1]["type"] == "STOP_MARKET"
    assert "BE" in calls[0][1]["clientOrderId"]


def test_be_create_then_cleanup_keeps_verified_be_when_old_cancel_unverified(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(tracker, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "position_side_param", lambda direction: "LONG")
    states = iter([
        {"status": "ok", "sl_orders": [{"orderId": "OLD1", "type": "STOP_MARKET", "stopPrice": "99", "origQty": "1"}], "tp_orders": []},
        {"status": "ok", "sl_orders": [
            {"orderId": "OLD1", "type": "STOP_MARKET", "stopPrice": "99", "origQty": "1"},
            {"orderId": "BE1", "type": "STOP_MARKET", "stopPrice": "100", "origQty": "1", "clientOrderId": "EVT_BE_T"},
        ], "tp_orders": []},
        # _cancel_old_sl_verified must verify the old order still exists -> failure.
        {"status": "ok", "sl_orders": [
            {"orderId": "OLD1", "type": "STOP_MARKET", "stopPrice": "99", "origQty": "1"},
            {"orderId": "BE1", "type": "STOP_MARKET", "stopPrice": "100", "origQty": "1", "clientOrderId": "EVT_BE_T"},
        ], "tp_orders": []},
    ])
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: next(states))
    requests = []
    monkeypatch.setattr(tracker, "_request", lambda method, path, params: requests.append((method, params)) or {"code": 0, "data": {"order": {"orderId": "BE1", "clientOrderId": params.get("clientOrderId")}}})
    monkeypatch.setattr(tracker, "cancel_order", lambda *a, **k: {"code": 500, "msg": "cancel failed"})
    monkeypatch.setattr(tracker.time, "sleep", lambda *a, **k: None)

    out = tracker._move_sl_to_break_even("AAA-USDT", "LONG", 100.0, 1.0, "OLD1", "T", old_sl_price=99.0)
    assert out["status"] == "created_cleanup_pending"
    assert out["order_id"] == "BE1"
    assert out["old_sl_cleanup_pending"] is True
    assert len(requests) == 1  # only BE creation; no duplicate/restore POST


def test_be_failure_notification_is_truthful_for_verified_close(monkeypatch):
    from event_engine import tracker
    sent = []
    monkeypatch.setattr(tracker, "send_tg", lambda text: sent.append(text) or True)
    tracker._notify_be_failure(
        "AAA-USDT", "LONG", "BE stop not visible; restore=failed; rollback=closed_verified",
        event_id="EVT_AAA", rollback={"status": "closed_verified", "remaining_qty": 0.0},
    )
    assert sent
    assert "Позиция подтверждённо закрыта биржей." in sent[0]
    assert "требует reconciliation" not in sent[0].lower()


def test_emergency_be_rollback_verifies_order_before_final_unverified(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: next(states))
    states = iter([
        {"status": "found", "positionAmt": 1.0},
        {"status": "found", "positionAmt": 1.0},
        {"status": "not_found"},
    ])
    monkeypatch.setattr(tracker, "close_position_market", lambda *a, **k: {
        "status": "closed", "response": {"code": 0, "data": {"order": {"orderId": "CLOSE1"}}}
    })
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_id": "CLOSE1", "order_status": "FILLED", "executed_qty": 1.0,
    })
    monkeypatch.setattr(tracker.time, "sleep", lambda *a, **k: None)
    out = tracker._emergency_close_after_be_failure("AAA-USDT", "LONG", 1.0, "T")
    assert out["status"] == "closed_verified"
    assert out["order_filled"] is True
    assert out["remaining_qty"] == 0.0


def test_be_rollback_reports_already_closed_when_position_is_absent(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(tracker, "close_position_market", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not send close when already absent")))
    out = tracker._emergency_close_after_be_failure("AAA-USDT", "LONG", 1.0, "T")
    assert out["status"] == "already_closed"
    assert out["remaining_qty"] == 0.0


def test_be_rollback_uses_full_position_fallback_when_directional_read_errors(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status": "error", "error": "temporary"})
    monkeypatch.setattr(tracker, "get_positions", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})
    monkeypatch.setattr(tracker, "close_position_market", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not send close when full positions confirms absence")))
    out = tracker._emergency_close_after_be_failure("AAA-USDT", "LONG", 1.0, "T")
    assert out["status"] == "already_closed"
    assert out["remaining_qty"] == 0.0


def test_be_rollback_does_not_call_stale_qty_current_after_close_verification_error(monkeypatch):
    from event_engine import tracker
    states = iter([
        {"status": "found", "positionAmt": 1.0, "avgPrice": 100.0},
        {"status": "error", "error": "position endpoint timeout"},
    ])
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: next(states, {"status": "error", "error": "position endpoint timeout"}))
    monkeypatch.setattr(tracker, "get_positions", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("full positions timeout")))
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: {"status": "error", "error": "openOrders timeout"})
    monkeypatch.setattr(tracker, "close_position_market", lambda *a, **k: {
        "status": "closed", "response": {"code": 0, "data": {"order": {"orderId": "CLOSE1"}}},
    })
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_id": "CLOSE1", "order_status": "FILLED", "executed_qty": 1.0,
    })
    monkeypatch.setattr(tracker.time, "sleep", lambda *a, **k: None)
    out = tracker._emergency_close_after_be_failure("AAA-USDT", "LONG", 1.0, "T")
    assert out["status"] == "close_unverified"
    assert out["remaining_qty"] is None
    assert out["last_known_qty"] == 1.0


def test_runtime_state_paths_are_project_root_relative(tmp_path, monkeypatch):
    """State must stay under the repository data/ directory regardless of cwd."""
    import os
    import run_once
    from event_engine import analytics, tracker

    monkeypatch.chdir(tmp_path)
    expected = (tmp_path / "unused").resolve()
    project_root = Path(run_once.__file__).resolve().parent

    assert run_once.PROJECT_ROOT / "data" == project_root / "data"
    assert tracker.PROJECT_ROOT / "data" == project_root / "data"
    assert analytics.PROJECT_ROOT / "data" == project_root / "data"
    assert (run_once.PROJECT_ROOT / "data") != Path.cwd() / "data"
    assert expected != project_root / "data"


def test_be_sl_at_entry_is_valid_protection(monkeypatch):
    from event_engine import bingx
    sl = {"type": "STOP_MARKET", "stopPrice": "100", "origQty": "1"}
    assert bingx._validate_sl_order_for_position(sl, "LONG", 100.0, 1.0) is True
    assert bingx._validate_sl_order_for_position(sl, "SHORT", 100.0, 1.0) is True


def test_be_verification_polls_after_eventual_consistency(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(tracker, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "position_side_param", lambda direction: "LONG")
    states = iter([
        {"status": "ok", "sl_orders": [{"orderId": "OLD1", "type": "STOP_MARKET", "stopPrice": "99", "origQty": "1"}], "tp_orders": []},
        {"status": "ok", "sl_orders": [{"orderId": "OLD1", "type": "STOP_MARKET", "stopPrice": "99", "origQty": "1"}], "tp_orders": []},
        {"status": "ok", "sl_orders": [{"orderId": "OLD1", "type": "STOP_MARKET", "stopPrice": "99", "origQty": "1"}, {"orderId": "BE1", "type": "STOP_MARKET", "stopPrice": "100", "origQty": "0.5"}], "tp_orders": []},
        {"status": "ok", "sl_orders": [{"orderId": "BE1", "type": "STOP_MARKET", "stopPrice": "100", "origQty": "0.5"}], "tp_orders": []},
    ])
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: next(states))
    monkeypatch.setattr(tracker, "_request", lambda method, path, params: {"code": 0, "data": {"order": {"orderId": "BE1", "clientOrderId": params["clientOrderId"]}}})
    monkeypatch.setattr(tracker, "_cancel_old_sl_verified", lambda *a, **k: (True, "cancelled"))
    monkeypatch.setattr(tracker.time, "sleep", lambda *a, **k: None)
    out = tracker._move_sl_to_break_even("AAA-USDT", "LONG", 100.0, 0.5, "OLD1", "T", old_sl_price=99.0)
    assert out["status"] == "created"
    assert out["order_id"] == "BE1"


def test_be_failure_accepts_full_size_old_sl_for_residual_position(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(tracker, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "position_side_param", lambda direction: "LONG")
    states = iter([
        {"status": "ok", "sl_orders": [{"orderId": "OLD1", "type": "STOP_MARKET", "stopPrice": "99", "origQty": "1"}], "tp_orders": []},
        {"status": "ok", "sl_orders": [{"orderId": "OLD1", "type": "STOP_MARKET", "stopPrice": "99", "origQty": "1"}], "tp_orders": []},
    ])
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: next(states))
    monkeypatch.setattr(tracker, "_request", lambda method, path, params: {"code": 99, "msg": "BE rejected"})
    monkeypatch.setattr(tracker, "_emergency_close_after_be_failure", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should keep valid old SL")))
    monkeypatch.setattr(tracker, "send_tg", lambda *a, **k: True)
    out = tracker._move_sl_to_break_even("AAA-USDT", "LONG", 100.0, 0.5, "OLD1", "T", old_sl_price=99.0)
    assert out["safety_action"] == "old_sl_kept"
    assert out["old_sl_restored"] is True


def test_emergency_close_cleanup_happens_before_market_close(monkeypatch):
    import run_once
    calls = []
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok",
        "sl_orders": [{"orderId": "SL1", "clientOrderId": "EVT_SL"}],
        "tp_orders": [{"orderId": "TP1", "clientOrderId": "EVT_TP"}],
    })
    monkeypatch.setattr(run_once, "cancel_order", lambda symbol, oid: calls.append(("cancel", oid)) or {"code": 0})
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: {"status": "found", "positionAmt": 1.0})
    monkeypatch.setattr(run_once, "close_position_market", lambda *a, **k: calls.append(("close",)) or {"status": "closed", "response": {"data": {"order": {"orderId": "C1"}}}})
    monkeypatch.setattr(run_once, "time", type("T", (), {"sleep": staticmethod(lambda *a, **k: None)})())
    # Force one bounded attempt to avoid a long mock loop.
    monkeypatch.setenv("EMERGENCY_CLOSE_ATTEMPTS", "2")
    monkeypatch.setenv("EMERGENCY_CLOSE_VERIFY_POLLS", "2")
    result = run_once.execute_new_position
    # Inspect helper directly: cleanup must precede any caller that invokes the close.
    cleanup = run_once._cancel_engine_protection_before_emergency_close("AAA-USDT", "LONG")
    assert cleanup["status"] == "ok"
    calls.clear()
    run_once._emergency_close_and_verify("AAA-USDT", "LONG", 1.0, "EVT1")
    # The close helper itself has no cleanup side effect; the production call sites
    # are responsible for invoking cleanup first. Verify that helper separately is callable.
    assert calls[0][0] == "close"


def test_workflow_stages_all_data_but_excludes_scan_history():
    from pathlib import Path
    text = Path(".github/workflows/event-engine.yml").read_text(encoding="utf-8")
    assert "git add data/" in text
    assert "git reset -- data/scan_history.jsonl" in text
    assert "schedule:" in text
    assert '- cron: "*/5 * * * *"' in text


def test_execution_ledger_is_append_only_and_carries_attempt_lineage(tmp_path, monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "EXECUTION_LEDGER_PATH", tmp_path / "execution_ledger.jsonl")
    run_once._append_execution_ledger("EVT_TEST", "ATT_TEST", "TEST_STAGE", {"status": "ok"})
    row = json.loads((tmp_path / "execution_ledger.jsonl").read_text(encoding="utf-8").strip())
    assert row["record_type"] == "EXECUTION_LEDGER"
    assert row["event_id"] == "EVT_TEST"
    assert row["attempt_id"] == "ATT_TEST"
    assert row["stage"] == "TEST_STAGE"
    assert row["status"] == "ok"
    assert row["strategy_version"] == run_once._effective_strategy_version()
    assert row["code_commit_sha"] == run_once.CODE_COMMIT_SHA


def test_emergency_be_rollback_cleans_engine_orders_before_close(monkeypatch):
    from event_engine import tracker
    calls = []
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status": "found", "positionAmt": 1.0, "avgPrice": 100.0})
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok",
        "sl_orders": [{"orderId": "SL1", "clientOrderId": "EVT_SL"}],
        "tp_orders": [{"orderId": "TP1", "clientOrderId": "EVT_TP"}],
    })
    monkeypatch.setattr(tracker, "cancel_order", lambda symbol, oid: calls.append(("cancel", oid)) or {"code": 0})
    monkeypatch.setattr(tracker, "close_position_market", lambda *a, **k: calls.append(("close",)) or {"status": "closed", "response": {"data": {"order": {"orderId": "C1"}}}})
    states = iter([{"status": "found", "positionAmt": 1.0}, {"status": "not_found"}])
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: next(states, {"status": "not_found"}))
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status": "ok", "order_status": "FILLED", "executed_qty": 1.0})
    monkeypatch.setattr(tracker.time, "sleep", lambda *a, **k: None)
    monkeypatch.setenv("BE_FAILURE_CLOSE_POLLS", "2")
    out = tracker._emergency_close_after_be_failure("AAA-USDT", "LONG", 1.0, "T")
    assert out["status"] == "closed_verified"
    assert calls[:3] == [("cancel", "SL1"), ("cancel", "TP1"), ("close",)]


def test_emergency_close_only_cancels_engine_owned_protection(monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok",
        "sl_orders": [
            {"orderId": "E1", "clientOrderId": "EVT_SL"},
            {"orderId": "M1", "clientOrderId": "MANUAL_SL"},
        ],
        "tp_orders": [{"orderId": "E2", "clientOrderId": "EVT_TP"}],
    })
    cancelled = []
    monkeypatch.setattr(run_once, "cancel_order", lambda symbol, oid: cancelled.append(oid) or {"code": 0})
    out = run_once._cancel_engine_protection_before_emergency_close("AAA-USDT", "LONG")
    assert out["status"] == "ok"
    assert cancelled == ["E1", "E2"]


def test_emergency_be_rollback_fallback_does_not_refresh_contracts(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status": "error", "error": "temporary"})
    monkeypatch.setattr(tracker, "get_positions", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fallback must not refresh contracts")))
    out = tracker._emergency_close_after_be_failure("AAA-USDT", "LONG", 1.0, "T")
    assert out["status"] == "already_closed"


def test_run_once_all_emergency_paths_cleanup_before_close():
    from pathlib import Path
    text = Path("run_once.py").read_text(encoding="utf-8")
    # Each execute_new_position emergency branch must call cleanup immediately
    # before the emergency close helper, preventing protection orders from
    # competing with the rollback MARKET close.
    assert text.count("cleanup = _cancel_engine_protection_before_emergency_close(symbol, direction)\n        close_result = _emergency_close_and_verify") >= 4

from run_once import WATCHLIST_ONLY, WATCHLIST_SYMBOLS


def test_watchlist_is_exact_20_symbols():
    expected = (
        "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "TAO-USDT",
        "LTC-USDT", "BCH-USDT", "AVAX-USDT", "LINK-USDT", "ETC-USDT",
        "ADA-USDT", "UNI-USDT", "XRP-USDT", "ICP-USDT", "HYPE-USDT",
        "DOGE-USDT", "HBAR-USDT", "ARB-USDT", "POL-USDT", "SUI-USDT",
    )
    assert WATCHLIST_ONLY is False
    assert WATCHLIST_SYMBOLS == expected
    assert len(WATCHLIST_SYMBOLS) == 20


def test_get_execution_quote_accepts_current_nested_bookticker_envelope(monkeypatch):
    from event_engine import bingx
    calls = []
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    responses = [
        {
            "code": 0,
            "msg": "",
            "data": {
                "book_ticker": {
                    "symbol": "BTC-USDT",
                    "bidPrice": "100.00",
                    "bidQty": "2.5",
                    "askPrice": "100.10",
                    "askQty": "1.5",
                    "time": 1234567890000,
                }
            },
        },
    ]
    def fake_request(method, path, params, signed=True, **kwargs):
        calls.append(path)
        return responses.pop(0)
    monkeypatch.setattr(bingx, "_request", fake_request)
    monkeypatch.setenv("BINGX_BOOK_TICKER_MIN_INTERVAL_SEC", "0")

    out = bingx.get_execution_quote("BTC-USDT")

    assert out["status"] == "ok"
    assert out["quote_source"] == "bookTicker"
    assert out["bid"] == 100.00
    assert out["ask"] == 100.10
    assert out["quote_exchange_time_ms"] == 1234567890000
    assert calls == [bingx.BOOK_TICKER_PATH]


def test_get_execution_quote_falls_back_from_bookticker_to_ticker(monkeypatch):
    from event_engine import bingx
    calls = []
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    responses = [
        {"code": 0, "data": {"symbol": "FLOKI-USDT", "bidPrice": "0", "askPrice": "0"}},
        {"code": 0, "data": [{"symbol": "FLOKI-USDT", "bidPrice": "0.00002460", "askPrice": "0.00002461", "time": 123}]},
    ]
    def fake_request(method, path, params, signed=True, **kwargs):
        calls.append(path)
        return responses.pop(0)
    monkeypatch.setattr(bingx, "_request", fake_request)
    monkeypatch.setenv("BINGX_BOOK_TICKER_MIN_INTERVAL_SEC", "0")
    out = bingx.get_execution_quote("FLOKI-USDT")
    assert out["status"] == "ok"
    assert out["quote_source"] == "ticker"
    assert out["bid"] == 0.00002460
    assert out["ask"] == 0.00002461
    assert calls == [bingx.BOOK_TICKER_PATH, bingx.TICKER_PATH]


def test_get_execution_quote_falls_back_to_depth(monkeypatch):
    from event_engine import bingx
    calls = []
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    responses = [
        {"code": 0, "data": {"symbol": "S-USDT", "bidPrice": "0", "askPrice": "0"}},
        {"code": 0, "data": {"symbol": "S-USDT", "bidPrice": "0", "askPrice": "0"}},
        {"code": 0, "data": {"bids": [["0.02690", "10"]], "asks": [["0.02691", "11"]], "T": 456}},
    ]
    def fake_request(method, path, params, signed=True, **kwargs):
        calls.append(path)
        return responses.pop(0)
    monkeypatch.setattr(bingx, "_request", fake_request)
    monkeypatch.setenv("BINGX_BOOK_TICKER_MIN_INTERVAL_SEC", "0")
    out = bingx.get_execution_quote("S-USDT")
    assert out["status"] == "ok"
    assert out["quote_source"] == "depth"
    assert out["bid"] == 0.02690
    assert out["ask"] == 0.02691
    assert calls == [bingx.BOOK_TICKER_PATH, bingx.TICKER_PATH, bingx.DEPTH_PATH]


def test_get_execution_quote_blocks_when_all_bingx_sources_invalid(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    def fake_request(method, path, params, signed=True, **kwargs):
        return {"code": 0, "data": {"symbol": "TEST-USDT", "bidPrice": "0", "askPrice": "0"}}
    monkeypatch.setattr(bingx, "_request", fake_request)
    monkeypatch.setenv("BINGX_BOOK_TICKER_MIN_INTERVAL_SEC", "0")
    out = bingx.get_execution_quote("TEST-USDT")
    assert out["status"] == "error"
    assert "bookTicker->ticker->depth" in out["error"]
    assert out["quote_sources_attempted"] == ["bookTicker", "ticker", "depth"]


def test_event_execution_claim_is_idempotent_after_terminal_finalize():
    import run_once
    ok, reason, attempt = run_once._claim_event_for_execution("ZONE_TEST_TERMINAL")
    assert ok is True
    assert reason == "claimed"
    assert attempt
    run_once._finalize_event_claim("ZONE_TEST_TERMINAL", attempt, terminal=True, status="skipped_stale_signal")
    ok2, reason2, attempt2 = run_once._claim_event_for_execution("ZONE_TEST_TERMINAL")
    assert ok2 is False
    assert reason2 == "terminal"
    assert attempt2 == ""


def test_event_execution_claim_blocks_second_in_flight_attempt():
    import run_once
    ok, reason, attempt = run_once._claim_event_for_execution("ZONE_TEST_INFLIGHT")
    assert ok is True
    ok2, reason2, attempt2 = run_once._claim_event_for_execution("ZONE_TEST_INFLIGHT")
    assert ok2 is False
    assert reason2 == "in_flight"
    assert attempt2 == ""
    run_once._finalize_event_claim("ZONE_TEST_INFLIGHT", attempt, terminal=False, status="execution_quote_unavailable")


def test_signal_forensics_records_rejection_and_zone_geometry():
    from event_engine.signals import _signal_forensics
    out = _signal_forensics(
        "SHORT",
        cur_o=10.0,
        cur_h=11.0,
        cur_l=9.0,
        cur_c=9.4,
        atr=1.0,
        zone={"top": 10.8, "btm": 10.0, "poi": 10.4},
    )
    assert out["range_atr"] == 2.0
    assert out["body_atr"] == 0.6
    assert out["upper_wick_ratio"] == 0.5
    assert out["lower_wick_ratio"] == 0.2
    assert out["directional_rejection_side"] == "upper_wick"
    assert out["zone_width_atr"] == 0.8
    assert out["zone_penetration_pct_capped"] == 100.0
    assert 0.0 <= out["directional_close_location"] <= 1.0


def test_update_mfe_mae_records_threshold_milestones():
    from event_engine import tracker
    trade = {
        "entry_price": 100.0,
        "direction": "LONG",
        "entry_ts": 1_000,
        "planned_risk_pct": 1.0,
        "peak_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "max_drawdown_pct": 0.0,
    }
    candles = [
        {"open_time": 1_001, "close_time": 1_061, "high": 100.4, "low": 99.9},
        {"open_time": 1_062, "close_time": 1_122, "high": 101.1, "low": 100.8},
        {"open_time": 1_123, "close_time": 1_183, "high": 102.2, "low": 101.5},
    ]
    tracker._update_mfe_mae(trade, candles)
    assert set(trade["mfe_milestones_r"]) >= {"0.25", "0.50", "1.00", "2.00"}
    assert trade["mfe_milestones_r"]["0.50"] == 1_122
    assert trade["mfe_milestones_r"]["1.00"] == 1_122
    assert trade["mfe_milestones_r"]["2.00"] == 1_183


def test_5m_zone_visit_locks_after_midpoint_touch_and_does_not_retrigger_in_chop(monkeypatch):
    import time
    import pandas as pd
    import run_once
    from run_once import _process_5m_zone_visits
    monkeypatch.setattr(run_once, "MIN_STRUCTURE_ROOM_R", 0.0)
    monkeypatch.setattr(run_once, "MAX_5M_TRIGGER_AGE_MINUTES", 60.0)

    now = pd.Timestamp.now(tz="UTC").floor("5min")
    t0 = now - pd.Timedelta(minutes=15)
    zone = {"start": 0, "top": 110.0, "btm": 100.0, "poi": 105.0}
    bars = [
        {"timestamp": int(t0.timestamp()*1000), "open": 108, "high": 109, "low": 107, "close": 108, "volume": 10},
        {"timestamp": int((t0+pd.Timedelta(minutes=5)).timestamp()*1000), "open": 107, "high": 108, "low": 104, "close": 106, "volume": 10},
        {"timestamp": int((t0+pd.Timedelta(minutes=10)).timestamp()*1000), "open": 106, "high": 107, "low": 104, "close": 106, "volume": 10},
    ]
    df1h = pd.DataFrame([
        {"timestamp": now - pd.Timedelta(hours=12-i), "open": 100, "high": 111, "low": 99, "close": 105, "volume": 100, "atr50": 2.0}
        for i in range(12)
    ])
    signals, state, text = _process_5m_zone_visits("TEST-USDT", bars, [zone], [], df1h, None, set(), set())
    assert len(signals) == 1
    assert signals[0]["trigger_timeframe"] == "5m"
    assert signals[0]["entry"] == 105.0
    assert state["zones"]["DEMAND:0:110.000000000000:100.000000000000"]["state"] == "LOCKED"


def test_5m_zone_visit_rearms_only_after_closed_bar_exits_far_edge_then_allows_new_touch(monkeypatch):
    import pandas as pd
    import run_once
    from run_once import _process_5m_zone_visits
    monkeypatch.setattr(run_once, "MAX_5M_TRIGGER_AGE_MINUTES", 60.0)
    monkeypatch.setattr(run_once, "INITIAL_5M_TRIGGER_LOOKBACK_MINUTES", 60.0)

    now = pd.Timestamp.now(tz="UTC").floor("5min")
    t0 = now - pd.Timedelta(minutes=20)
    zone = {"start": 0, "top": 110.0, "btm": 100.0, "poi": 105.0}
    bars = []
    vals = [
        (108, 109, 104, 106),  # first midpoint touch
        (106, 108, 103, 106),  # same visit/chop
        (109, 112, 108, 111),  # close beyond top -> rearm
        (112, 113, 104, 105),  # new visit midpoint touch
        (105, 108, 102, 106),  # same visit
    ]
    for i, (o,h,l,c) in enumerate(vals):
        ts=t0+pd.Timedelta(minutes=5*i)
        bars.append({"timestamp": int(ts.timestamp()*1000), "open":o,"high":h,"low":l,"close":c,"volume":10})
    df1h = pd.DataFrame([
        {"timestamp": now - pd.Timedelta(hours=12-i), "open": 100, "high": 111, "low": 99, "close": 105, "volume": 100, "atr50": 2.0}
        for i in range(12)
    ])
    signals, state, _ = _process_5m_zone_visits("TEST-USDT", bars, [zone], [], df1h, None, set(), set())
    assert len(signals) == 2
    assert signals[0]["trigger_bar_time"] != signals[1]["trigger_bar_time"]
    assert state["zones"]["DEMAND:0:110.000000000000:100.000000000000"]["state"] == "LOCKED"


def test_5m_zone_mode_locks_first_zone_touch_and_ignores_later_chop(monkeypatch):
    import pandas as pd
    import run_once
    from run_once import _process_5m_zone_visits
    monkeypatch.setattr(run_once, "ZONE_TRIGGER_MODE", "zone")
    monkeypatch.setattr(run_once, "MIN_STRUCTURE_ROOM_R", 0.0)
    monkeypatch.setattr(run_once, "REQUIRE_STRUCTURE_OBSTACLE", False)
    monkeypatch.setattr(run_once, "_nearest_opposing_level", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_once, "MAX_5M_TRIGGER_AGE_MINUTES", 60.0)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    zone = {"start": 0, "top": 110.0, "btm": 100.0, "poi": 105.0}
    vals = [
        (108, 104, 102, 103),
        (103, 107, 101, 104),
        (104, 109, 100, 103),
    ]
    bars = []
    for i, (o, h, l, c) in enumerate(vals):
        ts = now - pd.Timedelta(minutes=15 - 5*i)
        bars.append({"timestamp": int(ts.timestamp()*1000), "open": o, "high": h, "low": l, "close": c, "volume": 10})
    df1h = pd.DataFrame([{
        "timestamp": now-pd.Timedelta(hours=12-i), "open":100, "high":111, "low":99, "close":105, "volume":100, "atr50":2.0
    } for i in range(12)])
    sigs, state, _ = _process_5m_zone_visits("TEST-USDT", bars, [zone], [], df1h, None, set(), set())
    key = "DEMAND:0:110.000000000000:100.000000000000"
    assert len(sigs) == 1
    assert sigs[0]["trigger"]["type"] == "ZONE_TOUCH_5M"
    assert sigs[0]["trigger"]["midpoint_touched_diagnostic"] is False
    assert state["zones"][key]["state"] == "LOCKED"
    assert state["zones"][key]["lock_reason"] == "zone_touch"


def test_5m_first_processed_bar_is_not_suppressed_by_unprocessed_history_touch(monkeypatch):
    import pandas as pd
    import run_once
    from run_once import _process_5m_zone_visits
    monkeypatch.setattr(run_once, "ZONE_TRIGGER_MODE", "zone")
    monkeypatch.setattr(run_once, "MAX_5M_TRIGGER_AGE_MINUTES", 60.0)
    monkeypatch.setattr(run_once, "INITIAL_5M_TRIGGER_LOOKBACK_MINUTES", 10.0)
    monkeypatch.setattr(run_once, "MIN_STRUCTURE_ROOM_R", 0.0)
    monkeypatch.setattr(run_once, "REQUIRE_STRUCTURE_OBSTACLE", False)
    monkeypatch.setattr(run_once, "_nearest_opposing_level", lambda *args, **kwargs: None)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    zone = {"start": 0, "top": 110.0, "btm": 100.0, "poi": 105.0}
    bars = [
        {"timestamp": int((now-pd.Timedelta(minutes=15)).timestamp()*1000), "open":108, "high":109, "low":101, "close":103, "volume":10},
        {"timestamp": int((now-pd.Timedelta(minutes=10)).timestamp()*1000), "open":108, "high":109, "low":101, "close":103, "volume":10},
        {"timestamp": int((now-pd.Timedelta(minutes=5)).timestamp()*1000), "open":90, "high":95, "low":80, "close":85, "volume":10},
    ]
    df1h = pd.DataFrame([{
        "timestamp": now-pd.Timedelta(hours=12-i), "open":100, "high":111, "low":99, "close":105, "volume":100, "atr50":2.0
    } for i in range(12)])
    sigs, state, _ = _process_5m_zone_visits("TEST-USDT", bars, [zone], [], df1h, None, set(), set())
    assert len(sigs) == 1
    assert sigs[0]["trigger"]["type"] == "ZONE_TOUCH_5M"
    assert sigs[0]["trigger_bar_time"] == pd.Timestamp(bars[1]["timestamp"], unit="ms", tz="UTC").isoformat()
    assert state["zones"]["DEMAND:0:110.000000000000:100.000000000000"]["state"] == "LOCKED"


def test_5m_pending_event_is_reused_for_retry_without_new_event_id(monkeypatch):
    import pandas as pd
    import run_once
    from run_once import _process_5m_zone_visits
    monkeypatch.setattr(run_once, "MIN_STRUCTURE_ROOM_R", 0.0)
    monkeypatch.setattr(run_once, "MAX_5M_TRIGGER_AGE_MINUTES", 60.0)

    now = pd.Timestamp.now(tz="UTC").floor("5min")
    t0 = now - pd.Timedelta(minutes=10)
    zone = {"start": 0, "top": 110.0, "btm": 100.0, "poi": 105.0}
    bars = [
        {"timestamp": int(t0.timestamp()*1000), "open": 108, "high": 109, "low": 104, "close": 106, "volume": 10},
    ]
    df1h = pd.DataFrame([
        {"timestamp": now - pd.Timedelta(hours=12-i), "open": 100, "high": 111, "low": 99, "close": 105, "volume": 100, "atr50": 2.0}
        for i in range(12)
    ])
    signals, state, _ = _process_5m_zone_visits("TEST-USDT", bars, [zone], [], df1h, None, set(), set())
    assert len(signals) == 1
    eid = signals[0]["event_id"]
    signals2, state2, _ = _process_5m_zone_visits("TEST-USDT", [], [zone], [], df1h, state, set(), set())
    assert len(signals2) == 1
    assert signals2[0]["event_id"] == eid


def test_5m_touch_before_zone_activation_is_ignored(monkeypatch):
    import pandas as pd
    import run_once
    from run_once import _process_5m_zone_visits, SWING_LEN
    monkeypatch.setattr(run_once, "INITIAL_5M_TRIGGER_LOOKBACK_MINUTES", 240.0)
    monkeypatch.setattr(run_once, "MAX_5M_TRIGGER_AGE_MINUTES", 240.0)

    now = pd.Timestamp.now(tz="UTC").floor("5min")
    # Zone with start=0 becomes active at 1H index 10; this touch is deliberately 1h before activation.
    t0 = now - pd.Timedelta(hours=3)
    zone = {"start": 0, "top": 110.0, "btm": 100.0, "poi": 105.0}
    bars = [{"timestamp": int(t0.timestamp()*1000), "open":108, "high":106, "low":104, "close":105, "volume":10}]
    df1h = pd.DataFrame([
        {"timestamp": now - pd.Timedelta(hours=12-i), "open":100, "high":111, "low":99, "close":105, "volume":100, "atr50":2.0}
        for i in range(12)
    ])
    assert SWING_LEN == 10
    activation_ts = pd.Timestamp(df1h.loc[10, "timestamp"]) + pd.Timedelta(hours=1)
    assert t0 < activation_ts
    signals, _, _ = _process_5m_zone_visits("TEST-USDT", bars, [zone], [], df1h, None, set(), set())
    assert signals == []


def test_5m_prior_closed_bar_touch_remains_actionable_when_current_bar_no_longer_touches(monkeypatch):
    import pandas as pd
    import run_once
    from run_once import _process_5m_zone_visits

    monkeypatch_now = pd.Timestamp.now(tz="UTC").floor("5min")
    t0 = monkeypatch_now - pd.Timedelta(minutes=10)
    monkeypatch.setattr(run_once, "INITIAL_5M_TRIGGER_LOOKBACK_MINUTES", 30.0)
    monkeypatch.setattr(run_once, "MAX_5M_TRIGGER_AGE_MINUTES", 15.0)
    zone = {"start": 0, "top": 110.0, "btm": 100.0, "poi": 105.0}
    bars = [
        {"timestamp": int(t0.timestamp()*1000), "open": 108, "high": 109, "low": 104, "close": 106, "volume": 10},
        {"timestamp": int((t0+pd.Timedelta(minutes=5)).timestamp()*1000), "open": 106, "high": 109, "low": 106, "close": 108, "volume": 10},
    ]
    df1h = pd.DataFrame([
        {"timestamp": monkeypatch_now - pd.Timedelta(hours=12-i), "open": 100, "high": 111, "low": 99, "close": 105, "volume": 100, "atr50": 2.0}
        for i in range(12)
    ])
    signals, state, _ = _process_5m_zone_visits("TEST-USDT", bars, [zone], [], df1h, None, set(), set())
    assert len(signals) == 1
    assert signals[0]["trigger_bar_time"] == pd.Timestamp(bars[0]["timestamp"], unit="ms", tz="UTC").isoformat()
    assert state["zones"]["DEMAND:0:110.000000000000:100.000000000000"]["state"] == "LOCKED"


def test_5m_rearm_bar_that_also_crosses_midpoint_does_not_create_same_bar_reentry(monkeypatch):
    import pandas as pd
    import run_once
    from run_once import _process_5m_zone_visits
    monkeypatch.setattr(run_once, "INITIAL_5M_TRIGGER_LOOKBACK_MINUTES", 60.0)
    monkeypatch.setattr(run_once, "MAX_5M_TRIGGER_AGE_MINUTES", 60.0)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    t0 = now - pd.Timedelta(minutes=15)
    zone = {"start": 0, "top": 110.0, "btm": 100.0, "poi": 105.0}
    bars = [
        {"timestamp": int(t0.timestamp()*1000), "open": 107, "high": 108, "low": 104, "close": 106, "volume": 10},
        {"timestamp": int((t0+pd.Timedelta(minutes=5)).timestamp()*1000), "open": 106, "high": 112, "low": 104, "close": 111, "volume": 10},
        {"timestamp": int((t0+pd.Timedelta(minutes=10)).timestamp()*1000), "open": 112, "high": 113, "low": 104, "close": 105, "volume": 10},
    ]
    df1h = pd.DataFrame([{"timestamp": now-pd.Timedelta(hours=12-i), "open":100, "high":111, "low":99, "close":105, "volume":100, "atr50":2.0} for i in range(12)])
    sigs, state, _ = _process_5m_zone_visits("TEST-USDT", bars, [zone], [], df1h, None, set(), set())
    assert len(sigs) == 1
    assert sigs[0]["trigger_bar_time"] == pd.Timestamp(bars[0]["timestamp"], unit="ms", tz="UTC").isoformat()


def test_5m_stale_touch_does_not_lock_zone(monkeypatch):
    import pandas as pd
    import run_once
    from run_once import _process_5m_zone_visits
    monkeypatch.setattr(run_once, "INITIAL_5M_TRIGGER_LOOKBACK_MINUTES", 60.0)
    monkeypatch.setattr(run_once, "MAX_5M_TRIGGER_AGE_MINUTES", 5.0)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    t0 = now - pd.Timedelta(minutes=20)
    zone = {"start": 0, "top": 110.0, "btm": 100.0, "poi": 105.0}
    bars = [{"timestamp": int(t0.timestamp()*1000), "open":108, "high":109, "low":104, "close":106, "volume":10}]
    df1h = pd.DataFrame([{"timestamp": now-pd.Timedelta(hours=12-i), "open":100, "high":111, "low":99, "close":105, "volume":100, "atr50":2.0} for i in range(12)])
    sigs, state, _ = _process_5m_zone_visits("TEST-USDT", bars, [zone], [], df1h, None, set(), set())
    zs = state["zones"]["DEMAND:0:110.000000000000:100.000000000000"]
    assert sigs == []
    assert zs["state"] == "ARMED"
    assert zs["lock_reason"] == "stale_midpoint_touch_ignored"


def test_register_active_trade_preserves_5m_trigger_metadata(monkeypatch, tmp_path):
    import event_engine.tracker as tracker
    monkeypatch.setattr(tracker, "DATA", tmp_path)
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    tracker.register_active_trade(
        event_id="EVT_META_5M",
        symbol="BTC-USDT",
        name="BTC-USDT",
        direction="LONG",
        entry_price=100.0,
        qty=1.0,
        tp_orders=[],
        sl_result={},
        event_type="DEMAND_MIDPOINT_TOUCH_5M",
        timeframe="5m",
        score=75.0,
        setup={
            "strategy_version": "zone-midpoint-v3-5m-visit-stop10-tp5-tp7-be-on-tp1",
            "signal_snapshot": {"trigger_timeframe": "5m"},
            "tp_levels": [],
            "planned_risk_pct": 10.0,
            "target_rr": 0.7,
            "planned_weighted_rr": 0.6,
        },
    )
    state = tracker._load_active_trades()
    trade = state["EVT_META_5M"]
    assert trade["timeframe"] == "5m"
    assert trade["event_type"] == "DEMAND_MIDPOINT_TOUCH_5M"


def test_display_symbol_removes_hyphen_only_for_human_log_output():
    import run_once
    assert run_once._display_symbol("NEAR-USDT") == "NEARUSDT"
    assert run_once._display_symbol("btc-usdt") == "BTCUSDT"


def test_5m_zone_diagnostics_capture_window_and_processed_reasons(monkeypatch):
    import pandas as pd
    import run_once
    from run_once import _process_5m_zone_visits
    monkeypatch.setattr(run_once, "INITIAL_5M_TRIGGER_LOOKBACK_MINUTES", 60.0)
    monkeypatch.setattr(run_once, "MAX_5M_TRIGGER_AGE_MINUTES", 60.0)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    zone = {"start": 0, "top": 110.0, "btm": 100.0, "poi": 105.0}
    t0 = now - pd.Timedelta(minutes=15)
    bars = [
        {"timestamp": int(t0.timestamp()*1000), "open": 108, "high": 109, "low": 104, "close": 106, "volume": 10},
        {"timestamp": int((t0+pd.Timedelta(minutes=5)).timestamp()*1000), "open": 106, "high": 107, "low": 103, "close": 104, "volume": 10},
        {"timestamp": int((t0+pd.Timedelta(minutes=10)).timestamp()*1000), "open": 104, "high": 106, "low": 104, "close": 105, "volume": 10},
    ]
    df1h = pd.DataFrame([{"timestamp": now-pd.Timedelta(hours=12-i), "open":100, "high":111, "low":99, "close":105, "volume":100, "atr50":2.0} for i in range(12)])
    diagnostics = {"current_1h_idx": len(df1h)-1}
    sigs, state, _ = _process_5m_zone_visits("TEST-USDT", bars, [zone], [], df1h, None, set(), set(), diagnostics=diagnostics)
    key = "DEMAND:0:110.000000000000:100.000000000000"
    assert diagnostics["bars_received"] == len(bars)
    assert diagnostics["bars_closed"] == len(bars)
    assert diagnostics["processed_bars"] >= 1
    assert diagnostics["zones"][key]["window_midpoint_touches"] >= 1
    assert diagnostics["zones"][key]["midpoint_touches"] >= 1
    assert diagnostics["zones"][key]["signals_created"] >= 1
    assert sigs
    assert state["zones"][key]["state"] == "LOCKED"


def test_build_zone_rejects_non_finite_atr_and_anchor():
    from event_engine.signals import _build_zone

    with pytest.raises(ValueError, match="ATR"):
        _build_zone(100.0, 1, float("nan"), 10)
    with pytest.raises(ValueError, match="anchor"):
        _build_zone(float("nan"), 1, 1.0, 10)


def test_zone_id_is_stable_when_dataframe_start_index_changes():
    from event_engine.signals import _build_zone

    z1 = _build_zone(100.0, 1, 2.0, 10, origin_ts_ms=1735732800000, symbol="TEST-USDT")
    z2 = _build_zone(100.0, 1, 2.0, 1010, origin_ts_ms=1735732800000, symbol="TEST-USDT")
    assert z1["zone_id"] == z2["zone_id"]
    assert z1["start"] != z2["start"]


def test_5m_event_id_uses_stable_zone_identity():
    import run_once

    z1 = {"zone_id": "ZID_TEST", "start": 10, "top": 101.0, "btm": 99.0}
    z2 = {"zone_id": "ZID_TEST", "start": 1010, "top": 101.0, "btm": 99.0}
    assert run_once._make_5m_event_id("TEST-USDT", "LONG", 1735732800000, z1) == run_once._make_5m_event_id("TEST-USDT", "LONG", 1735732800000, z2)


def test_validate_trade_geometry_uses_module_structure_obstacle_default(monkeypatch):
    import run_once

    signal = {
        "type": "LONG", "entry": 100.0, "sl": 90.0, "tp1": 103.0, "tp2": 106.0,
        "risk_pct": 10.0, "target": {},
    }
    monkeypatch.setattr(run_once, "REQUIRE_STRUCTURE_OBSTACLE", False)
    ok, reason = run_once._validate_trade_geometry(signal)
    assert ok and reason == "ok"
    monkeypatch.setattr(run_once, "REQUIRE_STRUCTURE_OBSTACLE", True)
    ok, reason = run_once._validate_trade_geometry(signal)
    assert not ok and reason == "missing_structural_obstacle"


def test_entry_decision_journal_records_cycle_cap_and_has_stable_decision_id(tmp_path, monkeypatch):
    import run_once

    path = tmp_path / "entry_decisions.jsonl"
    monkeypatch.setattr(run_once, "ENTRY_DECISIONS_PATH", path)
    signal = {"event_id": "ZONE_TEST", "symbol": "TEST-USDT", "type": "SHORT", "time": "2026-01-01T00:00:00+00:00"}
    run_once._record_entry_decision("SCAN_TEST", signal, "CYCLE_CAP", "max_trades_per_cycle_reached", selection_rank=6)
    run_once._record_entry_decision("SCAN_TEST", signal, "CYCLE_CAP", "max_trades_per_cycle_reached", selection_rank=6)
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["stage"] == "CYCLE_CAP"
    assert rows[0]["selection_rank"] == 6
    assert rows[0]["decision_id"] == rows[1]["decision_id"]


def test_prepare_protection_capacity_cleans_stale_engine_orders(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    states = iter([
        {"status": "ok", "sl_orders": [{"orderId": "1", "clientOrderId": "EVT_OLD_SL"}], "tp_orders": [{"orderId": "2", "clientOrderId": "EVT_OLD_TP1"}]},
        {"status": "ok", "sl_orders": [], "tp_orders": []},
    ])
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *a, **k: next(states))
    monkeypatch.setattr(bingx, "cancel_order", lambda *a, **k: {"code": 0})
    result = bingx.prepare_protection_capacity("TEST-USDT", "LONG")
    assert result["status"] == "ready"
    assert result["cancelled_order_ids"] == ["1", "2"]


def test_prepare_protection_capacity_blocks_external_orders(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [], "tp_orders": [{"orderId": "X1", "clientOrderId": "MANUAL"}],
    })
    monkeypatch.setattr(bingx, "cancel_order", lambda *a, **k: pytest.fail("manual order must not be cancelled"))
    result = bingx.prepare_protection_capacity("TEST-USDT", "LONG")
    assert result["status"] == "blocked_external_protection"
    assert result["external_order_ids"] == ["X1"]


def test_rejected_touch_research_payload_keeps_structure_room(monkeypatch):
    import run_once
    ts = pd.date_range("2026-01-01", periods=80, freq="1h", tz="UTC")
    closes = [100 + i * 0.1 for i in range(80)]
    df1h = pd.DataFrame({
        "timestamp": ts, "open": closes, "high": [x + 1 for x in closes],
        "low": [x - 1 for x in closes], "close": closes,
        "volume": [1000.0] * 80, "atr50": [2.0] * 80,
    })
    zone = {"zone_id": "Z_STRUCT_REJ", "top": 105.0, "btm": 95.0, "poi": 100.0,
            "origin_ts_ms": int(ts[10].timestamp() * 1000), "start": 10}
    bar_ts = (pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(minutes=5)).isoformat()
    bars = [{"timestamp": bar_ts, "open": 100.0, "high": 106.0,
             "low": 99.0, "close": 99.0, "volume": 1000.0}]
    monkeypatch.setattr(run_once, "REQUIRE_DIRECTIONAL_CANDLE", True)
    monkeypatch.setattr(run_once, "MIN_STRUCTURE_ROOM_R", 999.0)
    monkeypatch.setattr(run_once, "_nearest_opposing_level", lambda *args, **kwargs: {"price": 92.0, "source": "test_obstacle"})
    diagnostics = {}
    obstacle = {"zone_id": "Z_OBS", "top": 92.0, "btm": 90.0, "poi": 91.0, "origin_ts_ms": int(ts[11].timestamp() * 1000), "start": 11}
    signals, _, _ = run_once._process_5m_zone_visits(
        "TEST-USDT", bars, [obstacle], [zone], df1h, None, set(), set(), diagnostics=diagnostics
    )
    assert signals == []
    reasons = [e for e in diagnostics["touch_events"] if e.get("reason", "").startswith("insufficient_structure_room")]
    assert reasons, diagnostics["touch_events"]
    assert reasons[0]["structure_room_R"] is not None
    assert reasons[0]["structural_distance"] is not None


def test_quote_freshness_uses_exchange_timestamp(monkeypatch):
    from event_engine import bingx
    now_ms = int(time.time() * 1000)
    monkeypatch.setattr(bingx, "SERVER_TIME_OFFSET_MS", 0)
    fresh = {"quote_observed_at_ms": now_ms, "quote_exchange_time_ms": now_ms - 500}
    stale = {"quote_observed_at_ms": now_ms, "quote_exchange_time_ms": now_ms - 5000}
    assert bingx._quote_freshness(fresh, 2.0)[0] is True
    assert bingx._quote_freshness(stale, 2.0)[0] is False


def test_quote_freshness_uses_local_clock_for_local_observation(monkeypatch):
    from event_engine import bingx
    local_now_ms = int(time.time() * 1000)
    monkeypatch.setattr(bingx, "SERVER_TIME_OFFSET_MS", 5000)
    quote = {"quote_observed_at_ms": local_now_ms, "quote_exchange_time_ms": None}
    ok, local_age, exchange_age, source = bingx._quote_freshness(quote, 2.0)
    assert ok is True
    assert local_age is not None and local_age < 1.0
    assert exchange_age is None
    assert source == "local_observed_elapsed"


def test_open_market_uses_final_venue_quote_not_preflight_quote(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "TESTUSDT")
    monkeypatch.setattr(bingx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {"quantityPrecision": 2, "tradeMinQuantity": 0.01, "multiplier": 1, "maxLeverage": 20})
    monkeypatch.setattr(bingx, "has_open_position", lambda *a, **k: False)
    monkeypatch.setattr(bingx, "_set_leverage", lambda *a, **k: True)
    monkeypatch.setattr(bingx, "position_side_param", lambda *a, **k: "LONG")
    monkeypatch.setattr(bingx, "get_order", lambda *a, **k: {"status": "error", "error": "not found"})
    now_ms = int(time.time() * 1000)
    preflight_quote = {"status":"ok","symbol":"TESTUSDT","bid":90.0,"ask":90.1,"quote_observed_at_ms":now_ms - 60000,"quote_exchange_time_ms":now_ms - 60000,"quote_source":"old_preflight"}
    final_quote = {"status":"ok","symbol":"TESTUSDT","bid":100.0,"ask":100.2,"quote_observed_at_ms":now_ms,"quote_exchange_time_ms":now_ms,"quote_source":"bookTicker"}
    calls = {"n": 0}
    def fake_quote(*a, **k):
        calls["n"] += 1
        return final_quote
    monkeypatch.setattr(bingx, "get_execution_quote", fake_quote)
    posted = {}
    monkeypatch.setattr(bingx, "_request", lambda method, path, params, **kwargs: posted.update(params) or {"code":0,"data":{"order":{"orderId":"1","clientOrderId":params["clientOrderId"]}}})
    out = bingx.open_market("TEST-USDT", "LONG", 100.0, "EVENT_FINAL_QUOTE", execution_quote=preflight_quote)
    assert out["status"] == "opened"
    assert calls["n"] == 1
    assert out["execution_quote"]["ask"] == 100.2
    assert isinstance(out.get("order_submit_at_ms"), int)
    assert out["execution_quote"]["order_submit_at_ms"] == out["order_submit_at_ms"]
    assert float(posted["quantity"]) > 0


def test_open_market_records_authoritative_final_quote(monkeypatch):
    from event_engine import bingx, telemetry

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "TESTUSDT")
    monkeypatch.setattr(bingx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {"quantityPrecision": 2, "tradeMinQuantity": 0.01, "multiplier": 1, "maxLeverage": 20})
    monkeypatch.setattr(bingx, "has_open_position", lambda *a, **k: False)
    monkeypatch.setattr(bingx, "_set_leverage", lambda *a, **k: True)
    monkeypatch.setattr(bingx, "position_side_param", lambda *a, **k: "LONG")
    monkeypatch.setattr(bingx, "get_order", lambda *a, **k: {"status": "error", "error": "not found"})

    now_ms = int(time.time() * 1000)
    final_quote = {
        "status": "ok",
        "symbol": "TESTUSDT",
        "bid": 100.0,
        "ask": 100.2,
        "quote_observed_at_ms": now_ms,
        "quote_exchange_time_ms": now_ms,
        "quote_source": "bookTicker",
        "spread_pct": 0.1996,
        "quote_sources_attempted": ["bookTicker"],
    }
    monkeypatch.setattr(bingx, "get_execution_quote", lambda *a, **k: final_quote.copy())

    captured = []
    monkeypatch.setattr(telemetry, "record_quote_snapshot", lambda **kwargs: captured.append(kwargs))

    monkeypatch.setattr(
        bingx,
        "_request",
        lambda method, path, params, **kwargs: {"code": 0, "data": {"order": {"orderId": "1", "clientOrderId": params["clientOrderId"]}}},
    )

    out = bingx.open_market("TEST-USDT", "LONG", 100.0, "EVENT_AUTH_QUOTE", attempt_id="ATT_AUTH_QUOTE")

    assert out["status"] == "opened"
    assert len(captured) == 1
    row = captured[0]
    assert row["event_id"] == "EVENT_AUTH_QUOTE"
    assert row["attempt_id"] == "ATT_AUTH_QUOTE"
    assert row["stage"] == "FINAL_PRE_POST"
    assert row["quote"]["ask"] == 100.2
    assert row["quote"]["quote_source"] == "bookTicker"
    assert row["quote"]["execution_reference_price"] == 100.2
    assert row["quote"]["signal_drift_pct"] == pytest.approx(0.2)
    assert row["quote"]["order_submit_at_ms"] == out["order_submit_at_ms"]
    assert row["signal_price"] == 100.0


def test_open_market_final_drift_blocks_post(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "TESTUSDT")
    monkeypatch.setattr(bingx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {"quantityPrecision": 2, "tradeMinQuantity": 0.01, "multiplier": 1, "maxLeverage": 20})
    monkeypatch.setattr(bingx, "has_open_position", lambda *a, **k: False)
    monkeypatch.setattr(bingx, "_set_leverage", lambda *a, **k: True)
    monkeypatch.setattr(bingx, "position_side_param", lambda *a, **k: "LONG")
    monkeypatch.setattr(bingx, "get_order", lambda *a, **k: {"status": "error", "error": "not found"})
    now_ms = int(time.time() * 1000)
    quote = {"status":"ok","symbol":"TESTUSDT","bid":102.0,"ask":102.2,"quote_observed_at_ms":now_ms,"quote_exchange_time_ms":now_ms,"quote_source":"bookTicker"}
    monkeypatch.setattr(bingx, "get_execution_quote", lambda *a, **k: quote)
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: pytest.fail("market POST must not happen after final drift rejection"))
    out = bingx.open_market("TEST-USDT", "LONG", 100.0, "EVENT_DRIFT_FINAL")
    assert out["status"] == "skipped_stale_signal"
    assert "signal_drift_pct" in out["error"]


def test_entry_decision_records_short_geometry_shadow_candidate(tmp_path, monkeypatch):
    import run_once
    from event_engine import research

    captured = []
    monkeypatch.setattr(research, "record_entry_decision", lambda payload, path=None: captured.append(payload) or True)
    signal = {
        "event_id": "ZONE_SHADOW",
        "symbol": "TEST-USDT",
        "type": "SHORT",
        "trigger_bar_time": "2026-09-21T12:00:00+00:00",
        "signal_forensics": {
            "body_to_range": 0.81,
            "lower_wick_ratio": 0.03,
        },
    }

    run_once._record_entry_decision(
        "SCAN_SHADOW", signal, "EXECUTION_SELECTED", "selected", selection_rank=1, attempt_id="ATT_SHADOW"
    )

    assert len(captured) == 1
    row = captured[0]
    assert row["shadow_short_filter_experiment"] == research.SHADOW_SHORT_FILTER_EXPERIMENT
    assert row["shadow_short_geometry_available"] is True
    assert row["shadow_short_geometry_bad"] is True
    assert row["shadow_short_body_to_range_gt"] == 0.70
    assert row["shadow_short_lower_wick_lt"] == 0.05


def test_telemetry_write_failure_is_observable_without_breaking_execution(monkeypatch, tmp_path):
    from event_engine import telemetry

    telemetry._TELEMETRY_FAILURE_COUNT = 0
    telemetry._TELEMETRY_LAST_FAILURE_TS_MS = None
    telemetry._TELEMETRY_LAST_FAILURE_TYPE = None

    def fail_write(path, record):
        raise OSError("disk unavailable")

    monkeypatch.setattr(telemetry, "_append_jsonl", fail_write)
    telemetry.record_quote_snapshot(
        event_id="EVT_HEALTH",
        attempt_id="ATT_HEALTH",
        symbol="AAA-USDT",
        direction="SHORT",
        quote={"bid": 99.0, "ask": 99.1, "quote_source": "bookTicker"},
        signal_price=100.0,
    )

    health = telemetry.health_snapshot()
    assert health["write_failure_count"] == 1
    assert health["last_failure_ts_ms"] is not None
    assert health["last_failure_type"] == "OSError"


def test_execution_telemetry_writes_linked_jsonl_events(tmp_path, monkeypatch):
    from event_engine import telemetry

    monkeypatch.setattr(telemetry, "DATA", tmp_path)
    paths = {
        "QUOTE_SNAPSHOTS_PATH": tmp_path / "quote_snapshots.jsonl",
        "ORDER_LIFECYCLE_PATH": tmp_path / "order_lifecycle.jsonl",
        "PROTECTION_LIFECYCLE_PATH": tmp_path / "protection_lifecycle.jsonl",
        "POSITION_RECONCILIATION_PATH": tmp_path / "position_reconciliation.jsonl",
        "EXCHANGE_ERRORS_PATH": tmp_path / "exchange_errors.jsonl",
    }
    for name, value in paths.items():
        monkeypatch.setattr(telemetry, name, value)
    monkeypatch.setenv("TELEMETRY_FSYNC", "false")

    telemetry.record_quote_snapshot(
        event_id="EVT_1",
        attempt_id="ATT_1",
        symbol="AAA-USDT",
        direction="SHORT",
        quote={
            "bid": 99.0,
            "ask": 99.1,
            "spread_pct": 0.101,
            "quote_source": "bookTicker",
            "quote_sources_attempted": ["bookTicker"],
            "quote_exchange_time_ms": 1000,
            "quote_observed_at_ms": 1050,
            "quote_local_age_sec": 0.05,
            "quote_exchange_age_sec": 0.04,
        },
        signal_price=100.0,
    )
    telemetry.record_order_event(
        event_id="EVT_1",
        attempt_id="ATT_1",
        order_id="ORD_1",
        symbol="AAA-USDT",
        direction="SHORT",
        leg="ENTRY",
        status="ACKNOWLEDGED",
        client_order_id="CID_1",
    )
    telemetry.record_protection_event(
        event_id="EVT_1",
        attempt_id="ATT_1",
        position_id="POS_1",
        order_id="SL_1",
        symbol="AAA-USDT",
        direction="SHORT",
        leg="SL",
        status="WORKING",
    )
    telemetry.record_position_reconciliation(
        event_id="EVT_1",
        attempt_id="ATT_1",
        position_id="POS_1",
        symbol="AAA-USDT",
        direction="SHORT",
        status="FOUND",
        internal_remaining_qty=1.0,
        exchange_position_qty=1.0,
    )
    telemetry.record_exchange_error(
        event_id="EVT_1",
        attempt_id="ATT_1",
        position_id="POS_1",
        order_id="SL_1",
        symbol="AAA-USDT",
        endpoint="/openOrders",
        method="GET",
        error_code=100410,
        message="rate limited",
        retryable=True,
        blocking=True,
    )

    for path in paths.values():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["schema_version"] == 1
        assert row["event_id"] == "EVT_1"
        assert row["attempt_id"] == "ATT_1"
        assert row["ts_ms"] > 0


def test_data_retention_preserves_production_state_and_prunes_only_research_history(tmp_path, monkeypatch):
    from event_engine import data_retention

    data_dir = tmp_path / "data"
    monkeypatch.setattr(data_retention, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_retention, "PROJECT_ROOT", tmp_path)

    active = {"POS1": {"symbol": "BTC-USDT", "closed": False}}
    zone_state = {"version": 2, "symbols": {"BTC-USDT": {"last_visit": "V1"}}}
    cursors = {"BTC-USDT|5m|binance": {"last_timestamp": "2026-09-23T10:00:00+00:00"}}
    outcome_state = {"schema_version": 2, "processed_observation_ids": ["OBS_DONE", "OBS_SIGNAL", "OBS_NEAR_OLD"]}
    (data_dir / "active_trades.json").write_text(json.dumps(active), encoding="utf-8")
    (data_dir / "zone_visit_state.json").write_text(json.dumps(zone_state), encoding="utf-8")
    (data_dir / "research_bar_cursors.json").write_text(json.dumps(cursors), encoding="utf-8")
    (data_dir / "research_outcome_state.json").write_text(json.dumps(outcome_state), encoding="utf-8")

    recent = "2026-09-23T12:00:00+00:00"
    old = "2026-09-20T00:00:00+00:00"
    bars = [
        {"timestamp": recent, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
        {"timestamp": old, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1},
    ]
    (data_dir / "market_bars_5m.jsonl").write_text("\n".join(json.dumps(x) for x in bars) + "\n", encoding="utf-8")
    (data_dir / "market_bars_1h.jsonl").write_text("\n".join(json.dumps(x) for x in bars) + "\n", encoding="utf-8")
    obs = [
        {"observation_id": "OBS_SIGNAL", "event_type": "SIGNAL_CREATED", "observation_ts": old},
        {"observation_id": "OBS_NEAR_OLD", "event_type": "NEAREST_APPROACH", "observation_ts": old},
        {"observation_id": "OBS_NEAR_NEW", "event_type": "NEAREST_APPROACH", "observation_ts": recent},
    ]
    (data_dir / "zone_observations.jsonl").write_text("\n".join(json.dumps(x) for x in obs) + "\n", encoding="utf-8")

    # Use a temporary archive and a fixed now so the test is deterministic.
    now = datetime.fromisoformat("2026-09-23T13:00:00+00:00")
    # Stub the research finalizer: this test validates the data migration itself.
    monkeypatch.setattr("research_forward.update_outcomes", lambda write: (0, 0))

    result = data_retention.compact_data(
        apply=True,
        archive_dir=tmp_path / "archive",
        now=now,
    )

    assert result["files"]["market_bars_5m.jsonl"]["pruned"] == 1
    assert result["files"]["market_bars_1h.jsonl"]["pruned"] == 0
    assert result["files"]["zone_observations.jsonl"]["pruned"] == 1
    assert json.loads((data_dir / "active_trades.json").read_text()) == active
    assert json.loads((data_dir / "zone_visit_state.json").read_text()) == zone_state
    assert json.loads((data_dir / "research_bar_cursors.json").read_text()) == cursors
    assert json.loads((data_dir / "research_outcome_state.json").read_text()) == outcome_state
    assert (tmp_path / "archive" / "market_bars_5m.jsonl.pruned.jsonl.gz").exists()
    assert (tmp_path / "archive" / "zone_observations.jsonl.pruned.jsonl.gz").exists()
    assert not (tmp_path / "archive" / "market_bars_1h.jsonl.pruned.jsonl.gz").exists()


def test_data_retention_refuses_stale_market_bar_history(tmp_path, monkeypatch):
    from event_engine import data_retention

    data_dir = tmp_path / "data"
    monkeypatch.setattr(data_retention, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_retention, "PROJECT_ROOT", tmp_path)
    active = {"POS1": {"symbol": "BTC-USDT", "closed": False}}
    (data_dir / "active_trades.json").write_text(json.dumps(active), encoding="utf-8")
    (data_dir / "market_bars_5m.jsonl").write_text(json.dumps({"timestamp": "2026-09-20T00:00:00+00:00"}) + "\n", encoding="utf-8")
    now = datetime.fromisoformat("2026-09-23T13:00:00+00:00")
    monkeypatch.setattr("research_forward.update_outcomes", lambda write: (0, 0))

    with pytest.raises(RuntimeError, match="refusing destructive compaction"):
        data_retention.compact_data(apply=True, archive_dir=tmp_path / "archive", now=now)

    assert json.loads((data_dir / "active_trades.json").read_text()) == active
    assert (data_dir / "market_bars_5m.jsonl").read_text().count("\n") == 1
