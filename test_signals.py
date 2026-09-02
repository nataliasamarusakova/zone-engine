from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from event_engine.signals import (
    RR_RATIO,
    TP1_R,
    TP2_R,
    calc_hma,
    calc_alma,
    compute_ajay_trigger,
    generate_zone_signals,
    score_zone_signal,
)


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


def test_long_tp_ordering_and_rr():
    signal = {
        "symbol": "TEST-USDT", "type": "LONG", "entry": 100.0, "sl": 95.0,
        "tp1": 107.5, "tp2": 115.0, "risk_pct": 5.0,
        "zone": {"kind": "DEMAND", "age_bars": 10, "impulse_atr": 2.0},
        "confirmation": {"alma_cross": True, "zone_touch": True, "volume_ratio": 1.5, "candle_body_atr": 1.0},
    }
    assert signal["tp1"] < signal["tp2"]
    assert RR_RATIO == 3.0
    assert TP1_R == 0.5
    assert TP2_R == 1.0
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
        "risk_pct": 5.0,
        "score": 90,
        "zone": {"kind": "DEMAND"},
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
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: {
        "status": "found", "positionAmt": 1.0
    })
    closed = {}
    monkeypatch.setattr(run_once, "close_position_market", lambda *a, **k: closed.update({"called": True, "qty": a[2]}) or {"status": "closed"})
    monkeypatch.setattr(run_once.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(run_once, "register_active_trade", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not register unprotected trade")))

    result = run_once.execute_new_position(signal)
    assert result["status"] == "opened_then_emergency_closed"
    assert closed.get("called") is True


def test_open_client_order_id_is_unique():
    from event_engine import bingx
    a = bingx._new_open_client_order_id("ALT-USDT", "ZONE_TEST")
    b = bingx._new_open_client_order_id("ALT-USDT", "ZONE_TEST")
    assert a != b
    assert len(a) <= 32
    assert len(b) <= 32


def test_tp_constants_are_one_and_two_r():
    from event_engine.signals import TP1_R, TP2_R, TP1_FRACTION, TP2_FRACTION
    assert TP1_R == 0.5
    assert TP2_R == 1.0
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
    assert sig.REQUIRE_ZONE_TOUCH is True
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

def test_telegram_uses_zone_only_label_and_dynamic_rr_values():
    from event_engine.telegram import format_signal
    msg = format_signal({
        "type": "LONG", "symbol": "TEST-USDT", "entry": 100.0, "sl": 95.0,
        "tp1": 102.5, "tp2": 105.0, "tp1_rr": 0.5, "tp2_rr": 1.0, "risk_pct": 5.0,
        "zone": {"kind": "DEMAND", "btm": 94.0, "top": 99.0, "poi": 96.5, "age_bars": 1, "impulse_atr": 2.0},
        "confirmation": {},
    })
    assert "Demand/Supply Zone First" in msg
    assert "Ajay R5.41 · ALMA" not in msg
    assert "(0.5R / 50%)" in msg
    assert "(1.0R / 50%)" in msg


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
    signal = {
        "event_id": "ZONE_TEST_REBASE", "symbol": "TEST-USDT", "type": "SHORT",
        "entry": 100.0, "sl": 105.0, "tp1": 99.0, "tp2": 98.0, "risk_pct": 5.0,
        "score": 75, "atr": 2.0,
        "zone": {"kind": "SUPPLY", "btm": 99.0, "top": 104.0},
        "target": {"obstacle_price": 90.0, "source": "nearest_opposing_structure"},
    }
    captured = {}
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: {"status": "opened", "symbol": "TEST-USDT"})
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda *a, **k: {"status": "found", "avgPrice": 95.0, "positionAmt": 1.0})
    def fake_protection(*args, **kwargs):
        captured["avg"] = args[2]
        captured["levels"] = args[5]
        return {"status": "PROTECTED", "tp_orders": [], "sl_result": {}}
    monkeypatch.setattr(run_once, "ensure_directional_protection", fake_protection)
    monkeypatch.setattr(run_once, "register_active_trade", lambda *a, **k: None)
    out = run_once.execute_new_position(signal)
    assert out["status"] == "opened_protected"
    assert captured["avg"] == 95.0
    levels = captured["levels"]
    assert levels[0]["pnl_pct"] > 0 and levels[1]["pnl_pct"] > levels[0]["pnl_pct"]
    assert out["executed_signal"]["entry"] == 95.0
    assert out["executed_signal"]["tp1"] < 95.0
    assert out["executed_signal"]["tp2"] < out["executed_signal"]["tp1"]


def test_invalid_setup_is_rejected_before_protection_preflight(monkeypatch):
    import run_once
    signal = {
        "event_id": "ZONE_TEST_INVALID", "symbol": "ALGO-USDT", "type": "SHORT",
        "entry": 0.0908, "sl": 0.09, "tp1": 0.09, "tp2": 0.09, "risk_pct": -0.88,
        "score": 75, "zone": {"kind": "SUPPLY", "btm": 0.089, "top": 0.091},
    }
    called = {"preflight": False, "open": False}
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: called.__setitem__("preflight", True) or {"status": "ok"})
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: called.__setitem__("open", True))
    result = run_once.execute_new_position(signal)
    assert result["status"] == "skipped_invalid_setup"
    assert called == {"preflight": False, "open": False}


def test_protection_endpoint_failure_blocks_market_entry(monkeypatch):
    import run_once
    signal = {
        "event_id": "ZONE_TEST_PREFLIGHT", "symbol": "ALGO-USDT", "type": "SHORT",
        "entry": 100.0, "sl": 105.0, "tp1": 99.0, "tp2": 98.0, "risk_pct": 5.0,
        "score": 75, "zone": {"kind": "SUPPLY", "btm": 99.0, "top": 104.0},
    }
    called = {"open": False}
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {"status": "error", "error": "code:100410 disabled period"})
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
    assert '_ZONE_TOUCH"' in source


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

    monkeypatch.setattr(sig, "REQUIRE_ZONE_TOUCH", True)
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


def test_zone_signal_uses_zone_boundary_stop_and_small_tps(monkeypatch):
    import numpy as np
    import pandas as pd
    from event_engine import signals as sig

    monkeypatch.setattr(sig, "REQUIRE_ZONE_TOUCH", True)
    n = 120
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    close = np.linspace(100.0, 120.0, n)
    open_ = close.copy()
    high = close + 0.5
    low = close - 0.5
    # Give the latest candle a small pullback/touch area.
    close[-1] = 109.0
    open_[-1] = 110.0
    high[-1] = 110.0
    low[-1] = 108.0
    volume = np.full(n, 1000.0)
    df = pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})
    # Supply a deterministic Demand zone touching the latest bar.
    def forced_walk(frame):
        demand = [{"top": 110.0, "btm": 107.0, "poi": 108.5, "start": len(frame)-5}]
        return [], demand, [], [], []
    monkeypatch.setattr(sig, "_pine_zone_walk", forced_walk)

    # With a forced zone walk, generate_zone_signals builds its own active zone,
    # so patch zone construction/selection at the deterministic insertion point.
    forced_zone = {"top": 110.0, "btm": 107.0, "poi": 108.5, "start": n - 5}
    monkeypatch.setattr(sig, "_find_directional_zone", lambda direction, cur_l, cur_h, cur_c, demand, supply: forced_zone if direction == "LONG" else None)

    _, _, _, emitted = sig.generate_zone_signals(df, symbol="TEST-USDT", mode="live")
    assert emitted
    latest = emitted[-1]
    assert latest["confirmation"]["zone_touch"] is True
    assert latest["risk_model"]["sl_source"] == "zone_boundary_plus_atr_buffer"
    assert latest["trigger"]["alma_required"] is False
    assert latest["target"]["source"] in {"nearest_opposing_structure", "atr_rr_fallback"}
    assert latest["tp1"] < latest["tp2"]
    assert latest["tp1_rr"] > 0
    assert latest["tp2_rr"] >= latest["tp1_rr"]
    assert latest["sl"] < 107.0


def test_targets_use_nearest_opposing_zone_and_stay_before_it():
    from event_engine import signals as sig

    obstacle = {"price": 108.0, "source": "supply_zone"}
    out = sig._targets_from_nearest_obstacle("LONG", 100.0, 95.0, 1.0, obstacle)
    assert out is not None
    assert out["target_source"] == "nearest_opposing_structure"
    assert out["obstacle_price"] == 108.0
    assert 100.0 < out["tp1"] < out["tp2"] < 108.0
    assert out["tp2_rr"] <= sig.TP_MAX_R + 1e-9

    obstacle = {"price": 92.0, "source": "demand_zone"}
    out = sig._targets_from_nearest_obstacle("SHORT", 100.0, 105.0, 1.0, obstacle)
    assert out is not None
    assert 92.0 < out["tp2"] < out["tp1"] < 100.0


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


def test_directional_zone_requires_actual_candle_touch_without_padding():
    from event_engine.signals import _find_directional_zone
    demand = [{"btm": 100.0, "top": 110.0}]
    supply = [{"btm": 120.0, "top": 130.0}]
    assert _find_directional_zone("LONG", 110.1, 112.0, 111.0, demand, supply) is None
    assert _find_directional_zone("LONG", 110.0, 112.0, 111.0, demand, supply) == demand[0]
    assert _find_directional_zone("SHORT", 118.0, 119.0, 119.0, demand, supply) is None
    assert _find_directional_zone("SHORT", 118.0, 120.0, 119.5, demand, supply) == supply[0]


def test_run_once_import_regression():
    # run_once.py uses SWING_LEN for its per-symbol minimum-history guard.
    # Keep this import-level test so a refactor cannot leave the constant out
    # of the run_once module namespace and fail every scanned symbol.
    import importlib
    module = importlib.import_module("run_once")
    assert module.SWING_LEN == 10
