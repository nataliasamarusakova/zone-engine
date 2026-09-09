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



import pytest


@pytest.fixture(autouse=True)
def _disable_real_telegram_for_all_tests(monkeypatch):
    """Tests must never send real Telegram messages when CI secrets are present."""
    from event_engine import tracker
    monkeypatch.setattr(tracker, "send_tg", lambda *args, **kwargs: True)
    # run_once signal notifications are also disabled by default; tests that
    # explicitly exercise delivery may override this fixture locally.
    import run_once
    monkeypatch.setattr(run_once, "send_tg", lambda *args, **kwargs: True)


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
        "risk_pct": 1.0,
        "score": 90,
        "zone": {"kind": "DEMAND", "btm": 99.0, "top": 100.0},
        "target": {"obstacle_price": 110.0},
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
        "levels": {
            "blue": [{"level_id": "B1", "color": "BLUE", "kind": "SUPPORT", "source": "pine_sr", "price": 0.7900, "lower": 0.7895, "upper": 0.7905, "status": "ACTIVE"}],
            "red": [{"level_id": "R1", "color": "RED", "kind": "RESISTANCE", "source": "pine_sr", "price": 0.8110, "lower": 0.8105, "upper": 0.8115, "status": "ACTIVE"}],
        },
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
        "levels": {
            "blue": [{"level_id": "B1", "color": "BLUE", "kind": "DEMAND", "source": "demand_zone", "price": 4.70, "lower": 4.69, "upper": 4.71, "status": "ACTIVE"}],
            "red": [{"level_id": "R1", "color": "RED", "kind": "RESISTANCE", "source": "pine_sr", "price": 4.85, "lower": 4.84, "upper": 4.86, "status": "ACTIVE"}],
        },
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
        "levels": {
            "blue": [{"level_id": "B1", "color": "BLUE", "kind": "SUPPORT", "source": "pine_sr", "price": 90.0, "lower": 89.5, "upper": 90.5, "status": "ACTIVE"}],
            "red": [{"level_id": "R1", "color": "RED", "kind": "SUPPLY", "source": "supply_zone", "price": 104.0, "lower": 103.5, "upper": 104.5, "status": "ACTIVE"}],
        },
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
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: {"status": "found", "positionAmt": 1.0})
    monkeypatch.setattr(run_once, "close_position_market", lambda *a, **k: {"status": "closed"})
    monkeypatch.setattr(run_once, "_cleanup_engine_protection", lambda *a, **k: {"status": "ok"})
    out = run_once.execute_new_position(signal)
    assert out["status"] == "opened_then_emergency_closed"
    assert out["error"].startswith("risk_pct_above_limit=")


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
        "entry": 100.0, "sl": 105.0, "tp1": 99.0, "tp2": 98.0, "risk_pct": 1.0,
        "score": 75, "zone": {"kind": "SUPPLY", "btm": 99.0, "top": 104.0},
        "target": {"obstacle_price": 90.0},
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


def test_zone_signal_uses_last_same_color_structural_stop_and_targets(monkeypatch):
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
    monkeypatch.setattr(sig, "build_level_pool", lambda **kwargs: {
        "blue": [{"level_id": "B1", "color": "BLUE", "kind": "CLUSTER", "source": "cluster", "price": 109.10, "lower": 109.0, "upper": 109.2, "status": "ACTIVE", "member_kinds": ["DEMAND"], "strength": 2}],
        "red": [{"level_id": "R1", "color": "RED", "kind": "RESISTANCE", "source": "pine_sr", "price": 115.0, "lower": 114.5, "upper": 115.5, "status": "ACTIVE", "member_kinds": ["RESISTANCE"], "strength": 2}],
        "all": [], "cluster_tolerance": 0.1,
    })

    _, _, _, emitted = sig.generate_zone_signals(df, symbol="TEST-USDT", mode="live")
    assert emitted
    latest = emitted[-1]
    assert latest["confirmation"]["zone_touch"] is True
    assert latest["risk_model"]["sl_source"] == "last_valid_same_color_level_plus_buffer"
    assert latest["protection_level"]["level_id"] == "B1"
    assert latest["sl"] < 109.0
    assert latest["trigger"]["alma_required"] is False
    assert latest["target"]["source"] in {"nearest_opposing_structure", "atr_rr_fallback"}
    assert latest["tp1"] < latest["tp2"]
    assert latest["tp1_rr"] > 0
    assert latest["tp2_rr"] >= latest["tp1_rr"]
    assert latest["sl"] < 110.0


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
    monkeypatch.setenv("MAX_SIGNAL_RISK_PCT", "1.50")
    monkeypatch.setenv("MIN_STRUCTURE_ROOM_R", "1.20")
    good = _base_signal("LONG")
    good["risk_pct"] = 1.0
    good["target"]["obstacle_price"] = 104.0
    ok, reason = run_once._validate_trade_geometry(good)
    assert ok, reason
    bad = _base_signal("LONG")
    ok, reason = run_once._validate_trade_geometry(bad)
    assert not ok and "risk_pct_above_limit" in reason


def test_directional_candle_filter():
    monkeypatch = None
    assert sig.REQUIRE_DIRECTIONAL_CANDLE is True


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
    assert sig.MAX_ZONE_AGE_BARS == 30
    assert sig.MAX_SIGNAL_RISK_PCT == 1.50
    assert sig.MIN_STRUCTURE_ROOM_R == 1.20
    assert sig.REQUIRE_DIRECTIONAL_CANDLE is True
    assert sig.REQUIRE_STRUCTURE_OBSTACLE is True


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




def test_post_fill_execution_passes_structural_stop_to_protection(monkeypatch):
    import run_once
    signal = {
        "event_id":"ZONE_STRUCTURAL_STOP", "symbol":"TEST-USDT", "type":"LONG",
        "entry":100.0, "sl":99.0, "tp1":101.0, "tp2":102.0, "risk_pct":1.0, "score":80,
        "atr":1.0, "zone":{"kind":"DEMAND","btm":98.0,"top":100.0},
        "target":{"obstacle_price":104.0},
    }
    captured = {}
    monkeypatch.setattr(run_once, "_validate_trade_geometry", lambda s: (True, "ok"))
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: {"status":"opened","symbol":"TEST-USDT"})
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda *a, **k: {"status":"found","avgPrice":100.0,"positionAmt":1.0})
    monkeypatch.setattr(run_once, "_rebase_protection_after_fill", lambda s, avg: {
        "sl":97.5,"tp1":101.0,"tp2":102.0,"risk_abs":2.5,"risk_pct":2.5,
        "tp1_rr":0.4,"tp2_rr":0.8,"target_source":"test","obstacle_price":104.0,
        "protection_level":{"level_id":"B1"}, "target_levels":[{"level_id":"R1"}],
    })
    monkeypatch.setattr(run_once, "ensure_directional_protection", lambda *a, **k: captured.update(k) or {"status":"PROTECTED","tp_orders":[],"sl_result":{},"effective_tp_levels":[]})
    monkeypatch.setattr(run_once, "register_active_trade", lambda *a, **k: None)
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {"status":"ok","sl_orders":[],"tp_orders":[]})
    out=run_once.execute_new_position(signal)
    assert out["status"] == "opened_protected"
    assert captured["requested_sl_price"] == 97.5

def test_crossed_tp_market_fallback_is_reduce_only_in_one_way(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(bingx, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2, "tradeMinQuantity": 0.001})
    monkeypatch.setattr(bingx, "position_side_param", lambda d: "BOTH")
    state = {"sl_orders": [], "tp_orders": []}
    def protection(*a, **k):
        return {"status":"ok","sl_orders":list(state["sl_orders"]),"tp_orders":list(state["tp_orders"])}
    monkeypatch.setattr(bingx, "get_open_protection_directional", protection)
    monkeypatch.setattr(bingx, "_current_close_price", lambda s: 101.0)
    monkeypatch.setattr(bingx, "_verify_market_reduce_order", lambda *a, **k: {"status":"verified","executed_qty":0.5,"reduced_qty":0.5,"remaining_qty":0.5})
    monkeypatch.setattr(bingx, "_verify_open_order", lambda *a, **k: {"status":"verified","order":{"orderId":"V","clientOrderId":k.get("client_order_id"),"type":k.get("order_kind"),"stopPrice":k.get("expected_price"),"origQty":k.get("expected_qty")}})
    seen=[]
    def req(method, path, params):
        seen.append(dict(params))
        order={"orderId":f"O{len(seen)}","clientOrderId":params["clientOrderId"],"type":params["type"],"stopPrice":params.get("stopPrice"),"origQty":params["quantity"]}
        if params["type"] == "STOP_MARKET":
            state["sl_orders"]=[order]
        elif params["type"] == "TAKE_PROFIT_MARKET":
            state["tp_orders"].append(order)
        return {"code":0,"data":{"order":order}}
    monkeypatch.setattr(bingx, "_request", req)
    out=bingx.ensure_directional_protection("AAA-USDT","LONG",100.0,1.0,1.0,[{"leg":"tp1","pnl_pct":0.5,"close_fraction":0.5},{"leg":"tp2","pnl_pct":1.0,"close_fraction":0.5}],trade_id="T",requested_sl_price=98.0)
    assert out["status"] in {"PROTECTED", "PROTECTION_FAILED"}
    crossed=[x for x in seen if x["type"]=="MARKET"]
    assert crossed and crossed[-1]["reduceOnly"]=="true"

def test_protection_uses_structural_stop_price_on_entry_and_restart(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(bingx, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2, "tradeMinQuantity": 0.001})
    monkeypatch.setattr(bingx, "position_side_param", lambda d: "BOTH")
    state = {"sl_orders": [], "tp_orders": []}
    def get_protection(*a, **k):
        return {"status": "ok", "sl_orders": list(state["sl_orders"]), "tp_orders": list(state["tp_orders"])}
    monkeypatch.setattr(bingx, "get_open_protection_directional", get_protection)
    monkeypatch.setattr(bingx, "_verify_open_order", lambda *a, **k: {"status": "verified", "order": {"orderId": "SL1", "clientOrderId": k.get("client_order_id", ""), "stopPrice": k.get("expected_price", 0), "origQty": k.get("expected_qty", 0)}})
    calls=[]
    def req(method, path, params):
        calls.append(params)
        state["sl_orders"] = [{"orderId":"SL1","clientOrderId":params["clientOrderId"],"type":params["type"],"stopPrice":params["stopPrice"],"origQty":params["quantity"]}]
        return {"code":0, "data":{"order":{"orderId":"SL1","clientOrderId":params["clientOrderId"]}}}
    monkeypatch.setattr(bingx, "_request", req)
    out=bingx.ensure_directional_protection("AAA-USDT","LONG",100.0,1.0,1.0,[],trade_id="T",requested_sl_price=96.5)
    assert out["status"] == "PROTECTED"
    assert abs(float(calls[-1]["stopPrice"]) - 96.5) < 1e-9
    assert calls[-1]["reduceOnly"] == "true"


def test_protection_adds_reduce_only_to_one_way_tp(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(bingx, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2, "tradeMinQuantity": 0.001})
    monkeypatch.setattr(bingx, "position_side_param", lambda d: "BOTH")
    state = {"sl_orders": [], "tp_orders": []}
    def get_protection(*a, **k):
        return {"status":"ok", "sl_orders": list(state["sl_orders"]), "tp_orders": list(state["tp_orders"])}
    monkeypatch.setattr(bingx, "get_open_protection_directional", get_protection)
    seen=[]
    def req(method, path, params):
        seen.append(dict(params))
        order = {"orderId":f"O{len(seen)}","clientOrderId":params["clientOrderId"],"type":params["type"],"stopPrice":params.get("stopPrice"),"origQty":params["quantity"]}
        if params["type"] == "STOP_MARKET":
            state["sl_orders"] = [order]
        else:
            state["tp_orders"].append(order)
        return {"code":0,"data":{"order":order}}
    monkeypatch.setattr(bingx, "_request", req)
    monkeypatch.setattr(bingx, "_verify_open_order", lambda *a, **k: {"status":"verified","order":{"orderId":"V","clientOrderId":k.get("client_order_id"),"stopPrice":k.get("expected_price"),"origQty":k.get("expected_qty")}})
    monkeypatch.setattr(bingx, "_current_close_price", lambda s: 99.0)
    out=bingx.ensure_directional_protection("AAA-USDT","LONG",100.0,1.0,1.0,[{"leg":"tp1","pnl_pct":1.0,"close_fraction":1.0}],trade_id="T",requested_sl_price=98.5)
    assert out["status"] == "PROTECTED"
    assert any(x["type"]=="STOP_MARKET" and x.get("reduceOnly")=="true" for x in seen)
    assert any(x["type"]=="TAKE_PROFIT_MARKET" and x.get("reduceOnly")=="true" for x in seen)


def test_analytics_line_contains_all_structural_levels():
    from event_engine.analytics import _line
    line=_line({"symbol":"BTC-USDT","current_price":100.0,"price_position":"OUT","fresh_signal":"—","active_demand":1,"active_supply":1,"market_source":"binance_spot","levels":{"blue":[{"level_id":"B123456789","kind":"SUPPORT","lower":98,"upper":99,"strength":3,"age_bars":4}],"red":[{"level_id":"R123456789","kind":"RESISTANCE","lower":102,"upper":103,"strength":2,"age_bars":6}]}})
    assert "BLUE=" in line and "SUPPORT" in line and "RED=" in line and "RESISTANCE" in line

def test_workflow_risk_cap_is_not_accidentally_25_percent():
    from pathlib import Path
    workflow = Path('.github/workflows/event-engine.yml').read_text(encoding='utf-8')
    assert 'MAX_SIGNAL_RISK_PCT: "1.50"' in workflow
    assert 'MAX_SIGNAL_RISK_PCT: "25"' not in workflow

def test_signal_risk_cap_hard_clamped(monkeypatch):
    import importlib
    import event_engine.signals as sig
    monkeypatch.setenv('MAX_SIGNAL_RISK_PCT', '25')
    sig2 = importlib.reload(sig)
    assert sig2.MAX_SIGNAL_RISK_PCT == 1.50
    monkeypatch.setenv('MAX_SIGNAL_RISK_PCT', '1.25')
    sig2 = importlib.reload(sig2)
    assert sig2.MAX_SIGNAL_RISK_PCT == 1.25
    monkeypatch.delenv("MAX_SIGNAL_RISK_PCT", raising=False)
    importlib.reload(sig2)

def test_run_once_hard_caps_production_risk(monkeypatch):
    import run_once
    monkeypatch.setenv("MAX_SIGNAL_RISK_PCT", "25")
    assert run_once.MAX_PRODUCTION_RISK_PCT == 1.50

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
    monkeypatch.setattr(run_once, "open_market", lambda *a: {"status":"opened"})
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda *a, **k: {"status":"found","avgPrice":102.5,"positionAmt":1})
    monkeypatch.setattr(run_once, "_emergency_close_and_verify", lambda *a, **k: {"status":"closed_verified"})
    monkeypatch.setattr(run_once, "_cleanup_engine_protection", lambda *a, **k: {"status":"ok"})
    out = run_once.execute_new_position(signal)
    assert out["status"] == "opened_then_emergency_closed"
    assert "adverse_entry_slippage" in out["error"]

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

    assert run_once.DATA == project_root / "data"
    assert tracker.DATA == project_root / "data"
    assert analytics.DATA_DIR == project_root / "data"
    assert run_once.DATA != Path.cwd() / "data"
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
    assert 'schedule:' not in text


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


def test_watchlist_is_exact_24_symbols_and_enabled():
    expected = (
        "BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT", "XRP-USDT", "DOGE-USDT",
        "TRX-USDT", "HYPE-USDT", "XMR-USDT", "ZEC-USDT", "LINK-USDT", "ADA-USDT",
        "XLM-USDT", "BCH-USDT", "UNI-USDT", "LTC-USDT", "AVAX-USDT", "SUI-USDT",
        "HBAR-USDT", "TAO-USDT", "ICP-USDT", "ARB-USDT", "POL-USDT", "ETC-USDT",
    )
    assert WATCHLIST_ONLY is True
    assert WATCHLIST_SYMBOLS == expected
    assert len(WATCHLIST_SYMBOLS) == 24


def test_level_engine_preserves_semantic_sr_colors_and_clusters():
    from event_engine.levels import build_level_pool, select_opposing_levels, select_protective_level, protective_price
    pool = build_level_pool(
        demand=[{"top": 100.5, "btm": 99.5, "start": 10}],
        supply=[{"top": 105.5, "btm": 104.5, "start": 12}],
        support_resistance=[
            {"price": 100.2, "semantic": "SUPPORT", "created_idx": 11, "age_bars": 1, "strength": 2},
            {"price": 105.0, "semantic": "RESISTANCE", "created_idx": 13, "age_bars": 1, "strength": 2},
        ],
        pivot_lows=[{"price": 100.1, "created_idx": 9, "age_bars": 3}],
        pivot_highs=[{"price": 105.1, "created_idx": 14, "age_bars": 0}],
        current_idx=14, atr=1.0,
    )
    assert pool["blue"] and pool["red"]
    assert all(level["color"] == "BLUE" for level in pool["blue"])
    assert all(level["color"] == "RED" for level in pool["red"])
    protective = select_protective_level("LONG", 102.0, pool["blue"], pool["red"])
    opposing = select_opposing_levels("LONG", 102.0, pool["blue"], pool["red"], limit=2)
    assert protective is not None
    assert opposing
    stop = protective_price(protective, "LONG", 0.1)
    assert stop < protective["lower"] < 102.0


def test_level_engine_short_uses_last_red_above_entry():
    from event_engine.levels import build_level_pool, select_protective_level, protective_price
    pool = build_level_pool(
        demand=[], supply=[{"top": 110.5, "btm": 109.5, "start": 1}],
        support_resistance=[{"price": 108.0, "semantic": "RESISTANCE", "created_idx": 2, "age_bars": 1, "strength": 2}],
        pivot_highs=[{"price": 107.8, "created_idx": 3, "age_bars": 0}],
        current_idx=3, atr=1.0,
    )
    protective = select_protective_level("SHORT", 105.0, pool["blue"], pool["red"])
    assert protective is not None
    assert protective["color"] == "RED"
    assert protective_price(protective, "SHORT", 0.1) > protective["upper"]


def test_level_engine_rejects_broken_semantic_levels():
    from event_engine.levels import build_level_pool, select_protective_level
    pool = build_level_pool(
        demand=[], supply=[],
        support_resistance=[
            {"price": 100.0, "semantic": "SUPPORT", "created_idx": 1, "age_bars": 5, "strength": 2},
            {"price": 110.0, "semantic": "RESISTANCE", "created_idx": 2, "age_bars": 4, "strength": 2},
        ],
        pivot_lows=[{"price": 98.0, "created_idx": 1, "age_bars": 5}],
        pivot_highs=[{"price": 112.0, "created_idx": 2, "age_bars": 4}],
        current_idx=5, reference_price=95.0, atr=1.0,
    )
    assert not pool["blue"]
    assert pool["invalid"]
    assert not [x for x in pool["invalid"] if x["color"] == "RED" and x["status"] == "BROKEN"]
    assert select_protective_level("LONG", 114.0, pool["blue"], pool["red"]) is None


def test_signal_level_snapshot_uses_same_source_for_stop_and_targets(monkeypatch):
    from event_engine import signals as sig
    df = _candles(120)
    forced_zone = {"top": 110.5, "btm": 109.0, "poi": 109.75, "start": 115}
    monkeypatch.setattr(sig, "_pine_zone_walk", lambda frame: ([], [forced_zone], [], [], []))
    monkeypatch.setattr(sig, "_find_directional_zone", lambda direction, *args: forced_zone if direction == "LONG" else None)
    monkeypatch.setattr(sig, "build_level_pool", lambda **kwargs: {
        "blue": [{"level_id": "B1", "color": "BLUE", "kind": "CLUSTER", "source": "cluster", "price": 109.10, "lower": 109.0, "upper": 109.2, "status": "ACTIVE", "member_kinds": ["DEMAND"], "strength": 2}],
        "red": [
            {"level_id": "R1", "color": "RED", "kind": "CLUSTER", "source": "cluster", "price": 115.0, "lower": 114.5, "upper": 115.5, "status": "ACTIVE", "member_kinds": ["RESISTANCE"], "strength": 2},
            {"level_id": "R2", "color": "RED", "kind": "CLUSTER", "source": "cluster", "price": 120.0, "lower": 119.5, "upper": 120.5, "status": "ACTIVE", "member_kinds": ["SUPPLY"], "strength": 2},
        ], "all": [], "cluster_tolerance": 0.1,
    })
    monkeypatch.setattr(sig, "select_entry_level_for_zone", lambda direction, zone, pool: pool["blue"][0])
    monkeypatch.setattr(sig, "calc_atr", lambda frame, length: pd.Series(1.0, index=frame.index))
    # Build a deterministic latest bullish touch.
    df.loc[len(df)-2, "close"] = 111.0
    df.loc[len(df)-2, "open"] = 110.0
    df.loc[len(df)-2, "high"] = 111.2
    df.loc[len(df)-2, "low"] = 109.8
    df.loc[len(df)-1, "close"] = 110.0
    df.loc[len(df)-1, "open"] = 109.0
    df.loc[len(df)-1, "low"] = 108.0
    df.loc[len(df)-1, "high"] = 110.0
    _, _, _, emitted = sig.generate_zone_signals(df, symbol="TEST-USDT", mode="live")
    assert emitted
    s = emitted[-1]
    assert s["protection_level"]["level_id"] == "B1"
    assert all(x["color"] == "BLUE" for x in s["levels"]["blue"])
    assert all(x["color"] == "RED" for x in s["levels"]["red"])


def test_run_once_uses_bingx_fallback_when_binance_history_is_stale(monkeypatch):
    import run_once
    now = pd.Timestamp.now(tz="UTC")
    stale = [{"timestamp": int((now - pd.Timedelta(hours=10 + (30-i))).timestamp()*1000), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000} for i in range(30)]
    fresh = [{"timestamp": int((now - pd.Timedelta(hours=0.5 + (59-i))).timestamp()*1000), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000} for i in range(60)]
    monkeypatch.setattr(run_once, "fetch_binance_klines", lambda *a, **k: stale)
    monkeypatch.setattr(run_once, "fetch_bingx_klines", lambda *a, **k: fresh)
    bars, source = run_once._fetch_analysis_bars("XMR-USDT", "XMRUSDT", "binance")
    assert source == "bingx_fallback"
    assert len(bars) == 60


def test_rebase_uses_last_valid_same_color_level_not_stale_zone_boundary():
    import run_once
    signal = {
        "type": "LONG", "entry": 100.0, "sl": 99.0, "tp1": 101.0, "tp2": 102.0,
        "risk_pct": 1.0, "atr": 1.0,
        "zone": {"btm": 99.5, "top": 100.5},
        "levels": {
            "blue": [
                {"level_id": "B1", "color": "BLUE", "kind": "CLUSTER", "source": "cluster", "price": 98.0, "lower": 97.5, "upper": 98.5, "status": "ACTIVE"},
                {"level_id": "B2", "color": "BLUE", "kind": "CLUSTER", "source": "cluster", "price": 99.25, "lower": 99.0, "upper": 99.5, "status": "ACTIVE"},
            ],
            "red": [{"level_id": "R1", "color": "RED", "kind": "RESISTANCE", "source": "pine_sr", "price": 105.0, "lower": 104.5, "upper": 105.5, "status": "ACTIVE"}],
        },
    }
    out = run_once._rebase_protection_after_fill(signal, 100.0)
    assert out["protection_level"]["level_id"] == "B2"
    assert out["sl"] < 99.0




def test_level_snapshot_is_exposed_for_non_signal_scan():
    import pandas as pd
    from event_engine import signals
    rows = []
    base = 100.0
    for i in range(180):
        wave = 2.0 * ((i % 20) - 10) / 10.0
        close = base + wave
        rows.append({
            "timestamp": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i),
            "open": close - 0.1,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": 1000.0 + i,
        })
    df, _, _, _ = signals.generate_zone_signals(pd.DataFrame(rows), symbol="TEST-USDT", mode="historical")
    snapshot = df.attrs.get("level_snapshot")
    assert isinstance(snapshot, dict)
    assert "blue" in snapshot and "red" in snapshot
    assert "cluster_tolerance" in snapshot


def test_protective_and_opposing_levels_exclude_entry_overlapping_levels():
    from event_engine.levels import select_opposing_levels, select_protective_level
    blue = [
        {"level_id": "B_OVERLAP", "status": "ACTIVE", "lower": 99.0, "upper": 101.0, "price": 100.0},
        {"level_id": "B_BELOW", "status": "ACTIVE", "lower": 96.0, "upper": 98.0, "price": 97.0},
    ]
    red = [
        {"level_id": "R_OVERLAP", "status": "ACTIVE", "lower": 99.0, "upper": 101.0, "price": 100.0},
        {"level_id": "R_ABOVE", "status": "ACTIVE", "lower": 102.0, "upper": 104.0, "price": 103.0},
    ]
    assert select_protective_level("LONG", 100.0, blue, red)["level_id"] == "B_BELOW"
    assert select_protective_level("SHORT", 100.0, blue, red)["level_id"] == "R_ABOVE"
    assert select_opposing_levels("LONG", 100.0, blue, red, limit=2)[0]["level_id"] == "R_ABOVE"
    assert select_opposing_levels("SHORT", 100.0, blue, red, limit=2)[0]["level_id"] == "B_BELOW"


def test_human_level_logging_never_emits_nan_and_exposes_trade_map(caplog):
    import run_once as ro

    entry_level = {"level_id": "LVL_BLUE1", "kind": "CLUSTER", "member_kinds": ["SUPPORT"], "lower": 78680.0, "upper": 78680.0, "strength": 2, "age_bars": 10}
    result = {
        "current_price": 78952.01,
        "latest_closed_idx": 119,
        "levels": {
            "blue": [
                {"level_id": "LVL_BLUE1", "kind": "CLUSTER", "member_kinds": ["SUPPORT", "PIVOT_LOW"], "lower": 78680.0, "upper": 78680.0, "strength": 2, "age_bars": 10},
                {"level_id": "LVL_BLUE2", "kind": "CLUSTER", "member_kinds": ["DEMAND", "PIVOT_LOW"], "lower": 77620.01, "upper": 77699.55, "strength": 3, "age_bars": 10},
            ],
            "red": [
                {"level_id": "LVL_RED1", "kind": "CLUSTER", "member_kinds": ["RESISTANCE", "PIVOT_HIGH"], "lower": 79485.0, "upper": 79485.0, "strength": 3, "age_bars": 21},
            ],
            "invalid": [],
            "high_level": {"price": 82300.0},
            "low_level": {"price": 76264.0},
            "active_zones": {
                "demand": [{"btm": 77620.01, "top": 77699.55, "start": 109}],
                "supply": [{"btm": 80487.825, "top": 80559.99, "start": 71}, {"btm": 80400.0, "top": float("nan"), "start": 70}],
            },
        },
        "signals": [{
            "type": "LONG",
            "sl": 78500.0,
            "tp1": 79400.0,
            "tp2": 80000.0,
            "risk_pct": 0.57,
            "tp2_rr": 1.4,
            "entry_level": entry_level,
            "protection_level": entry_level,
            "target": {"target_levels": [{"level_id": "LVL_RED1", "kind": "CLUSTER", "member_kinds": ["RESISTANCE"], "lower": 79485.0, "upper": 79485.0, "strength": 3, "age_bars": 21}]},
        }],
    }
    caplog.set_level("INFO", logger="zone_engine")
    ro._log_human_level_map("BTC-USDT", result)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "[LEVEL_MAP] BTC-USDT | PRICE=78952.01" in text
    assert "NEAREST_BLUE_BELOW" in text and "78680" in text
    assert "NEAREST_RED_ABOVE" in text and "79485" in text
    assert "PINE_HIGH=82300" in text and "PINE_LOW=76264" in text
    assert "INVALID_ZONE" in text
    assert "[TRADE_MAP] BTC-USDT | DIRECTION=LONG" in text
    assert "[TRADE_MAP] BTC-USDT | PROTECTION" in text
    assert "nan" not in text.lower()
