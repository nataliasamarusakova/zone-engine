from __future__ import annotations

import numpy as np
import pandas as pd

from event_engine.signals import (
    RR_RATIO,
    TP1_R,
    TP2_R,
    calc_hma,
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
        assert s["sl"] > 0
        assert s["tp1"] > 0
        assert s["tp2"] > 0
        assert s["risk_pct"] > 0
        assert s["event_id"].startswith("ZONE_")
        assert s["zone"]["kind"] in {"DEMAND", "SUPPLY"}
        assert s["confirmation"]["hma_cross"] is True


def test_long_tp_ordering_and_rr():
    signal = {
        "symbol": "TEST-USDT", "type": "LONG", "entry": 100.0, "sl": 95.0,
        "tp1": 107.5, "tp2": 115.0, "risk_pct": 5.0,
        "zone": {"kind": "DEMAND", "age_bars": 10, "impulse_atr": 2.0},
        "confirmation": {"hma_cross": True, "volume_ratio": 1.5, "candle_body_atr": 1.0},
    }
    assert signal["tp1"] < signal["tp2"]
    assert RR_RATIO == 3.0
    assert TP1_R == 1.0
    assert TP2_R == 2.0
    assert score_zone_signal(signal) >= 70


def test_short_tp_ordering():
    signal = {
        "symbol": "TEST-USDT", "type": "SHORT", "entry": 100.0, "sl": 105.0,
        "tp1": 92.5, "tp2": 85.0, "risk_pct": 5.0,
        "zone": {"kind": "SUPPLY", "age_bars": 20, "impulse_atr": 1.5},
        "confirmation": {"hma_cross": True, "volume_ratio": 1.2, "candle_body_atr": 0.8},
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
    assert TP1_R == 1.0
    assert TP2_R == 2.0
    assert TP1_FRACTION == 0.50
    assert TP2_FRACTION == 0.50
