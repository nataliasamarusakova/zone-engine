from __future__ import annotations

from event_engine import bingx


def test_existing_verified_protection_uses_one_open_orders_snapshot(monkeypatch):
    """Already protected positions should not do a redundant second openOrders call."""
    sl = {"orderId": "SL1", "clientOrderId": "EVT_TEST_SL", "type": "STOP_MARKET", "stopPrice": "90.0", "origQty": "1.0"}
    tp1 = {"orderId": "TP1", "clientOrderId": "EVT_TEST_TP1", "type": "TAKE_PROFIT_MARKET", "stopPrice": "103.0", "origQty": "0.5"}
    tp2 = {"orderId": "TP2", "clientOrderId": "EVT_TEST_TP2", "type": "TAKE_PROFIT_MARKET", "stopPrice": "106.0", "origQty": "0.5"}
    calls = []

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {"quantityPrecision": 3, "pricePrecision": 1, "tradeMinQuantity": 0.001})
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *args, **kwargs: (calls.append((args, kwargs)) or {"status": "ok", "sl_orders": [sl], "tp_orders": [tp1, tp2]}))
    monkeypatch.setattr(bingx, "_request", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no order writes expected")))

    result = bingx.ensure_directional_protection(
        "AAA-USDT", "LONG", 100.0, 1.0, 10.0,
        [
            {"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 0.5},
            {"leg": "tp2", "pnl_pct": 6.0, "close_fraction": 0.5},
        ],
        trade_id="EVT_TEST",
    )

    assert result["status"] == "PROTECTED"
    assert len(calls) == 1
    assert result["sl_result"]["order_id"] == "SL1"


def test_deferred_research_bar_prefetch_preserves_symbol_mapping(monkeypatch):
    import run_once

    calls = []

    def fake_binance(symbol, interval, *, limit, retryable):
        calls.append(("binance", symbol, interval, limit, retryable))
        return [{"symbol": symbol, "timestamp": 1}]

    def fake_bingx(symbol, interval, *, limit, retryable):
        calls.append(("bingx", symbol, interval, limit, retryable))
        return [{"symbol": symbol, "timestamp": 2}]

    monkeypatch.setattr(run_once, "fetch_binance_klines", fake_binance)
    monkeypatch.setattr(run_once, "fetch_bingx_klines", fake_bingx)
    payloads = [
        {"symbol": "AAA-USDT", "provider": "binance", "persist_bars_for_forward": True, "bars_5m": []},
        {"symbol": "BBB-USDT", "provider": "bingx", "persist_bars_for_forward": True, "bars_5m": []},
        {"symbol": "CCC-USDT", "provider": "binance", "persist_bars_for_forward": True, "bars_5m": [{"timestamp": 3}]},
        {"symbol": "DDD-USDT", "provider": "binance", "persist_bars_for_forward": False, "bars_5m": []},
    ]
    prefetch = run_once._prefetch_missing_research_5m_bars(
        payloads,
        {"AAA-USDT": {"binance_symbol": "AAAUSDT"}},
        limit=24,
        max_workers=2,
    )

    assert prefetch[id(payloads[0])] == [{"symbol": "AAAUSDT", "timestamp": 1}]
    assert prefetch[id(payloads[1])] == [{"symbol": "BBB-USDT", "timestamp": 2}]
    assert prefetch[id(payloads[2])] == [{"symbol": "CCCUSDT", "timestamp": 1}]
    assert prefetch[id(payloads[3])] == [{"symbol": "DDDUSDT", "timestamp": 1}]
    assert len(calls) == 4
    assert {call[0] for call in calls} == {"binance", "bingx"}


def test_deferred_research_prefetch_uses_bounded_four_worker_pool(monkeypatch):
    import threading
    import time
    import run_once

    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_fetch(symbol, interval, *, limit, retryable):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            return [{"symbol": symbol, "timestamp": 1}]
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(run_once, "fetch_binance_klines", fake_fetch)
    payloads = [
        {"symbol": f"S{i}-USDT", "provider": "binance", "persist_bars_for_forward": True, "bars_5m": []}
        for i in range(8)
    ]
    result = run_once._prefetch_missing_research_5m_bars(
        payloads, {}, limit=24, max_workers=8,
    )

    assert len(result) == len(payloads)
    assert max_active == 4
    assert max_active <= 4
