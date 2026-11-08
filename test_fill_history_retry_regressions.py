from __future__ import annotations

import pytest


def test_get_fill_orders_retries_109500_once_then_uses_confirmed_fill(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    monkeypatch.setattr(bingx, "credentials_available", lambda: True)
    sleeps: list[float] = []
    calls: list[tuple[tuple, dict]] = []
    responses = [
        {"code": 109500, "msg": "internal RPC timeout"},
        {
            "code": 0,
            "data": [{
                "tradeId": "T-109500-1",
                "orderId": "O-109500-1",
                "symbol": "AAA-USDT",
                "side": "SELL",
                "price": "101.25",
                "qty": "0.4",
                "realizedPnl": "1.75",
                "fee": "-0.02",
                "time": 1767225500000,
            }],
        },
    ]

    def fake_request(*args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(bingx, "_request", fake_request)
    monkeypatch.setattr(bingx.time, "sleep", lambda seconds: sleeps.append(seconds))

    fills = bingx.get_fill_orders("AAA-USDT", 100, 200, order_id="O-109500-1")

    assert len(calls) == 2
    assert sleeps == [pytest.approx(1.0)]
    assert calls[0][0] == calls[1][0]
    assert calls[0][1] == calls[1][1]
    assert calls[0][0][2]["tradingUnit"] == "COIN"
    assert calls[0][0][2]["orderId"] == "O-109500-1"
    assert len(fills) == 1
    assert fills[0]["trade_id"] == "T-109500-1"
    assert fills[0]["qty"] == pytest.approx(0.4)
    assert fills[0]["realized_pnl"] == pytest.approx(1.75)


def test_get_fill_orders_does_not_retry_non_109500_exchange_errors(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    monkeypatch.setattr(bingx, "credentials_available", lambda: True)
    calls = []
    monkeypatch.setattr(
        bingx,
        "_request",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"code": 109400, "msg": "bad parameters"},
    )
    monkeypatch.setattr(bingx.time, "sleep", lambda _seconds: pytest.fail("unexpected retry delay"))

    with pytest.raises(RuntimeError, match="109400"):
        bingx.get_fill_orders("AAA-USDT", 100, 200)

    assert len(calls) == 1


def test_research_account_snapshot_retries_109500_once_and_captures_fills(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "get_credentials", lambda: ("TEST_KEY", "TEST_SECRET"))
    monkeypatch.setattr(bingx.time, "sleep", lambda _seconds: None)
    fill_calls = 0

    def fake_request(method, path, params=None, signed=True, **kwargs):
        nonlocal fill_calls
        if path == bingx.BALANCE_PATH:
            return {"code": 0, "data": [{"asset": "USDT", "balance": "95.5", "equity": "95.5", "realisedProfit": "0", "availableMargin": "95.5", "usedMargin": "0"}]}
        if path == bingx.ALL_FILL_ORDERS_PATH:
            fill_calls += 1
            if fill_calls == 1:
                return {"code": 109500, "msg": "internal RPC timeout"}
            return {"code": 0, "data": [{
                "tradeId": "T-SNAPSHOT-1", "orderId": "O-SNAPSHOT-1",
                "symbol": "AAA-USDT", "side": "SELL", "price": "101.25",
                "qty": "0.4", "realizedPnl": "1.75", "fee": "-0.02",
                "time": 1767225500000,
            }]}
        if path in {bingx.INCOME_PATH, bingx.FORCE_ORDERS_PATH, bingx.POSITION_PATH}:
            return {"code": 0, "data": []}
        if path == bingx.COMMISSION_RATE_PATH:
            return {"code": 0, "data": {"commission": {"takerCommissionRate": "0.0005", "makerCommissionRate": "0.0002"}}}
        raise AssertionError(path)

    monkeypatch.setattr(bingx, "_request", fake_request)
    snapshot = bingx.fetch_research_account_snapshot()

    assert fill_calls == 2
    assert snapshot["recent_fill_status"] == "OK"
    assert snapshot["recent_fill_count"] == 1
    assert snapshot["recent_fill_source"] == "ALL_FILL_ORDERS"
    assert snapshot["recent_realized_pnl_total"] == pytest.approx(1.75)
    assert snapshot["recent_fill_fee_total"] == pytest.approx(-0.02)


def test_research_account_snapshot_keeps_fills_unknown_after_retry_fails(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "get_credentials", lambda: ("TEST_KEY", "TEST_SECRET"))
    monkeypatch.setattr(bingx.time, "sleep", lambda _seconds: None)
    fill_calls = 0

    def fake_request(method, path, params=None, signed=True, **kwargs):
        nonlocal fill_calls
        if path == bingx.BALANCE_PATH:
            return {"code": 0, "data": [{"asset": "USDT", "balance": "95.5", "equity": "95.5", "realisedProfit": "0", "availableMargin": "95.5", "usedMargin": "0"}]}
        if path == bingx.ALL_FILL_ORDERS_PATH:
            fill_calls += 1
            return {"code": 109500, "msg": "internal RPC timeout"}
        if path in {bingx.INCOME_PATH, bingx.FORCE_ORDERS_PATH, bingx.POSITION_PATH}:
            return {"code": 0, "data": []}
        if path == bingx.COMMISSION_RATE_PATH:
            return {"code": 0, "data": {"commission": {"takerCommissionRate": "0.0005", "makerCommissionRate": "0.0002"}}}
        raise AssertionError(path)

    monkeypatch.setattr(bingx, "_request", fake_request)
    snapshot = bingx.fetch_research_account_snapshot()

    assert fill_calls == 2
    assert snapshot["status"] == "partial"
    assert snapshot["recent_fill_status"] == "error"
    assert snapshot["recent_fill_source"] == "UNAVAILABLE"
    assert snapshot["recent_realized_pnl_total"] is None
