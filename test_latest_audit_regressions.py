"""Regression tests for four confirmed audit findings in ZoneEngine 5.5.1."""
from __future__ import annotations

import threading

import pytest


def test_weighted_rr_uses_exchange_rounded_effective_tp_weights():
    from event_engine import tracker

    trade = {
        "initial_qty": 3.0,
        "remaining_qty": 3.0,
        "planned_risk_pct": 10.0,
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "tp_levels": [
            {"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 0.5},
            {"leg": "tp2", "pnl_pct": 6.0, "close_fraction": 0.5},
        ],
        "effective_tp_levels": [
            {"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 2.0 / 3.0, "qty": 2.0},
            {"leg": "tp2", "pnl_pct": 6.0, "close_fraction": 1.0 / 3.0, "qty": 1.0},
        ],
        "hit_legs": [],
        "completed_tp_legs": [],
    }

    effective, realized, remaining = tracker._weighted_rr_snapshot(trade)
    assert realized == pytest.approx(0.0)
    assert effective == pytest.approx(0.4)
    assert remaining == pytest.approx(0.4)


def test_weighted_rr_uses_nested_effective_levels_when_top_level_is_legacy_default():
    from event_engine import tracker

    planned = [
        {"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 0.5},
        {"leg": "tp2", "pnl_pct": 6.0, "close_fraction": 0.5},
    ]
    trade = {
        "initial_qty": 3.0,
        "remaining_qty": 3.0,
        "planned_risk_pct": 10.0,
        "tp_levels": planned,
        # Legacy load migration fills this with planned TP levels.
        "effective_tp_levels": planned,
        "setup": {"protection_result": {"effective_tp_levels": [
            {"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 2 / 3, "qty": 2.0},
            {"leg": "tp2", "pnl_pct": 6.0, "close_fraction": 1 / 3, "qty": 1.0},
        ]}},
        "hit_legs": [],
        "completed_tp_legs": [],
    }
    assert tracker._weighted_rr_snapshot(trade)[0] == pytest.approx(0.4)


def test_adverse_completed_recovered_tp_clears_recovery_marker_but_not_hit_leg():
    from event_engine import tracker

    trade = {
        "tp_recovery_pending": True,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1", "recovered_execution_pending": True}],
        "completed_tp_legs": ["tp1"],
        "hit_legs": [],  # adverse TP is deliberately not a strategy hit/BE trigger
    }
    assert tracker._clear_completed_tp_recovery_markers(trade) is True
    assert trade["tp_recovery_pending"] is False
    assert "recovered_execution_pending" not in trade["tp_orders"][0]
    assert trade["hit_legs"] == []


def test_unfilled_recovered_tp_keeps_recovery_marker_pending():
    from event_engine import tracker

    trade = {
        "tp_recovery_pending": True,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1", "recovered_execution_pending": True}],
        "completed_tp_legs": [],
        "hit_legs": [],
    }
    assert tracker._clear_completed_tp_recovery_markers(trade) is False
    assert trade["tp_recovery_pending"] is True
    assert trade["tp_orders"][0]["recovered_execution_pending"] is True


def _open_protection_from_orders(orders):
    return {
        "status": "ok",
        "sl_orders": [x for x in list(orders) if str(x.get("type", "")).upper() in {"STOP", "STOP_MARKET"}],
        "tp_orders": [x for x in list(orders) if str(x.get("type", "")).upper() in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}],
    }


def test_sl_repair_uses_unique_client_id_before_canceling_stale_sl(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {"quantityPrecision": 1, "pricePrecision": 2, "tradeMinQuantity": 0.1})
    monkeypatch.setattr(bingx, "position_side_param", lambda direction, **kwargs: str(direction).upper())
    monkeypatch.setattr(bingx, "_current_close_price", lambda symbol: 100.0)
    base_id = bingx.build_sl_client_order_id("TRADE-1")
    orders = [{
        "orderId": "SL_OLD", "clientOrderId": base_id, "type": "STOP_MARKET",
        "side": "SELL", "positionSide": "LONG", "stopPrice": "90.00", "origQty": "1.0",
    }]
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *a, **k: _open_protection_from_orders(orders))
    cancelled = []

    def cancel(symbol, order_id):
        cancelled.append(str(order_id))
        orders[:] = [x for x in orders if str(x.get("orderId")) != str(order_id)]
        return {"code": 0}

    monkeypatch.setattr(bingx, "cancel_order", cancel)
    posts = []

    def request(method, path, params=None, **kwargs):
        posts.append(dict(params or {}))
        created = {
            "orderId": "SL_NEW", "clientOrderId": params["clientOrderId"], "type": params["type"],
            "side": params["side"], "positionSide": params["positionSide"],
            "stopPrice": params["stopPrice"], "origQty": params["quantity"],
        }
        orders.append(created)
        return {"code": 0, "data": {"order": created}}

    monkeypatch.setattr(bingx, "_request", request)
    result = bingx.ensure_directional_protection("AAA-USDT", "LONG", 100.0, 0.5, 10.0, [], trade_id="TRADE-1")

    assert result["status"] == "PROTECTED"
    assert len(posts) == 1
    assert posts[0]["clientOrderId"] != base_id
    assert bingx._client_order_id_matches_base(posts[0]["clientOrderId"], base_id)
    assert cancelled == ["SL_OLD"]
    assert [x["orderId"] for x in orders] == ["SL_NEW"]


@pytest.mark.parametrize(
    "old_price,old_qty",
    [(102.0, 2.0), (103.0, 1.0)],  # wrong price and wrong quantity
)
def test_tp_repair_uses_unique_id_and_cleans_old_order_after_verification(monkeypatch, old_price, old_qty):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {"quantityPrecision": 0, "pricePrecision": 2, "tradeMinQuantity": 1.0})
    monkeypatch.setattr(bingx, "position_side_param", lambda direction, **kwargs: str(direction).upper())
    monkeypatch.setattr(bingx, "_current_close_price", lambda symbol: 100.0)
    trade_id = "TRADE-TP"
    base_tp_id = bingx.build_tp_client_order_id("tp1", trade_id)
    base_sl_id = bingx.build_sl_client_order_id(trade_id)
    old_tp_id = "TP_OLD"
    orders = [
        {"orderId": "SL_KEEP", "clientOrderId": base_sl_id, "type": "STOP_MARKET", "side": "SELL", "positionSide": "LONG", "stopPrice": "90.00", "origQty": "2"},
        {"orderId": old_tp_id, "clientOrderId": base_tp_id, "type": "TAKE_PROFIT_MARKET", "side": "SELL", "positionSide": "LONG", "stopPrice": f"{old_price:.2f}", "origQty": str(old_qty)},
    ]
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *a, **k: _open_protection_from_orders(orders))
    posts = []
    cancelled = []

    def request(method, path, params=None, **kwargs):
        posts.append(dict(params or {}))
        created = {
            "orderId": "TP_NEW", "clientOrderId": params["clientOrderId"], "type": params["type"],
            "side": params["side"], "positionSide": params["positionSide"],
            "stopPrice": params["stopPrice"], "origQty": params["quantity"],
        }
        orders.append(created)
        return {"code": 0, "data": {"order": created}}

    def cancel(symbol, order_id):
        cancelled.append(str(order_id))
        orders[:] = [x for x in orders if str(x.get("orderId")) != str(order_id)]
        return {"code": 0}

    def verify(symbol, direction, *, client_order_id, order_kind, expected_price, expected_qty, price_precision, max_attempts=3):
        matched = next((x for x in orders if x.get("clientOrderId") == client_order_id), None)
        if matched is None:
            return {"status": "not_found", "client_order_id": client_order_id}
        return {"status": "verified", "order": matched}

    monkeypatch.setattr(bingx, "_request", request)
    monkeypatch.setattr(bingx, "cancel_order", cancel)
    monkeypatch.setattr(bingx, "_verify_open_order", verify)
    result = bingx.ensure_directional_protection(
        "AAA-USDT", "LONG", 100.0, 2.0, 10.0,
        [{"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 1.0}], trade_id=trade_id,
    )

    assert result["status"] == "PROTECTED"
    assert posts[0]["clientOrderId"] != base_tp_id
    assert bingx._client_order_id_matches_base(posts[0]["clientOrderId"], base_tp_id)
    assert cancelled == [old_tp_id]
    assert not any(x.get("orderId") == old_tp_id for x in orders)
    assert any(x.get("orderId") == "TP_NEW" for x in orders)


def test_be_repair_uses_unique_id_and_cleans_stale_be_after_new_be_verified(monkeypatch):
    from event_engine import bingx, tracker

    monkeypatch.setattr(bingx, "get_position_mode", lambda **kwargs: "HEDGE")
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {"quantityPrecision": 1, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "get_contract", lambda symbol: {"quantityPrecision": 1, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda symbol: symbol)
    monkeypatch.setattr(tracker, "_load_active_trades_required", lambda: {})
    monkeypatch.setattr(tracker, "_order_ids_owned_by_other_trades", lambda *a, **k: {})
    trade_token = "TRADE-BE"
    base_be_id = f"EVT_BE_{trade_token}"
    orders = [
        {"orderId": "SL_OLD", "clientOrderId": "EVT_OLD_SL", "type": "STOP_MARKET", "side": "SELL", "positionSide": "LONG", "stopPrice": "90.00", "origQty": "1.0"},
        {"orderId": "BE_OLD", "clientOrderId": base_be_id, "type": "STOP_MARKET", "side": "SELL", "positionSide": "LONG", "stopPrice": "100.00", "origQty": "0.5"},
    ]
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: _open_protection_from_orders(orders))
    posts = []
    cancelled = []

    def request(method, path, params=None, **kwargs):
        posts.append(dict(params or {}))
        created = {
            "orderId": "BE_NEW", "clientOrderId": params["clientOrderId"], "type": params["type"],
            "side": params["side"], "positionSide": params["positionSide"],
            "stopPrice": params["stopPrice"], "origQty": params["quantity"],
        }
        orders.append(created)
        return {"code": 0, "data": {"order": created}}

    def cancel(symbol, order_id):
        cancelled.append(str(order_id))
        orders[:] = [x for x in orders if str(x.get("orderId")) != str(order_id)]
        return {"code": 0}

    monkeypatch.setattr(tracker, "_request", request)
    monkeypatch.setattr(tracker, "cancel_order", cancel)
    result = tracker._move_sl_to_break_even(
        "AAA-USDT", "LONG", 100.0, 1.0, "SL_OLD", trade_token, old_sl_price=90.0,
        owner_event_id="EVT_CURRENT",
    )

    assert result["status"] == "created", result
    assert posts[0]["clientOrderId"] != base_be_id
    assert bingx._client_order_id_matches_base(posts[0]["clientOrderId"], base_be_id)
    assert set(cancelled) == {"BE_OLD", "SL_OLD"}
    assert [x["orderId"] for x in orders] == ["BE_NEW"]


def test_binance_retryable_flag_reaches_fetch_layer(monkeypatch):
    from event_engine import binance

    observed = {}
    monkeypatch.setattr(binance, "_get", lambda *args, **kwargs: observed.update(kwargs) or [])
    assert binance.fetch_klines("BTCUSDT", interval="5m", limit=1, retryable=False) == []
    assert observed["retryable"] is False


def test_binance_sessions_have_distinct_retry_policies(monkeypatch):
    from event_engine import binance

    monkeypatch.setattr(binance, "_LOCAL", threading.local())
    retrying = binance._session(retryable=True)
    no_retry = binance._session(retryable=False)
    assert retrying.get_adapter("https://").max_retries.total == 2
    assert no_retry.get_adapter("https://").max_retries.total == 0
    assert retrying is not no_retry


def test_repaired_tp_is_reused_on_next_reconciliation_instead_of_replaced_again(monkeypatch):
    """A successful repair ID must remain recognizable as the same TP leg."""
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {"quantityPrecision": 0, "pricePrecision": 2, "tradeMinQuantity": 1.0})
    monkeypatch.setattr(bingx, "position_side_param", lambda direction, **kwargs: str(direction).upper())
    trade_id = "TRADE-TP-SECOND"
    base_tp_id = bingx.build_tp_client_order_id("tp1", trade_id)
    repaired_tp_id = bingx._repair_client_order_id(base_tp_id)
    base_sl_id = bingx.build_sl_client_order_id(trade_id)
    orders = [
        {"orderId": "SL_KEEP", "clientOrderId": base_sl_id, "type": "STOP_MARKET", "side": "SELL", "positionSide": "LONG", "stopPrice": "90.00", "origQty": "2"},
        {"orderId": "TP_REPAIRED", "clientOrderId": repaired_tp_id, "type": "TAKE_PROFIT_MARKET", "side": "SELL", "positionSide": "LONG", "stopPrice": "103.00", "origQty": "2"},
    ]
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *a, **k: _open_protection_from_orders(orders))
    monkeypatch.setattr(bingx, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: pytest.fail("valid repaired TP must not be posted again"))
    monkeypatch.setattr(bingx, "cancel_order", lambda *a, **k: pytest.fail("valid repaired TP must not be cancelled"))

    result = bingx.ensure_directional_protection(
        "AAA-USDT", "LONG", 100.0, 2.0, 10.0,
        [{"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 1.0}],
        trade_id=trade_id,
    )

    assert result["status"] == "PROTECTED"
    assert result["tp_orders"][0]["status"] == "already_exists"
    assert result["tp_orders"][0]["order_id"] == "TP_REPAIRED"
