from __future__ import annotations

import pytest


def test_get_order_preserves_trigger_execution_and_lifecycle_metadata(monkeypatch):
    from event_engine import bingx

    raw_order = {
        "orderId": "1234567890123456789",
        "symbol": "AAA-USDT",
        "side": "SELL",
        "positionSide": "LONG",
        "type": "STOP_MARKET",
        "status": "NEW",
        "origQty": "1.25",
        "executedQty": "0.10",
        "avgPrice": "101.5",
        "stopPrice": "100.0",
        "workingType": "MARK_PRICE",
        "reduceOnly": True,
        "closePosition": False,
        "timeInForce": "GTC",
        "time": 1_700_000_000_000,
        "updateTime": 1_700_000_001_000,
    }
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "AAA-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *_a, **_k: {"code": 0, "data": {"order": raw_order}})

    result = bingx.get_order("AAA-USDT", "1234567890123456789")

    assert result["status"] == "ok"
    assert result["order_id"] == "1234567890123456789"
    assert result["order_type"] == "STOP_MARKET"
    assert result["trigger_price"] == pytest.approx(100.0)
    assert result["orig_qty"] == pytest.approx(1.25)
    assert result["executed_qty"] == pytest.approx(0.10)
    assert result["executed_qty_known"] is True
    assert result["working_type"] == "MARK_PRICE"
    assert result["reduce_only"] is True
    assert result["close_position"] is False
    assert result["time_in_force"] == "GTC"
    assert result["time_ms"] == 1_700_000_000_000
    assert result["update_time_ms"] == 1_700_000_001_000


def test_get_order_keeps_unreported_order_flags_unknown_not_false(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "AAA-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *_a, **_k: {
        "code": 0,
        "data": {"order": {
            "orderId": "ORDER-UNKNOWN-FLAGS", "symbol": "AAA-USDT",
            "side": "SELL", "positionSide": "LONG", "type": "STOP_MARKET",
            "status": "NEW", "origQty": "1", "stopPrice": "90",
        }},
    })

    result = bingx.get_order("AAA-USDT", "ORDER-UNKNOWN-FLAGS")

    assert result["status"] == "ok"
    assert result["working_type"] is None
    assert result["reduce_only"] is None
    assert result["close_position"] is None
    assert result["time_in_force"] is None


def test_exchange_boolean_metadata_parsing_is_tristate_and_accepts_wire_strings():
    from event_engine.bingx import _parse_optional_exchange_bool

    assert _parse_optional_exchange_bool(True) is True
    assert _parse_optional_exchange_bool(False) is False
    assert _parse_optional_exchange_bool("true") is True
    assert _parse_optional_exchange_bool("false") is False
    assert _parse_optional_exchange_bool("1") is True
    assert _parse_optional_exchange_bool("0") is False
    assert _parse_optional_exchange_bool(None) is None
    assert _parse_optional_exchange_bool("") is None
    assert _parse_optional_exchange_bool("unknown") is None


def test_protection_order_lifecycle_metadata_keeps_exchange_times_and_unverified_flags(monkeypatch):
    from event_engine import bingx

    metadata = bingx._order_lifecycle_metadata(
        {
            "orderId": "SL-1",
            "clientOrderId": "EVT_SL_TEST",
            "type": "STOP_MARKET",
            "status": "NEW",
            "side": "SELL",
            "positionSide": "LONG",
            "stopPrice": "90.0",
            "origQty": "1.0",
            "workingType": "MARK_PRICE",
            "reduceOnly": "true",
            "timeInForce": "GTC",
            "time": 1_700_000_000_000,
            "updateTime": 1_700_000_001_000,
        },
        verification_observed_at_ms=1_700_000_002_000,
    )
    assert metadata["order_id"] == "SL-1"
    assert metadata["client_order_id"] == "EVT_SL_TEST"
    assert metadata["order_type"] == "STOP_MARKET"
    assert metadata["exchange_order_status"] == "NEW"
    assert metadata["working_type"] == "MARK_PRICE"
    assert metadata["reduce_only"] is True
    assert metadata["close_position"] is None
    assert metadata["time_in_force"] == "GTC"
    assert metadata["exchange_order_time_ms"] == 1_700_000_000_000
    assert metadata["exchange_order_update_time_ms"] == 1_700_000_001_000
    assert metadata["verification_observed_at_ms"] == 1_700_000_002_000


def test_protection_order_lifecycle_metadata_does_not_invent_exchange_times_or_flags():
    from event_engine import bingx

    metadata = bingx._order_lifecycle_metadata({"orderId": "SL-UNKNOWN"})
    assert metadata["working_type"] is None
    assert metadata["exchange_order_status"] is None
    assert metadata["reduce_only"] is None
    assert metadata["close_position"] is None
    assert metadata["time_in_force"] is None
    assert metadata["exchange_order_time_ms"] is None
    assert metadata["exchange_order_update_time_ms"] is None


@pytest.mark.parametrize("raw_executed_qty", ["NaN", "Infinity", "-0.1", "not-a-number"])
def test_get_order_marks_malformed_executed_quantity_as_unknown(monkeypatch, raw_executed_qty):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "AAA-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *_a, **_k: {
        "code": 0,
        "data": {"order": {
            "orderId": "ORDER-BAD-EXECUTED-QTY", "symbol": "AAA-USDT",
            "side": "SELL", "positionSide": "LONG", "type": "TAKE_PROFIT_MARKET",
            "status": "PARTIALLY_FILLED", "origQty": "0.5", "executedQty": raw_executed_qty,
        }},
    })

    result = bingx.get_order("AAA-USDT", "ORDER-BAD-EXECUTED-QTY")

    assert result["status"] == "ok"
    assert result["executed_qty_known"] is False


def test_get_order_keeps_a_real_zero_executed_quantity_known(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "AAA-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *_a, **_k: {
        "code": 0,
        "data": {"order": {
            "orderId": "ORDER-ZERO-EXECUTED-QTY", "symbol": "AAA-USDT",
            "side": "SELL", "positionSide": "LONG", "type": "TAKE_PROFIT_MARKET",
            "status": "NEW", "origQty": "0.5", "executedQty": "0",
        }},
    })

    result = bingx.get_order("AAA-USDT", "ORDER-ZERO-EXECUTED-QTY")

    assert result["executed_qty"] == 0.0
    assert result["executed_qty_known"] is True


def test_protection_lifecycle_metadata_does_not_mark_negative_executed_quantity_known():
    from event_engine import bingx

    metadata = bingx._order_lifecycle_metadata({
        "orderId": "ORDER-NEGATIVE-QTY",
        "executedQty": "-0.1",
    })

    assert metadata["executed_qty"] is None
    assert metadata["executed_qty_known"] is False
