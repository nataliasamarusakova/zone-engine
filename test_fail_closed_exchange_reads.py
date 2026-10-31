from __future__ import annotations

import pytest

from event_engine import bingx, tracker


def _identity_symbol(value: str) -> str:
    return str(value).upper().replace("-", "")


def test_get_position_directional_unknown_envelope_returns_error_not_not_found(monkeypatch):
    monkeypatch.setattr(bingx, "to_bx_symbol", _identity_symbol)
    monkeypatch.setattr(
        bingx,
        "_request",
        lambda *args, **kwargs: {
            "code": 0,
            "data": {
                "unexpected": [{
                    "symbol": "BTCUSDT",
                    "positionSide": "LONG",
                    "positionAmt": "1",
                    "avgPrice": "100",
                }]
            },
        },
    )

    result = bingx.get_position_directional("BTC-USDT", "LONG")

    assert result["status"] == "error"
    assert "unrecognized response shape" in result["error"]


def test_get_position_directional_malformed_amount_returns_error_not_not_found(monkeypatch):
    monkeypatch.setattr(bingx, "to_bx_symbol", _identity_symbol)
    monkeypatch.setattr(
        bingx,
        "_request",
        lambda *args, **kwargs: {
            "code": 0,
            "data": [{
                "symbol": "BTCUSDT",
                "positionSide": "LONG",
                "positionAmt": "not-a-number",
                "avgPrice": "100",
            }],
        },
    )

    result = bingx.get_position_directional("BTC-USDT", "LONG")

    assert result["status"] == "error"
    assert "invalid positionAmt" in result["error"]


def test_get_position_directional_recognizes_explicit_empty_list_as_not_found(monkeypatch):
    monkeypatch.setattr(bingx, "to_bx_symbol", _identity_symbol)
    monkeypatch.setattr(bingx, "_request", lambda *args, **kwargs: {"code": 0, "data": []})

    result = bingx.get_position_directional("BTC-USDT", "LONG")

    assert result["status"] == "not_found"


def test_get_positions_rejects_non_finite_open_position_values(monkeypatch):
    monkeypatch.setattr(
        bingx,
        "_request",
        lambda *args, **kwargs: {
            "code": 0,
            "data": [{
                "symbol": "BTCUSDT",
                "positionSide": "LONG",
                "positionAmt": "NaN",
                "avgPrice": "100",
            }],
        },
    )

    with pytest.raises(RuntimeError, match="non-finite positionAmt"):
        bingx.get_positions()


def test_get_open_protection_directional_unknown_envelope_returns_error(monkeypatch):
    monkeypatch.setattr(bingx, "to_bx_symbol", _identity_symbol)
    monkeypatch.setattr(bingx, "position_side_param", lambda direction: direction)
    monkeypatch.setattr(
        bingx,
        "_request",
        lambda *args, **kwargs: {"code": 0, "data": {"unexpected": []}},
    )

    result = bingx.get_open_protection_directional("BTC-USDT", "LONG")

    assert result["status"] == "error"
    assert "unrecognized response shape" in result["error"]
    assert result["tp_orders"] == []
    assert result["sl_orders"] == []


def test_get_open_protection_directional_rejects_unclassifiable_order_row(monkeypatch):
    monkeypatch.setattr(bingx, "to_bx_symbol", _identity_symbol)
    monkeypatch.setattr(bingx, "position_side_param", lambda direction: direction)
    monkeypatch.setattr(
        bingx,
        "_request",
        lambda *args, **kwargs: {
            "code": 0,
            "data": [{"symbol": "BTCUSDT", "positionSide": "LONG", "orderId": "1"}],
        },
    )

    result = bingx.get_open_protection_directional("BTC-USDT", "LONG")

    assert result["status"] == "error"
    assert "missing order type" in result["error"]


def test_open_orders_requests_are_rate_limited_across_callers(monkeypatch):
    now = {"value": 100.0}
    sleeps = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now["value"] += seconds

    monkeypatch.delenv("ZONE_ENGINE_TESTS_ISOLATED", raising=False)
    monkeypatch.setenv("BINGX_OPEN_ORDERS_MIN_INTERVAL_SEC", "0.30")
    monkeypatch.setattr(bingx.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(bingx.time, "sleep", fake_sleep)
    monkeypatch.setattr(bingx, "_LAST_OPEN_ORDERS_REQUEST_TS", None)

    bingx._throttle_open_orders_request()
    assert sleeps == []
    bingx._throttle_open_orders_request()

    assert sleeps == pytest.approx([0.30])
    assert now["value"] == pytest.approx(100.30)


@pytest.mark.parametrize("exchange_error", [
    "openOrders failed: code=100410 msg=trigger frequency limit",
    "openOrders failed: code=109500 msg=temporary network error",
])
def test_ensure_directional_protection_distinguishes_unavailable_read_from_missing_orders(monkeypatch, exchange_error):
    monkeypatch.setattr(bingx, "to_bx_symbol", _identity_symbol)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {
        "symbol": "BTC-USDT", "quantityPrecision": 3,
        "pricePrecision": 2, "tradeMinQuantity": 0.001,
    })
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *args, **kwargs: {
        "status": "error",
        "error": exchange_error,
    })

    result = bingx.ensure_directional_protection(
        "BTC-USDT", "LONG", 100.0, 1.0, 10.0,
        [{"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 0.5}],
        trade_id="EVT_READ_LIMITED",
    )

    assert result["status"] == "PROTECTION_READ_UNAVAILABLE"
    assert result["failure_class"] == "EXCHANGE_READ_UNAVAILABLE"
    assert result["error"] == exchange_error


def test_has_open_position_does_not_convert_bad_quantity_to_flat(monkeypatch):
    monkeypatch.setattr(bingx, "to_bx_symbol", _identity_symbol)
    monkeypatch.setattr(
        bingx,
        "get_positions",
        lambda **kwargs: [{
            "symbol": "BTCUSDT",
            "positionSide": "LONG",
            "positionAmt": "not-a-number",
            "avgPrice": "100",
        }],
    )

    with pytest.raises(RuntimeError, match="invalid positionAmt"):
        bingx.has_open_position("BTC-USDT", "LONG")


def test_emergency_close_fallback_does_not_treat_malformed_position_as_flat(monkeypatch):
    close_calls = []
    cleanup_calls = []
    monkeypatch.setattr(tracker, "get_position_directional", lambda *args, **kwargs: {"status": "error", "error": "directional parse failed"})
    monkeypatch.setattr(
        tracker,
        "get_positions",
        lambda **kwargs: [{
            "symbol": "BTCUSDT",
            "positionSide": "LONG",
            "positionAmt": "not-a-number",
            "avgPrice": "100",
        }],
    )
    monkeypatch.setattr(tracker, "close_position_market", lambda *args, **kwargs: close_calls.append(args) or {"status": "closed"})
    monkeypatch.setattr(tracker, "_cancel_engine_protection_before_emergency_close", lambda *args, **kwargs: cleanup_calls.append(args) or {"status": "ok"})

    result = tracker._emergency_close_after_be_failure("BTC-USDT", "LONG", 1.0, "EVT_TEST")

    assert result["status"] == "close_unverified"
    assert close_calls == []
    assert cleanup_calls == []
