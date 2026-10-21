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
