from __future__ import annotations

import json

import pytest


def test_existing_verified_be_with_old_sl_cleanup_failure_is_cleanup_pending(monkeypatch):
    from event_engine import tracker

    monkeypatch.setattr(tracker, "to_bx_symbol", lambda value: value)
    monkeypatch.setattr(tracker, "get_contract", lambda _symbol: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "position_side_param", lambda direction: str(direction).upper())
    orders = [
        {
            "orderId": "SL_OLD", "clientOrderId": "EVT_TRADE1_SL", "type": "STOP_MARKET",
            "side": "SELL", "positionSide": "LONG", "stopPrice": "90.00", "origQty": "1.0",
        },
        {
            "orderId": "BE_EXISTING", "clientOrderId": "EVT_BE_TRADE1", "type": "STOP_MARKET",
            "side": "SELL", "positionSide": "LONG", "stopPrice": "100.00", "origQty": "1.0",
        },
    ]
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *_a, **_k: {
        "status": "ok", "sl_orders": orders, "tp_orders": [],
    })
    monkeypatch.setattr(tracker, "_load_active_trades_required", lambda: {})
    # This test isolates the state/status branch after a valid BE candidate was
    # exchange-observed; order field validation has separate direct coverage below.
    monkeypatch.setattr(tracker, "_validate_sl_order_for_position", lambda *_a, **_k: True)
    monkeypatch.setattr(tracker, "_cancel_old_sl_verified", lambda *_a, **_k: (False, "simulated timeout"))

    result = tracker._move_sl_to_break_even(
        "AAA-USDT", "LONG", 100.0, 1.0, "SL_OLD", trade_id="TRADE1", owner_event_id="EVT_TRADE1"
    )

    assert result["status"] == "created_cleanup_pending"
    assert result["order_id"] == "BE_EXISTING"
    assert result["old_sl_cleanup_pending"] is True
    assert result["old_sl_cleanup_pending_order_id"] == "SL_OLD"


def test_manual_sl_at_entry_is_not_adopted_as_engine_break_even(monkeypatch):
    from event_engine import tracker

    monkeypatch.setattr(tracker, "position_side_param", lambda direction: str(direction).upper())
    monkeypatch.setattr(tracker, "_validate_sl_order_for_position", lambda *_a, **_k: True)
    common = {
        "orderId": "MANUAL_1", "type": "STOP_MARKET", "side": "SELL", "positionSide": "LONG",
        "stopPrice": "100.00", "origQty": "1.0",
    }
    manual = {**common, "clientOrderId": "USER_SET_BE"}
    engine = {**common, "orderId": "ENGINE_BE", "clientOrderId": "EVT_BE_TRADE1"}

    assert tracker._is_valid_break_even_stop(manual, "LONG", 100.0, 1.0, 2) is False
    assert tracker._is_valid_break_even_stop(
        engine, "LONG", 100.0, 1.0, 2, expected_client_order_id="EVT_BE_TRADE1"
    ) is True
    assert tracker._is_valid_break_even_stop(
        engine, "LONG", 100.0, 1.0, 2, expected_client_order_id="EVT_BE_OTHER_TRADE"
    ) is False


def test_reconciliation_replacing_missing_be_with_normal_sl_resets_active_be_state(tmp_path, monkeypatch):
    from event_engine import tracker

    active_path = tmp_path / "active_trades.json"
    state = {
        "EVT_TRADE1": {
            "event_id": "EVT_TRADE1", "symbol": "AAA-USDT", "direction": "LONG", "closed": False,
            "entry_price": 100.0, "initial_qty": 1.0, "remaining_qty": 0.5,
            "planned_risk_pct": 10.0, "tp_levels": [], "tp_orders": [], "sl_order": {"order_id": "BE_OLD"},
            "hit_legs": ["tp1"], "be_activated": True, "be_required": False,
            "be_order_id": "BE_OLD", "be_client_order_id": "EVT_BE_TRADE1", "be_activation_ts": 12345,
        }
    }
    active_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active_path)

    assert tracker.update_active_trade_protection(
        "AAA-USDT", "LONG", [], {
            "status": "already_exists", "order_id": "SL_REBUILT", "client_order_id": "EVT_TRADE1_SL",
            "stop_price": 90.0, "qty": 0.5, "protection_order_verified": True,
        }, event_id="EVT_TRADE1"
    ) is True

    updated = json.loads(active_path.read_text(encoding="utf-8"))["EVT_TRADE1"]
    assert updated["sl_order"]["order_id"] == "SL_REBUILT"
    assert updated["be_activated"] is False
    assert updated["be_order_id"] is None
    assert updated["be_client_order_id"] is None
    assert updated["be_activation_ts"] is None
    assert updated["be_previous_order_id"] == "BE_OLD"
    assert updated["be_previous_activation_ts"] == 12345
    assert updated["be_required"] is True


def test_reconciliation_preserves_be_when_verified_exchange_order_id_changes(tmp_path, monkeypatch):
    from event_engine import tracker

    active_path = tmp_path / "active_trades.json"
    state = {
        "EVT_TRADE1": {
            "event_id": "EVT_TRADE1", "symbol": "AAA-USDT", "direction": "LONG", "closed": False,
            "entry_price": 100.0, "initial_qty": 1.0, "remaining_qty": 0.5,
            "planned_risk_pct": 10.0, "tp_levels": [], "tp_orders": [], "sl_order": {"order_id": "BE_OLD"},
            "hit_legs": ["tp1"], "be_activated": True, "be_required": False,
            "be_order_id": "BE_OLD", "be_client_order_id": "EVT_BE_TRADE1", "be_activation_ts": 12345,
        }
    }
    active_path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active_path)

    assert tracker.update_active_trade_protection(
        "AAA-USDT", "LONG", [], {
            "status": "already_exists", "order_id": "BE_REPAIRED", "client_order_id": "EVT_BE_TRADE1",
            "stop_price": 100.0, "qty": 0.5, "protection_order_verified": True,
        }, event_id="EVT_TRADE1"
    ) is True

    updated = json.loads(active_path.read_text(encoding="utf-8"))["EVT_TRADE1"]
    assert updated["be_activated"] is True
    assert updated["be_order_id"] == "BE_REPAIRED"
    assert updated["be_client_order_id"] == "EVT_BE_TRADE1"
    assert updated["be_activation_ts"] == 12345


def _fill_rows():
    return [
        {"order_id": "ENTRY1", "qty": 1.0, "price": 100.0, "fee": -0.1, "realized_pnl": 0.0, "time_ms": 100},
        {"order_id": "EXIT1", "qty": 0.5, "price": 102.0, "fee": -0.1, "realized_pnl": 1.0, "time_ms": 200},
    ]


def test_fill_accounting_never_verifies_short_exit_quantity(tmp_path, monkeypatch):
    from event_engine import tracker

    rows = _fill_rows()
    monkeypatch.setattr(tracker, "credentials_available", lambda: True)
    monkeypatch.setattr(tracker, "get_fill_orders", lambda _s, _a, _b, *, order_id=None, limit=1000: [
        row for row in rows if order_id is None or row["order_id"] == str(order_id)
    ])
    monkeypatch.setattr(tracker, "_get_order_execution_evidence", lambda _s, order_id, **_k: {
        "status": "ok", "order_id": order_id,
        "order_status": "FILLED" if order_id == "ENTRY1" else "PARTIALLY_FILLED",
        "executed_qty": 1.0 if order_id == "ENTRY1" else 0.5,
        "avg_price": 100.0 if order_id == "ENTRY1" else 102.0,
    })

    result = tracker._collect_trade_fill_accounting({
        "symbol": "AAA-USDT", "direction": "LONG", "entry_ts": 1,
        "entry_order": {"order_id": "ENTRY1"}, "exit_order_id": "EXIT1",
    }, 1000)

    assert result["status"] == "PARTIAL_ACCOUNTING"
    assert result["quantity_reconciliation"]["status"] == "PARTIAL_OR_UNVERIFIED"
    assert result["observed_entry_qty"] == pytest.approx(1.0)
    assert result["observed_exit_qty"] == pytest.approx(0.5)
    assert result["confirmed_exit_qty"] == pytest.approx(0.5)  # observed partial quantity, not a verified full close
    assert result["gross_realized_pnl_abs"] is None
    assert result["fees_paid_abs"] is None
    assert result["exchange_realized_pnl_abs"] is None


def test_fill_accounting_surfaces_order_status_timeout_and_marks_partial(monkeypatch):
    from event_engine import tracker

    rows = _fill_rows()
    monkeypatch.setattr(tracker, "credentials_available", lambda: True)
    fill_calls = []

    def get_fills(_s, _a, _b, *, order_id=None, limit=1000):
        fill_calls.append(order_id)
        return [row for row in rows if order_id is None or row["order_id"] == str(order_id)]

    monkeypatch.setattr(tracker, "get_fill_orders", get_fills)

    def execution_evidence(_symbol, order_id, **_kwargs):
        if order_id == "EXIT1":
            return {"status": "error", "error": "simulated exchange order-status API timeout"}
        return {
            "status": "ok", "order_id": order_id, "order_status": "FILLED",
            "executed_qty": 1.0, "avg_price": 100.0,
        }

    monkeypatch.setattr(tracker, "_get_order_execution_evidence", execution_evidence)
    result = tracker._collect_trade_fill_accounting({
        "symbol": "AAA-USDT", "direction": "LONG", "entry_ts": 1,
        "entry_order": {"order_id": "ENTRY1"}, "exit_order_id": "EXIT1",
    }, 1000)

    assert result["status"] == "PARTIAL_ACCOUNTING"
    assert result["quantity_reconciliation"]["status"] == "PARTIAL_OR_UNVERIFIED"
    assert None in fill_calls  # aggregate read
    assert "EXIT1" in fill_calls  # exact order-specific fallback after status lookup failed
    assert "EXIT1" in result["order_execution_evidence_errors"]
    assert "timeout" in result["order_execution_evidence_errors"]["EXIT1"]
    assert result["observed_exit_qty"] == pytest.approx(0.5)
    assert result["confirmed_exit_qty"] is None
    assert result["gross_realized_pnl_abs"] is None
    assert result["fees_paid_abs"] is None
