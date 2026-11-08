import pytest

from event_engine.tracker import _update_tp_reconciliation_state


def test_partial_tp_fill_persists_pending_state_and_first_observation_time():
    order = {"leg": "tp1", "order_id": "123"}

    state = _update_tp_reconciliation_state(order, "PARTIALLY_FILLED", 0.4, 1_000)
    _update_tp_reconciliation_state(order, "PARTIALLY_FILLED", 0.7, 2_000)

    assert state == "PENDING_PARTIAL_FILL"
    assert order["reconciliation_state"] == "PENDING_PARTIAL_FILL"
    assert order["last_exchange_executed_qty"] == 0.7
    assert order["partial_fill_since_ts"] == 1_000
    assert order["last_reconciled_ts"] == 2_000


def test_filled_tp_clears_pending_marker_only_after_exchange_terminal_status():
    order = {
        "leg": "tp1", "reconciliation_state": "PENDING_PARTIAL_FILL",
        "partial_fill_since_ts": 1_000,
    }

    state = _update_tp_reconciliation_state(order, "FILLED", 1.0, 3_000)

    assert state == "FILLED_CONFIRMED"
    assert order["last_exchange_executed_qty"] == 1.0
    assert "partial_fill_since_ts" not in order
    assert order["partial_fill_terminal_status"] is None


def test_cancelled_partially_filled_tp_is_not_misreported_as_fully_filled():
    order = {"leg": "tp1", "reconciliation_state": "PENDING_PARTIAL_FILL", "partial_fill_since_ts": 1_000}

    state = _update_tp_reconciliation_state(order, "CANCELED", 0.4, 4_000)

    assert state == "PARTIAL_FILL_TERMINAL"
    assert order["last_exchange_executed_qty"] == 0.4
    assert order["partial_fill_terminal_status"] == "CANCELED"
    assert "partial_fill_since_ts" not in order


def test_transient_unknown_status_does_not_clear_pending_partial_fill():
    order = {"leg": "tp1", "reconciliation_state": "PENDING_PARTIAL_FILL", "partial_fill_since_ts": 1_000}

    state = _update_tp_reconciliation_state(order, "", 0.4, 5_000)

    assert state == "PENDING_PARTIAL_FILL"
    assert order["partial_fill_since_ts"] == 1_000


def test_transient_unknown_status_preserves_terminal_partial_fill_state():
    order = {
        "leg": "tp1",
        "reconciliation_state": "PARTIAL_FILL_TERMINAL",
        "partial_fill_terminal_status": "CANCELED",
        "last_exchange_executed_qty": 0.4,
    }

    state = _update_tp_reconciliation_state(order, "", 0.0, 5_000)

    assert state == "PARTIAL_FILL_TERMINAL"
    assert order["last_exchange_executed_qty"] == 0.4
    assert order["partial_fill_terminal_status"] == "CANCELED"


def test_filled_status_with_executed_quantity_below_order_quantity_is_not_full_fill():
    order = {"leg": "tp1", "qty": 0.5}

    state = _update_tp_reconciliation_state(order, "FILLED", 0.3, 6_000, expected_qty=0.5)

    assert state == "PARTIAL_FILL_TERMINAL"
    assert order["last_exchange_executed_qty"] == 0.3
    assert order["partial_fill_terminal_status"] == "FILLED_QTY_MISMATCH"


def test_filled_confirmation_is_not_downgraded_by_stale_cancel_response():
    order = {"leg": "tp1", "reconciliation_state": "FILLED_CONFIRMED", "last_exchange_executed_qty": 0.5, "qty": 0.5}

    state = _update_tp_reconciliation_state(order, "CANCELED", 0.5, 7_000, expected_qty=0.5)

    assert state == "FILLED_CONFIRMED"
    assert order["partial_fill_terminal_status"] is None


def test_partial_fill_persists_the_average_price_with_its_cumulative_quantity():
    order = {"leg": "tp1", "qty": 0.5}

    state = _update_tp_reconciliation_state(
        order, "PARTIALLY_FILLED", 0.2, 8_000, expected_qty=0.5, avg_price=103.0,
    )

    assert state == "PENDING_PARTIAL_FILL"
    assert order["last_exchange_executed_qty"] == 0.2
    assert order["last_exchange_avg_price"] == 103.0
    assert order["last_exchange_avg_price_qty"] == 0.2


def test_incremental_tp_fill_average_price_is_derived_from_cumulative_order_average():
    from event_engine.tracker import _incremental_tp_fill_average_price

    result = _incremental_tp_fill_average_price(
        cumulative_qty=0.5,
        cumulative_avg_price=104.0,
        previous_accounted_qty=0.2,
        previous_accounted_notional=0.2 * 103.0,
    )

    assert result == pytest.approx((0.5 * 104.0 - 0.2 * 103.0) / 0.3)


def test_incremental_tp_fill_average_price_uses_cumulative_price_for_first_fill():
    from event_engine.tracker import _incremental_tp_fill_average_price

    assert _incremental_tp_fill_average_price(
        cumulative_qty=0.2,
        cumulative_avg_price=103.0,
        previous_accounted_qty=0.0,
        previous_accounted_notional=0.0,
    ) == pytest.approx(103.0)


def test_incremental_tp_fill_average_price_refuses_incomplete_or_invalid_accounting():
    from event_engine.tracker import _incremental_tp_fill_average_price

    assert _incremental_tp_fill_average_price(
        cumulative_qty=0.5,
        cumulative_avg_price=104.0,
        previous_accounted_qty=0.2,
        previous_accounted_notional=None,
    ) is None
    assert _incremental_tp_fill_average_price(
        cumulative_qty=0.1,
        cumulative_avg_price=104.0,
        previous_accounted_qty=0.2,
        previous_accounted_notional=20.0,
    ) is None
    assert _incremental_tp_fill_average_price(
        cumulative_qty=0.5,
        cumulative_avg_price=40.0,
        previous_accounted_qty=0.2,
        previous_accounted_notional=0.2 * 103.0,
    ) is None


def test_close_evidence_uses_cumulative_order_average_with_cumulative_tp_quantity():
    from event_engine.tracker import _close_execution_evidence

    trade = {
        "initial_qty": 1.0,
        "tp_fill_events": [
            {
                "leg": "tp1", "order_id": "TP1", "delta_qty": 0.2,
                "executed_qty_total": 0.2, "avg_price": 103.0,
                "cumulative_avg_price": 103.0, "order_update_time_ms": 100,
            },
            {
                "leg": "tp1", "order_id": "TP1", "delta_qty": 0.3,
                "executed_qty_total": 0.5, "avg_price": (0.5 * 104.0 - 0.2 * 103.0) / 0.3,
                "cumulative_avg_price": 104.0, "order_update_time_ms": 150,
            },
            {
                "leg": "tp2", "order_id": "TP2", "delta_qty": 0.5,
                "executed_qty_total": 0.5, "avg_price": 106.0,
                "cumulative_avg_price": 106.0, "order_update_time_ms": 200,
            },
        ],
    }
    result = _close_execution_evidence(trade, {
        "entry_qty_confirmed": 1.0,
        "fills": [],
        "order_execution_evidence": [],
    })

    assert result["status"] == "CONFIRMED"
    assert result["observed_exit_qty"] == pytest.approx(1.0)
    assert result["exit_vwap_confirmed"] == pytest.approx(105.0)
    assert result["latest"]["avg_price"] == pytest.approx(106.0)


def test_filled_status_with_executed_quantity_above_original_quantity_is_not_confirmed():
    order = {"leg": "tp1", "qty": 0.5}

    state = _update_tp_reconciliation_state(
        order, "FILLED", 0.6, 9_000, expected_qty=0.5,
    )

    assert state == "PARTIAL_FILL_TERMINAL"
    assert order["last_exchange_executed_qty"] == 0.6
    assert order["partial_fill_terminal_status"] == "FILLED_QTY_MISMATCH"
