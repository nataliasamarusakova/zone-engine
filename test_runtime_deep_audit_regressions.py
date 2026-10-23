from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_get_order_rejects_a_response_for_a_different_order_id(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "AAAUSDT")
    monkeypatch.setattr(bingx, "_request", lambda *_a, **_k: {
        "code": 0,
        "data": {"order": {"orderId": "OTHER_ORDER", "symbol": "AAAUSDT", "status": "FILLED", "executedQty": "1", "avgPrice": "100"}},
    })

    result = bingx.get_order("AAA-USDT", "REQUESTED_ORDER")

    assert result["status"] == "error"
    assert "orderId mismatch" in result["error"]


def test_get_order_does_not_fabricate_requested_id_when_response_id_is_missing(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "AAAUSDT")
    monkeypatch.setattr(bingx, "_request", lambda *_a, **_k: {
        "code": 0, "data": {"order": {"symbol": "AAAUSDT", "status": "FILLED", "executedQty": "1", "avgPrice": "100"}},
    })

    result = bingx.get_order("AAA-USDT", "REQUESTED_ORDER")

    assert result["status"] == "error"
    assert "returned no exchange orderId" in result["error"]


def test_order_execution_evidence_rejects_mismatched_id(monkeypatch):
    from event_engine import tracker

    monkeypatch.setattr(tracker, "get_order", lambda *_a, **_k: {
        "status": "ok", "order_id": "OTHER_ORDER", "order_status": "FILLED", "executed_qty": 1.0, "avg_price": 100.0,
    })
    result = tracker._get_order_execution_evidence("AAA-USDT", "REQUESTED_ORDER")
    assert result["status"] == "error"
    assert "ID mismatch" in result["error"]


def test_fill_accounting_never_verifies_exit_quantity_above_entry(monkeypatch):
    from event_engine import tracker

    fills = [
        {"order_id": "ENTRY", "qty": 1.0, "price": 100.0, "fee": -0.01, "realized_pnl": 0.0, "time_ms": 100},
        {"order_id": "TP1", "qty": 0.75, "price": 103.0, "fee": -0.01, "realized_pnl": 2.25, "time_ms": 200},
        {"order_id": "BE", "qty": 0.75, "price": 100.0, "fee": -0.01, "realized_pnl": 0.0, "time_ms": 300},
    ]
    monkeypatch.setattr(tracker, "credentials_available", lambda: True)
    monkeypatch.setattr(tracker, "get_fill_orders", lambda _s, _start, _end, *, order_id=None, limit=1000: [
        row for row in fills if order_id is None or row["order_id"] == str(order_id)
    ])
    monkeypatch.setattr(tracker, "_get_order_execution_evidence", lambda _s, oid, **_k: {
        "status": "ok", "order_id": oid, "order_status": "FILLED",
        "executed_qty": {"ENTRY": 1.0, "TP1": 0.75, "BE": 0.75}[oid],
        "avg_price": {"ENTRY": 100.0, "TP1": 103.0, "BE": 100.0}[oid],
    })

    result = tracker._collect_trade_fill_accounting({
        "symbol": "AAA-USDT", "direction": "LONG", "entry_ts": 1,
        "initial_qty": 1.0, "entry_order": {"order_id": "ENTRY"},
        "tp_orders": [{"leg": "TP1", "order_id": "TP1"}],
        "sl_order": {"order_id": "BE"},
    }, 1000)

    assert result["status"] == "PARTIAL_ACCOUNTING"
    assert result["quantity_reconciliation"]["status"] == "MISMATCH"
    assert result["observed_exit_qty"] == pytest.approx(1.5)
    assert result["confirmed_exit_qty"] is None
    assert result["exit_vwap_confirmed"] is None
    assert result["exit_vwap_observed"] == pytest.approx((0.75 * 103.0 + 0.75 * 100.0) / 1.5)
    assert result["gross_realized_pnl_abs"] is None
    assert result["fees_status"] == "UNAVAILABLE"


def test_close_evidence_prefers_exchange_rows_when_they_already_cover_trade_qty():
    from event_engine import tracker

    trade = {
        "initial_qty": 1.0,
        "exit_reason": "BREAK_EVEN",
        # This local event's order ID differs from the exchange status ID, but
        # the exchange rows already cover the entire expected exit quantity.
        "tp_fill_events": [{"order_id": "LOCAL_TP_ID", "leg": "TP1", "executed_qty_total": 0.5, "avg_price": 103.0, "ts": 200}],
    }
    fill_accounting = {
        "entry_qty_confirmed": 1.0,
        "fills": [
            {"order_id": "TP1_EXCHANGE", "role": "TP1", "qty": 0.5, "price": 103.0, "time_ms": 100},
            {"order_id": "BE_EXCHANGE", "role": "BE_SL", "qty": 0.5, "price": 100.0, "time_ms": 300},
        ],
        # Cumulative status rows for these same order IDs must not be added to fill rows.
        "order_execution_evidence": [
            {"order_id": "TP1_EXCHANGE", "role": "TP1", "executed_qty": 0.5, "avg_price": 103.0},
            {"order_id": "BE_EXCHANGE", "role": "BE_SL", "executed_qty": 0.5, "avg_price": 100.0},
        ],
    }
    result = tracker._close_execution_evidence(trade, fill_accounting)

    assert result["status"] == "CONFIRMED"
    assert result["confirmed_exit_qty"] == pytest.approx(1.0)
    assert result["exit_vwap_confirmed"] == pytest.approx(101.5)
    assert len(result["rows"]) == 2
    assert all(row["evidence_source"] == "ALL_FILL_ORDERS" for row in result["rows"])
    assert all(row["order_id"] != "LOCAL_TP_ID" for row in result["rows"])


def test_close_evidence_marks_over_exit_qty_as_mismatch_not_confirmed():
    from event_engine import tracker

    result = tracker._close_execution_evidence({"initial_qty": 1.0, "exit_reason": "BREAK_EVEN"}, {
        "entry_qty_confirmed": 1.0,
        "fills": [
            {"order_id": "TP1", "role": "TP1", "qty": 1.0, "price": 103.0, "time_ms": 100},
        ],
        "order_execution_evidence": [
            {"order_id": "BE_OR_ALIAS", "role": "BE_SL", "executed_qty": 0.5, "avg_price": 100.0},
        ],
    })
    assert result["status"] == "QTY_MISMATCH"
    assert result["authority"] == "UNVERIFIED_CONFLICT"
    assert result["confirmed_exit_qty"] is None
    assert result["observed_exit_qty"] == pytest.approx(1.5)
    assert result["exit_vwap_confirmed"] is None


def test_duplicate_close_journal_does_not_send_duplicate_telegram(monkeypatch):
    from event_engine import tracker

    sends = []
    monkeypatch.setattr(tracker, "_send_tracker_notification", lambda *a, **k: sends.append((a, k)) or True)
    result = tracker._notify_trade_close_if_new(
        journal_inserted=False, event_id="EVT_ALREADY_CLOSED", symbol="AAA-USDT", message="duplicate close",
    )
    assert result is False
    assert sends == []


def test_telegram_chat_ids_are_deduplicated_preserving_order(monkeypatch):
    from event_engine.telegram import _chat_ids

    monkeypatch.setenv("TG_CHAT_IDS", "123, 456;123,789,456")
    monkeypatch.delenv("TG_CHAT_ID", raising=False)
    assert _chat_ids() == ["123", "456", "789"]


def test_whole_trade_vwap_is_used_only_when_exit_quantity_exactly_reconciles():
    from event_engine import tracker

    verified = {
        "status": "VERIFIED_ORDER_EXECUTION",
        "entry_qty_confirmed": 373.1,
        "confirmed_exit_qty": 373.1,
        "exit_vwap_confirmed": 0.02638488608952,
        "gross_realized_pnl_abs": 0.169803,
        "quantity_reconciliation": {"status": "VERIFIED", "initial_qty": 373.1, "confirmed_exit_qty": 373.1},
        "incomplete_order_fill_ids": [],
        "order_execution_evidence_errors": {},
    }
    assert tracker._confirmed_whole_trade_exit_vwap(verified, 373.1) == pytest.approx(0.02638488608952)
    overfilled = {**verified, "confirmed_exit_qty": 559.7, "quantity_reconciliation": {"status": "MISMATCH"}}
    assert tracker._confirmed_whole_trade_exit_vwap(overfilled, 373.1) is None


def test_runtime_artifact_pruning_requires_downloaded_bundle_verification():
    workflow = Path(".github/workflows/event-engine.yml").read_text(encoding="utf-8")
    prune_step = workflow.split("- name: Retain only the two newest runtime-state artifacts", 1)[1].split("- name: Remove legacy runtime data", 1)[0]
    assert "steps.verify_uploaded_runtime_state_bundle.outcome == 'success'" in prune_step


def test_runtime_test_isolation_check_does_not_emit_cascade_error_if_restore_failed():
    workflow = Path(".github/workflows/event-engine.yml").read_text(encoding="utf-8")
    verify_step = workflow.split("- name: Verify tests did not mutate runtime data", 1)[1].split("- name: Record source fingerprint", 1)[0]
    assert "RESTORE_OUTCOME" in verify_step
    assert "Skipping runtime-data isolation verification because runtime-state restoration failed" in verify_step


def test_close_evidence_counts_generic_exit_fills_for_take_profit_full_and_skips_local_alias():
    from event_engine import tracker

    trade = {
        "initial_qty": 0.007,
        "exit_reason": "TAKE_PROFIT_FULL",
        # Legacy TP event used a different local ID, but the exchange fill ledger
        # already accounts for the full position and is the preferred evidence.
        "tp_fill_events": [{
            "order_id": "LOCAL_TP_ALIAS", "leg": "TP1",
            "executed_qty_total": 0.004, "avg_price": 1299.66, "order_status": "FILLED",
        }],
    }
    accounting = {
        "entry_qty_confirmed": 0.007,
        "fills": [
            {"order_id": "TP_FILL_1", "role": "EXIT", "qty": 0.004, "price": 1299.66, "time_ms": 100},
            {"order_id": "TP_FILL_2", "role": "EXIT", "qty": 0.003, "price": 1280.74, "time_ms": 200},
        ],
        "order_execution_evidence": [
            {"order_id": "TP_FILL_1", "role": "TP1", "executed_qty": 0.004, "avg_price": 1299.66, "order_status": "FILLED"},
        ],
    }
    result = tracker._close_execution_evidence(trade, accounting)
    assert result["status"] == "CONFIRMED"
    assert result["confirmed_exit_qty"] == pytest.approx(0.007)
    assert result["exit_vwap_confirmed"] == pytest.approx((0.004 * 1299.66 + 0.003 * 1280.74) / 0.007)
    assert len(result["rows"]) == 2
    assert all(row["evidence_source"] == "ALL_FILL_ORDERS" for row in result["rows"])
    assert all(row["order_id"] != "LOCAL_TP_ALIAS" for row in result["rows"])


def test_close_evidence_does_not_claim_confirmation_when_expected_trade_qty_is_unknown():
    from event_engine import tracker

    result = tracker._close_execution_evidence({"exit_reason": "TAKE_PROFIT_FULL"}, {
        "fills": [{"order_id": "TP1", "role": "TP1", "qty": 1.0, "price": 100.0, "time_ms": 100}],
        "order_execution_evidence": [],
    })
    assert result["status"] == "QTY_UNVERIFIABLE"
    assert result["authority"] == "UNVERIFIED_EXPECTED_EXIT_QTY_UNKNOWN"
    assert result["confirmed_exit_qty"] is None
    assert result["exit_vwap_confirmed"] is None
    assert result["observed_exit_qty"] == pytest.approx(1.0)
    assert result["exit_vwap_observed"] == pytest.approx(100.0)
