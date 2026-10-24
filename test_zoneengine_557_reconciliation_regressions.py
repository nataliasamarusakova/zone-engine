from __future__ import annotations

import pytest


def test_get_order_by_identity_recovers_mismatched_exchange_order_id_by_exact_client_id(monkeypatch):
    from event_engine import bingx

    calls = []
    def lookup(symbol, order_id=None, *, client_order_id=None):
        calls.append((symbol, order_id, client_order_id))
        if order_id:
            return {
                "status": "error", "error": "orderId mismatch",
                "requested_order_id": order_id, "reported_order_id": "OTHER_ORDER",
            }
        return {
            "status": "ok", "order_id": "ACTUAL_ORDER", "order_status": "FILLED",
            "symbol": "AAAUSDT", "client_order_id": "evt_edu_tp1",
            "exchange_reported_client_order_id": "evt_edu_tp1",
            "exchange_reported_symbol": "AAAUSDT",
            "executed_qty": 76.0, "avg_price": 0.06796,
        }

    out = bingx.get_order_by_identity(
        "AAA-USDT", "OLD_ORDER", client_order_id="EVT_EDU_TP1", order_lookup=lookup,
    )

    assert out["status"] == "ok"
    assert out["identity_resolution"] == "CLIENT_ORDER_ID_FALLBACK"
    assert out["requested_order_id"] == "OLD_ORDER"
    assert out["order_id"] == "ACTUAL_ORDER"
    assert calls == [("AAA-USDT", "OLD_ORDER", None), ("AAA-USDT", None, "EVT_EDU_TP1")]
    assert "exchange_reported_client_order_id" not in out


def test_get_order_by_identity_rejects_wrong_or_missing_client_id_on_fallback():
    from event_engine import bingx

    for reported_cid in ("EVT_SOME_OTHER_ORDER", ""):
        def lookup(symbol, order_id=None, *, client_order_id=None):
            if order_id:
                return {"status": "error", "error": "orderId mismatch"}
            return {
                "status": "ok", "order_id": "OTHER_ORDER", "order_status": "FILLED",
                "symbol": "AAAUSDT", "client_order_id": reported_cid,
                "exchange_reported_client_order_id": reported_cid,
                "exchange_reported_symbol": "AAAUSDT",
            }
        out = bingx.get_order_by_identity(
            "AAA-USDT", "OLD_ORDER", client_order_id="EVT_EXPECTED", order_lookup=lookup,
        )
        assert out["status"] == "error"
        assert "clientOrderId fallback identity mismatch" in out["error"]


def test_get_order_by_identity_rejects_wrong_symbol_even_when_client_id_matches():
    from event_engine import bingx

    def lookup(symbol, order_id=None, *, client_order_id=None):
        if order_id:
            return {"status": "error", "error": "orderId mismatch"}
        return {
            "status": "ok", "order_id": "OTHER_ORDER", "order_status": "FILLED",
            "symbol": "BBB-USDT", "client_order_id": "EVT_EXPECTED",
            "exchange_reported_client_order_id": "EVT_EXPECTED",
            "exchange_reported_symbol": "BBB-USDT",
        }
    out = bingx.get_order_by_identity("AAA-USDT", "OLD_ORDER", client_order_id="EVT_EXPECTED", order_lookup=lookup)
    assert out["status"] == "error"
    assert "symbol mismatch" in out["error"]


def test_execution_evidence_queries_fill_history_using_resolved_system_order_id(monkeypatch):
    from event_engine import tracker

    calls = []
    def lookup(symbol, order_id=None, *, client_order_id=None):
        if order_id:
            return {"status": "error", "error": "orderId mismatch"}
        return {
            "status": "ok", "order_id": "ACTUAL_ORDER", "order_status": "FILLED",
            "symbol": "AAAUSDT", "client_order_id": "EVT_TP1",
            "exchange_reported_client_order_id": "EVT_TP1",
            "exchange_reported_symbol": "AAAUSDT",
            "executed_qty": 0.0, "avg_price": 0.0, "executed_qty_known": True,
            "time_ms": 100, "update_time_ms": 200, "orig_qty": 2.0,
            "trigger_price": 101.0,
        }
    monkeypatch.setattr(tracker, "get_order", lookup)
    monkeypatch.setattr(tracker, "get_fill_orders", lambda symbol, start, end, *, order_id=None, limit=1000: calls.append(order_id) or [
        {"order_id": "ACTUAL_ORDER", "trade_id": "FILL-1", "qty": 2.0, "price": 101.0, "fee": None, "realized_pnl": None, "time_ms": 150},
    ])
    out = tracker._get_order_execution_evidence("AAA-USDT", "OLD_ORDER", entry_ts=50, client_order_id="EVT_TP1")
    assert out["status"] == "ok"
    assert out["identity_resolution"] == "CLIENT_ORDER_ID_FALLBACK"
    assert out["order_id"] == "ACTUAL_ORDER"
    assert calls == ["ACTUAL_ORDER"]
    assert out["execution_evidence_source"] == "ALL_FILL_ORDERS"
    assert out["executed_qty"] == pytest.approx(2.0)


def test_private_block_reason_distinguishes_credentials_from_preflight_failure():
    import run_once

    assert run_once._private_block_decision_reason({"status": "BLOCKED_MISSING_CREDENTIALS"}) == "missing_private_credentials"
    assert run_once._private_block_decision_reason({"status": "BLOCKED_PRIVATE_PREFLIGHT"}) == "private_preflight_unresolved"
    assert run_once._private_block_decision_reason({"status": "BLOCKED_OTHER"}) == "private_execution_blocked"


def test_collect_trade_fill_accounting_does_not_add_order_status_again_after_client_id_resolves_fill_id(monkeypatch):
    from event_engine import tracker

    monkeypatch.setattr(tracker, "credentials_available", lambda: True)

    def get_order(symbol, order_id=None, *, client_order_id=None):
        if client_order_id:
            assert client_order_id == "EVT_TP1"
            return {
                "status": "ok", "order_id": "ACTUAL_TP_ORDER", "order_status": "FILLED",
                "symbol": "AAAUSDT", "client_order_id": "EVT_TP1",
                "exchange_reported_client_order_id": "EVT_TP1", "exchange_reported_symbol": "AAAUSDT",
                "executed_qty": 1.0, "avg_price": 105.0,
                "identity_resolution": "CLIENT_ORDER_ID_FALLBACK",
            }
        if order_id == "ENTRY_ORDER":
            return {
                "status": "ok", "order_id": "ENTRY_ORDER", "order_status": "FILLED",
                "symbol": "AAAUSDT", "client_order_id": "EVT_ENTRY",
                "executed_qty": 1.0, "avg_price": 100.0,
            }
        return {"status": "error", "error": "orderId mismatch"}

    monkeypatch.setattr(tracker, "get_order", get_order)
    exact_calls = []

    def get_fills(symbol, start, end, *, order_id=None, limit=1000):
        if order_id is None:
            return [{"order_id": "ENTRY_ORDER", "trade_id": "ENTRY_FILL", "qty": 1.0, "price": 100.0, "fee": -0.01, "realized_pnl": 0.0, "time_ms": 100}]
        exact_calls.append(order_id)
        if order_id == "ACTUAL_TP_ORDER":
            return [{"order_id": "ACTUAL_TP_ORDER", "trade_id": "TP_FILL", "qty": 1.0, "price": 105.0, "fee": -0.01, "realized_pnl": 5.0, "time_ms": 200}]
        return []

    monkeypatch.setattr(tracker, "get_fill_orders", get_fills)
    trade = {
        "event_id": "EVT_ALIAS_ACCOUNTING", "symbol": "AAA-USDT", "direction": "LONG",
        "entry_ts": 50, "entry_order": {"order_id": "ENTRY_ORDER", "client_order_id": "EVT_ENTRY"},
        "tp_orders": [{"leg": "tp1", "order_id": "OLD_TP_ORDER", "client_order_id": "EVT_TP1"}],
        "setup": {"protection_result": {"tp_orders": [{"leg": "tp1", "order_id": "OLD_TP_ORDER", "client_order_id": "EVT_TP1"}]}},
        "sl_order": {}, "exit_reason": "TAKE_PROFIT_FULL", "initial_qty": 1.0,
    }

    result = tracker._collect_trade_fill_accounting(trade, 1000)

    assert result["status"] == "VERIFIED"
    assert result["entry_qty_confirmed"] == pytest.approx(1.0)
    assert result["confirmed_exit_qty"] == pytest.approx(1.0)
    assert result["gross_realized_pnl_abs"] == pytest.approx(5.0)
    assert exact_calls == ["ACTUAL_TP_ORDER"]
    assert len([row for row in result["fills"] if row.get("role") != "ENTRY"]) == 1
    assert result["order_execution_evidence"] == []


def test_close_evidence_does_not_add_local_tp_event_when_same_client_identity_is_already_observed(monkeypatch):
    from event_engine import tracker

    trade = {
        "initial_qty": 1.0,
        "exit_reason": "BREAK_EVEN",
        "setup": {"protection_result": {"tp_orders": [
            {"leg": "tp1", "order_id": "OLD_TP_ID", "client_order_id": "EVT_TP1"},
        ]}},
        "tp_fill_events": [{
            "order_id": "OLD_TP_ID", "leg": "TP1", "executed_qty_total": 0.5,
            "avg_price": 103.0, "order_status": "FILLED",
        }],
    }
    accounting = {
        "entry_qty_confirmed": 1.0,
        "fills": [],
        "order_execution_evidence": [
            {"order_id": "RESOLVED_TP_ID", "client_order_id": "EVT_TP1", "role": "TP1", "order_status": "FILLED", "executed_qty": 0.3, "avg_price": 103.0},
            {"order_id": "BE_ID", "client_order_id": "EVT_BE", "role": "BE_SL", "order_status": "FILLED", "executed_qty": 0.5, "avg_price": 100.0},
        ],
    }

    result = tracker._close_execution_evidence(trade, accounting)

    assert result["status"] == "PARTIAL"
    assert result["confirmed_exit_qty"] is None
    assert result["observed_exit_qty"] == pytest.approx(0.8)
    assert len(result["rows"]) == 2
    assert all(row["order_id"] != "OLD_TP_ID" for row in result["rows"])



def test_close_evidence_deduplicates_tp_event_and_order_status_by_client_order_id():
    from event_engine import tracker

    trade = {
        "initial_qty": 1.0,
        "exit_reason": "BREAK_EVEN",
        "setup": {"protection_result": {"tp_orders": [
            {"leg": "tp1", "order_id": "OLD_TP_ID", "client_order_id": "EVT_TP1"},
        ]}},
        "tp_fill_events": [{
            "order_id": "OLD_TP_ID", "leg": "TP1", "executed_qty_total": 0.5,
            "avg_price": 103.0, "order_status": "FILLED",
        }],
    }
    accounting = {
        "entry_qty_confirmed": 1.0,
        "fills": [],
        "order_execution_evidence": [
            {"order_id": "RESOLVED_TP_ID", "client_order_id": "evt_tp1", "role": "TP1", "order_status": "FILLED", "executed_qty": 0.5, "avg_price": 103.0},
            {"order_id": "BE_ID", "client_order_id": "EVT_BE", "role": "BE_SL", "order_status": "FILLED", "executed_qty": 0.5, "avg_price": 100.0},
        ],
    }

    result = tracker._close_execution_evidence(trade, accounting)

    assert result["status"] == "CONFIRMED"
    assert result["confirmed_exit_qty"] == pytest.approx(1.0)
    assert result["observed_exit_qty"] == pytest.approx(1.0)
    assert len(result["rows"]) == 2
    assert {row.get("client_order_id", "").casefold() for row in result["rows"]} == {"evt_tp1", "evt_be"}


def test_directly_confirmed_break_even_stop_is_not_mislabeled_stop_loss(monkeypatch):
    from event_engine import tracker

    monkeypatch.setattr(tracker, "_get_filled_order", lambda *a, **k: {
        "status": "ok", "order_status": "FILLED", "order_id": "BE123",
        "client_order_id": "EVT_BE_EVENT", "executed_qty": 1.0,
        "avg_price": 100.2,
    })
    out = tracker._reconcile_historical_exit_order(
        "AAA-USDT", "LONG", 100, 1.0, [],
        {"order_id": "BE123", "client_order_id": "EVT_BE_EVENT"},
        trade={"be_activated": True, "be_order_id": "BE123", "be_client_order_id": "EVT_BE_EVENT"},
    )

    assert out[0] == pytest.approx(100.2)
    assert out[1] == "BREAK_EVEN"
    assert out[2] == "historical_sl_order"
    assert out[3]["order_id"] == "BE123"
