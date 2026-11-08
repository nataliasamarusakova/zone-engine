from __future__ import annotations

import pytest


@pytest.mark.parametrize("module_name", ["run_once", "tracker"])
def test_emergency_cleanup_does_not_report_success_while_cancelled_order_remains_open(
    monkeypatch, module_name,
):
    from event_engine import bingx, tracker
    import run_once

    module = run_once if module_name == "run_once" else tracker
    monkeypatch.setattr(module, "get_position_directional", lambda *_a, **_k: {"status": "not_found"})
    order = {
        "orderId": "SL-STALE",
        "clientOrderId": "EVT_SL_STALE",
        "type": "STOP_MARKET",
    }
    calls = []

    # The initial discovery and every verification read continue to show the
    # order. A cancel acknowledgement alone must not be treated as cleanup.
    monkeypatch.setattr(module, "get_open_protection_directional", lambda *_a, **_k: {
        "status": "ok", "sl_orders": [dict(order)], "tp_orders": [],
    })
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *_a, **_k: {
        "status": "ok", "sl_orders": [dict(order)], "tp_orders": [],
    })
    fake_cancel = lambda *_a, **_k: calls.append("cancel") or {"code": 0}
    monkeypatch.setattr(module, "cancel_order", fake_cancel)
    monkeypatch.setattr(bingx, "cancel_order", fake_cancel)
    monkeypatch.setattr(bingx.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setenv("EMERGENCY_PROTECTION_CANCEL_VERIFY_ATTEMPTS", "2")

    result = module._cancel_engine_protection_before_emergency_close("AAA-USDT", "LONG")

    assert calls == ["cancel"]
    assert result["status"] == "error"
    assert result["verified"] is False
    assert result["cancelled"] == []
    assert result["remaining_order_ids"] == ["SL-STALE"]
    assert result["errors"]


@pytest.mark.parametrize("module_name", ["run_once", "tracker"])
def test_emergency_cleanup_reports_success_only_after_openorders_confirms_absence(
    monkeypatch, module_name,
):
    from event_engine import bingx, tracker
    import run_once

    module = run_once if module_name == "run_once" else tracker
    monkeypatch.setattr(module, "get_position_directional", lambda *_a, **_k: {"status": "not_found"})
    order = {
        "orderId": "TP-STALE",
        "clientOrderId": "EVT_TP_STALE",
        "type": "TAKE_PROFIT_MARKET",
    }
    calls = []
    rows = iter([
        {"status": "ok", "sl_orders": [], "tp_orders": [dict(order)]},  # discovery
        {"status": "ok", "sl_orders": [], "tp_orders": [dict(order)]},  # pre-cancel
        {"status": "ok", "sl_orders": [], "tp_orders": []},  # post-cancel
    ])

    def read_orders(*_a, **_k):
        try:
            return next(rows)
        except StopIteration:
            return {"status": "ok", "sl_orders": [], "tp_orders": []}

    monkeypatch.setattr(module, "get_open_protection_directional", read_orders)
    monkeypatch.setattr(bingx, "get_open_protection_directional", read_orders)
    fake_cancel = lambda *_a, **_k: calls.append("cancel") or {"code": 0}
    monkeypatch.setattr(module, "cancel_order", fake_cancel)
    monkeypatch.setattr(bingx, "cancel_order", fake_cancel)
    monkeypatch.setattr(bingx.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setenv("EMERGENCY_PROTECTION_CANCEL_VERIFY_ATTEMPTS", "2")

    result = module._cancel_engine_protection_before_emergency_close("AAA-USDT", "LONG")

    assert calls == ["cancel"]
    assert result["status"] == "ok"
    assert result["verified"] is True
    assert result["cancelled"] == ["TP-STALE"]
    assert result["remaining_order_ids"] == []


def test_run_once_emergency_cleanup_respects_known_order_id_scope(monkeypatch):
    import run_once
    from event_engine import bingx
    monkeypatch.setattr(run_once, "get_position_directional", lambda *_a, **_k: {"status": "not_found"})

    rows = [{
        "status": "ok",
        "sl_orders": [
            {"orderId": "SL-OWNED", "clientOrderId": "EVT_SL_OWNED"},
            {"orderId": "SL-OTHER", "clientOrderId": "EVT_SL_OTHER"},
            {"orderId": "SL-MANUAL", "clientOrderId": "MANUAL"},
        ],
        "tp_orders": [],
    }]
    rows += [rows[0], {
        "status": "ok",
        "sl_orders": [
            {"orderId": "SL-OTHER", "clientOrderId": "EVT_SL_OTHER"},
            {"orderId": "SL-MANUAL", "clientOrderId": "MANUAL"},
        ],
        "tp_orders": [],
    }]
    reads = iter(rows)
    cancelled = []
    def read_orders(*_a, **_k):
        return next(reads, rows[-1])
    monkeypatch.setattr(run_once, "get_open_protection_directional", read_orders)
    fake_cancel = lambda _s, oid: cancelled.append(oid) or {"code": 0}
    monkeypatch.setattr(run_once, "cancel_order", fake_cancel)
    monkeypatch.setattr(bingx, "cancel_order", fake_cancel)
    monkeypatch.setattr(bingx.time, "sleep", lambda *_a, **_k: None)

    result = run_once._cancel_engine_protection_before_emergency_close(
        "AAA-USDT", "LONG", known_order_ids={"SL-OWNED"},
    )

    assert result["verified"] is True
    assert cancelled == ["SL-OWNED"]


def test_cancel_transport_timeout_is_reconciled_by_confirmed_openorders_absence(monkeypatch):
    from event_engine import bingx

    order = {"orderId": "SL-TIMEOUT", "clientOrderId": "EVT_SL_TIMEOUT"}
    reads = iter([
        {"status": "ok", "sl_orders": [dict(order)], "tp_orders": []},
        {"status": "ok", "sl_orders": [], "tp_orders": []},
    ])
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *_a, **_k: next(reads))
    monkeypatch.setattr(bingx, "cancel_order", lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("read timeout after exchange accepted cancel")))
    monkeypatch.setattr(bingx.time, "sleep", lambda *_a, **_k: None)

    result = bingx._cancel_engine_protection_orders_verified_detailed(
        "AAA-USDT", "LONG", ["SL-TIMEOUT"], attempts=2,
    )

    assert result["status"] == "ok"
    assert result["verified"] is True
    assert result["cancelled"] == ["SL-TIMEOUT"]
    assert result["remaining_order_ids"] == []


def test_post_cancel_openorders_failure_never_claims_verified_cleanup(monkeypatch):
    from event_engine import bingx

    order = {"orderId": "TP-UNKNOWN", "clientOrderId": "EVT_TP_UNKNOWN"}
    reads = iter([
        {"status": "ok", "sl_orders": [], "tp_orders": [dict(order)]},
        {"status": "error", "error": "temporary openOrders timeout"},
        {"status": "error", "error": "temporary openOrders timeout"},
    ])
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *_a, **_k: next(reads))
    monkeypatch.setattr(bingx, "cancel_order", lambda *_a, **_k: {"code": 0})
    monkeypatch.setattr(bingx.time, "sleep", lambda *_a, **_k: None)

    result = bingx._cancel_engine_protection_orders_verified_detailed(
        "AAA-USDT", "LONG", ["TP-UNKNOWN"], attempts=2,
    )

    assert result["status"] == "error"
    assert result["verified"] is False
    assert result["cancelled"] == []
    assert result["remaining_order_ids"] == ["TP-UNKNOWN"]
    assert result["errors"]


def test_emergency_cleanup_guard_refuses_cancel_when_position_reappears(monkeypatch):
    from event_engine import bingx

    order = {"orderId": "SL-REOPENED", "clientOrderId": "EVT_SL_REOPENED"}
    calls = []
    reads = iter([
        {"status": "ok", "sl_orders": [dict(order)], "tp_orders": []},
        {"status": "ok", "sl_orders": [dict(order)], "tp_orders": []},
    ])
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *_a, **_k: next(reads))
    monkeypatch.setattr(bingx, "cancel_order", lambda *_a, **_k: calls.append("cancel") or {"code": 0})
    result = bingx._cancel_engine_protection_orders_verified_detailed(
        "AAA-USDT", "LONG", ["SL-REOPENED"], attempts=1,
        pre_cancel_guard=lambda: False,
    )

    assert calls == []
    assert result["verified"] is False
    assert result["remaining_order_ids"] == ["SL-REOPENED"]
    assert any("position is not authoritatively flat" in item for item in result["errors"])


def test_pending_emergency_cleanup_is_retried_after_restart_only_when_position_is_flat(
    monkeypatch, tmp_path,
):
    from event_engine import tracker, telemetry

    journal = tmp_path / "emergency_exits.jsonl"
    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", journal)
    telemetry.record_emergency_exit(
        event_id="EVT_RECOVERY_CLEANUP", attempt_id="ATT_RECOVERY_CLEANUP",
        position_id="POS_RECOVERY_CLEANUP", symbol="AAA-USDT", direction="LONG",
        reason="TEST_ROLLBACK", requested_qty=1.0,
        result={"status": "closed_verified", "attempts": [], "verification": {"status": "not_found"}},
    )

    position_reads = []
    def flat_position(*_a, **_k):
        position_reads.append("read")
        return {"status": "not_found"}

    order = {"orderId": "SL-ORPHAN", "clientOrderId": "EVT_SL_ORPHAN"}
    rows = iter([
        {"status": "ok", "sl_orders": [dict(order)], "tp_orders": []},  # discovery
        {"status": "ok", "sl_orders": [dict(order)], "tp_orders": []},  # pre-cancel
        {"status": "ok", "sl_orders": [], "tp_orders": []},  # post-cancel
    ])
    cancelled = []
    monkeypatch.setattr(tracker, "get_position_directional", flat_position)
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *_a, **_k: next(rows))
    monkeypatch.setattr(tracker, "cancel_order", lambda _s, oid: cancelled.append(oid) or {"code": 0})
    monkeypatch.setattr(tracker.time, "sleep", lambda *_a, **_k: None)

    summary = tracker.reconcile_pending_emergency_protection_cleanup(now_ms=2_000_000_000_000)

    assert summary["attempted"] == 1
    assert summary["verified"] == 1
    assert cancelled == ["SL-ORPHAN"]
    assert len(position_reads) >= 2  # initial flat read + guard immediately before cancel
    records = [__import__("json").loads(line) for line in journal.read_text().splitlines()]
    assert records[-1]["record_type"] == "EMERGENCY_PROTECTION_CLEANUP"
    assert records[-1]["verified"] is True
    assert records[-1]["cleanup_result"]["cancelled"] == ["SL-ORPHAN"]

    # A second run after the success journal is durable is a no-op.
    summary2 = tracker.reconcile_pending_emergency_protection_cleanup(now_ms=2_000_000_001_000)
    assert summary2["attempted"] == 0


def test_pending_emergency_cleanup_is_deferred_when_position_is_open_or_unknown(monkeypatch, tmp_path):
    from event_engine import tracker, telemetry

    journal = tmp_path / "emergency_exits.jsonl"
    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", journal)
    telemetry.record_emergency_exit(
        event_id="EVT_RECOVERY_DEFER", attempt_id=None, position_id="POS_RECOVERY_DEFER",
        symbol="AAA-USDT", direction="SHORT", reason="TEST_ROLLBACK", requested_qty=1.0,
        result={"status": "closed_verified", "attempts": [], "verification": {"status": "not_found"}},
    )
    cleanup_calls = []
    statuses = iter([{"status": "found", "positionAmt": "1"}, {"status": "error", "error": "API timeout"}])
    monkeypatch.setattr(tracker, "get_position_directional", lambda *_a, **_k: next(statuses))
    monkeypatch.setattr(tracker, "_cancel_engine_protection_before_emergency_close", lambda *_a, **_k: cleanup_calls.append("cleanup"))

    first = tracker.reconcile_pending_emergency_protection_cleanup(now_ms=2_000_000_000_000)
    second = tracker.reconcile_pending_emergency_protection_cleanup(now_ms=2_000_000_040_000)

    assert first["deferred_position_open"] == 1
    assert second["deferred_position_unknown"] == 1
    assert cleanup_calls == []


def test_pending_emergency_cleanup_retries_only_remaining_ids_from_previous_result(monkeypatch, tmp_path):
    from event_engine import tracker, telemetry

    journal = tmp_path / "emergency_exits.jsonl"
    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", journal)
    telemetry.record_emergency_exit(
        event_id="EVT_RETRY_IDS", attempt_id=None, position_id="POS_RETRY_IDS",
        symbol="AAA-USDT", direction="LONG", reason="TEST", requested_qty=1.0,
        result={
            "status": "closed_verified", "attempts": [],
            "cleanup_after_flat_verified": {
                "status": "partial", "verified": False,
                "remaining_order_ids": ["SL-LEFT"], "cancelled": ["TP-REMOVED"],
                "errors": ["cancel unresolved"],
            },
        },
    )
    monkeypatch.setattr(tracker, "get_position_directional", lambda *_a, **_k: {"status": "not_found"})
    observed = {}
    monkeypatch.setattr(tracker, "_cancel_engine_protection_before_emergency_close", lambda _s, _d, **kw: observed.update(kw) or {"status": "ok", "verified": True, "cancelled": ["SL-LEFT"], "remaining_order_ids": [], "errors": []})

    result = tracker.reconcile_pending_emergency_protection_cleanup(now_ms=2_000_000_000_000)

    assert result["verified"] == 1
    assert observed["known_order_ids"] == {"SL-LEFT"}
    assert observed["require_flat"] is True


def test_tracker_runs_emergency_cleanup_recovery_even_without_active_trades(monkeypatch):
    from event_engine import tracker

    calls = []
    monkeypatch.setattr(tracker, "_load_active_trades_required", lambda: {})
    monkeypatch.setattr(
        tracker, "reconcile_pending_emergency_protection_cleanup",
        lambda **_kwargs: calls.append("recovery") or {"pending": 0},
    )

    tracker.update_active_trades()

    assert calls == ["recovery"]


def test_failed_cleanup_retry_is_durable_and_can_succeed_on_later_tracker_cycle(monkeypatch, tmp_path):
    from event_engine import tracker, telemetry

    journal = tmp_path / "emergency_exits.jsonl"
    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", journal)
    telemetry.record_emergency_exit(
        event_id="EVT_RETRY_LATER", attempt_id="ATT_RETRY_LATER", position_id="POS_RETRY_LATER",
        symbol="AAA-USDT", direction="LONG", reason="TEST", requested_qty=1.0,
        result={"status": "closed_verified", "attempts": [], "verification": {"status": "not_found"}},
    )
    monkeypatch.setattr(tracker, "get_position_directional", lambda *_a, **_k: {"status": "not_found"})
    attempts = []

    def cleanup(_symbol, _direction, **kwargs):
        attempts.append(kwargs.get("known_order_ids"))
        if len(attempts) == 1:
            return {"status": "error", "verified": False, "cancelled": [], "remaining_order_ids": ["SL-PENDING"], "errors": ["openOrders timeout"]}
        return {"status": "ok", "verified": True, "cancelled": ["SL-PENDING"], "remaining_order_ids": [], "errors": []}

    monkeypatch.setattr(tracker, "_cancel_engine_protection_before_emergency_close", cleanup)
    first = tracker.reconcile_pending_emergency_protection_cleanup(now_ms=2_000_000_000_000)
    second = tracker.reconcile_pending_emergency_protection_cleanup(now_ms=2_000_000_031_000)

    assert first["attempted"] == 1 and first["errors"] == 1
    assert second["attempted"] == 1 and second["verified"] == 1
    assert attempts == [None, {"SL-PENDING"}]
    rows = [__import__("json").loads(line) for line in journal.read_text().splitlines()]
    retry_rows = [row for row in rows if row["record_type"] == "EMERGENCY_PROTECTION_CLEANUP"]
    assert [row["retry_number"] for row in retry_rows] == [1, 2]
    assert retry_rows[0]["verified"] is False
    assert retry_rows[1]["verified"] is True


def test_pending_emergency_cleanup_is_deferred_to_existing_active_trade_owner(monkeypatch, tmp_path):
    from event_engine import tracker, telemetry

    journal = tmp_path / "emergency_exits.jsonl"
    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", journal)
    telemetry.record_emergency_exit(
        event_id="EVT_OLD_CLOSE", attempt_id=None, position_id="POS_OLD_CLOSE",
        symbol="AAA-USDT", direction="LONG", reason="TEST", requested_qty=1.0,
        result={"status": "closed_verified", "attempts": [], "verification": {"status": "not_found"}},
    )
    monkeypatch.setattr(tracker, "get_position_directional", lambda *_a, **_k: pytest.fail("active owner path must be handled by normal reconciliation"))
    monkeypatch.setattr(tracker, "_cancel_engine_protection_before_emergency_close", lambda *_a, **_k: pytest.fail("must not cancel protection owned by current trade"))

    result = tracker.reconcile_pending_emergency_protection_cleanup(
        active_trades={"EVT_NEW": {"symbol": "AAA-USDT", "direction": "LONG", "closed": False}},
        now_ms=2_000_000_000_000,
    )

    assert result["deferred_active_trade_owner"] == 1
    assert result["attempted"] == 0


def test_recovery_uses_each_emergency_events_own_remaining_order_ids(monkeypatch, tmp_path):
    from event_engine import tracker, telemetry

    journal = tmp_path / "emergency_exits.jsonl"
    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", journal)
    for event_id, order_id in (("EVT_RECOVERY_A", "SL-A"), ("EVT_RECOVERY_B", "SL-B")):
        telemetry.record_emergency_exit(
            event_id=event_id, attempt_id=None, position_id=f"POS-{event_id}",
            symbol="AAA-USDT", direction="LONG", reason="TEST", requested_qty=1.0,
            result={"status": "closed_verified", "attempts": [], "verification": {"status": "not_found"}},
        )
        telemetry.record_emergency_protection_cleanup(
            event_id=event_id, attempt_id=None, position_id=f"POS-{event_id}",
            symbol="AAA-USDT", direction="LONG", source="PREVIOUS_ATTEMPT",
            close_event_ts_ms=None, retry_number=1,
            position_check={"status": "not_found"},
            result={
                "status": "partial", "verified": False,
                "cancelled": [], "remaining_order_ids": [order_id],
                "errors": [f"{order_id} remains open"],
            },
        )

    monkeypatch.setattr(tracker, "get_position_directional", lambda *_a, **_k: {"status": "not_found"})
    observed = []

    def cleanup(_symbol, _direction, **kwargs):
        observed.append(kwargs.get("known_order_ids"))
        return {
            "status": "ok", "verified": True, "cancelled": [],
            "already_absent": [], "remaining_order_ids": [], "errors": [],
        }

    monkeypatch.setattr(tracker, "_cancel_engine_protection_before_emergency_close", cleanup)
    summary = tracker.reconcile_pending_emergency_protection_cleanup(now_ms=2_000_000_000_000)

    assert summary["attempted"] == 2
    assert summary["verified"] == 2
    assert observed == [{"SL-A"}, {"SL-B"}]


def test_pending_emergency_cleanup_normalizes_symbol_before_matching_active_trade_owner(monkeypatch, tmp_path):
    from event_engine import tracker, telemetry

    journal = tmp_path / "emergency_exits.jsonl"
    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", journal)
    telemetry.record_emergency_exit(
        event_id="EVT_OLD_FORMAT", attempt_id=None, position_id="POS_OLD_FORMAT",
        symbol="AAA-USDT", direction="LONG", reason="TEST", requested_qty=1.0,
        result={"status": "closed_verified", "attempts": [], "verification": {"status": "not_found"}},
    )
    monkeypatch.setattr(
        tracker, "get_position_directional",
        lambda *_a, **_k: pytest.fail("active trade ownership must be checked before exchange reads"),
    )
    monkeypatch.setattr(
        tracker, "_cancel_engine_protection_before_emergency_close",
        lambda *_a, **_k: pytest.fail("must not cancel a different serialization of the active trade's protection"),
    )

    result = tracker.reconcile_pending_emergency_protection_cleanup(
        active_trades={"EVT_NEW": {"symbol": "AAAUSDT", "direction": "LONG", "closed": False}},
        now_ms=2_000_000_000_000,
    )

    assert result["deferred_active_trade_owner"] == 1
    assert result["attempted"] == 0


def test_cancel_cleanup_waits_for_delayed_openorders_propagation(monkeypatch):
    from event_engine import bingx

    order = {"orderId": "SL-DELAYED", "clientOrderId": "EVT_SL_DELAYED"}
    reads = iter([
        {"status": "ok", "sl_orders": [dict(order)], "tp_orders": []},  # pre-cancel
        {"status": "ok", "sl_orders": [dict(order)], "tp_orders": []},  # propagation lag 1
        {"status": "ok", "sl_orders": [dict(order)], "tp_orders": []},  # propagation lag 2
        {"status": "ok", "sl_orders": [], "tp_orders": []},  # eventually absent
    ])
    cancelled = []
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *_a, **_k: next(reads))
    monkeypatch.setattr(bingx, "cancel_order", lambda _s, oid: cancelled.append(oid) or {"code": 0})
    monkeypatch.setattr(bingx.time, "sleep", lambda *_a, **_k: None)

    result = bingx._cancel_engine_protection_orders_verified_detailed(
        "AAA-USDT", "LONG", ["SL-DELAYED"], attempts=3,
    )

    assert cancelled == ["SL-DELAYED"]
    assert result["status"] == "ok"
    assert result["verified"] is True
    assert result["cancelled"] == ["SL-DELAYED"]


def test_recovery_rechecks_position_if_same_symbol_direction_reopens_between_reads(monkeypatch, tmp_path):
    from event_engine import tracker, telemetry

    journal = tmp_path / "emergency_exits.jsonl"
    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", journal)
    telemetry.record_emergency_exit(
        event_id="EVT_REOPEN_RACE", attempt_id=None, position_id="POS_REOPEN_RACE",
        symbol="AAA-USDT", direction="LONG", reason="TEST", requested_qty=1.0,
        result={"status": "closed_verified", "attempts": [], "verification": {"status": "not_found"}},
    )
    order = {"orderId": "SL-NEW-POSITION", "clientOrderId": "EVT_SL_NEW_POSITION"}
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *_a, **_k: {
        "status": "ok", "sl_orders": [dict(order)], "tp_orders": [],
    })
    position_reads = iter([
        {"status": "not_found"},  # first read says old position is flat
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},  # new same-side position appears before cancel
    ])
    monkeypatch.setattr(tracker, "get_position_directional", lambda *_a, **_k: next(position_reads))
    cancelled = []
    monkeypatch.setattr(tracker, "cancel_order", lambda _s, oid: cancelled.append(oid) or {"code": 0})
    monkeypatch.setattr(tracker.time, "sleep", lambda *_a, **_k: None)

    summary = tracker.reconcile_pending_emergency_protection_cleanup(
        active_trades={}, now_ms=2_000_000_000_000,
    )

    assert summary["attempted"] == 1
    assert summary["errors"] == 1
    assert cancelled == []


def test_recovery_retry_budget_does_not_starve_events_behind_rate_limited_rows(
    monkeypatch, tmp_path,
):
    import json
    from event_engine import tracker, telemetry

    journal = tmp_path / "emergency_exits.jsonl"
    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", journal)
    now_ms = 2_000_000_000_000
    rows = [
        {
            "record_type": "EMERGENCY_EXIT", "event_id": "EVT_RATE_LIMITED",
            "symbol": "AAA-USDT", "direction": "LONG", "status": "closed_verified",
            "verified_flat": True,
        },
        {
            "record_type": "EMERGENCY_PROTECTION_CLEANUP", "event_id": "EVT_RATE_LIMITED",
            "symbol": "AAA-USDT", "direction": "LONG", "ts_ms": now_ms - 1_000,
            "status": "partial", "verified": False,
            "cleanup_result": {"status": "partial", "verified": False, "remaining_order_ids": ["SL-A"]},
        },
        {
            "record_type": "EMERGENCY_EXIT", "event_id": "EVT_READY",
            "symbol": "BBB-USDT", "direction": "SHORT", "status": "closed_verified",
            "verified_flat": True,
        },
    ]
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    monkeypatch.setenv("EMERGENCY_CLEANUP_RETRY_INTERVAL_SEC", "30")
    cleanup_symbols = []
    monkeypatch.setattr(tracker, "get_position_directional", lambda *_a, **_k: {"status": "not_found"})
    monkeypatch.setattr(
        tracker, "_cancel_engine_protection_before_emergency_close",
        lambda symbol, _direction, **_kwargs: cleanup_symbols.append(symbol) or {
            "status": "ok", "verified": True, "cancelled": [],
            "already_absent": [], "remaining_order_ids": [], "errors": [],
        },
    )
    monkeypatch.setattr(telemetry, "record_emergency_protection_cleanup", lambda **_kwargs: None)

    summary = tracker.reconcile_pending_emergency_protection_cleanup(
        max_events=1, now_ms=now_ms,
    )

    assert summary["pending"] == 2
    assert summary["attempted"] == 1
    assert summary["verified"] == 1
    assert cleanup_symbols == ["BBB-USDT"]


@pytest.mark.parametrize("deferred_state", ["active_owner", "position_open", "position_unknown"])
def test_recovery_deferrals_do_not_consume_cleanup_retry_budget(
    monkeypatch, tmp_path, deferred_state,
):
    import json
    from event_engine import tracker, telemetry

    journal = tmp_path / "emergency_exits.jsonl"
    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", journal)
    rows = [
        {
            "record_type": "EMERGENCY_EXIT", "event_id": "EVT_DEFERRED",
            "symbol": "AAA-USDT", "direction": "LONG", "status": "closed_verified",
            "verified_flat": True, "ts_ms": 1,
        },
        {
            "record_type": "EMERGENCY_EXIT", "event_id": "EVT_ELIGIBLE",
            "symbol": "BBB-USDT", "direction": "SHORT", "status": "closed_verified",
            "verified_flat": True, "ts_ms": 2,
        },
    ]
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    active_trades = (
        {"EVT_CURRENT": {"symbol": "AAAUSDT", "direction": "LONG", "closed": False}}
        if deferred_state == "active_owner" else {}
    )
    def position(symbol, _direction):
        if symbol == "AAA-USDT":
            if deferred_state == "position_open":
                return {"status": "found", "positionAmt": 0.25}
            if deferred_state == "position_unknown":
                return {"status": "error", "error": "temporary timeout"}
        return {"status": "not_found"}

    cleanup_symbols = []
    monkeypatch.setattr(tracker, "get_position_directional", position)
    monkeypatch.setattr(
        tracker, "_cancel_engine_protection_before_emergency_close",
        lambda symbol, _direction, **_kwargs: cleanup_symbols.append(symbol) or {
            "status": "ok", "verified": True, "cancelled": [],
            "already_absent": [], "remaining_order_ids": [], "errors": [],
        },
    )
    monkeypatch.setattr(telemetry, "record_emergency_protection_cleanup", lambda **_kwargs: None)

    summary = tracker.reconcile_pending_emergency_protection_cleanup(
        max_events=1, now_ms=2_000_000_000_000, active_trades=active_trades,
    )

    assert summary["attempted"] == 1
    assert summary["verified"] == 1
    assert cleanup_symbols == ["BBB-USDT"]
    if deferred_state == "active_owner":
        assert summary["deferred_active_trade_owner"] == 1
    elif deferred_state == "position_open":
        assert summary["deferred_position_open"] == 1
    else:
        assert summary["deferred_position_unknown"] == 1
