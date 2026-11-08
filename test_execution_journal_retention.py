from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from event_engine import data_retention as retention


JOURNALS = (
    "pre_entry_features.jsonl",
    "execution_quality.jsonl",
    "emergency_exits.jsonl",
)


@pytest.mark.parametrize("journal_name", JOURNALS)
def test_size_guard_plan_retains_recent_unknown_and_protected_rows(journal_name: str):
    now = retention.datetime(2026, 11, 8, 12, 0, tzinfo=retention.timezone.utc)
    old = {
        "ts_ms": int((now - timedelta(days=180)).timestamp() * 1000),
        "event_id": "OLD_EVENT",
    }
    recent = {
        "ts_ms": int((now - timedelta(hours=1)).timestamp() * 1000),
        "event_id": "RECENT_EVENT",
    }
    unknown = {"event_id": "UNKNOWN_TS"}
    protected = {
        "ts_ms": int((now - timedelta(days=180)).timestamp() * 1000),
        "event_id": "ACTIVE_EVENT",
    }

    policies = retention._size_guard_plan(
        Path(journal_name),
        now=now,
        pending_ids=set(),
        pending_bar_cutoffs={},
        protected_observation_ids=set(),
        protected_refs={"ACTIVE_EVENT"},
    )

    assert policies, f"no size-guard retention policy registered for {journal_name}"
    _name, keep = policies[0]
    assert keep(old) is False
    assert keep(recent) is True
    assert keep(unknown) is True
    assert keep(protected) is True


@pytest.mark.parametrize("journal_name", JOURNALS)
def test_size_guard_archives_telemetry_journal_and_rollback_restores_exact_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    journal_name: str,
):
    project = tmp_path / "repo"
    data_dir = project / "data"
    archive_dir = data_dir / "retention_archive"
    data_dir.mkdir(parents=True)
    archive_dir.mkdir(parents=True)

    now = retention.datetime(2026, 11, 8, 12, 0, tzinfo=retention.timezone.utc)
    old_ms = int((now - timedelta(days=180)).timestamp() * 1000)
    recent_ms = int((now - timedelta(hours=1)).timestamp() * 1000)
    rows = [
        {"record_type": "OLD", "ts_ms": old_ms, "event_id": f"OLD-{index}", "payload": "x" * 500}
        for index in range(4)
    ]
    rows.extend([
        {"record_type": "RECENT", "ts_ms": recent_ms, "event_id": "RECENT"},
        {"record_type": "UNKNOWN_TIME", "event_id": "UNKNOWN"},
    ])
    original = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
    journal_path = data_dir / journal_name
    journal_path.write_bytes(original)

    monkeypatch.setattr(retention, "PROJECT_ROOT", project)
    monkeypatch.setattr(retention, "DATA_DIR", data_dir)
    monkeypatch.setattr(retention, "REPO_RETENTION_ARCHIVE_ROOT", archive_dir)
    monkeypatch.setattr(retention, "GITHUB_SIZE_GUARD_BYTES", 1_000)
    monkeypatch.setattr(retention, "GITHUB_SIZE_TARGET_BYTES", 800)
    monkeypatch.setattr(retention, "_validate_state_before", lambda: {})
    monkeypatch.setattr(retention, "_validate_state_unchanged", lambda _before: None)
    monkeypatch.setattr(retention, "_pending_observations", lambda: [])
    monkeypatch.setattr(retention, "_pending_bar_cutoffs", lambda **_kwargs: {})
    monkeypatch.setattr(retention, "_protected_audit_refs", lambda **_kwargs: (set(), set()))

    result = retention.apply_size_guard(now=now, archive_root=archive_dir)

    file_result = result["files"].get(journal_name)
    assert file_result is not None, f"size guard did not inspect {journal_name}"
    assert file_result["triggered"] is True
    assert file_result["failed"] is False
    assert journal_path.stat().st_size <= 800
    compacted_rows = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line]
    compacted_ids = {row["event_id"] for row in compacted_rows}
    assert "RECENT" in compacted_ids
    assert "UNKNOWN" in compacted_ids
    assert all(f"OLD-{index}" not in compacted_ids for index in range(4))

    snapshot_chunks = list(archive_dir.rglob(f"{journal_name}.before.part-*.jsonl.gz"))
    pruned_chunks = list(archive_dir.rglob("*.pruned*.jsonl.gz"))
    assert snapshot_chunks
    assert pruned_chunks

    restored = retention.restore_pre_compaction_snapshots(
        manifest_path=data_dir / "data_retention_manifest.json",
        archive_dir=archive_dir,
    )
    assert restored["status"] == "restored"
    assert journal_path.read_bytes() == original


def test_tp1_partial_to_full_fill_reconciles_exchange_residual_and_activates_be_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from event_engine import tracker

    active_path = tmp_path / "active_trades.json"
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    monkeypatch.setattr(tracker, "fetch_klines", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tracker, "_tracker_parallel_first_tp_read_enabled", lambda: False)
    monkeypatch.setattr(tracker, "_send_tracker_notification", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(tracker, "_enrich_tp_fill_event_from_exchange_history", lambda _symbol, event, **_kwargs: event)

    current_order_evidence = {
        "status": "ok",
        "order_status": "PARTIALLY_FILLED",
        "order_id": "TP1-ORDER",
        "executed_qty": 0.2,
        "avg_price": 103.0,
        "execution_evidence_source": "ORDER_STATUS",
    }
    monkeypatch.setattr(
        tracker,
        "_get_order_execution_evidence",
        lambda *_args, **_kwargs: dict(current_order_evidence),
    )

    position_qty = {"value": 0.8}
    monkeypatch.setattr(
        tracker,
        "get_position_directional",
        lambda *_args, **_kwargs: {
            "status": "found",
            "positionAmt": str(position_qty["value"]),
            "avgPrice": "100",
        },
    )

    # Exercise the real BE replacement routine. Only the exchange boundary is
    # simulated: the old SL remains visible until the new residual-sized BE is
    # accepted and verified, then the old SL is cancelled and disappearance is
    # verified through the same open-orders reader.
    exchange_sl_orders = [{
        "orderId": "SL-ORDER",
        "clientOrderId": "EVT_TP1_OLD_SL",
        "type": "STOP_MARKET",
        "stopPrice": "90.00",
        "origQty": "1.000",
        "positionSide": "BOTH",
        "side": "SELL",
    }]
    exchange_calls: list[tuple] = []

    def fake_open_protection(*_args, **_kwargs):
        exchange_calls.append(("OPEN_ORDERS", tuple(order["orderId"] for order in exchange_sl_orders)))
        return {"status": "ok", "sl_orders": [dict(order) for order in exchange_sl_orders], "tp_orders": []}

    def fake_create_be(method, path, params):
        assert method == "POST"
        assert params["type"] == "STOP_MARKET"
        order = {
            "orderId": "BE-ORDER",
            "clientOrderId": params["clientOrderId"],
            "type": params["type"],
            "status": "NEW",
            "stopPrice": params["stopPrice"],
            "origQty": params["quantity"],
            "positionSide": params["positionSide"],
            "side": params["side"],
            "workingType": "MARK_PRICE",
            "reduceOnly": params.get("reduceOnly"),
            "time": 1_700_000_300_000,
            "updateTime": 1_700_000_301_000,
        }
        exchange_sl_orders.append(order)
        exchange_calls.append(("POST_BE", float(params["quantity"]), tuple(item["orderId"] for item in exchange_sl_orders)))
        return {"code": 0, "data": {"order": dict(order)}}

    def fake_cancel_protection(_symbol, order_id):
        exchange_calls.append(("CANCEL_OLD_SL", str(order_id), tuple(item["orderId"] for item in exchange_sl_orders)))
        exchange_sl_orders[:] = [item for item in exchange_sl_orders if str(item["orderId"]) != str(order_id)]
        return {"code": 0}

    monkeypatch.setattr(tracker, "to_bx_symbol", lambda _symbol: "AAAUSDT")
    monkeypatch.setattr(tracker, "get_contract", lambda _symbol: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "position_side_param", lambda _direction: "BOTH")
    monkeypatch.setattr(tracker, "get_open_protection_directional", fake_open_protection)
    monkeypatch.setattr(tracker, "_request", fake_create_be)
    monkeypatch.setattr(tracker, "cancel_order", fake_cancel_protection)
    monkeypatch.setattr(tracker.time, "sleep", lambda *_args, **_kwargs: None)

    trade = {
        "event_id": "EVT_TP1_INTEGRATION",
        "attempt_id": "ATT_TP1_INTEGRATION",
        "position_id": "POS_TP1_INTEGRATION",
        "symbol": "AAA-USDT",
        "direction": "LONG",
        "name": "AAA",
        "entry_price": 100.0,
        "initial_qty": 1.0,
        "remaining_qty": 1.0,
        "entry_ts": 1_700_000_000_000,
        "closed": False,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1-ORDER", "qty": 0.5}],
        "sl_order": {"order_id": "SL-ORDER", "stop_price": 90.0, "qty": 1.0},
        "hit_legs": [],
        "tp_filled_qty": {},
        "tp_fill_events": [],
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "peak_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "planned_risk_pct": 10.0,
        "planned_weighted_rr": 0.45,
        "effective_weighted_rr": 0.45,
        "setup": {},
        "tp_levels": [],
        "effective_tp_levels": [],
    }
    active_path.write_text(json.dumps({trade["event_id"]: trade}), encoding="utf-8")

    # First cycle: exchange reports only 0.2 of the 0.5 TP1 target filled.
    tracker.update_active_trades()
    after_partial = json.loads(active_path.read_text(encoding="utf-8"))[trade["event_id"]]
    assert after_partial["tp_filled_qty"]["tp1"] == pytest.approx(0.2)
    assert after_partial["realized_pnl_qty"] == pytest.approx(0.2)
    assert after_partial["remaining_qty"] == pytest.approx(0.8)
    assert after_partial.get("be_activated") is not True
    assert [event["delta_qty"] for event in after_partial["tp_fill_events"]] == [pytest.approx(0.2)]
    assert after_partial["tp_fill_events"][0]["avg_price"] == pytest.approx(103.0)
    assert after_partial["tp_fill_events"][0]["avg_price_source"] == "ORDER_CUMULATIVE_AVG_FIRST_FILL"
    assert [item["orderId"] for item in exchange_sl_orders] == ["SL-ORDER"]

    # Second cycle: cumulative exchange execution reaches 0.5 and leaves 0.5 open.
    current_order_evidence.update({"order_status": "FILLED", "executed_qty": 0.5, "avg_price": 104.0})
    position_qty["value"] = 0.5
    tracker.update_active_trades()
    after_full = json.loads(active_path.read_text(encoding="utf-8"))[trade["event_id"]]
    assert after_full["tp_filled_qty"]["tp1"] == pytest.approx(0.5)
    assert after_full["realized_pnl_qty"] == pytest.approx(0.5)
    assert after_full["remaining_qty"] == pytest.approx(0.5)
    assert after_full["be_activated"] is True
    assert after_full["sl_order"]["order_id"] == "BE-ORDER"
    assert after_full["sl_order"]["working_type"] == "MARK_PRICE"
    assert after_full["sl_order"]["exchange_order_status"] == "NEW"
    assert after_full["sl_order"]["reduce_only"] is True
    assert after_full["sl_order"]["exchange_order_time_ms"] == 1_700_000_300_000
    assert after_full["sl_order"]["exchange_order_update_time_ms"] == 1_700_000_301_000
    assert after_full["sl_order"]["verification_observed_at_ms"] is not None
    assert [event["delta_qty"] for event in after_full["tp_fill_events"]] == [pytest.approx(0.2), pytest.approx(0.3)]
    assert after_full["tp_fill_events"][1]["cumulative_avg_price"] == pytest.approx(104.0)
    assert after_full["tp_fill_events"][1]["avg_price"] == pytest.approx((0.5 * 104.0 - 0.2 * 103.0) / 0.3)
    assert after_full["tp_fill_events"][1]["avg_price_source"] == "ORDER_CUMULATIVE_AVG_DIFFERENCED"
    assert after_full["realized_pnl_weighted_sum"] == pytest.approx(2.0)
    post_index = next(index for index, call in enumerate(exchange_calls) if call[0] == "POST_BE")
    be_post = exchange_calls[post_index]
    verified_new_be_index = next(
        index for index, call in enumerate(exchange_calls)
        if call[0] == "OPEN_ORDERS" and call[1] == ("SL-ORDER", "BE-ORDER")
    )
    cancel_index = next(index for index, call in enumerate(exchange_calls) if call[0] == "CANCEL_OLD_SL")
    old_sl_cancel = exchange_calls[cancel_index]
    assert be_post[1] == pytest.approx(0.5)
    assert be_post[2] == ("SL-ORDER", "BE-ORDER")
    assert post_index < verified_new_be_index < cancel_index
    assert old_sl_cancel[1] == "SL-ORDER"
    assert old_sl_cancel[2] == ("SL-ORDER", "BE-ORDER")
    assert [item["orderId"] for item in exchange_sl_orders] == ["BE-ORDER"]
    assert float(exchange_sl_orders[0]["origQty"]) == pytest.approx(0.5)

    # Third cycle: same cumulative FILLED status is seen again; no second delta or BE.
    tracker.update_active_trades()
    after_repeat = json.loads(active_path.read_text(encoding="utf-8"))[trade["event_id"]]
    assert after_repeat["tp_filled_qty"]["tp1"] == pytest.approx(0.5)
    assert after_repeat["realized_pnl_qty"] == pytest.approx(0.5)
    assert after_repeat["remaining_qty"] == pytest.approx(0.5)
    assert after_repeat["be_activated"] is True
    assert len(after_repeat["tp_fill_events"]) == 2
    assert len([call for call in exchange_calls if call[0] == "POST_BE"]) == 1
    assert len([call for call in exchange_calls if call[0] == "CANCEL_OLD_SL"]) == 1


def test_protected_audit_refs_ignore_closed_history_but_keep_active_and_recent_unresolved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    now = retention.datetime(2026, 11, 8, 12, 0, tzinfo=retention.timezone.utc)
    old_ms = int((now - timedelta(days=180)).timestamp() * 1000)
    recent_ms = int((now - timedelta(minutes=30)).timestamp() * 1000)
    stale_ms = int((now - timedelta(hours=3)).timestamp() * 1000)

    (data_dir / "active_trades.json").write_text(json.dumps({
        "EVT_ACTIVE": {
            "event_id": "EVT_ACTIVE",
            "attempt_id": "ATT_ACTIVE",
            "position_id": "POS_ACTIVE",
            "closed": False,
        }
    }), encoding="utf-8")
    (data_dir / "trades.jsonl").write_text("\n".join([
        json.dumps({
            "record_type": "TRADE_OPEN", "event_id": "EVT_CLOSED",
            "attempt_id": "ATT_CLOSED", "ts_ms": old_ms,
            "signal": {"scan_id": "SCAN_CLOSED", "observation_id": "OBS_CLOSED"},
        }),
        json.dumps({
            "record_type": "TRADE_CLOSE", "event_id": "EVT_CLOSED",
            "attempt_id": "ATT_CLOSED", "closed_ts": old_ms,
        }),
        json.dumps({
            "record_type": "TRADE_OPEN", "event_id": "EVT_ACTIVE",
            "attempt_id": "ATT_ACTIVE", "ts_ms": old_ms,
            "signal": {"scan_id": "SCAN_ACTIVE", "observation_id": "OBS_ACTIVE"},
        }),
        json.dumps({
            "record_type": "TRADE_ROLLBACK", "event_id": "EVT_ROLLBACK_PENDING",
            "status": "opened_registration_failed_close_unverified", "ts_ms": recent_ms,
        }),
        json.dumps({
            "record_type": "TRADE_ROLLBACK", "event_id": "EVT_ROLLBACK_CLOSED",
            "status": "opened_then_emergency_closed", "ts_ms": old_ms,
        }),
    ]) + "\n", encoding="utf-8")
    (data_dir / "entry_decisions.jsonl").write_text("\n".join([
        json.dumps({
            "record_type": "ENTRY_DECISION", "event_id": "EVT_CLOSED",
            "attempt_id": "ATT_CLOSED", "terminal": False,
            "decision_ts": now.replace(year=2026, month=5, day=12).isoformat(),
            "scan_id": "SCAN_CLOSED",
        }),
        json.dumps({
            "record_type": "ENTRY_DECISION", "event_id": "EVT_PENDING",
            "attempt_id": "ATT_PENDING", "terminal": False,
            "ts": recent_ms, "scan_id": "SCAN_PENDING",
        }),
        json.dumps({
            "record_type": "ENTRY_DECISION", "event_id": "EVT_STALE_PENDING",
            "attempt_id": "ATT_STALE_PENDING", "terminal": False,
            "ts": stale_ms, "scan_id": "SCAN_STALE_PENDING",
        }),
        json.dumps({
            "record_type": "ENTRY_DECISION", "event_id": "EVT_UNKNOWN_PENDING",
            "attempt_id": "ATT_UNKNOWN_PENDING", "terminal": False,
            "scan_id": "SCAN_UNKNOWN_PENDING",
        }),
    ]) + "\n", encoding="utf-8")

    monkeypatch.setattr(retention, "DATA_DIR", data_dir)
    protected_observations, protected_refs = retention._protected_audit_refs(now=now)

    assert "EVT_CLOSED" not in protected_refs
    assert "ATT_CLOSED" not in protected_refs
    assert "SCAN_CLOSED" not in protected_refs
    assert "OBS_CLOSED" not in protected_observations
    assert {"EVT_ACTIVE", "ATT_ACTIVE", "POS_ACTIVE", "SCAN_ACTIVE"}.issubset(protected_refs)
    assert "OBS_ACTIVE" in protected_observations
    assert {"EVT_PENDING", "ATT_PENDING", "SCAN_PENDING"}.issubset(protected_refs)
    assert not ({"EVT_STALE_PENDING", "ATT_STALE_PENDING", "SCAN_STALE_PENDING"} & protected_refs)
    assert {"EVT_UNKNOWN_PENDING", "ATT_UNKNOWN_PENDING", "SCAN_UNKNOWN_PENDING"}.issubset(protected_refs)
    assert "EVT_ROLLBACK_PENDING" in protected_refs
    assert "EVT_ROLLBACK_CLOSED" not in protected_refs


def test_size_guard_compacts_closed_event_telemetry_without_monkeypatching_protected_refs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    project = tmp_path / "repo"
    data_dir = project / "data"
    archive_dir = data_dir / "retention_archive"
    archive_dir.mkdir(parents=True)
    now = retention.datetime(2026, 11, 8, 12, 0, tzinfo=retention.timezone.utc)
    old_ms = int((now - timedelta(days=180)).timestamp() * 1000)
    recent_ms = int((now - timedelta(hours=1)).timestamp() * 1000)

    (data_dir / "active_trades.json").write_text(json.dumps({
        "EVT_ACTIVE": {"event_id": "EVT_ACTIVE", "attempt_id": "ATT_ACTIVE", "position_id": "POS_ACTIVE", "closed": False}
    }), encoding="utf-8")
    (data_dir / "trades.jsonl").write_text("\n".join([
        json.dumps({"record_type": "TRADE_OPEN", "event_id": "EVT_CLOSED", "ts_ms": old_ms}),
        json.dumps({"record_type": "TRADE_CLOSE", "event_id": "EVT_CLOSED", "closed_ts": old_ms}),
        json.dumps({"record_type": "TRADE_OPEN", "event_id": "EVT_ACTIVE", "ts_ms": old_ms}),
    ]) + "\n", encoding="utf-8")
    (data_dir / "entry_decisions.jsonl").write_text(json.dumps({
        "record_type": "ENTRY_DECISION", "event_id": "EVT_CLOSED", "terminal": True, "ts": old_ms,
    }) + "\n", encoding="utf-8")

    rows = [
        {"event_id": "EVT_CLOSED", "ts_ms": old_ms, "payload": "x" * 500}
        for _ in range(4)
    ]
    rows.extend([
        {"event_id": "EVT_ACTIVE", "ts_ms": old_ms, "payload": "active"},
        {"event_id": "EVT_RECENT", "ts_ms": recent_ms, "payload": "recent"},
        {"event_id": "EVT_UNKNOWN", "payload": "unknown timestamp"},
    ])
    journal = data_dir / "pre_entry_features.jsonl"
    original = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
    journal.write_bytes(original)

    monkeypatch.setattr(retention, "PROJECT_ROOT", project)
    monkeypatch.setattr(retention, "DATA_DIR", data_dir)
    monkeypatch.setattr(retention, "REPO_RETENTION_ARCHIVE_ROOT", archive_dir)
    monkeypatch.setattr(retention, "GITHUB_SIZE_GUARD_BYTES", 1_000)
    monkeypatch.setattr(retention, "GITHUB_SIZE_TARGET_BYTES", 800)
    monkeypatch.setattr(retention, "_validate_state_before", lambda: {})
    monkeypatch.setattr(retention, "_validate_state_unchanged", lambda _before: None)
    monkeypatch.setattr(retention, "_pending_observations", lambda: [])
    monkeypatch.setattr(retention, "_pending_bar_cutoffs", lambda **_kwargs: {})

    result = retention.apply_size_guard(now=now, archive_root=archive_dir)
    file_result = result["files"]["pre_entry_features.jsonl"]
    assert file_result["failed"] is False
    assert journal.stat().st_size <= 800
    compacted = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line]
    compacted_events = [row["event_id"] for row in compacted]
    assert "EVT_ACTIVE" in compacted_events
    assert "EVT_RECENT" in compacted_events
    assert "EVT_UNKNOWN" in compacted_events
    assert "EVT_CLOSED" not in compacted_events

    restored = retention.restore_pre_compaction_snapshots(
        manifest_path=data_dir / "data_retention_manifest.json",
        archive_dir=archive_dir,
    )
    assert restored["status"] == "restored"
    assert journal.read_bytes() == original


def test_terminal_cancelled_tp_partial_fill_is_accounted_without_activating_be(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from event_engine import tracker

    active_path = tmp_path / "active_trades.json"
    event_id = "EVT_TP1_CANCELLED_PARTIAL"
    trade = {
        "event_id": event_id,
        "attempt_id": "ATT_TP1_CANCELLED_PARTIAL",
        "position_id": "POS_TP1_CANCELLED_PARTIAL",
        "symbol": "AAA-USDT",
        "direction": "LONG",
        "name": "AAA",
        "entry_price": 100.0,
        "initial_qty": 1.0,
        "remaining_qty": 0.8,
        "entry_ts": 1_700_000_000_000,
        "closed": False,
        "tp_orders": [{
            "leg": "tp1", "order_id": "TP1-ORDER", "qty": 0.5,
            "reconciliation_state": "PENDING_PARTIAL_FILL",
            "partial_fill_since_ts": 1_700_000_100_000,
            "last_exchange_executed_qty": 0.3,
            "last_exchange_reported_executed_qty": 0.0,
            "last_exchange_avg_price": 103.0,
            "last_exchange_avg_price_qty": 0.3,
        }],
        "sl_order": {"order_id": "SL-ORDER", "stop_price": 90.0, "qty": 0.8},
        "hit_legs": [],
        "completed_tp_legs": [],
        "tp_filled_qty": {"tp1": 0.2},
        "tp_fill_events": [{
            "leg": "tp1", "order_id": "TP1-ORDER", "executed_qty_total": 0.2,
            "delta_qty": 0.2, "avg_price": 103.0,
        }],
        "realized_pnl_qty": 0.2,
        "realized_pnl_weighted_sum": 0.6,
        "peak_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "planned_risk_pct": 10.0,
        "planned_weighted_rr": 0.45,
        "effective_weighted_rr": 0.45,
        "setup": {},
        "tp_levels": [],
        "effective_tp_levels": [],
    }
    active_path.write_text(json.dumps({event_id: trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    monkeypatch.setattr(tracker, "fetch_klines", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tracker, "_tracker_parallel_first_tp_read_enabled", lambda: False)
    monkeypatch.setattr(tracker, "_send_tracker_notification", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(tracker, "_enrich_tp_fill_event_from_exchange_history", lambda _symbol, event, **_kwargs: event)
    monkeypatch.setattr(tracker, "_get_order_execution_evidence", lambda *_args, **_kwargs: {
        "status": "ok",
        "order_id": "TP1-ORDER",
        "order_status": "CANCELED",
        "executed_qty": 0.0,
        "avg_price": 0.0,
        "execution_evidence_source": "ORDER_STATUS",
        "side": "SELL",
        "position_side": "LONG",
        "order_type": "TAKE_PROFIT_MARKET",
        "working_type": "MARK_PRICE",
        "reduce_only": None,
        "close_position": None,
        "time_in_force": "GTC",
        "time_ms": 1_700_000_200_000,
        "update_time_ms": 1_700_000_201_000,
        "executed_qty_known": True,
    })
    monkeypatch.setattr(tracker, "get_position_directional", lambda *_args, **_kwargs: {
        "status": "found", "positionAmt": "0.7", "avgPrice": "100",
    })

    tracker.update_active_trades()

    saved = json.loads(active_path.read_text(encoding="utf-8"))[event_id]
    assert saved["tp_filled_qty"]["tp1"] == pytest.approx(0.3)
    assert saved["realized_pnl_qty"] == pytest.approx(0.3)
    assert saved["remaining_qty"] == pytest.approx(0.7)
    assert saved["tp_orders"][0]["reconciliation_state"] == "PARTIAL_FILL_TERMINAL"
    assert saved["tp_orders"][0]["partial_fill_terminal_status"] == "CANCELED"
    assert saved["tp_fill_events"][-1]["order_status"] == "CANCELED"
    assert saved["tp_fill_events"][-1]["executed_qty_total"] == pytest.approx(0.3)
    assert saved["tp_fill_events"][-1]["delta_qty"] == pytest.approx(0.1)
    assert saved["tp_fill_events"][-1]["order_type"] == "TAKE_PROFIT_MARKET"
    assert saved["tp_fill_events"][-1]["working_type"] == "MARK_PRICE"
    assert saved["tp_fill_events"][-1]["reduce_only"] is None
    assert saved["tp_fill_events"][-1]["close_position"] is None
    assert saved["tp_fill_events"][-1]["time_in_force"] == "GTC"
    assert saved["tp_fill_events"][-1]["order_time_ms"] == 1_700_000_200_000
    assert saved["tp_fill_events"][-1]["order_update_time_ms"] == 1_700_000_201_000
    assert saved["tp_fill_events"][-1]["executed_qty_known"] is True
    assert saved["tp_fill_events"][-1]["executed_qty_observed"] == pytest.approx(0.0)
    assert saved["tp_fill_events"][-1]["executed_qty_source"] == "PERSISTED_EXCHANGE_HIGH_WATER_MARK"
    assert saved["tp_fill_events"][-1]["avg_price_source"] == "CUMULATIVE_ORDER_AVG_FALLBACK_PRIOR_LEDGER_UNVERIFIED"
    assert "tp1" not in saved["hit_legs"]
    assert saved.get("be_activated") is not True
    assert saved.get("be_required") is not True

    # The same stale terminal response cannot account the persisted cumulative
    # quantity a second time.
    event_count = len(saved["tp_fill_events"])
    tracker.update_active_trades()
    repeated = json.loads(active_path.read_text(encoding="utf-8"))[event_id]
    assert repeated["tp_filled_qty"]["tp1"] == pytest.approx(0.3)
    assert repeated["realized_pnl_qty"] == pytest.approx(0.3)
    assert len(repeated["tp_fill_events"]) == event_count
    assert repeated.get("be_activated") is not True


def test_filled_tp_status_with_cumulative_quantity_mismatch_books_only_real_fill_and_does_not_activate_be(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from event_engine import tracker

    event_id = "EVT_TP1_FILLED_QTY_MISMATCH"
    active_path = tmp_path / "active_trades.json"
    trade = {
        "event_id": event_id, "attempt_id": "ATT_FILLED_QTY_MISMATCH",
        "position_id": "POS_FILLED_QTY_MISMATCH", "symbol": "AAA-USDT",
        "direction": "LONG", "entry_price": 100.0, "initial_qty": 1.0,
        "remaining_qty": 0.7, "entry_ts": 1_700_000_000_000, "closed": False,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1-MISMATCH", "qty": 0.5}],
        "sl_order": {"order_id": "SL-KEEP", "stop_price": 90.0, "qty": 0.7},
        "hit_legs": [], "completed_tp_legs": [], "tp_filled_qty": {}, "tp_fill_events": [],
        "realized_pnl_qty": 0.0, "realized_pnl_weighted_sum": 0.0,
        "peak_pnl_pct": 0.0, "mae_pct": 0.0, "max_drawdown_pct": 0.0,
        "planned_risk_pct": 10.0, "planned_weighted_rr": 0.45,
        "effective_weighted_rr": 0.45, "setup": {}, "tp_levels": [], "effective_tp_levels": [],
    }
    active_path.write_text(json.dumps({event_id: trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    monkeypatch.setattr(tracker, "fetch_klines", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tracker, "_tracker_parallel_first_tp_read_enabled", lambda: False)
    monkeypatch.setattr(tracker, "_send_tracker_notification", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(tracker, "_enrich_tp_fill_event_from_exchange_history", lambda _symbol, event, **_kwargs: event)
    monkeypatch.setattr(tracker, "_get_order_execution_evidence", lambda *_args, **_kwargs: {
        "status": "ok", "order_id": "TP1-MISMATCH", "order_status": "FILLED",
        "executed_qty": 0.3, "orig_qty": 0.5, "executed_qty_known": True,
        "avg_price": 103.0, "execution_evidence_source": "ORDER_STATUS",
    })
    monkeypatch.setattr(tracker, "get_position_directional", lambda *_args, **_kwargs: {
        "status": "found", "positionAmt": "0.7", "avgPrice": "100",
    })

    tracker.update_active_trades()

    saved = json.loads(active_path.read_text(encoding="utf-8"))[event_id]
    assert saved["tp_filled_qty"]["tp1"] == pytest.approx(0.3)
    assert saved["realized_pnl_qty"] == pytest.approx(0.3)
    assert saved["remaining_qty"] == pytest.approx(0.7)
    assert saved["tp_orders"][0]["reconciliation_state"] == "PARTIAL_FILL_TERMINAL"
    assert saved["tp_orders"][0]["partial_fill_terminal_status"] == "FILLED_QTY_MISMATCH"
    assert saved["tp_fill_events"][-1]["delta_qty"] == pytest.approx(0.3)
    assert "tp1" not in saved["hit_legs"]
    assert "tp1" not in saved["completed_tp_legs"]
    assert saved.get("be_activated") is not True
