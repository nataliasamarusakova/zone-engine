from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path

import pytest


def _bar(ts_ms: int, close: float, *, symbol: str = "TEST-USDT", provider: str = "binance") -> dict:
    return {
        "record_type": "MARKET_BAR",
        "symbol": symbol,
        "provider": provider,
        "timestamp": ts_ms,
        "close_time": ts_ms + 5 * 60_000,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 10,
    }


def test_observation_cursor_survives_inode_change_when_bytes_match(tmp_path):
    from research_forward import OBS_CURSOR_SCHEMA_VERSION, _cursor_is_valid, _journal_signature

    path = tmp_path / "zone_observations.jsonl"
    payload = b'{"row":"' + b"x" * 80_000 + b'"}\n'
    path.write_bytes(payload)
    cursor = {"schema_version": OBS_CURSOR_SCHEMA_VERSION, "offset": len(payload), "size": len(payload), **_journal_signature(path, len(payload))}
    old_inode = path.stat().st_ino

    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(payload)
    os.replace(replacement, path)
    assert path.stat().st_ino != old_inode
    assert _cursor_is_valid(path, cursor) is True


def test_observation_cursor_rejects_content_change_and_truncation(tmp_path):
    from research_forward import OBS_CURSOR_SCHEMA_VERSION, _cursor_is_valid, _journal_signature

    path = tmp_path / "zone_observations.jsonl"
    payload = b'{"row":"' + b"x" * 80_000 + b'"}\n'
    path.write_bytes(payload)
    cursor = {"schema_version": OBS_CURSOR_SCHEMA_VERSION, "offset": len(payload), "size": len(payload), **_journal_signature(path, len(payload))}

    changed = bytearray(payload)
    changed[9] = ord("y")
    path.write_bytes(changed)
    assert _cursor_is_valid(path, cursor) is False

    path.write_bytes(payload[:100])
    assert _cursor_is_valid(path, cursor) is False


def test_observation_cursor_rejects_same_size_middle_rewrite(tmp_path):
    from research_forward import OBS_CURSOR_SCHEMA_VERSION, _cursor_is_valid, _journal_signature

    path = tmp_path / "zone_observations.jsonl"
    # The edit is outside both sampled 64 KiB windows and preserves file size.
    payload = bytearray(b"x" * (64 * 1024 * 3 + 17))
    path.write_bytes(payload)
    offset = len(payload)
    cursor = {
        "schema_version": OBS_CURSOR_SCHEMA_VERSION,
        "offset": offset,
        **_journal_signature(path, offset),
    }
    payload[64 * 1024 * 2] = ord("y")
    path.write_bytes(payload)
    assert path.stat().st_size == offset
    assert _cursor_is_valid(path, cursor) is False


def test_forward_bars_loads_retention_deltas_and_live_rows_without_legacy_seed(monkeypatch, tmp_path):
    from event_engine import research
    from research_forward import _load_bars

    data_dir = tmp_path / "data"
    live_path = data_dir / "market_bars_5m.jsonl"
    delta_dir = data_dir / "retention_archive" / "run1" / "market_bars_5m.jsonl" / "pruned"
    delta_dir.mkdir(parents=True)
    legacy_seed = data_dir / "research_history_seed" / "2026-10-16"
    legacy_seed.mkdir(parents=True)

    t0 = 1_760_000_000_000
    t1 = t0 + 5 * 60_000
    t2 = t1 + 5 * 60_000
    delta_path = delta_dir / "market_bars_5m.jsonl.pruned.0001.jsonl.gz"
    with gzip.open(delta_path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(_bar(t1, 102.0)) + "\n")
    # Old recovery-seed directories must not affect forward research after clean start.
    with gzip.open(legacy_seed / "market_bars_5m.jsonl.legacy.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(_bar(t0, 99.0)) + "\n")
    data_dir.mkdir(parents=True, exist_ok=True)
    live_path.write_text(json.dumps(_bar(t0, 100.5)) + "\n" + json.dumps(_bar(t2, 103.0)) + "\n", encoding="utf-8")
    monkeypatch.setattr(research, "MARKET_BARS_5M_PATH", live_path)

    bars = _load_bars(symbols={"TEST-USDT"})
    assert "TEST-USDT|binance" in bars
    df = bars["TEST-USDT|binance"]
    assert len(df) == 3
    assert list(df["close"]) == pytest.approx([100.5, 102.0, 103.0])
    assert df["timestamp"].is_monotonic_increasing


def test_stale_five_minute_trigger_is_rejected_inside_market_submitter(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {
        "quantityPrecision": 4, "pricePrecision": 4, "tradeMinQuantity": 0.0001,
        "multiplier": 1, "maxLeverage": 10,
    })
    monkeypatch.setattr(bingx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bingx, "has_open_position", lambda *args, **kwargs: False)
    monkeypatch.setattr(bingx, "position_side_param", lambda direction: direction)
    monkeypatch.setattr(bingx, "get_order", lambda *args, **kwargs: {"status": "not_found"})
    monkeypatch.setattr(bingx, "_set_leverage", lambda *args, **kwargs: True)
    now_ms = int(time.time() * 1000)
    monkeypatch.setattr(bingx, "get_execution_quote", lambda *args, **kwargs: {
        "status": "ok", "bid": 100.0, "ask": 100.0,
        "quote_observed_at_ms": now_ms, "quote_exchange_time_ms": now_ms,
    })
    monkeypatch.setattr(bingx, "_quote_freshness", lambda quote, max_age: (True, 0.0, 0.0, "TEST"))
    post_calls = []
    monkeypatch.setattr(bingx, "_request", lambda method, path, params=None, **kwargs: post_calls.append((method, path, params)) or {"code": 0, "data": {"orderId": "O1"}})
    monkeypatch.setattr(bingx, "_new_open_client_order_id", lambda *args, **kwargs: "TEST-OPEN-1")

    result = bingx.open_market(
        "TEST-USDT", "LONG", 100.0, "EVENT-1",
        trigger_bar_time_ms=now_ms - 11 * 60_000,
        max_trigger_age_minutes=10,
    )
    assert result["status"] == "skipped_stale_signal"
    assert "5m_trigger_too_old_at_post" in result["error"]
    assert result["order_submitted"] is False
    assert post_calls == []


def test_stale_pending_zone_visit_is_released_only_for_matching_event():
    import run_once

    state = {
        "symbols": {
            "TEST-USDT": {"zones": {
                "zone-1": {"state": "LOCKED", "trigger_event_id": "EVENT-1", "pending_signal": {"event_id": "EVENT-1"}},
                "zone-2": {"state": "LOCKED", "trigger_event_id": "EVENT-2", "pending_signal": {"event_id": "EVENT-2"}},
            }}
        }
    }
    assert run_once._clear_stale_pending_zone_visit(state, {"symbol": "test-usdt", "event_id": "EVENT-1"}) is True
    zone1 = state["symbols"]["TEST-USDT"]["zones"]["zone-1"]
    zone2 = state["symbols"]["TEST-USDT"]["zones"]["zone-2"]
    assert zone1["state"] == "ARMED"
    assert zone1["pending_signal"] is None
    assert zone1["trigger_event_id"] is None
    assert zone2["state"] == "LOCKED"
    assert zone2["pending_signal"]["event_id"] == "EVENT-2"


def test_be_close_accounting_includes_prior_tp_fill_from_event_ledger(monkeypatch):
    from event_engine import tracker

    entry_ms = 1_760_000_000_000
    fills = [
        {"order_id": "ENTRY-1", "trade_id": "F1", "time_ms": entry_ms + 1, "qty": 1.0, "price": 100.0, "fee": -0.01, "realized_pnl": 0.0},
        {"order_id": "TP1-1", "trade_id": "F2", "time_ms": entry_ms + 2, "qty": 0.5, "price": 103.0, "fee": -0.005, "realized_pnl": 1.5},
        {"order_id": "BE-1", "trade_id": "F3", "time_ms": entry_ms + 3, "qty": 0.5, "price": 100.0, "fee": -0.005, "realized_pnl": 0.0},
    ]
    monkeypatch.setattr(
        tracker, "get_fill_orders",
        lambda symbol, start, end, *, order_id=None, limit=1000: [] if order_id else fills,
    )
    monkeypatch.setattr(tracker, "_get_order_execution_evidence", lambda *args, **kwargs: None)
    trade = {
        "event_id": "EVENT-1", "symbol": "TEST-USDT", "direction": "LONG",
        "entry_ts": entry_ms, "entry_order": {"order_id": "ENTRY-1"},
        # TP1 has already been removed from the live tp_orders list after its fill.
        "tp_orders": [],
        "tp_fill_events": [{"order_id": "TP1-1", "leg": "TP1", "delta_qty": 0.5, "avg_price": 103.0, "fill_time_ms": entry_ms + 2}],
        "be_order_id": "BE-1", "exit_reason": "BREAK_EVEN",
    }
    accounting = tracker._collect_trade_fill_accounting(trade, entry_ms + 10_000)
    assert accounting["confirmed_exit_qty"] == pytest.approx(1.0)
    assert accounting["quantity_reconciliation"]["status"] == "VERIFIED"
    tp1_fill = next(row for row in accounting["fills"] if row["order_id"] == "TP1-1")
    assert tp1_fill["role"] == "TP1"

    evidence = tracker._close_execution_evidence(trade, accounting)
    assert evidence["status"] == "CONFIRMED"
    assert evidence["confirmed_exit_qty"] == pytest.approx(1.0)
    assert {row["leg"] for row in evidence["rows"]} == {"TP1", "BE_SL"}
    assert evidence["authority"] == "EXCHANGE_FILL_HISTORY"


def test_archive_chain_markers_are_protected_from_retention_rewrites():
    from event_engine.data_retention import NEVER_REWRITE_STATE_FILES

    assert "retention_archive_state.json" in NEVER_REWRITE_STATE_FILES
    assert "research_outcomes_restore_state.json" not in NEVER_REWRITE_STATE_FILES


def test_retention_archive_generation_names_never_overwrite(monkeypatch, tmp_path):
    from event_engine import data_retention

    now = data_retention.datetime(2026, 10, 18, 10, 0, tzinfo=data_retention.timezone.utc)
    archive_base = tmp_path / "retention_archive"
    first = data_retention._safe_archive_dir(now, archive_base)
    second = data_retention._safe_archive_dir(now, archive_base)
    assert first != second
    assert first.exists() and second.exists()


def test_workflow_persists_cumulative_archives_across_clean_runners():
    workflow = Path(".github/workflows/event-engine.yml").read_text(encoding="utf-8")
    assert "actions: write" in workflow
    assert "Restore fingerprint-matched cumulative retention archive" in workflow
    assert 'repos/${GITHUB_REPOSITORY}/actions/artifacts?name=zone-engine-retention-archive&per_page=100' in workflow
    assert 'repos/${GITHUB_REPOSITORY}/actions/workflows/event-engine.yml' in workflow
    assert 'repos/${GITHUB_REPOSITORY}/actions/runs/${run_id}' in workflow
    assert '.workflow_id == $expected_workflow_id' in workflow
    assert '.event == "repository_dispatch"' in workflow
    assert '.conclusion == "success"' in workflow
    assert 'actions/workflows/event-engine.yml/runs?branch=main&event=repository_dispatch&status=success&per_page=100' not in workflow
    assert 'gh run download "$run_id" --name zone-engine-retention-archive' in workflow
    assert 'gh run download --name zone-engine-retention-archive' not in workflow
    assert "recovery_safe=false" in workflow
    assert "recovery_safe=true" in workflow
    assert "data/retention_archive_state.json" in workflow
    archive_policy = Path("event_engine/retention_archive.py").read_text(encoding="utf-8")
    assert "invalid retention archive-chain state" in archive_policy
    assert "Decide whether retention archive upload is needed" in workflow
    assert "event_engine.retention_archive --decide-upload" in workflow
    assert "event_engine.retention_archive --mark-upload" in workflow
    assert "event_engine.retention_archive --verify-upload" in workflow
    assert "archive_upload_required == 'true'" in workflow
    assert "--refresh-days 45" in workflow
    assert "Verify retention archive is durable before commit" in workflow
    assert "steps.retention_archive_commit_gate.outcome == 'success'" in workflow
    assert "Upload cumulative retention archive for next clean runner" in workflow
    assert "name: zone-engine-retention-archive" in workflow
    assert "retention-days: 90" in workflow
    assert "data/retention_archive/" in workflow


def test_size_guard_preserves_pruned_rows_across_multiple_policy_escalations(monkeypatch, tmp_path):
    from event_engine import data_retention

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    now = data_retention.datetime(2026, 10, 18, 10, 0, tzinfo=data_retention.timezone.utc)
    path = data_dir / "market_bars_5m.jsonl"
    rows = []
    now_ms = int(now.timestamp() * 1000)
    for index in range(300):
        ts = now_ms - index * 12 * 60_000
        rows.append({
            "record_type": "MARKET_BAR", "symbol": "TEST-USDT", "provider": "binance",
            "timestamp": ts, "close_time": ts + 5 * 60_000,
            "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1,
            "archive_test_id": f"BAR-{index:03d}", "payload": "x" * 250,
        })
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(data_retention, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_retention, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(data_retention, "REPO_RETENTION_ARCHIVE_ROOT", data_dir / "retention_archive")
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_GUARD_BYTES", 50_000)
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_TARGET_BYTES", 40_000)

    result = data_retention.apply_size_guard(now=now, archive_root=data_dir / "retention_archive")
    assert result["files"]["market_bars_5m.jsonl"]["triggered"] is True

    delta_paths = sorted((data_dir / "retention_archive").rglob("market_bars_5m.jsonl.pruned*.jsonl.gz"))
    assert len(delta_paths) >= 2, "each pruning escalation must keep a separate immutable delta"
    archived_ids = set()
    for delta_path in delta_paths:
        with gzip.open(delta_path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    archived_ids.add(json.loads(line)["archive_test_id"])
    kept_ids = {
        json.loads(line)["archive_test_id"]
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    original_ids = {row["archive_test_id"] for row in rows}
    assert archived_ids.isdisjoint(kept_ids)
    assert archived_ids | kept_ids == original_ids
    assert len(archived_ids) == len(original_ids) - len(kept_ids)


def test_tracker_local_tp_fallback_is_not_double_counted_for_repeated_partial_events():
    from event_engine import tracker

    trade = {
        "exit_reason": "BREAK_EVEN",
        "tp_fill_events": [
            {"order_id": "TP1", "leg": "TP1", "delta_qty": 0.25, "executed_qty_total": 0.25, "avg_price": 103.0, "ts": 1000},
            {"order_id": "TP1", "leg": "TP1", "delta_qty": 0.25, "executed_qty_total": 0.5, "avg_price": 103.2, "ts": 2000},
        ],
    }
    evidence = tracker._close_execution_evidence(trade, {"fills": [], "order_execution_evidence": []})
    assert evidence["confirmed_exit_qty"] == pytest.approx(0.5)
    assert len(evidence["rows"]) == 1
    assert evidence["rows"][0]["executed_qty"] == pytest.approx(0.5)
    assert evidence["rows"][0]["avg_price"] == pytest.approx(103.2)


def test_emergency_close_transport_sentinel_minus_one_never_resubmits(monkeypatch):
    import run_once

    close_calls = []
    reads = iter([
        {"status": "found", "positionAmt": 1.0},
        {"status": "found", "positionAmt": 1.0},
    ])
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: next(reads, {"status": "found", "positionAmt": 1.0}))
    monkeypatch.setattr(run_once, "close_position_market", lambda *a, **k: close_calls.append(1) or {
        "status": "error", "response": {"code": -1, "msg": "ReadTimeout: outcome unknown"}
    })
    monkeypatch.setattr(run_once, "get_order", lambda *a, **k: {"status": "error", "error": "no order id"})
    monkeypatch.setattr(run_once.time, "sleep", lambda *a, **k: None)
    monkeypatch.setenv("EMERGENCY_CLOSE_ATTEMPTS", "4")
    monkeypatch.setenv("EMERGENCY_CLOSE_VERIFY_POLLS", "2")

    result = run_once._emergency_close_and_verify("AAA-USDT", "LONG", 1.0, "EVT_UNKNOWN_CLOSE")
    assert result["status"] == "close_unverified"
    assert len(close_calls) == 1


def test_cancel_old_sl_refuses_manual_order_without_delete(monkeypatch):
    from event_engine import tracker

    deleted = []
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [{
            "orderId": "MANUAL-1", "clientOrderId": "user-stop", "type": "STOP_MARKET",
            "side": "SELL", "positionSide": "BOTH", "stopPrice": "99", "origQty": "0.5",
        }], "tp_orders": [],
    })
    monkeypatch.setattr(tracker, "cancel_order", lambda *a, **k: deleted.append(a) or {"code": 0})

    cancelled, reason = tracker._cancel_old_sl_verified("AAA-USDT", "LONG", "MANUAL-1", owner_event_id="EVT_X")
    assert cancelled is False
    assert "manual SL" in reason
    assert deleted == []


def test_emergency_rollback_order_ids_are_included_in_fill_accounting(monkeypatch):
    from event_engine import tracker

    calls = []
    def fills(symbol, start_ms, end_ms, *, order_id=None, limit=1000):
        calls.append(order_id)
        rows = [
            {"order_id":"ENTRY1", "qty":1.0, "price":100.0, "fee":None, "realized_pnl":None, "time_ms":100},
            {"order_id":"TP1", "qty":0.5, "price":103.0, "fee":None, "realized_pnl":None, "time_ms":200},
        ]
        return [x for x in rows if order_id is None or x["order_id"] == order_id]
    monkeypatch.setattr(tracker, "get_fill_orders", fills)
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status":"error", "error":"not available"})
    trade = {
        "symbol":"AAA-USDT", "direction":"LONG", "entry_ts":1,
        "entry_order":{"order_id":"ENTRY1"},
        "tp_fill_events":[{"order_id":"TP1", "leg":"TP1"}],
        "be_failure_rollback":{"status":"closed_verified", "attempts":[
            {"response":{"code":0, "data":{"order":{"orderId":"EMERGENCY1"}}}}
        ]},
    }
    result = tracker._collect_trade_fill_accounting(trade, 1000)
    assert "EMERGENCY1" in result["order_ids"]
    assert "EMERGENCY1" in result["exact_order_queries_attempted"]
    assert "EMERGENCY1" in calls


def test_unverified_residual_close_does_not_publish_partial_tp_pnl_as_final(tmp_path, monkeypatch):
    import json
    from event_engine import tracker

    active_path = tmp_path / "active_trades.json"
    trades_path = tmp_path / "trades.jsonl"
    actions_path = tmp_path / "actions.jsonl"
    trade = {
        "event_id":"EVT_PARTIAL_UNVERIFIED", "symbol":"AAA-USDT", "direction":"LONG", "name":"AAA",
        "entry_price":100.0, "initial_qty":1.0, "remaining_qty":0.5, "entry_ts":1, "closed":False,
        "entry_order":{"order_id":"ENTRY1"}, "tp_orders":[], "sl_order":{},
        "hit_legs":["tp1"], "tp_filled_qty":{"tp1":0.5}, "realized_pnl_qty":0.5,
        "realized_pnl_weighted_sum":1.5, "tp_fill_events":[{
            "order_id":"TP1", "leg":"TP1", "delta_qty":0.5, "executed_qty_total":0.5,
            "avg_price":103.0, "pnl_pct":3.0, "ts":200, "fill_time_ms":200,
        }],
        "be_failure_rollback":{"status":"closed_verified", "attempts":[
            {"response":{"code":0, "data":{"order":{"orderId":"EMERGENCY1", "executedQty":"0.5", "avgPrice":"99"}}}}
        ]},
        "peak_pnl_pct":3.0, "mae_pct":-1.0, "max_drawdown_pct":-1.0,
        "planned_risk_pct":10.0, "planned_weighted_rr":0.45, "effective_weighted_rr":0.45,
        "setup":{}, "tp_levels":[], "effective_tp_levels":[],
    }
    active_path.write_text(json.dumps({trade["event_id"]:trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tracker, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tracker, "ACTIONS_PATH", actions_path)
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status":"not_found"})
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_fill_orders", lambda symbol, start, end, *, order_id=None, limit=1000: [
        row for row in [
            {"order_id":"ENTRY1", "qty":1.0, "price":100.0, "fee":None, "realized_pnl":None, "time_ms":100},
            {"order_id":"TP1", "qty":0.5, "price":103.0, "fee":None, "realized_pnl":None, "time_ms":200},
        ] if order_id is None or row["order_id"] == order_id
    ])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status":"error", "error":"no fill evidence"})
    monkeypatch.setattr(tracker, "get_all_orders", lambda *a, **k: [])
    sent = []
    monkeypatch.setattr(tracker, "send_tg", lambda text: sent.append(text) or True)

    tracker.update_active_trades(position_snapshot=[])
    close = next(json.loads(line) for line in trades_path.read_text(encoding="utf-8").splitlines() if json.loads(line).get("record_type") == "TRADE_CLOSE")
    assert close["realized_pnl_reconciliation"]["status"] == "UNVERIFIED_RESIDUAL_EXIT"
    assert close["realized_pnl_pct"] is None
    assert close["realized_pnl_source"] == "UNVERIFIED_RESIDUAL_EXIT"
    assert close["realized_pnl_pct_confirmed_partial"] == 1.5
    assert close["realized_rr"] is None
    assert sent and "не подтверждён" in sent[0]


def test_partial_aggregate_fill_response_forces_partial_accounting(monkeypatch):
    from event_engine import tracker

    def fills(symbol, start_ms, end_ms, *, order_id=None, limit=1000):
        rows = [
            {"order_id":"ENTRY1", "qty":1.0, "price":100.0, "fee":-0.1, "realized_pnl":0.0, "time_ms":100},
            {"order_id":"EXIT1", "qty":0.5, "price":102.0, "fee":-0.1, "realized_pnl":1.0, "time_ms":200},
        ]
        return [row for row in rows if order_id is None or row["order_id"] == order_id]

    monkeypatch.setattr(tracker, "credentials_available", lambda: True)
    monkeypatch.setattr(tracker, "get_fill_orders", fills)
    monkeypatch.setattr(tracker, "_get_order_execution_evidence", lambda symbol, order_id, **kwargs: {
        "status":"ok", "order_id":order_id, "order_status":"PARTIALLY_FILLED", "executed_qty":1.0,
        "avg_price":102.0, "time_ms":200, "update_time_ms":200,
    } if order_id == "EXIT1" else {"status":"ok", "order_id":order_id, "order_status":"FILLED", "executed_qty":1.0, "avg_price":100.0})

    result = tracker._collect_trade_fill_accounting({
        "symbol":"AAA-USDT", "direction":"LONG", "entry_ts":1,
        "entry_order":{"order_id":"ENTRY1"}, "exit_order_id":"EXIT1",
    }, 1000)
    assert result["incomplete_order_fill_ids"] == ["EXIT1"]
    assert result["status"] == "PARTIAL_ACCOUNTING"
    assert result["confirmed_exit_qty"] is None
    assert result["gross_realized_pnl_abs"] is None
    assert result["fees_paid_abs"] is None


def test_zone_forensics_and_be_retry_keep_the_state_they_report():
    from pathlib import Path
    import inspect
    from event_engine import tracker
    import run_once

    run_source = Path(inspect.getsourcefile(run_once)).read_text(encoding="utf-8")
    tracker_source = inspect.getsource(tracker.update_active_trades)
    assert "private_checkpoint_state_trusted = False" in run_source
    assert "if private_checkpoint_state_trusted:" in run_source
    assert "private_layer_available = private_ready" not in run_source
    preflight_start = run_source.index("private_ready = _private_layer_ready()")
    strict_load = run_source.index("_load_active_trades_file(fresh_runtime_start=fresh_runtime_start)", preflight_start)
    position_mode_check = run_source.index("mode = get_position_mode(", strict_load)
    reconciliation_call = run_source.index("reconciliation_ok = reconcile_all_open_positions()", position_mode_check)
    trust_mark = run_source.index("private_checkpoint_state_trusted = True", reconciliation_call)
    assert strict_load < position_mode_check < reconciliation_call < trust_mark
    assert "BLOCKED_PRIVATE_PREFLIGHT" in run_source
    assert "_build_5m_zone_signal(symbol, direction, zone, bar, prev, df_1h, demand, supply, state, df_5m=x)" in run_source
    retry_branch = tracker_source.split('if retry_status in {"created", "created_cleanup_pending"}:', 1)[1].split("elif retry_status", 1)[0]
    for key in ("be_trigger_ts", "be_trigger_peak_r", "be_trigger_rule", "be_trigger_price"):
        assert f'trade["{key}"]' in retry_branch


def test_cancel_ack_is_not_called_verified_while_order_remains_open(monkeypatch, tmp_path):
    import json
    from event_engine import tracker

    active_path = tmp_path / "active_trades.json"
    active_path.write_text(json.dumps({"EVT_CURRENT": {
        "event_id":"EVT_CURRENT", "symbol":"AAA-USDT", "direction":"LONG", "closed":False,
        "sl_order":{"order_id":"SL1"}, "tp_orders":[],
    }}), encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active_path)
    still_open = {"status":"ok", "sl_orders":[{
        "orderId":"SL1", "clientOrderId":"EVT_CURRENT_SL", "type":"STOP_MARKET",
        "side":"SELL", "positionSide":"BOTH", "stopPrice":"99", "origQty":"1",
    }], "tp_orders":[]}
    snapshots = iter([still_open, still_open, still_open])
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: next(snapshots, still_open))
    deleted = []
    monkeypatch.setattr(tracker, "cancel_order", lambda *a, **k: deleted.append(a) or {"code":0})
    monkeypatch.setattr(tracker.time, "sleep", lambda *a, **k: None)

    cancelled, reason = tracker._cancel_old_sl_verified(
        "AAA-USDT", "LONG", "SL1", max_attempts=1, owner_event_id="EVT_CURRENT"
    )
    assert cancelled is False
    assert "remains visible" in reason
    assert len(deleted) == 1


def test_semantically_untrusted_duplicate_owner_state_cannot_enable_final_tracker(monkeypatch, tmp_path):
    """Credential availability must not bypass strict ownership validation at final checkpoint."""
    import inspect
    from event_engine import tracker
    import run_once

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    active_path = data_dir / "active_trades.json"
    duplicated = {
        event_id: {
            "event_id": event_id,
            "symbol": "AAA-USDT",
            "direction": "LONG",
            "closed": False,
            "initial_qty": 1.0,
            "remaining_qty": 1.0,
            "entry_price": 100.0,
            "entry_ts": 1,
            "entry_order": {},
            "sl_order": {},
            "tp_orders": [],
        }
        for event_id in ("EVT_A", "EVT_B")
    }
    active_path.write_text(json.dumps(duplicated), encoding="utf-8")
    monkeypatch.setattr(run_once, "DATA", data_dir)
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active_path)

    # Strict run preflight rejects two local owners of one exchange position.
    with pytest.raises(RuntimeError, match="cross-event position ownership collision"):
        run_once._load_active_trades_file()
    # The tracker-level JSON loader alone is more permissive, so it must never be
    # run after strict preflight rejects the state.
    assert len(tracker._load_active_trades_required()) == 2

    source = Path(inspect.getsourcefile(run_once)).read_text(encoding="utf-8")
    assert "private_checkpoint_state_trusted = False" in source
    assert "private_checkpoint_state_trusted = True" in source
    assert "if private_checkpoint_state_trusted:" in source
    assert "private_layer_available = private_ready" not in source


def test_complete_emergency_exit_fills_upgrade_residual_pnl_from_unverified_to_confirmed(tmp_path, monkeypatch):
    import json
    from event_engine import tracker

    active_path = tmp_path / "active_trades.json"
    trades_path = tmp_path / "trades.jsonl"
    actions_path = tmp_path / "actions.jsonl"
    trade = {
        "event_id":"EVT_PARTIAL_CONFIRMED", "symbol":"AAA-USDT", "direction":"LONG", "name":"AAA",
        "entry_price":100.0, "initial_qty":1.0, "remaining_qty":0.5, "entry_ts":1, "closed":False,
        "entry_order":{"order_id":"ENTRY1"}, "tp_orders":[], "sl_order":{},
        "hit_legs":["tp1"], "tp_filled_qty":{"tp1":0.5}, "realized_pnl_qty":0.5,
        "realized_pnl_weighted_sum":1.5, "tp_fill_events":[{
            "order_id":"TP1", "leg":"TP1", "delta_qty":0.5, "executed_qty_total":0.5,
            "avg_price":103.0, "pnl_pct":3.0, "ts":200, "fill_time_ms":200,
        }],
        "be_failure_rollback":{"status":"closed_verified", "attempts":[
            {"response":{"code":0, "data":{"order":{"orderId":"EMERGENCY1"}}}}
        ]},
        "peak_pnl_pct":3.0, "mae_pct":-1.0, "max_drawdown_pct":-1.0,
        "planned_risk_pct":10.0, "planned_weighted_rr":0.45, "effective_weighted_rr":0.45,
        "setup":{}, "tp_levels":[], "effective_tp_levels":[],
    }
    active_path.write_text(json.dumps({trade["event_id"]:trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tracker, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tracker, "ACTIONS_PATH", actions_path)
    monkeypatch.setattr(tracker, "credentials_available", lambda: True)
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status":"not_found"})
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    fills = [
        {"order_id":"ENTRY1", "qty":1.0, "price":100.0, "fee":-0.1, "realized_pnl":0.0, "time_ms":100},
        {"order_id":"TP1", "qty":0.5, "price":103.0, "fee":-0.1, "realized_pnl":1.5, "time_ms":200},
        {"order_id":"EMERGENCY1", "qty":0.5, "price":99.0, "fee":-0.1, "realized_pnl":-0.5, "time_ms":300},
    ]
    monkeypatch.setattr(tracker, "get_fill_orders", lambda symbol, start, end, *, order_id=None, limit=1000: [x for x in fills if order_id is None or x["order_id"] == order_id])
    executions = {
        "ENTRY1": (1.0, 100.0), "TP1": (0.5, 103.0), "EMERGENCY1": (0.5, 99.0),
    }
    def execution_evidence(symbol, order_id, **kwargs):
        if order_id not in executions:
            return {"status":"error", "error":"no order id", "order_id":order_id}
        return {
            "status":"ok", "order_id":order_id, "order_status":"FILLED", "executed_qty":executions[order_id][0],
            "avg_price":executions[order_id][1], "time_ms":100, "update_time_ms":300,
        }
    monkeypatch.setattr(tracker, "_get_order_execution_evidence", execution_evidence)
    monkeypatch.setattr(tracker, "get_all_orders", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "send_tg", lambda text: True)

    tracker.update_active_trades(position_snapshot=[])
    close = next(json.loads(line) for line in trades_path.read_text(encoding="utf-8").splitlines() if json.loads(line).get("record_type") == "TRADE_CLOSE")
    assert close["realized_pnl_reconciliation"]["status"] == "VERIFIED"
    assert close["realized_pnl_pct"] == pytest.approx(1.0)
    assert close["realized_pnl_pct_confirmed_partial"] is None
    assert close["realized_pnl_source"] == "confirmed_fill_accounting"


def test_entry_not_filled_is_terminal_and_recovered_from_trade_journal(tmp_path, monkeypatch):
    import run_once

    assert run_once._is_terminal_execution_failure({"status": "entry_not_filled"}) is True
    journal = tmp_path / "trades.jsonl"
    journal.write_text(json.dumps({
        "event_id": "EVT_NO_FILL", "result": {"status": "entry_not_filled"},
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_once, "TRADES_PATH", journal)
    assert "EVT_NO_FILL" in run_once._load_terminal_event_ids()


def test_displacement_rejection_is_case_insensitive_for_zone_retirement():
    import run_once

    assert run_once._is_stale_entry_rejection({
        "status": "SKIPPED_STALE_SIGNAL",
        "error": "signal_price_displacement_pct=1.25 exceeds limit",
    }) is True
    assert run_once._is_stale_entry_rejection({
        "status": "SKIPPED_STALE_SIGNAL", "error": "other reject",
    }) is False


def test_reconciliation_risk_uses_setup_fallback_and_never_guesses_one_percent():
    import run_once

    assert run_once._reconciliation_stop_loss_pct({
        "planned_risk_pct": None, "setup": {"risk_pct": "10.0"},
    }) == 10.0
    assert run_once._reconciliation_stop_loss_pct({
        "planned_risk_pct": "corrupt", "setup": {"risk_pct": 10.0},
    }) == 10.0
    assert run_once._reconciliation_stop_loss_pct({
        "planned_risk_pct": "corrupt", "setup": {"risk_pct": "also-corrupt"},
    }) is None


def test_entry_timestamp_observed_after_fill_is_passed_into_trade_registration():
    import inspect
    import run_once

    source = inspect.getsource(run_once.execute_new_position)
    assert "fill_observed_ts_ms = int(fill_observed_ts.timestamp() * 1000)" in source
    assert "entry_ts_ms=fill_observed_ts_ms" in source
