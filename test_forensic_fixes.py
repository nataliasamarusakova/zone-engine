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
    from research_forward import _cursor_is_valid, _journal_signature

    path = tmp_path / "zone_observations.jsonl"
    payload = b'{"row":"' + b"x" * 80_000 + b'"}\n'
    path.write_bytes(payload)
    cursor = {"offset": len(payload), "size": len(payload), **_journal_signature(path, len(payload))}
    old_inode = path.stat().st_ino

    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(payload)
    os.replace(replacement, path)
    assert path.stat().st_ino != old_inode
    assert _cursor_is_valid(path, cursor) is True


def test_observation_cursor_rejects_content_change_and_truncation(tmp_path):
    from research_forward import _cursor_is_valid, _journal_signature

    path = tmp_path / "zone_observations.jsonl"
    payload = b'{"row":"' + b"x" * 80_000 + b'"}\n'
    path.write_bytes(payload)
    cursor = {"offset": len(payload), "size": len(payload), **_journal_signature(path, len(payload))}

    changed = bytearray(payload)
    changed[9] = ord("y")
    path.write_bytes(changed)
    assert _cursor_is_valid(path, cursor) is False

    path.write_bytes(payload[:100])
    assert _cursor_is_valid(path, cursor) is False


def test_forward_bars_merge_recovery_seed_pruned_deltas_and_live_values(monkeypatch, tmp_path):
    from event_engine import research
    from research_forward import _load_bars

    data_dir = tmp_path / "data"
    live_path = data_dir / "market_bars_5m.jsonl"
    seed_dir = data_dir / "research_history_seed" / "2026-10-16"
    delta_dir = data_dir / "retention_archive" / "run1" / "market_bars_5m.jsonl" / "pruned"
    seed_dir.mkdir(parents=True)
    delta_dir.mkdir(parents=True)

    t0 = 1_760_000_000_000
    t1 = t0 + 5 * 60_000
    t2 = t1 + 5 * 60_000
    seed_path = seed_dir / "market_bars_5m.jsonl.before_2026-10-16.jsonl.gz"
    with gzip.open(seed_path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(_bar(t0, 100.0)) + "\n")
        fh.write(json.dumps(_bar(t1, 101.0, symbol="OTHER-USDT")) + "\n")
    delta_path = delta_dir / "market_bars_5m.jsonl.pruned.0001.jsonl.gz"
    with gzip.open(delta_path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(_bar(t1, 102.0)) + "\n")
    data_dir.mkdir(parents=True, exist_ok=True)
    live_t0 = _bar(t0, 100.5)
    # Live correction wins even when its close_time differs and sorts earlier.
    live_t0["close_time"] = t0 + 4 * 60_000
    live_path.write_text(
        json.dumps(live_t0) + "\n" + json.dumps(_bar(t2, 103.0)) + "\n",
        encoding="utf-8",
    )
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
    assert "actions: read" in workflow
    assert "Restore cumulative retention archive from previous successful runs" in workflow
    assert 'gh run list --repo "$GITHUB_REPOSITORY" --workflow event-engine.yml --status success' in workflow
    assert 'gh run download "$run_id" --name zone-engine-retention-archive' in workflow
    assert 'gh run download --name zone-engine-retention-archive' not in workflow
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
