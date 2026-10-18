from __future__ import annotations

from event_engine import bingx


def test_existing_verified_protection_uses_one_open_orders_snapshot(monkeypatch):
    """Already protected positions should not do a redundant second openOrders call."""
    sl = {"orderId": "SL1", "clientOrderId": "EVT_TEST_SL", "type": "STOP_MARKET", "stopPrice": "90.0", "origQty": "1.0"}
    tp1 = {"orderId": "TP1", "clientOrderId": "EVT_TEST_TP1", "type": "TAKE_PROFIT_MARKET", "stopPrice": "103.0", "origQty": "0.5"}
    tp2 = {"orderId": "TP2", "clientOrderId": "EVT_TEST_TP2", "type": "TAKE_PROFIT_MARKET", "stopPrice": "106.0", "origQty": "0.5"}
    calls = []

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {"quantityPrecision": 3, "pricePrecision": 1, "tradeMinQuantity": 0.001})
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *args, **kwargs: (calls.append((args, kwargs)) or {"status": "ok", "sl_orders": [sl], "tp_orders": [tp1, tp2]}))
    monkeypatch.setattr(bingx, "_request", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no order writes expected")))

    result = bingx.ensure_directional_protection(
        "AAA-USDT", "LONG", 100.0, 1.0, 10.0,
        [
            {"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 0.5},
            {"leg": "tp2", "pnl_pct": 6.0, "close_fraction": 0.5},
        ],
        trade_id="EVT_TEST",
    )

    assert result["status"] == "PROTECTED"
    assert len(calls) == 1
    assert result["sl_result"]["order_id"] == "SL1"


def test_deferred_research_bar_prefetch_preserves_symbol_mapping(monkeypatch):
    import run_once

    calls = []

    def fake_binance(symbol, interval, *, limit, retryable):
        calls.append(("binance", symbol, interval, limit, retryable))
        return [{"symbol": symbol, "timestamp": 1}]

    def fake_bingx(symbol, interval, *, limit, retryable):
        calls.append(("bingx", symbol, interval, limit, retryable))
        return [{"symbol": symbol, "timestamp": 2}]

    monkeypatch.setattr(run_once, "fetch_binance_klines", fake_binance)
    monkeypatch.setattr(run_once, "fetch_bingx_klines", fake_bingx)
    payloads = [
        {"symbol": "AAA-USDT", "provider": "binance", "persist_bars_for_forward": True, "bars_5m": []},
        {"symbol": "BBB-USDT", "provider": "bingx", "persist_bars_for_forward": True, "bars_5m": []},
        {"symbol": "CCC-USDT", "provider": "binance", "persist_bars_for_forward": True, "bars_5m": [{"timestamp": 3}]},
        {"symbol": "DDD-USDT", "provider": "binance", "persist_bars_for_forward": False, "bars_5m": []},
    ]
    prefetch = run_once._prefetch_missing_research_5m_bars(
        payloads,
        {"AAA-USDT": {"binance_symbol": "AAAUSDT"}},
        limit=24,
        max_workers=2,
    )

    assert prefetch[id(payloads[0])] == [{"symbol": "AAAUSDT", "timestamp": 1}]
    assert prefetch[id(payloads[1])] == [{"symbol": "BBB-USDT", "timestamp": 2}]
    assert id(payloads[2]) not in prefetch
    assert id(payloads[3]) not in prefetch
    assert len(calls) == 2
    assert {call[0] for call in calls} == {"binance", "bingx"}


def test_retention_upload_failure_restores_verified_pre_compaction_snapshot(tmp_path, monkeypatch):
    import gzip
    import hashlib
    import json
    from event_engine import data_retention

    data_dir = tmp_path / "data"
    archive_root = data_dir / "retention_archive"
    snapshot_dir = archive_root / "size-guard-test" / "market_bars_5m.jsonl" / "snapshot"
    snapshot_dir.mkdir(parents=True)
    original = b'{"row":1}\n{"row":2}\n'
    compacted = b'{"row":2}\n'
    target = data_dir / "market_bars_5m.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(compacted)
    chunk_path = snapshot_dir / "market_bars_5m.jsonl.before.part-0001.jsonl.gz"
    with gzip.open(chunk_path, "wb") as gz:
        gz.write(original)
    manifest_path = data_dir / "data_retention_manifest.json"
    manifest_path.write_text(json.dumps({
        "mode": "size_guard",
        "files": {
            "market_bars_5m.jsonl": {
                "triggered": True,
                "bytes_after": len(compacted),
                "rows_after": 1,
                "snapshot": {
                    "source_sha256": hashlib.sha256(original).hexdigest(),
                    "source_bytes": len(original),
                    "source_rows": 2,
                    "chunks": [{"path": str(chunk_path)}],
                },
            },
            "market_bars_1h.jsonl": {"triggered": False},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(data_retention, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_retention, "_utc_now", lambda: data_retention.datetime(2026, 10, 18, tzinfo=data_retention.timezone.utc))

    result = data_retention.restore_size_guard_snapshots(manifest_path=manifest_path)

    assert result["restored_any"] is True
    assert [item["file"] for item in result["restored"]] == ["market_bars_5m.jsonl"]
    assert target.read_bytes() == original
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["retention_upload_recovery"]["status"] == "restored_pre_compaction_snapshots"
    assert saved_manifest["files"]["market_bars_5m.jsonl"]["bytes_after"] == len(original)
    assert saved_manifest["files"]["market_bars_5m.jsonl"]["restored_after_upload_failure"] is True


def test_retention_rollback_refuses_corrupt_snapshot_without_replacing_live_journal(tmp_path, monkeypatch):
    import gzip
    import hashlib
    import json
    import pytest
    from event_engine import data_retention

    data_dir = tmp_path / "data"
    archive_root = data_dir / "retention_archive"
    snapshot_dir = archive_root / "size-guard-test" / "market_bars_5m.jsonl" / "snapshot"
    snapshot_dir.mkdir(parents=True)
    original = b'{"row":1}\n'
    compacted = b'{"row":2}\n'
    target = data_dir / "market_bars_5m.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(compacted)
    chunk_path = snapshot_dir / "market_bars_5m.jsonl.before.part-0001.jsonl.gz"
    with gzip.open(chunk_path, "wb") as gz:
        gz.write(original)
    manifest_path = data_dir / "data_retention_manifest.json"
    manifest_path.write_text(json.dumps({
        "mode": "size_guard",
        "files": {"market_bars_5m.jsonl": {
            "triggered": True,
            "snapshot": {
                "source_sha256": hashlib.sha256(b"not original").hexdigest(),
                "source_bytes": len(b"not original"),
                "source_rows": 1,
                "chunks": [{"path": str(chunk_path)}],
            },
        }},
    }), encoding="utf-8")
    monkeypatch.setattr(data_retention, "DATA_DIR", data_dir)

    with pytest.raises(RuntimeError, match="snapshot verification failed"):
        data_retention.restore_size_guard_snapshots(manifest_path=manifest_path)

    assert target.read_bytes() == compacted


def test_workflow_paginates_retention_artifacts_and_gates_commit_on_recovery():
    from pathlib import Path

    text = Path(".github/workflows/event-engine.yml").read_text(encoding="utf-8")
    assert "gh api --paginate --slurp" in text
    assert "id: retention_upload" in text
    assert "id: retention_recovery" in text
    assert "--restore-size-guard-snapshots" in text
    assert "steps.retention_upload.outcome == 'success' || steps.retention_recovery.outcome == 'success'" in text
    assert "--check-staged --limit-bytes \"$staged_limit\"" in text


def test_market_order_gate_rejects_old_5m_trigger_immediately_before_post(monkeypatch):
    import time

    now = 1_800_000_000.0
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: symbol)
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {
        "quantityPrecision": 3,
        "pricePrecision": 2,
        "tradeMinQuantity": 0.001,
        "multiplier": 1,
        "maxLongLeverage": 10,
    })
    monkeypatch.setattr(bingx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bingx, "has_open_position", lambda symbol, direction: False)
    monkeypatch.setattr(bingx, "position_side_param", lambda direction: "LONG")
    monkeypatch.setattr(bingx, "get_order", lambda *args, **kwargs: {"status": "not_found"})
    monkeypatch.setattr(bingx, "get_execution_quote", lambda *args, **kwargs: {
        "status": "ok", "bid": 100.0, "ask": 100.01, "timestamp": int(now * 1000),
    })
    monkeypatch.setattr(bingx, "_quote_freshness", lambda *args, **kwargs: (True, 0.0, 0.0, "exchange"))
    monkeypatch.setattr(bingx, "_set_leverage", lambda *args, **kwargs: True)
    monkeypatch.setattr(bingx.time, "time", lambda: now)
    monkeypatch.setattr(bingx, "_request", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("MARKET POST must not happen for stale trigger")))

    result = bingx.open_market(
        "AAA-USDT", "LONG", 100.0, "EVT_STALE_TRIGGER",
        trigger_bar_time_ms=int((now - 20 * 60) * 1000),
        max_trigger_age_minutes=10,
    )

    assert result["status"] == "skipped_stale_signal"
    assert result["order_submitted"] is False
    assert result["trigger_age_minutes"] == 20.0
