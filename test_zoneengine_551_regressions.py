from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import timedelta

import pytest


def test_zone_visit_state_invalid_version_values_fall_back_instead_of_crashing(monkeypatch, tmp_path):
    import run_once

    path = tmp_path / "zone_visit_state.json"
    monkeypatch.setattr(run_once, "ZONE_VISIT_STATE_PATH", path)
    expected = {
        "version": run_once.ZONE_VISIT_STATE_VERSION,
        "trigger_mode": run_once.ZONE_TRIGGER_MODE,
        "symbols": {},
    }
    for invalid in (None, "abc", [], {}, True, 1.5):
        path.write_text(json.dumps({"version": invalid, "symbols": {}}), encoding="utf-8")
        assert run_once._load_zone_visit_state() == expected


def test_active_trade_validation_rejects_null_operational_containers():
    from event_engine.state_validation import validate_active_trade_record

    for field in ("tp_orders", "hit_legs", "tp_filled_qty", "tp_fill_events"):
        with pytest.raises(ValueError, match=field):
            validate_active_trade_record("EVT_BAD", {"event_id": "EVT_BAD", field: None})


def test_top_of_book_parser_rejects_wrong_symbol_in_single_and_nested_rows():
    from event_engine.bingx import _parse_top_of_book_payload

    wrong = {"symbol": "WRONG-USDT", "bidPrice": "10", "askPrice": "11"}
    assert _parse_top_of_book_payload({"code": 0, "data": [wrong]}, "AAA-USDT", "bookTicker") is None
    assert _parse_top_of_book_payload(
        {"code": 0, "data": {"book_ticker": [wrong]}}, "AAA-USDT", "bookTicker"
    ) is None


def test_top_of_book_parser_accepts_matching_symbol_in_list_payload():
    from event_engine.bingx import _parse_top_of_book_payload

    good = {"symbol": "AAAUSDT", "bidPrice": "10", "askPrice": "11", "time": 123}
    assert _parse_top_of_book_payload({"code": 0, "data": [good]}, "AAA-USDT", "bookTicker") == (10.0, 11.0, 123)


def test_zone_observation_retention_keeps_unreplayable_pending_but_can_prune_replayable():
    from event_engine import data_retention

    now = data_retention.datetime(2026, 10, 21, 12, 0, tzinfo=data_retention.timezone.utc)
    row = {"observation_id": "OBS_OLD", "observation_ts": (now - timedelta(days=5)).isoformat()}
    plan_unreplayable = data_retention._size_guard_plan(
        data_retention.Path("zone_observations.jsonl"), now=now, pending_ids={"OBS_OLD"},
        pending_bar_cutoffs={}, protected_observation_ids=set(), protected_refs=set(),
        replayable_observation_ids=set(),
    )
    assert plan_unreplayable[0][1](row) is True
    plan_replayable = data_retention._size_guard_plan(
        data_retention.Path("zone_observations.jsonl"), now=now, pending_ids={"OBS_OLD"},
        pending_bar_cutoffs={}, protected_observation_ids=set(), protected_refs=set(),
        replayable_observation_ids={"OBS_OLD"},
    )
    assert plan_replayable[0][1](row) is False


def test_pending_observation_scan_merges_live_journal_rows_not_in_pending_metadata(monkeypatch, tmp_path):
    from event_engine import data_retention

    monkeypatch.setattr(data_retention, "DATA_DIR", tmp_path)
    now = data_retention.datetime(2026, 10, 21, 12, 0, tzinfo=data_retention.timezone.utc)
    (tmp_path / "research_outcome_state.json").write_text(
        json.dumps({"processed_observation_ids": [], "pending_observations_v1": {"OLD": {
            "observation_id": "OLD", "symbol": "AAA-USDT", "direction": "LONG",
            "observation_ts": (now - timedelta(days=2)).isoformat(), "reference_price": 10,
            "event_type": "SIGNAL_CREATED", "provider": "binance", "source": "test",
        }}}), encoding="utf-8"
    )
    (tmp_path / "zone_observations.jsonl").write_text(
        json.dumps({"observation_id": "NEW", "symbol": "BBB-USDT", "direction": "SHORT",
                    "observation_ts": (now - timedelta(days=2)).isoformat(), "reference_price": 20,
                    "event_type": "SIGNAL_CREATED", "provider": "binance", "source": "test"}) + "\n", encoding="utf-8"
    )
    rows = {row["observation_id"]: row for row in data_retention._pending_observations()}
    assert rows["OLD"]["replayable"] is True
    assert rows["NEW"]["replayable"] is False


def test_size_guard_reads_file_metadata_only_after_acquiring_file_lock(monkeypatch, tmp_path):
    from event_engine import data_retention

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "market_bars_5m.jsonl"
    path.write_text(
        json.dumps({"id": "keep", "payload": "k" * 1400}) + "\n"
        + json.dumps({"id": "drop", "payload": "d" * 1400}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(data_retention, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_GUARD_BYTES", 2000)
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_TARGET_BYTES", 1000)
    monkeypatch.setattr(data_retention, "_size_guard_plan", lambda *a, **k: [("test", lambda row: row.get("id") == "keep")])

    real_meta = data_retention._file_meta
    lock_state = {"held": False}

    @contextmanager
    def fake_lock(_path):
        assert lock_state["held"] is False
        lock_state["held"] = True
        try:
            yield
        finally:
            lock_state["held"] = False

    def checked_meta(target, **kwargs):
        assert lock_state["held"] is True, "metadata must be sampled under the compaction lock"
        return real_meta(target, **kwargs)

    monkeypatch.setattr(data_retention.research, "_FileLock", fake_lock)
    monkeypatch.setattr(data_retention, "_file_meta", checked_meta)
    result = data_retention._apply_size_guard_to_file(
        path, now=data_retention.datetime(2026, 10, 21, 12, 0, tzinfo=data_retention.timezone.utc),
        archive_root=data_dir / "retention_archive", pending_ids=set(), pending_bar_cutoffs={},
        protected_observation_ids=set(), protected_refs=set(),
    )
    assert result["triggered"] is True
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["id"] == "keep"


def test_settle_entry_order_cancels_partial_remainder_and_waits_for_terminal_status(monkeypatch):
    import run_once

    calls = []
    states = iter([
        {"status": "ok", "order_id": "O1", "client_order_id": "EVT_ENTRY", "order_status": "PARTIALLY_FILLED", "executed_qty": 0.3},
        {"status": "ok", "order_id": "O1", "client_order_id": "EVT_ENTRY", "order_status": "CANCELED", "executed_qty": 0.3},
    ])
    monkeypatch.setattr(run_once, "get_order", lambda *a, **k: next(states))
    monkeypatch.setattr(run_once, "cancel_order", lambda *a, **k: calls.append((a, k)) or {"code": 0})
    monkeypatch.setattr(run_once.time, "sleep", lambda *_: None)
    result = run_once._settle_entry_order_before_protection(
        "AAA-USDT", {"status": "entry_pending", "order_id": "O1", "client_order_id": "EVT_ENTRY", "order_status": "PARTIALLY_FILLED"},
        timeout_sec=1,
    )
    assert result["status"] == "settled"
    assert result["order_status"] == "CANCELED"
    assert result["executed_qty"] == pytest.approx(0.3)
    assert result["position_refresh_required"] is True
    assert len(calls) == 1


def test_one_way_protection_params_are_reduce_only_but_hedge_params_are_unchanged():
    from event_engine.bingx import _add_reduce_only_for_one_way

    one_way = {"positionSide": "BOTH", "type": "STOP_MARKET"}
    hedge = {"positionSide": "LONG", "type": "STOP_MARKET"}
    assert _add_reduce_only_for_one_way(one_way) == {"positionSide": "BOTH", "type": "STOP_MARKET", "reduceOnly": "true"}
    assert _add_reduce_only_for_one_way(hedge) == hedge


def test_one_way_position_conflict_blocks_opposite_direction_entry(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "AAAUSDT")
    monkeypatch.setattr(bingx, "get_positions", lambda: [{"symbol": "AAAUSDT", "positionSide": "BOTH", "positionAmt": "-0.5"}])
    assert bingx.has_position_conflict_for_open("AAA-USDT", "LONG") is True
    assert bingx.has_position_conflict_for_open("AAA-USDT", "SHORT") is True


def test_sr_diagnostic_limit_300_returns_300_closed_bars(monkeypatch):
    import pine_sr_diagnostic

    hour_ms = 3_600_000
    now_ms = 1_798_000_000_000
    current_hour_start = now_ms // hour_ms * hour_ms
    monkeypatch.setattr(pine_sr_diagnostic.time, "time", lambda: now_ms / 1000)
    calls = []
    rows = []
    for index in range(300):
        open_time = current_hour_start - (index + 1) * hour_ms
        rows.append([open_time, "1", "2", "0.5", "1.5", "100", open_time + hour_ms - 1])

    def fake_get(_url, params, **_kwargs):
        calls.append(dict(params))
        return rows

    monkeypatch.setattr(pine_sr_diagnostic, "_get", fake_get)
    candles = pine_sr_diagnostic.fetch_1h("AAA-USDT", limit=300)
    assert len(candles) == 300
    assert calls[0]["limit"] == 300
    assert calls[0]["endTime"] == current_hour_start - 1


def test_filled_label_with_mismatched_executed_and_original_qty_is_not_trusted(monkeypatch):
    import run_once

    bad_state = {
        "status": "ok", "order_id": "O1", "client_order_id": "EVT_ENTRY",
        "order_status": "FILLED", "executed_qty": 0.3, "orig_qty": 1.0,
        "executed_qty_known": True,
    }
    monkeypatch.setattr(run_once, "get_order", lambda *a, **k: dict(bad_state))
    monkeypatch.setattr(run_once, "cancel_order", lambda *a, **k: pytest.fail("a contradictory FILLED record is not an active order"))
    monkeypatch.setattr(run_once.time, "sleep", lambda *_: None)
    result = run_once._settle_entry_order_before_protection(
        "AAA-USDT", {
            "status": "opened", "order_id": "O1", "client_order_id": "EVT_ENTRY",
            "order_status": "FILLED", "qty": 1.0, "executed_qty": 0.3,
        }, timeout_sec=1,
    )
    assert result["status"] == "unverified"
    assert "without consistent executedQty/origQty evidence" in result["error"]


def test_close_is_not_finalized_when_exchange_protection_cancellation_is_unverified(monkeypatch, tmp_path):
    from event_engine import tracker

    active = tmp_path / "active_trades.json"
    trades_path = tmp_path / "trades.jsonl"
    actions_path = tmp_path / "actions.jsonl"
    event_id = "EVT_CLEANUP_UNVERIFIED"
    trade = {
        "event_id": event_id, "symbol": "AAA-USDT", "direction": "LONG", "name": "AAA",
        "entry_price": 100.0, "initial_qty": 1.0, "remaining_qty": 1.0, "entry_ts": 1, "closed": False,
        "entry_order": {"order_id": "ENTRY1"},
        "tp_orders": [{"leg": "tp1", "order_id": "TP1"}],
        "sl_order": {"order_id": "SL1"}, "hit_legs": [], "tp_filled_qty": {},
        "realized_pnl_qty": 0.0, "realized_pnl_weighted_sum": 0.0,
        "peak_pnl_pct": 0.0, "mae_pct": 0.0, "max_drawdown_pct": 0.0,
        "planned_risk_pct": 1.0, "planned_weighted_rr": 0.5, "effective_weighted_rr": 0.5,
        "setup": {}, "tp_levels": [], "effective_tp_levels": [],
    }
    active.write_text(json.dumps({event_id: trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active)
    monkeypatch.setattr(tracker, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tracker, "ACTIONS_PATH", actions_path)
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_all_orders", lambda *a, **k: [{
        "orderId": "CLOSE1", "status": "FILLED", "side": "SELL", "type": "MARKET",
        "executedQty": "1", "avgPrice": "102", "updateTime": 200,
    }])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status": "error", "error": "not found"})
    monkeypatch.setattr(tracker, "get_fill_orders", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "_cancel_engine_protection_orders_verified", lambda *a, **k: (False, "cancel not verifiable"))
    monkeypatch.setattr(tracker, "send_tg", lambda *a, **k: True)

    tracker.update_active_trades()

    saved = json.loads(active.read_text(encoding="utf-8"))
    assert event_id in saved
    assert saved[event_id]["protection_cleanup_pending"]["order_ids"] == ["TP1", "SL1"]
    assert not trades_path.exists() or trades_path.read_text(encoding="utf-8").strip() == ""


def test_pending_observation_metadata_missing_replay_field_is_not_marked_replayable(monkeypatch, tmp_path):
    from event_engine import data_retention

    monkeypatch.setattr(data_retention, "DATA_DIR", tmp_path)
    now = data_retention.datetime(2026, 10, 21, 12, 0, tzinfo=data_retention.timezone.utc)
    (tmp_path / "research_outcome_state.json").write_text(json.dumps({
        "processed_observation_ids": [],
        "pending_observations_v1": {"OBS_INCOMPLETE": {
            "observation_id": "OBS_INCOMPLETE", "event_type": "SIGNAL_CREATED",
            "symbol": "AAA-USDT", "direction": "LONG",
            "observation_ts": (now - timedelta(days=3)).isoformat(), "reference_price": 10,
            # provider and source are required by research_forward._pending_meta_is_replayable
        }},
    }), encoding="utf-8")
    (tmp_path / "zone_observations.jsonl").write_text("", encoding="utf-8")
    rows = data_retention._pending_observations()
    assert len(rows) == 1
    assert rows[0]["observation_id"] == "OBS_INCOMPLETE"
    assert rows[0]["replayable"] is False


def test_filled_entry_without_original_quantity_evidence_is_not_trusted(monkeypatch):
    import run_once

    responses = iter([
        {"status": "ok", "order_id": "O1", "client_order_id": "EVT_ENTRY", "order_status": "FILLED", "executed_qty": 1.0},
        {"status": "ok", "order_id": "O1", "client_order_id": "EVT_ENTRY", "order_status": "FILLED", "executed_qty": 1.0},
    ])
    monkeypatch.setattr(run_once, "get_order", lambda *a, **k: next(responses))
    monkeypatch.setattr(run_once, "cancel_order", lambda *a, **k: pytest.fail("FILLED order with missing origQty is not cancellable"))
    monkeypatch.setattr(run_once.time, "sleep", lambda *_: None)
    result = run_once._settle_entry_order_before_protection(
        "AAA-USDT", {
            "status": "entry_pending", "order_id": "O1", "client_order_id": "EVT_ENTRY",
            "order_status": "FILLED", "executed_qty": 1.0,
        }, timeout_sec=1,
    )
    assert result["status"] == "unverified"
    assert "without consistent executedQty/origQty evidence" in result["error"]
