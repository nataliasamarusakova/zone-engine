from __future__ import annotations

import json
from datetime import timedelta

import pytest


def test_one_way_opposite_position_prevents_stale_protection_cleanup(monkeypatch):
    from event_engine import bingx

    position_reads = []
    cancel_calls = []

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "AAAUSDT")
    monkeypatch.setattr(bingx, "get_position_mode", lambda **_kwargs: "ONE_WAY")
    def fake_request(method, path, params=None, **_kwargs):
        position_reads.append((method, path, dict(params or {})))
        if path == bingx.POSITION_PATH:
            return {"code": 0, "data": [{
                "symbol": "AAAUSDT", "positionSide": "BOTH", "positionAmt": "-0.5", "avgPrice": "100",
            }]}
        pytest.fail(f"unexpected exchange request while testing stale protection cleanup: {method} {path}")
    monkeypatch.setattr(bingx, "_request", fake_request)
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *_a, **_k: pytest.fail("must not inspect/delete stale protection while opposite position exists"))
    monkeypatch.setattr(bingx, "cancel_order", lambda *a, **k: cancel_calls.append((a, k)) or {"code": 0})

    # Exercise the real get_position_directional parser: it correctly calls the
    # opposite signed BOTH row not_found for LONG and found for SHORT.
    assert bingx.get_position_directional("AAA-USDT", "LONG")["status"] == "not_found"
    result = bingx.prepare_protection_capacity("AAA-USDT", "LONG")

    assert result["status"] == "blocked_existing_position"
    assert result["position"]["positionAmt"] == 0.5
    assert result["position"]["positionSide"] == "SHORT"
    assert len(position_reads) == 3  # direct assertion above plus the two preparation reads
    assert cancel_calls == []


def test_successful_sl_post_is_not_accepted_when_exact_new_order_is_not_visible(monkeypatch):
    from event_engine import bingx

    old_sl = {
        "orderId": "OLD_SL", "clientOrderId": "EVT_PREVIOUS_SL", "type": "STOP_MARKET",
        "stopPrice": "90.0", "origQty": "1.0", "side": "SELL", "positionSide": "LONG",
    }
    requests = []
    cancellations = []

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "AAAUSDT")
    monkeypatch.setattr(bingx, "get_contract", lambda _symbol: {"quantityPrecision": 2, "pricePrecision": 2, "tradeMinQuantity": 0.01})
    monkeypatch.setattr(bingx, "get_position_mode", lambda **_kwargs: "HEDGE")
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *_a, **_k: {"status": "ok", "sl_orders": [old_sl], "tp_orders": []})
    monkeypatch.setattr(bingx, "_current_close_price", lambda _symbol: 100.0)
    monkeypatch.setattr(bingx, "_request", lambda method, path, params=None, **_kwargs: requests.append(dict(params or {})) or {
        "code": 0, "data": {"order": {"orderId": "ACK_NEW", "clientOrderId": (params or {}).get("clientOrderId")}}
    })
    monkeypatch.setattr(bingx, "cancel_order", lambda *_a, **_k: cancellations.append(1) or {"code": 0})
    monkeypatch.setattr(bingx.time, "sleep", lambda *_a: None)

    result = bingx.ensure_directional_protection("AAA-USDT", "LONG", 100.0, 1.0, 10.0, [], trade_id="EVT_CURRENT")

    assert requests and requests[0]["type"] == "STOP_MARKET"
    assert result["status"] == "SL_UNVERIFIED"
    assert result["sl_result"]["protection_order_verified"] is False
    assert "old SL retained" in result["error"]
    assert cancellations == [], "old stop must not be cancelled before exact replacement is exchange-visible"


def test_entry_state_unverified_protected_is_success_like_and_never_released_as_failure(tmp_path, monkeypatch):
    import run_once

    events = tmp_path / "trades.jsonl"
    events.write_text(json.dumps({"event_id": "EVT_UNVERIFIED_PROTECTED", "result": {"status": "entry_state_unverified_protected"}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_once, "TRADES_PATH", events)

    outcome = {"status": "entry_state_unverified_protected"}
    assert run_once._is_protected_execution_status(outcome) is True
    assert run_once._is_terminal_execution_failure(outcome) is False
    assert "EVT_UNVERIFIED_PROTECTED" in run_once._load_successful_trade_ids()
    assert "EVT_UNVERIFIED_PROTECTED" in run_once._load_terminal_event_ids()


def test_tracker_retains_active_owner_when_entry_remainder_unverified_and_position_temporarily_absent(tmp_path, monkeypatch):
    from event_engine import tracker

    active = tmp_path / "active_trades.json"
    trades = tmp_path / "trades.jsonl"
    event_id = "EVT_ENTRY_REMAINDER_UNVERIFIED"
    trade = {
        "event_id": event_id, "symbol": "AAA-USDT", "direction": "LONG", "name": "AAA",
        "entry_price": 100.0, "initial_qty": 1.0, "remaining_qty": 0.3, "entry_ts": 1,
        "closed": False, "entry_order": {"client_order_id": "EVT_ENTRY", "order_status": "PARTIALLY_FILLED"},
        "entry_remainder_unverified": True,
        "entry_order_reconciliation": {"status": "unverified", "error": "order status unavailable"},
        "setup": {"entry_remainder_unverified": True},
        "tp_orders": [], "sl_order": {"order_id": "SL1"}, "hit_legs": [], "tp_filled_qty": {},
        "realized_pnl_qty": 0.0, "realized_pnl_weighted_sum": 0.0,
    }
    active.write_text(json.dumps({event_id: trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active)
    monkeypatch.setattr(tracker, "TRADES_PATH", trades)
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    monkeypatch.setattr(tracker, "get_position_directional", lambda *_a, **_k: {"status": "not_found"})
    monkeypatch.setattr(tracker, "fetch_klines", lambda *_a, **_k: [])
    monkeypatch.setattr(tracker, "send_tg", lambda *_a, **_k: True)

    tracker.update_active_trades()

    saved = json.loads(active.read_text(encoding="utf-8"))
    assert event_id in saved
    assert saved[event_id]["closed"] is False
    assert saved[event_id]["entry_remainder_unverified"] is True
    assert not trades.exists() or not trades.read_text(encoding="utf-8").strip()


def test_entry_remainder_detector_recognizes_top_level_reconciliation_marker():
    import run_once

    assert run_once._trade_has_unverified_entry_remainder({
        "entry_order_reconciliation": {"status": "unverified"},
    }) is True


def test_be_verified_close_marker_is_cleared_and_be_retried_if_position_is_currently_found(tmp_path, monkeypatch):
    from event_engine import tracker, telemetry

    active = tmp_path / "active_trades.json"
    event_id = "EVT_BE_ROLLBACK_BUT_POSITION_LIVE"
    prior_marker = {"status": "closed_verified", "remaining_qty": 0.0, "last_known_qty": 0.0}
    trade = {
        "event_id": event_id, "symbol": "AAA-USDT", "direction": "LONG", "name": "AAA",
        "entry_price": 100.0, "initial_qty": 1.0, "remaining_qty": 0.5, "entry_ts": 1,
        "closed": False, "tp_orders": [], "sl_order": {"order_id": "OLD_SL", "stop_price": 90.0},
        "hit_legs": ["tp1"], "tp_filled_qty": {"tp1": 0.5}, "be_activated": False,
        "be_failure_rollback": prior_marker, "be_failure_rollback_verified_ts": 123,
        "realized_pnl_qty": 0.5, "realized_pnl_weighted_sum": 1.5,
        "peak_pnl_pct": 3.0, "mae_pct": 0.0, "max_drawdown_pct": 0.0,
        "planned_risk_pct": 10.0, "setup": {"planned_risk_pct": 10.0},
    }
    active.write_text(json.dumps({event_id: trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active)
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    reads = iter([
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},  # initial read contradicts old marker
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},  # forced fresh confirmation
    ])
    monkeypatch.setattr(tracker, "get_position_directional", lambda *_a, **_k: next(reads, {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0}))
    monkeypatch.setattr(tracker, "fetch_klines", lambda *_a, **_k: [])
    be_calls = []
    monkeypatch.setattr(tracker, "_move_sl_to_break_even", lambda *a, **k: be_calls.append(1) or {
        "status": "created", "order_id": "BE_NEW", "client_order_id": "EVT_BE_NEW", "stop_price": 100.0,
        "qty": 0.5, "side": "SELL", "positionSide": "LONG",
    })
    monkeypatch.setattr(tracker, "_get_exit_from_sl", lambda *a, **k: (None, None))
    monkeypatch.setattr(tracker, "_get_filled_order", lambda *a, **k: None)
    monkeypatch.setattr(tracker, "get_all_orders", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status": "error"})
    monkeypatch.setattr(tracker, "get_fill_orders", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "_send_tracker_notification", lambda *a, **k: True)
    for name in ("record_position_reconciliation", "record_protection_event", "record_exchange_error", "record_state_conflict", "record_performance_metric"):
        if hasattr(telemetry, name):
            monkeypatch.setattr(telemetry, name, lambda *a, **k: None)

    tracker.update_active_trades()

    saved = json.loads(active.read_text(encoding="utf-8"))[event_id]
    assert be_calls == [1]
    assert saved["be_activated"] is True
    assert "be_failure_rollback" not in saved
    assert saved["be_failure_rollback_stale"] == prior_marker


def test_retention_does_not_mark_pending_metadata_without_scan_id_or_observation_id_replayable(monkeypatch, tmp_path):
    from event_engine import data_retention

    monkeypatch.setattr(data_retention, "DATA_DIR", tmp_path)
    now = data_retention.datetime(2026, 10, 21, 12, 0, tzinfo=data_retention.timezone.utc)
    complete = {
        "observation_id": "OBS_COMPLETE", "scan_id": "SCAN_1", "event_type": "SIGNAL_CREATED",
        "symbol": "AAA-USDT", "direction": "LONG", "provider": "binance", "source": "test",
        "observation_ts": (now - timedelta(days=2)).isoformat(), "reference_price": 10,
    }
    missing_scan = {**complete, "observation_id": "OBS_NO_SCAN"}
    missing_scan.pop("scan_id")
    missing_obs_id = {**complete}
    missing_obs_id.pop("observation_id")
    state = {"processed_observation_ids": [], "pending_observations_v1": {
        "OBS_COMPLETE": complete, "OBS_NO_SCAN": missing_scan, "OBS_NO_ID": missing_obs_id,
    }}
    (tmp_path / "research_outcome_state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "zone_observations.jsonl").write_text("", encoding="utf-8")

    by_id = {row["observation_id"]: row for row in data_retention._pending_observations()}
    assert by_id["OBS_COMPLETE"]["replayable"] is True
    assert by_id["OBS_NO_SCAN"]["replayable"] is False
    assert by_id["OBS_NO_ID"]["replayable"] is False


def test_reconciliation_retries_unsettled_entry_and_repairs_protection_from_fresh_qty(monkeypatch):
    import run_once
    from event_engine import telemetry

    event_id = "EVT_ENTRY_REMAINDER_REPAIR"
    trade = {
        "event_id": event_id, "symbol": "AAA-USDT", "direction": "LONG", "closed": False,
        "entry_price": 100.0, "initial_qty": 1.0, "remaining_qty": 0.3, "planned_risk_pct": 10.0,
        "entry_remainder_unverified": True,
        "entry_order": {"order_id": "O_ENTRY", "client_order_id": "EVT_ENTRY", "order_status": "PARTIALLY_FILLED"},
        "tp_orders": [], "hit_legs": [], "tp_filled_qty": {},
        "setup": {
            "entry_remainder_unverified": True,
            "entry_order_reconciliation": {"status": "unverified"},
            "tp_levels": [
                {"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 0.5},
                {"leg": "tp2", "pnl_pct": 6.0, "close_fraction": 0.5},
            ],
        },
    }
    positions = iter([
        [{"symbol": "AAAUSDT", "positionSide": "BOTH", "positionAmt": "0.3", "avgPrice": "100"}],
        # The order remainder fills during the first reconciliation attempt.
        [{"symbol": "AAAUSDT", "positionSide": "BOTH", "positionAmt": "0.8", "avgPrice": "100"}],
    ])
    monkeypatch.setattr(run_once, "get_positions", lambda **_kwargs: next(positions))
    monkeypatch.setattr(run_once, "_load_active_trades_file", lambda **_kwargs: {event_id: trade})
    settlements = []
    monkeypatch.setattr(run_once, "_settle_entry_order_before_protection", lambda *a, **k: settlements.append((a, k)) or {
        "status": "unverified", "order_status": "PARTIALLY_FILLED", "error": "simulated delayed exchange state",
    })
    monkeypatch.setattr(run_once, "update_active_trade_entry_reconciliation", lambda *_a, **_k: True)
    protected_qtys = []
    monkeypatch.setattr(run_once, "ensure_directional_protection", lambda symbol, direction, avg, qty, *a, **k: protected_qtys.append(qty) or {
        "status": "PROTECTED", "tp_orders": [],
        "sl_result": {"status": "verified", "order_id": "SL_REPAIRED", "client_order_id": "EVT_REPAIR_SL", "stop_price": 90.0, "qty": qty},
        "effective_tp_levels": [], "tp_mode": "none", "effective_weighted_rr": 0.0,
    })
    monkeypatch.setattr(run_once, "update_active_trade_protection", lambda *a, **k: True)
    monkeypatch.setattr(run_once, "_append_execution_ledger", lambda *a, **k: None)
    for name in ("record_position_reconciliation", "record_protection_event", "record_exchange_error", "record_state_conflict"):
        if hasattr(telemetry, name):
            monkeypatch.setattr(telemetry, name, lambda *a, **k: None)

    result = run_once.reconcile_all_open_positions()

    assert result is False, "unsettled entry order must block further entries"
    assert len(settlements) == 1, "entry settlement is retried by reconciliation"
    assert protected_qtys == [pytest.approx(0.8)], "protection must be repaired against the post-cancel fresh quantity, not the stale 0.3 snapshot"
