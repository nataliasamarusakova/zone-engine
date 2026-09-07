import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from event_engine import signals as sig


def _base_signal(direction="LONG"):
    return {
        "symbol": "TEST-USDT", "type": direction, "entry": 100.0,
        "sl": 98.0 if direction == "LONG" else 102.0,
        "tp1": 101.2 if direction == "LONG" else 98.8,
        "tp2": 102.4 if direction == "LONG" else 97.6,
        "risk_pct": 2.0,
        "target": {"obstacle_price": 104.0 if direction == "LONG" else 96.0},
        "zone": {"kind": "DEMAND" if direction == "LONG" else "SUPPLY", "btm": 97.0, "top": 99.0, "start": 100, "age_bars": 5},
    }


def test_atr_does_not_backfill_future_values():
    df = pd.DataFrame({"high":[1,2,3],"low":[0,1,2],"close":[0.5,1.5,2.5]})
    out = sig.calc_atr(df, 5)
    assert out.isna().all()


def test_run_once_validation_enforces_risk_and_structure(monkeypatch):
    import run_once
    monkeypatch.setenv("MAX_SIGNAL_RISK_PCT", "1.50")
    monkeypatch.setenv("MIN_STRUCTURE_ROOM_R", "1.20")
    good = _base_signal("LONG")
    good["risk_pct"] = 1.0
    good["target"]["obstacle_price"] = 104.0
    ok, reason = run_once._validate_trade_geometry(good)
    assert ok, reason
    bad = _base_signal("LONG")
    ok, reason = run_once._validate_trade_geometry(bad)
    assert not ok and "risk_pct_above_limit" in reason


def test_directional_candle_filter():
    monkeypatch = None
    assert sig.REQUIRE_DIRECTIONAL_CANDLE is True


def test_server_time_offset_function_exists():
    from event_engine import bingx
    class R:
        headers = {"Date": "Wed, 02 Sep 2026 19:00:00 GMT"}
    old = bingx.SERVER_TIME_OFFSET_MS
    assert bingx._update_server_time_offset(R()) is True
    assert isinstance(bingx.SERVER_TIME_OFFSET_MS, int)
    bingx.SERVER_TIME_OFFSET_MS = old


def test_zone_entry_filters_are_configured_safely():
    from event_engine import signals as sig
    assert sig.MAX_ZONE_AGE_BARS == 30
    assert sig.MAX_SIGNAL_RISK_PCT == 1.50
    assert sig.MIN_STRUCTURE_ROOM_R == 1.20
    assert sig.REQUIRE_DIRECTIONAL_CANDLE is True
    assert sig.REQUIRE_STRUCTURE_OBSTACLE is True


def test_tracker_close_once_is_idempotent(tmp_path, monkeypatch):
    from event_engine import tracker
    path = tmp_path / "trades.jsonl"
    monkeypatch.setattr(tracker, "TRADES_PATH", path)
    rec = {"record_type": "TRADE_CLOSE", "event_id": "EVT_TEST"}
    assert tracker._append_trade_close_once(rec) is True
    assert tracker._append_trade_close_once(rec) is False
    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1


def test_open_market_blocks_two_tp_min_quantity_before_order(monkeypatch):
    from event_engine import bingx
    monkeypatch.setenv("BINGX_MARGIN_USDT", "1")
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {
        "symbol": "BNB-USDT", "quantityPrecision": 2, "tradeMinQuantity": 0.01,
        "multiplier": 1, "maxLeverage": 10,
    })
    monkeypatch.setattr(bingx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bingx, "_current_close_price", lambda symbol: 1000.0)
    monkeypatch.setattr(bingx, "has_open_position", lambda *args, **kwargs: False)
    out = bingx.open_market("BNB-USDT", "LONG", 1000.0, "EVT_TEST")
    assert out["status"] == "skipped_tp_min_qty"


def test_sl_order_validation_is_strictly_one_sided():
    from event_engine import bingx
    assert bingx._validate_sl_order_for_position({"type":"STOP_MARKET","stopPrice":"99","origQty":"1"}, "LONG", 100.0)
    assert not bingx._validate_sl_order_for_position({"type":"STOP_MARKET","stopPrice":"100.2","origQty":"1"}, "LONG", 100.0)
    assert bingx._validate_sl_order_for_position({"type":"STOP_MARKET","stopPrice":"101","origQty":"1"}, "SHORT", 100.0)
    assert not bingx._validate_sl_order_for_position({"type":"STOP_MARKET","stopPrice":"99.8","origQty":"1"}, "SHORT", 100.0)

def test_workflow_risk_cap_is_not_accidentally_25_percent():
    from pathlib import Path
    workflow = Path('.github/workflows/event-engine.yml').read_text(encoding='utf-8')
    assert 'MAX_SIGNAL_RISK_PCT: "1.50"' in workflow
    assert 'MAX_SIGNAL_RISK_PCT: "25"' not in workflow

def test_signal_risk_cap_hard_clamped(monkeypatch):
    import importlib
    import event_engine.signals as sig
    monkeypatch.setenv('MAX_SIGNAL_RISK_PCT', '25')
    sig2 = importlib.reload(sig)
    assert sig2.MAX_SIGNAL_RISK_PCT == 1.50
    monkeypatch.setenv('MAX_SIGNAL_RISK_PCT', '1.25')
    sig2 = importlib.reload(sig2)
    assert sig2.MAX_SIGNAL_RISK_PCT == 1.25
    monkeypatch.delenv("MAX_SIGNAL_RISK_PCT", raising=False)
    importlib.reload(sig2)

def test_run_once_hard_caps_production_risk(monkeypatch):
    import run_once
    monkeypatch.setenv("MAX_SIGNAL_RISK_PCT", "25")
    assert run_once.MAX_PRODUCTION_RISK_PCT == 1.50

def test_request_does_not_fallback_post_after_network_error(monkeypatch):
    from event_engine import bingx
    class Session:
        def __init__(self): self.calls=[]
        def request(self, **kwargs):
            self.calls.append(kwargs['url'])
            raise bingx.requests.Timeout('network')
    session = Session()
    monkeypatch.setattr(bingx, 'SESSION', session)
    monkeypatch.setattr(bingx, '_get_fast_session', lambda: session)
    monkeypatch.setenv('BINGX_API_KEY', 'k')
    monkeypatch.setenv('BINGX_SECRET_KEY', 's')
    out = bingx._request('POST', '/foo', {'a':'1'}, signed=True, retryable=True)
    assert out['code'] == -1
    assert len(session.calls) == 1


def test_sl_validator_covers_full_position_qty():
    from event_engine import bingx
    good = {"type": "STOP_MARKET", "stopPrice": "99", "origQty": "1.0"}
    bad = {"type": "STOP_MARKET", "stopPrice": "99", "origQty": "0.5"}
    assert bingx._validate_sl_order_for_position(good, "LONG", 100.0, 1.0)
    assert not bingx._validate_sl_order_for_position(bad, "LONG", 100.0, 1.0)


def test_tp_success_requires_exchange_reconciliation(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})
    out = bingx._verify_open_order("AAA-USDT", "LONG", client_order_id="EVT_X_TP1", order_kind="TAKE_PROFIT_MARKET", expected_price=101.0, expected_qty=1.0, price_precision=2, max_attempts=1)
    assert out["status"] == "not_found"


def test_failed_signal_registry_roundtrip(tmp_path, monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "DATA", tmp_path)
    monkeypatch.setattr(run_once, "FAILED_SIGNALS_PATH", tmp_path / "failed_signals.json")
    monkeypatch.setattr(run_once, "FAILED_SIGNAL_TTL_SEC", 3600)
    run_once._mark_failed_signal("EVT_TEST", "blocked_protection_preflight", "x")
    loaded = run_once._load_failed_signal_ids()
    assert "EVT_TEST" in loaded


def test_be_uses_detected_one_way_position_side(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(tracker, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "position_side_param", lambda direction: "BOTH")
    order_box = {}
    def fake_open_orders(*a, **k):
        if order_box:
            return {"status": "ok", "sl_orders": [order_box.copy()], "tp_orders": []}
        return {"status": "ok", "sl_orders": [], "tp_orders": []}
    monkeypatch.setattr(tracker, "get_open_protection_directional", fake_open_orders)
    calls = []
    def fake_request(method, path, params):
        calls.append((method, params))
        if method == "POST":
            order_box.update({"orderId": "123", "clientOrderId": params.get("clientOrderId"), "type": "STOP_MARKET", "stopPrice": params.get("stopPrice"), "origQty": params.get("quantity")})
            return {"code": 0, "data": {"order": order_box.copy()}}
        return {"code": 0}
    monkeypatch.setattr(tracker, "_request", fake_request)
    out = tracker._move_sl_to_break_even("AAA-USDT", "LONG", 100.0, 1.0, None, "T")
    assert out["status"] == "created"
    assert calls[0][1]["positionSide"] == "BOTH"

def test_signed_get_and_delete_use_explicit_query(monkeypatch):
    from event_engine import bingx
    monkeypatch.setenv("BINGX_API_KEY", "key")
    monkeypatch.setenv("BINGX_SECRET_KEY", "secret")
    captured = []
    class Resp:
        headers = {}
        def json(self): return {"code": 0, "data": {}}
    class Session:
        def request(self, **kwargs):
            captured.append(kwargs)
            return Resp()
    monkeypatch.setattr(bingx, "SESSION", Session())
    bingx._request("GET", "/p", {"symbol":"BTC-USDT", "orderId":"123"}, signed=True)
    bingx._request("DELETE", "/p", {"symbol":"BTC-USDT", "orderId":"123"}, signed=True)
    for call in captured:
        assert "?" in call["url"]
        assert call.get("params") is None
        assert "signature=" in call["url"]


def test_close_position_omits_reduce_only_in_hedge(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(bingx, "get_contract", lambda s: {"quantityPrecision": 3})
    monkeypatch.setattr(bingx, "position_side_param", lambda d: d)
    calls = []
    monkeypatch.setattr(bingx, "_request", lambda method, path, params: calls.append(params) or {"code":0})
    out = bingx.close_position_market("AAA-USDT", "SHORT", 1.0, reduce_only=True, trade_id="T")
    assert out["status"] == "closed"
    assert calls and "reduceOnly" not in calls[-1]
    assert calls[-1]["positionSide"] == "SHORT"


def test_close_position_uses_reduce_only_in_one_way(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(bingx, "get_contract", lambda s: {"quantityPrecision": 3})
    monkeypatch.setattr(bingx, "position_side_param", lambda d: "BOTH")
    calls = []
    monkeypatch.setattr(bingx, "_request", lambda method, path, params: calls.append(params) or {"code":0})
    out = bingx.close_position_market("AAA-USDT", "LONG", 1.0, reduce_only=True, trade_id="T")
    assert out["status"] == "closed"
    assert calls[-1]["reduceOnly"] == "true"


def test_failed_signal_is_recorded_after_emergency_like_status(tmp_path, monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "DATA", tmp_path)
    monkeypatch.setattr(run_once, "FAILED_SIGNALS_PATH", tmp_path / "failed_signals.json")
    monkeypatch.setattr(run_once, "FAILED_SIGNAL_TTL_SEC", 3600)
    # The execution-status branch used by main must treat any non-protected
    # outcome as terminal, including statuses whose prefix is 'opened'.
    status = "opened_then_emergency_closed"
    if status != "opened_protected":
        run_once._mark_failed_signal("EVT_EMERGENCY", status, "rollback")
    assert "EVT_EMERGENCY" in run_once._load_failed_signal_ids()


def test_scan_history_rotates_and_keeps_complete_records(tmp_path, monkeypatch):
    from event_engine import analytics
    monkeypatch.setattr(analytics, "DATA_DIR", tmp_path)
    monkeypatch.setattr(analytics, "SCAN_JSONL", tmp_path / "scan_history.jsonl")
    monkeypatch.setattr(analytics, "MAX_SCAN_HISTORY_BYTES", 1024)
    for i in range(80):
        analytics._append_jsonl(analytics.SCAN_JSONL, {"i": i, "payload": "x" * 40})
        analytics._rotate_scan_history()
    assert analytics.SCAN_JSONL.stat().st_size <= 1400
    rows = [json.loads(x) for x in analytics.SCAN_JSONL.read_text().splitlines() if x.strip()]
    assert rows and rows[-1]["i"] == 79


def test_all_orders_helper_uses_historical_endpoint(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda s: s)
    captured = {}
    def fake_request(method, path, params, signed=True):
        captured.update(method=method, path=path, params=params, signed=signed)
        return {"code": 0, "data": [{"orderId": "1"}]}
    monkeypatch.setattr(bingx, "_request", fake_request)
    out = bingx.get_all_orders("AAA-USDT", 100, 200, 50)
    assert out == [{"orderId": "1"}]
    assert captured["path"] == "/openApi/swap/v2/trade/allOrders"
    assert captured["params"]["startTime"] == 100
    assert captured["params"]["endTime"] == 200


def test_historical_exit_reconciliation_does_not_use_tp_for_residual(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "_get_filled_order", lambda *a, **k: None)
    monkeypatch.setattr(tracker, "get_all_orders", lambda *a, **k: [
        {"orderId":"TP1", "status":"FILLED", "side":"SELL", "type":"TAKE_PROFIT_MARKET", "executedQty":"1", "avgPrice":"110", "updateTime":200},
        {"orderId":"SL1", "status":"FILLED", "side":"SELL", "type":"STOP_MARKET", "executedQty":"1", "avgPrice":"95", "updateTime":300},
    ])
    out = tracker._reconcile_historical_exit_order("AAA-USDT", "LONG", 100, 1, [{"order_id":"TP1"}], {"order_id":"SL1"})
    assert out[0] == 95.0
    assert out[1] == "STOP_LOSS"


def test_post_fill_slippage_guard_emergency_closes(monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "MAX_ENTRY_SLIPPAGE_PCT", 1.0)
    signal = {"event_id":"EVT_SLIP", "symbol":"AAA-USDT", "type":"LONG", "entry":100.0, "sl":99.0, "tp1":101.0, "tp2":102.0, "risk_pct":1.0,
              "zone":{"kind":"DEMAND","btm":98,"top":99}, "target":{"obstacle_price":110}}
    monkeypatch.setattr(run_once, "_validate_trade_geometry", lambda s: (True, ""))
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {"status":"ok","sl_orders":[],"tp_orders":[]})
    monkeypatch.setattr(run_once, "_build_setup", lambda s: {"zone":{"kind":"DEMAND"}})
    monkeypatch.setattr(run_once, "open_market", lambda *a: {"status":"opened"})
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda *a, **k: {"status":"found","avgPrice":102.5,"positionAmt":1})
    monkeypatch.setattr(run_once, "_emergency_close_and_verify", lambda *a, **k: {"status":"closed_verified"})
    monkeypatch.setattr(run_once, "_cleanup_engine_protection", lambda *a, **k: {"status":"ok"})
    out = run_once.execute_new_position(signal)
    assert out["status"] == "opened_then_emergency_closed"
    assert "adverse_entry_slippage" in out["error"]

def test_wait_for_position_fill_retries_transient_errors(monkeypatch):
    from event_engine import bingx
    seq = [
        {"status":"error", "error":"temporary"},
        {"status":"not_found"},
        {"status":"found", "avgPrice":100, "positionAmt":1},
    ]
    monkeypatch.setattr(bingx, "get_position_directional", lambda *a, **k: seq.pop(0))
    monkeypatch.setattr(bingx.time, "sleep", lambda *a, **k: None)
    out = bingx.wait_for_position_fill_directional("AAA-USDT", "LONG", timeout_sec=1, poll_interval=0)
    assert out["status"] == "found"


def test_execute_reconciles_position_after_fill_poll_timeout(monkeypatch):
    import run_once
    signal = {
        "event_id": "ZONE_TEST_FILL_RECON",
        "symbol": "TEST-USDT",
        "type": "LONG",
        "entry": 100.0,
        "sl": 99.0,
        "tp1": 101.0,
        "tp2": 102.0,
        "risk_pct": 1.0,
        "score": 80,
        "atr": 2.0,
        "zone": {"kind": "DEMAND", "btm": 98.0, "top": 100.0},
        "target": {"obstacle_price": 104.0},
    }
    monkeypatch.setattr(run_once, "_validate_trade_geometry", lambda s: (True, "ok"))
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})
    monkeypatch.setattr(run_once, "_build_setup", lambda s: {"zone": {"kind": "DEMAND"}, "tp_levels": []})
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: {"status": "opened", "symbol": "TEST-USDT"})
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda *a, **k: {"status": "timeout", "symbol": "TEST-USDT"})
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: {"status": "found", "avgPrice": 100.5, "positionAmt": 1.0})
    monkeypatch.setattr(run_once, "_rebase_protection_after_fill", lambda s, avg: {
        "sl": 98.5, "tp1": 101.25, "tp2": 102.5, "risk_abs": 2.0, "risk_pct": 1.99,
        "tp1_rr": 0.5, "tp2_rr": 1.0, "target_source": "test", "obstacle_price": 104.0,
    })
    monkeypatch.setenv("MAX_SIGNAL_RISK_PCT", "2.5")
    monkeypatch.setattr(run_once, "ensure_directional_protection", lambda *a, **k: {
        "status": "PROTECTED", "tp_orders": [], "sl_result": {},
    })
    registered = {}
    monkeypatch.setattr(run_once, "register_active_trade", lambda *a, **k: registered.update(k))
    out = run_once.execute_new_position(signal)
    assert out["status"] == "opened_protected"
    assert out["position"]["avgPrice"] == 100.5
    assert registered["entry_price"] == 100.5


def test_execute_marks_entry_not_filled_after_timeout_and_reconcile(monkeypatch):
    import run_once
    signal = {
        "event_id": "ZONE_TEST_FILL_ABSENT",
        "symbol": "TEST-USDT",
        "type": "SHORT",
        "entry": 100.0,
        "sl": 101.0,
        "tp1": 99.5,
        "tp2": 99.0,
        "risk_pct": 1.0,
        "score": 80,
        "zone": {"kind": "SUPPLY", "btm": 100.0, "top": 102.0},
        "target": {"obstacle_price": 96.0},
    }
    monkeypatch.setattr(run_once, "_validate_trade_geometry", lambda s: (True, "ok"))
    monkeypatch.setattr(run_once, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})
    monkeypatch.setattr(run_once, "_build_setup", lambda s: {"zone": {"kind": "SUPPLY"}, "tp_levels": []})
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: {"status": "opened", "symbol": "TEST-USDT"})
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda *a, **k: {"status": "timeout", "symbol": "TEST-USDT"})
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: {"status": "not_found", "symbol": "TEST-USDT", "positionSide": "SHORT"})
    monkeypatch.setattr(run_once, "ensure_directional_protection", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not protect absent position")))
    out = run_once.execute_new_position(signal)
    assert out["status"] == "entry_not_filled"
    assert out["position"]["status"] == "not_found"


def test_position_keys_normalizes_one_way_both_by_position_amount():
    import run_once
    positions = [
        {"symbol": "AAA-USDT", "positionSide": "BOTH", "positionAmt": "1.25"},
        {"symbol": "BBB-USDT", "positionSide": "BOTH", "positionAmt": "-2.50"},
        {"symbol": "CCC-USDT", "positionSide": "BOTH", "positionAmt": "0"},
        {"symbol": "DDD-USDT", "positionSide": "LONG", "positionAmt": "1"},
    ]
    assert run_once._position_keys(positions) == {
        ("AAA-USDT", "LONG"),
        ("BBB-USDT", "SHORT"),
        ("DDD-USDT", "LONG"),
    }


def test_reconcile_handles_one_way_both_position(monkeypatch, tmp_path):
    import run_once
    monkeypatch.setattr(run_once, "_load_active_trades_file", lambda: {})
    monkeypatch.setattr(run_once, "get_positions", lambda **kwargs: [
        {"symbol": "AAA-USDT", "positionSide": "BOTH", "positionAmt": "-1.0", "avgPrice": "100.0"},
    ])
    calls = {}
    monkeypatch.setattr(run_once, "ensure_directional_protection", lambda *args, **kwargs: calls.update({
        "symbol": args[0], "direction": args[1], "avg": args[2], "qty": args[3],
    }) or {"status": "PROTECTED", "tp_orders": [], "sl_result": {}, "effective_tp_levels": []})
    run_once.reconcile_all_open_positions()
    assert calls == {"symbol": "AAA-USDT", "direction": "SHORT", "avg": 100.0, "qty": 1.0}


def test_tracker_uses_exchange_qty_as_remaining_qty(monkeypatch, tmp_path):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    trade = {
        "event_id": "EVT_QTY_RECON",
        "symbol": "AAA-USDT",
        "direction": "LONG",
        "name": "AAA",
        "entry_price": 100.0,
        "initial_qty": 1.0,
        "remaining_qty": 1.0,
        "entry_ts": 0,
        "closed": False,
        "tp_orders": [],
        "sl_order": {},
        "hit_legs": [],
        "tp_filled_qty": {},
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "peak_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "max_drawdown_pct": 0.0,
    }
    (tmp_path / "active_trades.json").write_text(json.dumps({"EVT_QTY_RECON": trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {
        "status": "found", "positionAmt": 0.4, "avgPrice": 100.0,
    })
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status": "ok", "order_status": "NEW"})
    monkeypatch.setattr(tracker, "send_tg", lambda *a, **k: True)
    tracker.update_active_trades()
    saved = json.loads((tmp_path / "active_trades.json").read_text(encoding="utf-8"))
    assert saved["EVT_QTY_RECON"]["remaining_qty"] == 0.4
    assert saved["EVT_QTY_RECON"]["current_position_qty"] == 0.4
    assert saved["EVT_QTY_RECON"]["closed"] is False


def test_be_existing_with_old_sl_cancel_failure_is_not_success(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(tracker, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok",
        "sl_orders": [{"orderId": "BE1", "clientOrderId": "EVT_BE_T", "type": "STOP_MARKET", "stopPrice": "100", "origQty": "1"}],
        "tp_orders": [],
    })
    monkeypatch.setattr(tracker, "_cancel_old_sl_verified", lambda *a, **k: (False, "cancel timeout"))
    out = tracker._move_sl_to_break_even("AAA-USDT", "LONG", 100.0, 1.0, "OLD1", "T", old_sl_price=99.0)
    assert out["status"] == "error"
    assert "old SL cancel failed" in out["error"]


def test_be_restore_requires_exchange_verification(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(tracker, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "position_side_param", lambda direction: "BOTH")
    calls = []
    responses = iter([
        {"status": "ok", "sl_orders": [], "tp_orders": []},
        {"status": "ok", "sl_orders": [], "tp_orders": []},
    ])
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: next(responses))
    monkeypatch.setattr(tracker, "_cancel_old_sl_verified", lambda *a, **k: (True, "cancelled"))
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status": "found", "positionAmt": 1.0})
    monkeypatch.setattr(tracker, "close_position_market", lambda *a, **k: {"status": "closed"})
    monkeypatch.setattr(tracker.time, "sleep", lambda *a, **k: None)
    def fake_request(method, path, params):
        calls.append(params)
        return {"code": 0, "data": {"order": {"orderId": "RST1", "clientOrderId": params["clientOrderId"]}}}
    monkeypatch.setattr(tracker, "_request", fake_request)
    # First POST (BE) succeeds but its verification does not see the order,
    # forcing restore. Restore POST also succeeds, but restore verification does not.
    out = tracker._move_sl_to_break_even("AAA-USDT", "LONG", 100.0, 1.0, "OLD1", "T", old_sl_price=99.0)
    assert out["status"] == "error"
    assert out["old_sl_restored"] is False
    assert len(calls) == 2


def test_be_failure_without_restore_triggers_emergency_close(monkeypatch):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(tracker, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tracker, "position_side_param", lambda direction: "LONG")
    # Initial protection exists, then BE is accepted but invisible; restore is
    # also not provable. The safe outcome is emergency close + verified absence.
    responses = iter([
        {"status": "ok", "sl_orders": [], "tp_orders": []},
        {"status": "ok", "sl_orders": [], "tp_orders": []},
    ])
    monkeypatch.setattr(tracker, "get_open_protection_directional", lambda *a, **k: next(responses))
    monkeypatch.setattr(tracker, "_cancel_old_sl_verified", lambda *a, **k: (True, "cancelled"))
    requests = []
    def fake_request(method, path, params):
        requests.append(params)
        return {"code": 0, "data": {"order": {"orderId": "RST1", "clientOrderId": params["clientOrderId"]}}}
    monkeypatch.setattr(tracker, "_request", fake_request)
    pos_states = iter([
        {"status": "found", "positionAmt": 1.0},
        {"status": "not_found"},
    ])
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: next(pos_states))
    closes = []
    monkeypatch.setattr(tracker, "close_position_market", lambda *a, **k: closes.append((a, k)) or {"status": "closed"})
    monkeypatch.setattr(tracker, "send_tg", lambda *a, **k: True)
    monkeypatch.setattr(tracker.time, "sleep", lambda *a, **k: None)
    out = tracker._move_sl_to_break_even("AAA-USDT", "LONG", 100.0, 1.0, "OLD1", "T", old_sl_price=99.0)
    assert out["status"] == "error"
    assert out["safety_action"] == "emergency_close"
    assert out["rollback"]["status"] == "closed_verified"
    assert closes


def test_stale_scan_branch_does_not_reference_uninitialized_live_price():
    import ast
    source = Path("run_once.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    scan_one = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "scan_one"
    )
    stale = next(
        n for n in ast.walk(scan_one)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and any(isinstance(x, ast.Name) and x.id == "MAX_DATA_STALENESS_HOURS" for x in ast.walk(n.test))
    )
    names = [n.id for n in ast.walk(stale) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
    assert "binance_live_price" not in names


def test_crossed_tp_market_close_requires_execution_verification(monkeypatch):
    from event_engine import bingx
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda s: s)
    monkeypatch.setattr(bingx, "get_contract", lambda s: {
        "symbol": "AAA-USDT", "quantityPrecision": 3, "pricePrecision": 2, "tradeMinQuantity": 0.1,
    })
    monkeypatch.setattr(bingx, "position_side_param", lambda d: "BOTH")
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [{"orderId": "SL1", "type": "STOP_MARKET", "stopPrice": "99", "origQty": "1"}], "tp_orders": []
    })
    monkeypatch.setattr(bingx, "_current_close_price", lambda s: 101.0)
    monkeypatch.setattr(bingx, "_verify_open_order", lambda *a, **k: {"status": "verified"})
    monkeypatch.setattr(bingx, "cancel_order", lambda *a, **k: {"code": 0})
    monkeypatch.setattr(bingx, "_request", lambda method, path, params: {"code": 0, "data": {"order": {
        "orderId": "TPM1", "clientOrderId": params.get("clientOrderId")
    }}} if method == "POST" else {"code": 0})
    # POST is acknowledged, but neither order execution nor position reduction is visible.
    monkeypatch.setattr(bingx, "get_order", lambda *a, **k: {
        "status": "ok", "order_id": "TPM1", "order_status": "NEW", "executed_qty": 0.0, "avg_price": 0.0
    })
    monkeypatch.setattr(bingx, "get_position_directional", lambda *a, **k: {
        "status": "found", "positionAmt": 1.0, "avgPrice": 100.0
    })
    monkeypatch.setattr(bingx.time, "sleep", lambda *a, **k: None)
    out = bingx._verify_market_reduce_order("AAA-USDT", "LONG", "TPM1", 0.5, 1.0, attempts=1)
    assert out["status"] == "unverified"


def test_failed_signal_registry_uses_bounded_backoff_and_terminal_state(tmp_path, monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "DATA", tmp_path)
    monkeypatch.setattr(run_once, "FAILED_SIGNALS_PATH", tmp_path / "failed_signals.json")
    monkeypatch.setattr(run_once, "FAILED_SIGNAL_TTL_SEC", 3600)
    monkeypatch.setattr(run_once, "FAILED_SIGNAL_MAX_RETRIES", 3)
    monkeypatch.setattr(run_once, "FAILED_SIGNAL_RETRY_BASE_SEC", 10)
    monkeypatch.setattr(run_once, "FAILED_SIGNAL_RETRY_MAX_SEC", 100)
    monkeypatch.setattr(run_once.time, "time", lambda: 1000)
    run_once._mark_failed_signal("EVT_RETRY", "execution_error", "x")
    run_once._mark_failed_signal("EVT_RETRY", "execution_error", "y")
    run_once._mark_failed_signal("EVT_RETRY", "execution_error", "z")
    raw = json.loads((tmp_path / "failed_signals.json").read_text())
    record = raw["EVT_RETRY"]
    assert record["retry_count"] == 3
    assert record["terminal"] is True
    assert record["next_retry_at"] == 0
    assert run_once._failed_signal_is_blocked(record, now=2000) is True


def test_execution_outcome_categories_separate_technical_failures():
    import run_once
    assert run_once._execution_outcome_category("opened_protected") == "PROTECTED_ENTRY"
    assert run_once._execution_outcome_category("opened_then_emergency_closed") == "EMERGENCY_EXIT"
    assert run_once._execution_outcome_category("entry_state_unverified") == "EXECUTION_UNVERIFIED"
    assert run_once._execution_outcome_category("blocked_protection_preflight") == "PROTECTION_FAILURE"


def test_exit_outcome_categories_separate_strategy_and_unverified_closes():
    from event_engine import tracker
    assert tracker._exit_outcome_category("STOP_LOSS") == "STRATEGY_EXIT"
    assert tracker._exit_outcome_category("TAKE_PROFIT_FULL") == "STRATEGY_EXIT"
    assert tracker._exit_outcome_category("BREAK_EVEN") == "STRATEGY_EXIT"
    assert tracker._exit_outcome_category("POSITION_CLOSED_UNVERIFIED") == "UNVERIFIED_CLOSE"


def test_active_trade_state_save_uses_lock_and_atomic_replace(tmp_path, monkeypatch):
    from event_engine import tracker
    target = tmp_path / "active_trades.json"
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", target)
    tracker._save_active_trades({"EVT": {"symbol": "AAA-USDT", "closed": False}})
    assert json.loads(target.read_text(encoding="utf-8"))["EVT"]["symbol"] == "AAA-USDT"
    assert target.with_suffix(target.suffix + ".lock").exists()
    assert not target.with_name(target.name + ".tmp").exists()



def test_tracker_sends_and_records_tp_notification(monkeypatch, tmp_path):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    trade = {
        "event_id": "EVT_TP_NOTIFY",
        "symbol": "AAA-USDT",
        "direction": "LONG",
        "name": "AAA",
        "entry_price": 100.0,
        "initial_qty": 1.0,
        "remaining_qty": 1.0,
        "entry_ts": 1,
        "closed": False,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1"}],
        "sl_order": {},
        "hit_legs": [],
        "tp_filled_qty": {},
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "peak_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "planned_risk_pct": 1.0,
        "planned_weighted_rr": 0.75,
        "effective_weighted_rr": 0.75,
        "setup": {},
        "tp_levels": [],
        "effective_tp_levels": [],
    }
    (tmp_path / "active_trades.json").write_text(json.dumps({trade["event_id"]: trade}), encoding="utf-8")
    states = iter([
        {"status": "found", "positionAmt": 1.0, "avgPrice": 100.0},
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},
    ])
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: next(states))
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "FILLED", "executed_qty": 0.5,
        "avg_price": 102.0, "order_id": "TP1",
    })
    # TP1 notification is the subject of this test; avoid entering the BE
    # network path that normally follows a real TP1 fill.
    monkeypatch.setattr(tracker, "_move_sl_to_break_even", lambda *a, **k: {
        "status": "created", "order_id": "BE1", "stop_price": 100.0,
    })
    sent = []
    monkeypatch.setattr(tracker, "send_tg", lambda text: sent.append(text) or True)
    tracker.update_active_trades()
    assert sent and "tp1" in sent[0].lower()
    actions = [json.loads(x) for x in (tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert actions[-1]["action"] == "TP_HIT"
    assert actions[-1]["telegram_ok"] is True



def test_tp_notification_reports_exchange_remaining_qty_after_fill(monkeypatch, tmp_path):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    trade = {
        "event_id": "EVT_TP_REMAINING", "symbol": "AAA-USDT", "direction": "LONG", "name": "AAA",
        "entry_price": 100.0, "initial_qty": 1.0, "remaining_qty": 0.5, "entry_ts": 1, "closed": False,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1"}], "sl_order": {}, "hit_legs": [], "tp_filled_qty": {},
        "realized_pnl_qty": 0.0, "realized_pnl_weighted_sum": 0.0, "peak_pnl_pct": 0.0, "mae_pct": 0.0,
        "max_drawdown_pct": 0.0, "planned_risk_pct": 1.0, "planned_weighted_rr": 0.75,
        "effective_weighted_rr": 0.75, "setup": {}, "tp_levels": [], "effective_tp_levels": [],
    }
    (tmp_path / "active_trades.json").write_text(json.dumps({trade["event_id"]: trade}), encoding="utf-8")
    states = iter([
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},
        {"status": "found", "positionAmt": 0.5, "avgPrice": 100.0},
    ])
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: next(states))
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status": "ok", "order_status": "FILLED", "executed_qty": 0.5, "avg_price": 102.0, "order_id": "TP1"})
    monkeypatch.setattr(tracker, "_move_sl_to_break_even", lambda *a, **k: {"status": "created", "order_id": "BE1", "stop_price": 100.0})
    sent=[]
    monkeypatch.setattr(tracker, "send_tg", lambda text: sent.append(text) or True)
    tracker.update_active_trades()
    assert sent and "0.50000000" in sent[0]
    assert "50.0%" in sent[0]

def test_tracker_sends_and_records_close_notification(monkeypatch, tmp_path):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    trade = {
        "event_id": "EVT_CLOSE_NOTIFY",
        "symbol": "AAA-USDT",
        "direction": "LONG",
        "name": "AAA",
        "entry_price": 100.0,
        "initial_qty": 1.0,
        "remaining_qty": 1.0,
        "entry_ts": 1,
        "closed": False,
        "tp_orders": [],
        "sl_order": {},
        "hit_legs": [],
        "tp_filled_qty": {},
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "peak_pnl_pct": 2.0,
        "mae_pct": -1.0,
        "max_drawdown_pct": -1.0,
        "planned_risk_pct": 1.0,
        "planned_weighted_rr": 0.75,
        "effective_weighted_rr": 0.75,
        "setup": {},
        "tp_levels": [],
        "effective_tp_levels": [],
    }
    (tmp_path / "active_trades.json").write_text(json.dumps({trade["event_id"]: trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_all_orders", lambda *a, **k: [{
        "orderId": "CLOSE1", "status": "FILLED", "side": "SELL", "type": "MARKET",
        "executedQty": "1", "avgPrice": "102", "updateTime": 200,
    }])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status": "error", "error": "not found"})
    sent = []
    monkeypatch.setattr(tracker, "send_tg", lambda text: sent.append(text) or True)
    tracker.update_active_trades()
    assert sent and "сделка закрыта" in sent[0]
    assert "+2.00%" in sent[0]
    actions = [json.loads(x) for x in (tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert actions[-1]["action"] == "TRADE_CLOSE"
    assert actions[-1]["telegram_ok"] is True



def test_tracker_sends_negative_close_notification(monkeypatch, tmp_path):
    from event_engine import tracker
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "ACTIONS_PATH", tmp_path / "actions.jsonl")
    trade = {
        "event_id": "EVT_CLOSE_NOTIFY_NEG",
        "symbol": "AAA-USDT",
        "direction": "LONG",
        "name": "AAA",
        "entry_price": 100.0,
        "initial_qty": 1.0,
        "remaining_qty": 1.0,
        "entry_ts": 1,
        "closed": False,
        "tp_orders": [],
        "sl_order": {},
        "hit_legs": [],
        "tp_filled_qty": {},
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "peak_pnl_pct": 0.0,
        "mae_pct": -3.0,
        "max_drawdown_pct": -3.0,
        "planned_risk_pct": 1.0,
        "planned_weighted_rr": 0.75,
        "effective_weighted_rr": 0.75,
        "setup": {},
        "tp_levels": [],
        "effective_tp_levels": [],
    }
    (tmp_path / "active_trades.json").write_text(json.dumps({trade["event_id"]: trade}), encoding="utf-8")
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(tracker, "fetch_klines", lambda *a, **k: [])
    monkeypatch.setattr(tracker, "get_all_orders", lambda *a, **k: [{
        "orderId": "CLOSE_NEG", "status": "FILLED", "side": "SELL", "type": "MARKET",
        "executedQty": "1", "avgPrice": "97", "updateTime": 200,
    }])
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {"status": "error", "error": "not found"})
    sent = []
    monkeypatch.setattr(tracker, "send_tg", lambda text: sent.append(text) or True)
    tracker.update_active_trades()
    assert sent
    assert "💔" in sent[0]
    assert "-3.00%" in sent[0]
    actions = [json.loads(x) for x in (tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert actions[-1]["action"] == "TRADE_CLOSE"
    assert actions[-1]["telegram_ok"] is True
