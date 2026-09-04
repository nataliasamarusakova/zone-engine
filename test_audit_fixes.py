import json
import os

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
