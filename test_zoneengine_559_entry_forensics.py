from __future__ import annotations

import pytest


def _base_tp_plan():
    return [
        {"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 0.5},
        {"leg": "tp2", "pnl_pct": 6.0, "close_fraction": 0.5},
    ]


def test_partial_invalid_tp_plan_fails_closed_before_exchange_reads(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(
        bingx, "get_open_protection_directional",
        lambda *args, **kwargs: pytest.fail("invalid TP plan must be rejected before exchange reads"),
    )
    plan = _base_tp_plan()
    plan[1]["close_fraction"] = float("nan")

    result = bingx.ensure_directional_protection(
        "AAA-USDT", "LONG", 100.0, 1.0, 10.0, plan, trade_id="EVT_PARTIAL_BAD_TP",
    )

    assert result["status"] == "PROTECTION_FAILED"
    assert result["failure_class"] == "INVALID_TP_CONFIGURATION"
    assert "tp2" in result["error"]


def test_duplicate_case_insensitive_tp_legs_fail_closed(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(
        bingx, "get_open_protection_directional",
        lambda *args, **kwargs: pytest.fail("duplicate TP identities must be rejected before exchange reads"),
    )
    plan = [
        {"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 0.5},
        {"leg": "TP1", "pnl_pct": 6.0, "close_fraction": 0.5},
    ]

    result = bingx.ensure_directional_protection(
        "AAA-USDT", "LONG", 100.0, 1.0, 10.0, plan, trade_id="EVT_DUPLICATE_TP",
    )

    assert result["status"] == "PROTECTION_FAILED"
    assert result["failure_class"] == "INVALID_TP_CONFIGURATION"
    assert "duplicate TP leg" in result["error"]


def test_active_trade_validation_rejects_malformed_nested_tp_plan():
    from event_engine.state_validation import validate_active_trade_record

    trade = {
        "event_id": "EVT_BAD_TP_STATE",
        "setup": {"tp_levels": _base_tp_plan()},
    }
    trade["setup"]["tp_levels"][1]["pnl_pct"] = 0

    with pytest.raises(ValueError, match=r"setup\.tp_levels\[1\]\.pnl_pct"):
        validate_active_trade_record("EVT_BAD_TP_STATE", trade)


def test_active_trade_validation_rejects_duplicate_nested_tp_identities():
    from event_engine.state_validation import validate_active_trade_record

    trade = {
        "event_id": "EVT_DUPLICATE_TP_STATE",
        "setup": {
            "tp_levels": [
                {"leg": "tp1", "pnl_pct": 3.0, "close_fraction": 0.5},
                {"leg": "TP1", "pnl_pct": 6.0, "close_fraction": 0.5},
            ]
        },
    }

    with pytest.raises(ValueError, match="duplicate leg"):
        validate_active_trade_record("EVT_DUPLICATE_TP_STATE", trade)


def test_active_trade_validation_accepts_empty_tp_plan_during_recovery():
    from event_engine.state_validation import validate_active_trade_record

    validate_active_trade_record(
        "EVT_TP_RECOVERY",
        {"event_id": "EVT_TP_RECOVERY", "setup": {"tp_levels": []}},
    )
