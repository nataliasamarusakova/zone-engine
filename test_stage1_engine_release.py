from __future__ import annotations

import json
import math


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_tp_fill_timestamp_enrichment_uses_only_new_cumulative_delta(monkeypatch):
    from event_engine import tracker

    calls = []

    def fills(symbol, start_ms, end_ms, *, order_id=None, limit=1000, timeout_sec=None):
        calls.append((symbol, order_id, start_ms, end_ms, limit, timeout_sec))
        return [
            {"order_id": "TP1", "trade_id": "T-1", "qty": 0.2, "price": 103.0, "time_ms": 1_100},
            {"order_id": "TP1", "trade_id": "T-2", "qty": 0.3, "price": 103.1, "time_ms": 1_200},
        ]

    monkeypatch.setattr(tracker, "get_fill_orders", fills)
    event = {"order_id": "TP1", "order_update_time_ms": 1_300, "fill_time_ms": None}
    out = tracker._enrich_tp_fill_event_from_exchange_history(
        "AAA-USDT", event, entry_ts_ms=1_000,
        previous_qty=0.2, executed_qty=0.5, delta_qty=0.3, now_ms=1_400,
    )

    assert len(calls) == 1
    assert calls[0][-1] == 1.0
    assert out["fill_time_ms"] == 1_200
    assert out["fill_time_start_ms"] == 1_200
    assert out["fill_time_source"] == "EXCHANGE_FILL_HISTORY"
    assert out["fill_timestamp_status"] == "EXACT_FILL_TIMESTAMP_CONFIRMED"
    assert out["fill_trade_ids"] == ["T-2"]
    assert out["fill_history_delta_qty"] == 0.3
    # Aggregate quantity/price and the order-update fallback are not overwritten.
    assert out["order_update_time_ms"] == 1_300


def test_tp_fill_timestamp_enrichment_keeps_fallback_on_incomplete_history(monkeypatch):
    from event_engine import tracker

    monkeypatch.setattr(
        tracker,
        "get_fill_orders",
        lambda *args, **kwargs: [
            {"order_id": "TP1", "trade_id": "T-1", "qty": 0.2, "price": 103.0, "time_ms": 1_100}
        ],
    )
    # Even a stale candidate derived from the whole order history must be cleared
    # when the exact newly-accounted delta cannot be proven.
    event = {
        "order_id": "TP1", "order_update_time_ms": 1_300,
        "fill_time_ms": 1_100, "fill_time_source": "EXCHANGE_FILL_HISTORY",
    }
    out = tracker._enrich_tp_fill_event_from_exchange_history(
        "AAA-USDT", event, entry_ts_ms=1_000,
        previous_qty=0.0, executed_qty=0.5, delta_qty=0.5, now_ms=1_400,
    )

    assert out["fill_time_ms"] is None
    assert out["fill_time_source"] == "ORDER_UPDATE_TIME_FALLBACK"
    assert out["fill_timestamp_status"] == "FILL_HISTORY_CUMULATIVE_QTY_MISMATCH"
    assert out["order_update_time_ms"] == 1_300


def test_pre_entry_feature_and_execution_quality_telemetry_are_linked_and_sanitized(tmp_path, monkeypatch):
    from event_engine import telemetry

    monkeypatch.setattr(telemetry, "PRE_ENTRY_FEATURES_PATH", tmp_path / "pre_entry_features.jsonl")
    monkeypatch.setattr(telemetry, "EXECUTION_QUALITY_PATH", tmp_path / "execution_quality.jsonl")
    signal = {
        "strategy_version": "strategy-test",
        "trigger_timeframe": "5m",
        "entry": 100.0,
        "sl": 95.0,
        "tp1": 103.0,
        "tp2": 106.0,
        "zone": {"top": 100.0, "bottom": 98.0},
        "signal_forensics": {"penetration": 0.8, "not_finite": float("nan")},
        "confirmation": {"volume_ratio": 1.4},
        "market_snapshot": {"analysis_provider": "binance"},
        "pre_execution_market_context": {
            "status": "partial",
            "funding_rate": 0.0001,
            "order_book": {"status": "ok", "bid_quote_5": 1_000.0},
            "recent_trades": {"status": "ok", "sample_count": 50},
        },
    }
    quote = {
        "bid": 99.9, "ask": 100.1, "spread_pct": 0.2,
        "quote_source": "bookTicker", "quote_observed_at_ms": 1_000,
        "quote_local_age_sec": 0.1, "quote_freshness_status": "FRESH",
    }

    telemetry.record_pre_entry_feature_snapshot(
        event_id="EVT_STAGE1", attempt_id="ATT_STAGE1", symbol="AAA-USDT",
        direction="LONG", signal=signal, quote=quote, code_commit_sha="abc123",
    )
    telemetry.record_execution_quality_snapshot(
        event_id="EVT_STAGE1", attempt_id="ATT_STAGE1", symbol="AAA-USDT",
        direction="LONG", stage="PRE_ENTRY", quote=quote,
        signal_price=100.0, execution_reference_price=100.1,
        market_context=signal["pre_execution_market_context"],
    )
    telemetry.record_execution_quality_snapshot(
        event_id="EVT_STAGE1", attempt_id="ATT_STAGE1", symbol="AAA-USDT",
        direction="LONG", stage="POST_FILL", quote=quote,
        signal_price=100.0, execution_reference_price=100.1,
        fill_price=100.2, signed_slippage_pct=0.0999,
        adverse_slippage_pct=0.0999,
        market_context=signal["pre_execution_market_context"],
    )

    feature_row = _read_jsonl(tmp_path / "pre_entry_features.jsonl")[0]
    quality_rows = _read_jsonl(tmp_path / "execution_quality.jsonl")
    assert feature_row["event_id"] == "EVT_STAGE1"
    assert feature_row["signal_features"]["zone"]["top"] == 100.0
    assert feature_row["signal_features"]["signal_forensics"]["not_finite"] is None
    assert feature_row["feature_availability"]["order_book_depth"] == "ok"
    assert feature_row["feature_collection_mode"] == "LOCAL_SNAPSHOT_NO_ADDITIONAL_MARKET_REQUESTS"
    assert len(quality_rows) == 2
    assert [row["stage"] for row in quality_rows] == ["PRE_ENTRY", "POST_FILL"]
    assert quality_rows[1]["actual_fill_price"] == 100.2
    assert quality_rows[1]["adverse_slippage_pct"] == 0.0999
    # A telemetry call must not mutate the signal that continues into execution.
    assert math.isnan(signal["signal_forensics"]["not_finite"])


def test_emergency_close_records_reason_and_verified_result(tmp_path, monkeypatch):
    from event_engine import telemetry
    import run_once

    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", tmp_path / "emergency_exits.jsonl")
    monkeypatch.setattr(run_once, "get_position_directional", lambda symbol, direction: {"status": "not_found"})

    result = run_once._emergency_close_and_verify(
        "AAA-USDT", "LONG", 0.5, "EVT_STAGE1_CLOSE",
        reason="TEST_CONFIRMED_EMERGENCY_REASON",
    )

    rows = _read_jsonl(tmp_path / "emergency_exits.jsonl")
    assert result["status"] == "closed_verified"
    assert len(rows) == 1
    assert rows[0]["event_id"] == "EVT_STAGE1_CLOSE"
    assert rows[0]["reason"] == "TEST_CONFIRMED_EMERGENCY_REASON"
    assert rows[0]["verified_flat"] is True
    assert rows[0]["requested_qty"] == 0.5


def test_tracker_be_emergency_rollback_records_owner_and_reason(tmp_path, monkeypatch):
    from event_engine import tracker, telemetry

    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", tmp_path / "emergency_exits.jsonl")
    monkeypatch.setattr(tracker, "get_position_directional", lambda *args, **kwargs: {"status": "not_found"})
    monkeypatch.setattr(
        tracker,
        "_cancel_engine_protection_before_emergency_close",
        lambda *args, **kwargs: {
            "status": "partial",
            "verified": False,
            "cancelled": ["SL-OLD"],
            "errors": ["TP-OLD: cancel rejected"],
        },
    )

    result = tracker._emergency_close_after_be_failure(
        "AAA-USDT", "LONG", 0.5, "TRADE-TOKEN",
        event_id="EVT_BE_ROLLBACK", attempt_id="ATT_BE_ROLLBACK",
    )

    rows = _read_jsonl(tmp_path / "emergency_exits.jsonl")
    assert result["status"] == "already_closed"
    assert len(rows) == 1
    assert rows[0]["event_id"] == "EVT_BE_ROLLBACK"
    assert rows[0]["attempt_id"] == "ATT_BE_ROLLBACK"
    assert rows[0]["reason"] == "BE_PROTECTION_FAILURE_OLD_SL_UNVERIFIED"
    assert rows[0]["verified_flat"] is True
    assert rows[0]["cleanup_after_flat_verified"] == {
        "status": "partial",
        "verified": False,
        "cancelled": ["SL-OLD"],
        "errors": ["TP-OLD: cancel rejected"],
    }


def test_pre_entry_snapshot_marks_disabled_optional_context_as_not_captured(tmp_path, monkeypatch):
    from event_engine import telemetry

    monkeypatch.setattr(telemetry, "PRE_ENTRY_FEATURES_PATH", tmp_path / "pre_entry_features.jsonl")
    telemetry.record_pre_entry_feature_snapshot(
        event_id="EVT_NO_CONTEXT", attempt_id="ATT_NO_CONTEXT",
        symbol="AAA-USDT", direction="SHORT",
        signal={"score": 81.0, "volume_ratio_1h20": 1.7, "decision_ts": 1_234},
        quote={"bid": 99.0, "ask": 99.2, "spread_pct": 0.2},
    )

    row = _read_jsonl(tmp_path / "pre_entry_features.jsonl")[0]
    assert row["signal_features"]["score"] == 81.0
    assert row["signal_features"]["volume_ratio_1h20"] == 1.7
    assert row["signal_features"]["decision_ts"] == 1_234
    assert row["feature_availability"]["pre_execution_context"] == "NOT_CAPTURED"
    assert row["feature_availability"]["funding_context"] == "NOT_CAPTURED"
    assert row["feature_availability"]["order_book_depth"] == "NOT_CAPTURED"


def test_fill_history_timeout_can_be_bounded_for_tracker_enrichment(monkeypatch):
    from event_engine import bingx

    captured = {}
    monkeypatch.setattr(bingx, "credentials_available", lambda: True)
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "AAA-USDT")
    monkeypatch.setattr(bingx, "_normalize_all_fill_orders_data", lambda raw: [])

    def request(*args, **kwargs):
        captured.update(kwargs)
        return {"code": 0, "data": []}

    monkeypatch.setattr(bingx, "_request", request)
    result = bingx.get_fill_orders(
        "AAA-USDT", 1_000, 2_000, order_id="TP1", timeout_sec=1.0,
    )
    assert result == []
    assert captured["timeout_sec"] == 1.0


def test_stage1_telemetry_journals_are_included_in_runtime_state_bundle(tmp_path):
    from event_engine.runtime_state_bundle import _collect_source_files

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    journal_names = (
        "pre_entry_features.jsonl",
        "execution_quality.jsonl",
        "emergency_exits.jsonl",
    )
    for name in journal_names:
        (data_dir / name).write_text('{"record_type":"TEST"}\n', encoding="utf-8")

    bundled = {relative for _source, relative in _collect_source_files(data_dir)}
    assert {f"data/{name}" for name in journal_names} <= bundled


def test_emergency_exit_journal_persists_verified_protection_cleanup(tmp_path, monkeypatch):
    from event_engine import telemetry

    monkeypatch.setattr(telemetry, "EMERGENCY_EXITS_PATH", tmp_path / "emergency_exits.jsonl")
    telemetry.record_emergency_exit(
        event_id="EVT_CLEANUP_AUDIT",
        attempt_id="ATT_CLEANUP_AUDIT",
        position_id="POS_CLEANUP_AUDIT",
        symbol="AAA-USDT",
        direction="LONG",
        reason="BE_PROTECTION_FAILURE_OLD_SL_UNVERIFIED",
        requested_qty=0.5,
        result={
            "status": "closed_verified",
            "remaining_qty": 0.0,
            "verification": {"status": "not_found"},
            "attempts": [],
            "cleanup_after_flat_verified": {
                "status": "partial",
                "verified": False,
                "cancelled": ["SL-OLD"],
                "errors": ["TP-OLD: cancel rejected"],
            },
        },
    )

    rows = _read_jsonl(tmp_path / "emergency_exits.jsonl")
    assert len(rows) == 1
    cleanup = rows[0]["cleanup_after_flat_verified"]
    assert cleanup["status"] == "partial"
    assert cleanup["verified"] is False
    assert cleanup["cancelled"] == ["SL-OLD"]
    assert cleanup["errors"] == ["TP-OLD: cancel rejected"]


def test_pre_entry_snapshot_captures_available_btc_mtf_vwap_basis_and_account_context(tmp_path, monkeypatch):
    from event_engine import telemetry

    monkeypatch.setattr(telemetry, "PRE_ENTRY_FEATURES_PATH", tmp_path / "pre_entry_features.jsonl")
    signal = {
        "event_id": "EVT_FULL_CONTEXT",
        "attempt_id": "ATT_FULL_CONTEXT",
        "score": 88.0,
        "zone": {"top": 101.0, "bottom": 99.0},
        "confirmation": {"mtf_zone_confluence": {"status": "confluent", "poi_distance_pct": 0.12}},
        "trend_filter": {"btc_regime": "RISK_ON", "btc_bar_close_ts": 1_700_000_000_000, "btc_chg_1h_pct": 0.8},
        "vwap_context": {"status": "PRESENT", "kind": "UTC_DAY_TP_VWAP", "session_vwap_approx": 100.2, "session_vwap_distance_pct": -0.2},
        "pre_entry_market_features": {"5m_atr14": 0.4, "btc_return_1h_pct": 0.8},
        "market_snapshot": {
            "binance_live_price": 100.0,
            "bingx_live_price": 100.1,
            "market_spread_pct": 0.1,
            "market_spread_metric": "cross_venue_price_deviation",
            "scan_time": "2026-11-08T10:00:00Z",
        },
        "portfolio_context": {"status": "captured", "open_positions_count": 2},
        "open_positions_snapshot": [{"symbol": "BBB-USDT", "direction": "SHORT", "notional_usdt": 50.0}],
        "margin_snapshot": {"equity": 500.0, "used_margin": 100.0, "available_margin": 400.0},
        "signal_forensics": {"zone_penetration_ratio": 0.82},
    }
    telemetry.record_pre_entry_feature_snapshot(
        event_id="EVT_FULL_CONTEXT", attempt_id="ATT_FULL_CONTEXT",
        symbol="AAA-USDT", direction="LONG", signal=signal,
        quote={"bid": 99.9, "ask": 100.1, "spread_pct": 0.2},
        code_commit_sha="commit-test",
    )

    row = _read_jsonl(tmp_path / "pre_entry_features.jsonl")[0]
    assert row["signal_features"]["confirmation"]["mtf_zone_confluence"]["status"] == "confluent"
    assert row["signal_features"]["trend_filter"]["btc_regime"] == "RISK_ON"
    assert row["signal_features"]["pre_entry_market_features"]["btc_return_1h_pct"] == 0.8
    assert row["supplemental_context"]["vwap_context"]["session_vwap_approx"] == 100.2
    assert row["supplemental_context"]["cross_venue_basis"] is None
    assert row["supplemental_context"]["cross_venue_price_deviation"]["market_spread_pct"] == 0.1
    assert row["supplemental_context"]["portfolio_context"]["open_positions_count"] == 2
    assert row["supplemental_context"]["margin_snapshot"]["available_margin"] == 400.0
    assert row["feature_availability"]["mtf_zone_confluence"] == "confluent"
    assert row["feature_availability"]["btc_context"] == "PRESENT"
    assert row["feature_availability"]["vwap_context"] == "PRESENT"
    assert row["feature_availability"]["cross_venue_basis"] == "NOT_CAPTURED"
    assert row["feature_availability"]["cross_venue_price_deviation"] == "PRESENT"
    assert row["feature_availability"]["portfolio_context"] == "PRESENT"
    assert row["feature_availability"]["margin_snapshot"] == "PRESENT"
    assert row["feature_availability"]["open_positions_snapshot"] == "PRESENT"
    assert row["feature_collection_mode"] == "LOCAL_SNAPSHOT_NO_ADDITIONAL_MARKET_REQUESTS"


def test_pre_entry_snapshot_marks_unavailable_research_context_explicitly(tmp_path, monkeypatch):
    from event_engine import telemetry

    monkeypatch.setattr(telemetry, "PRE_ENTRY_FEATURES_PATH", tmp_path / "pre_entry_features.jsonl")
    telemetry.record_pre_entry_feature_snapshot(
        event_id="EVT_CONTEXT_ABSENT", attempt_id="ATT_CONTEXT_ABSENT",
        symbol="AAA-USDT", direction="SHORT", signal={"entry": 100.0}, quote={},
    )
    row = _read_jsonl(tmp_path / "pre_entry_features.jsonl")[0]
    for key in (
        "btc_context", "mtf_zone_confluence", "vwap_context", "cross_venue_basis",
        "portfolio_context", "open_positions_snapshot", "margin_snapshot",
    ):
        assert row["feature_availability"][key] == "NOT_CAPTURED"


def test_pre_entry_snapshot_records_when_optional_shadow_basis_request_was_attempted(tmp_path, monkeypatch):
    from event_engine import telemetry

    monkeypatch.setattr(telemetry, "PRE_ENTRY_FEATURES_PATH", tmp_path / "pre_entry_features.jsonl")
    telemetry.record_pre_entry_feature_snapshot(
        event_id="EVT_BASIS_ATTEMPT", attempt_id="ATT_BASIS_ATTEMPT",
        symbol="AAA-USDT", direction="LONG",
        signal={"cross_venue_basis": {
            "status": "UNSYNCHRONIZED", "shadow_only": True,
            "collection_mode": "OPTIONAL_BINANCE_BOOK_TICKER_REQUEST",
        }},
        quote={},
    )
    row = _read_jsonl(tmp_path / "pre_entry_features.jsonl")[0]
    assert row["feature_availability"]["cross_venue_basis"] == "NOT_CAPTURED"
    assert row["feature_collection_mode"] == "LOCAL_SNAPSHOT_PLUS_OPTIONAL_SHADOW_BINANCE_BOOK_TICKER"
