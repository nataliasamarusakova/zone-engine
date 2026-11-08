from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _bars(now: pd.Timestamp) -> tuple[list[dict], list[dict], pd.DataFrame]:
    hour = now.floor("h") - pd.Timedelta(hours=2)
    bars_1h = []
    for offset in range(2):
        ts = hour + pd.Timedelta(hours=offset)
        bars_1h.append({
            "timestamp": ts.isoformat(),
            "close_time": (ts + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)).isoformat(),
            "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 100.0,
        })
    five = now.floor("5min") - pd.Timedelta(minutes=15)
    bars_5m = []
    for offset in range(3):
        ts = five + pd.Timedelta(minutes=5 * offset)
        bars_5m.append({
            "timestamp": ts.isoformat(),
            "close_time": (ts + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)).isoformat(),
            "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.1, "volume": 20.0,
        })
    return bars_1h, bars_5m, pd.DataFrame(bars_1h)


def _patch_research_paths(monkeypatch, tmp_path: Path, research) -> None:
    mapping = {
        "DATA_DIR": tmp_path,
        "ZONE_OBSERVATIONS_PATH": tmp_path / "zone_observations.jsonl",
        "ENTRY_DECISIONS_PATH": tmp_path / "entry_decisions.jsonl",
        "MARKET_BARS_1H_PATH": tmp_path / "market_bars_1h.jsonl",
        "MARKET_BARS_5M_PATH": tmp_path / "market_bars_5m.jsonl",
        "RESEARCH_BAR_CURSORS_PATH": tmp_path / "research_bar_cursors.json",
        "RESEARCH_MANIFEST_PATH": tmp_path / "research_manifest.json",
        "RESEARCH_ERRORS_PATH": tmp_path / "research_persistence_errors.jsonl",
        "RESEARCH_OUTCOMES_PATH": tmp_path / "research_outcomes.jsonl",
        "RESEARCH_OUTCOME_STATE_PATH": tmp_path / "research_outcome_state.json",
        "COUNTERFACTUAL_EXPERIMENTS_PATH": tmp_path / "counterfactual_experiments.jsonl",
        "MARKET_CONTEXT_PATH": tmp_path / "market_context.jsonl",
        "ACCOUNT_CONTEXT_PATH": tmp_path / "account_context.jsonl",
    }
    for name, value in mapping.items():
        monkeypatch.setattr(research, name, value)
    research._MARKET_BAR_INDEX_CACHE.clear()
    research._OBSERVATION_SEEN_IDS = None
    research._OBSERVATION_SEEN_IDS_PATH = None


def test_deferred_bar_only_path_avoids_dataframe_and_persists_bars(tmp_path, monkeypatch):
    from event_engine import research

    now = pd.Timestamp.now(tz="UTC")
    bars_1h, bars_5m, df_1h = _bars(now)
    _patch_research_paths(monkeypatch, tmp_path, research)

    original_dataframe = pd.DataFrame

    class ForbiddenDataFrame(original_dataframe):
        def __init__(self, *args, **kwargs):
            raise AssertionError("bar-only deferred research must not construct pandas DataFrames")

    monkeypatch.setattr(research.pd, "DataFrame", ForbiddenDataFrame)
    with research.batch_persistence() as batch:
        result = research.record_scan_symbol(
            scan_id="SCAN_FAST", symbol="FAST-USDT", strategy_version="v1", code_commit_sha="abc",
            provider="binance", source="binance_spot", bars_1h=bars_1h, bars_5m=bars_5m,
            df_1h=df_1h, demand=[], supply=[], diagnostics={"touch_events": [], "rearm_events": [], "zones": {}},
            symbol_state={}, signals=[], decision_ts=now.isoformat(), persist_bars_for_forward=True,
        )
    assert result["observations"] == 0
    assert result["bars_1h"] == len(bars_1h)
    assert result["bars_5m"] == len(bars_5m)
    assert batch["flush_result"]["persistence_errors"] == 0
    assert len((tmp_path / "market_bars_1h.jsonl").read_text().splitlines()) == len(bars_1h)
    assert len((tmp_path / "market_bars_5m.jsonl").read_text().splitlines()) == len(bars_5m)
    assert not (tmp_path / "zone_observations.jsonl").exists()
    cursors = json.loads((tmp_path / "research_bar_cursors.json").read_text())
    assert "FAST-USDT|1h|binance" in cursors
    assert "FAST-USDT|5m|binance" in cursors


def test_deferred_fast_path_does_not_skip_nearest_approach_observation(tmp_path, monkeypatch):
    from event_engine import research

    now = pd.Timestamp.now(tz="UTC")
    bars_1h, bars_5m, df_1h = _bars(now)
    _patch_research_paths(monkeypatch, tmp_path, research)
    monkeypatch.setattr(research, "build_research_features", lambda **kwargs: {"test_feature": 1})
    bar = {**bars_5m[-1], "distance_pct": 0.5}
    diagnostics = {
        "touch_events": [], "rearm_events": [],
        "zones": {"DEMAND:Z1": {
            "direction": "LONG", "closest_midpoint_distance_pct": 0.5,
            "closest_midpoint_bar": bar,
        }},
    }
    result = research.record_scan_symbol(
        scan_id="SCAN_NEAR", symbol="NEAR-USDT", strategy_version="v1", code_commit_sha="abc",
        provider="binance", source="binance_spot", bars_1h=bars_1h, bars_5m=bars_5m,
        df_1h=df_1h, demand=[{"zone_id": "Z1", "top": 10.2, "btm": 9.8, "poi": 10.0}],
        supply=[], diagnostics=diagnostics, symbol_state={}, signals=[], decision_ts=now.isoformat(),
        persist_bars_for_forward=True,
    )
    assert result["observations"] == 1
    rows = [json.loads(line) for line in (tmp_path / "zone_observations.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event_type"] == "NEAREST_APPROACH"


def test_funding_rate_fallback_is_used_when_premium_index_has_no_timestamp(monkeypatch):
    from event_engine import bingx

    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    calls = []
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "XRP-USDT")

    def fake_get(endpoint_key, path, params, **kwargs):
        calls.append((endpoint_key, path, params))
        if endpoint_key == "premiumIndex":
            return {"code": 0, "data": {"symbol": "XRP-USDT", "lastFundingRate": "0.00012", "nextFundingTime": now_ms + 3_600_000}}
        if endpoint_key == "fundingRate":
            return {"code": 0, "data": [{
                "symbol": "XRP-USDT", "fundingRate": "0.00012",
                "fundingTime": now_ms - 3_600_000,
                "nextFundingTime": now_ms + 3_600_000,
            }]}
        raise AssertionError(f"unexpected endpoint {endpoint_key}")

    monkeypatch.setattr(bingx, "_research_public_get", fake_get)
    result = bingx.fetch_funding_snapshot("XRP-USDT")
    assert result["status"] == "ok"
    assert result["funding_rate"] == pytest.approx(0.00012)
    assert result["funding_rate_pct"] == pytest.approx(0.012)
    assert result["funding_source"] == "fundingRate"
    assert result["timestamp_source"] == "funding_period"
    assert result["funding_time_ms"] == now_ms - 3_600_000
    assert result["next_funding_time_ms"] == now_ms + 3_600_000
    assert [call[0] for call in calls] == ["premiumIndex", "fundingRate"]


def test_funding_fallback_rejects_expired_or_invalid_funding_period(monkeypatch):
    from event_engine import bingx

    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(
        bingx, "_research_public_get",
        lambda endpoint_key, path, params, **kwargs: (
            {"code": 0, "data": {"symbol": "BTC-USDT", "lastFundingRate": "0.0001"}}
            if endpoint_key == "premiumIndex" else
            {"code": 0, "data": {"symbol": "BTC-USDT", "fundingRate": "0.0001",
                                  "fundingTime": now_ms - 8 * 3_600_000,
                                  "nextFundingTime": now_ms - 60_000}}
        ),
    )
    result = bingx.fetch_funding_snapshot("BTC-USDT")
    assert result["status"] == "stale"
    assert "funding_rate_pct" not in result


def test_telegram_execution_error_is_in_status_and_error_field_is_removed():
    from event_engine.telegram import format_signal

    message = format_signal(
        {"type": "LONG", "symbol": "BTC-USDT", "entry": 100, "sl": 90, "tp1": 103, "tp2": 106,
         "risk_pct": 10, "tp1_rr": 0.3, "tp2_rr": 0.6, "target": {"tp1_pct": 3, "tp2_pct": 6}},
        execution={"status": "protection_failed", "error": "SL confirmation timeout",
                   "order": {"order_id": "123"}, "protection": {"status": "UNPROTECTED"}},
        funding={"status": "unavailable"},
    )
    assert "Status: <code>protection_failed · SL confirmation timeout</code>" in message
    assert "Protection: <code>UNPROTECTED</code>" in message
    assert "Error:" not in message
    assert "Funding: <code>UNAVAILABLE</code>" in message


def test_funding_formatter_shows_unverified_state_instead_of_dash():
    from event_engine.telegram import _fmt_funding

    assert _fmt_funding({"status": "unavailable"}) == "UNAVAILABLE"
    assert _fmt_funding({"status": "unverified"}) == "UNVERIFIED"
    assert _fmt_funding({"status": "stale"}) == "STALE"
    assert _fmt_funding({"status": "ok", "funding_rate_pct": 0.012}) == "+0.0120%"


def test_workflow_uses_bounded_faster_scan_without_sampling_or_parallel_writers():
    from pathlib import Path

    workflow = (Path(__file__).parent / ".github" / "workflows" / "event-engine.yml").read_text(encoding="utf-8")
    assert 'SCAN_WORKERS: "16"' in workflow
    assert 'SCAN_BATCH_SIZE: "64"' in workflow
    assert 'MAX_SCAN_SYMBOLS: "0"' in workflow
    assert 'WATCHLIST_ONLY: "false"' in workflow
    assert "cancel-in-progress: false" in workflow


def test_funding_fallback_selects_latest_exchange_period_not_response_order(monkeypatch):
    from event_engine import bingx

    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "ETH-USDT")
    seen = {}

    def fake_get(endpoint_key, path, params, **kwargs):
        if endpoint_key == "premiumIndex":
            return {"code": 0, "data": {"symbol": "ETH-USDT", "lastFundingRate": "0.0001"}}
        seen.update(params)
        return {"code": 0, "data": [
            {"symbol": "ETH-USDT", "fundingRate": "0.0001", "fundingTime": now_ms - 8 * 3_600_000, "nextFundingTime": now_ms + 8 * 3_600_000},
            {"symbol": "ETH-USDT", "fundingRate": "0.0002", "fundingTime": now_ms - 3_600_000, "nextFundingTime": now_ms + 3_600_000},
        ]}

    monkeypatch.setattr(bingx, "_research_public_get", fake_get)
    result = bingx.fetch_funding_snapshot("ETH-USDT")
    assert result["status"] == "ok"
    assert result["funding_rate"] == pytest.approx(0.0002)
    assert result["funding_time_ms"] == now_ms - 3_600_000
    assert seen == {"symbol": "ETH-USDT", "limit": 1}


def test_funding_lookup_never_accepts_a_single_row_for_another_symbol(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "XRP-USDT")
    monkeypatch.setattr(
        bingx, "_research_public_get",
        lambda endpoint_key, path, params, **kwargs: {
            "code": 0,
            "data": [{"symbol": "BTC-USDT", "lastFundingRate": "0.0001", "time": 1760000000000,
                      "fundingRate": "0.0001", "fundingTime": 1759990000000, "nextFundingTime": 1760000000000}],
        },
    )
    result = bingx.fetch_funding_snapshot("XRP-USDT")
    assert result["status"] == "unverified"
    assert "funding_rate_pct" not in result
