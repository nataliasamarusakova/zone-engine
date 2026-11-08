from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from event_engine import research, data_retention
from run_once import _prepare_deferred_research_5m_bars, _prepare_pre_entry_research_5m_bars


def _bar(ts: datetime, close: float, *, volume: float = 10.0) -> dict:
    return {
        "timestamp": ts.isoformat(),
        "close_time": (ts + timedelta(minutes=5) - timedelta(milliseconds=1)).isoformat(),
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": volume,
        "quote_volume": close * volume,
        "taker_buy_quote": close * volume * 0.6,
        "bar_delta_usdt": close * volume * 0.2,
    }


def test_load_market_bar_streams_reads_only_requested_symbol_provider_streams(tmp_path, monkeypatch):
    path = tmp_path / "market_bars_5m.jsonl"
    ts = datetime(2026, 11, 8, 0, 0, tzinfo=timezone.utc)
    rows = [
        {**_bar(ts, 100), "symbol": "BTC-USDT", "timeframe": "5m", "provider": "binance"},
        {**_bar(ts, 200), "symbol": "ETH-USDT", "timeframe": "5m", "provider": "binance"},
        {**_bar(ts, 300), "symbol": "BTC-USDT", "timeframe": "5m", "provider": "bingx"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(research, "MARKET_BARS_5M_PATH", path)

    loaded = research.load_market_bar_streams([
        ("BTC-USDT", "5m", "binance"),
        ("ETH-USDT", "5m", "binance"),
    ])

    assert len(loaded[("BTC-USDT", "5m", "binance")]) == 1
    assert len(loaded[("ETH-USDT", "5m", "binance")]) == 1
    assert ("BTC-USDT", "5m", "bingx") not in loaded


def test_merge_market_bar_history_sorts_deduplicates_and_prefers_current_rows():
    start = datetime(2026, 11, 8, 0, 0, tzinfo=timezone.utc)
    persisted = [_bar(start + timedelta(minutes=5 * i), 100 + i) for i in range(300)]
    current = [_bar(start + timedelta(minutes=5 * i), 200 + i) for i in range(156, 301)]

    merged = research.merge_market_bar_history(persisted, current, timeframe="5m")

    assert len(merged) == 301
    stamps = [pd.Timestamp(row["timestamp"]) for row in merged]
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == 301
    assert merged[156]["close"] == 356
    assert merged[-1]["close"] == 500


def test_derived_session_metrics_expose_full_utc_day_coverage_and_long_returns():
    start = pd.Timestamp("2026-11-08T00:00:00Z")
    bars = [_bar((start + pd.Timedelta(minutes=5 * i)).to_pydatetime(), 100 + i * 0.01) for i in range(150)]
    decision_ts = "2026-11-08T12:30:00+00:00"
    df = pd.DataFrame(bars)

    features = research._derive_time_series_features(
        direction="LONG", df_5m=df, df_1h=None, decision_ts=decision_ts,
        btc_df_5m=None, btc_df_1h=None,
    )

    assert features["session_vwap_coverage"] == "FULL"
    assert features["session_cvd_coverage"] == "FULL"
    assert features["session_vwap_available_bars"] == 150
    assert features["session_vwap_expected_bars"] == 150
    assert features["5m_return_12h_pct"] is not None
    assert features["5m_return_12h_status"] == "AVAILABLE"
    assert features["5m_return_24h_pct"] is None  # only 12.5h is supplied
    assert features["5m_return_24h_status"] == "INSUFFICIENT_HISTORY"


def test_initial_1h_research_persistence_keeps_1000_bars():
    assert research.INITIAL_1H_BARS >= 1000


def test_workflow_explicitly_configures_1000_hourly_history_and_retention():
    workflow = (Path(__file__).parent / ".github/workflows/event-engine.yml").read_text(encoding="utf-8")
    assert 'RESEARCH_INITIAL_1H_BARS: "1000"' in workflow
    assert 'DATA_RETENTION_1H_HOURS: "1000"' in workflow


def test_size_guard_never_prunes_hourly_research_history_below_800_hours(tmp_path, monkeypatch):
    now = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
    path = tmp_path / "market_bars_1h.jsonl"
    policies = data_retention._size_guard_plan(
        path, now=now, pending_ids=set(), pending_bar_cutoffs={},
        protected_observation_ids=set(), protected_refs=set(),
    )
    hours = [int(name.split("_")[3].removesuffix("h")) for name, _ in policies]
    assert hours
    assert min(hours) >= 804


def test_deferred_research_pipeline_merges_persisted_5m_history_for_24h_features():
    start = datetime(2026, 11, 7, 0, 0, tzinfo=timezone.utc)
    persisted = [
        {**_bar(start + timedelta(minutes=5 * i), 100 + i * 0.01),
         "symbol": "BTC-USDT", "timeframe": "5m", "provider": "binance"}
        for i in range(300)
    ]
    current = [
        {**_bar(start + timedelta(minutes=5 * i), 100 + i * 0.01),
         "symbol": "BTC-USDT", "timeframe": "5m", "provider": "binance"}
        for i in range(156, 301)
    ]
    history = {("BTC-USDT", "5m", "binance"): persisted}
    payload = {"symbol": "BTC-USDT", "provider": "binance", "bars_5m": current}

    merged = _prepare_deferred_research_5m_bars(history, payload, None)
    features = research._derive_time_series_features(
        direction="LONG", df_5m=pd.DataFrame(merged), df_1h=None,
        decision_ts="2026-11-08T01:00:00+00:00", btc_df_5m=None, btc_df_1h=None,
    )

    assert len(merged) == 301
    assert features["5m_history_bars_available"] == 300
    assert features["5m_return_24h_status"] == "AVAILABLE"
    assert features["5m_return_24h_pct"] is not None
    assert features["session_vwap_coverage"] == "FULL"
    assert features["session_cvd_coverage"] == "FULL"
    assert len(history[("BTC-USDT", "5m", "binance")]) == 301


def test_partial_utc_day_is_never_labelled_as_full_session_vwap_or_cvd():
    start = pd.Timestamp("2026-11-08T00:00:00Z")
    bars = [_bar((start + pd.Timedelta(minutes=5 * i)).to_pydatetime(), 100 + i * 0.01) for i in range(144)]
    bars[10].pop("bar_delta_usdt")
    features = research._derive_time_series_features(
        direction="LONG", df_5m=pd.DataFrame(bars), df_1h=None,
        decision_ts="2026-11-08T12:30:00+00:00", btc_df_5m=None, btc_df_1h=None,
    )

    assert features["session_vwap_coverage"] == "PARTIAL"
    assert features["session_vwap_kind"] == "PARTIAL_UTC_DAY_TP_VWAP"
    assert features["session_vwap_available_bars"] == 144
    assert features["session_vwap_expected_bars"] == 150
    assert features["session_cvd_coverage"] == "PARTIAL"


def test_1h_retention_default_keeps_more_than_804_hours():
    assert data_retention.MARKET_1H_RETENTION_HOURS >= 1000


def test_persist_market_bars_bootstraps_1000_hourly_source_rows(tmp_path, monkeypatch):
    start = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1000)
    rows = []
    for i in range(1000):
        ts = start + pd.Timedelta(hours=i)
        row = {
            "timestamp": ts.isoformat(),
            "close_time": (ts + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)).isoformat(),
            "open": 100 + i * 0.01,
            "high": 102 + i * 0.01,
            "low": 99 + i * 0.01,
            "close": 101 + i * 0.01,
            "volume": 10.0,
            "quote_volume": (101 + i * 0.01) * 10.0,
            "taker_buy_base": 5.0,
            "taker_buy_quote": (101 + i * 0.01) * 5.0,
            "bar_delta_usdt": 0.0,
        }
        rows.append(row)
    monkeypatch.setattr(research, "MARKET_BARS_1H_PATH", tmp_path / "market_bars_1h.jsonl")
    monkeypatch.setattr(research, "RESEARCH_BAR_CURSORS_PATH", tmp_path / "research_bar_cursors.json")
    monkeypatch.setattr(research, "RESEARCH_MANIFEST_PATH", tmp_path / "research_manifest.json")
    monkeypatch.setattr(research, "RESEARCH_ENABLED", True)
    monkeypatch.setattr(research, "INITIAL_1H_BARS", 1000)
    monkeypatch.setattr(research, "_bump_manifest", lambda *_args, **_kwargs: None)

    written = research.persist_market_bars(
        "BTC-USDT", "1h", rows, provider="binance", source="test",
        scan_id="SCAN_TEST", code_commit_sha="test-sha",
    )

    saved = [json.loads(line) for line in (tmp_path / "market_bars_1h.jsonl").read_text(encoding="utf-8").splitlines()]
    assert written == 1000
    assert len(saved) == 1000


def test_1000_hourly_bars_provide_200_complete_4h_candles_for_ema200():
    start = pd.Timestamp.now(tz="UTC").floor("4h") - pd.Timedelta(hours=1000)
    rows = []
    for i in range(1000):
        ts = start + pd.Timedelta(hours=i)
        close = 100 + i * 0.02
        rows.append({
            "timestamp": ts.isoformat(),
            "close_time": (ts + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)).isoformat(),
            "open": close - 0.2, "high": close + 1.0, "low": close - 1.0,
            "close": close, "volume": 10.0,
        })
    decision_ts = (start + pd.Timedelta(hours=1000)).isoformat()
    features = research._derive_time_series_features(
        direction="LONG", df_5m=None, df_1h=pd.DataFrame(rows),
        decision_ts=decision_ts, btc_df_5m=None, btc_df_1h=None,
    )
    assert features["1h_history_bars_available"] == 1000
    assert features["4h_complete_bars_available"] == 250
    assert features["htf4h_ema200_status"] == "AVAILABLE"
    assert features["htf4h_ema200_distance_pct"] is not None


def test_pre_entry_market_context_reuses_persisted_5m_history_without_new_market_fetch(tmp_path, monkeypatch):
    path = tmp_path / "market_bars_5m.jsonl"
    start = datetime(2026, 11, 7, 0, 0, tzinfo=timezone.utc)
    persisted = [
        {**_bar(start + timedelta(minutes=5 * i), 100 + i * 0.01),
         "symbol": "BTC-USDT", "timeframe": "5m", "provider": "binance"}
        for i in range(300)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in persisted), encoding="utf-8")
    monkeypatch.setattr(research, "MARKET_BARS_5M_PATH", path)
    current = [
        {**_bar(start + timedelta(minutes=5 * i), 100 + i * 0.01),
         "symbol": "BTC-USDT", "timeframe": "5m", "provider": "binance"}
        for i in range(156, 301)
    ]

    merged = _prepare_pre_entry_research_5m_bars("BTC-USDT", "binance", current)
    features = research._derive_time_series_features(
        direction="LONG", df_5m=pd.DataFrame(merged), df_1h=None,
        decision_ts="2026-11-08T01:00:00+00:00", btc_df_5m=None, btc_df_1h=None,
    )

    assert len(merged) == 301
    assert features["5m_return_24h_status"] == "AVAILABLE"
    assert features["session_vwap_coverage"] == "FULL"


def test_merge_preserves_persisted_market_fields_when_current_row_has_nulls():
    ts = datetime(2026, 11, 8, 0, 0, tzinfo=timezone.utc)
    persisted = [{**_bar(ts, 100), "quote_volume": 1000.0, "bar_delta_usdt": 200.0}]
    current = [{"timestamp": ts.isoformat(), "close": 101.0, "quote_volume": None, "bar_delta_usdt": None}]
    merged = research.merge_market_bar_history(persisted, current, timeframe="5m")
    assert len(merged) == 1
    assert merged[0]["close"] == 101.0
    assert merged[0]["quote_volume"] == 1000.0
    assert merged[0]["bar_delta_usdt"] == 200.0


def test_time_based_returns_are_not_calculated_across_missing_5m_candles():
    start = pd.Timestamp("2026-11-08T00:00:00Z")
    bars = [
        _bar((start + pd.Timedelta(minutes=5 * i)).to_pydatetime(), 100 + i * 0.01)
        for i in range(146) if i != 70
    ]
    features = research._derive_time_series_features(
        direction="LONG", df_5m=pd.DataFrame(bars), df_1h=None,
        decision_ts="2026-11-08T12:15:00+00:00", btc_df_5m=None, btc_df_1h=None,
    )
    assert features["5m_return_12h_pct"] is None
    assert features["5m_return_12h_status"] == "GAPPED_HISTORY"


def test_missing_session_history_uses_not_captured_vwap_kind():
    features = research._derive_time_series_features(
        direction="LONG", df_5m=pd.DataFrame(), df_1h=pd.DataFrame(),
        decision_ts="2026-11-08T12:00:00+00:00", btc_df_5m=pd.DataFrame(), btc_df_1h=pd.DataFrame(),
    )
    assert features["session_vwap_kind"] == "NOT_CAPTURED"
    assert features["5m_return_24h_status"] == "NOT_CAPTURED"
    assert features["htf4h_ema200_status"] == "NOT_CAPTURED"
    assert features["btc_ema200_status"] == "NOT_CAPTURED"
    assert features["session_vwap_coverage"] == "NOT_CAPTURED"
    assert features["session_vwap_approx"] is None
