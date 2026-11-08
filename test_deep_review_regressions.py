from __future__ import annotations

import numpy as np
import pandas as pd


def _bars(n: int, interval_ms: int, *, start: int = 1_700_000_000_000, step: float = 1.0) -> pd.DataFrame:
    rows = []
    for i in range(n):
        close = 100.0 + i * step
        ts = start + i * interval_ms
        rows.append({
            "timestamp": ts,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
            "close_time": ts + interval_ms,
        })
    return pd.DataFrame(rows)


def test_trend_filter_rejects_gapped_hourly_and_four_hour_inputs():
    from event_engine.trend_filter import evaluate_trend_filter

    h1 = _bars(450, 3_600_000).drop(index=list(range(100, 125))).reset_index(drop=True)
    h4 = _bars(450, 14_400_000, step=2.0).drop(index=list(range(100, 125))).reset_index(drop=True)
    btc = _bars(10, 3_600_000)
    decision = int(max(h1.close_time.max(), h4.close_time.max()))
    out = evaluate_trend_filter(
        symbol="TESTUSDT", direction="LONG", event_type="ZONE_5M",
        df_1h=h1, df_4h=h4, btc_1h_df=btc, decision_ts_ms=decision,
        min_bars_1h=400, min_bars_4h=400,
    )
    assert out["trend_decision"] == "REJECT"
    assert out["trend_reject_reason"] in {"TREND_1H_HISTORY_GAP", "TREND_4H_HISTORY_GAP"}


def test_btc_context_is_unknown_when_hourly_history_has_gap():
    from event_engine.trend_filter import _classify_btc

    btc = _bars(10, 3_600_000).drop(index=5).reset_index(drop=True)
    out = _classify_btc(btc, int(btc.close_time.max()))
    assert out["state"] == "UNKNOWN"
    assert out.get("history_status") == "GAPPED"


def test_live_alma_does_not_compress_missing_complete_eight_hour_bucket():
    from event_engine.signals import compute_ajay_trigger

    bars = _bars(40, 3_600_000, start=1_780_000_000_000)
    # Remove the entire 16:00-23:00 UTC bucket. The 24:00 bucket must not
    # calculate its historical 8H ALMA against 08:00 as though adjacent.
    bars = bars.drop(index=list(range(16, 24))).reset_index(drop=True)
    out = compute_ajay_trigger(bars, mode="live")
    bucket_24 = pd.Timestamp(1_780_000_000_000, unit="ms", tz="UTC").floor("8h") + pd.Timedelta(hours=24)
    rows = out[out["timestamp"].dt.floor("8h") == bucket_24]
    assert rows["alma_close_alt"].isna().all()
    assert rows["alma_open_alt"].isna().all()


def test_price_position_reports_overlapping_demand_and_supply_as_ambiguous():
    from run_once import _price_position

    demand = [{"btm": 99.0, "top": 101.0}]
    supply = [{"btm": 100.0, "top": 102.0}]
    value = _price_position(100.5, demand, supply)
    assert "ПЕРЕКРЫТИЕ" in value.upper()
    assert "не сигнал" in value.lower()


def test_scan_symbol_timing_outcome_is_not_zone_position(monkeypatch):
    import event_engine.request_metrics as metrics

    monkeypatch.setattr(metrics, "_ENABLED", True)
    metrics.reset()
    wrapped = metrics.instrument_scan_symbol(lambda _symbol: {"price_position": "🟢 Внутри 1h DEMAND (не сигнал)"})
    wrapped("TURBOUSDT")
    out = metrics.snapshot()["operations"]["scan_symbol"]["items"][0]["outcome"]
    assert out == "ok"


def test_research_features_infer_decision_boundary_from_bar_when_argument_is_omitted():
    import pytest
    from event_engine.research import build_research_features

    ts = pd.date_range("2026-01-01T00:00:00Z", periods=30, freq="5min")
    closes = [100.0 + i for i in range(30)]
    closes[11:] = [10_000.0 + i for i in range(19)]
    df5 = pd.DataFrame({
        "timestamp": ts,
        "close_time": ts + pd.Timedelta(minutes=5),
        "open": closes,
        "high": [x + 1 for x in closes],
        "low": [x - 1 for x in closes],
        "close": closes,
        "volume": [100.0] * len(ts),
    })
    trigger_index = 10
    boundary = (ts[trigger_index] + pd.Timedelta(minutes=5)).isoformat()
    bar = {
        "timestamp": ts[trigger_index].isoformat(),
        "decision_ts": boundary,
        "open": closes[trigger_index], "high": closes[trigger_index] + 1,
        "low": closes[trigger_index] - 1, "close": closes[trigger_index],
        "volume": 100.0,
    }
    zone = {"zone_id": "Z", "top": 111.0, "btm": 99.0, "poi": 105.0}
    features = build_research_features(symbol="TEST-USDT", direction="LONG", zone=zone, bar=bar, df_5m=df5)

    assert features["feature_availability_ts"] == boundary
    assert features["5m_history_end_ts"] == ts[trigger_index].isoformat()
    assert features["5m_return_5m_pct"] == pytest.approx((closes[10] / closes[9] - 1) * 100)


def test_research_features_do_not_use_unbounded_history_without_any_event_timestamp():
    from event_engine.research import build_research_features

    ts = pd.date_range("2026-01-01T00:00:00Z", periods=30, freq="5min")
    df5 = pd.DataFrame({
        "timestamp": ts,
        "close_time": ts + pd.Timedelta(minutes=5),
        "open": [100.0] * len(ts),
        "high": [101.0] * len(ts),
        "low": [99.0] * len(ts),
        "close": [100.0] * len(ts),
        "volume": [100.0] * len(ts),
    })
    features = build_research_features(
        symbol="TEST-USDT", direction="LONG",
        zone={"zone_id": "Z", "top": 101.0, "btm": 99.0, "poi": 100.0},
        bar={"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 10.0},
        df_5m=df5,
    )
    assert features["feature_availability_ts"] is None
    assert features["5m_history_bars_available"] == 0
    assert features["5m_return_5m_pct"] is None
