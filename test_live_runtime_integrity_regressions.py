import pandas as pd
import pytest

from event_engine import shadow


def _prior_5m_bars(trigger_ts: pd.Timestamp, count: int = 24) -> pd.DataFrame:
    opens = pd.date_range(
        end=trigger_ts - pd.Timedelta(minutes=5),
        periods=count,
        freq="5min",
        tz="UTC",
    )
    rows = []
    for index, stamp in enumerate(opens):
        close = 100.0 + index * 0.01
        rows.append({
            "timestamp": stamp,
            "close_time": stamp + pd.Timedelta(minutes=5),
            "open": close - 0.01,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": 100.0,
            "bar_delta_usdt": 5.0,
        })
    return pd.DataFrame(rows)


def _live_signal(trigger_ts: pd.Timestamp) -> dict:
    return {
        "event_id": "EVT_LIVE_TRIGGER",
        "symbol": "TEST-USDT",
        "type": "LONG",
        "entry": 100.28,
        "trigger_bar_time": trigger_ts.isoformat(),
        "decision_boundary_ts": (trigger_ts + pd.Timedelta(minutes=5)).isoformat(),
        "zone": {"top": 100.5, "btm": 99.5},
        "entry_bar": {
            "timestamp": trigger_ts.isoformat(),
            "open": 100.24,
            "high": 100.30,
            "low": 100.20,
            "close": 100.28,
            "volume": 300.0,
            "quote_volume": 30_000.0,
            "taker_buy_base": 200.0,
            "taker_buy_quote": 20_000.0,
            "taker_flow_valid": True,
            "trade_count": 120,
            "bar_delta_usdt": 10_000.0,
        },
    }


def test_shadow_uses_canonical_signal_entry_bar_when_frame_omits_trigger():
    trigger_ts = pd.Timestamp("2026-11-08T15:25:00Z")
    signal = _live_signal(trigger_ts)
    snap = shadow.build_entry_snapshot(
        signal,
        df_5m=_prior_5m_bars(trigger_ts),
        df_1h=None,
        decision_ts=signal["decision_boundary_ts"],
    )
    experiments = snap["experiments"]

    volume = experiments["CF_VOLUME_5M"]
    assert volume["status"] == "AVAILABLE"
    assert volume["trigger_volume"] == pytest.approx(300.0)
    assert volume["baseline_sma20_previous_bars"] == pytest.approx(100.0)
    assert volume["ratio"] == pytest.approx(3.0)

    atr = experiments["CF_ATR_MIN_5M"]
    assert atr["status"] == "AVAILABLE"
    assert atr["atr14_5m"] is not None

    delta = experiments["CF_DELTA_CONFIRM"]
    assert delta["bar_delta_usdt"] == pytest.approx(10_000.0)
    assert delta["would_pass"] is True


def test_shadow_does_not_use_mismatched_canonical_entry_bar_timestamp():
    trigger_ts = pd.Timestamp("2026-11-08T15:25:00Z")
    signal = _live_signal(trigger_ts)
    signal["entry_bar"]["timestamp"] = (trigger_ts - pd.Timedelta(minutes=5)).isoformat()
    snap = shadow.build_entry_snapshot(
        signal,
        df_5m=_prior_5m_bars(trigger_ts),
        df_1h=None,
        decision_ts=signal["decision_boundary_ts"],
    )
    assert snap["experiments"]["CF_VOLUME_5M"]["status"] == "MISSING_TRIGGER_BAR"
    assert snap["experiments"]["CF_ATR_MIN_5M"]["status"] == "MISSING_TRIGGER_BAR"


def test_5m_signal_preserves_exchange_flow_fields_for_shadow(monkeypatch):
    import run_once

    trigger_ts = pd.Timestamp("2026-11-08T17:25:00Z")
    hourly_opens = pd.date_range(end=trigger_ts, periods=24, freq="1h", tz="UTC")
    df_1h = pd.DataFrame({
        "timestamp": hourly_opens,
        "close_time": hourly_opens + pd.Timedelta(hours=1),
        "open": [100.0] * 24,
        "high": [101.0] * 24,
        "low": [99.0] * 24,
        "close": [100.0] * 24,
        "volume": [1_000.0] * 24,
        "atr50": [2.0] * 24,
    })
    bar = pd.Series({
        "timestamp": trigger_ts,
        "close_time": int((trigger_ts + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1)).timestamp() * 1000),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 300.0,
        "quote_volume": 30_000.0,
        "taker_buy_base": 200.0,
        "taker_buy_quote": 20_000.0,
        "taker_flow_valid": True,
        "trade_count": 120,
        "bar_delta_usdt": 10_000.0,
    })
    zone = {"zone_id": "Z_LIVE_BAR", "top": 101.0, "btm": 99.0, "poi": 100.0, "start": 0}

    monkeypatch.setattr(run_once, "ZONE_TRIGGER_MODE", "zone")
    monkeypatch.setattr(run_once, "REQUIRE_DIRECTIONAL_CANDLE", False)
    monkeypatch.setattr(run_once, "REQUIRE_STRUCTURE_OBSTACLE", False)
    monkeypatch.setattr(run_once, "_nearest_opposing_level", lambda *args, **kwargs: None)

    signal = run_once._build_5m_zone_signal(
        "TEST-USDT", "LONG", zone, bar, None, df_1h, [zone], [], {"visit_id": "V1"},
    )
    for field in (
        "quote_volume", "taker_buy_base", "taker_buy_quote", "taker_flow_valid",
        "trade_count", "bar_delta_usdt",
    ):
        assert signal["entry_bar"][field] == bar[field]


def test_shadow_enriches_existing_trigger_row_with_canonical_flow_metadata():
    trigger_ts = pd.Timestamp("2026-11-08T15:25:00Z")
    signal = _live_signal(trigger_ts)
    bars = _prior_5m_bars(trigger_ts)
    trigger_row = signal["entry_bar"].copy()
    trigger_row.pop("bar_delta_usdt")
    trigger_row["timestamp"] = pd.Timestamp(trigger_row["timestamp"])
    trigger_row["close_time"] = trigger_ts + pd.Timedelta(minutes=5)
    bars = pd.concat([bars, pd.DataFrame([trigger_row])], ignore_index=True)

    snap = shadow.build_entry_snapshot(
        signal, df_5m=bars, df_1h=None, decision_ts=signal["decision_boundary_ts"],
    )
    delta = snap["experiments"]["CF_DELTA_CONFIRM"]
    assert delta["bar_delta_usdt"] == pytest.approx(10_000.0)
    assert delta["would_pass"] is True


def test_shadow_does_not_accept_delta_when_signal_marks_flow_invalid():
    trigger_ts = pd.Timestamp("2026-11-08T15:25:00Z")
    signal = _live_signal(trigger_ts)
    signal["entry_bar"]["taker_flow_valid"] = False
    snap = shadow.build_entry_snapshot(
        signal, df_5m=_prior_5m_bars(trigger_ts), df_1h=None,
        decision_ts=signal["decision_boundary_ts"],
    )
    delta = snap["experiments"]["CF_DELTA_CONFIRM"]
    assert delta["bar_delta_usdt"] is None
    assert delta["would_pass"] is None


def test_utc_session_coverage_counts_last_closed_exchange_candle_at_999ms_boundary():
    from event_engine import research

    start = pd.Timestamp("2026-11-08T00:00:00Z")
    stamps = pd.date_range(start=start, periods=190, freq="5min")
    bars = pd.DataFrame({
        "timestamp": stamps,
        "close_time": stamps + pd.Timedelta(minutes=5) - pd.Timedelta(milliseconds=1),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 10.0,
        "bar_delta_usdt": 2.0,
    })
    decision_ts = "2026-11-08T15:49:59.999Z"

    complete = research._derive_time_series_features(
        direction="LONG", df_5m=bars, df_1h=None,
        decision_ts=decision_ts, btc_df_5m=None, btc_df_1h=None,
    )
    assert complete["session_vwap_expected_bars"] == 190
    assert complete["session_vwap_available_bars"] == 190
    assert complete["session_vwap_coverage"] == "FULL"
    assert complete["session_cvd_expected_bars"] == 190
    assert complete["session_cvd_coverage"] == "FULL"

    missing_last = research._derive_time_series_features(
        direction="LONG", df_5m=bars.iloc[:-1].copy(), df_1h=None,
        decision_ts=decision_ts, btc_df_5m=None, btc_df_1h=None,
    )
    assert missing_last["session_vwap_expected_bars"] == 190
    assert missing_last["session_vwap_available_bars"] == 189
    assert missing_last["session_vwap_coverage"] == "PARTIAL"
    assert missing_last["session_cvd_expected_bars"] == 190
    assert missing_last["session_cvd_coverage"] == "PARTIAL"


def test_shadow_removes_existing_bar_delta_when_canonical_trigger_marks_flow_invalid():
    trigger_ts = pd.Timestamp("2026-11-08T15:25:00Z")
    signal = _live_signal(trigger_ts)
    signal["entry_bar"]["taker_flow_valid"] = False
    bars = _prior_5m_bars(trigger_ts)
    trigger_row = signal["entry_bar"].copy()
    trigger_row["timestamp"] = trigger_ts
    trigger_row["close_time"] = trigger_ts + pd.Timedelta(minutes=5)
    trigger_row["bar_delta_usdt"] = 5.0  # stale/invalid exchange diagnostic
    bars = pd.concat([bars, pd.DataFrame([trigger_row])], ignore_index=True)

    snap = shadow.build_entry_snapshot(
        signal, df_5m=bars, df_1h=None, decision_ts=signal["decision_boundary_ts"],
    )
    delta = snap["experiments"]["CF_DELTA_CONFIRM"]
    assert delta["bar_delta_usdt"] is None
    assert delta["would_pass"] is None
