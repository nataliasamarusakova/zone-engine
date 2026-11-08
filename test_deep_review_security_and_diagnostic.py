from __future__ import annotations

import requests


def test_telegram_transport_error_does_not_log_bot_token_or_chat_id(monkeypatch, capsys):
    from event_engine import telegram

    token = "123456:SUPER_SECRET_TOKEN"
    chat_id = "987654321"
    monkeypatch.setenv("TG_BOT_TOKEN", token)
    monkeypatch.setenv("TG_CHAT_IDS", chat_id)

    def fail_post(*args, **kwargs):
        raise requests.ConnectionError(
            f"failed request to https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}"
        )

    monkeypatch.setattr(telegram.requests, "post", fail_post)
    assert telegram.send("test") is False
    output = capsys.readouterr().out
    assert token not in output
    assert chat_id not in output


def test_pine_diagnostic_format_price_rounds_to_tick_multiple():
    from pine_sr_diagnostic import format_price

    assert format_price(1.03, 0.05) == "1.05"
    assert format_price(1.12, 0.25) == "1.00"


def test_forward_gap_counter_includes_missing_trailing_bar_before_horizon():
    import pandas as pd

    from research_forward import _BarIndex, _path_gap_count_indexed

    observation = pd.Timestamp("2026-11-01T00:00:00Z")
    target = observation + pd.Timedelta(hours=24)
    # Complete data exists after the target, but the final expected candle
    # immediately before the horizon is missing.
    opens = list(pd.date_range(observation, target - pd.Timedelta(minutes=10), freq="5min"))
    opens += [target + pd.Timedelta(minutes=5)]
    bars = pd.DataFrame({
        "timestamp": opens,
        "close_time": [ts + pd.Timedelta(minutes=5) for ts in opens],
        "open": [1.0] * len(opens),
        "high": [1.1] * len(opens),
        "low": [0.9] * len(opens),
        "close": [1.0] * len(opens),
    })
    index = _BarIndex(bars)
    gaps = _path_gap_count_indexed(
        index, int(observation.value), int(target.value), "CLOSED_5M_EVENT_CLOSE"
    )
    assert gaps == 1


def _research_bar(timestamp, close=100.0):
    import pandas as pd

    ts = pd.Timestamp(timestamp)
    return {
        "timestamp": ts.isoformat(),
        "close_time": (ts + pd.Timedelta(minutes=5)).isoformat(),
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 100.0,
        "bar_delta_usdt": 1.0,
    }


def test_research_5m_ema_atr_and_structure_are_not_computed_across_recent_gaps():
    import pandas as pd
    from event_engine import research

    start = pd.Timestamp("2026-11-01T00:00:00Z")
    rows = [
        _research_bar(start + pd.Timedelta(minutes=5 * i), 100.0 + i * 0.1)
        for i in range(80) if i != 70
    ]
    features = research._derive_time_series_features(
        direction="LONG", df_5m=pd.DataFrame(rows), df_1h=None,
        decision_ts=(start + pd.Timedelta(minutes=5 * 80)).isoformat(),
        btc_df_5m=None, btc_df_1h=None,
    )
    assert features["5m_ema20_distance_pct"] is None
    assert features["5m_ema50_distance_pct"] is None
    assert features["5m_atr14"] is None
    assert features["structure_high_20"] is None
    assert features["5m_ema20_status"] == "GAPPED_HISTORY"


def test_research_4h_ema200_requires_200_contiguous_complete_buckets():
    import pandas as pd
    from event_engine import research

    start = pd.Timestamp("2026-10-01T00:00:00Z")
    rows = []
    for i in range(1000):
        if i == 920:
            continue
        ts = start + pd.Timedelta(hours=i)
        close = 100.0 + i * 0.02
        rows.append({
            "timestamp": ts.isoformat(),
            "close_time": (ts + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)).isoformat(),
            "open": close - 0.1, "high": close + 1, "low": close - 1,
            "close": close, "volume": 100.0,
        })
    decision = start + pd.Timedelta(hours=1000)
    features = research._derive_time_series_features(
        direction="LONG", df_5m=None, df_1h=pd.DataFrame(rows),
        decision_ts=decision.isoformat(), btc_df_5m=None, btc_df_1h=None,
    )
    assert features["htf4h_ema200_status"] == "GAPPED_OR_INCOMPLETE_HISTORY"
    assert features["htf4h_ema200_distance_pct"] is None


def test_btc_ema200_status_requires_contiguous_hourly_history():
    import pandas as pd
    from event_engine import research

    start = pd.Timestamp("2026-10-01T00:00:00Z")
    rows = []
    for i in range(240):
        if i == 220:
            continue
        ts = start + pd.Timedelta(hours=i)
        close = 100.0 + i * 0.05
        rows.append({
            "timestamp": ts.isoformat(),
            "close_time": (ts + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)).isoformat(),
            "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": 100.0,
        })
    decision = start + pd.Timedelta(hours=240)
    features = research._derive_time_series_features(
        direction="LONG", df_5m=None, df_1h=None,
        decision_ts=decision.isoformat(), btc_df_5m=None, btc_df_1h=pd.DataFrame(rows),
    )
    assert features["btc_ema200_status"] == "GAPPED_HISTORY"
    assert features["btc_ema200_distance_pct"] is None


def test_research_volume_ratio_requires_20_consecutive_prior_5m_bars():
    import pandas as pd
    from event_engine import research

    start = pd.Timestamp("2026-11-01T00:00:00Z")
    rows = [
        _research_bar(start + pd.Timedelta(minutes=5 * i), 100.0)
        for i in range(25) if i != 12
    ]
    trigger_ts = start + pd.Timedelta(minutes=125)
    features = research.build_research_features(
        symbol="TEST-USDT", direction="LONG", zone={"top": 101.0, "btm": 99.0, "poi": 100.0},
        bar={"timestamp": trigger_ts.isoformat(), "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 200.0},
        df_5m=pd.DataFrame(rows), decision_ts=(trigger_ts + pd.Timedelta(minutes=5)).isoformat(),
    )
    assert features["volume_ratio_5m20"] is None
    assert features["volume_ratio_5m20_status"] == "GAPPED_HISTORY"


def test_24h_forward_outcome_stays_pending_when_trailing_bar_is_missing():
    import pandas as pd
    from research_forward import _BarIndex, _calculate_forward_outcome_indexed

    observation = pd.Timestamp("2026-11-01T00:00:00Z")
    target = observation + pd.Timedelta(hours=24)
    opens = list(pd.date_range(observation, target - pd.Timedelta(minutes=10), freq="5min"))
    opens += [target + pd.Timedelta(minutes=5)]
    bars = pd.DataFrame({
        "timestamp": opens,
        "close_time": [ts + pd.Timedelta(minutes=5) for ts in opens],
        "open": [100.0] * len(opens),
        "high": [101.0] * len(opens),
        "low": [99.0] * len(opens),
        "close": [100.0] * len(opens),
    })
    observation_row = {
        "observation_id": "OBS_GAP_TRAILING",
        "event_id": "EV_GAP_TRAILING",
        "scan_id": "SCAN_GAP_TRAILING",
        "event_type": "TOUCH_REJECTED",
        "symbol": "TEST-USDT",
        "direction": "LONG",
        "observation_ts": observation.isoformat(),
        "source_event_ts": observation.isoformat(),
        "decision_boundary_kind": "CLOSED_5M_EVENT_CLOSE",
        "reference_price": 100.0,
    }
    outcome = _calculate_forward_outcome_indexed(
        observation_row, _BarIndex(bars), require_complete_24h=True
    )
    assert outcome is None


def _hourly_research_bar(timestamp, close):
    import pandas as pd
    ts = pd.Timestamp(timestamp)
    return {
        "timestamp": ts.isoformat(),
        "close_time": (ts + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)).isoformat(),
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 100.0,
    }


def test_build_research_features_does_not_compress_gaps_for_1h_returns_or_ema():
    import pandas as pd
    from event_engine import research

    start = pd.Timestamp("2026-10-01T00:00:00Z")
    rows = [
        _hourly_research_bar(start + pd.Timedelta(hours=i), 100.0 + i * 0.1)
        for i in range(230) if i != 228
    ]
    trigger = start + pd.Timedelta(hours=230)
    features = research.build_research_features(
        symbol="TEST-USDT", direction="LONG",
        zone={"top": 102.0, "btm": 100.0, "poi": 101.0},
        bar={"timestamp": trigger.isoformat(), "open": 123.0, "high": 124.0, "low": 122.0, "close": 123.5, "volume": 10.0},
        df_1h=pd.DataFrame(rows), decision_ts=trigger.isoformat(),
    )
    assert features.get("return_1h_pct") is None
    assert features.get("return_3h_pct") is None
    assert features.get("ema50_distance_pct") is None
    assert features.get("ema200_distance_pct") is None
    assert features.get("ema50_history_status") == "GAPPED_HISTORY"
    assert features.get("ema200_history_status") == "GAPPED_HISTORY"


def test_approach_returns_and_streak_do_not_cross_missing_5m_candles():
    import pandas as pd
    from event_engine import research

    start = pd.Timestamp("2026-10-01T00:00:00Z")
    rows = [
        _research_bar(start + pd.Timedelta(minutes=5 * i), 100.0 + i)
        for i in range(30) if i != 27
    ]
    trigger = start + pd.Timedelta(minutes=150)
    features = research.build_research_features(
        symbol="TEST-USDT", direction="LONG",
        zone={"top": 130.0, "btm": 120.0, "poi": 125.0},
        bar={"timestamp": trigger.isoformat(), "open": 130.0, "high": 131.0, "low": 129.0, "close": 130.5, "volume": 10.0},
        df_5m=pd.DataFrame(rows), decision_ts=(trigger + pd.Timedelta(minutes=5)).isoformat(),
    )
    assert features.get("approach_return_15m_pct") is None
    assert features.get("approach_return_30m_pct") is None
    assert features.get("approach_directional_streak") == 1


def test_departure_volume_baseline_is_unavailable_across_hourly_gap():
    import pandas as pd
    from event_engine import research

    start = pd.Timestamp("2026-10-01T00:00:00Z")
    rows = []
    for i in range(80):
        if i == 35:
            continue
        row = _hourly_research_bar(start + pd.Timedelta(hours=i), 100.0 + i * 0.1)
        row["atr50"] = 1.0
        rows.append(row)
    trigger = start + pd.Timedelta(hours=80)
    features = research.build_research_features(
        symbol="TEST-USDT", direction="LONG",
        zone={"top": 103.0, "btm": 101.0, "poi": 102.0, "start": 50, "origin_ts_ms": int((start + pd.Timedelta(hours=50)).timestamp() * 1000)},
        bar={"timestamp": trigger.isoformat(), "open": 110.0, "high": 111.0, "low": 109.0, "close": 110.5, "volume": 10.0},
        df_1h=pd.DataFrame(rows), decision_ts=trigger.isoformat(),
    )
    assert features.get("departure_volume_ratio_20") is None
    assert features.get("departure_volume_ratio_20_status") == "GAPPED_HISTORY"


def test_shadow_entry_snapshot_does_not_use_gapped_5m_volume_or_atr():
    import pandas as pd
    from event_engine import shadow

    bars = shadow_test_bars_5m(35)
    bars = bars.loc[bars["timestamp"] != pd.Timestamp("2026-09-28T09:30:00Z")].reset_index(drop=True)
    snap = shadow.build_entry_snapshot(_shadow_test_signal(), df_5m=bars, df_1h=shadow_test_bars_1h())
    exps = snap["experiments"]
    assert exps["CF_VOLUME_5M"]["ratio"] is None
    assert exps["CF_VOLUME_5M"]["status"] == "GAPPED_HISTORY"
    assert exps["CF_ATR_MIN_5M"]["atr14_5m"] is None
    assert exps["CF_ATR_MIN_5M"]["status"] == "GAPPED_HISTORY"


def test_shadow_entry_snapshot_does_not_calculate_ema200_across_hourly_gap():
    import pandas as pd
    from event_engine import shadow

    bars1h = shadow_test_bars_1h(260)
    gap_ts = pd.Timestamp("2026-09-28T08:00:00Z")
    bars1h = bars1h.loc[bars1h["timestamp"] != gap_ts].reset_index(drop=True)
    snap = shadow.build_entry_snapshot(_shadow_test_signal(), df_5m=shadow_test_bars_5m(35), df_1h=bars1h)
    exp = snap["experiments"]["CF_EMA200_1H"]
    assert exp["ema200_1h"] is None
    assert exp["status"] == "GAPPED_HISTORY"


def test_shadow_entry_snapshot_requires_exact_trigger_bar():
    import pandas as pd
    from event_engine import shadow

    bars5m = shadow_test_bars_5m(35)
    trigger_ts = pd.Timestamp("2026-09-28T10:00:00Z")
    bars5m = bars5m.loc[bars5m["timestamp"] != trigger_ts].reset_index(drop=True)
    snap = shadow.build_entry_snapshot(_shadow_test_signal(), df_5m=bars5m, df_1h=shadow_test_bars_1h())
    exp = snap["experiments"]["CF_VOLUME_5M"]
    assert exp["trigger_volume"] is None
    assert exp["status"] == "MISSING_TRIGGER_BAR"


def shadow_test_bars_5m(n):
    import pandas as pd
    from event_engine import shadow
    ts = pd.date_range("2026-09-28T07:10:00Z", periods=n, freq="5min")
    return pd.DataFrame({
        "timestamp": ts,
        "close_time": ts + pd.Timedelta(minutes=5),
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0] * n, "volume": [10.0] * n, "bar_delta_usdt": [1.0] * n,
    })


def shadow_test_bars_1h(n=240):
    import pandas as pd
    ts = pd.date_range("2026-09-18T00:00:00Z", periods=n, freq="1h")
    return pd.DataFrame({
        "timestamp": ts,
        "close_time": ts + pd.Timedelta(hours=1),
        "open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
        "close": [100.0] * n, "volume": [1000.0] * n,
    })


def _shadow_test_signal():
    return {
        "event_id": "EV_SHADOW_GAP",
        "symbol": "TEST-USDT",
        "type": "LONG",
        "entry": 100.0,
        "trigger_bar_time": "2026-09-28T10:00:00Z",
        "zone": {"top": 101.0, "btm": 99.0},
        "atr": 1.0,
    }


def test_shadow_trailing_atr_is_unavailable_after_candle_gap():
    import pandas as pd
    from event_engine import shadow

    ts = pd.date_range("2026-10-01T00:00:00Z", periods=40, freq="5min")
    bars = pd.DataFrame({
        "timestamp": ts.delete(28),
        "close_time": ts.delete(28) + pd.Timedelta(minutes=5),
        "open": [100.0] * 39, "high": [101.0] * 39, "low": [99.0] * 39,
        "close": [100.0] * 39,
    })
    prepared = shadow.prepare_counterfactual_bars(bars)
    atr_col = shadow._SHADOW_ATR_BEFORE_CLOSE
    # At the first candle after the missing interval, the preceding ATR path is
    # not a valid consecutive 5m sequence and must not be used to move a stop.
    row_after_gap = prepared.loc[prepared["timestamp"] == pd.Timestamp("2026-10-01T02:25:00Z")]
    assert len(row_after_gap) == 1
    assert pd.isna(row_after_gap.iloc[0][atr_col])


def test_shadow_prepare_counterfactual_bars_deduplicates_overlapping_history_and_refresh():
    import pandas as pd
    from event_engine import shadow

    ts = pd.Timestamp("2026-10-01T00:00:00Z")
    old = {"timestamp": ts, "close_time": ts + pd.Timedelta(minutes=5), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}
    refreshed = {**old, "high": 105.0, "close": 104.0}
    prepared = shadow.prepare_counterfactual_bars(pd.DataFrame([old, refreshed]))
    assert len(prepared) == 1
    assert prepared.iloc[0]["close"] == 104.0
    assert prepared.iloc[0]["high"] == 105.0


def test_departure_impulse_is_unavailable_across_hourly_gap_inside_window():
    import pandas as pd
    from event_engine import research

    start = pd.Timestamp("2026-10-01T00:00:00Z")
    rows = []
    for i in range(90):
        if i == 53:
            continue
        row = _hourly_research_bar(start + pd.Timedelta(hours=i), 100.0 + i * 0.1)
        row["atr50"] = 1.0
        rows.append(row)
    trigger = start + pd.Timedelta(hours=90)
    features = research.build_research_features(
        symbol="TEST-USDT", direction="LONG",
        zone={"top": 103.0, "btm": 101.0, "poi": 102.0, "start": 50},
        bar={"timestamp": trigger.isoformat(), "open": 110.0, "high": 111.0, "low": 109.0, "close": 110.5, "volume": 10.0},
        df_1h=pd.DataFrame(rows), decision_ts=trigger.isoformat(),
    )
    assert features.get("departure_impulse_atr") is None
    assert features.get("departure_impulse_status") == "GAPPED_HISTORY"
