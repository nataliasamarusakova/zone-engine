from __future__ import annotations

import pandas as pd
import pytest

from event_engine.trend_filter import evaluate_trend_filter

BASE_TS = 1_700_000_000_000
H1 = 3_600_000
H4 = 14_400_000


def frame(n: int, step: float, interval_ms: int, *, base: float = 100.0, start_ts: int = BASE_TS) -> pd.DataFrame:
    rows=[]
    for i in range(n):
        close=base+step*i
        rows.append({"open":close,"high":close+0.5,"low":close-0.5,"close":close,"volume":1000.0,"close_time":start_ts+i*interval_ms})
    return pd.DataFrame(rows)


def result(df1, df4, direction="LONG", decision_ts=None, **kwargs):
    ts=int(decision_ts if decision_ts is not None else max(df1.close_time.iloc[-1],df4.close_time.iloc[-1]))
    return evaluate_trend_filter(symbol="TESTUSDT",direction=direction,event_type="ZONE_5M",df_1h=df1,df_4h=df4,btc_1h_df=frame(10,1.0,H1),decision_ts_ms=ts,min_bars_1h=400,min_bars_4h=400,**kwargs)


def test_shadow_alignment_is_causal():
    out=result(frame(450,1.0,H1),frame(450,2.0,H4),"LONG")
    assert out["trend_decision"]=="ALIGNED"
    assert out["trend_4h"]=="BULL"
    assert out["trend_1h"]=="LONG"
    assert out["trend_persistence"]=="PERSISTENT"
    assert out["trend_1h_bar_close_ts"]<=out["decision_ts"]
    assert out["trend_4h_bar_close_ts"]<=out["decision_ts"]


def test_direction_mismatch_rejects():
    out=result(frame(450,1.0,H1),frame(450,2.0,H4),"SHORT")
    assert out["trend_decision"]=="REJECT"
    assert out["trend_reject_reason"]=="TREND_4H_DIRECTION_MISMATCH"


def test_unknown_history_is_explicit():
    out=result(frame(300,1.0,H1),frame(300,2.0,H4),"LONG")
    assert out["trend_decision"]=="REJECT"
    assert out["trend_reject_reason"]=="TREND_4H_UNKNOWN"


def test_future_bars_do_not_change_decision():
    df1=frame(450,1.0,H1); df4=frame(450,2.0,H4)
    decision=max(df1.close_time.iloc[-1],df4.close_time.iloc[-1])
    future1=frame(40,-8.0,H1,base=float(df1.close.iloc[-1]),start_ts=int(decision+H1))
    future4=frame(40,-16.0,H4,base=float(df4.close.iloc[-1]),start_ts=int(decision+H4))
    out=result(pd.concat([df1,future1]),pd.concat([df4,future4]),"LONG",decision_ts=decision)
    assert out["trend_decision"]=="ALIGNED"
    assert out["trend_1h_bar_close_ts"]==int(df1.close_time.iloc[-1])
    assert out["trend_4h_bar_close_ts"]==int(df4.close_time.iloc[-1])


def test_optional_persistence_is_separate_from_direction():
    df4=frame(450,2.0,H4); df1=frame(450,1.0,H1)
    last=len(df1)-1
    for idx,val in zip(range(last-5,last+1),[550,545,540,537,535,533]): df1.loc[idx,"close"]=val
    df1["open"]=df1["close"];df1["high"]=df1["close"]+0.5;df1["low"]=df1["close"]-0.5
    out=result(df1,df4,"LONG",require_persistence=True)
    assert out["trend_4h"]=="BULL"
    assert out["trend_reject_reason"] in {"TREND_PERSISTENCE_TRANSITION","TREND_1H_TRANSITION","TREND_1H_DIRECTION_MISMATCH"}


def test_btc_is_context_only():
    df1=frame(450,1.0,H1);df4=frame(450,2.0,H4)
    out=evaluate_trend_filter(symbol="TESTUSDT",direction="LONG",event_type="ZONE_5M",df_1h=df1,df_4h=df4,btc_1h_df=frame(10,-1.0,H1),decision_ts_ms=int(df4.close_time.iloc[-1]),min_bars_1h=400,min_bars_4h=400)
    assert out["trend_decision"]=="ALIGNED"
    assert out["btc_regime"]=="BEARISH"
    assert out["btc_context_veto"] is False


def test_structure_is_recorded_and_causal():
    df1=frame(450,1.0,H1);df4=frame(450,2.0,H4)
    out=result(df1,df4,"LONG")
    assert "structure_1h" in out and "structure_4h" in out
    assert out["structure_1h"]["last_swing_high_ts"] is None or out["structure_1h"]["last_swing_high_ts"]<=out["decision_ts"]
    assert out["structure_4h"]["last_swing_low_ts"] is None or out["structure_4h"]["last_swing_low_ts"]<=out["decision_ts"]


def test_invalid_direction_is_rejected():
    out=result(frame(450,1.0,H1),frame(450,2.0,H4),"SIDEWAYS")
    assert out["trend_reject_reason"]=="TREND_EVENT_DIRECTION_UNKNOWN"


def test_enforce_helper_is_fail_closed_for_unknown(monkeypatch):
    import run_once
    assert run_once._trend_filter_enforce_reject({"trend_decision":"REJECT"},"shadow") is False
    assert run_once._trend_filter_enforce_reject({"trend_decision":"REJECT"},"enforce") is True
    assert run_once._trend_filter_enforce_reject({"trend_decision":"ALIGNED"},"enforce") is False
    assert run_once._trend_filter_enforce_reject(None,"enforce") is True


def test_structure_timestamps_preserve_epoch_milliseconds():
    df1 = frame(450, 1.0, H1)
    df4 = frame(450, 2.0, H4)
    out = result(df1, df4, "LONG")
    structure = out["structure_1h"]
    confirmation_ts = structure.get("last_swing_high_confirmation_ts")
    pivot_ts = structure.get("last_swing_high_ts")
    if confirmation_ts is not None and pivot_ts is not None:
        assert confirmation_ts > pivot_ts
        # The synthetic bars are one hour apart; a nanosecond interpretation
        # would be far outside the decision-time epoch range.
        assert 1_600_000_000_000 <= pivot_ts < 2_000_000_000_000
        assert 1_600_000_000_000 <= confirmation_ts < 2_000_000_000_000


def test_trend_version_is_current_in_runtime_source():
    from pathlib import Path
    source = Path(__file__).with_name("run_once.py").read_text(encoding="utf-8")
    assert "trend-v1-2026-10-04" not in source
    assert "trend-v1-2026-10-06" in source


def test_trend_filter_accepts_iso_close_time_columns_without_unknowning_every_bar():
    import pandas as pd
    import event_engine.trend_filter as trend_filter
    ts = pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "close_time": (ts + pd.Timedelta(hours=1)).astype(str),
        "close": [100, 101, 102, 103, 104, 105],
        "open": [99, 100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105, 106],
        "low": [98, 99, 100, 101, 102, 103],
    })
    out = trend_filter._clean_closed_frame(df, int(pd.Timestamp("2026-01-01T06:00:00Z").timestamp() * 1000))
    assert len(out) == 6
    assert int(out["close_time"].iloc[-1]) == int(pd.Timestamp("2026-01-01T06:00:00Z").timestamp() * 1000)
