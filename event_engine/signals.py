from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

SWING_LEN = 10
BOX_WIDTH = 2.5
RR_RATIO = 3.0
TP1_R = 1.0
TP2_R = 2.0
TP1_FRACTION = 0.50
TP2_FRACTION = 0.50


def _wma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    weights = np.arange(1, length + 1, dtype=float)
    denom = weights.sum()
    return series.rolling(length, min_periods=length).apply(lambda x: float(np.dot(x, weights) / denom), raw=True)


def calc_hma(series: pd.Series, length: int) -> pd.Series:
    half = max(1, length // 2)
    sqrt_len = max(1, int(round(math.sqrt(length))))
    return _wma(2.0 * _wma(series, half) - _wma(series, length), sqrt_len)


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length, min_periods=length).mean().bfill()


def _make_event_id(symbol: str, direction: str, timestamp: int, zone_start: int, entry: float) -> str:
    raw = f"ZONE:{symbol.upper()}:{direction.upper()}:{timestamp}:{zone_start}:{entry:.12f}"
    return "ZONE_" + hashlib.sha256(raw.encode()).hexdigest().upper()[:24]


def _zone_age(start_idx: int, current_idx: int) -> int:
    return max(0, int(current_idx) - int(start_idx))


def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _zone_context(zone: dict, current_idx: int, df: pd.DataFrame) -> dict[str, Any]:
    start = int(zone["start"])
    move_idx = min(start + 6, current_idx)
    base = _safe_num(df.loc[start, "close"])
    end_close = _safe_num(df.loc[move_idx, "close"])
    atr = max(_safe_num(df.loc[start, "atr"], 0.0), 1e-12)
    impulse_atr = abs(end_close - base) / atr
    age = _zone_age(start, current_idx)
    return {"start_idx": start, "age_bars": age, "impulse_atr": round(impulse_atr, 3)}


def generate_zone_signals(df: pd.DataFrame, symbol: str = "") -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    df = df.copy().reset_index(drop=True)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    timestamp_series = df["timestamp"]
    if pd.api.types.is_numeric_dtype(timestamp_series):
        df["timestamp"] = pd.to_datetime(timestamp_series, unit="ms", utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(timestamp_series, utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
    n = len(df)
    if n < SWING_LEN * 2 + 5:
        return df, [], [], []

    df["atr"] = _atr(df, 14)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr50"] = tr.rolling(50, min_periods=1).mean().bfill()
    df["fast_ma"] = calc_hma(df["close"], 5)
    df["slow_ma"] = calc_hma(df["close"], 13)
    df["vol_sma20"] = df["volume"].rolling(20, min_periods=5).mean().bfill()
    df["body_atr"] = (df["close"] - df["open"]).abs() / df["atr"].replace(0, np.nan)
    df["range_atr"] = (df["high"] - df["low"]) / df["atr"].replace(0, np.nan)

    active_supply: list[dict] = []
    active_demand: list[dict] = []
    signals: list[dict] = []

    for i in range(SWING_LEN * 2, n):
        p_idx = i - SWING_LEN
        h = float(df.loc[p_idx, "high"])
        l = float(df.loc[p_idx, "low"])
        buf = max(float(df.loc[p_idx, "atr50"]) * (BOX_WIDTH / 10.0), 1e-12)

        if all(h >= float(df.loc[p_idx - k, "high"]) for k in range(1, SWING_LEN + 1)) and all(
            h >= float(df.loc[p_idx + k, "high"]) for k in range(1, SWING_LEN + 1)
        ):
            active_supply.append({"top": h, "btm": h - buf, "poi": h - buf / 2.0, "start": p_idx})

        if all(l <= float(df.loc[p_idx - k, "low"]) for k in range(1, SWING_LEN + 1)) and all(
            l <= float(df.loc[p_idx + k, "low"]) for k in range(1, SWING_LEN + 1)
        ):
            active_demand.append({"top": l + buf, "btm": l, "poi": l + buf / 2.0, "start": p_idx})

        current_close = float(df.loc[i, "close"])
        active_supply = [s for s in active_supply[-4:] if current_close <= float(s["top"])]
        active_demand = [d for d in active_demand[-4:] if current_close >= float(d["btm"])]

        cur_l = float(df.loc[i, "low"])
        cur_h = float(df.loc[i, "high"])
        cur_c = current_close
        cur_o = float(df.loc[i, "open"])
        c_atr = max(float(df.loc[i, "atr"]), 1e-12)

        fast = df.loc[i, "fast_ma"]
        slow = df.loc[i, "slow_ma"]
        fast_prev = df.loc[i - 1, "fast_ma"]
        slow_prev = df.loc[i - 1, "slow_ma"]
        bull_cross = bool(pd.notna(fast) and pd.notna(slow) and pd.notna(fast_prev) and pd.notna(slow_prev) and fast > slow and fast_prev <= slow_prev)
        bear_cross = bool(pd.notna(fast) and pd.notna(slow) and pd.notna(fast_prev) and pd.notna(slow_prev) and fast < slow and fast_prev >= slow_prev)

        near_demand = [d for d in active_demand if cur_l <= float(d["top"]) * 1.003 and cur_c >= float(d["btm"])]
        if near_demand and bull_cross:
            zone = near_demand[-1]
            sl = round(float(zone["btm"]) - 0.3 * c_atr, 8)
            risk = max(cur_c - sl, 1e-12)
            tp1 = round(cur_c + TP1_R * risk, 8)
            tp2 = round(cur_c + TP2_R * risk, 8)
            event_id = _make_event_id(symbol or "UNKNOWN", "LONG", int(df.loc[i, "timestamp"].timestamp() * 1000), int(zone["start"]), cur_c)
            vol_ratio = cur_vol = None
            avg_vol = _safe_num(df.loc[i, "vol_sma20"], 0.0)
            if avg_vol > 0:
                cur_vol = _safe_num(df.loc[i, "volume"])
                vol_ratio = cur_vol / avg_vol
            signals.append(
                {
                    "event_id": event_id,
                    "idx": i,
                    "time": df.loc[i, "timestamp"].isoformat(),
                    "type": "LONG",
                    "symbol": symbol.upper(),
                    "entry": cur_c,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "risk_pct": round((risk / cur_c) * 100.0, 4),
                    "risk_abs": risk,
                    "rr_ratio": RR_RATIO,
                    "zone": {**zone, **_zone_context(zone, i, df), "kind": "DEMAND"},
                    "confirmation": {
                        "hma_cross": True,
                        "volume_ratio": round(vol_ratio, 3) if vol_ratio is not None else None,
                        "candle_body_atr": round(_safe_num(df.loc[i, "body_atr"]), 3),
                        "range_atr": round(_safe_num(df.loc[i, "range_atr"]), 3),
                        "bullish_candle": cur_c >= cur_o,
                    },
                    "source_bar_close": cur_c,
                }
            )

        near_supply = [s for s in active_supply if cur_h >= float(s["btm"]) * 0.997 and cur_c <= float(s["top"])]
        if near_supply and bear_cross:
            zone = near_supply[-1]
            sl = round(float(zone["top"]) + 0.3 * c_atr, 8)
            risk = max(sl - cur_c, 1e-12)
            tp1 = round(cur_c - TP1_R * risk, 8)
            tp2 = round(cur_c - TP2_R * risk, 8)
            event_id = _make_event_id(symbol or "UNKNOWN", "SHORT", int(df.loc[i, "timestamp"].timestamp() * 1000), int(zone["start"]), cur_c)
            vol_ratio = None
            avg_vol = _safe_num(df.loc[i, "vol_sma20"], 0.0)
            if avg_vol > 0:
                vol_ratio = _safe_num(df.loc[i, "volume"]) / avg_vol
            signals.append(
                {
                    "event_id": event_id,
                    "idx": i,
                    "time": df.loc[i, "timestamp"].isoformat(),
                    "type": "SHORT",
                    "symbol": symbol.upper(),
                    "entry": cur_c,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "risk_pct": round((risk / cur_c) * 100.0, 4),
                    "risk_abs": risk,
                    "rr_ratio": RR_RATIO,
                    "zone": {**zone, **_zone_context(zone, i, df), "kind": "SUPPLY"},
                    "confirmation": {
                        "hma_cross": True,
                        "volume_ratio": round(vol_ratio, 3) if vol_ratio is not None else None,
                        "candle_body_atr": round(_safe_num(df.loc[i, "body_atr"]), 3),
                        "range_atr": round(_safe_num(df.loc[i, "range_atr"]), 3),
                        "bearish_candle": cur_c <= cur_o,
                    },
                    "source_bar_close": cur_c,
                }
            )

    return df, active_supply, active_demand, signals


def score_zone_signal(signal: dict[str, Any]) -> float:
    confirmation = signal.get("confirmation", {})
    zone = signal.get("zone", {})
    score = 60.0
    if confirmation.get("hma_cross"):
        score += 10.0
    vol_ratio = _safe_num(confirmation.get("volume_ratio"), 0.0)
    if vol_ratio >= 1.5:
        score += 10.0
    elif vol_ratio >= 1.2:
        score += 5.0
    body_atr = _safe_num(confirmation.get("candle_body_atr"), 0.0)
    if body_atr >= 0.8:
        score += 5.0
    age = int(zone.get("age_bars", 9999) or 9999)
    if age <= 36:
        score += 5.0
    if _safe_num(zone.get("impulse_atr"), 0.0) >= 1.0:
        score += 10.0
    return min(score, 100.0)
