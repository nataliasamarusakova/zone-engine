from __future__ import annotations

import math
import numbers
from datetime import timezone
from typing import Any

import pandas as pd
import numpy as np


SHADOW_SCHEMA_VERSION = 1
SHADOW_EXPERIMENT_VERSION = "counterfactual-v3"

VOLUME_5M_THRESHOLDS = (0.8, 1.0, 1.2, 1.5, 1.8, 2.0)
VOLUME_1H_THRESHOLDS = (0.8, 1.0, 1.2, 1.5, 1.8, 2.0)
ATR_5M_MIN_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)
TRAIL_ATR_MULTIPLIERS = (0.75, 1.0, 1.5)
DELAYED_BE_BARS = (1, 2)
MFE_PROTECTION_ACTIVATION_PCT = 2.0
MFE_PROTECTION_LOCK_LEVELS_PCT = (-1.0, 0.0, 0.5)
PRODUCTION_TP1_FRACTION = 0.50
PRODUCTION_TP2_FRACTION = 0.25
PRODUCTION_TP3_FRACTION = 0.25
PRODUCTION_STOP_PCT = 10.0
PRODUCTION_TP1_PCT = 3.0
PRODUCTION_TP2_PCT = 6.0
PRODUCTION_TP3_PCT = 9.0
STRUCTURAL_SL_ATR_MULT = 0.5
STRUCTURAL_SL_MIN_PCT = 1.0
STRUCTURAL_SL_MAX_PCT = 6.0
TP1_R_MULT = 1.5
TP2_R_MULT = 3.0
ROUNDTRIP_SLIPPAGE_SCENARIOS_PCT = (0.0, 0.10, 0.20, 0.50)


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _parse_ts(value: Any) -> pd.Timestamp | None:
    """Parse ISO timestamps and numeric epoch values deterministically.

    Numeric values in this project are epoch milliseconds (for example
    ``origin_ts_ms``), while ISO strings may contain an explicit timezone.
    ``pd.Timestamp(numeric)`` defaults to nanoseconds and silently produces
    dates near 1970, so numeric input must use ``unit="ms"`` explicitly.
    """
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, numbers.Real):
            numeric = float(value)
            if not math.isfinite(numeric):
                return None
            magnitude = abs(numeric)
            unit = (
                "ns" if magnitude >= 1e17 else
                "us" if magnitude >= 1e14 else
                "ms" if magnitude >= 1e11 else
                "s" if magnitude >= 1e8 else None
            )
            if unit is None:
                return None
            ts = pd.to_datetime(value, unit=unit, utc=True)
        elif isinstance(value, str):
            raw = value.strip()
            numeric = pd.to_numeric(raw, errors="coerce")
            if pd.notna(numeric):
                numeric = float(numeric)
                magnitude = abs(numeric)
                unit = (
                    "ns" if magnitude >= 1e17 else
                    "us" if magnitude >= 1e14 else
                    "ms" if magnitude >= 1e11 else
                    "s" if magnitude >= 1e8 else None
                )
                if unit is None:
                    return None
                ts = pd.to_datetime(numeric, unit=unit, utc=True)
            else:
                ts = pd.Timestamp(raw)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                else:
                    ts = ts.tz_convert("UTC")
        else:
            ts = pd.Timestamp(value)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return None


def _timestamp_series_to_utc(series: pd.Series) -> pd.Series:
    """Normalize ISO and epoch ns/us/ms/s values without pandas nanosecond inference."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    ratio = float(numeric.notna().mean()) if len(series) else 0.0
    if ratio >= 0.99:
        finite = numeric.dropna().abs()
        magnitude = float(finite.median()) if not finite.empty else 0.0
        unit = (
            "ns" if magnitude >= 1e17 else
            "us" if magnitude >= 1e14 else
            "ms" if magnitude >= 1e11 else
            "s" if magnitude >= 1e8 else None
        )
        if unit is not None:
            return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _closed_5m(df: pd.DataFrame | None, entry_ts: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x["timestamp"] = _timestamp_series_to_utc(x["timestamp"])
    if "close_time" not in x.columns:
        x["close_time"] = x["timestamp"] + pd.Timedelta(minutes=5)
    else:
        x["close_time"] = _timestamp_series_to_utc(x["close_time"])
    for col in ("open", "high", "low", "close", "volume"):
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["timestamp", "close_time", "open", "high", "low", "close", "volume"])
    # The signal is created from a CLOSED trigger candle whose timestamp is its
    # candle-open time. Include that trigger bar by extending the close boundary
    # exactly one 5m interval; do not include any later bar.
    x = x.loc[x["close_time"] <= (entry_ts + pd.Timedelta(minutes=5))].copy()
    return x.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _closed_1h(df: pd.DataFrame | None, entry_ts: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x["timestamp"] = _timestamp_series_to_utc(x["timestamp"])
    if "close_time" not in x.columns:
        x["close_time"] = x["timestamp"] + pd.Timedelta(hours=1)
    else:
        x["close_time"] = _timestamp_series_to_utc(x["close_time"])
    for col in ("open", "high", "low", "close", "volume"):
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["timestamp", "close_time", "open", "high", "low", "close", "volume"])
    x = x.loc[x["close_time"] <= entry_ts].copy()
    return x.sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _has_contiguous_tail(df: pd.DataFrame, count: int, interval: pd.Timedelta) -> bool:
    if df is None or len(df) < count or "timestamp" not in df.columns:
        return False
    timestamps = _timestamp_series_to_utc(df["timestamp"].tail(count))
    if timestamps.isna().any():
        return False
    if len(timestamps) > 1 and not timestamps.diff().dropna().eq(interval).all():
        return False
    interval_ns = int(interval.value)
    return bool((timestamps.astype("int64") % interval_ns == 0).all())


def _contiguous_suffix(df: pd.DataFrame, interval: pd.Timedelta) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=df.columns if isinstance(df, pd.DataFrame) else None)
    x = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    timestamps = _timestamp_series_to_utc(x["timestamp"])
    valid = timestamps.notna()
    x = x.loc[valid].reset_index(drop=True)
    timestamps = timestamps.loc[valid].reset_index(drop=True)
    if len(x) <= 1:
        return x
    interval_ns = int(interval.value)
    ts_ns = timestamps.astype("int64").to_numpy()
    breaks = [
        (int(ts_ns[i]) - int(ts_ns[i - 1])) != interval_ns
        or (int(ts_ns[i]) % interval_ns) != 0
        or (int(ts_ns[i - 1]) % interval_ns) != 0
        for i in range(1, len(ts_ns))
    ]
    last_break = max((i for i, value in enumerate(breaks) if value), default=-1)
    return x.iloc[last_break + 1:].reset_index(drop=True)


def _atr14_5m(x: pd.DataFrame) -> float | None:
    # Fourteen true-range values need their preceding close as well; all
    # fifteen candle opens must be consecutive 5m intervals.
    if len(x) < 15 or not _has_contiguous_tail(x, 15, pd.Timedelta(minutes=5)):
        return None
    prev = x["close"].shift(1)
    tr = pd.concat(
        [x["high"] - x["low"], (x["high"] - prev).abs(), (x["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    vals = tr.tail(14).dropna()
    if len(vals) < 14:
        return None
    value = vals.mean()
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None


def _ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline <= 0:
        return None
    return float(value / baseline)


def _grid(value: float | None, thresholds: tuple[float, ...], *, ge: bool = True) -> dict[str, bool | None]:
    if value is None:
        return {str(t): None for t in thresholds}
    if ge:
        return {str(t): bool(value >= t) for t in thresholds}
    return {str(t): bool(value <= t) for t in thresholds}


def _pct_return(direction: str, entry: float, price: float) -> float:
    if entry <= 0:
        return 0.0
    return ((price / entry) - 1.0) * 100.0 if direction == "LONG" else (1.0 - (price / entry)) * 100.0


def _directional_hit(direction: str, high: float, low: float, price: float, favorable: bool) -> bool:
    if direction == "LONG":
        return high >= price if favorable else low <= price
    return low <= price if favorable else high >= price


def _stop_or_target_hit(direction: str, high: float, low: float, stop: float | None, target: float | None) -> tuple[str | None, bool]:
    stop_hit = stop is not None and _directional_hit(direction, high, low, float(stop), False)
    target_hit = target is not None and _directional_hit(direction, high, low, float(target), True)
    if stop_hit and target_hit:
        return "AMBIGUOUS_SAME_BAR", True
    if stop_hit:
        return "STOP", False
    if target_hit:
        return "TARGET", False
    return None, False


def build_entry_snapshot(
    signal: dict[str, Any],
    *,
    df_5m: pd.DataFrame | None,
    df_1h: pd.DataFrame | None,
    account_context: dict[str, Any] | None = None,
    decision_ts: Any | None = None,
) -> dict[str, Any]:
    direction = str(signal.get("type", "")).upper()
    entry = _finite(signal.get("entry"))
    entry_ts = _parse_ts(signal.get("trigger_bar_time") or signal.get("time"))
    decision_ts_parsed = _parse_ts(decision_ts or signal.get("decision_ts") or signal.get("observation_ts"))
    if decision_ts_parsed is None:
        # Standalone/unit-test snapshots have no separate decision boundary;
        # retain trigger-time semantics unless the caller supplies one explicitly.
        decision_ts_parsed = entry_ts or pd.Timestamp.now(tz="UTC")
    zone = signal.get("zone") if isinstance(signal.get("zone"), dict) else {}
    top = _finite(zone.get("top"))
    bottom = _finite(zone.get("btm"))
    if direction not in {"LONG", "SHORT"} or entry is None or entry <= 0 or entry_ts is None:
        return {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "experiment_version": SHADOW_EXPERIMENT_VERSION,
            "applied": False,
            "status": "INSUFFICIENT_ENTRY_DATA",
            "experiments": {},
        }

    # 5M trigger history is anchored to trigger-bar open because _closed_5m
    # includes that closed trigger bar (open + 5m).  HTF history is anchored to
    # the actual canonical decision boundary so an hourly candle that closes
    # between trigger open and trigger close is included when it was knowable.
    x5 = _closed_5m(df_5m, entry_ts)
    # The signal's entry_bar is the canonical candle that actually created the
    # production event. Some call paths pass a history frame that does not yet
    # contain the just-selected event candle. Reconcile the event candle into
    # the frame
    # even when a same-timestamp row exists: the event's OHLCV is authoritative,
    # while non-null exchange diagnostics from either source are retained.
    raw_entry_bar = signal.get("entry_bar")
    if isinstance(raw_entry_bar, dict):
        canonical_ts = _parse_ts(raw_entry_bar.get("timestamp"))
        canonical_values = {
            key: _finite(raw_entry_bar.get(key))
            for key in ("open", "high", "low", "close", "volume")
        }
        if canonical_ts is not None and canonical_ts == entry_ts and all(
            value is not None for value in canonical_values.values()
        ):
            open_price = canonical_values["open"]
            high_price = canonical_values["high"]
            low_price = canonical_values["low"]
            close_price = canonical_values["close"]
            volume = canonical_values["volume"]
            if (
                min(open_price, high_price, low_price, close_price) > 0
                and volume >= 0
                and high_price >= max(open_price, close_price, low_price)
                and low_price <= min(open_price, close_price, high_price)
            ):
                existing_rows = x5.loc[x5["timestamp"] == entry_ts] if not x5.empty else pd.DataFrame()
                canonical_row = existing_rows.iloc[-1].to_dict() if not existing_rows.empty else {}
                canonical_row.update(canonical_values)
                canonical_row.update({
                    "timestamp": entry_ts,
                    "close_time": entry_ts + pd.Timedelta(minutes=5),
                })
                flow_valid_raw = raw_entry_bar.get("taker_flow_valid")
                flow_valid = None if flow_valid_raw is None else bool(flow_valid_raw)
                if flow_valid is False:
                    # The event bar's explicit invalid-flow flag overrides a
                    # stale diagnostic value that may exist in the history row.
                    canonical_row.pop("bar_delta_usdt", None)
                for field in ("quote_volume", "taker_buy_base", "taker_buy_quote", "bar_delta_usdt"):
                    value = _finite(raw_entry_bar.get(field))
                    if value is not None and not (field == "bar_delta_usdt" and flow_valid is False):
                        canonical_row[field] = value
                if flow_valid is not None:
                    canonical_row["taker_flow_valid"] = flow_valid
                trade_count = raw_entry_bar.get("trade_count")
                if trade_count is not None:
                    try:
                        canonical_row["trade_count"] = int(trade_count)
                    except (TypeError, ValueError, OverflowError):
                        pass
                if not x5.empty:
                    x5 = x5.loc[x5["timestamp"] != entry_ts].copy()
                x5 = _closed_5m(
                    pd.concat([x5, pd.DataFrame([canonical_row])], ignore_index=True),
                    entry_ts,
                )
    trigger_rows = x5.loc[x5["timestamp"] == entry_ts] if not x5.empty else pd.DataFrame()
    x1 = _closed_1h(df_1h, decision_ts_parsed)
    trigger_bar = trigger_rows.iloc[-1] if not trigger_rows.empty else None
    prior5 = x5.loc[x5["timestamp"] < entry_ts] if not x5.empty else pd.DataFrame()
    prev5 = prior5.tail(20)
    vol5 = _finite(trigger_bar["volume"]) if trigger_bar is not None else None
    if trigger_bar is None:
        volume_5m_status = "MISSING_TRIGGER_BAR"
        baseline5 = None
    elif len(prev5) < 20:
        volume_5m_status = "INSUFFICIENT_HISTORY"
        baseline5 = None
    elif not _has_contiguous_tail(prev5, 20, pd.Timedelta(minutes=5)) or prior5.empty or prior5["timestamp"].iloc[-1] + pd.Timedelta(minutes=5) != entry_ts:
        volume_5m_status = "GAPPED_HISTORY"
        baseline5 = None
    else:
        baseline5 = _finite(prev5["volume"].mean())
        volume_5m_status = "AVAILABLE" if baseline5 is not None and baseline5 > 0 and vol5 is not None else "INVALID_BASELINE"
    volume_5m_ratio = _ratio(vol5, baseline5)

    last1h = x1.iloc[-1] if not x1.empty else None
    prev1h = x1.iloc[:-1].tail(20) if len(x1) >= 21 else pd.DataFrame()
    vol1h = _finite(last1h["volume"]) if last1h is not None else None
    if last1h is None or len(prev1h) < 20:
        volume_1h_status = "INSUFFICIENT_HISTORY"
        baseline1h = None
    elif not _has_contiguous_tail(prev1h, 20, pd.Timedelta(hours=1)) or prev1h["timestamp"].iloc[-1] + pd.Timedelta(hours=1) != last1h["timestamp"]:
        volume_1h_status = "GAPPED_HISTORY"
        baseline1h = None
    else:
        baseline1h = _finite(prev1h["volume"].mean())
        volume_1h_status = "AVAILABLE" if baseline1h is not None and baseline1h > 0 and vol1h is not None else "INVALID_BASELINE"
    volume_1h_ratio = _ratio(vol1h, baseline1h)

    atr5 = _atr14_5m(x5) if trigger_bar is not None else None
    if trigger_bar is None:
        atr5_status = "MISSING_TRIGGER_BAR"
    elif atr5 is not None:
        atr5_status = "AVAILABLE"
    elif len(x5) >= 15:
        atr5_status = "GAPPED_HISTORY"
    else:
        atr5_status = "INSUFFICIENT_HISTORY"
    atr5_pct = (atr5 / entry * 100.0) if atr5 is not None and entry > 0 else None
    delta = _finite(trigger_bar.get("bar_delta_usdt")) if trigger_bar is not None and hasattr(trigger_bar, "get") else None
    delta_confirmed = None if delta is None else (delta > 0 if direction == "LONG" else delta < 0)

    ema200 = None
    ema200_distance_pct = None
    ema200_trend_up = None
    contiguous_1h = _contiguous_suffix(x1, pd.Timedelta(hours=1))
    if len(contiguous_1h) >= 200:
        closes = contiguous_1h["close"]
        ema_series = closes.ewm(span=200, adjust=False, min_periods=200).mean()
        ema200 = _finite(ema_series.iloc[-1])
        if ema200 is not None:
            ema200_distance_pct = (entry / ema200 - 1.0) * 100.0
            if len(ema_series) >= 2 and pd.notna(ema_series.iloc[-2]):
                ema200_trend_up = bool(float(ema_series.iloc[-1]) > float(ema_series.iloc[-2]))
        ema200_status = "AVAILABLE_CONTIGUOUS_SUFFIX" if len(contiguous_1h) < len(x1) else "AVAILABLE"
    else:
        ema200_status = "GAPPED_HISTORY" if len(x1) >= 200 else "INSUFFICIENT_HISTORY"
    macro_alignment = None if ema200 is None else (entry > ema200 if direction == "LONG" else entry < ema200)

    far_edge = bottom if direction == "LONG" else top
    structural_sl = None
    structural_risk_pct = None
    structural_available = far_edge is not None and atr5 is not None and atr5 > 0
    if structural_available:
        structural_sl = far_edge - atr5 * STRUCTURAL_SL_ATR_MULT if direction == "LONG" else far_edge + atr5 * STRUCTURAL_SL_ATR_MULT
        structural_risk_pct = abs(entry - structural_sl) / entry * 100.0 if structural_sl > 0 else None
    structural_pass = None if structural_risk_pct is None else bool(STRUCTURAL_SL_MIN_PCT <= structural_risk_pct <= STRUCTURAL_SL_MAX_PCT)
    structural_risk_abs = abs(entry - structural_sl) if structural_sl is not None else None
    tp1_rr = entry + structural_risk_abs * TP1_R_MULT if direction == "LONG" and structural_risk_abs is not None else entry - structural_risk_abs * TP1_R_MULT if direction == "SHORT" and structural_risk_abs is not None else None
    tp2_rr = entry + structural_risk_abs * TP2_R_MULT if direction == "LONG" and structural_risk_abs is not None else entry - structural_risk_abs * TP2_R_MULT if direction == "SHORT" and structural_risk_abs is not None else None

    penetration_ratio = None
    if trigger_bar is not None and top is not None and bottom is not None and top > bottom:
        trigger_low = _finite(trigger_bar.get("low"))
        trigger_high = _finite(trigger_bar.get("high"))
        width = top - bottom
        if direction == "LONG" and trigger_low is not None:
            penetration_ratio = (top - trigger_low) / width
        elif direction == "SHORT" and trigger_high is not None:
            penetration_ratio = (trigger_high - bottom) / width
    penetration_priority = (
        "HIGH" if penetration_ratio is not None and 0.70 <= penetration_ratio <= 1.00
        else "NORMAL" if penetration_ratio is not None else "UNKNOWN"
    )

    trigger_session_hour = int(entry_ts.hour)
    decision_session_hour = int(decision_ts_parsed.hour)
    session_hour = decision_session_hour
    asia_block_window = 0 <= session_hour < 6
    if asia_block_window:
        session = "ASIA_OFF_HOURS"
    elif 6 <= session_hour < 13:
        session = "LONDON"
    elif 13 <= session_hour < 16:
        session = "LONDON_NY_OVERLAP"
    elif 16 <= session_hour < 24:
        session = "NEW_YORK"
    else:
        session = "UNKNOWN"

    body_close_invalid_before_entry = False
    invalidation_ts = None
    if top is not None and bottom is not None and not x5.empty:
        origin_ts = _parse_ts(zone.get("origin_ts_ms"))
        prior_bars = x5.loc[x5["timestamp"] < entry_ts].copy()
        if origin_ts is not None:
            prior_bars = prior_bars.loc[prior_bars["timestamp"] >= origin_ts]
        for _, row in prior_bars.iterrows():
            c = _finite(row["close"])
            if c is None:
                continue
            invalid = (direction == "LONG" and c < bottom) or (direction == "SHORT" and c > top)
            if invalid:
                body_close_invalid_before_entry = True
                invalidation_ts = _parse_ts(row["close_time"]).isoformat() if _parse_ts(row["close_time"]) is not None else None
                break

    open_symbols: list[str] = []
    if isinstance(account_context, dict):
        for pos in account_context.get("positions") or []:
            if isinstance(pos, dict) and pos.get("symbol"):
                open_symbols.append(str(pos["symbol"]).upper())
    account_context_available = isinstance(account_context, dict)
    open_symbols = sorted(set(open_symbols))

    taker_fee_rate = _finite((account_context or {}).get("taker_commission_rate")) if account_context_available else None
    maker_fee_rate = _finite((account_context or {}).get("maker_commission_rate")) if account_context_available else None
    roundtrip_taker_fee_pct = taker_fee_rate * 2.0 * 100.0 if taker_fee_rate is not None else None

    production_sl = _finite(signal.get("sl"))
    production_tp1 = _finite(signal.get("tp1"))
    production_tp2 = _finite(signal.get("tp2"))
    production_tp3 = _finite(signal.get("tp3"))
    if production_sl is None:
        production_risk_abs = entry * PRODUCTION_STOP_PCT / 100.0
        production_sl = entry - production_risk_abs if direction == "LONG" else entry + production_risk_abs
    if production_tp1 is None:
        production_tp1_distance = entry * PRODUCTION_TP1_PCT / 100.0
        production_tp1 = entry + production_tp1_distance if direction == "LONG" else entry - production_tp1_distance
    if production_tp2 is None:
        production_tp2_distance = entry * PRODUCTION_TP2_PCT / 100.0
        production_tp2 = entry + production_tp2_distance if direction == "LONG" else entry - production_tp2_distance
    return {
        "schema_version": SHADOW_SCHEMA_VERSION,
        "experiment_version": SHADOW_EXPERIMENT_VERSION,
        "applied": False,
        "decision_anchor": {
            "event_id": signal.get("event_id"),
            "symbol": signal.get("symbol"),
            "direction": direction,
            "entry": entry,
            "trigger_bar_time": entry_ts.isoformat(),
            "decision_ts": decision_ts_parsed.isoformat(),
            "counterfactual_path_start_ts": decision_ts_parsed.isoformat(),
            "source": "closed_5m_signal_bar",
        },
        "experiments": {
            "CF_STRUCTURAL_SL": {
                "version": 1,
                "rule": "far_zone_edge_plus_0.5x_ATR_5m",
                "far_zone_edge": far_edge,
                "atr_5m": atr5,
                "atr_multiplier": STRUCTURAL_SL_ATR_MULT,
                "hypothetical_sl": structural_sl,
                "risk_pct": structural_risk_pct,
                "min_risk_pct": STRUCTURAL_SL_MIN_PCT,
                "max_risk_pct": STRUCTURAL_SL_MAX_PCT,
                "would_pass": structural_pass,
            },
            "CF_TP_RR_15_30": {
                "version": 1,
                "risk_pct": structural_risk_pct,
                "tp1_rr": TP1_R_MULT,
                "tp2_rr": TP2_R_MULT,
                "hypothetical_tp1": tp1_rr,
                "hypothetical_tp2": tp2_rr,
                "would_be_valid": structural_risk_abs is not None and structural_risk_abs > 0,
            },
            "CF_VOLUME_5M": {
                "version": 1,
                "trigger_volume": vol5,
                "baseline_sma20_previous_bars": baseline5,
                "ratio": volume_5m_ratio,
                "status": volume_5m_status,
                "threshold_grid": _grid(volume_5m_ratio, VOLUME_5M_THRESHOLDS),
            },
            "CF_VOLUME_1H": {
                "version": 1,
                "current_closed_1h_volume": vol1h,
                "baseline_sma20_previous_bars": baseline1h,
                "ratio": volume_1h_ratio,
                "status": volume_1h_status,
                "threshold_grid": _grid(volume_1h_ratio, VOLUME_1H_THRESHOLDS),
            },
            "CF_DELTA_CONFIRM": {
                "version": 1,
                "bar_delta_usdt": delta,
                "rule": "LONG>0 / SHORT<0",
                "would_pass": delta_confirmed,
            },
            "CF_EMA200_1H": {
                "version": 1,
                "ema200_1h": ema200,
                "distance_pct": ema200_distance_pct,
                "ema200_trend_up": ema200_trend_up,
                "status": ema200_status,
                "rule": "LONG price>EMA200 / SHORT price<EMA200",
                "would_pass": macro_alignment,
            },
            "CF_ATR_MIN_5M": {
                "version": 1,
                "atr14_5m": atr5,
                "atr_pct": atr5_pct,
                "status": atr5_status,
                "threshold_grid_pct": _grid(atr5_pct, ATR_5M_MIN_THRESHOLDS),
            },
            "CF_SESSION": {
                "version": 1,
                "session_utc": session,
                "utc_hour": session_hour,
                "trigger_session_utc": (
                    "ASIA_OFF_HOURS" if 0 <= trigger_session_hour < 6 else
                    "LONDON" if 6 <= trigger_session_hour < 13 else
                    "LONDON_NY_OVERLAP" if 13 <= trigger_session_hour < 16 else
                    "NEW_YORK" if 16 <= trigger_session_hour < 24 else "UNKNOWN"
                ),
                "trigger_utc_hour": trigger_session_hour,
                "decision_session_utc": session,
                "decision_utc_hour": decision_session_hour,
                "asia_block_window": "00:00<=UTC<06:00",
                "asia_off_hours": asia_block_window,
                "would_pass_if_asia_blocked": not asia_block_window,
            },
            "CF_ZONE_INVALIDATION": {
                "version": 1,
                "zone_top": top,
                "zone_bottom": bottom,
                "body_close_invalid_before_entry": body_close_invalid_before_entry,
                "invalidation_ts": invalidation_ts,
                "would_pass": not body_close_invalid_before_entry,
            },
            "CF_CORRELATION": {
                "version": 1,
                "threshold": 0.80,
                "open_position_symbols": open_symbols,
                "max_open_position_correlation": None,
                "would_pass": None,
                "status": (
                    "CONTEXT_UNAVAILABLE"
                    if not account_context_available
                    else "DEFERRED_TO_AUDIT_WITH_STORED_MARKET_BARS" if open_symbols
                    else "NO_OPEN_PEERS"
                ),
            },
            "CF_PRODUCTION_MANAGEMENT": {
                "version": 3 if production_tp3 is not None else 2,
                "stop": production_sl,
                "tp1": production_tp1,
                "tp2": production_tp2,
                "tp3": production_tp3,
                "tp1_fraction": PRODUCTION_TP1_FRACTION,
                "tp2_fraction": PRODUCTION_TP2_FRACTION if production_tp3 is not None else 0.50,
                "tp3_fraction": PRODUCTION_TP3_FRACTION if production_tp3 is not None else None,
                "management_baseline": "current_production_SL_TP1_50_TP2_25_TP3_25_BE_after_TP1" if production_tp3 is not None else "legacy_production_SL_TP1_TP2_50_50_BE_after_TP1",
            },
            "CF_COST_MODEL": {
                "version": 1,
                "taker_fee_rate": taker_fee_rate,
                "maker_fee_rate": maker_fee_rate,
                "roundtrip_taker_fee_pct": roundtrip_taker_fee_pct,
                "roundtrip_cost_scenarios_pct": {
                    str(x): (None if roundtrip_taker_fee_pct is None else roundtrip_taker_fee_pct + x)
                    for x in ROUNDTRIP_SLIPPAGE_SCENARIOS_PCT
                },
                "slippage_scenario_semantics": "roundtrip_pct_added_to_roundtrip_taker_fee",
            },
            "CF_DELAYED_BE": {
                "version": 2,
                "activation": "after_production_TP1_then_N_completed_closed_bars",
                "bars_grid": list(DELAYED_BE_BARS),
                "production_sl": production_sl,
                "production_tp1": production_tp1,
                "production_tp2": production_tp2,
                "production_tp3": production_tp3,
                "tp1_fraction": PRODUCTION_TP1_FRACTION,
                "tp2_fraction": PRODUCTION_TP2_FRACTION if production_tp3 is not None else 0.50,
                "tp3_fraction": PRODUCTION_TP3_FRACTION if production_tp3 is not None else None,
                "standalone_management_only": True,
                "production_gate_applied": False,
            },
            "CF_TRAIL_ATR": {
                "version": 2,
                "activation": "after_production_TP1",
                "atr_multiplier_grid": list(TRAIL_ATR_MULTIPLIERS),
                "production_sl": production_sl,
                "production_tp1": production_tp1,
                "production_tp2": production_tp2,
                "production_tp3": production_tp3,
                "tp1_fraction": PRODUCTION_TP1_FRACTION,
                "tp2_fraction": PRODUCTION_TP2_FRACTION if production_tp3 is not None else 0.50,
                "tp3_fraction": PRODUCTION_TP3_FRACTION if production_tp3 is not None else None,
                "standalone_management_only": True,
                "production_gate_applied": False,
            },
            "CF_PENETRATION_PRIORITY": {
                "version": 1,
                "zone_penetration_ratio_raw": penetration_ratio,
                "priority": penetration_priority,
                "priority_threshold": {"min_inclusive": 0.70, "max_inclusive": 1.00},
                "standalone_management_only": True,
                "production_gate_applied": False,
                "signal_score_changed": False,
                "selection_order_changed": False,
            },
            "CF_MFE_PROTECTION_PLUS_2": {
                "version": 1,
                "activation_mfe_pct": MFE_PROTECTION_ACTIVATION_PCT,
                "locked_profit_levels_pct": list(MFE_PROTECTION_LOCK_LEVELS_PCT),
                "activation_semantics": "after_completed_future_bar_reaches_mfe_threshold",
                "stop_semantics": "new_stop_first_active_on_next_completed_bar",
                "same_bar_policy": "ambiguous_when_mfe_coexists_with_stop_or_tp1",
                "tp1_fraction": PRODUCTION_TP1_FRACTION,
                "tp2_fraction": PRODUCTION_TP2_FRACTION if production_tp3 is not None else 0.50,
                "tp3_fraction": PRODUCTION_TP3_FRACTION if production_tp3 is not None else None,
                "standalone_management_only": True,
                "production_gate_applied": False,
            },
        },
    }


def _simulate_single_exit(
    future: pd.DataFrame,
    *,
    direction: str,
    entry: float,
    stop: float | None,
    target: float | None,
) -> dict[str, Any]:
    result = {
        "exit": None,
        "exit_price": None,
        "exit_ts": None,
        "pnl_pct": None,
        "ambiguous": False,
        "bars_to_exit": None,
    }
    if future.empty:
        return result
    for idx, (high_value, low_value, close_time) in enumerate(
        future.reset_index(drop=True)[["high", "low", "close_time"]].itertuples(index=False, name=None)
    ):
        event, ambiguous = _stop_or_target_hit(direction, float(high_value), float(low_value), stop, target)
        if event is None:
            continue
        result["exit"] = event
        result["ambiguous"] = ambiguous
        result["exit_price"] = target if event == "TARGET" else stop if event == "STOP" else None
        parsed_close_time = _parse_ts(close_time)
        result["exit_ts"] = parsed_close_time.isoformat() if parsed_close_time is not None else None
        result["bars_to_exit"] = idx + 1
        if result["exit_price"] is not None:
            result["pnl_pct"] = _pct_return(direction, entry, float(result["exit_price"]))
        return result
    return result


def _weighted_partial_exit_pnl(
    direction: str,
    entry: float,
    tp1_price: float,
    residual_price: float,
    tp1_fraction: float = PRODUCTION_TP1_FRACTION,
) -> float:
    tp1_pnl = _pct_return(direction, entry, tp1_price)
    residual_pnl = _pct_return(direction, entry, residual_price)
    return float(tp1_pnl * tp1_fraction + residual_pnl * (1.0 - tp1_fraction))


def _weighted_tp_pnl(direction: str, entry: float, realized_levels: list[tuple[float, float]], remaining_fraction: float = 0.0, residual_price: float | None = None) -> float:
    """Weighted whole-position PnL from confirmed target legs and an optional residual exit."""
    total = sum(_pct_return(direction, entry, price) * fraction for price, fraction in realized_levels)
    if remaining_fraction > 0 and residual_price is not None:
        total += _pct_return(direction, entry, residual_price) * remaining_fraction
    return float(total)


_SHADOW_ATR_BEFORE_CLOSE = "_shadow_atr_before_close"


def _precompute_atr_before_close(bars: pd.DataFrame) -> np.ndarray | None:
    """Precompute the exact trailing ATR used by counterfactual simulations.

    For each current candle, the old implementation selected all rows with
    ``close_time < current.close_time`` and averaged the last 14 true-range
    values, but only when at least 15 historical rows existed. This vectorized
    version preserves that strict prior-close boundary and warm-up rule.

    Return ``None`` for non-monotonic close times; the caller then uses the
    conservative legacy calculation for those unusual inputs.
    """
    if bars.empty or "close_time" not in bars.columns:
        return None
    close_time = bars["close_time"]
    if not close_time.is_monotonic_increasing:
        return None

    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(high).all() and np.isfinite(low).all() and np.isfinite(close).all()):
        return None

    previous_close = np.empty_like(close)
    previous_close[0:1] = np.nan
    if close.size > 1:
        previous_close[1:] = close[:-1]
    # pandas DataFrame.max(axis=1) skips NaN, so the first TR equals high-low.
    true_range = np.fmax.reduce((high - low, np.abs(high - previous_close), np.abs(low - previous_close)))
    atr14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().to_numpy(dtype=float)

    close_ns = close_time.astype("int64").to_numpy()
    open_ns = bars["timestamp"].astype("int64").to_numpy()
    # Strictly exclude the current candle: the historical implementation used
    # close_time < current_close_time, not <=. searchsorted-left also handles
    # duplicate close times without allowing same-close candles into history.
    history_ends = np.searchsorted(close_ns, close_ns, side="left")
    result = np.full(len(bars), np.nan, dtype=float)
    valid_positions = np.flatnonzero(history_ends >= 15)
    interval_ns = int(pd.Timedelta(minutes=5).value)
    for position in valid_positions:
        end = int(history_ends[position])
        start = end - 15
        if start < 0:
            continue
        # ATR14 at this candle must be based on 14 consecutive true ranges,
        # including the previous close for the first range, and the current
        # candle must immediately follow that closed history.
        cadence = np.concatenate((open_ns[start:end], open_ns[position:position + 1]))
        if len(cadence) != 16 or not np.all(np.diff(cadence) == interval_ns):
            continue
        if np.any(cadence % interval_ns != 0):
            continue
        atr_index = end - 1
        value = atr14[atr_index]
        if np.isfinite(value):
            result[position] = value
    return result


def _simulate_management_only(
    all_bars: pd.DataFrame,
    future: pd.DataFrame,
    *,
    direction: str,
    entry: float,
    production_stop: float | None,
    production_tp1: float | None,
    production_tp2: float | None,
    production_tp3: float | None = None,
    mode: str,
    parameter: float | int,
) -> dict[str, Any]:
    """Simulate management variants using the persisted production TP plan.

    DELAYED_BE keeps every planned TP leg active. TRAIL_ATR is an alternative
    management experiment: TP1 remains a 50% partial exit and the residual is
    managed by the ATR trail rather than fixed TP2/TP3 exits. Historical
    snapshots without TP3 retain the old 50/50 allocation.
    """
    if future is None or future.empty or production_stop is None or production_tp1 is None or production_tp2 is None:
        return {"status": "INSUFFICIENT_PATH", "pnl_pct": None, "exit": None, "standalone_management_only": True}

    direction = str(direction).upper()
    mode = str(mode).upper()
    if direction not in {"LONG", "SHORT"} or mode not in {"DELAYED_BE", "TRAIL_ATR"}:
        return {"status": "INVALID_INPUT", "pnl_pct": None, "exit": None, "standalone_management_only": True}

    legacy_two_tp = production_tp3 is None
    production_rows = [
        (float(production_tp1), 0.50, "tp1"),
        (float(production_tp2), 0.50 if legacy_two_tp else 0.25, "tp2"),
    ]
    if not legacy_two_tp:
        production_rows.append((float(production_tp3), 0.25, "tp3"))
    prices = [row[0] for row in production_rows]
    fractions = [row[1] for row in production_rows]
    if abs(sum(fractions) - 1.0) > 1e-9:
        return {"status": "INVALID_TARGET_ALLOCATION", "pnl_pct": None, "exit": None, "standalone_management_only": True}
    if (direction == "LONG" and not all(a < b for a, b in zip(prices, prices[1:]))) or (direction == "SHORT" and not all(a > b for a, b in zip(prices, prices[1:]))):
        return {"status": "INVALID_TARGET_LADDER", "pnl_pct": None, "exit": None, "standalone_management_only": True}

    all_bars = all_bars.copy().reset_index(drop=True)
    future = future.reset_index(drop=True)
    realized_levels: list[tuple[float, float]] = []
    realized_legs: list[str] = []
    realized_fraction = 0.0
    tp1_fill_idx: int | None = None
    tp1_fill_ts: str | None = None
    current_stop = float(production_stop)
    has_precomputed_atr = _SHADOW_ATR_BEFORE_CLOSE in future.columns
    if not has_precomputed_atr:
        all_bars["timestamp"] = _timestamp_series_to_utc(all_bars["timestamp"])
        all_bars["close_time"] = _timestamp_series_to_utc(all_bars["close_time"])
        for col in ("open", "high", "low", "close"):
            all_bars[col] = pd.to_numeric(all_bars[col], errors="coerce")
    future_atr = future[_SHADOW_ATR_BEFORE_CLOSE].to_numpy(dtype=float, copy=False) if has_precomputed_atr else None

    for idx, row in future.iterrows():
        high = _finite(row.get("high"))
        low = _finite(row.get("low"))
        close_time = _parse_ts(row.get("close_time"))
        if high is None or low is None or close_time is None:
            return {"status": "INVALID_FUTURE_BAR", "exit": None, "pnl_pct": None, "standalone_management_only": True}

        # Recalculate the stop before checking this candle. Every input below
        # comes strictly from previously closed bars, so the current candle's
        # high/low cannot move a stop and then trigger that stop retroactively.
        if tp1_fill_idx is not None and idx > tp1_fill_idx:
            if mode == "DELAYED_BE":
                activation_idx = tp1_fill_idx + int(parameter) + 1
                if idx >= activation_idx:
                    current_stop = float(entry)
            else:  # TRAIL_ATR
                if has_precomputed_atr:
                    atr = _finite(future_atr[idx])
                else:
                    current_close_ts = close_time
                    hist = all_bars.loc[all_bars["close_time"] < current_close_ts]
                    current_open_ts = current_close_ts - pd.Timedelta(minutes=5)
                    history_ts = _timestamp_series_to_utc(hist["timestamp"].tail(15)) if len(hist) >= 15 else pd.Series(dtype="datetime64[ns, UTC]")
                    cadence_ok = len(history_ts) == 15 and not history_ts.isna().any() and history_ts.diff().dropna().eq(pd.Timedelta(minutes=5)).all() and history_ts.iloc[-1] + pd.Timedelta(minutes=5) == current_open_ts
                    if cadence_ok:
                        previous_close = hist["close"].shift(1)
                        true_range = pd.concat([hist["high"] - hist["low"], (hist["high"] - previous_close).abs(), (hist["low"] - previous_close).abs()], axis=1).max(axis=1)
                        atr_values = true_range.dropna().tail(14)
                        atr = _finite(atr_values.mean()) if len(atr_values) == 14 else None
                    else:
                        atr = None
                if atr is not None:
                    # Preserve the experiment's historical convention: the TP1
                    # price anchors the first trail; later closed bars extend it.
                    prior_future = future.iloc[tp1_fill_idx + 1:idx]
                    if direction == "LONG":
                        prior_extreme = max([float(production_tp1)] + [float(x) for x in prior_future["high"].dropna().tolist()])
                        current_stop = max(current_stop, prior_extreme - atr * float(parameter))
                    else:
                        prior_extreme = min([float(production_tp1)] + [float(x) for x in prior_future["low"].dropna().tolist()])
                        current_stop = min(current_stop, prior_extreme + atr * float(parameter))

        # In the ATR alternative TP1 is retained but the remaining quantity is
        # trailing-managed. DELAYED_BE continues to honor TP2 and TP3.
        if mode == "TRAIL_ATR" and tp1_fill_idx is not None:
            target_hits = [False] * len(production_rows)
        else:
            target_hits = [
                leg not in realized_legs and _directional_hit(direction, high, low, price, True)
                for price, _, leg in production_rows
            ]
        stop_hit = _directional_hit(direction, high, low, current_stop, False)
        remaining_fraction = max(0.0, 1.0 - realized_fraction)

        if stop_hit and any(target_hits):
            return {
                "status": "AMBIGUOUS_STOP_AND_TARGET_SAME_BAR", "exit": "AMBIGUOUS", "exit_price": None,
                "exit_ts": close_time.isoformat(), "pnl_pct": None,
                "realized_target_legs": list(realized_legs),
                "realized_target_fraction": realized_fraction, "residual_fraction": remaining_fraction,
                "standalone_management_only": True,
            }
        if stop_hit:
            pnl = _weighted_tp_pnl(direction, entry, realized_levels, remaining_fraction, current_stop)
            residual_pnl = _pct_return(direction, entry, current_stop) if remaining_fraction > 0 else 0.0
            tp1_pnl = _pct_return(direction, entry, float(production_tp1)) if "tp1" in realized_legs else None
            status = "STOP_BEFORE_TP1" if realized_fraction <= 1e-9 else "EXIT_AFTER_PARTIAL_TPS"
            exit_name = "PRODUCTION_SL" if not realized_legs else ("BE_AFTER_TP1" if abs(current_stop - entry) <= max(1e-12, abs(entry) * 1e-12) else mode)
            return {
                "status": status, "exit": exit_name, "exit_price": float(current_stop), "tp1_fill_ts": tp1_fill_ts,
                "tp1_pnl_pct": tp1_pnl, "residual_exit_pnl_pct": residual_pnl,
                "realized_target_legs": list(realized_legs), "realized_target_fraction": realized_fraction,
                "residual_fraction": remaining_fraction, "pnl_pct": pnl,
                "bars_to_exit": idx + 1, "exit_ts": close_time.isoformat(), "standalone_management_only": True,
            }

        hit_indexes = [i for i, hit in enumerate(target_hits) if hit]
        if hit_indexes:
            highest_hit = max(hit_indexes)
            for target_index, (price, fraction, leg) in enumerate(production_rows):
                if target_index <= highest_hit and leg not in realized_legs:
                    realized_levels.append((price, fraction))
                    realized_legs.append(leg)
                    realized_fraction += fraction
                    if leg == "tp1" and tp1_fill_idx is None:
                        tp1_fill_idx = idx
                        tp1_fill_ts = close_time.isoformat()
            if realized_fraction >= 1.0 - 1e-9:
                last_leg = production_rows[-1][2].upper()
                return {
                    "status": "ALL_TP_LEGS_REACHED" if not legacy_two_tp else "TP1_AND_TP2_REACHED",
                    "exit": f"PRODUCTION_{last_leg}", "exit_price": production_rows[-1][0],
                    "tp1_fill_ts": tp1_fill_ts, "realized_target_legs": list(realized_legs),
                    "realized_target_fraction": realized_fraction, "residual_fraction": 0.0,
                    "pnl_pct": _weighted_tp_pnl(direction, entry, realized_levels),
                    "bars_to_exit": idx + 1, "exit_ts": close_time.isoformat(), "standalone_management_only": True,
                }
            continue

    if tp1_fill_idx is None:
        return {"status": "PATH_NOT_REACHED_TP1", "exit": None, "exit_price": None, "pnl_pct": None, "standalone_management_only": True}
    return {
        "status": "PATH_NOT_EXITED", "exit": None, "exit_price": None,
        "tp1_fill_ts": tp1_fill_ts, "tp1_pnl_pct": _pct_return(direction, entry, float(production_tp1)),
        "realized_target_legs": list(realized_legs), "realized_target_fraction": realized_fraction,
        "residual_fraction": max(0.0, 1.0 - realized_fraction), "pnl_pct": None,
        "standalone_management_only": True,
    }


def _simulate_mfe_protection(
    future: pd.DataFrame,
    *,
    direction: str,
    entry: float,
    production_stop: float,
    production_tp1: float,
    production_tp2: float,
    activation_mfe_pct: float,
    locked_profit_pct: float,
    production_tp3: float | None = None,
) -> dict[str, Any]:
    """Simulate an MFE stop lock with production TP allocation and no look-ahead.

    TP1/TP2-only historical snapshots retain their old 50/50 model; new
    observations with TP3 use the production 50/25/25 plan.
    """
    legacy_two_tp = production_tp3 is None
    target_rows = [
        (float(production_tp1), 0.50, "tp1"),
        (float(production_tp2), 0.50 if legacy_two_tp else 0.25, "tp2"),
    ]
    if not legacy_two_tp:
        target_rows.append((float(production_tp3), 0.25, "tp3"))
    prices = [x[0] for x in target_rows]
    fractions = [x[1] for x in target_rows]
    base = {
        "standalone_management_only": True,
        "production_gate_applied": False,
        "activation_mfe_pct": float(activation_mfe_pct),
        "locked_profit_pct": float(locked_profit_pct),
        "mfe_activation_ts": None,
        "tp1_fill_ts": None,
        "exit_ts": None,
        "exit_price": None,
        "pnl_pct": None,
        "bars_to_exit": None,
    }
    if future is None or future.empty or entry <= 0:
        return {**base, "status": "INSUFFICIENT_PATH", "exit": None}
    direction = str(direction).upper()
    if direction not in {"LONG", "SHORT"}:
        return {**base, "status": "INVALID_DIRECTION", "exit": None}
    if abs(sum(fractions) - 1.0) > 1e-9:
        return {**base, "status": "INVALID_TARGET_ALLOCATION", "exit": None}
    if (direction == "LONG" and not all(a < b for a, b in zip(prices, prices[1:]))) or (direction == "SHORT" and not all(a > b for a, b in zip(prices, prices[1:]))):
        return {**base, "status": "INVALID_TARGET_LADDER", "exit": None}

    mfe_trigger = entry * (1.0 + activation_mfe_pct / 100.0) if direction == "LONG" else entry * (1.0 - activation_mfe_pct / 100.0)
    lock_stop = entry * (1.0 + locked_profit_pct / 100.0) if direction == "LONG" else entry * (1.0 - locked_profit_pct / 100.0)
    activated = False
    activation_ts = None
    tp1_fill_ts = None
    realized_levels: list[tuple[float, float]] = []
    realized_legs: list[str] = []
    realized_fraction = 0.0
    current_stop = float(production_stop)
    future = future.reset_index(drop=True)

    def finish(status: str, exit_name: str | None, price: float | None, idx: int, ts: Any) -> dict[str, Any]:
        out = dict(base)
        residual = max(0.0, 1.0 - realized_fraction)
        out.update({
            "status": status,
            "exit": exit_name,
            "exit_price": float(price) if price is not None else None,
            "exit_ts": ts.isoformat() if ts is not None else None,
            "bars_to_exit": int(idx) + 1,
            "mfe_activation_ts": activation_ts.isoformat() if activation_ts is not None else None,
            "tp1_fill_ts": tp1_fill_ts.isoformat() if tp1_fill_ts is not None else None,
            "realized_target_legs": list(realized_legs),
            "realized_target_fraction": realized_fraction,
            "residual_fraction": residual,
        })
        if price is not None:
            out["pnl_pct"] = _weighted_tp_pnl(direction, entry, realized_levels, residual, float(price))
            if "tp1" in realized_legs:
                out["tp1_pnl_pct"] = _pct_return(direction, entry, float(production_tp1))
                out["residual_exit_pnl_pct"] = _pct_return(direction, entry, float(price)) if residual > 0 else 0.0
        return out

    for idx, row in future.iterrows():
        high = _finite(row.get("high"))
        low = _finite(row.get("low"))
        close_time = _parse_ts(row.get("close_time"))
        if high is None or low is None or close_time is None:
            return finish("INVALID_FUTURE_BAR", None, None, idx, close_time)

        target_hits = [
            leg not in realized_legs and _directional_hit(direction, high, low, price, True)
            for price, _, leg in target_rows
        ]
        stop_hit = _directional_hit(direction, high, low, current_stop, False)
        mfe_hit = not activated and not realized_legs and _directional_hit(direction, high, low, mfe_trigger, True)

        if stop_hit and mfe_hit and not any(target_hits):
            return finish("AMBIGUOUS_MFE_AND_PRODUCTION_STOP_SAME_BAR", "AMBIGUOUS", None, idx, close_time)
        if stop_hit and any(target_hits):
            return finish("AMBIGUOUS_MFE_STOP_AND_TARGET_SAME_BAR", "AMBIGUOUS", None, idx, close_time)
        if stop_hit:
            if activated and not realized_legs:
                return finish("STOP_AFTER_MFE_PROTECTION", "MFE_PROTECTION", current_stop, idx, close_time)
            if realized_legs:
                status = "EXIT_AFTER_TP1_AND_MFE" if activated or abs(current_stop - entry) <= max(1e-12, abs(entry) * 1e-12) else "EXIT_AFTER_PARTIAL_TPS"
                return finish(status, "MFE_PROTECTION_OR_BE" if activated or abs(current_stop - entry) <= max(1e-12, abs(entry) * 1e-12) else "PRODUCTION_STOP", current_stop, idx, close_time)
            return finish("STOP_BEFORE_TP1", "PRODUCTION_SL", current_stop, idx, close_time)
        if mfe_hit and any(target_hits):
            return finish("AMBIGUOUS_MFE_AND_TARGET_SAME_BAR", "AMBIGUOUS", None, idx, close_time)
        if mfe_hit:
            activated = True
            activation_ts = close_time
            current_stop = lock_stop
            # New stop becomes effective only on the following closed candle.
            continue

        hit_indexes = [i for i, hit in enumerate(target_hits) if hit]
        if hit_indexes:
            highest_hit = max(hit_indexes)
            for index, (price, fraction, leg) in enumerate(target_rows):
                if index <= highest_hit and leg not in realized_legs:
                    realized_levels.append((price, fraction))
                    realized_legs.append(leg)
                    realized_fraction += fraction
                    if leg == "tp1" and tp1_fill_ts is None:
                        tp1_fill_ts = close_time
            if realized_fraction >= 1.0 - 1e-9:
                last_leg = target_rows[-1][2].upper()
                return finish("ALL_TP_LEGS_REACHED" if not legacy_two_tp else "TP1_AND_TP2_REACHED", f"PRODUCTION_{last_leg}", prices[-1], idx, close_time)
            if "tp1" in realized_legs:
                # Existing BE rule activates only after TP1 is confirmed; it
                # applies from the next bar in this closed-bar simulation.
                be_stop = float(entry)
                current_stop = max(current_stop, be_stop) if direction == "LONG" else min(current_stop, be_stop)
            continue

    status = "PATH_NOT_EXITED" if activated else ("TP1_FILLED_PATH_NOT_EXITED" if "tp1" in realized_legs else "MFE_NOT_REACHED")
    out = dict(base)
    out.update({
        "status": status,
        "exit": None,
        "mfe_activation_ts": activation_ts.isoformat() if activation_ts is not None else None,
        "tp1_fill_ts": tp1_fill_ts.isoformat() if tp1_fill_ts is not None else None,
        "realized_target_legs": list(realized_legs),
        "realized_target_fraction": realized_fraction,
        "residual_fraction": max(0.0, 1.0 - realized_fraction),
        "pnl_pct": None,
    })
    return out


def prepare_counterfactual_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Normalize one symbol's bars once for repeated counterfactual calculations.

    The returned frame is detached from caller-owned data and can safely be
    reused for multiple observations on the same symbol/provider. ATR is based
    only on candles closed strictly before each candidate candle.
    """
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["timestamp", "close_time", "open", "high", "low", "close"])
    x = bars.copy()
    x["timestamp"] = _timestamp_series_to_utc(x["timestamp"])
    if "close_time" not in x.columns:
        x["close_time"] = x["timestamp"] + pd.Timedelta(minutes=5)
    else:
        x["close_time"] = _timestamp_series_to_utc(x["close_time"])
    for col in ("open", "high", "low", "close"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["timestamp", "close_time", "open", "high", "low", "close"]).copy()
    # When persisted history and a freshly fetched overlap are concatenated,
    # keep the last supplied version of each candle. Processing both would
    # count one market bar twice and distort the simulated path.
    x["_shadow_input_order"] = np.arange(len(x), dtype=np.int64)
    x = (
        x.sort_values(["timestamp", "_shadow_input_order"], kind="stable")
        .drop_duplicates("timestamp", keep="last")
        .drop(columns=["_shadow_input_order"])
        .reset_index(drop=True)
    )
    # Never trust a caller-supplied value for this internal acceleration field.
    x = x.drop(columns=[_SHADOW_ATR_BEFORE_CLOSE], errors="ignore")
    atr_before_close = _precompute_atr_before_close(x)
    if atr_before_close is not None:
        x[_SHADOW_ATR_BEFORE_CLOSE] = atr_before_close
    return x


def calculate_counterfactual_outcomes(
    observation: dict[str, Any],
    bars: pd.DataFrame,
    *,
    bars_prepared: bool = False,
) -> dict[str, Any]:
    """Calculate counterfactual paths; prepared frames may be reused by symbol.

    ``bars_prepared`` is an internal fast path for frames returned by
    :func:`prepare_counterfactual_bars`. It avoids re-copying, re-normalizing,
    and re-sorting the same market history for each observation.
    """
    shadow = ((observation.get("features") or {}).get("shadow_experiments") or {})
    # Production observations persist the complete snapshot envelope returned by
    # build_entry_snapshot(). Older tests/journals may contain only the flat
    # experiments mapping, so support both forms without mutating the source.
    if isinstance(shadow, dict) and isinstance(shadow.get("experiments"), dict):
        shadow = shadow["experiments"]
    direction = str(observation.get("direction", "")).upper()
    entry = _finite(observation.get("reference_price"))
    source_ts = _parse_ts(
        observation.get("counterfactual_path_start_ts")
        or observation.get("observation_ts")
        or observation.get("source_event_ts")
    )
    if direction not in {"LONG", "SHORT"} or entry is None or entry <= 0 or source_ts is None or bars is None or bars.empty:
        return {"schema_version": SHADOW_SCHEMA_VERSION, "status": "INSUFFICIENT_INPUT"}
    x = bars if bars_prepared else prepare_counterfactual_bars(bars)
    future = x.loc[(x["timestamp"] >= source_ts) & (x["close_time"] > source_ts)].copy().reset_index(drop=True)
    if future.empty:
        return {"schema_version": SHADOW_SCHEMA_VERSION, "status": "NO_FUTURE_PATH"}

    structural = shadow.get("CF_STRUCTURAL_SL") if isinstance(shadow, dict) else None
    rr = shadow.get("CF_TP_RR_15_30") if isinstance(shadow, dict) else None
    structural_sl = _finite((structural or {}).get("hypothetical_sl")) if isinstance(structural, dict) else None
    tp1 = _finite((rr or {}).get("hypothetical_tp1")) if isinstance(rr, dict) else None
    tp2 = _finite((rr or {}).get("hypothetical_tp2")) if isinstance(rr, dict) else None

    out: dict[str, Any] = {"schema_version": SHADOW_SCHEMA_VERSION, "status": "OK", "experiments": {}}
    if structural_sl is not None:
        if tp1 is not None:
            out["experiments"]["CF_STRUCTURAL_SL"] = _simulate_single_exit(future, direction=direction, entry=entry, stop=structural_sl, target=tp1)
        if tp2 is not None:
            out["experiments"]["CF_TP_RR_15_30"] = {
                "target_tp1": _simulate_single_exit(future, direction=direction, entry=entry, stop=structural_sl, target=tp1),
                "target_tp2": _simulate_single_exit(future, direction=direction, entry=entry, stop=structural_sl, target=tp2),
            }

    # These experiments intentionally model the production management baseline
    # independently of structural-stop availability.
    production = shadow.get("CF_PRODUCTION_MANAGEMENT") if isinstance(shadow, dict) else None
    production_stop = _finite((production or {}).get("stop")) if isinstance(production, dict) else None
    production_tp1 = _finite((production or {}).get("tp1")) if isinstance(production, dict) else None
    production_tp2 = _finite((production or {}).get("tp2")) if isinstance(production, dict) else None
    production_tp3 = _finite((production or {}).get("tp3")) if isinstance(production, dict) else None
    if production_stop is not None and production_tp1 is not None and production_tp2 is not None:
        for bars_delay in DELAYED_BE_BARS:
            out["experiments"][f"CF_DELAYED_BE_CURRENT_{bars_delay}BAR"] = _simulate_management_only(
                x, future, direction=direction, entry=entry, production_stop=production_stop,
                production_tp1=production_tp1, production_tp2=production_tp2, production_tp3=production_tp3,
                mode="DELAYED_BE", parameter=bars_delay,
            )
        for atr_mult in TRAIL_ATR_MULTIPLIERS:
            out["experiments"][f"CF_TRAIL_CURRENT_{str(atr_mult).replace('.', '_')}ATR"] = _simulate_management_only(
                x, future, direction=direction, entry=entry, production_stop=production_stop,
                production_tp1=production_tp1, production_tp2=production_tp2, production_tp3=production_tp3,
                mode="TRAIL_ATR", parameter=atr_mult,
            )
        for locked_profit_pct in MFE_PROTECTION_LOCK_LEVELS_PCT:
            token = (
                "MINUS_1_0" if locked_profit_pct == -1.0
                else "0_0" if locked_profit_pct == 0.0
                else "PLUS_0_5" if locked_profit_pct == 0.5
                else str(locked_profit_pct).replace("-", "MINUS_").replace("+", "PLUS_").replace(".", "_")
            )
            out["experiments"][f"CF_MFE_PLUS_2_LOCK_{token}"] = _simulate_mfe_protection(
                future, direction=direction, entry=entry, production_stop=production_stop,
                production_tp1=production_tp1, production_tp2=production_tp2, production_tp3=production_tp3,
                activation_mfe_pct=MFE_PROTECTION_ACTIVATION_PCT,
                locked_profit_pct=locked_profit_pct,
            )

    filter_specs = {
        "CF_VOLUME_5M": "1.5",
        "CF_VOLUME_1H": "1.5",
    }
    for experiment_id, threshold in filter_specs.items():
        exp = shadow.get(experiment_id) if isinstance(shadow, dict) else None
        out["experiments"][experiment_id] = {
            "would_have_executed": ((exp or {}).get("threshold_grid") or {}).get(threshold) if isinstance(exp, dict) else None,
            "forward_return_24h_pct": None,
        }
    return out
