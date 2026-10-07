from __future__ import annotations

import math
import numbers
from datetime import timezone
from typing import Any

import pandas as pd


SHADOW_SCHEMA_VERSION = 1
SHADOW_EXPERIMENT_VERSION = "counterfactual-v2"

VOLUME_5M_THRESHOLDS = (0.8, 1.0, 1.2, 1.5, 1.8, 2.0)
VOLUME_1H_THRESHOLDS = (0.8, 1.0, 1.2, 1.5, 1.8, 2.0)
ATR_5M_MIN_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)
TRAIL_ATR_MULTIPLIERS = (0.75, 1.0, 1.5)
DELAYED_BE_BARS = (1, 2)
PRODUCTION_TP1_FRACTION = 0.50
PRODUCTION_STOP_PCT = 10.0
PRODUCTION_TP1_PCT = 3.0
PRODUCTION_TP2_PCT = 6.0
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


def _atr14_5m(x: pd.DataFrame) -> float | None:
    if len(x) < 2:
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
    x1 = _closed_1h(df_1h, decision_ts_parsed)
    trigger_bar = x5.iloc[-1] if not x5.empty else None
    prev5 = x5.iloc[:-1].tail(20) if len(x5) >= 21 else pd.DataFrame()
    vol5 = _finite(trigger_bar["volume"]) if trigger_bar is not None else None
    baseline5 = _finite(prev5["volume"].mean()) if not prev5.empty else None
    volume_5m_ratio = _ratio(vol5, baseline5)

    last1h = x1.iloc[-1] if not x1.empty else None
    prev1h = x1.iloc[:-1].tail(20) if len(x1) >= 21 else pd.DataFrame()
    vol1h = _finite(last1h["volume"]) if last1h is not None else None
    baseline1h = _finite(prev1h["volume"].mean()) if not prev1h.empty else None
    volume_1h_ratio = _ratio(vol1h, baseline1h)

    atr5 = _atr14_5m(x5)
    atr5_pct = (atr5 / entry * 100.0) if atr5 is not None and entry > 0 else None
    delta = _finite(trigger_bar.get("bar_delta_usdt")) if trigger_bar is not None and hasattr(trigger_bar, "get") else None
    delta_confirmed = None if delta is None else (delta > 0 if direction == "LONG" else delta < 0)

    ema200 = None
    ema200_distance_pct = None
    ema200_trend_up = None
    if len(x1) >= 200:
        closes = x1["close"]
        ema_series = closes.ewm(span=200, adjust=False, min_periods=200).mean()
        ema200 = _finite(ema_series.iloc[-1])
        if ema200 is not None:
            ema200_distance_pct = (entry / ema200 - 1.0) * 100.0
            if len(ema_series) >= 2 and pd.notna(ema_series.iloc[-2]):
                ema200_trend_up = bool(float(ema_series.iloc[-1]) > float(ema_series.iloc[-2]))
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
        prior_bars = x5.iloc[:-1]
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
                "threshold_grid": _grid(volume_5m_ratio, VOLUME_5M_THRESHOLDS),
            },
            "CF_VOLUME_1H": {
                "version": 1,
                "current_closed_1h_volume": vol1h,
                "baseline_sma20_previous_bars": baseline1h,
                "ratio": volume_1h_ratio,
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
                "rule": "LONG price>EMA200 / SHORT price<EMA200",
                "would_pass": macro_alignment,
            },
            "CF_ATR_MIN_5M": {
                "version": 1,
                "atr14_5m": atr5,
                "atr_pct": atr5_pct,
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
                "version": 1,
                "stop": production_sl,
                "tp1": production_tp1,
                "tp2": production_tp2,
                "tp1_fraction": PRODUCTION_TP1_FRACTION,
                "management_baseline": "current_production_SL_TP1_TP2_BE_after_TP1",
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
                "tp1_fraction": PRODUCTION_TP1_FRACTION,
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
                "tp1_fraction": PRODUCTION_TP1_FRACTION,
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
    for idx, row in future.reset_index(drop=True).iterrows():
        event, ambiguous = _stop_or_target_hit(direction, float(row["high"]), float(row["low"]), stop, target)
        if event is None:
            continue
        result["exit"] = event
        result["ambiguous"] = ambiguous
        result["exit_price"] = target if event == "TARGET" else stop if event == "STOP" else None
        result["exit_ts"] = _parse_ts(row["close_time"]).isoformat() if _parse_ts(row["close_time"]) is not None else None
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


def _simulate_management_only(
    all_bars: pd.DataFrame,
    future: pd.DataFrame,
    *,
    direction: str,
    entry: float,
    production_stop: float | None,
    production_tp1: float | None,
    production_tp2: float | None,
    mode: str,
    parameter: float | int,
) -> dict[str, Any]:
    """Simulate only post-entry management changes against current production entry/SL/TP.

    Before TP1, production SL and TP1 are unchanged. After TP1, 50% is realized at TP1;
    only the residual 50% is managed by delayed-BE or ATR trailing. This isolates management
    effects from structural-SL/1.5R experiments and returns weighted trade-level PnL.
    """
    if future.empty or production_stop is None or production_tp1 is None or production_tp2 is None:
        return {"status": "INSUFFICIENT_PATH", "pnl_pct": None, "exit": None, "standalone_management_only": True}

    tp1_hit_idx: int | None = None
    tp1_fill_ts: str | None = None
    tp1_pnl_pct = _pct_return(direction, entry, production_tp1)

    future = future.reset_index(drop=True)
    all_bars = all_bars.copy().reset_index(drop=True)
    all_bars["timestamp"] = _timestamp_series_to_utc(all_bars["timestamp"])
    all_bars["close_time"] = _timestamp_series_to_utc(all_bars["close_time"])
    for col in ("open", "high", "low", "close"):
        all_bars[col] = pd.to_numeric(all_bars[col], errors="coerce")

    for idx, row in future.iterrows():
        high = float(row["high"])
        low = float(row["low"])
        stop_hit = _directional_hit(direction, high, low, float(production_stop), False)
        tp1_hit = _directional_hit(direction, high, low, float(production_tp1), True)
        tp2_hit = _directional_hit(direction, high, low, float(production_tp2), True)
        if stop_hit and (tp1_hit or tp2_hit):
            return {
                "status": "AMBIGUOUS_PRE_TP1_BAR",
                "exit": "AMBIGUOUS",
                "exit_price": None,
                "exit_ts": _parse_ts(row["close_time"]).isoformat() if _parse_ts(row["close_time"]) is not None else None,
                "pnl_pct": None,
                "standalone_management_only": True,
            }
        if stop_hit:
            stop_pnl = _pct_return(direction, entry, float(production_stop))
            return {
                "status": "STOP_BEFORE_TP1",
                "exit": "PRODUCTION_SL",
                "exit_price": float(production_stop),
                "tp1_pnl_pct": None,
                "residual_exit_pnl_pct": stop_pnl,
                "pnl_pct": stop_pnl,
                "bars_to_exit": idx + 1,
                "standalone_management_only": True,
            }
        if tp2_hit:
            weighted = _weighted_partial_exit_pnl(direction, entry, production_tp1, production_tp2)
            return {
                "status": "TP1_AND_TP2_REACHED_SAME_BAR",
                "exit": "PRODUCTION_TP2",
                "exit_price": float(production_tp2),
                "tp1_pnl_pct": tp1_pnl_pct,
                "residual_exit_pnl_pct": _pct_return(direction, entry, production_tp2),
                "pnl_pct": weighted,
                "bars_to_exit": idx + 1,
                "standalone_management_only": True,
            }
        if tp1_hit:
            tp1_hit_idx = int(idx)
            tp1_fill_ts = _parse_ts(row["close_time"]).isoformat() if _parse_ts(row["close_time"]) is not None else None
            break

    if tp1_hit_idx is None:
        return {
            "status": "PATH_NOT_REACHED_TP1",
            "exit": None,
            "exit_price": None,
            "pnl_pct": None,
            "standalone_management_only": True,
        }

    activation_idx = tp1_hit_idx + int(parameter) + 1 if mode == "DELAYED_BE" else tp1_hit_idx + 1
    current_stop = float(production_stop)
    highest = None
    lowest = None

    for idx in range(tp1_hit_idx + 1, len(future)):
        row = future.iloc[idx]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])

        if mode == "DELAYED_BE" and idx >= activation_idx:
            current_stop = float(entry)
        elif mode == "DELAYED_BE":
            current_stop = float(production_stop)

        current_close_ts = _parse_ts(row["close_time"])
        if mode == "TRAIL_ATR":
            # No intrabar look-ahead: build the trail only from bars that closed
            # before the current bar. The current bar can trigger the already-set stop,
            # but cannot retroactively move that stop using its own high/low.
            hist = all_bars.loc[all_bars["close_time"] < current_close_ts]
            if len(hist) >= 15:
                prev = hist["close"].shift(1)
                tr = pd.concat([
                    hist["high"] - hist["low"],
                    (hist["high"] - prev).abs(),
                    (hist["low"] - prev).abs(),
                ], axis=1).max(axis=1)
                atr_vals = tr.dropna().tail(14)
                atr = _finite(atr_vals.mean()) if len(atr_vals) == 14 else None
            else:
                atr = None
            if atr is not None:
                if direction == "LONG":
                    highest = max(float(highest), float(production_tp1)) if highest is not None else float(production_tp1)
                    current_stop = max(current_stop, highest - atr * float(parameter))
                else:
                    lowest = min(float(lowest), float(production_tp1)) if lowest is not None else float(production_tp1)
                    current_stop = min(current_stop, lowest + atr * float(parameter))

        stop_hit = _directional_hit(direction, high, low, current_stop, False)
        target_hit = mode == "DELAYED_BE" and _directional_hit(direction, high, low, production_tp2, True)
        if stop_hit and target_hit:
            return {
                "status": "AMBIGUOUS_POST_TP1_BAR",
                "exit": "AMBIGUOUS",
                "exit_price": None,
                "tp1_pnl_pct": tp1_pnl_pct,
                "residual_exit_pnl_pct": None,
                "pnl_pct": None,
                "tp1_fill_ts": tp1_fill_ts,
                "standalone_management_only": True,
            }
        if target_hit:
            weighted = _weighted_partial_exit_pnl(direction, entry, production_tp1, production_tp2)
            return {
                "status": "TP2_AFTER_TP1",
                "exit": "PRODUCTION_TP2",
                "exit_price": float(production_tp2),
                "tp1_pnl_pct": tp1_pnl_pct,
                "residual_exit_pnl_pct": _pct_return(direction, entry, production_tp2),
                "pnl_pct": weighted,
                "tp1_fill_ts": tp1_fill_ts,
                "exit_ts": current_close_ts.isoformat(),
                "bars_to_exit": idx + 1,
                "standalone_management_only": True,
            }
        if stop_hit:
            weighted = _weighted_partial_exit_pnl(direction, entry, production_tp1, current_stop)
            return {
                "status": "EXIT_AFTER_TP1",
                "exit": mode,
                "exit_price": float(current_stop),
                "tp1_pnl_pct": tp1_pnl_pct,
                "residual_exit_pnl_pct": _pct_return(direction, entry, current_stop),
                "pnl_pct": weighted,
                "tp1_fill_ts": tp1_fill_ts,
                "exit_ts": current_close_ts.isoformat(),
                "bars_to_exit": idx + 1,
                "standalone_management_only": True,
            }

        # Only a completed bar may move a trailing extreme used by the next bar.
        if mode == "TRAIL_ATR":
            if direction == "LONG":
                highest = high if highest is None else max(float(highest), high)
            else:
                lowest = low if lowest is None else min(float(lowest), low)

    return {
        "status": "PATH_NOT_EXITED",
        "exit": None,
        "exit_price": None,
        "tp1_pnl_pct": tp1_pnl_pct,
        "residual_exit_pnl_pct": None,
        "pnl_pct": None,
        "tp1_fill_ts": tp1_fill_ts,
        "standalone_management_only": True,
    }

def calculate_counterfactual_outcomes(observation: dict[str, Any], bars: pd.DataFrame) -> dict[str, Any]:
    shadow = ((observation.get("features") or {}).get("shadow_experiments") or {})
    direction = str(observation.get("direction", "")).upper()
    entry = _finite(observation.get("reference_price"))
    source_ts = _parse_ts(
        observation.get("counterfactual_path_start_ts")
        or observation.get("observation_ts")
        or observation.get("source_event_ts")
    )
    if direction not in {"LONG", "SHORT"} or entry is None or entry <= 0 or source_ts is None or bars is None or bars.empty:
        return {"schema_version": SHADOW_SCHEMA_VERSION, "status": "INSUFFICIENT_INPUT"}
    x = bars.copy()
    x["timestamp"] = _timestamp_series_to_utc(x["timestamp"])
    if "close_time" not in x.columns:
        x["close_time"] = x["timestamp"] + pd.Timedelta(minutes=5)
    else:
        x["close_time"] = _timestamp_series_to_utc(x["close_time"])
    for col in ("open", "high", "low", "close"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["timestamp", "close_time", "open", "high", "low", "close"]).sort_values("timestamp")
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
    if production_stop is not None and production_tp1 is not None and production_tp2 is not None:
        for bars_delay in DELAYED_BE_BARS:
            out["experiments"][f"CF_DELAYED_BE_CURRENT_{bars_delay}BAR"] = _simulate_management_only(
                x, future, direction=direction, entry=entry, production_stop=production_stop,
                production_tp1=production_tp1, production_tp2=production_tp2, mode="DELAYED_BE", parameter=bars_delay,
            )
        for atr_mult in TRAIL_ATR_MULTIPLIERS:
            out["experiments"][f"CF_TRAIL_CURRENT_{str(atr_mult).replace('.', '_')}ATR"] = _simulate_management_only(
                x, future, direction=direction, entry=entry, production_stop=production_stop,
                production_tp1=production_tp1, production_tp2=production_tp2, mode="TRAIL_ATR", parameter=atr_mult,
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
