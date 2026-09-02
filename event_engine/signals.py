from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Literal

import numpy as np
import pandas as pd

# =============================================================================
# Ajay R5.41 — parameters copied from the supplied Pine source
# =============================================================================
ALMA_TIMEFRAME_MINUTES = 480  # 1H chart * intRes(8) = 480m = 8H
ALMA_TIMEFRAME_HOURS = 8
ALMA_BASIS_TYPE = "ALMA"
ALMA_BASIS_LEN = 2
ALMA_SIGMA = 5.0
ALMA_OFFSET = 0.85
DELAY_OFFSET = 0
USE_ALTERNATE_SIGNALS = True
INT_RES = 8

SWING_LEN = 10
ZONE_HISTORY = 20
BOX_WIDTH = 2.5
RR_RATIO = 3.0
TP1_R = 1.0
TP2_R = 2.0
TP1_FRACTION = 0.50
TP2_FRACTION = 0.50

MIN_BARS = 70

# When an Ajay ALMA signal has no directional Demand/Supply zone touching the
# signal bar, production still needs a deterministic protective stop. This
# fallback is deliberately expressed in ATR rather than inventing a zone.
FALLBACK_SL_ATR_MULTIPLIER = float(os.environ.get("FALLBACK_SL_ATR_MULTIPLIER", "1.5"))

# Pine visual S/R settings (diagnostic layer; not an entry filter)
SR_ENABLE = True
SR_STRENGTH = 2
SR_RB = 10
SR_PRD = 284
SR_CHANNEL_W = 10
SR_LABEL_LOC = 55
SR_ZONE_WIDTH_PCT = 2
SR_USE_ZONES = True
SR_USE_HL_ZONES = True
SR_EXPAND = True
SR_SUPPORT_COLOR = "#00DBFF"
SR_RESISTANCE_COLOR = "#E91E63"



def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _wilder_rma(series: pd.Series, length: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if len(s) < length:
        return out
    first = s.iloc[:length].mean()
    out.iloc[length - 1] = first
    alpha = 1.0 / float(length)
    for i in range(length, len(s)):
        prev = out.iloc[i - 1]
        value = s.iloc[i]
        out.iloc[i] = prev + alpha * (value - prev) if pd.notna(value) else prev
    return out


def calc_atr(df: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _wilder_rma(tr, length).bfill()




def _wma(series: pd.Series, length: int) -> pd.Series:
    length = max(1, int(length))
    weights = np.arange(1, length + 1, dtype=float)
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return numeric.rolling(length, min_periods=length).apply(
        lambda x: float(np.dot(x, weights) / weights.sum()), raw=True
    )


def calc_hma(series: pd.Series, length: int) -> pd.Series:
    """Legacy helper retained only for backwards-compatible tests; not used by Ajay trigger."""
    length = max(1, int(length))
    half = max(1, length // 2)
    sqrt_len = max(1, int(round(math.sqrt(length))))
    return _wma(2.0 * _wma(series, half) - _wma(series, length), sqrt_len)


def calc_alma(series: pd.Series, length: int = 2, offset: float = 0.85, sigma: float = 5.0) -> pd.Series:
    """Port of TradingView ta.alma() with the Pine defaults from Ajay R5.41."""
    length = max(1, int(length))
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if length == 1:
        return numeric.copy()
    m = float(offset) * (length - 1)
    s = length / max(float(sigma), 1e-12)
    weights = np.array(
        [math.exp(-((i - m) ** 2) / (2.0 * s * s)) for i in range(length)],
        dtype=float,
    )
    weights /= weights.sum()
    return numeric.rolling(length, min_periods=length).apply(
        lambda x: float(np.dot(x, weights)), raw=True
    )


def _ema(series: pd.Series, length: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return numeric.ewm(span=int(length), adjust=False, min_periods=1).mean()


def compute_pine_keltner_channels(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the eight visible channel lines from Ajay R5.41.

    Pine: ta.kc(close, 80, multiplier), followed by ta.ema(..., 50).
    """
    out = _normalize_1h(df)
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    basis = _ema(out["close"], 80)
    span = _ema(tr, 80)
    bands = {}
    for name, mult in (("kc1", 10.5), ("kc2", 9.5), ("kc3", 8.0), ("kc4", 3.0)):
        upper = _ema(basis + span * mult, 50)
        lower = _ema(basis - span * mult, 50)
        bands[f"{name}_upper"] = upper
        bands[f"{name}_lower"] = lower
    return pd.DataFrame(bands, index=out.index)


def _make_event_id(symbol: str, direction: str, timestamp: int, zone_start: int | None, entry: float) -> str:
    raw = f"AJAY_R541:{symbol.upper()}:{direction.upper()}:{timestamp}:{zone_start if zone_start is not None else -1}:{entry:.12f}"
    return "ZONE_" + hashlib.sha256(raw.encode()).hexdigest().upper()[:24]


def _normalize_1h(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing candle columns: {sorted(missing)}")

    raw_ts = out["timestamp"]
    # Exchange APIs in this project return epoch milliseconds.  Calling
    # pd.to_datetime() without a unit silently interprets those integers as
    # nanoseconds, turning 2026 timestamps into dates around 1970.
    # Preserve already-datetime values, otherwise infer the epoch unit from
    # the magnitude of numeric timestamps.
    if pd.api.types.is_datetime64_any_dtype(raw_ts):
        out["timestamp"] = pd.to_datetime(raw_ts, utc=True, errors="coerce")
    else:
        numeric_ts = pd.to_numeric(raw_ts, errors="coerce")
        numeric_ratio = float(numeric_ts.notna().mean()) if len(numeric_ts) else 0.0
        if numeric_ratio >= 0.99:
            finite = numeric_ts.dropna().abs()
            magnitude = float(finite.median()) if not finite.empty else 0.0
            if magnitude >= 1e17:
                unit = "ns"
            elif magnitude >= 1e14:
                unit = "us"
            elif magnitude >= 1e11:
                unit = "ms"
            elif magnitude >= 1e8:
                unit = "s"
            else:
                unit = None
            if unit is not None:
                out["timestamp"] = pd.to_datetime(numeric_ts, unit=unit, utc=True, errors="coerce")
            else:
                out["timestamp"] = pd.to_datetime(raw_ts, utc=True, errors="coerce")
        else:
            out["timestamp"] = pd.to_datetime(raw_ts, utc=True, errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # A 1H candle sequence compressed into milliseconds is always a malformed
    # timestamp conversion, not valid market data. Fail loudly instead of
    # allowing all candles to land in one 8H bucket and suppress signals.
    if len(out) >= 3:
        deltas = out["timestamp"].diff().dropna()
        if not deltas.empty and deltas.median() < pd.Timedelta(minutes=30):
            raise ValueError(
                "Invalid 1H timestamp spacing after normalization: "
                f"median_delta={deltas.median()}"
            )
    return out


def _drop_incomplete_1h(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Ajay strategy has calc_on_every_tick=false; use only closed chart bars."""
    out = _normalize_1h(df)
    if out.empty:
        return out
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    last_ts = out["timestamp"].iloc[-1]
    # Binance timestamp is candle-open time. A 1H bar is complete one hour after start.
    if last_ts + pd.Timedelta(hours=1) > now:
        out = out.iloc[:-1].reset_index(drop=True)
    return out


def _aggregate_8h(df_1h: pd.DataFrame) -> pd.DataFrame:
    x = _normalize_1h(df_1h)
    x["bucket"] = x["timestamp"].dt.floor("8h")
    g = x.groupby("bucket", sort=True, observed=True)
    tf = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        bars=("close", "size"),
        last_ts=("timestamp", "last"),
    ).reset_index()
    return tf


def _attach_exact_alternate_series(df: pd.DataFrame, mode: Literal["historical", "live"] = "live") -> pd.DataFrame:
    """Build the 8H alternate series used by Pine's ``reso()``.

    Historical mode reproduces the historical ``lookahead_on`` mapping.

    Live mode reproduces the important Pine execution distinction for the latest
    realtime chart bar: historical chart bars receive only confirmed HTF values,
    while the current chart bar may receive the developing HTF value. For the
    current 8H bucket, therefore, every *previous* 1H bar uses the last confirmed
    8H ALMA and only the latest closed 1H bar uses the developing 8H ALMA.
    """
    out = _normalize_1h(df)
    tf = _aggregate_8h(out)
    tf["alma_close"] = calc_alma(tf["close"], ALMA_BASIS_LEN, ALMA_OFFSET, ALMA_SIGMA)
    tf["alma_open"] = calc_alma(tf["open"], ALMA_BASIS_LEN, ALMA_OFFSET, ALMA_SIGMA)

    # Allow callers/tests to pass a dataframe that was already enriched by a
    # previous trigger computation. Recompute trigger columns from raw OHLCV.
    out = out.drop(
        columns=["alma_close_alt", "alma_open_alt", "pine_buy", "pine_sell", "alternate_mode"],
        errors="ignore",
    ).copy()
    out["_bucket"] = out["timestamp"].dt.floor("8h")

    if mode == "historical":
        mapping = tf.set_index("bucket")[["alma_close", "alma_open"]]
        out = out.join(mapping, on="_bucket")
        out.rename(columns={"alma_close": "alma_close_alt", "alma_open": "alma_open_alt"}, inplace=True)
        out["alternate_mode"] = "historical_lookahead_on"
    else:
        # Start with the normal historical request.security mapping.
        tf_idx = tf.set_index("bucket")
        mapping = tf_idx[["alma_close", "alma_open"]]
        out = out.join(mapping, on="_bucket")
        out.rename(columns={"alma_close": "alma_close_alt", "alma_open": "alma_open_alt"}, inplace=True)

        latest_bucket = out["_bucket"].iloc[-1]
        current_tf = tf_idx.loc[latest_bucket]
        previous_buckets = tf_idx.loc[tf_idx.index < latest_bucket]

        # If the current bucket has a confirmed predecessor, historical bars
        # inside the current bucket must see that predecessor, not the developing
        # current HTF value. This is the key difference from the previous Python
        # implementation, which compared two developing states inside one 8H bar.
        if not previous_buckets.empty:
            prev_tf = previous_buckets.iloc[-1]
            confirmed_prev_close = float(prev_tf["close"])
            confirmed_prev_open = float(prev_tf["open"])
            # Pine ta.alma(length=2): weight[0] applies to the older sample and
            # weight[1] to the newer sample.
            m = ALMA_OFFSET * (ALMA_BASIS_LEN - 1)
            sigma_scale = ALMA_BASIS_LEN / ALMA_SIGMA
            weights = np.array(
                [
                    math.exp(-((i - m) ** 2) / (2.0 * sigma_scale * sigma_scale))
                    for i in range(ALMA_BASIS_LEN)
                ],
                dtype=float,
            )
            weights /= weights.sum()

            developing_close = weights[0] * confirmed_prev_close + weights[1] * float(current_tf["close"])
            developing_open = weights[0] * confirmed_prev_open + weights[1] * float(current_tf["open"])

            current_bucket_mask = out["_bucket"].eq(latest_bucket)
            # All prior chart bars in the current HTF bucket are historical at the
            # moment the latest 1H bar closes; they see only the last confirmed HTF
            # ALMA. The latest bar alone receives the developing HTF value.
            prev_confirmed_alma_close = float(prev_tf["alma_close"])
            prev_confirmed_alma_open = float(prev_tf["alma_open"])
            out.loc[current_bucket_mask, "alma_close_alt"] = prev_confirmed_alma_close
            out.loc[current_bucket_mask, "alma_open_alt"] = prev_confirmed_alma_open
            out.loc[out.index[-1], "alma_close_alt"] = developing_close
            out.loc[out.index[-1], "alma_open_alt"] = developing_open

        out["alternate_mode"] = "live_safe"

    out.drop(columns=["_bucket"], inplace=True, errors="ignore")
    out["pine_buy"] = (
        (out["alma_close_alt"] > out["alma_open_alt"])
        & (out["alma_close_alt"].shift(1) <= out["alma_open_alt"].shift(1))
    ).fillna(False)
    out["pine_sell"] = (
        (out["alma_close_alt"] < out["alma_open_alt"])
        & (out["alma_close_alt"].shift(1) >= out["alma_open_alt"].shift(1))
    ).fillna(False)
    return out


def _pine_overlap_check(new_poi: float, zones: list[dict[str, Any]], atr: float) -> bool:
    """Literal behavior of the supplied Pine f_check_overlapping()."""
    if not zones:
        return True
    atr_threshold = atr * 2.0
    okay = True
    for zone in zones:
        top = float(zone["top"])
        bottom = float(zone["btm"])
        poi = (top + bottom) / 2.0
        upper = poi + atr_threshold
        lower = poi - atr_threshold
        if new_poi >= lower and new_poi <= upper:
            okay = False
            break
        else:
            okay = True
    return okay


def _build_zone(top_or_bottom: float, box_type: int, atr: float, start: int) -> dict[str, Any]:
    buffer = atr * (BOX_WIDTH / 10.0)
    if box_type == 1:
        top = top_or_bottom
        bottom = top - buffer
    else:
        bottom = top_or_bottom
        top = bottom + buffer
    return {
        "top": top,
        "btm": bottom,
        "poi": (top + bottom) / 2.0,
        "start": int(start),
    }


def _pine_zone_walk(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    supply: list[dict[str, Any]] = []
    demand: list[dict[str, Any]] = []
    supply_bos: list[dict[str, Any]] = []
    demand_bos: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    n = len(df)
    for i in range(SWING_LEN * 2, n):
        p = i - SWING_LEN
        h = float(df.loc[p, "high"])
        l = float(df.loc[p, "low"])
        atr = max(float(df.loc[i, "atr50"]), 1e-12)

        is_ph = all(h >= float(df.loc[p-k, "high"]) for k in range(1, SWING_LEN+1)) and all(
            h >= float(df.loc[p+k, "high"]) for k in range(1, SWING_LEN+1)
        )
        is_pl = all(l <= float(df.loc[p-k, "low"]) for k in range(1, SWING_LEN+1)) and all(
            l <= float(df.loc[p+k, "low"]) for k in range(1, SWING_LEN+1)
        )

        # Pine is if swing_high, else if swing_low.
        if is_ph:
            zone = _build_zone(h, 1, atr, p)
            if _pine_overlap_check(zone["poi"], supply, atr):
                supply.insert(0, zone)
                del supply[ZONE_HISTORY:]
        elif is_pl:
            zone = _build_zone(l, -1, atr, p)
            if _pine_overlap_check(zone["poi"], demand, atr):
                demand.insert(0, zone)
                del demand[ZONE_HISTORY:]

        close = float(df.loc[i, "close"])
        # Logical representation of f_sd_to_bos(): broken boxes are removed
        # from the active list and an immutable BOS record is retained for plots/logs.
        kept_supply = []
        for z in supply:
            if close >= float(z["top"]):
                supply_bos.append({**z, "break_idx": i, "break_time": df.loc[i, "timestamp"].isoformat(), "type": "SUPPLY_BOS"})
            else:
                kept_supply.append(z)
        supply = kept_supply

        kept_demand = []
        for z in demand:
            if close <= float(z["btm"]):
                demand_bos.append({**z, "break_idx": i, "break_time": df.loc[i, "timestamp"].isoformat(), "type": "DEMAND_BOS"})
            else:
                kept_demand.append(z)
        demand = kept_demand

        snapshots.append(
            {
                "idx": i,
                "time": df.loc[i, "timestamp"].isoformat(),
                "supply": [dict(z) for z in supply],
                "demand": [dict(z) for z in demand],
            }
        )

    return supply, demand, supply_bos, demand_bos, snapshots


def _zone_context(zone: dict[str, Any], current_idx: int, df: pd.DataFrame) -> dict[str, Any]:
    start = int(zone["start"])
    move_idx = min(start + 6, current_idx)
    base = _safe_num(df.loc[start, "close"])
    end_close = _safe_num(df.loc[move_idx, "close"])
    atr = max(_safe_num(df.loc[start, "atr50"], 0.0), 1e-12)
    return {
        "start_idx": start,
        "age_bars": max(0, current_idx - start),
        "impulse_atr": round(abs(end_close - base) / atr, 3),
    }


def _find_directional_zone(direction: str, cur_l: float, cur_h: float, cur_c: float, demand: list[dict[str, Any]], supply: list[dict[str, Any]]) -> dict[str, Any] | None:
    if direction == "LONG":
        candidates = [z for z in demand if cur_l <= float(z["top"]) * 1.003 and cur_c >= float(z["btm"])]
    else:
        candidates = [z for z in supply if cur_h >= float(z["btm"]) * 0.997 and cur_c <= float(z["top"])]
    return candidates[0] if candidates else None


def _nearest_zone(direction: str, cur_c: float, demand: list[dict[str, Any]], supply: list[dict[str, Any]]) -> dict[str, Any] | None:
    zones = demand if direction == "LONG" else supply
    return min(zones, key=lambda z: abs(cur_c - float(z["poi"])), default=None)



def _pivot_series_pine(df: pd.DataFrame, rb: int = SR_RB) -> tuple[pd.Series, pd.Series]:
    """Series of pivot prices at their Pine confirmation bar (not pivot bar)."""
    x = _normalize_1h(df)
    n = len(x)
    ph = pd.Series(np.nan, index=x.index, dtype=float)
    pl = pd.Series(np.nan, index=x.index, dtype=float)
    for confirm in range(2 * rb, n):
        p = confirm - rb
        h = float(x.loc[p, "high"])
        l = float(x.loc[p, "low"])
        left_h = x.loc[p-rb:p-1, "high"].to_numpy(dtype=float)
        right_h = x.loc[p+1:p+rb, "high"].to_numpy(dtype=float)
        left_l = x.loc[p-rb:p-1, "low"].to_numpy(dtype=float)
        right_l = x.loc[p+1:p+rb, "low"].to_numpy(dtype=float)
        if np.all(h >= left_h) and np.all(h >= right_h):
            ph.iloc[confirm] = h
        if np.all(l <= left_l) and np.all(l <= right_l):
            pl.iloc[confirm] = l
    return ph, pl


def _pine_sr_event_levels(x: pd.DataFrame, event_i: int, ph: pd.Series, pl: pd.Series) -> dict[str, Any]:
    """Recreate the supplied Pine SR clustering at one bar where ph/pl fires."""
    start = max(0, event_i - SR_PRD + 1)
    window = x.iloc[start:event_i + 1]
    window_high = float(window["high"].max())
    window_low = float(window["low"].min())
    cwidth = (window_high - window_low) * SR_CHANNEL_W / 100.0
    zone_start = max(0, event_i - 300 + 1)
    zone_window = x.iloc[zone_start:event_i + 1]
    zone_perc = (float(zone_window["high"].max()) - float(zone_window["low"].min())) * SR_ZONE_WIDTH_PCT / 100.0

    pivots = []
    for xx in range(0, SR_PRD + 1):
        idx = event_i - xx
        if idx < 0:
            break
        if pd.notna(ph.iloc[idx]):
            pivots.append((idx, float(ph.iloc[idx]), "H"))
        if pd.notna(pl.iloc[idx]):
            pivots.append((idx, float(pl.iloc[idx]), "L"))
        if len(pivots) >= 41:
            break

    highestph = window_low
    lowestpl = window_high
    for _, price, _ in pivots:
        highestph = max(highestph, price)
        lowestpl = min(lowestpl, price)

    aas = [True] * 41
    sr_levels = [None] * 21
    countpp = 0
    for _, pivot_price, ptype in pivots:
        countpp += 1
        if countpp > 40:
            break
        if not aas[countpp]:
            continue
        upl = pivot_price + cwidth
        dnl = pivot_price - cwidth
        tmp = [True] * 41
        tpoint = 0
        cnt = 0
        for _, price2, _ in pivots:
            cnt += 1
            if cnt > 40:
                break
            if aas[cnt] and dnl <= price2 <= upl:
                tpoint += 1
                tmp[cnt] = False
        if tpoint >= SR_STRENGTH:
            for g in range(41):
                if not tmp[g]:
                    aas[g] = False
            if countpp < 21:
                sr_levels[countpp] = pivot_price

    levels = [v for v in sr_levels if v is not None and math.isfinite(v)]
    return {
        "levels": levels,
        "highestph": highestph,
        "lowestpl": lowestpl,
        "cwidth": cwidth,
        "zonePerc": zone_perc,
        "event_idx": event_i,
    }


def compute_pine_sr_visual(df: pd.DataFrame) -> dict[str, Any]:
    """Visual/diagnostic reproduction of the supplied Pine Ajay R5.41 S/R block."""
    x = _normalize_1h(df)
    if len(x) < SR_PRD:
        return {"levels": [], "highestph": None, "lowestpl": None, "cwidth": None, "zonePerc": None, "event_idx": None, "events": []}
    ph, pl = _pivot_series_pine(x, SR_RB)
    current = None
    events = []
    for i in range(len(x)):
        if pd.notna(ph.iloc[i]) or pd.notna(pl.iloc[i]):
            current = _pine_sr_event_levels(x, i, ph, pl)
            current["timestamp"] = x.iloc[i]["timestamp"]
            events.append(dict(current))
    if current is None:
        current = {"levels": [], "highestph": None, "lowestpl": None, "cwidth": None, "zonePerc": None, "event_idx": None}
    current = dict(current)
    current["events"] = events
    current["ph"] = ph
    current["pl"] = pl
    return current

def compute_pine_zone_records(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Return historical Supply/Demand box lifetimes and BOS records for plotting."""
    x = _normalize_1h(df)
    if len(x) < MIN_BARS:
        return {"supply": [], "demand": [], "supply_bos": [], "demand_bos": []}
    x["atr50"] = calc_atr(x, 50)
    supply, demand, supply_bos, demand_bos, snapshots = _pine_zone_walk(x)
    records: dict[str, dict[tuple, dict[str, Any]]] = {"supply": {}, "demand": {}}
    for snap in snapshots:
        idx = int(snap["idx"])
        for kind, key in (("supply", "supply"), ("demand", "demand")):
            for z in snap[key]:
                ident = (kind, int(z["start"]), round(float(z["top"]), 12), round(float(z["btm"]), 12))
                rec = records[kind].get(ident)
                if rec is None:
                    rec = {**z, "kind": kind.upper(), "active_from_idx": idx, "end_idx": None}
                    records[kind][ident] = rec
    for bos in supply_bos:
        ident = ("supply", int(bos["start"]), round(float(bos["top"]), 12), round(float(bos["btm"]), 12))
        if ident in records["supply"]:
            records["supply"][ident]["end_idx"] = int(bos["break_idx"])
    for bos in demand_bos:
        ident = ("demand", int(bos["start"]), round(float(bos["top"]), 12), round(float(bos["btm"]), 12))
        if ident in records["demand"]:
            records["demand"][ident]["end_idx"] = int(bos["break_idx"])
    return {
        "supply": list(records["supply"].values()),
        "demand": list(records["demand"].values()),
        "supply_bos": supply_bos,
        "demand_bos": demand_bos,
    }


def compute_ajay_trigger(df: pd.DataFrame, mode: Literal["historical", "live"] = "live") -> pd.DataFrame:
    """Return 1H data with Ajay R5.41 alternate-timeframe ALMA trigger fields."""
    out = _drop_incomplete_1h(df)
    return _attach_exact_alternate_series(out, mode=mode)


def generate_zone_signals(
    df: pd.DataFrame,
    symbol: str = "",
    mode: Literal["historical", "live"] = "live",
) -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    """Ajay R5.41 strategy core plus Supply/Demand context.

    The strategy trigger is ONLY the ALMA crossover from the supplied Pine.
    Zones are reconstructed independently and used for context/SL planning.
    """
    df = _drop_incomplete_1h(df)
    if len(df) < MIN_BARS:
        return df, [], [], []

    df["atr50"] = calc_atr(df, 50)
    df["atr30"] = calc_atr(df, 30)
    df["vol_sma20"] = df["volume"].rolling(20, min_periods=5).mean().bfill()
    df["body_atr"] = (df["close"] - df["open"]).abs() / df["atr50"].replace(0, np.nan)
    df["range_atr"] = (df["high"] - df["low"]) / df["atr50"].replace(0, np.nan)

    df = _attach_exact_alternate_series(df, mode=mode)
    _, _, _, _, snapshots = _pine_zone_walk(df)

    active_supply: list[dict[str, Any]] = []
    active_demand: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []

    for i in range(SWING_LEN * 2, len(df)):
        p = i - SWING_LEN
        h = float(df.loc[p, "high"])
        l = float(df.loc[p, "low"])
        atr = max(float(df.loc[i, "atr50"]), 1e-12)

        is_ph = all(h >= float(df.loc[p-k, "high"]) for k in range(1, SWING_LEN+1)) and all(
            h >= float(df.loc[p+k, "high"]) for k in range(1, SWING_LEN+1)
        )
        is_pl = all(l <= float(df.loc[p-k, "low"]) for k in range(1, SWING_LEN+1)) and all(
            l <= float(df.loc[p+k, "low"]) for k in range(1, SWING_LEN+1)
        )
        if is_ph:
            zone = _build_zone(h, 1, atr, p)
            if _pine_overlap_check(zone["poi"], active_supply, atr):
                active_supply.insert(0, zone)
                del active_supply[ZONE_HISTORY:]
        elif is_pl:
            zone = _build_zone(l, -1, atr, p)
            if _pine_overlap_check(zone["poi"], active_demand, atr):
                active_demand.insert(0, zone)
                del active_demand[ZONE_HISTORY:]

        close = float(df.loc[i, "close"])
        active_supply = [z for z in active_supply if close < float(z["top"])]
        active_demand = [z for z in active_demand if close > float(z["btm"])]

        direction = "LONG" if bool(df.loc[i, "pine_buy"]) else "SHORT" if bool(df.loc[i, "pine_sell"]) else None
        if not direction:
            continue

        cur_l, cur_h, cur_c, cur_o = map(float, [df.loc[i, "low"], df.loc[i, "high"], df.loc[i, "close"], df.loc[i, "open"]])
        trade_zone = _find_directional_zone(direction, cur_l, cur_h, cur_c, active_demand, active_supply)
        context_zone = trade_zone or _nearest_zone(direction, cur_c, active_demand, active_supply)

        stop = risk = tp1 = tp2 = risk_pct = None
        sl_source = None
        if trade_zone is not None:
            if direction == "LONG":
                stop = float(trade_zone["btm"]) - 0.3 * atr
                risk = cur_c - stop
            else:
                stop = float(trade_zone["top"]) + 0.3 * atr
                risk = stop - cur_c
            sl_source = "zone"
        else:
            # Pine's strategy.entry() does not require a Demand/Supply touch.
            # Therefore a valid ALMA signal must remain a valid production signal
            # even when there is no directional zone on the trigger bar. Use a
            # deterministic ATR stop only for risk construction; the zone itself
            # remains context, not an entry gate.
            fallback_risk = max(FALLBACK_SL_ATR_MULTIPLIER * atr, cur_c * 0.0001)
            if direction == "LONG":
                stop = cur_c - fallback_risk
                risk = fallback_risk
            else:
                stop = cur_c + fallback_risk
                risk = fallback_risk
            sl_source = "atr_fallback"

        if risk is not None and risk > 0 and cur_c > 0:
            risk_pct = (risk / cur_c) * 100.0
            tp1 = cur_c + TP1_R * risk if direction == "LONG" else cur_c - TP1_R * risk
            tp2 = cur_c + TP2_R * risk if direction == "LONG" else cur_c - TP2_R * risk
        else:
            stop = risk = tp1 = tp2 = risk_pct = None

        avg_vol = _safe_num(df.loc[i, "vol_sma20"], 0.0)
        vol_ratio = (_safe_num(df.loc[i, "volume"]) / avg_vol) if avg_vol > 0 else None
        event_ts = int(df.loc[i, "timestamp"].timestamp() * 1000)
        zone_start = int(context_zone["start"]) if context_zone is not None else None
        event_id = _make_event_id(symbol or "UNKNOWN", direction, event_ts, zone_start, cur_c)
        zone_ctx = {}
        if context_zone:
            zone_ctx = {
                **context_zone,
                **_zone_context(context_zone, i, df),
                "kind": "DEMAND" if direction == "LONG" else "SUPPLY",
            }

        signals.append(
            {
                "event_id": event_id,
                "idx": i,
                "time": df.loc[i, "timestamp"].isoformat(),
                "type": direction,
                "symbol": symbol.upper(),
                "entry": cur_c,
                "sl": round(stop, 8) if stop is not None else None,
                "tp1": round(tp1, 8) if tp1 is not None else None,
                "tp2": round(tp2, 8) if tp2 is not None else None,
                "risk_pct": round(risk_pct, 4) if risk_pct is not None else None,
                "risk_abs": round(risk, 8) if risk is not None else None,
                "rr_ratio": RR_RATIO,
                "strategy": "Ajay R5.41",
                "trigger": {
                    "type": "ALMA_CROSS",
                    "basis_type": ALMA_BASIS_TYPE,
                    "basis_len": ALMA_BASIS_LEN,
                    "offset": ALMA_OFFSET,
                    "sigma": ALMA_SIGMA,
                    "alternate_timeframe": "8h",
                    "mode": mode,
                    "buy": bool(df.loc[i, "pine_buy"]),
                    "sell": bool(df.loc[i, "pine_sell"]),
                    "alma_close_alt": float(df.loc[i, "alma_close_alt"]),
                    "alma_open_alt": float(df.loc[i, "alma_open_alt"]),
                    "signal_bar_timestamp": df.loc[i, "timestamp"].isoformat(),
                },
                "zone": zone_ctx,
                "risk_model": {
                    "sl_source": sl_source,
                    "fallback_atr_multiplier": FALLBACK_SL_ATR_MULTIPLIER if sl_source == "atr_fallback" else None,
                },
                "confirmation": {
                    "alma_cross": True,
                    "zone_touch": trade_zone is not None,
                    "volume_ratio": round(vol_ratio, 3) if vol_ratio is not None else None,
                    "candle_body_atr": round(_safe_num(df.loc[i, "body_atr"]), 3),
                    "range_atr": round(_safe_num(df.loc[i, "range_atr"]), 3),
                    "bullish_candle": cur_c >= cur_o,
                    "bearish_candle": cur_c <= cur_o,
                },
                "source_bar_close": cur_c,
            }
        )

    return df, active_supply, active_demand, signals


def score_zone_signal(signal: dict[str, Any]) -> float:
    """Research-only score; never changes the Ajay trigger."""
    confirmation = signal.get("confirmation", {})
    zone = signal.get("zone", {})
    score = 50.0
    if confirmation.get("alma_cross"):
        score += 20.0
    if confirmation.get("zone_touch"):
        score += 10.0
    vol_ratio = _safe_num(confirmation.get("volume_ratio"), 0.0)
    if vol_ratio >= 2.0:
        score += 10.0
    elif vol_ratio >= 1.2:
        score += 5.0
    body = _safe_num(confirmation.get("candle_body_atr"), 0.0)
    if body >= 0.8:
        score += 5.0
    if int(zone.get("age_bars", 9999) or 9999) <= 36:
        score += 5.0
    return min(score, 100.0)
