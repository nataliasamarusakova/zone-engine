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
TP1_R = float(os.environ.get("TP1_R", "0.5"))  # fallback only
TP2_R = float(os.environ.get("TP2_R", "1.0"))  # fallback only
TP1_FRACTION = 0.50
TP2_FRACTION = 0.50
ZONE_SL_ATR_BUFFER = float(os.environ.get("ZONE_SL_ATR_BUFFER", "0.10"))
REQUIRE_ZONE_TOUCH = os.environ.get("REQUIRE_ZONE_TOUCH", "true").lower() == "true"
TP1_OBSTACLE_FRACTION = float(os.environ.get("TP1_OBSTACLE_FRACTION", "0.50"))
TP2_OBSTACLE_FRACTION = float(os.environ.get("TP2_OBSTACLE_FRACTION", "0.90"))
TP_OBSTACLE_BUFFER_ATR = float(os.environ.get("TP_OBSTACLE_BUFFER_ATR", "0.10"))
TP_MAX_R = float(os.environ.get("TP_MAX_R", "1.50"))
TP_MIN_R = float(os.environ.get("TP_MIN_R", "0.50"))

MIN_BARS = 70
# Production entry filters selected from the last completed audit. Keep them
# explicit and small so their effect remains observable in the new trade set.
MAX_ZONE_AGE_BARS = int(os.environ.get("MAX_ZONE_AGE_BARS", "30"))
MAX_SIGNAL_RISK_PCT = min(float(os.environ.get("MAX_SIGNAL_RISK_PCT", "1.50")), 1.50)
MIN_STRUCTURE_ROOM_R = float(os.environ.get("MIN_STRUCTURE_ROOM_R", "1.20"))
REQUIRE_DIRECTIONAL_CANDLE = os.environ.get("REQUIRE_DIRECTIONAL_CANDLE", "true").lower() == "true"
REQUIRE_STRUCTURE_OBSTACLE = os.environ.get("REQUIRE_STRUCTURE_OBSTACLE", "true").lower() == "true"

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
    return _wilder_rma(tr, length)




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
    # TradingView ta.alma() applies the ALMA weights from the OLDEST sample
    # in the window to the NEWEST sample. With pandas rolling(), x[0] is the
    # oldest value and x[-1] is the newest, so this dot product matches Pine.
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
    """Attach Ajay R5.41 ``request.security(..., 480, ..., lookahead_on)`` values.

    ``historical`` reproduces TradingView's stable historical HTF mapping: the
    final 8H ALMA value is repeated on every 1H bar belonging to that 8H bucket.

    ``live`` reproduces the important realtime distinction without corrupting
    historical bars: every *completed* 8H bucket uses the historical mapping,
    while only the current 8H bucket is reconstructed as a developing HTF bar.
    Thus, for the current 8H bucket, each closed 1H bar gets its own developing
    ALMA state and consecutive bars inside that bucket can legitimately cross.
    Earlier 8H buckets are never rebuilt as developing bars, which prevents the
    large number of synthetic historical BUY/SELL markers seen in earlier builds.
    """
    out = _normalize_1h(df)
    out = out.drop(
        columns=["alma_close_alt", "alma_open_alt", "pine_buy", "pine_sell", "alternate_mode"],
        errors="ignore",
    ).copy()
    out["_bucket"] = out["timestamp"].dt.floor("8h")

    tf = _aggregate_8h(out)
    tf["alma_close"] = calc_alma(tf["close"], ALMA_BASIS_LEN, ALMA_OFFSET, ALMA_SIGMA)
    tf["alma_open"] = calc_alma(tf["open"], ALMA_BASIS_LEN, ALMA_OFFSET, ALMA_SIGMA)
    tf_idx = tf.set_index("bucket")

    # Start from the exact historical lookahead_on mapping for every bar.
    mapped = tf_idx[["alma_close", "alma_open"]]
    out = out.join(mapped, on="_bucket")
    out.rename(columns={"alma_close": "alma_close_alt", "alma_open": "alma_open_alt"}, inplace=True)

    if mode == "historical":
        out["alternate_mode"] = "historical_lookahead_on"
    else:
        # Only the latest available 8H bucket is still developing.  Keep every
        # prior bucket exactly as TradingView historical request.security() does.
        latest_bucket = out["_bucket"].iloc[-1]
        mask = out["_bucket"].eq(latest_bucket)

        m = ALMA_OFFSET * (ALMA_BASIS_LEN - 1)
        sigma_scale = ALMA_BASIS_LEN / ALMA_SIGMA
        weights = np.array(
            [
                math.exp(-((j - m) ** 2) / (2.0 * sigma_scale * sigma_scale))
                for j in range(ALMA_BASIS_LEN)
            ],
            dtype=float,
        )
        weights /= weights.sum()

        # The current 8H ALMA(length=2) uses the previous confirmed 8H value
        # and the current developing 8H value.  For openSeries the current 8H
        # open is fixed for the entire bucket; for closeSeries it is the most
        # recent 1H close at each closed 1H bar.
        buckets = out["_bucket"]
        unique_buckets = list(tf_idx.index)
        if len(unique_buckets) >= 2:
            prev_bucket = unique_buckets[-2]
            prev_row = tf_idx.loc[prev_bucket]
            prev_close = float(prev_row["close"])
            prev_open = float(prev_row["open"])

            current_rows = out.loc[mask].copy()
            current_close = pd.to_numeric(current_rows["close"], errors="coerce").to_numpy(dtype=float)
            current_open = float(current_rows["open"].iloc[0])

            live_close = weights[0] * prev_close + weights[1] * current_close
            live_open = np.full(len(current_rows), weights[0] * prev_open + weights[1] * current_open, dtype=float)

            out.loc[mask, "alma_close_alt"] = live_close
            out.loc[mask, "alma_open_alt"] = live_open

        out["alternate_mode"] = "live_current_8h_developing"

    out.drop(columns=["_bucket"], inplace=True, errors="ignore")

    # Pine ta.crossover()/ta.crossunder() are evaluated bar-by-bar using the
    # current value and the immediately previous chart-bar value.
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
        # Exact zone-touch geometry: the candle must actually reach the
        # Demand zone. No 0.3% proximity expansion.
        candidates = [z for z in demand if cur_l <= float(z["top"]) and cur_c >= float(z["btm"])]
    else:
        # Exact zone-touch geometry: the candle must actually reach the
        # Supply zone. No 0.3% proximity expansion.
        candidates = [z for z in supply if cur_h >= float(z["btm"]) and cur_c <= float(z["top"])]
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


def _nearest_opposing_level(
    direction: str,
    entry: float,
    active_demand: list[dict[str, Any]],
    active_supply: list[dict[str, Any]],
    df: pd.DataFrame,
    current_idx: int,
) -> dict[str, Any] | None:
    """Find the nearest structural obstacle in the profit direction.

    Primary source is the opposite Demand/Supply zone. If none exists, use the
    most recent confirmed swing level from the same 10/10 pivot structure.
    """
    candidates: list[dict[str, Any]] = []
    if direction == "LONG":
        for z in active_supply:
            level = _safe_num(z.get("btm"), 0.0)
            if level > entry:
                candidates.append({"price": level, "source": "supply_zone", "zone": dict(z)})
        # confirmed pivot highs only; the pivot must be fully confirmed before i
        for p in range(max(SWING_LEN, current_idx - 120), current_idx - SWING_LEN + 1):
            h = _safe_num(df.loc[p, "high"], 0.0)
            if h <= entry:
                continue
            if all(h >= _safe_num(df.loc[p-k, "high"], 0.0) for k in range(1, SWING_LEN + 1)) and all(
                h >= _safe_num(df.loc[p+k, "high"], 0.0) for k in range(1, SWING_LEN + 1)
            ):
                candidates.append({"price": h, "source": "pivot_high", "pivot_idx": p})
        if not candidates:
            return None
        return min(candidates, key=lambda x: float(x["price"]))

    for z in active_demand:
        level = _safe_num(z.get("top"), 0.0)
        if 0 < level < entry:
            candidates.append({"price": level, "source": "demand_zone", "zone": dict(z)})
    for p in range(max(SWING_LEN, current_idx - 120), current_idx - SWING_LEN + 1):
        l = _safe_num(df.loc[p, "low"], 0.0)
        if l >= entry or l <= 0:
            continue
        if all(l <= _safe_num(df.loc[p-k, "low"], 0.0) for k in range(1, SWING_LEN + 1)) and all(
            l <= _safe_num(df.loc[p+k, "low"], 0.0) for k in range(1, SWING_LEN + 1)
        ):
            candidates.append({"price": l, "source": "pivot_low", "pivot_idx": p})
    if not candidates:
        return None
    return max(candidates, key=lambda x: float(x["price"]))


def _targets_from_nearest_obstacle(
    direction: str,
    entry: float,
    stop: float,
    atr: float,
    obstacle: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Place small targets before the nearest support/resistance obstacle."""
    risk = abs(entry - stop)
    if risk <= 0 or entry <= 0:
        return None
    if obstacle is None:
        # Deterministic fallback only when no structural obstacle exists.
        tp2 = entry + TP2_R * risk if direction == "LONG" else entry - TP2_R * risk
        tp1 = entry + TP1_R * risk if direction == "LONG" else entry - TP1_R * risk
        return {
            "tp1": tp1,
            "tp2": tp2,
            "tp1_rr": abs(tp1 - entry) / risk,
            "tp2_rr": abs(tp2 - entry) / risk,
            "target_source": "atr_rr_fallback",
            "obstacle_price": None,
            "obstacle_source": None,
        }

    level = float(obstacle["price"])
    buffer = max(float(atr) * TP_OBSTACLE_BUFFER_ATR, entry * 0.0002)
    if direction == "LONG":
        usable_distance = level - buffer - entry
        if usable_distance <= 0:
            return None
        tp2_distance = usable_distance * TP2_OBSTACLE_FRACTION
        tp2_distance = min(tp2_distance, TP_MAX_R * risk)
        tp1_distance = tp2_distance * TP1_OBSTACLE_FRACTION
        tp1 = entry + tp1_distance
        tp2 = entry + tp2_distance
    else:
        usable_distance = entry - (level + buffer)
        if usable_distance <= 0:
            return None
        tp2_distance = usable_distance * TP2_OBSTACLE_FRACTION
        tp2_distance = min(tp2_distance, TP_MAX_R * risk)
        tp1_distance = tp2_distance * TP1_OBSTACLE_FRACTION
        tp1 = entry - tp1_distance
        tp2 = entry - tp2_distance

    tp1_rr = abs(tp1 - entry) / risk
    tp2_rr = abs(tp2 - entry) / risk
    if tp2_rr < TP_MIN_R or tp1_rr <= 0 or tp2_rr <= tp1_rr:
        return None
    return {
        "tp1": tp1,
        "tp2": tp2,
        "tp1_rr": tp1_rr,
        "tp2_rr": tp2_rr,
        "target_source": "nearest_opposing_structure",
        "obstacle_price": level,
        "obstacle_source": obstacle.get("source"),
    }


def generate_zone_signals(
    df: pd.DataFrame,
    symbol: str = "",
    mode: Literal["historical", "live"] = "live",
) -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    """Zone-first strategy: trade fresh Demand/Supply touches, no ALMA required."""
    df = _drop_incomplete_1h(df)
    if len(df) < MIN_BARS:
        return df, [], [], []

    df["atr50"] = calc_atr(df, 50)
    df["atr30"] = calc_atr(df, 30)
    df["vol_sma20"] = df["volume"].rolling(20, min_periods=20).mean()
    df["body_atr"] = (df["close"] - df["open"]).abs() / df["atr50"].replace(0, np.nan)
    df["range_atr"] = (df["high"] - df["low"]) / df["atr50"].replace(0, np.nan)

    # Keep ALMA columns available for diagnostics/backward-compatible analytics,
    # but they are deliberately NOT an entry condition in this version.
    df = _attach_exact_alternate_series(df, mode=mode)
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
        cur_l, cur_h, cur_c, cur_o = map(float, [df.loc[i, "low"], df.loc[i, "high"], df.loc[i, "close"], df.loc[i, "open"]])

        demand_zone = _find_directional_zone("LONG", cur_l, cur_h, cur_c, active_demand, active_supply)
        supply_zone = _find_directional_zone("SHORT", cur_l, cur_h, cur_c, active_demand, active_supply)
        if demand_zone is not None and supply_zone is not None:
            # Ambiguous overlap: do not guess a direction.
            continue
        direction = "LONG" if demand_zone is not None else "SHORT" if supply_zone is not None else None
        trade_zone = demand_zone if direction == "LONG" else supply_zone
        if direction is None or trade_zone is None:
            continue

        # Fresh touch only: previous close must have been outside the zone.
        prev_close = float(df.loc[i - 1, "close"])
        if direction == "LONG":
            if not (prev_close > float(trade_zone["top"]) and cur_l <= float(trade_zone["top"]) and cur_c >= float(trade_zone["btm"])):
                continue
        else:
            if not (prev_close < float(trade_zone["btm"]) and cur_h >= float(trade_zone["btm"]) and cur_c <= float(trade_zone["top"])):
                continue

        # Audit-derived entry filters. These are applied only after a literal
        # fresh zone touch, never to ordinary in-zone observations.
        zone_age_bars = max(0, int(i - int(trade_zone.get("start", i))))
        if zone_age_bars > MAX_ZONE_AGE_BARS:
            continue
        if REQUIRE_DIRECTIONAL_CANDLE:
            if direction == "LONG" and cur_c < cur_o:
                continue
            if direction == "SHORT" and cur_c > cur_o:
                continue

        if not math.isfinite(atr) or atr <= 0:
            continue

        zone_top = float(trade_zone["top"])
        zone_bottom = float(trade_zone["btm"])
        sl_buffer = ZONE_SL_ATR_BUFFER * atr
        if direction == "LONG":
            stop = zone_bottom - sl_buffer
            risk = cur_c - stop
        else:
            stop = zone_top + sl_buffer
            risk = stop - cur_c
        if risk <= 0:
            continue

        obstacle = _nearest_opposing_level(direction, cur_c, active_demand, active_supply, df, i)
        if REQUIRE_STRUCTURE_OBSTACLE and obstacle is None:
            continue
        if obstacle is not None:
            obstacle_price = float(obstacle["price"])
            structural_distance = (obstacle_price - cur_c) if direction == "LONG" else (cur_c - obstacle_price)
            if risk <= 0 or structural_distance <= 0 or (structural_distance / risk) < MIN_STRUCTURE_ROOM_R:
                continue
        risk_pct = (risk / cur_c) * 100.0
        if risk_pct > MAX_SIGNAL_RISK_PCT:
            continue
        targets = _targets_from_nearest_obstacle(direction, cur_c, stop, atr, obstacle)
        if targets is None:
            continue
        tp1 = float(targets["tp1"])
        tp2 = float(targets["tp2"])
        tp1_rr = float(targets["tp1_rr"])
        tp2_rr = float(targets["tp2_rr"])
        avg_vol = _safe_num(df.loc[i, "vol_sma20"], 0.0)
        vol_ratio = (_safe_num(df.loc[i, "volume"]) / avg_vol) if avg_vol > 0 else None
        event_ts = int(df.loc[i, "timestamp"].timestamp() * 1000)
        zone_start = int(trade_zone["start"])
        event_id = _make_event_id(symbol or "UNKNOWN", direction, event_ts, zone_start, cur_c)
        zone_ctx = {
            **trade_zone,
            **_zone_context(trade_zone, i, df),
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
                "sl": round(stop, 8),
                "tp1": round(tp1, 8),
                "tp2": round(tp2, 8),
                "risk_pct": round(risk_pct, 4),
                "risk_abs": round(risk, 8),
                "atr": round(float(atr), 8),
                "tp1_rr": round(tp1_rr, 4),
                "tp2_rr": round(tp2_rr, 4),
                "rr_ratio": round(tp2_rr, 4),
                "strategy": "Demand/Supply Zone First",
                "trigger": {
                    "type": "ZONE_TOUCH",
                    "alma_required": False,
                    "alternate_timeframe": "8h",
                    "mode": mode,
                    "zone_touch": True,
                    "zone_entry_rule": "fresh_touch_from_outside",
                },
                "zone": zone_ctx,
                "target": {
                    "source": targets["target_source"],
                    "obstacle_source": targets["obstacle_source"],
                    "obstacle_price": targets["obstacle_price"],
                    "tp1_fraction_to_tp2": TP1_OBSTACLE_FRACTION,
                    "tp2_fraction_to_obstacle": TP2_OBSTACLE_FRACTION,
                    "obstacle_buffer_atr": TP_OBSTACLE_BUFFER_ATR,
                    "tp_max_r": TP_MAX_R,
                },
                "risk_model": {
                    "sl_source": "zone_boundary_plus_atr_buffer",
                    "zone_sl_buffer_atr": ZONE_SL_ATR_BUFFER,
                    "max_signal_risk_pct": MAX_SIGNAL_RISK_PCT,
                },
                "confirmation": {
                    "alma_cross": False,
                    "directional_candle_required": REQUIRE_DIRECTIONAL_CANDLE,
                    "directional_candle_ok": (cur_c >= cur_o) if direction == "LONG" else (cur_c <= cur_o),
                    "zone_age_limit_bars": MAX_ZONE_AGE_BARS,
                    "zone_age_bars": zone_age_bars,
                    "minimum_structure_room_r": MIN_STRUCTURE_ROOM_R,
                    "zone_touch": True,
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
    """Research-only setup score; never changes the zone entry trigger."""
    confirmation = signal.get("confirmation", {})
    zone = signal.get("zone", {})
    score = 50.0
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
