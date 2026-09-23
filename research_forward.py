from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


class _BarIndex:
    """Numpy-backed index over one symbol/provider 5m bar series.

    The research formulas stay the same; this only avoids copying/filtering a full
    DataFrame and iterating pandas rows for every observation.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        if "close_time" not in df.columns:
            df = df.copy()
            df["close_time"] = df["timestamp"] + pd.Timedelta(minutes=5)
        self.df = df
        self.ts_ns = df["timestamp"].astype("int64").to_numpy()
        self.close_ts_ns = df["close_time"].astype("int64").to_numpy()
        self.open = df["open"].to_numpy(dtype=float)
        self.high = df["high"].to_numpy(dtype=float)
        self.low = df["low"].to_numpy(dtype=float)
        self.close = df["close"].to_numpy(dtype=float)
        self.n = len(df)

    def start_at_timestamp(self, ts_ns: int) -> int:
        return int(np.searchsorted(self.ts_ns, ts_ns, side="left"))

    def end_at_close(self, ts_ns: int) -> int:
        return int(np.searchsorted(self.close_ts_ns, ts_ns, side="right"))

    def close_slice(self, start: int, target_ns: int) -> slice:
        return slice(start, self.end_at_close(target_ns))



from event_engine import research

log = logging.getLogger("zone_engine.research_forward")

HORIZONS_MINUTES = (5, 15, 30, 60, 180, 360, 720, 1440)
MFE_MAE_HORIZONS = (15, 30, 60, 180, 360, 720, 1440)
FAVORABLE_THRESHOLDS = (3, 5, 6, 7, 10)
ADVERSE_THRESHOLDS = (3, 5, 7, 10)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    non_object = 0
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    non_object += 1
            except json.JSONDecodeError:
                malformed += 1
    if malformed or non_object:
        log.error("[RESEARCH_JOURNAL_CORRUPT] path=%s malformed=%d non_object=%d", path, malformed, non_object)
    return rows


def _load_bars(*, symbols: set[str] | None = None) -> dict[str, pd.DataFrame]:
    rows = _read_jsonl(research.MARKET_BARS_5M_PATH)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    providers_by_symbol: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        provider = str(row.get("provider", "")).lower()
        if symbols is not None and symbol not in symbols:
            continue
        if symbol:
            grouped[(symbol, provider)].append(row)
            providers_by_symbol[symbol].add(provider)
    out: dict[str, pd.DataFrame] = {}
    for (symbol, provider), items in grouped.items():
        df = pd.DataFrame(items)
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if "close_time" in df.columns:
            df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")
        else:
            df["close_time"] = df["timestamp"] + pd.Timedelta(minutes=5)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["timestamp", "close_time", "open", "high", "low", "close", "volume"])
        df = df.loc[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)]
        df = df.loc[(df["high"] >= df[["open", "close"]].max(axis=1)) & (df["low"] <= df[["open", "close"]].min(axis=1)) & (df["high"] >= df["low"])]
        df = df.sort_values(["close_time", "timestamp"]).drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        key = f"{symbol}|{provider}" if provider else symbol
        out[key] = df
        if len(providers_by_symbol[symbol]) == 1:
            out[symbol] = df
    return out


def _path_gap_count(bars: pd.DataFrame, observation_ts: pd.Timestamp, target_ts: pd.Timestamp) -> int:
    """Count missing 5m bars up to target_ts using candle-open timestamps."""
    path = bars.loc[(bars["timestamp"] >= observation_ts) & (bars["close_time"] <= target_ts), ["timestamp", "close_time"]].sort_values("timestamp")
    if path.empty:
        return 0
    next_expected = observation_ts.floor("5min") + pd.Timedelta(minutes=5)
    first_delta = path.iloc[0]["timestamp"] - next_expected
    gaps = max(0, int(round(first_delta.total_seconds() / 300.0)))
    diffs = path["timestamp"].diff().dropna()
    for delta in diffs:
        gaps += max(0, int(round(delta.total_seconds() / 300.0)) - 1)
    return gaps


def _path_gap_count_indexed(index: _BarIndex, observation_ts_ns: int, target_ts_ns: int) -> int:
    start = index.start_at_timestamp(observation_ts_ns)
    end = index.end_at_close(target_ts_ns)
    if end <= start:
        return 0
    ts = index.ts_ns[start:end]
    if ts.size == 0:
        return 0
    next_expected_ns = ((observation_ts_ns // (5 * 60 * 1_000_000_000)) + 1) * (5 * 60 * 1_000_000_000)
    gaps = max(0, int(round((int(ts[0]) - next_expected_ns) / (5 * 60 * 1_000_000_000))))
    if ts.size > 1:
        diffs = np.diff(ts) // (5 * 60 * 1_000_000_000)
        gaps += int(np.maximum(diffs - 1, 0).sum())
    return gaps


def _pct_return(direction: str, reference: float, price: float) -> float | None:
    if not reference or not math.isfinite(reference) or not math.isfinite(price):
        return None
    raw = (price / reference - 1.0) * 100.0
    return raw if direction == "LONG" else -raw


def _favorable_mfe_mae(direction: str, reference: float, window: pd.DataFrame) -> tuple[float | None, float | None]:
    if window.empty or reference <= 0:
        return None, None
    if direction == "LONG":
        favorable = (float(window["high"].max()) / reference - 1.0) * 100.0
        adverse = (1.0 - float(window["low"].min()) / reference) * 100.0
    else:
        favorable = (1.0 - float(window["low"].min()) / reference) * 100.0
        adverse = (float(window["high"].max()) / reference - 1.0) * 100.0
    return max(0.0, favorable), max(0.0, adverse)


def _time_to_threshold(direction: str, reference: float, bars: pd.DataFrame, threshold_pct: float, *, favorable: bool) -> float | None:
    if reference <= 0 or bars.empty:
        return None
    for _, row in bars.iterrows():
        if favorable:
            hit = float(row["high"]) >= reference * (1.0 + threshold_pct / 100.0) if direction == "LONG" else float(row["low"]) <= reference * (1.0 - threshold_pct / 100.0)
        else:
            hit = float(row["low"]) <= reference * (1.0 - threshold_pct / 100.0) if direction == "LONG" else float(row["high"]) >= reference * (1.0 + threshold_pct / 100.0)
        if hit:
            ts = pd.Timestamp(row["close_time"] if "close_time" in row else row["timestamp"])
            return max(0.0, (ts - bars.attrs["observation_ts"]).total_seconds() / 60.0)
    return None


def _calculate_forward_outcome_indexed(observation: dict[str, Any], index: _BarIndex) -> dict[str, Any] | None:
    direction = str(observation.get("direction", "")).upper()
    symbol = str(observation.get("symbol", "")).upper()
    if direction not in {"LONG", "SHORT"} or index.n == 0:
        return None
    try:
        obs_ts = pd.Timestamp(observation.get("observation_ts"))
        if obs_ts.tzinfo is None:
            obs_ts = obs_ts.tz_localize("UTC")
        else:
            obs_ts = obs_ts.tz_convert("UTC")
        reference = float(observation.get("reference_price"))
    except Exception:
        return None
    if reference <= 0 or not math.isfinite(reference):
        return None

    obs_ns = int(obs_ts.value)
    future_start = index.start_at_timestamp(obs_ns)
    if future_start >= index.n:
        return None
    future_end = index.n
    last_close_ns = int(index.close_ts_ns[future_end - 1])
    last_close_ts = pd.Timestamp(last_close_ns, tz="UTC")
    result: dict[str, Any] = {
        "outcome_id": research.stable_id("forward-v1", observation.get("observation_id"), last_close_ts.isoformat(), prefix="OUT_"),
        "schema_version": research.RESEARCH_SCHEMA_VERSION, "record_type": "FORWARD_OUTCOME",
        "observation_id": observation.get("observation_id"), "event_id": observation.get("event_id"),
        "scan_id": observation.get("scan_id"), "event_type": observation.get("event_type"),
        "outcome_class": "SIGNAL_FORWARD_OUTCOME" if observation.get("event_type") == "SIGNAL_CREATED" else "HYPOTHETICAL_FORWARD_OUTCOME",
        "symbol": symbol, "direction": direction, "provider": observation.get("provider"), "source": observation.get("source"),
        "observation_ts": obs_ts.isoformat(),
        "reference_price": reference, "data_last_ts": last_close_ts.isoformat(),
        "bars_available_after_observation": int(future_end - future_start),
    }

    close_start = int(np.searchsorted(index.close_ts_ns, obs_ns, side="right"))
    for minutes in HORIZONS_MINUTES:
        target_ns = obs_ns + int(minutes * 60 * 1_000_000_000)
        end = index.end_at_close(target_ns)
        key = f"forward_return_{minutes}m_pct"
        if end <= close_start:
            result[key] = None
            result[f"censored_{minutes}m"] = True
            result[f"forward_return_{minutes}m_sample_ts"] = None
        else:
            sample_close = float(index.close[end - 1])
            result[key] = _pct_return(direction, reference, sample_close)
            result[f"censored_{minutes}m"] = False
            result[f"forward_return_{minutes}m_sample_ts"] = pd.Timestamp(int(index.close_ts_ns[end - 1]), tz="UTC").isoformat()

    for minutes in MFE_MAE_HORIZONS:
        target_ns = obs_ns + int(minutes * 60 * 1_000_000_000)
        end = index.end_at_close(target_ns)
        gap_count = _path_gap_count_indexed(index, obs_ns, target_ns)
        path_complete = (end > future_start) and last_close_ns >= target_ns and gap_count == 0
        if path_complete:
            highs = index.high[future_start:end]
            lows = index.low[future_start:end]
            if direction == "LONG":
                mfe = max(0.0, (float(np.max(highs)) / reference - 1.0) * 100.0)
                mae = max(0.0, (1.0 - float(np.min(lows)) / reference) * 100.0)
            else:
                mfe = max(0.0, (1.0 - float(np.min(lows)) / reference) * 100.0)
                mae = max(0.0, (float(np.max(highs)) / reference - 1.0) * 100.0)
        else:
            mfe = mae = None
        result[f"forward_mfe_{minutes}m_pct"] = mfe
        result[f"forward_mae_{minutes}m_pct"] = mae
        result[f"forward_gap_count_{minutes}m"] = gap_count
        result[f"censored_path_{minutes}m"] = not path_complete

    horizon_24_ns = obs_ns + int(24 * 60 * 60 * 1_000_000_000)
    result["forward_gap_count_24h"] = _path_gap_count_indexed(index, obs_ns, horizon_24_ns)
    result["forward_path_complete_24h"] = result["forward_gap_count_24h"] == 0 and last_close_ns >= horizon_24_ns

    threshold_end = index.start_at_timestamp(horizon_24_ns)
    threshold_end = min(threshold_end, index.n)
    for threshold in FAVORABLE_THRESHOLDS:
        target = reference * (1.0 + threshold / 100.0) if direction == "LONG" else reference * (1.0 - threshold / 100.0)
        arr = index.high if direction == "LONG" else index.low
        if direction == "LONG":
            hits = np.flatnonzero(arr[future_start:threshold_end] >= target)
        else:
            hits = np.flatnonzero(arr[future_start:threshold_end] <= target)
        if hits.size:
            hit_idx = future_start + int(hits[0])
            hit_ts = int(index.close_ts_ns[hit_idx])
            result[f"time_to_plus_{threshold}pct_min"] = max(0.0, (hit_ts - obs_ns) / 60_000_000_000.0)
        else:
            result[f"time_to_plus_{threshold}pct_min"] = None
    for threshold in ADVERSE_THRESHOLDS:
        target = reference * (1.0 - threshold / 100.0) if direction == "LONG" else reference * (1.0 + threshold / 100.0)
        arr = index.low if direction == "LONG" else index.high
        if direction == "LONG":
            hits = np.flatnonzero(arr[future_start:threshold_end] <= target)
        else:
            hits = np.flatnonzero(arr[future_start:threshold_end] >= target)
        if hits.size:
            hit_idx = future_start + int(hits[0])
            hit_ts = int(index.close_ts_ns[hit_idx])
            result[f"time_to_minus_{threshold}pct_min"] = max(0.0, (hit_ts - obs_ns) / 60_000_000_000.0)
        else:
            result[f"time_to_minus_{threshold}pct_min"] = None

    result["plus_5_before_minus_10"] = False
    result["minus_10_before_plus_5"] = False
    plus5_t = result["time_to_plus_5pct_min"]; minus10_t = result["time_to_minus_10pct_min"]
    result["threshold_order_ambiguous_same_bar"] = False
    if plus5_t is not None and minus10_t is not None:
        if plus5_t < minus10_t:
            result["plus_5_before_minus_10"] = True
        elif minus10_t < plus5_t:
            result["minus_10_before_plus_5"] = True
        else:
            result["plus_5_before_minus_10"] = None
            result["minus_10_before_plus_5"] = None
            result["threshold_order_ambiguous_same_bar"] = True
    elif plus5_t is not None:
        result["plus_5_before_minus_10"] = True
    elif minus10_t is not None:
        result["minus_10_before_plus_5"] = True
    return result


def calculate_forward_outcome(observation: dict[str, Any], bars: pd.DataFrame) -> dict[str, Any] | None:
    return _calculate_forward_outcome_indexed(observation, _BarIndex(bars))


def _load_state() -> dict[str, Any]:
    raw = research._load_json(research.RESEARCH_OUTCOME_STATE_PATH, {})
    return raw if isinstance(raw, dict) else {}


def update_outcomes(*, write: bool = False) -> tuple[int, int]:
    started = time.perf_counter()
    state = _load_state()
    processed = set(str(x) for x in state.get("processed_observation_ids", []) if x)
    bootstrap_complete = bool(state.get("outcome_state_bootstrap_v1"))

    # Stream observations and keep only candidates that are both unprocessed and mature.
    # We still scan the append-only journal, but avoid building a second full in-memory copy.
    now = pd.Timestamp.now(tz="UTC")
    observations: list[dict[str, Any]] = []
    candidate_symbols: set[str] = set()
    total_observations = 0
    raw_path = research.ZONE_OBSERVATIONS_PATH
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as fh:
            seen_ids: set[str] = set()
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    obs = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obs, dict):
                    continue
                oid = str(obs.get("observation_id", ""))
                if not oid or oid in seen_ids:
                    continue
                seen_ids.add(oid)
                total_observations += 1
                if oid in processed:
                    continue
                try:
                    obs_ts = pd.Timestamp(obs.get("observation_ts"))
                    if obs_ts.tzinfo is None:
                        obs_ts = obs_ts.tz_localize("UTC")
                    else:
                        obs_ts = obs_ts.tz_convert("UTC")
                except Exception:
                    continue
                if now < obs_ts + pd.Timedelta(hours=24):
                    continue
                symbol = str(obs.get("symbol", "")).upper()
                if not symbol:
                    continue
                observations.append(obs)
                candidate_symbols.add(symbol)

    scan_seconds = time.perf_counter() - started
    log.info("[RESEARCH_FORWARD_SCAN] unique_observations=%d matured_candidates=%d symbols=%d seconds=%.3f", total_observations, len(observations), len(candidate_symbols), scan_seconds)
    if not observations or not candidate_symbols:
        if write and not bootstrap_complete:
            # Preserve every pre-existing state key; only add the bootstrap marker.
            state_payload = dict(state)
            state_payload.update({
                "schema_version": research.RESEARCH_SCHEMA_VERSION,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "processed_observation_ids": sorted(processed),
                "outcome_state_bootstrap_v1": True,
            })
            research._atomic_json_locked(research.RESEARCH_OUTCOME_STATE_PATH, state_payload)
        return 0, total_observations

    bars_started = time.perf_counter()
    bars_by_symbol = _load_bars(symbols=candidate_symbols)
    log.info("[RESEARCH_FORWARD_BARS] symbols=%d groups=%d seconds=%.3f", len(candidate_symbols), len(bars_by_symbol), time.perf_counter() - bars_started)
    if not bars_by_symbol:
        return 0, total_observations

    # Cache the compact numeric index once per symbol/provider and reuse it for all observations.
    indexed: dict[str, _BarIndex] = {key: _BarIndex(df) for key, df in bars_by_symbol.items()}

    # One-time legacy bootstrap: only if the state file has not yet recorded the bootstrap marker.
    existing_outcome_ids: set[str] = set()
    if not bootstrap_complete and research.RESEARCH_OUTCOMES_PATH.exists():
        for line in research.RESEARCH_OUTCOMES_PATH.open("r", encoding="utf-8"):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("outcome_id"):
                existing_outcome_ids.add(str(row["outcome_id"]))

    calc_started = time.perf_counter()
    ready: list[dict[str, Any]] = []
    for obs in observations:
        symbol = str(obs.get("symbol", "")).upper()
        provider = str(obs.get("provider", "")).lower()
        key = f"{symbol}|{provider}" if provider else symbol
        index = indexed.get(key) or indexed.get(symbol)
        if index is None:
            continue
        outcome = _calculate_forward_outcome_indexed(obs, index)
        if outcome is None or not bool(outcome.get("forward_path_complete_24h")):
            continue
        oid = str(obs.get("observation_id", ""))
        if not bootstrap_complete and str(outcome.get("outcome_id", "")) in existing_outcome_ids:
            processed.add(oid)
            continue
        ready.append(outcome)

    log.info("[RESEARCH_FORWARD_CALC] candidates=%d ready=%d seconds=%.3f", len(observations), len(ready), time.perf_counter() - calc_started)
    if write:
        unique_ready: list[dict[str, Any]] = []
        seen_outcome_ids = set(existing_outcome_ids)
        for row in ready:
            outcome_id = str(row.get("outcome_id", ""))
            if outcome_id and outcome_id in seen_outcome_ids:
                continue
            if outcome_id:
                seen_outcome_ids.add(outcome_id)
            unique_ready.append(row)
        if unique_ready:
            research._append_jsonl_locked(research.RESEARCH_OUTCOMES_PATH, unique_ready)
            processed.update(str(x["observation_id"]) for x in unique_ready if x.get("observation_id"))
            ready = unique_ready
        state_payload = dict(state)
        state_payload.update({
            "schema_version": research.RESEARCH_SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed_observation_ids": sorted(processed),
            "outcome_state_bootstrap_v1": True,
        })
        research._atomic_json_locked(research.RESEARCH_OUTCOME_STATE_PATH, state_payload)
        if ready:
            research._bump_manifest("forward_outcomes_written", len(ready))
    return len(ready), total_observations


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description="Update 24h forward outcomes for Zone Engine shadow observations")
    parser.add_argument("--write", action="store_true", help="Persist matured forward outcomes")
    args = parser.parse_args()
    ready, total = update_outcomes(write=args.write)
    print(json.dumps({"observations_total": total, "matured_outcomes": ready, "write": bool(args.write)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
