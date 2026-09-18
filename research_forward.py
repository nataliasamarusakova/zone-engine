from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

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


def _load_bars() -> dict[str, pd.DataFrame]:
    rows = _read_jsonl(research.MARKET_BARS_5M_PATH)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    providers_by_symbol: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        provider = str(row.get("provider", "")).lower()
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


def calculate_forward_outcome(observation: dict[str, Any], bars: pd.DataFrame) -> dict[str, Any] | None:
    symbol = str(observation.get("symbol", "")).upper()
    direction = str(observation.get("direction", "")).upper()
    if direction not in {"LONG", "SHORT"} or bars.empty:
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
    x = bars.copy()
    if "close_time" not in x.columns:
        x["close_time"] = x["timestamp"] + pd.Timedelta(minutes=5)
    # A completed candle whose opening timestamp is before observation_ts contains
    # price action that may have happened before the decision. Exclude such a partial
    # candle rather than leaking its pre-observation high/low into the outcome.
    future = x.loc[x["timestamp"] >= obs_ts].copy().reset_index(drop=True)
    if future.empty:
        return None
    future.attrs["observation_ts"] = obs_ts
    result: dict[str, Any] = {
        "outcome_id": research.stable_id("forward-v1", observation.get("observation_id"), future["close_time"].iloc[-1].isoformat(), prefix="OUT_"),
        "schema_version": research.RESEARCH_SCHEMA_VERSION, "record_type": "FORWARD_OUTCOME",
        "observation_id": observation.get("observation_id"), "event_id": observation.get("event_id"),
        "scan_id": observation.get("scan_id"), "event_type": observation.get("event_type"),
        "outcome_class": "SIGNAL_FORWARD_OUTCOME" if observation.get("event_type") == "SIGNAL_CREATED" else "HYPOTHETICAL_FORWARD_OUTCOME",
        "symbol": symbol, "direction": direction, "provider": observation.get("provider"), "source": observation.get("source"),
        "observation_ts": obs_ts.isoformat(),
        "reference_price": reference, "data_last_ts": pd.Timestamp(future["close_time"].iloc[-1]).isoformat(),
        "bars_available_after_observation": int(len(future)),
    }
    last_close_ts = pd.Timestamp(future["close_time"].iloc[-1])
    # Forward returns are sampled only from a completed close at or before the
    # requested horizon. This avoids introducing post-horizon price information;
    # the exact horizon is represented by the nearest completed 5m bar close.
    close_path = x.loc[x["close_time"] > obs_ts].sort_values("close_time").reset_index(drop=True)
    for minutes in HORIZONS_MINUTES:
        target_ts = obs_ts + pd.Timedelta(minutes=minutes)
        eligible = close_path.loc[close_path["close_time"] <= target_ts]
        key = f"forward_return_{minutes}m_pct"
        if eligible.empty:
            result[key] = None; result[f"censored_{minutes}m"] = True; result[f"forward_return_{minutes}m_sample_ts"] = None
        else:
            sample = eligible.iloc[-1]
            result[key] = _pct_return(direction, reference, float(sample["close"]))
            result[f"censored_{minutes}m"] = False
            result[f"forward_return_{minutes}m_sample_ts"] = pd.Timestamp(sample["close_time"]).isoformat()
    for minutes in MFE_MAE_HORIZONS:
        target_ts = obs_ts + pd.Timedelta(minutes=minutes)
        eligible = future.loc[future["close_time"] <= target_ts]
        gap_count = _path_gap_count(x, obs_ts, target_ts)
        path_complete = (not eligible.empty) and last_close_ts >= target_ts and gap_count == 0
        mfe, mae = _favorable_mfe_mae(direction, reference, eligible) if path_complete else (None, None)
        result[f"forward_mfe_{minutes}m_pct"] = mfe
        result[f"forward_mae_{minutes}m_pct"] = mae
        result[f"forward_gap_count_{minutes}m"] = gap_count
        result[f"censored_path_{minutes}m"] = not path_complete
    result["forward_gap_count_24h"] = _path_gap_count(x, obs_ts, obs_ts + pd.Timedelta(hours=24))
    result["forward_path_complete_24h"] = result["forward_gap_count_24h"] == 0 and last_close_ts >= obs_ts + pd.Timedelta(hours=24)
    threshold_path = future.loc[future["timestamp"] < obs_ts + pd.Timedelta(hours=24)].copy()
    for threshold in FAVORABLE_THRESHOLDS:
        result[f"time_to_plus_{threshold}pct_min"] = _time_to_threshold(direction, reference, threshold_path, threshold, favorable=True)
    for threshold in ADVERSE_THRESHOLDS:
        result[f"time_to_minus_{threshold}pct_min"] = _time_to_threshold(direction, reference, threshold_path, threshold, favorable=False)
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


def _load_state() -> dict[str, Any]:
    raw = research._load_json(research.RESEARCH_OUTCOME_STATE_PATH, {})
    return raw if isinstance(raw, dict) else {}


def update_outcomes(*, write: bool = False) -> tuple[int, int]:
    raw_observations = _read_jsonl(research.ZONE_OBSERVATIONS_PATH)
    # Collapse duplicate observation records before outcome generation. This makes
    # the consumer idempotent even if the append-only journal contains repeated rows.
    observations_by_id: dict[str, dict[str, Any]] = {}
    for obs in raw_observations:
        oid = str(obs.get("observation_id", ""))
        if oid and oid not in observations_by_id:
            observations_by_id[oid] = obs
    observations = list(observations_by_id.values())
    bars_by_symbol = _load_bars()
    if not observations or not bars_by_symbol:
        return 0, 0
    state = _load_state()
    processed = set(str(x) for x in state.get("processed_observation_ids", []) if x)
    existing_outcomes = _read_jsonl(research.RESEARCH_OUTCOMES_PATH)
    existing_outcome_ids = {str(x.get("outcome_id", "")) for x in existing_outcomes if x.get("outcome_id")}
    ready: list[dict[str, Any]] = []
    now = pd.Timestamp.now(tz="UTC")
    for obs in observations:
        oid = str(obs.get("observation_id", ""))
        if not oid or oid in processed:
            continue
        try:
            obs_ts = pd.Timestamp(obs.get("observation_ts"))
            if obs_ts.tzinfo is None:
                obs_ts = obs_ts.tz_localize("UTC")
            else:
                obs_ts = obs_ts.tz_convert("UTC")
        except Exception:
            continue
        # Wait until a complete 24h forward path should exist; this makes the final
        # stored record non-censored for the requested 24h horizon whenever data exists.
        if now < obs_ts + pd.Timedelta(hours=24):
            continue
        symbol = str(obs.get("symbol", "")).upper()
        provider = str(obs.get("provider", "")).lower()
        bars = bars_by_symbol.get(f"{symbol}|{provider}") if provider else bars_by_symbol.get(symbol)
        if bars is None:
            bars = bars_by_symbol.get(symbol, pd.DataFrame())
        outcome = calculate_forward_outcome(obs, bars)
        if outcome is None or not bool(outcome.get("forward_path_complete_24h")):
            continue
        if str(outcome.get("outcome_id", "")) in existing_outcome_ids:
            processed.add(oid)
            continue
        ready.append(outcome)
    if write and ready:
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
            ready = unique_ready
            processed.update(str(x["observation_id"]) for x in ready)
        research._atomic_json_locked(research.RESEARCH_OUTCOME_STATE_PATH, {
            "schema_version": research.RESEARCH_SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed_observation_ids": sorted(processed),
        })
        research._bump_manifest("forward_outcomes_written", len(ready))
    return len(ready), len(observations)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update 24h forward outcomes for Zone Engine shadow observations")
    parser.add_argument("--write", action="store_true", help="Persist matured forward outcomes")
    args = parser.parse_args()
    ready, total = update_outcomes(write=args.write)
    print(json.dumps({"observations_total": total, "matured_outcomes": ready, "write": bool(args.write)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
