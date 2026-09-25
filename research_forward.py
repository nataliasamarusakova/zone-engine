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

OBS_CURSOR_STATE_KEY = "observation_journal_cursor_v1"
PENDING_OBSERVATIONS_STATE_KEY = "pending_observations_v1"
OBS_CURSOR_SCHEMA_VERSION = 1
OBS_SIGNATURE_BYTES = 64 * 1024
PENDING_META_SCHEMA_VERSION = 2
PENDING_REPLAY_FIELDS = (
    "observation_id", "event_id", "scan_id", "event_type", "symbol",
    "direction", "provider", "source", "observation_ts", "reference_price",
)


def _pending_meta_from_observation(obs: dict[str, Any], *, offset: int) -> dict[str, Any]:
    """Store the minimal observation payload needed to finish a 24h outcome.

    The full raw observation remains in the journal/snapshot archive; this compact
    state copy lets retention move old pending rows out of the working JSONL
    without losing the ability to calculate their forward outcome later.
    """
    meta: dict[str, Any] = {
        "schema_version": PENDING_META_SCHEMA_VERSION,
        "offset": int(offset),
    }
    for field in PENDING_REPLAY_FIELDS:
        value = obs.get(field)
        if value not in (None, ""):
            meta[field] = value
    return meta


def _pending_meta_is_replayable(meta: dict[str, Any]) -> bool:
    return all(meta.get(field) not in (None, "") for field in PENDING_REPLAY_FIELDS if field != "event_id")


def _observation_from_pending_meta(meta: dict[str, Any], oid: str) -> dict[str, Any] | None:
    if not isinstance(meta, dict) or not _pending_meta_is_replayable(meta):
        return None
    obs = {field: meta.get(field) for field in PENDING_REPLAY_FIELDS if meta.get(field) not in (None, "")}
    obs["observation_id"] = str(oid)
    return obs


def _sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _journal_signature(path: Path, offset: int) -> dict[str, Any]:
    """Return a cheap append-only journal signature around the resume offset."""
    if not path.exists():
        return {"size": 0, "prefix_sha256": "", "window_sha256": "", "window_start": 0}
    size = path.stat().st_size
    with path.open("rb") as fh:
        prefix = fh.read(min(OBS_SIGNATURE_BYTES, size))
        window_start = max(0, min(int(offset) - OBS_SIGNATURE_BYTES, size))
        fh.seek(window_start)
        window_len = max(0, min(OBS_SIGNATURE_BYTES, int(offset) - window_start))
        window = fh.read(window_len)
    return {
        "size": int(size),
        "inode": int(path.stat().st_ino),
        "prefix_sha256": _sha256_bytes(prefix),
        "window_start": int(window_start),
        "window_sha256": _sha256_bytes(window),
    }


def _cursor_is_valid(path: Path, cursor: dict[str, Any]) -> bool:
    if not path.exists() or not isinstance(cursor, dict):
        return False
    try:
        offset = int(cursor.get("offset", -1))
        recorded_size = int(cursor.get("size", -1))
    except (TypeError, ValueError):
        return False
    if offset < 0 or recorded_size < 0:
        return False
    current_size = path.stat().st_size
    if current_size < offset or current_size < recorded_size:
        return False
    sig = _journal_signature(path, offset)
    recorded_inode = cursor.get("inode")
    if recorded_inode is not None and int(recorded_inode) != int(sig.get("inode", -1)):
        return False
    return (
        sig.get("prefix_sha256") == cursor.get("prefix_sha256")
        and sig.get("window_sha256") == cursor.get("window_sha256")
        and int(cursor.get("window_start", -1)) == int(sig.get("window_start", -2))
    )


def _read_observation_at(path: Path, offset: int, expected_id: str) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            fh.seek(int(offset))
            line = fh.readline()
        row = json.loads(line)
        if not isinstance(row, dict):
            return None
        if str(row.get("observation_id", "")) != str(expected_id):
            return None
        return row
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _store_observation_cursor(state: dict[str, Any], path: Path, offset: int) -> None:
    sig = _journal_signature(path, offset)
    state[OBS_CURSOR_STATE_KEY] = {
        "schema_version": OBS_CURSOR_SCHEMA_VERSION,
        "offset": int(offset),
        **sig,
    }


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
    raw_path = research.ZONE_OBSERVATIONS_PATH
    now = pd.Timestamp.now(tz="UTC")

    cursor = state.get(OBS_CURSOR_STATE_KEY) if isinstance(state.get(OBS_CURSOR_STATE_KEY), dict) else {}
    cursor_valid = _cursor_is_valid(raw_path, cursor)
    full_rebuild = not cursor_valid
    pending: dict[str, dict[str, Any]] = {
        str(k): dict(v) for k, v in (state.get(PENDING_OBSERVATIONS_STATE_KEY) or {}).items()
        if isinstance(v, dict) and k and _pending_meta_is_replayable(v)
    }
    legacy_pending_present = any(
        isinstance(v, dict) and not _pending_meta_is_replayable(v)
        for v in (state.get(PENDING_OBSERVATIONS_STATE_KEY) or {}).values()
    )
    if full_rebuild:
        read_offset = 0
    else:
        read_offset = int(cursor.get("offset", 0))
    # Older state files stored only offsets. If such entries remain, ignore them
    # until the full journal rebuild reconstructs replayable metadata.
    if legacy_pending_present:
        pending = {}
        full_rebuild = True
        read_offset = 0

    observations: list[dict[str, Any]] = []
    pending_now: dict[str, dict[str, Any]] = dict(pending)
    candidate_symbols: set[str] = set()
    candidate_meta: dict[str, dict[str, Any]] = {}
    new_unique = 0
    seen_new_ids: set[str] = set()
    scan_started = time.perf_counter()
    scan_end_offset = read_offset

    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as fh:
            fh.seek(read_offset)
            while True:
                line_offset = fh.tell()
                line = fh.readline()
                if not line:
                    scan_end_offset = fh.tell()
                    break
                scan_end_offset = fh.tell()
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
                if not oid or oid in processed or oid in pending_now or oid in seen_new_ids:
                    continue
                seen_new_ids.add(oid)
                new_unique += 1
                try:
                    obs_ts = pd.Timestamp(obs.get("observation_ts"))
                    if obs_ts.tzinfo is None:
                        obs_ts = obs_ts.tz_localize("UTC")
                    else:
                        obs_ts = obs_ts.tz_convert("UTC")
                except Exception:
                    continue
                age_ready = now >= obs_ts + pd.Timedelta(hours=24)
                if age_ready:
                    observations.append(obs)
                    candidate_meta[oid] = {"offset": int(line_offset), "observation_ts": obs_ts.isoformat()}
                    symbol = str(obs.get("symbol", "")).upper()
                    if symbol:
                        candidate_symbols.add(symbol)
                else:
                    pending_now[oid] = _pending_meta_from_observation(obs, offset=int(line_offset))

    # Revisit only the small pending set that can have matured since the last run.
    stale_pending: list[str] = []
    for oid, meta in sorted(pending_now.items(), key=lambda item: int(item[1].get("offset", 0))):
        try:
            obs_ts = pd.Timestamp(meta.get("observation_ts"))
            if obs_ts.tzinfo is None:
                obs_ts = obs_ts.tz_localize("UTC")
            else:
                obs_ts = obs_ts.tz_convert("UTC")
        except Exception:
            stale_pending.append(oid)
            continue
        if now < obs_ts + pd.Timedelta(hours=24):
            continue
        obs = _observation_from_pending_meta(meta, oid)
        if obs is None:
            obs = _read_observation_at(raw_path, int(meta.get("offset", 0)), oid)
        if obs is None:
            # The journal changed behind the cursor. Force a full rebuild on the next write.
            full_rebuild = True
            break
        # Upgrade legacy pending metadata in place so the next retention pass can
        # safely archive the raw journal row.
        pending_now[oid] = _pending_meta_from_observation(obs, offset=int(meta.get("offset", 0)))
        observations.append(obs)
        candidate_meta[oid] = dict(meta)
        symbol = str(obs.get("symbol", "")).upper()
        if symbol:
            candidate_symbols.add(symbol)

    scan_seconds = time.perf_counter() - scan_started
    known_total = int(state.get("observation_journal_unique_count", 0))
    if full_rebuild:
        # We will recompute the total from the records observed in the rebuilt journal below.
        known_total = 0
    total_observations = known_total + new_unique
    if full_rebuild:
        # A full rebuild includes all unique IDs in the current journal, including processed rows.
        full_seen: set[str] = set()
        with raw_path.open("r", encoding="utf-8") if raw_path.exists() else _NullContext() as fh:
            if raw_path.exists():
                for line in fh:
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        row = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        rid = str(row.get("observation_id", ""))
                        if rid:
                            full_seen.add(rid)
        total_observations = len(full_seen)

    log.info(
        "[RESEARCH_FORWARD_SCAN] mode=%s cursor_offset=%d unique_total=%d new=%d pending=%d matured_candidates=%d symbols=%d seconds=%.3f",
        "full_rebuild" if full_rebuild else "incremental", read_offset, total_observations, new_unique, len(pending_now), len(observations), len(candidate_symbols), scan_seconds,
    )

    if full_rebuild:
        # Rebuild from the current journal, preserving replayable pending metadata
        # that may now live only in the archive after retention compaction.
        rebuilt_pending: dict[str, dict[str, Any]] = {
            str(k): dict(v) for k, v in pending.items()
            if isinstance(v, dict) and _pending_meta_is_replayable(v)
        }
        seen: set[str] = set()
        if raw_path.exists():
            with raw_path.open("r", encoding="utf-8") as fh:
                while True:
                    line_offset = fh.tell()
                    line = fh.readline()
                    if not line:
                        break
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
                    if not oid or oid in seen or oid in processed:
                        continue
                    seen.add(oid)
                    try:
                        obs_ts = pd.Timestamp(obs.get("observation_ts"))
                        if obs_ts.tzinfo is None:
                            obs_ts = obs_ts.tz_localize("UTC")
                        else:
                            obs_ts = obs_ts.tz_convert("UTC")
                    except Exception:
                        continue
                    if now < obs_ts + pd.Timedelta(hours=24):
                        rebuilt_pending[oid] = _pending_meta_from_observation(obs, offset=int(line_offset))
        
        # Keep already-matured candidates out of pending; they'll be completed below.
        for obs in observations:
            oid = str(obs.get("observation_id", ""))
            rebuilt_pending.pop(oid, None)
        pending_now = rebuilt_pending

    if not observations or not candidate_symbols:
        if write:
            state_payload = dict(state)
            state_payload.update({
                "schema_version": research.RESEARCH_SCHEMA_VERSION,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "processed_observation_ids": sorted(processed),
                "outcome_state_bootstrap_v1": True,
                OBS_CURSOR_STATE_KEY: {
                    **_journal_signature(raw_path, scan_end_offset),
                    "schema_version": OBS_CURSOR_SCHEMA_VERSION,
                    "offset": int(scan_end_offset),
                },
                PENDING_OBSERVATIONS_STATE_KEY: pending_now,
                "observation_journal_unique_count": int(total_observations),
            })
            research._atomic_json_locked(research.RESEARCH_OUTCOME_STATE_PATH, state_payload)
        return 0, total_observations

    bars_started = time.perf_counter()
    bars_by_symbol = _load_bars(symbols=candidate_symbols)
    log.info("[RESEARCH_FORWARD_BARS] symbols=%d groups=%d seconds=%.3f", len(candidate_symbols), len(bars_by_symbol), time.perf_counter() - bars_started)
    if not bars_by_symbol:
        if write:
            state_payload = dict(state)
            state_payload.update({
                "schema_version": research.RESEARCH_SCHEMA_VERSION,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "processed_observation_ids": sorted(processed),
                "outcome_state_bootstrap_v1": True,
                OBS_CURSOR_STATE_KEY: {**_journal_signature(raw_path, scan_end_offset), "schema_version": OBS_CURSOR_SCHEMA_VERSION, "offset": int(scan_end_offset)},
                PENDING_OBSERVATIONS_STATE_KEY: pending_now,
                "observation_journal_unique_count": int(total_observations),
            })
            research._atomic_json_locked(research.RESEARCH_OUTCOME_STATE_PATH, state_payload)
        return 0, total_observations

    indexed: dict[str, _BarIndex] = {key: _BarIndex(df) for key, df in bars_by_symbol.items()}

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
            if oid := str(obs.get("observation_id", "")):
                pending_now[oid] = dict(candidate_meta.get(oid) or {"offset": 0, "observation_ts": str(obs.get("observation_ts", ""))})
            continue
        outcome = _calculate_forward_outcome_indexed(obs, index)
        oid = str(obs.get("observation_id", ""))
        if outcome is None or not bool(outcome.get("forward_path_complete_24h")):
            if oid:
                # If we cannot complete the 24h horizon yet, revisit next run.
                meta = pending_now.get(oid) or candidate_meta.get(oid)
                if meta is None:
                    meta = {"offset": 0, "observation_ts": str(obs.get("observation_ts", ""))}
                pending_now[oid] = dict(meta)
            continue
        if not bootstrap_complete and str(outcome.get("outcome_id", "")) in existing_outcome_ids:
            processed.add(oid)
            pending_now.pop(oid, None)
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
            for x in unique_ready:
                pending_now.pop(str(x.get("observation_id", "")), None)
            ready = unique_ready
        state_payload = dict(state)
        state_payload.update({
            "schema_version": research.RESEARCH_SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "processed_observation_ids": sorted(processed),
            "outcome_state_bootstrap_v1": True,
            OBS_CURSOR_STATE_KEY: {**_journal_signature(raw_path, scan_end_offset), "schema_version": OBS_CURSOR_SCHEMA_VERSION, "offset": int(scan_end_offset)},
            PENDING_OBSERVATIONS_STATE_KEY: pending_now,
            "observation_journal_unique_count": int(total_observations),
        })
        research._atomic_json_locked(research.RESEARCH_OUTCOME_STATE_PATH, state_payload)
        if ready:
            research._bump_manifest("forward_outcomes_written", len(ready))
    return len(ready), total_observations


class _NullContext:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
    def __iter__(self):
        return iter(())


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
