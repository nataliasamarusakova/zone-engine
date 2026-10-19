from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import time
import zlib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


def _timestamp_series_to_utc(series: pd.Series) -> pd.Series:
    """Normalize ISO/epoch ns/us/ms/s market timestamps without nanosecond inference."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    ratio = float(numeric.notna().mean()) if len(series) else 0.0
    if ratio >= 0.99:
        finite = numeric.dropna().abs()
        magnitude = float(finite.median()) if not finite.empty else 0.0
        unit = ("ns" if magnitude >= 1e17 else "us" if magnitude >= 1e14 else "ms" if magnitude >= 1e11 else "s" if magnitude >= 1e8 else None)
        if unit is not None:
            return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


class _BarIndex:
    """Numpy-backed index over one symbol/provider 5m bar series.

    The research formulas stay the same; this only avoids copying/filtering a full
    DataFrame and iterating pandas rows for every observation.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        work = df.copy()
        work["timestamp"] = _timestamp_series_to_utc(work["timestamp"])
        if "close_time" not in work.columns:
            work["close_time"] = work["timestamp"] + pd.Timedelta(minutes=5)
        else:
            work["close_time"] = _timestamp_series_to_utc(work["close_time"])
        required = ["timestamp", "close_time", "open", "high", "low", "close"]
        work = work.dropna(subset=required)
        # _load_bars() already delivers sorted/deduplicated frames.  Avoid a
        # second full sort in the hot path, but retain deterministic repair for
        # direct callers that provide unsorted or duplicate rows.
        needs_repair = (
            not work["timestamp"].is_monotonic_increasing
            or not work["close_time"].is_monotonic_increasing
            or work["timestamp"].duplicated().any()
        )
        if needs_repair:
            work = work.sort_values(["close_time", "timestamp"]).drop_duplicates("timestamp", keep="last")
        work = work.reset_index(drop=True)
        self.df = work
        self.ts_ns = work["timestamp"].astype("int64").to_numpy()
        self.close_ts_ns = work["close_time"].astype("int64").to_numpy()
        # OHLC arrays MUST follow the exact same sorted/deduplicated rows as the
        # timestamp arrays. Previously they were read from the pre-sort input,
        # which could silently pair one bar's time with another bar's OHLC data.
        self.open = work["open"].to_numpy(dtype=float)
        self.high = work["high"].to_numpy(dtype=float)
        self.low = work["low"].to_numpy(dtype=float)
        self.close = work["close"].to_numpy(dtype=float)
        self.n = len(work)

    def start_at_timestamp(self, ts_ns: int) -> int:
        return int(np.searchsorted(self.ts_ns, ts_ns, side="left"))

    def end_at_close(self, ts_ns: int) -> int:
        return int(np.searchsorted(self.close_ts_ns, ts_ns, side="right"))

    def close_slice(self, start: int, target_ns: int) -> slice:
        return slice(start, self.end_at_close(target_ns))


def _build_bar_indexes(bars_by_key: dict[str, pd.DataFrame]) -> dict[str, _BarIndex]:
    """Build one numpy index per underlying frame, reusing symbol aliases.

    ``_load_bars`` maps a single-provider frame under both ``SYMBOL|PROVIDER``
    and ``SYMBOL``. The frames are the same object; indexing both copies the full
    DataFrame and its numpy arrays needlessly. Reuse by object identity without
    merging genuinely distinct provider histories.
    """
    indexed: dict[str, _BarIndex] = {}
    by_frame_id: dict[int, _BarIndex] = {}
    for key, frame in bars_by_key.items():
        frame_id = id(frame)
        index = by_frame_id.get(frame_id)
        if index is None:
            index = _BarIndex(frame)
            by_frame_id[frame_id] = index
        indexed[key] = index
    return indexed


from event_engine import research, shadow

log = logging.getLogger("zone_engine.research_forward")

HORIZONS_MINUTES = (5, 15, 30, 60, 180, 360, 720, 1440)
MFE_MAE_HORIZONS = (15, 30, 60, 180, 360, 720, 1440)
FAVORABLE_THRESHOLDS = (3, 5, 6, 7, 10)
ADVERSE_THRESHOLDS = (3, 5, 7, 10)

OBS_CURSOR_STATE_KEY = "observation_journal_cursor_v1"
PENDING_OBSERVATIONS_STATE_KEY = "pending_observations_v1"
OBS_CURSOR_SCHEMA_VERSION = 2
OBS_SIGNATURE_BYTES = 64 * 1024
PENDING_META_SCHEMA_VERSION = 2
PENDING_REPLAY_FIELDS = (
    "observation_id", "event_id", "scan_id", "event_type", "symbol",
    "direction", "provider", "source", "observation_ts", "source_event_ts", "reference_price",
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
    # source_event_ts was added for temporal-integrity validation but remains optional
    # for legacy pending rows so old clean state can still be replayed after upgrade.
    required = [field for field in PENDING_REPLAY_FIELDS if field not in {"event_id", "source_event_ts"}]
    return all(meta.get(field) not in (None, "") for field in required)


def _observation_from_pending_meta(meta: dict[str, Any], oid: str) -> dict[str, Any] | None:
    if not isinstance(meta, dict) or not _pending_meta_is_replayable(meta):
        return None
    obs = {field: meta.get(field) for field in PENDING_REPLAY_FIELDS if meta.get(field) not in (None, "")}
    obs["observation_id"] = str(oid)
    return obs


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file_prefix(path: Path, length: int) -> str:
    """Hash every byte before the cursor using bounded memory.

    Sampling the first and last 64 KiB alone can miss a same-size rewrite in the
    middle of a large JSONL journal. The complete processed prefix is the cursor's
    identity; streaming it keeps memory usage bounded and the cost small compared
    with loading/replaying the research journal.
    """
    digest = hashlib.sha256()
    remaining = max(0, int(length))
    with path.open("rb") as fh:
        while remaining:
            chunk = fh.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _journal_signature(path: Path, offset: int) -> dict[str, Any]:
    """Return a content-verified append-only journal signature at a byte offset."""
    if not path.exists():
        return {
            "size": 0, "prefix_sha256": "", "processed_prefix_sha256": _sha256_bytes(b""),
            "window_sha256": "", "window_start": 0,
        }
    stat = path.stat()
    size = int(stat.st_size)
    safe_offset = max(0, min(int(offset), size))
    with path.open("rb") as fh:
        prefix = fh.read(min(OBS_SIGNATURE_BYTES, size))
        window_start = max(0, min(safe_offset - OBS_SIGNATURE_BYTES, size))
        fh.seek(window_start)
        window_len = max(0, min(OBS_SIGNATURE_BYTES, safe_offset - window_start))
        window = fh.read(window_len)
    return {
        "size": size,
        "inode": int(stat.st_ino),
        # Keep the original sampled fields for diagnostics; the full processed-prefix
        # digest below is the authoritative content check for cursor validity.
        "prefix_sha256": _sha256_bytes(prefix),
        "processed_prefix_sha256": _sha256_file_prefix(path, safe_offset),
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
    # inode is runner-local metadata, not journal identity. GitHub-hosted
    # runners receive a fresh inode after every checkout even when the append-only
    # journal bytes are unchanged, so content signatures—not inode—decide validity.
    # Old sampled-only cursors are intentionally invalidated once. They do not
    # prove that bytes between the sampled prefix and end window remained intact.
    return (
        cursor.get("schema_version") == OBS_CURSOR_SCHEMA_VERSION
        and bool(cursor.get("processed_prefix_sha256"))
        and sig.get("processed_prefix_sha256") == cursor.get("processed_prefix_sha256")
        and sig.get("prefix_sha256") == cursor.get("prefix_sha256")
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
    """Load live 5m bars plus immutable recovery/pruned journals.

    The working JSONL is retained as the highest-priority source. Older bars that
    were safely compacted are read from gzip *pruned deltas* and the one-time
    recovery seed; full snapshots are intentionally skipped because they duplicate
    kept and pruned rows. All sources are deduplicated by symbol/provider/timestamp.
    """
    bar_path = Path(research.MARKET_BARS_5M_PATH)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    providers_by_symbol: dict[str, set[str]] = defaultdict(set)
    source_priority = 0

    def add_row(row: dict[str, Any], priority: int) -> None:
        symbol = str(row.get("symbol", "")).upper()
        provider = str(row.get("provider", "")).lower()
        if not symbol or (symbols is not None and symbol not in symbols):
            return
        row_copy = dict(row)
        row_copy["_archive_source_priority"] = priority
        grouped[(symbol, provider)].append(row_copy)
        providers_by_symbol[symbol].add(provider)

    def read_gzip_jsonl(path: Path, priority: int, expected: dict[str, Any] | None = None) -> int:
        count = 0
        source_rows = 0
        source_bytes = 0
        digest = hashlib.sha256()
        if expected is not None:
            try:
                expected_compressed_bytes = int(expected["compressed_bytes"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid compressed_bytes in recovery-seed manifest") from exc
            if path.stat().st_size != expected_compressed_bytes:
                raise ValueError(
                    f"{path}: compressed byte-count mismatch "
                    f"(expected {expected_compressed_bytes}, got {path.stat().st_size})"
                )
        try:
            # Binary mode is deliberate: seed SHA-256 and source-byte counts are
            # defined over the exact uncompressed JSONL bytes, not normalized text.
            with gzip.open(path, "rb") as fh:
                for line_no, raw_line in enumerate(fh, 1):
                    digest.update(raw_line)
                    source_bytes += len(raw_line)
                    if not raw_line.strip():
                        continue
                    try:
                        row = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        if expected is not None:
                            raise ValueError(f"{path}:{line_no}: invalid JSON/UTF-8 in verified recovery seed") from exc
                        log.error("[RESEARCH_ARCHIVE_CORRUPT] path=%s line=%d reason=invalid_json", path, line_no)
                        continue
                    if not isinstance(row, dict):
                        if expected is not None:
                            raise ValueError(f"{path}:{line_no}: non-object row in verified recovery seed")
                        continue
                    source_rows += 1
                    if row.get("record_type") not in (None, "", "MARKET_BAR"):
                        continue
                    add_row(row, priority)
                    count += 1
        except (OSError, EOFError, gzip.BadGzipFile, zlib.error) as exc:
            if expected is not None:
                raise ValueError(f"{path}: recovery seed cannot be read: {exc}") from exc
            log.error("[RESEARCH_ARCHIVE_UNREADABLE] path=%s error=%s", path, exc)
            return count

        if expected is not None:
            expected_sha = str(expected.get("source_sha256", "")).lower()
            if digest.hexdigest() != expected_sha:
                raise ValueError(
                    f"{path}: source SHA-256 mismatch "
                    f"(expected {expected_sha}, got {digest.hexdigest()})"
                )
            try:
                expected_source_bytes = int(expected["source_bytes"])
                expected_source_rows = int(expected["source_rows"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid source byte/row count in recovery-seed manifest") from exc
            if source_bytes != expected_source_bytes:
                raise ValueError(
                    f"{path}: source byte-count mismatch "
                    f"(expected {expected_source_bytes}, got {source_bytes})"
                )
            if source_rows != expected_source_rows:
                raise ValueError(
                    f"{path}: source row-count mismatch "
                    f"(expected {expected_source_rows}, got {source_rows})"
                )
        return count

    def _expected_seed_metadata(archive: Path, seed_root: Path) -> dict[str, Any] | None:
        """Return manifest checks for packaged bootstrap bars; legacy fallback is optional."""
        bootstrap_root = (PROJECT_ROOT / "bootstrap" / "research_history_seed").resolve()
        is_bootstrap_seed = archive.resolve().is_relative_to(bootstrap_root)
        manifest_path = archive.parent / "manifest.json"
        if not manifest_path.is_file():
            if is_bootstrap_seed:
                raise ValueError(f"{archive}: packaged recovery seed has no manifest.json")
            return None
        if not is_bootstrap_seed:
            # A legacy runtime seed is a migration fallback; don't break older
            # deployments that predate the manifest format.
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{manifest_path}: cannot read recovery-seed manifest") from exc
        if not isinstance(manifest, dict) or manifest.get("record_type") != "RESEARCH_HISTORY_RECOVERY_SEED":
            raise ValueError(f"{manifest_path}: unexpected recovery-seed manifest record_type")
        files = manifest.get("files")
        meta = files.get(bar_path.name) if isinstance(files, dict) else None
        if not isinstance(meta, dict):
            raise ValueError(f"{manifest_path}: missing metadata for {bar_path.name}")
        archive_rel = Path(str(meta.get("archive_path", "")))
        if archive_rel.is_absolute():
            raise ValueError(f"{manifest_path}: absolute archive path is not allowed")
        expected_path = (PROJECT_ROOT / archive_rel).resolve()
        if not expected_path.is_relative_to(seed_root.resolve()) or expected_path != archive.resolve():
            raise ValueError(f"{manifest_path}: archive_path does not identify {archive}")
        return meta

    # Immutable recovery seed is shipped separately from runtime data so a source-only
    # checkout has no stale active positions or journals. Read any legacy data/ fallback
    # first, then bootstrap last so the packaged, manifest-verified source wins conflicts.
    seed_roots = [bar_path.parent / "research_history_seed", PROJECT_ROOT / "bootstrap" / "research_history_seed"]
    seen_seed_paths: set[str] = set()
    seed_files: list[tuple[Path, Path]] = []
    for seed_root in seed_roots:
        if not seed_root.exists():
            continue
        for archive in sorted(p for p in seed_root.rglob("*.gz") if p.name.startswith(bar_path.name + ".")):
            normalized = str(archive.resolve())
            if normalized not in seen_seed_paths:
                seen_seed_paths.add(normalized)
                seed_files.append((archive, seed_root))
    for archive, seed_root in seed_files:
        source_priority += 1
        expected = _expected_seed_metadata(archive, seed_root)
        read_gzip_jsonl(archive, source_priority, expected=expected)

    # Size-guard and time-retention compaction write deleted rows as gzip deltas.
    # Read these rows on later clean runners; snapshots themselves duplicate rows
    # and are kept only as integrity/recovery evidence.
    retention_root = bar_path.parent / "retention_archive"
    delta_files = sorted(
        p for p in retention_root.rglob("*.gz")
        if p.name.startswith(bar_path.name + ".pruned")
    ) if retention_root.exists() else []
    for archive in delta_files:
        source_priority += 1
        read_gzip_jsonl(archive, source_priority)

    # The current working file wins any overlapping timestamp because it can
    # contain a provider correction or a newly reconciled row.
    source_priority += 1
    if bar_path.exists():
        with bar_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    log.error("[RESEARCH_JOURNAL_CORRUPT] path=%s line=%d reason=invalid_json", bar_path, line_no)
                    continue
                if isinstance(row, dict):
                    add_row(row, source_priority)

    if seed_files or delta_files:
        log.info(
            "[RESEARCH_BARS_ARCHIVE_LOAD] seed_files=%d pruned_delta_files=%d candidate_symbols=%d rows_loaded=%d",
            len(seed_files), len(delta_files), len(grouped), sum(len(v) for v in grouped.values()),
        )

    out: dict[str, pd.DataFrame] = {}
    for (symbol, provider), items in grouped.items():
        df = pd.DataFrame(items)
        if df.empty:
            continue
        df["timestamp"] = _timestamp_series_to_utc(df["timestamp"])
        if "close_time" in df.columns:
            df["close_time"] = _timestamp_series_to_utc(df["close_time"])
        else:
            df["close_time"] = df["timestamp"] + pd.Timedelta(minutes=5)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["timestamp", "close_time", "open", "high", "low", "close", "volume"])
        df = df.loc[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0) & (df["volume"] >= 0)]
        df = df.loc[(df["high"] >= df[["open", "close"]].max(axis=1)) & (df["low"] <= df[["open", "close"]].min(axis=1)) & (df["high"] >= df["low"])]
        # Select the winning source by timestamp + source priority first. Sorting
        # close_time ahead of priority could otherwise let an older archived row
        # override a live correction when the corrected close_time changed.
        df = df.sort_values(["timestamp", "_archive_source_priority", "close_time"]).drop_duplicates("timestamp", keep="last")
        df = df.sort_values(["close_time", "timestamp"]).reset_index(drop=True)
        df = df.drop(columns=["_archive_source_priority"], errors="ignore")
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
            ts = research._parse_timestamp(row["close_time"] if "close_time" in row else row["timestamp"])
            if ts is None:
                return None
            return max(0.0, (ts - bars.attrs["observation_ts"]).total_seconds() / 60.0)
    return None


def _calculate_forward_outcome_indexed(
    observation: dict[str, Any],
    index: _BarIndex,
    *,
    counterfactual_bars: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    direction = str(observation.get("direction", "")).upper()
    symbol = str(observation.get("symbol", "")).upper()
    if direction not in {"LONG", "SHORT"} or index.n == 0:
        return None
    obs_ts = research._parse_timestamp(observation.get("observation_ts"))
    if obs_ts is None:
        return None
    try:
        reference = float(observation.get("reference_price"))
    except (TypeError, ValueError):
        return None
    if reference <= 0 or not math.isfinite(reference):
        return None

    source_event_ts = research._parse_timestamp(observation.get("source_event_ts"))
    if source_event_ts is not None:
        delta_sec = (obs_ts - source_event_ts).total_seconds()
        if delta_sec < -1.0 or delta_sec > 6 * 60.0:
            return {
                "outcome_id": research.stable_id("forward-invalid-v1", observation.get("observation_id"), obs_ts.isoformat(), prefix="OUT_"),
                "schema_version": research.RESEARCH_SCHEMA_VERSION,
                "record_type": "FORWARD_OUTCOME",
                "observation_id": observation.get("observation_id"),
                "event_id": observation.get("event_id"),
                "scan_id": observation.get("scan_id"),
                "event_type": observation.get("event_type"),
                "outcome_class": "INVALID_RESEARCH_OBSERVATION",
                "symbol": symbol, "direction": direction,
                "provider": observation.get("provider"), "source": observation.get("source"),
                "observation_ts": obs_ts.isoformat(), "reference_price": reference,
                "data_last_ts": None, "bars_available_after_observation": 0,
                "forward_gap_count_24h": None, "forward_path_complete_24h": False,
                "temporal_integrity_status": "INVALID_OBSERVATION_BOUNDARY",
                "temporal_integrity_source_event_ts": source_event_ts.isoformat(),
                "temporal_integrity_delta_sec": delta_sec,
            }

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
    # Counterfactual experiments use only the persisted pre-entry snapshot plus
    # future market bars. They never modify production trade outcomes.
    try:
        if counterfactual_bars is None:
            result["counterfactual_experiments"] = shadow.calculate_counterfactual_outcomes(observation, index.df)
        else:
            result["counterfactual_experiments"] = shadow.calculate_counterfactual_outcomes(
                observation, counterfactual_bars, bars_prepared=True,
            )
    except Exception as exc:
        result["counterfactual_experiments"] = {"schema_version": shadow.SHADOW_SCHEMA_VERSION, "status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

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
    cf = result.get("counterfactual_experiments")
    if isinstance(cf, dict):
        for cf_id in ("CF_VOLUME_5M", "CF_VOLUME_1H"):
            if isinstance(cf.get("experiments", {}).get(cf_id), dict):
                cf["experiments"][cf_id]["forward_return_24h_pct"] = result.get("forward_return_1440m_pct")

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


def _invalid_temporal_outcome(observation: dict[str, Any]) -> dict[str, Any] | None:
    """Return a terminal outcome when observation_ts cannot belong to its source 5m event."""
    obs_ts = research._parse_timestamp(observation.get("observation_ts"))
    source_ts = research._parse_timestamp(observation.get("source_event_ts"))
    if obs_ts is None or source_ts is None:
        return None
    delta_sec = (obs_ts - source_ts).total_seconds()
    if -1.0 <= delta_sec <= 6 * 60.0:
        return None
    try:
        reference = float(observation.get("reference_price"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(reference) or reference <= 0:
        return None
    return {
        "outcome_id": research.stable_id("forward-invalid-v1", observation.get("observation_id"), obs_ts.isoformat(), prefix="OUT_"),
        "schema_version": research.RESEARCH_SCHEMA_VERSION,
        "record_type": "FORWARD_OUTCOME",
        "observation_id": observation.get("observation_id"),
        "event_id": observation.get("event_id"),
        "scan_id": observation.get("scan_id"),
        "event_type": observation.get("event_type"),
        "outcome_class": "INVALID_RESEARCH_OBSERVATION",
        "symbol": str(observation.get("symbol", "")).upper(),
        "direction": str(observation.get("direction", "")).upper(),
        "provider": observation.get("provider"),
        "source": observation.get("source"),
        "observation_ts": obs_ts.isoformat(),
        "reference_price": reference,
        "data_last_ts": None,
        "bars_available_after_observation": 0,
        "forward_gap_count_24h": None,
        "forward_path_complete_24h": False,
        "temporal_integrity_status": "INVALID_OBSERVATION_BOUNDARY",
        "temporal_integrity_source_event_ts": source_ts.isoformat(),
        "temporal_integrity_delta_sec": delta_sec,
    }


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
    terminal_invalid: list[dict[str, Any]] = []
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
                invalid = _invalid_temporal_outcome(obs)
                if invalid is not None:
                    terminal_invalid.append(invalid)
                    continue
                obs_ts = research._parse_timestamp(obs.get("observation_ts"))
                if obs_ts is None:
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
        obs_ts = research._parse_timestamp(meta.get("observation_ts"))
        if obs_ts is None:
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
                        obs_ts = research._parse_timestamp(obs.get("observation_ts"))
                        if obs_ts is None:
                            continue
                    except Exception:
                        continue
                    if now < obs_ts + pd.Timedelta(hours=24):
                        rebuilt_pending[oid] = _pending_meta_from_observation(obs, offset=int(line_offset))
        
        # Keep already-matured candidates out of pending; they'll be completed below.
        for obs in observations:
            oid = str(obs.get("observation_id", ""))
            rebuilt_pending.pop(oid, None)
        pending_now = rebuilt_pending

    if not observations and not candidate_symbols and not terminal_invalid:
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
    if not bars_by_symbol and not terminal_invalid:
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

    index_started = time.perf_counter()
    indexed = _build_bar_indexes(bars_by_symbol)
    index_seconds = time.perf_counter() - index_started

    # Counterfactual simulations need the same sorted OHLC path for every
    # observation. Prepare it once per unique symbol/provider index, not once per
    # observation; formulas and strict close-time boundaries remain unchanged.
    prepare_started = time.perf_counter()
    counterfactual_bars_by_index_id: dict[int, pd.DataFrame] = {}
    for index in indexed.values():
        index_id = id(index)
        if index_id not in counterfactual_bars_by_index_id:
            counterfactual_bars_by_index_id[index_id] = shadow.prepare_counterfactual_bars(index.df)
    prepare_seconds = time.perf_counter() - prepare_started
    unique_indexes = {id(index): index for index in indexed.values()}
    indexed_rows = sum(index.n for index in unique_indexes.values())
    log.info(
        "[RESEARCH_FORWARD_INDEX] frame_keys=%d unique_frames=%d bar_rows=%d index_seconds=%.3f prepare_seconds=%.3f total_seconds=%.3f",
        len(indexed), len(unique_indexes), indexed_rows, index_seconds, prepare_seconds,
        index_seconds + prepare_seconds,
    )

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
    ready: list[dict[str, Any]] = list(terminal_invalid)
    for obs in observations:
        symbol = str(obs.get("symbol", "")).upper()
        provider = str(obs.get("provider", "")).lower()
        key = f"{symbol}|{provider}" if provider else symbol
        index = indexed.get(key) or indexed.get(symbol)
        if index is None:
            if oid := str(obs.get("observation_id", "")):
                pending_now[oid] = dict(candidate_meta.get(oid) or {"offset": 0, "observation_ts": str(obs.get("observation_ts", ""))})
            continue
        outcome = _calculate_forward_outcome_indexed(
            obs, index, counterfactual_bars=counterfactual_bars_by_index_id.get(id(index)),
        )
        oid = str(obs.get("observation_id", ""))
        invalid_observation = isinstance(outcome, dict) and outcome.get("outcome_class") == "INVALID_RESEARCH_OBSERVATION"
        if outcome is None or (not invalid_observation and not bool(outcome.get("forward_path_complete_24h"))):
            if oid:
                # Incomplete but otherwise valid observations stay pending; explicit
                # temporal-integrity failures are terminal research records instead.
                meta = pending_now.get(oid) or candidate_meta.get(oid)
                if meta is None:
                    meta = {"offset": 0, "observation_ts": str(obs.get("observation_ts", "")), "source_event_ts": obs.get("source_event_ts")}
                pending_now[oid] = dict(meta)
            continue
        if invalid_observation:
            ready.append(outcome)
            continue
        if str(outcome.get("outcome_id", "")) in existing_outcome_ids:
            if oid:
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
            observations_by_id: dict[str, dict[str, Any]] = {}
            for observation_row in observations:
                observation_id = str(observation_row.get("observation_id", ""))
                if observation_id and observation_id not in observations_by_id:
                    observations_by_id[observation_id] = observation_row
            for outcome_row in unique_ready:
                try:
                    obs = observations_by_id.get(str(outcome_row.get("observation_id", "")))
                    if obs is not None and outcome_row.get("outcome_class") != "INVALID_RESEARCH_OBSERVATION":
                        research.record_counterfactual_outcome(observation=obs, outcomes=outcome_row.get("counterfactual_experiments") or {}, outcome_id=str(outcome_row.get("outcome_id") or ""))
                except Exception as cf_exc:
                    log.warning("[COUNTERFACTUAL_OUTCOME] event_id=%s error=%s", outcome_row.get("event_id"), cf_exc)
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
