from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from . import research

log = logging.getLogger("event_engine.data_retention")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Conservative retention. These values are intentionally larger than the 24h
# forward-outcome horizon so compaction cannot race the research consumer.
MARKET_5M_RETENTION_HOURS = max(24, int(os.environ.get("DATA_RETENTION_5M_HOURS", "36")))
MARKET_1H_RETENTION_HOURS = max(72, int(os.environ.get("DATA_RETENTION_1H_HOURS", "168")))
NEAREST_APPROACH_RETENTION_HOURS = max(24, int(os.environ.get("DATA_RETENTION_NEAREST_HOURS", "36")))

ARCHIVE_ROOT = Path(os.environ.get("DATA_RETENTION_ARCHIVE_DIR", str(PROJECT_ROOT.parent / "zone-engine-data-archive")))

PROTECTED_DATA_FILES = {
    "trades.jsonl",
    "active_trades.json",
    "entry_decisions.jsonl",
    "execution_ledger.jsonl",
    "account_context.jsonl",
    "failed_signals.json",
    "research_outcomes.jsonl",
    "research_outcome_state.json",
    "research_bar_cursors.json",
    "research_manifest.json",
    "zone_visit_state.json",
    "event_execution_claims.json",
    "actions.jsonl",
    "signal_history.jsonl",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            v = float(value)
            if abs(v) >= 1e11:
                return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(v, tz=timezone.utc)
        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            v = float(text)
            if abs(v) >= 1e11:
                v /= 1000.0
            return datetime.fromtimestamp(v, tz=timezone.utc)
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield line_no, row


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                count += 1
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        return count
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _archive_rows(path: Path, rows: list[dict[str, Any]], archive_root: Path) -> Path | None:
    if not rows:
        return None
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / (path.name + ".pruned.jsonl.gz")
    # append is intentional: repeated idempotent maintenance runs can add a new
    # generation without overwriting an earlier archive.
    mode = "ab" if target.exists() else "wb"
    with gzip.open(target, mode) as gz:
        for row in rows:
            gz.write((json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8"))
    return target


def _max_jsonl_timestamp(path: Path, field: str) -> datetime | None:
    latest: datetime | None = None
    if not path.exists():
        return None
    for _, row in _iter_jsonl(path):
        ts = _parse_ts(row.get(field))
        if ts is not None and (latest is None or ts > latest):
            latest = ts
    return latest


def _pending_observations() -> list[dict[str, Any]]:
    obs_path = DATA_DIR / "zone_observations.jsonl"
    state_path = DATA_DIR / "research_outcome_state.json"
    if not obs_path.exists() or not state_path.exists():
        return []
    state = _read_json(state_path)
    processed = {str(x) for x in (state.get("processed_observation_ids") or []) if x}
    pending: list[dict[str, Any]] = []
    for _, row in _iter_jsonl(obs_path):
        oid = str(row.get("observation_id", ""))
        if not oid or oid in processed:
            continue
        ts = _parse_ts(row.get("observation_ts"))
        if ts is None:
            continue
        pending.append({
            "observation_id": oid,
            "event_type": str(row.get("event_type", "")),
            "observation_ts": ts.isoformat(),
            "symbol": str(row.get("symbol", "")).upper(),
        })
    return pending


def _pending_summary(pending: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("event_type", "UNKNOWN")) for row in pending)
    timestamps = sorted(str(row.get("observation_ts", "")) for row in pending if row.get("observation_ts"))
    return {
        "count": len(pending),
        "by_event_type": dict(counts),
        "oldest_observation_ts": timestamps[0] if timestamps else None,
        "sample": pending[:5],
    }


def _pending_bar_cutoffs(*, five_m_cutoff: datetime, pending: list[dict[str, Any]]) -> dict[str, datetime]:
    """Return extra per-symbol 5m retention needed for unprocessed observations."""
    out: dict[str, datetime] = {}
    for row in pending:
        ts = _parse_ts(row.get("observation_ts"))
        symbol = str(row.get("symbol", "")).upper()
        if ts is None or not symbol:
            continue
        if ts < five_m_cutoff:
            current = out.get(symbol)
            if current is None or ts < current:
                out[symbol] = ts
    return out


def _compact_jsonl(
    path: Path,
    keep_fn,
    *,
    archive_root: Path,
) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "before": 0, "after": 0, "pruned": 0, "archive": None, "sha256_after": None}
    before = 0
    kept: list[dict[str, Any]] = []
    pruned: list[dict[str, Any]] = []
    for _, row in _iter_jsonl(path):
        before += 1
        if keep_fn(row):
            kept.append(row)
        else:
            pruned.append(row)
    with research._FileLock(path):
        archive = _archive_rows(path, pruned, archive_root)
        after = _write_jsonl_atomic(path, kept)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "before": before,
        "after": after,
        "pruned": before - after,
        "archive": str(archive) if archive else None,
        "sha256_after": _sha256(path),
    }


def _validate_state_before() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name in ("active_trades.json", "zone_visit_state.json", "research_bar_cursors.json", "research_outcome_state.json"):
        path = DATA_DIR / name
        if not path.exists():
            checks[name] = {"exists": False}
            continue
        payload = _read_json(path)
        checks[name] = {
            "exists": True,
            "sha256": _sha256(path),
            "entries": len(payload) if isinstance(payload, (dict, list)) else None,
        }
    return checks


def _validate_state_unchanged(before: dict[str, Any]) -> None:
    after = _validate_state_before()
    for name, snapshot in before.items():
        if not snapshot.get("exists"):
            continue
        if after.get(name, {}).get("sha256") != snapshot.get("sha256"):
            raise RuntimeError(f"protected state changed during migration: {name}")


def _safe_archive_dir(now: datetime) -> Path:
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    target = ARCHIVE_ROOT / stamp
    target.mkdir(parents=True, exist_ok=True)
    return target


def compact_data(*, apply: bool = False, archive_dir: Path | None = None, now: datetime | None = None) -> dict[str, Any]:
    now = now or _utc_now()
    archive_root = archive_dir or _safe_archive_dir(now)
    archive_root.mkdir(parents=True, exist_ok=True)

    # One migration process at a time. The three target JSONL files additionally
    # use the same per-file locks as the research writer.
    with research._FileLock(DATA_DIR / "data_retention.lock"):
        return _compact_data_locked(apply=apply, archive_root=archive_root, now=now)


def _compact_data_locked(*, apply: bool, archive_root: Path, now: datetime) -> dict[str, Any]:
    state_before = _validate_state_before()

    pending = _pending_observations()

    if not apply:
        return {
            "mode": "dry_run",
            "retention_hours": {
                "market_5m": MARKET_5M_RETENTION_HOURS,
                "market_1h": MARKET_1H_RETENTION_HOURS,
                "nearest_approach": NEAREST_APPROACH_RETENTION_HOURS,
            },
            "protected_state": state_before,
            "pending_observations": _pending_summary(pending),
        }

    five_m_cutoff = now - timedelta(hours=MARKET_5M_RETENTION_HOURS)
    one_h_cutoff = now - timedelta(hours=MARKET_1H_RETENTION_HOURS)
    nearest_cutoff = now - timedelta(hours=NEAREST_APPROACH_RETENTION_HOURS)
    pending = _pending_observations()
    pending_bar_cutoffs = _pending_bar_cutoffs(five_m_cutoff=five_m_cutoff, pending=pending)

    # Never compact a bar journal if its newest record is already outside the
    # configured window. That situation means the runtime data is stale and a
    # blind cleanup could delete the entire research history.
    latest_5m = _max_jsonl_timestamp(DATA_DIR / "market_bars_5m.jsonl", "timestamp")
    if latest_5m is not None and latest_5m < five_m_cutoff:
        raise RuntimeError(
            f"market_bars_5m.jsonl newest timestamp {latest_5m.isoformat()} is "
            f"older than retention cutoff {five_m_cutoff.isoformat()}; refusing destructive compaction"
        )

    results: dict[str, Any] = {
        "mode": "apply",
        "started_at": now.isoformat(),
        "retention_hours": {
            "market_5m": MARKET_5M_RETENTION_HOURS,
            "market_1h": MARKET_1H_RETENTION_HOURS,
            "nearest_approach": NEAREST_APPROACH_RETENTION_HOURS,
        },
        "pending_observations": _pending_summary(pending),
        "files": {},
    }

    def keep_5m(row: dict[str, Any]) -> bool:
        ts = _parse_ts(row.get("timestamp"))
        if ts is None:
            return True
        symbol = str(row.get("symbol", "")).upper()
        symbol_cutoff = pending_bar_cutoffs.get(symbol)
        if symbol_cutoff is not None:
            return ts >= symbol_cutoff
        return ts >= five_m_cutoff

    def keep_1h(row: dict[str, Any]) -> bool:
        ts = _parse_ts(row.get("timestamp"))
        return ts is None or ts >= one_h_cutoff

    pending_ids = {str(row["observation_id"]) for row in pending}

    def keep_observation(row: dict[str, Any]) -> bool:
        oid = str(row.get("observation_id", ""))
        if oid in pending_ids:
            return True
        if str(row.get("event_type", "")).upper() != "NEAREST_APPROACH":
            return True
        ts = _parse_ts(row.get("observation_ts"))
        return ts is None or ts >= nearest_cutoff

    results["files"]["market_bars_5m.jsonl"] = _compact_jsonl(
        DATA_DIR / "market_bars_5m.jsonl", keep_5m, archive_root=archive_root
    )
    results["files"]["market_bars_1h.jsonl"] = _compact_jsonl(
        DATA_DIR / "market_bars_1h.jsonl", keep_1h, archive_root=archive_root
    )
    results["files"]["zone_observations.jsonl"] = _compact_jsonl(
        DATA_DIR / "zone_observations.jsonl", keep_observation, archive_root=archive_root
    )

    # State files are deliberately not rewritten. Their checksums are a hard
    # post-condition because changing them could cause trigger/research replay.
    _validate_state_unchanged(state_before)

    results["protected_state"] = _validate_state_before()
    results["archive_dir"] = str(archive_root)
    results["finished_at"] = _utc_now().isoformat()

    manifest_path = DATA_DIR / "data_retention_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safe one-time compaction of Zone Engine research/runtime data")
    parser.add_argument("--apply", action="store_true", help="apply compaction; without it only validates the plan")
    parser.add_argument("--archive-dir", type=Path, default=None, help="external directory for pruned JSONL.gz rows")
    args = parser.parse_args(argv)
    result = compact_data(apply=args.apply, archive_dir=args.archive_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
