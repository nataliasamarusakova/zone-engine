from __future__ import annotations

import argparse
from contextlib import nullcontext
import gzip
import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import research

log = logging.getLogger("event_engine.data_retention")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Conservative operational retention. Archived rows are preserved outside the
# repository, while the working data stays small enough for routine audit work.
MARKET_5M_RETENTION_HOURS = max(24, int(os.environ.get("DATA_RETENTION_5M_HOURS", "36")))
MARKET_1H_RETENTION_HOURS = max(72, int(os.environ.get("DATA_RETENTION_1H_HOURS", "168")))
NEAREST_APPROACH_RETENTION_HOURS = max(24, int(os.environ.get("DATA_RETENTION_NEAREST_HOURS", "36")))
RESEARCH_OUTCOME_RETENTION_HOURS = max(24, int(os.environ.get("DATA_RETENTION_OUTCOME_HOURS", "2160")))  # 90d
POSITION_RECONCILIATION_RETENTION_HOURS = max(24, int(os.environ.get("DATA_RETENTION_RECON_HOURS", "336")))  # 14d

# GitHub rejects individual Git blobs >= 100,000,000 bytes. We do not delete
# more data merely to hit this number; this value is reported as a warning so
# the operator can lower retention explicitly if necessary.
GITHUB_MAX_FILE_BYTES = 100_000_000
# Pre-commit guard: start compaction at 90 MB and aim for 80 MB so a normal
# workflow cycle has headroom before GitHub's hard 100 MB object limit.
GITHUB_SIZE_GUARD_BYTES = max(1_000_000, int(os.environ.get("DATA_RETENTION_SIZE_GUARD_BYTES", "90000000")))
GITHUB_SIZE_TARGET_BYTES = max(100_000, int(os.environ.get("DATA_RETENTION_SIZE_TARGET_BYTES", "80000000")))
GITHUB_WARNING_FILE_BYTES = GITHUB_SIZE_GUARD_BYTES
ARCHIVE_CHUNK_BYTES = 50_000_000
REPO_RETENTION_ARCHIVE_ROOT = DATA_DIR / "retention_archive"

ARCHIVE_ROOT = Path(
    os.environ.get(
        "DATA_RETENTION_ARCHIVE_DIR",
        str(PROJECT_ROOT.parent / "zone-engine-data-archive"),
    )
)

# These files are never rewritten by this migration. Their byte-for-byte hashes
# are captured before/after as a hard safety invariant.
NEVER_REWRITE_STATE_FILES = {
    "trades.jsonl",
    "active_trades.json",
    "entry_decisions.jsonl",
    "execution_ledger.jsonl",
    "account_context.jsonl",
    "failed_signals.json",
    "research_outcome_state.json",
    "research_bar_cursors.json",
    "research_manifest.json",
    "zone_visit_state.json",
    "event_execution_claims.json",
    "actions.jsonl",
    "signal_history.jsonl",
}


KeepFn = Callable[[int, dict[str, Any]], bool]


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


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def _file_meta(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "bytes": 0, "rows": 0, "sha256": None}
    return {
        "exists": True,
        "bytes": path.stat().st_size,
        "rows": _count_jsonl_rows(path) if rows is None else rows,
        "sha256": _sha256(path),
    }


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


def _snapshot_jsonl_locked(path: Path, snapshot_path: Path) -> dict[str, Any]:
    """Create a byte-faithful gzip snapshot of a JSONL file while holding its lock."""
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    rows = 0
    raw_bytes = 0
    fd, tmp_name = tempfile.mkstemp(
        prefix=snapshot_path.name + ".",
        suffix=".tmp",
        dir=str(snapshot_path.parent),
    )
    try:
        os.close(fd)
        with path.open("rb") as src, gzip.open(tmp_name, "wb") as gz:
            for line in src:
                gz.write(line)
                h.update(line)
                raw_bytes += len(line)
                if line.strip():
                    rows += 1
        os.replace(tmp_name, snapshot_path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return {
        "path": str(snapshot_path),
        "source_sha256": h.hexdigest(),
        "source_bytes": raw_bytes,
        "source_rows": rows,
        "compressed_bytes": snapshot_path.stat().st_size,
    }


def _snapshot_jsonl_chunked_locked(path: Path, snapshot_dir: Path) -> dict[str, Any]:
    """Create a lossless, Git-safe chunked snapshot of a JSONL file.

    Chunks contain the exact original bytes in order. Concatenating their
    decompressed streams reconstructs the source byte-for-byte. Each chunk is
    kept comfortably below GitHub's hard single-object limit.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    stem = path.name + ".before"
    h = hashlib.sha256()
    rows = 0
    raw_bytes = 0
    chunks: list[dict[str, Any]] = []
    gz = None
    part_index = 0
    part_raw_bytes = 0
    part_path: Path | None = None

    def open_part() -> None:
        nonlocal gz, part_index, part_raw_bytes, part_path
        if gz is not None:
            gz.close()
        part_index += 1
        part_raw_bytes = 0
        part_path = snapshot_dir / f"{stem}.part-{part_index:04d}.jsonl.gz"
        gz = gzip.open(part_path, "wb")

    try:
        open_part()
        with path.open("rb") as src:
            for line in src:
                if part_raw_bytes and part_raw_bytes + len(line) > ARCHIVE_CHUNK_BYTES:
                    open_part()
                assert gz is not None and part_path is not None
                gz.write(line)
                h.update(line)
                raw_bytes += len(line)
                part_raw_bytes += len(line)
                if line.strip():
                    rows += 1
        if gz is not None:
            gz.flush()
            gz.close()
            gz = None
    finally:
        if gz is not None:
            gz.close()

    for chunk in sorted(snapshot_dir.glob(f"{stem}.part-*.jsonl.gz")):
        chunks.append({
            "path": str(chunk),
            "compressed_bytes": chunk.stat().st_size,
        })
    return {
        "source_sha256": h.hexdigest(),
        "source_bytes": raw_bytes,
        "source_rows": rows,
        "chunks": chunks,
    }


def _iter_git_staged_paths() -> list[str]:
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "-z"],
        cwd=str(PROJECT_ROOT),
        check=True,
        stdout=subprocess.PIPE,
    )
    return [x for x in result.stdout.decode("utf-8", "surrogateescape").split("\x00") if x]


def check_staged_size_guard(*, limit_bytes: int = GITHUB_SIZE_GUARD_BYTES) -> dict[str, Any]:
    """Fail-safe check for every staged file under data/ before a commit."""
    oversized: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []
    import subprocess
    for rel in _iter_git_staged_paths():
        if not rel.startswith("data/") or rel.endswith("/"):
            continue
        try:
            raw_size = subprocess.check_output(
                ["git", "cat-file", "-s", f":{rel}"],
                cwd=str(PROJECT_ROOT),
                text=True,
            ).strip()
            size = int(raw_size)
        except subprocess.CalledProcessError:
            continue
        entry = {"path": rel, "bytes": size, "limit_bytes": limit_bytes}
        checked.append(entry)
        if size >= limit_bytes:
            oversized.append(entry)
    result = {"checked": checked, "oversized": oversized, "ok": not oversized}
    if oversized:
        raise RuntimeError(
            "staged data file(s) are at or above the pre-commit safety threshold "
            f"{limit_bytes} bytes: " + ", ".join(f"{x['path']}={x['bytes']}" for x in oversized)
        )
    return result


def _protected_audit_refs() -> tuple[set[str], set[str]]:
    """Return (observation_ids, generic_event_refs) tied to trade/decision state."""
    observation_ids: set[str] = set()
    generic_refs: set[str] = set()
    for name in ("trades.jsonl", "entry_decisions.jsonl"):
        path = DATA_DIR / name
        if not path.exists():
            continue
        for _, row in _iter_jsonl(path):
            for field in ("observation_id",):
                value = row.get(field)
                if value not in (None, ""):
                    observation_ids.add(str(value))
            for field in ("event_id", "attempt_id", "decision_id", "position_id", "signal_id"):
                value = row.get(field)
                if value not in (None, ""):
                    generic_refs.add(str(value))
    return observation_ids, generic_refs


def _size_guard_plan(path: Path, *, now: datetime, pending_ids: set[str], pending_bar_cutoffs: dict[str, datetime], protected_observation_ids: set[str], protected_refs: set[str]) -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    """Return progressively stronger *safe* size-reduction policies.

    Policies only archive records that are already outside the audit-critical
    working window. If a file still cannot be reduced below the guard without
    touching protected semantics, the workflow fails instead of deleting data.
    """
    name = path.name
    if name == "market_bars_5m.jsonl":
        hours = [36, 24, 12, 6, 3, 1]
        def keep_5m(row: dict[str, Any], h: int) -> bool:
            ts = _parse_ts(row.get("timestamp"))
            if ts is None:
                return True
            symbol = str(row.get("symbol", "")).upper()
            cutoff = now - timedelta(hours=h)
            pending_cutoff = pending_bar_cutoffs.get(symbol)
            if pending_cutoff is not None and pending_cutoff < cutoff:
                cutoff = pending_cutoff
            return ts >= cutoff
        return [(f"keep_newer_than_{h}h_with_pending_protection", lambda row, h=h: keep_5m(row, h)) for h in hours]
    if name == "market_bars_1h.jsonl":
        hours = [168, 72, 48, 24, 12]
        return [
            (f"keep_newer_than_{h}h", lambda row, h=h: (_parse_ts(row.get("timestamp")) is None or _parse_ts(row.get("timestamp")) >= now - timedelta(hours=h)))
            for h in hours
        ]
    if name == "research_outcomes.jsonl":
        hours = [2160, 720, 336, 168, 72, 24]
        def keep_outcome(row: dict[str, Any], h: int) -> bool:
            oid = str(row.get("observation_id", ""))
            refs = {str(row.get(f, "")) for f in ("event_id", "attempt_id", "decision_id", "position_id", "signal_id") if row.get(f) not in (None, "")}
            if oid in pending_ids or oid in protected_observation_ids or refs & protected_refs:
                return True
            ts = _parse_ts(row.get("observation_ts") or row.get("data_last_ts"))
            return ts is None or ts >= now - timedelta(hours=h)
        return [(f"keep_recent_outcome_{h}h_with_link_protection", lambda row, h=h: keep_outcome(row, h)) for h in hours]
    if name == "position_reconciliation.jsonl":
        # Size-based escalation for reconciliation is handled later by retaining
        # terminal/anomaly rows and the last healthy FOUND snapshot. The generic
        # time-only policy never drops reconciliation evidence here.
        return []
    if name == "zone_observations.jsonl":
        hours = [36, 24, 12, 6, 3, 1]
        def nearest_policy(row: dict[str, Any], h: int) -> bool:
            oid = str(row.get("observation_id", ""))
            if oid in pending_ids or oid in protected_observation_ids:
                return True
            if str(row.get("event_type", "")).upper() != "NEAREST_APPROACH":
                return True
            ts = _parse_ts(row.get("observation_ts"))
            return ts is None or ts >= now - timedelta(hours=h)
        return [(f"nearest_approach_{h}h", lambda row, h=h: nearest_policy(row, h)) for h in hours]
    return []


def _compact_reconciliation_size_guard(path: Path, *, now: datetime, run_root: Path, before: dict[str, Any]) -> list[dict[str, Any]]:
    closed_positions = _closed_position_times()
    active_refs = _active_trade_refs()
    applied: list[dict[str, Any]] = []
    # Work from a conservative set of closed-position rows only. The standard
    # reconciliation rule keeps anomalies/terminal rows and last healthy FOUND.
    for cutoff_hours in (336, 168, 72, 24):
        if _file_meta(path)["bytes"] < GITHUB_SIZE_GUARD_BYTES:
            break
        cutoff = now - timedelta(hours=cutoff_hours)
        last_healthy = _reconciliation_last_healthy_lines(
            path, active_refs=active_refs, closed_positions=closed_positions, cutoff=cutoff,
        )
        def keep(line_no: int, row: dict[str, Any], cutoff=cutoff, last_healthy=last_healthy) -> bool:
            position_id = str(row.get("position_id") or row.get("event_id") or "")
            event_id = str(row.get("event_id") or "")
            if not position_id or position_id in active_refs or event_id in active_refs:
                return True
            closed_ts = closed_positions.get(position_id) or closed_positions.get(event_id)
            if closed_ts is None or closed_ts >= cutoff:
                return True
            status = str(row.get("reconciliation_status", "")).upper()
            if status != "FOUND":
                return True
            return line_no in last_healthy
        applied.append(_compact_jsonl(
            path, keep, archive_root=run_root / "reconciliation", snapshot=False, lock_held=True,
        ))
    return applied


def _apply_size_guard_to_file(path: Path, *, now: datetime, archive_root: Path, pending_ids: set[str], pending_bar_cutoffs: dict[str, datetime], protected_observation_ids: set[str], protected_refs: set[str]) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size < GITHUB_SIZE_GUARD_BYTES:
        return {"path": str(path.relative_to(PROJECT_ROOT)), "triggered": False, "bytes_before": path.stat().st_size if path.exists() else 0}

    rel = path.relative_to(PROJECT_ROOT)
    run_root = archive_root / path.name
    run_root.mkdir(parents=True, exist_ok=True)
    before = _file_meta(path)
    with research._FileLock(path):
        snapshot = _snapshot_jsonl_chunked_locked(path, run_root / "snapshot")
        if snapshot["source_sha256"] != before["sha256"] or snapshot["source_bytes"] != before["bytes"]:
            raise RuntimeError(f"size-guard snapshot verification failed: {path}")

        applied: list[dict[str, Any]] = []
        current = before
        for name, policy in _size_guard_plan(path, now=now, pending_ids=pending_ids, pending_bar_cutoffs=pending_bar_cutoffs, protected_observation_ids=protected_observation_ids, protected_refs=protected_refs):
            if current["bytes"] < GITHUB_SIZE_GUARD_BYTES:
                break
            result = _compact_jsonl(path, lambda _line, row, policy=policy: policy(row), archive_root=run_root / "pruned", snapshot=False, lock_held=True)
            applied.append({"policy": name, **result})
            current = _file_meta(path)
            if current["bytes"] <= GITHUB_SIZE_TARGET_BYTES:
                break

        if path.name == "position_reconciliation.jsonl" and current["bytes"] >= GITHUB_SIZE_GUARD_BYTES:
            recon_results = _compact_reconciliation_size_guard(path, now=now, run_root=run_root, before=before)
            for item in recon_results:
                applied.append({"policy": "reconciliation_safe_history_compaction", **item})
                current = _file_meta(path)
                if current["bytes"] <= GITHUB_SIZE_TARGET_BYTES:
                    break

    if current["bytes"] >= GITHUB_SIZE_GUARD_BYTES:
        raise RuntimeError(
            f"size guard could not reduce {rel} below {GITHUB_SIZE_GUARD_BYTES} bytes; "
            "refusing to risk audit-critical deletion. No commit should be created."
        )
    return {
        "path": str(rel),
        "triggered": True,
        "bytes_before": before["bytes"],
        "bytes_after": current["bytes"],
        "rows_before": before["rows"],
        "rows_after": current["rows"],
        "snapshot": snapshot,
        "policies": applied,
        "target_bytes": GITHUB_SIZE_TARGET_BYTES,
        "guard_bytes": GITHUB_SIZE_GUARD_BYTES,
    }


def apply_size_guard(*, now: datetime | None = None, archive_root: Path | None = None) -> dict[str, Any]:
    now = now or _utc_now()
    archive_base = archive_root or REPO_RETENTION_ARCHIVE_ROOT
    archive_base.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    run_root = archive_base / stamp
    run_root.mkdir(parents=True, exist_ok=True)
    state_before = _validate_state_before()
    pending = _pending_observations()
    pending_ids = {str(row["observation_id"]) for row in pending}
    pending_bar_cutoffs = _pending_bar_cutoffs(
        five_m_cutoff=now - timedelta(hours=36), pending=pending,
    )
    protected_observation_ids, protected_refs = _protected_audit_refs()
    candidates = [
        DATA_DIR / "market_bars_5m.jsonl",
        DATA_DIR / "market_bars_1h.jsonl",
        DATA_DIR / "zone_observations.jsonl",
        DATA_DIR / "research_outcomes.jsonl",
        DATA_DIR / "position_reconciliation.jsonl",
    ]
    results = {"schema_version": 1, "mode": "size_guard", "started_at": now.isoformat(), "files": {}}
    for path in candidates:
        results["files"][path.name] = _apply_size_guard_to_file(
            path, now=now, archive_root=run_root, pending_ids=pending_ids,
            pending_bar_cutoffs=pending_bar_cutoffs,
            protected_observation_ids=protected_observation_ids, protected_refs=protected_refs,
        )
    _validate_state_unchanged(state_before)
    results["protected_state"] = _validate_state_before()
    results["finished_at"] = _utc_now().isoformat()
    manifest_path = DATA_DIR / "data_retention_manifest.json"
    manifest_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def _compact_jsonl(path: Path, keep_fn: KeepFn, *, archive_root: Path, snapshot: bool = True, lock_held: bool = False) -> dict[str, Any]:
    """Compact one JSONL file without loading the whole file into RAM.

    Before changing the working file, a byte-faithful gzip snapshot is made. The
    snapshot is the recovery source of truth; pruned rows are also kept in a
    smaller delta archive for convenience.
    """
    if not path.exists():
        return {
            "path": str(path),
            "before": 0,
            "after": 0,
            "pruned": 0,
            "archive": None,
            "snapshot": None,
            "sha256_before": None,
            "sha256_after": None,
        }

    snapshot_dir = archive_root / "snapshots_before"
    snapshot_path = snapshot_dir / (path.name + ".before.jsonl.gz")
    pruned_final = archive_root / (path.name + ".pruned.jsonl.gz")
    archive_root.mkdir(parents=True, exist_ok=True)

    lock_context = nullcontext() if lock_held else research._FileLock(path)
    with lock_context:
        before = _file_meta(path)
        snapshot_meta = None
        if snapshot:
            snapshot_meta = _snapshot_jsonl_locked(path, snapshot_path)
            if snapshot_meta["source_sha256"] != before["sha256"] or snapshot_meta["source_bytes"] != before["bytes"]:
                raise RuntimeError(f"pre-compaction snapshot verification failed: {path}")

        kept_tmp = None
        pruned_tmp = None
        kept_count = 0
        pruned_count = 0
        kept_bytes = 0
        try:
            fd_keep, kept_tmp = tempfile.mkstemp(
                prefix=path.name + ".kept.", suffix=".tmp", dir=str(path.parent)
            )
            with os.fdopen(fd_keep, "wb") as kept_fh:
                pruned_gz = None
                for line_no, row in _iter_jsonl(path):
                    encoded = (json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
                    if keep_fn(line_no, row):
                        kept_fh.write(encoded)
                        kept_count += 1
                        kept_bytes += len(encoded)
                    else:
                        if pruned_gz is None:
                            pruned_fd, pruned_tmp = tempfile.mkstemp(
                                prefix=path.name + ".pruned.", suffix=".tmp", dir=str(archive_root)
                            )
                            os.close(pruned_fd)
                            pruned_gz = gzip.open(pruned_tmp, "wb")
                        pruned_gz.write(encoded)
                        pruned_count += 1
                if pruned_gz is not None:
                    pruned_gz.flush()
                    pruned_gz.close()
                kept_fh.flush()
                os.fsync(kept_fh.fileno())

            archive = None
            if pruned_count:
                pruned_final.parent.mkdir(parents=True, exist_ok=True)
                os.replace(pruned_tmp, pruned_final)
                pruned_tmp = None
                archive = str(pruned_final)

            if pruned_count:
                os.replace(kept_tmp, path)
                kept_tmp = None
            else:
                # Avoid changing bytes when there is nothing to compact.
                os.unlink(kept_tmp)
                kept_tmp = None

            after = _file_meta(path, rows=kept_count if pruned_count else before["rows"])
            if pruned_count and kept_bytes != after["bytes"]:
                raise RuntimeError(f"post-compaction size accounting mismatch: {path}")
            if not pruned_count and after["sha256"] != before["sha256"]:
                raise RuntimeError(f"file changed despite zero pruning: {path}")
            return {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "before": before["rows"],
                "after": after["rows"],
                "pruned": pruned_count,
                "archive": archive,
                "snapshot": snapshot_meta,
                "sha256_before": before["sha256"],
                "sha256_after": after["sha256"],
                "bytes_before": before["bytes"],
                "bytes_after": after["bytes"],
            }
        finally:
            if kept_tmp:
                try:
                    os.unlink(kept_tmp)
                except FileNotFoundError:
                    pass
            if pruned_tmp:
                try:
                    os.unlink(pruned_tmp)
                except FileNotFoundError:
                    pass
            if pruned_gz is not None:
                try:
                    pruned_gz.close()
                except Exception:
                    pass


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


def _active_trade_refs() -> set[str]:
    path = DATA_DIR / "active_trades.json"
    if not path.exists():
        return set()
    with research._FileLock(path):
        payload = _read_json(path)
    refs: set[str] = set()
    if isinstance(payload, dict):
        for key, trade in payload.items():
            refs.add(str(key))
            if isinstance(trade, dict):
                for field in ("event_id", "position_id", "attempt_id"):
                    value = trade.get(field)
                    if value not in (None, ""):
                        refs.add(str(value))
    elif isinstance(payload, list):
        for trade in payload:
            if not isinstance(trade, dict):
                continue
            for field in ("event_id", "position_id", "attempt_id"):
                value = trade.get(field)
                if value not in (None, ""):
                    refs.add(str(value))
    return refs


def _closed_position_times() -> dict[str, datetime]:
    path = DATA_DIR / "trades.jsonl"
    out: dict[str, datetime] = {}
    if not path.exists():
        return out
    with research._FileLock(path):
        rows = list(_iter_jsonl(path))
    for _, row in rows:
        if str(row.get("record_type", "")).upper() != "TRADE_CLOSE":
            continue
        ts = _parse_ts(row.get("closed_ts") or row.get("close_ts") or row.get("timestamp"))
        if ts is None:
            continue
        refs = []
        for field in ("position_id", "event_id", "attempt_id"):
            value = row.get(field)
            if value not in (None, ""):
                refs.append(str(value))
        for ref in refs:
            previous = out.get(ref)
            if previous is None or ts > previous:
                out[ref] = ts
    return out


def _reconciliation_last_healthy_lines(
    path: Path,
    *,
    active_refs: set[str],
    closed_positions: dict[str, datetime],
    cutoff: datetime,
) -> set[int]:
    """Keep the final healthy FOUND snapshot for old closed positions."""
    if not path.exists():
        return set()
    last_line_by_position: dict[str, int] = {}
    for line_no, row in _iter_jsonl(path):
        position_id = str(row.get("position_id") or row.get("event_id") or "")
        if not position_id or position_id in active_refs:
            continue
        closed_ts = closed_positions.get(position_id) or closed_positions.get(str(row.get("event_id") or ""))
        if closed_ts is None or closed_ts >= cutoff:
            continue
        if str(row.get("reconciliation_status", "")).upper() != "FOUND":
            continue
        last_line_by_position[position_id] = line_no
    return set(last_line_by_position.values())


def _validate_state_before() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name in NEVER_REWRITE_STATE_FILES:
        path = DATA_DIR / name
        if not path.exists():
            checks[name] = {"exists": False}
            continue
        payload = None
        if path.suffix == ".json":
            try:
                payload = _read_json(path)
            except Exception:
                payload = None
        checks[name] = {
            "exists": True,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
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


def _safe_archive_dir(now: datetime, archive_base: Path | None = None) -> Path:
    base = archive_base or ARCHIVE_ROOT
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    target = base / stamp
    target.mkdir(parents=True, exist_ok=True)
    return target


def _size_status(bytes_value: int) -> dict[str, Any]:
    return {
        "bytes": bytes_value,
        "github_limit_bytes": GITHUB_MAX_FILE_BYTES,
        "github_warning_bytes": GITHUB_WARNING_FILE_BYTES,
        "over_github_limit": bytes_value >= GITHUB_MAX_FILE_BYTES,
        "near_github_limit": bytes_value >= GITHUB_WARNING_FILE_BYTES,
    }


def compact_data(*, apply: bool = False, archive_dir: Path | None = None, now: datetime | None = None) -> dict[str, Any]:
    now = now or _utc_now()
    archive_root = _safe_archive_dir(now, archive_dir)

    # One migration process at a time. Each JSONL target additionally uses the
    # same per-file lock as its normal writer, so the migration cannot race an
    # append to any file it compacts.
    with research._FileLock(DATA_DIR / "data_retention.lock"):
        return _compact_data_locked(apply=apply, archive_root=archive_root, now=now)


def _compact_data_locked(*, apply: bool, archive_root: Path, now: datetime) -> dict[str, Any]:
    state_before = _validate_state_before()
    pending = _pending_observations()
    active_refs = _active_trade_refs()
    closed_positions = _closed_position_times()

    if not apply:
        return {
            "mode": "dry_run",
            "archive_dir": str(archive_root),
            "retention_hours": {
                "market_5m": MARKET_5M_RETENTION_HOURS,
                "market_1h": MARKET_1H_RETENTION_HOURS,
                "nearest_approach": NEAREST_APPROACH_RETENTION_HOURS,
                "research_outcomes": RESEARCH_OUTCOME_RETENTION_HOURS,
                "position_reconciliation": POSITION_RECONCILIATION_RETENTION_HOURS,
            },
            "protected_state": state_before,
            "active_trade_refs": len(active_refs),
            "closed_position_refs": len(closed_positions),
            "pending_observations": _pending_summary(pending),
        }

    five_m_cutoff = now - timedelta(hours=MARKET_5M_RETENTION_HOURS)
    one_h_cutoff = now - timedelta(hours=MARKET_1H_RETENTION_HOURS)
    nearest_cutoff = now - timedelta(hours=NEAREST_APPROACH_RETENTION_HOURS)
    outcome_cutoff = now - timedelta(hours=RESEARCH_OUTCOME_RETENTION_HOURS)
    recon_cutoff = now - timedelta(hours=POSITION_RECONCILIATION_RETENTION_HOURS)

    pending = _pending_observations()
    pending_bar_cutoffs = _pending_bar_cutoffs(five_m_cutoff=five_m_cutoff, pending=pending)
    pending_ids = {str(row["observation_id"]) for row in pending}

    latest_5m = _max_jsonl_timestamp(DATA_DIR / "market_bars_5m.jsonl", "timestamp")
    if latest_5m is not None and latest_5m < five_m_cutoff:
        raise RuntimeError(
            f"market_bars_5m.jsonl newest timestamp {latest_5m.isoformat()} is "
            f"older than retention cutoff {five_m_cutoff.isoformat()}; refusing destructive compaction"
        )

    recon_path = DATA_DIR / "position_reconciliation.jsonl"
    last_healthy_recon_lines = _reconciliation_last_healthy_lines(
        recon_path,
        active_refs=active_refs,
        closed_positions=closed_positions,
        cutoff=recon_cutoff,
    )

    results: dict[str, Any] = {
        "schema_version": 2,
        "mode": "apply",
        "started_at": now.isoformat(),
        "retention_hours": {
            "market_5m": MARKET_5M_RETENTION_HOURS,
            "market_1h": MARKET_1H_RETENTION_HOURS,
            "nearest_approach": NEAREST_APPROACH_RETENTION_HOURS,
            "research_outcomes": RESEARCH_OUTCOME_RETENTION_HOURS,
            "position_reconciliation": POSITION_RECONCILIATION_RETENTION_HOURS,
        },
        "pending_observations": _pending_summary(pending),
        "active_trade_refs": len(active_refs),
        "closed_position_refs": len(closed_positions),
        "files": {},
    }

    def keep_5m(_: int, row: dict[str, Any]) -> bool:
        ts = _parse_ts(row.get("timestamp"))
        if ts is None:
            return True
        symbol = str(row.get("symbol", "")).upper()
        symbol_cutoff = pending_bar_cutoffs.get(symbol)
        if symbol_cutoff is not None:
            return ts >= symbol_cutoff
        return ts >= five_m_cutoff

    def keep_1h(_: int, row: dict[str, Any]) -> bool:
        ts = _parse_ts(row.get("timestamp"))
        return ts is None or ts >= one_h_cutoff

    def keep_observation(_: int, row: dict[str, Any]) -> bool:
        oid = str(row.get("observation_id", ""))
        if oid in pending_ids:
            return True
        if str(row.get("event_type", "")).upper() != "NEAREST_APPROACH":
            return True
        ts = _parse_ts(row.get("observation_ts"))
        return ts is None or ts >= nearest_cutoff

    def keep_outcome(_: int, row: dict[str, Any]) -> bool:
        oid = str(row.get("observation_id", ""))
        event_id = str(row.get("event_id", ""))
        if oid in pending_ids or event_id in active_refs:
            return True
        ts = _parse_ts(row.get("observation_ts") or row.get("data_last_ts"))
        return ts is None or ts >= outcome_cutoff

    def keep_reconciliation(line_no: int, row: dict[str, Any]) -> bool:
        position_id = str(row.get("position_id") or row.get("event_id") or "")
        event_id = str(row.get("event_id") or "")
        if not position_id:
            return True
        if position_id in active_refs or event_id in active_refs:
            return True
        closed_ts = closed_positions.get(position_id) or closed_positions.get(event_id)
        if closed_ts is None:
            return True
        if closed_ts >= recon_cutoff:
            return True
        # For old closed positions, preserve every anomaly/terminal event and
        # retain the last healthy FOUND snapshot so the terminal state is still
        # directly visible in the working journal. All original rows remain in
        # the pre-compaction snapshot for exact recovery.
        status = str(row.get("reconciliation_status", "")).upper()
        if status != "FOUND":
            return True
        return line_no in last_healthy_recon_lines

    results["files"]["market_bars_5m.jsonl"] = _compact_jsonl(
        DATA_DIR / "market_bars_5m.jsonl", keep_5m, archive_root=archive_root
    )
    results["files"]["market_bars_1h.jsonl"] = _compact_jsonl(
        DATA_DIR / "market_bars_1h.jsonl", keep_1h, archive_root=archive_root
    )
    results["files"]["zone_observations.jsonl"] = _compact_jsonl(
        DATA_DIR / "zone_observations.jsonl", keep_observation, archive_root=archive_root
    )
    results["files"]["research_outcomes.jsonl"] = _compact_jsonl(
        DATA_DIR / "research_outcomes.jsonl", keep_outcome, archive_root=archive_root
    )
    results["files"]["position_reconciliation.jsonl"] = _compact_jsonl(
        DATA_DIR / "position_reconciliation.jsonl", keep_reconciliation, archive_root=archive_root
    )

    _validate_state_unchanged(state_before)

    results["protected_state"] = _validate_state_before()
    results["finished_at"] = _utc_now().isoformat()
    results["git_size_status"] = {
        name: _size_status(meta.get("bytes_after", 0))
        for name, meta in results["files"].items()
        if meta.get("bytes_after") is not None
    }

    manifest_path = DATA_DIR / "data_retention_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safe one-time compaction and GitHub pre-commit data size guard")
    parser.add_argument("--apply", action="store_true", help="apply standard retention compaction")
    parser.add_argument("--size-guard", action="store_true", help="compact files that reach 90 MB before they can be staged")
    parser.add_argument("--check-staged", action="store_true", help="fail if any staged data file is >= 90 MB")
    parser.add_argument("--archive-dir", type=Path, default=None, help="archive root")
    args = parser.parse_args(argv)
    if args.check_staged:
        result = check_staged_size_guard()
    elif args.size_guard:
        result = apply_size_guard(now=_utc_now(), archive_root=args.archive_dir)
    else:
        result = compact_data(apply=args.apply, archive_dir=args.archive_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
