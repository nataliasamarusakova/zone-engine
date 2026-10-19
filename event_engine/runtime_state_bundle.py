from __future__ import annotations

"""Build, verify, and restore the persisted runtime state stored as a GH Actions artifact.

The Git repository should contain code and configuration, not the large per-run JSONL
journals. This module makes the state artifact self-verifying and fails closed before
trading if an artifact is missing or incomplete after migration.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

BUNDLE_FORMAT = "zone-engine-runtime-state"
BUNDLE_SCHEMA_VERSION = 1
MANIFEST_NAME = "runtime-state-manifest.json"

# Files managed by a separate retention-artifact chain or generated locally are not
# part of the portable snapshot. The retention archive must be restored and verified
# independently before size-guard compaction can run.
EXCLUDED_DATA_FILES = {
    "zone_engine_diagnostic.log",
    "scan_history.jsonl",
}
EXCLUDED_DATA_DIRS = {"retention_archive", "__pycache__"}
REQUIRED_FILES = (
    # Trade/execution/idempotency state and the journals needed by reconciliation.
    "data/active_trades.json",
    "data/event_execution_claims.json",
    "data/execution_ledger.jsonl",
    "data/failed_signals.json",
    "data/latest_scan.json",
    "data/latest_scan.txt",
    "data/order_lifecycle.jsonl",
    "data/position_reconciliation.jsonl",
    "data/protection_lifecycle.jsonl",
    "data/quote_snapshots.jsonl",
    "data/signal_history.jsonl",
    "data/trades.jsonl",
    "data/zone_visit_state.json",
    # Research state/cursors and the working history which research_forward reads.
    "data/account_context.jsonl",
    "data/counterfactual_experiments.jsonl",
    "data/entry_decisions.jsonl",
    "data/exchange_errors.jsonl",
    "data/market_bars_1h.jsonl",
    "data/market_bars_5m.jsonl",
    "data/market_context.jsonl",
    "data/research_bar_cursors.json",
    "data/research_manifest.json",
    "data/research_outcome_state.json",
    "data/research_outcomes.jsonl",
    "data/research_outcomes_restore_state.json",
    "data/zone_observations.jsonl",
    # Retention state/manifest are persisted here; the archive payload itself is separate.
    "data/data_retention_manifest.json",
    "data/retention_archive_state.json",
    "data/runtime_version.json",
    # Immutable seed history required to keep forward research reproducible.
    "bootstrap/research_history_seed/2026-10-16/manifest.json",
    "bootstrap/research_history_seed/2026-10-16/market_bars_5m.jsonl.before_2026-10-16.jsonl.gz",
    "bootstrap/research_history_seed/2026-10-16/research_outcomes.jsonl.before_2026-10-16.jsonl.gz",
)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_excluded_data_path(relative: Path) -> bool:
    if any(part in EXCLUDED_DATA_DIRS for part in relative.parts):
        return True
    name = relative.name
    if name in EXCLUDED_DATA_FILES or name.startswith("scan_history.jsonl."):
        return True
    if name.endswith((".tmp", ".lock", ".pyc")):
        return True
    return False


def _collect_source_files(data_dir: Path, seed_dir: Path) -> list[tuple[Path, str]]:
    result: list[tuple[Path, str]] = []
    if not data_dir.is_dir():
        raise RuntimeError(f"Runtime data directory is missing: {data_dir}")
    if not seed_dir.is_dir():
        raise RuntimeError(f"Research history seed directory is missing: {seed_dir}")

    for path in sorted(data_dir.rglob("*")):
        rel = path.relative_to(data_dir)
        if _is_excluded_data_path(rel):
            continue
        if path.is_symlink():
            raise RuntimeError(f"Refusing to bundle symlink in runtime data: {path}")
        if path.is_file():
            result.append((path, (Path("data") / rel).as_posix()))

    for path in sorted(seed_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Refusing to bundle symlink in research seed: {path}")
        if path.is_file():
            rel = path.relative_to(seed_dir)
            result.append((path, (Path("bootstrap") / "research_history_seed" / rel).as_posix()))

    return sorted(result, key=lambda item: item[1])


def create_bundle(
    *, data_dir: Path, seed_dir: Path, output_dir: Path,
    source_commit: str = "", run_id: str = "",
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    seed_dir = seed_dir.resolve()
    output_dir = output_dir.resolve()
    if _paths_overlap(output_dir, data_dir) or _paths_overlap(output_dir, seed_dir):
        raise RuntimeError("Bundle output must be outside runtime data and seed directories")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "data").mkdir(parents=True)
    (output_dir / "bootstrap" / "research_history_seed").mkdir(parents=True)

    source_files = _collect_source_files(data_dir, seed_dir)
    records: list[dict[str, Any]] = []
    for source, relative in source_files:
        target = output_dir / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({
            "path": relative,
            "size_bytes": target.stat().st_size,
            "sha256": _sha256(target),
        })

    _validate_required_paths({item["path"] for item in records})
    manifest = {
        "format": BUNDLE_FORMAT,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "run_id": run_id,
        "files": records,
    }
    manifest_path = output_dir / MANIFEST_NAME
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())

    # Read back all entries now so an incomplete local bundle can never be uploaded.
    verify_bundle(output_dir)
    total_bytes = sum(item["size_bytes"] for item in records)
    print(f"Created verified runtime-state bundle: files={len(records)} bytes={total_bytes}")
    return manifest


def check_legacy_snapshot(*, data_dir: Path, seed_dir: Path) -> None:
    data_dir = data_dir.resolve()
    seed_dir = seed_dir.resolve()
    missing = []
    for relative in REQUIRED_FILES:
        if relative.startswith("data/"):
            candidate = data_dir / relative.removeprefix("data/")
        else:
            candidate = seed_dir / Path(*PurePosixPath(relative).parts[2:])
        if not candidate.is_file() or candidate.is_symlink():
            missing.append(relative)
    if missing:
        raise RuntimeError("Legacy runtime snapshot is incomplete: " + ", ".join(sorted(missing)))
    print(f"Legacy runtime snapshot is complete: {len(REQUIRED_FILES)} required files")


def _validate_required_paths(paths: set[str]) -> None:
    missing = sorted(set(REQUIRED_FILES) - paths)
    if missing:
        raise RuntimeError("Runtime-state bundle is missing required files: " + ", ".join(missing))


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(f"Runtime-state manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Runtime-state manifest cannot be read: {exc}") from exc
    if manifest.get("format") != BUNDLE_FORMAT or manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise RuntimeError("Runtime-state bundle format/schema is unsupported")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Runtime-state manifest contains no file inventory")

    expected: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("Malformed runtime-state file inventory entry")
        rel = record.get("path")
        if not isinstance(rel, str) or not rel:
            raise RuntimeError("Runtime-state inventory has an invalid path")
        parsed = PurePosixPath(rel)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in rel:
            raise RuntimeError(f"Unsafe runtime-state manifest path: {rel!r}")
        if not (rel.startswith("data/") or rel.startswith("bootstrap/research_history_seed/")):
            raise RuntimeError(f"Unexpected path in runtime-state bundle: {rel!r}")
        if rel in expected:
            raise RuntimeError(f"Duplicate path in runtime-state manifest: {rel}")
        size = record.get("size_bytes")
        sha = record.get("sha256")
        if not isinstance(size, int) or size < 0 or not isinstance(sha, str) or len(sha) != 64:
            raise RuntimeError(f"Invalid size/checksum for runtime-state file: {rel}")
        expected[rel] = record

    _validate_required_paths(set(expected))
    actual: set[str] = set()
    for path in bundle_dir.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"Refusing symlink in downloaded runtime-state artifact: {path}")
        if not path.is_file() or path == manifest_path:
            continue
        rel = path.relative_to(bundle_dir).as_posix()
        actual.add(rel)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        bits = []
        if missing:
            bits.append("missing=" + ",".join(missing[:10]))
        if extra:
            bits.append("unexpected=" + ",".join(extra[:10]))
        raise RuntimeError("Runtime-state artifact inventory mismatch: " + " ".join(bits))

    for rel, record in expected.items():
        path = bundle_dir / Path(rel)
        if path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"Runtime-state file size mismatch: {rel}")
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Runtime-state SHA-256 mismatch: {rel}")
    print(f"Verified runtime-state bundle: files={len(expected)}")
    return manifest


def restore_bundle(*, bundle_dir: Path, data_dir: Path, seed_dir: Path) -> None:
    bundle_dir = bundle_dir.resolve()
    data_dir = data_dir.resolve()
    seed_dir = seed_dir.resolve()
    verify_bundle(bundle_dir)

    if data_dir == seed_dir or data_dir in seed_dir.parents or seed_dir in data_dir.parents:
        raise RuntimeError("Runtime data and research seed targets must be separate directories")
    if _paths_overlap(bundle_dir, data_dir) or _paths_overlap(bundle_dir, seed_dir):
        raise RuntimeError("Bundle path must not overlap restore targets")

    # Stage both directory trees before replacing either target, then retain backups
    # so a local filesystem failure can be rolled back before the bot starts.
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    seed_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zone-runtime-restore-") as temporary:
        tmp = Path(temporary)
        staged_data = tmp / "staged-data"
        staged_seed = tmp / "staged-seed"
        shutil.copytree(bundle_dir / "data", staged_data)
        shutil.copytree(bundle_dir / "bootstrap" / "research_history_seed", staged_seed)
        backup_data = tmp / "backup-data"
        backup_seed = tmp / "backup-seed"
        had_data = data_dir.exists()
        had_seed = seed_dir.exists()
        moved_data = moved_seed = installed_data = installed_seed = False
        try:
            if had_data:
                data_dir.rename(backup_data)
                moved_data = True
            if had_seed:
                seed_dir.rename(backup_seed)
                moved_seed = True
            staged_data.rename(data_dir)
            installed_data = True
            staged_seed.rename(seed_dir)
            installed_seed = True
        except Exception:
            if installed_seed and seed_dir.exists():
                shutil.rmtree(seed_dir)
            if installed_data and data_dir.exists():
                shutil.rmtree(data_dir)
            if moved_data and backup_data.exists():
                backup_data.rename(data_dir)
            if moved_seed and backup_seed.exists():
                backup_seed.rename(seed_dir)
            raise
    print("Restored verified runtime state and research-history seed from artifact.")


def _snapshot_excluded(relative: Path) -> bool:
    return _is_excluded_data_path(relative)


def data_snapshot(data_dir: Path) -> dict[str, dict[str, Any]]:
    if not data_dir.is_dir():
        raise RuntimeError(f"Runtime data directory is missing: {data_dir}")
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(data_dir.rglob("*")):
        relative = path.relative_to(data_dir)
        if _snapshot_excluded(relative):
            continue
        if path.is_symlink():
            raise RuntimeError(f"Refusing symlink in runtime data snapshot: {path}")
        if path.is_file():
            inventory[relative.as_posix()] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return inventory


def write_data_snapshot(*, data_dir: Path, output_path: Path) -> None:
    payload = {"schema_version": 1, "files": data_snapshot(data_dir)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    print(f"Snapshotted runtime data before tests: files={len(payload['files'])}")


def verify_data_unchanged(*, data_dir: Path, snapshot_path: Path) -> None:
    try:
        before_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Pre-test runtime-data snapshot is missing or invalid: {exc}") from exc
    if before_payload.get("schema_version") != 1 or not isinstance(before_payload.get("files"), dict):
        raise RuntimeError("Pre-test runtime-data snapshot has an unsupported format")
    after = data_snapshot(data_dir)
    before = before_payload["files"]
    if before != after:
        before_paths, after_paths = set(before), set(after)
        missing = sorted(before_paths - after_paths)
        added = sorted(after_paths - before_paths)
        changed = sorted(path for path in before_paths & after_paths if before[path] != after[path])
        details = []
        if added:
            details.append("added=" + ",".join(added[:8]))
        if missing:
            details.append("missing=" + ",".join(missing[:8]))
        if changed:
            details.append("changed=" + ",".join(changed[:8]))
        raise RuntimeError("Tests mutated production runtime data: " + " ".join(details))
    print(f"Runtime-data integrity verified after tests: files={len(after)} unchanged")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Build a self-verifying runtime-state bundle")
    create.add_argument("--data-dir", type=Path, default=Path("data"))
    create.add_argument("--seed-dir", type=Path, default=Path("bootstrap/research_history_seed"))
    create.add_argument("--output-dir", type=Path, default=Path("runtime-state-bundle"))
    create.add_argument("--source-commit", default=os.environ.get("GITHUB_SHA", ""))
    create.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", ""))

    legacy = sub.add_parser("check-legacy", help="Confirm a complete legacy Git state exists before first migration")
    legacy.add_argument("--data-dir", type=Path, default=Path("data"))
    legacy.add_argument("--seed-dir", type=Path, default=Path("bootstrap/research_history_seed"))

    verify = sub.add_parser("verify", help="Verify a downloaded bundle without restoring it")
    verify.add_argument("--bundle-dir", type=Path, required=True)

    restore = sub.add_parser("restore", help="Verify and restore an artifact bundle")
    restore.add_argument("--bundle-dir", type=Path, required=True)
    restore.add_argument("--data-dir", type=Path, default=Path("data"))
    restore.add_argument("--seed-dir", type=Path, default=Path("bootstrap/research_history_seed"))

    snapshot = sub.add_parser("snapshot", help="Snapshot persistent runtime data before pytest")
    snapshot.add_argument("--data-dir", type=Path, default=Path("data"))
    snapshot.add_argument("--output", type=Path, required=True)

    unchanged = sub.add_parser("verify-unchanged", help="Verify pytest did not alter runtime data")
    unchanged.add_argument("--data-dir", type=Path, default=Path("data"))
    unchanged.add_argument("--snapshot", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            create_bundle(
                data_dir=args.data_dir, seed_dir=args.seed_dir, output_dir=args.output_dir,
                source_commit=args.source_commit, run_id=args.run_id,
            )
        elif args.command == "check-legacy":
            check_legacy_snapshot(data_dir=args.data_dir, seed_dir=args.seed_dir)
        elif args.command == "verify":
            verify_bundle(args.bundle_dir)
        elif args.command == "restore":
            restore_bundle(bundle_dir=args.bundle_dir, data_dir=args.data_dir, seed_dir=args.seed_dir)
        elif args.command == "snapshot":
            write_data_snapshot(data_dir=args.data_dir, output_path=args.output)
        elif args.command == "verify-unchanged":
            verify_data_unchanged(data_dir=args.data_dir, snapshot_path=args.snapshot)
        else:  # pragma: no cover - argparse prevents this
            parser.error("unknown command")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
