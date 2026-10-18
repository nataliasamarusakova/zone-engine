#!/usr/bin/env python3
"""Idempotently restore archived research outcomes into the live JSONL journal.

The immutable research_history_seed is a recovery source, not a live execution
state. This utility restores only research_outcomes.jsonl, preserves source row
bytes, verifies the seed against its manifest, and writes a durable marker so a
later size-guard compaction does not cause the same old rows to be re-added.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_FILENAME = "research_outcomes_restore_state.json"
MANIFEST_RECORD_TYPE = "RESEARCH_HISTORY_RECOVERY_SEED"
STATE_RECORD_TYPE = "RESEARCH_OUTCOME_SEED_RESTORE_STATE"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _parse_jsonl(raw: bytes, label: str) -> list[tuple[str, bytes, dict[str, Any]]]:
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"{label}: JSONL file does not end with a newline; refusing to merge")
    result: list[tuple[str, bytes, dict[str, Any]]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.strip():
            continue
        content = line.rstrip(b"\r\n")
        try:
            row = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_no}: expected JSON object")
        outcome_id = str(row.get("outcome_id", "")).strip()
        if not outcome_id:
            raise ValueError(f"{label}:{line_no}: missing outcome_id")
        if outcome_id in seen:
            raise ValueError(f"{label}:{line_no}: duplicate outcome_id={outcome_id}")
        seen.add(outcome_id)
        result.append((outcome_id, line, row))
    return result


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _atomic_write(path, raw)


def restore_research_outcomes_seed(data_dir: Path) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    seed_root = data_dir.parent / "bootstrap" / "research_history_seed"
    target = data_dir / "research_outcomes.jsonl"
    # The marker belongs to runtime state so it survives the workflow's data commit.
    # The immutable seed itself is source/bootstrap material and is never mutated.
    state_path = data_dir / STATE_FILENAME

    if not seed_root.exists():
        return {"status": "no_seed_directory", "target": str(target), "merged_rows": 0}

    manifests = sorted(p for p in seed_root.rglob("manifest.json") if p.is_file())
    seed_specs: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        if manifest.get("record_type") != MANIFEST_RECORD_TYPE:
            continue
        files = manifest.get("files")
        outcomes_meta = files.get("research_outcomes.jsonl") if isinstance(files, dict) else None
        if not isinstance(outcomes_meta, dict):
            continue
        archive_rel = Path(str(outcomes_meta.get("archive_path", "")))
        if archive_rel.is_absolute():
            raise ValueError(f"{manifest_path}: absolute archive path is not allowed")
        archive_path = (data_dir.parent / archive_rel).resolve()
        if not archive_path.is_relative_to(seed_root.resolve()):
            raise ValueError(f"{manifest_path}: archive path escapes recovery seed directory")
        if not archive_path.is_file():
            raise FileNotFoundError(f"{manifest_path}: recovery seed archive is missing: {archive_path}")
        seed_specs.append((archive_path, manifest, outcomes_meta))

    if not seed_specs:
        return {"status": "no_outcome_seed", "target": str(target), "merged_rows": 0}

    if state_path.exists():
        state = _read_json(state_path)
        if state.get("record_type") != STATE_RECORD_TYPE or not isinstance(state.get("sources", {}), dict):
            raise ValueError(f"{state_path}: invalid restore-state schema")
    else:
        state = {"schema_version": 1, "record_type": STATE_RECORD_TYPE, "sources": {}}
    state_sources: dict[str, Any] = dict(state.get("sources", {}))

    current_raw = target.read_bytes() if target.exists() else b""
    current_rows = _parse_jsonl(current_raw, str(target)) if current_raw else []
    current_by_id = {outcome_id: line for outcome_id, line, _ in current_rows}
    all_actions: list[dict[str, Any]] = []
    total_added = 0

    for archive_path, manifest, meta in seed_specs:
        expected_sha = str(meta.get("source_sha256", "")).lower()
        if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
            raise ValueError(f"{archive_path}: invalid source_sha256 in seed manifest")

        # Once a source has been restored, its marker must survive subsequent
        # retention compaction. Do not re-inflate pruned journals on each runner.
        if expected_sha in state_sources:
            all_actions.append({
                "archive": str(archive_path.relative_to(data_dir.parent)),
                "source_sha256": expected_sha,
                "status": "already_marked_restored",
                "added_rows": 0,
            })
            continue

        expected_compressed_bytes = meta.get("compressed_bytes")
        if expected_compressed_bytes is not None and archive_path.stat().st_size != int(expected_compressed_bytes):
            raise ValueError(
                f"{archive_path}: compressed byte-count mismatch "
                f"(expected {int(expected_compressed_bytes)}, got {archive_path.stat().st_size})"
            )
        try:
            seed_raw = gzip.decompress(archive_path.read_bytes())
        except (OSError, EOFError, zlib.error) as exc:
            raise ValueError(f"{archive_path}: cannot decompress seed: {exc}") from exc

        actual_sha = _sha256(seed_raw)
        expected_bytes = int(meta.get("source_bytes", -1))
        if actual_sha != expected_sha:
            raise ValueError(f"{archive_path}: source SHA-256 mismatch (expected {expected_sha}, got {actual_sha})")
        if len(seed_raw) != expected_bytes:
            raise ValueError(f"{archive_path}: source byte-count mismatch (expected {expected_bytes}, got {len(seed_raw)})")

        seed_rows = _parse_jsonl(seed_raw, str(archive_path))
        expected_rows = int(meta.get("source_rows", -1))
        if len(seed_rows) != expected_rows:
            raise ValueError(f"{archive_path}: source row-count mismatch (expected {expected_rows}, got {len(seed_rows)})")

        seed_by_id = {outcome_id: line for outcome_id, line, _ in seed_rows}
        overlap = set(seed_by_id) & set(current_by_id)
        for outcome_id in overlap:
            if seed_by_id[outcome_id].rstrip(b"\r\n") != current_by_id[outcome_id].rstrip(b"\r\n"):
                raise ValueError(
                    f"{target}: conflicting contents for outcome_id={outcome_id}; "
                    "refusing to silently choose between live and seed rows"
                )

        missing_seed_rows = [(outcome_id, line, row) for outcome_id, line, row in seed_rows if outcome_id not in current_by_id]
        if missing_seed_rows:
            # Keep the recovered source block byte-identical, followed by current
            # rows that are not already present in the recovery source.
            seed_ids = set(seed_by_id)
            live_tail = b"".join(line for outcome_id, line, _ in current_rows if outcome_id not in seed_ids)
            merged_raw = seed_raw + live_tail
            _parse_jsonl(merged_raw, f"merged {target}")
            _atomic_write(target, merged_raw)
            current_raw = merged_raw
            current_rows = _parse_jsonl(current_raw, str(target))
            current_by_id = {outcome_id: line for outcome_id, line, _ in current_rows}

        target_sha = _sha256(current_raw)
        source_rows_present = len(seed_rows) - len(missing_seed_rows)
        record = {
            "archive": str(archive_path.relative_to(data_dir.parent)),
            "source_sha256": expected_sha,
            "source_bytes": len(seed_raw),
            "source_rows": len(seed_rows),
            "rows_added": len(missing_seed_rows),
            "source_rows_already_present": source_rows_present,
            "live_rows_after_merge": len(current_rows),
            "live_bytes_after_merge": len(current_raw),
            "live_sha256_after_merge": target_sha,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        state_sources[expected_sha] = record
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["sources"] = state_sources
        _atomic_write_json(state_path, state)
        total_added += len(missing_seed_rows)
        all_actions.append({**record, "status": "restored" if missing_seed_rows else "already_present"})

    return {
        "status": "ok",
        "target": str(target),
        "target_rows": len(current_rows),
        "target_bytes": len(current_raw),
        "target_sha256": _sha256(current_raw),
        "merged_rows": total_added,
        "sources": all_actions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restore immutable research outcome seed into live journal exactly once")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args(argv)
    try:
        result = restore_research_outcomes_seed(args.data_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") != "ok":
        # Missing seed is an observable deployment/data-recovery failure, not a
        # successful no-op. The workflow marks this step continue-on-error so
        # current position management can proceed safely.
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
