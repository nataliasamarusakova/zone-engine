from pathlib import Path
p=Path('/mnt/data/v6work/event_engine/data_retention.py')
s=p.read_text()
# Add size guard constants
old='''GITHUB_MAX_FILE_BYTES = 100_000_000\nGITHUB_WARNING_FILE_BYTES = 95_000_000\n'''
new='''GITHUB_MAX_FILE_BYTES = 100_000_000\n# Pre-commit guard: start compaction at 90 MB and aim for 80 MB so a normal\n# workflow cycle has headroom before GitHub's hard 100 MB object limit.\nGITHUB_SIZE_GUARD_BYTES = 90_000_000\nGITHUB_SIZE_TARGET_BYTES = 80_000_000\nGITHUB_WARNING_FILE_BYTES = GITHUB_SIZE_GUARD_BYTES\nARCHIVE_CHUNK_BYTES = 50_000_000\nREPO_RETENTION_ARCHIVE_ROOT = DATA_DIR / "retention_archive"\n'''
if old not in s: raise SystemExit('constants anchor missing')
s=s.replace(old,new)
# Insert helper functions before _compact_jsonl
anchor='''def _compact_jsonl(path: Path, keep_fn: KeepFn, *, archive_root: Path) -> dict[str, Any]:\n'''
insert=r'''def _snapshot_jsonl_chunked_locked(path: Path, snapshot_dir: Path) -> dict[str, Any]:
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
    return [x for x in result.stdout.decode("utf-8", "surrogateescape").split("\\x00") if x]


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


def _size_guard_plan(path: Path, *, now: datetime) -> list[tuple[str, Callable[[dict[str, Any]], bool]]]:
    """Return progressively stronger *safe* size-reduction policies.

    Policies only archive records that are already outside the audit-critical
    working window. If a file still cannot be reduced below the guard without
    touching protected semantics, the workflow fails instead of deleting data.
    """
    name = path.name
    if name == "market_bars_5m.jsonl":
        hours = [36, 24, 12, 6, 3, 1]
        return [
            (f"keep_newer_than_{h}h", lambda row, h=h: (_parse_ts(row.get("timestamp")) is None or _parse_ts(row.get("timestamp")) >= now - timedelta(hours=h)))
            for h in hours
        ]
    if name == "market_bars_1h.jsonl":
        hours = [168, 72, 48, 24, 12]
        return [
            (f"keep_newer_than_{h}h", lambda row, h=h: (_parse_ts(row.get("timestamp")) is None or _parse_ts(row.get("timestamp")) >= now - timedelta(hours=h)))
            for h in hours
        ]
    if name == "research_outcomes.jsonl":
        hours = [2160, 720, 336, 168, 72, 24]
        return [
            (f"keep_recent_outcome_{h}h", lambda row, h=h: (_parse_ts(row.get("observation_ts") or row.get("data_last_ts")) is None or _parse_ts(row.get("observation_ts") or row.get("data_last_ts")) >= now - timedelta(hours=h)))
            for h in hours
        ]
    if name == "position_reconciliation.jsonl":
        hours = [336, 168, 72, 24]
        return [
            (f"keep_recent_reconciliation_{h}h", lambda row, h=h: True)
            for h in hours
        ]
    if name == "zone_observations.jsonl":
        hours = [36, 24, 12, 6, 3, 1]
        def nearest_policy(row: dict[str, Any], h: int) -> bool:
            if str(row.get("event_type", "")).upper() != "NEAREST_APPROACH":
                return True
            ts = _parse_ts(row.get("observation_ts"))
            return ts is None or ts >= now - timedelta(hours=h)
        return [(f"nearest_approach_{h}h", lambda row, h=h: nearest_policy(row, h)) for h in hours]
    return []


def _apply_size_guard_to_file(path: Path, *, now: datetime, archive_root: Path) -> dict[str, Any]:
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
    for name, policy in _size_guard_plan(path, now=now):
        if current["bytes"] < GITHUB_SIZE_GUARD_BYTES:
            break
        result = _compact_jsonl(path, lambda _line, row, policy=policy: policy(row), archive_root=run_root / "pruned" )
        applied.append({"policy": name, **result})
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
    archive_base = archive_root or REPO_ARCHIVE_ROOT
    archive_base.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    run_root = archive_base / stamp
    run_root.mkdir(parents=True, exist_ok=True)
    state_before = _validate_state_before()
    candidates = [
        DATA_DIR / "market_bars_5m.jsonl",
        DATA_DIR / "market_bars_1h.jsonl",
        DATA_DIR / "zone_observations.jsonl",
        DATA_DIR / "research_outcomes.jsonl",
        DATA_DIR / "position_reconciliation.jsonl",
    ]
    results = {"schema_version": 1, "mode": "size_guard", "started_at": now.isoformat(), "files": {}}
    for path in candidates:
        results["files"][path.name] = _apply_size_guard_to_file(path, now=now, archive_root=run_root)
    _validate_state_unchanged(state_before)
    results["protected_state"] = _validate_state_before()
    results["finished_at"] = _utc_now().isoformat()
    manifest_path = DATA_DIR / "data_retention_manifest.json"
    manifest_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


'''
if anchor not in s: raise SystemExit('compact anchor missing')
s=s.replace(anchor,insert+anchor,1)
# Add CLI arguments and dispatch
old='''    parser = argparse.ArgumentParser(description="Safe one-time compaction of Zone Engine research/runtime data")\n    parser.add_argument("--apply", action="store_true", help="apply compaction; without it only validates the plan")\n    parser.add_argument("--archive-dir", type=Path, default=None, help="external archive root; a timestamped run directory is created inside it")\n    args = parser.parse_args(argv)\n    result = compact_data(apply=args.apply, archive_dir=args.archive_dir)\n    print(json.dumps(result, ensure_ascii=False, indent=2))\n    return 0\n'''
new='''    parser = argparse.ArgumentParser(description="Safe one-time compaction and GitHub pre-commit data size guard")\n    parser.add_argument("--apply", action="store_true", help="apply standard retention compaction")\n    parser.add_argument("--size-guard", action="store_true", help="compact files that reach 90 MB before they can be staged")\n    parser.add_argument("--check-staged", action="store_true", help="fail if any staged data file is >= 90 MB")\n    parser.add_argument("--archive-dir", type=Path, default=None, help="archive root")\n    args = parser.parse_args(argv)\n    if args.check_staged:\n        result = check_staged_size_guard()\n    elif args.size_guard:\n        result = apply_size_guard(now=_utc_now(), archive_root=args.archive_dir)\n    else:\n        result = compact_data(apply=args.apply, archive_dir=args.archive_dir)\n    print(json.dumps(result, ensure_ascii=False, indent=2))\n    return 0\n'''
if old not in s: raise SystemExit('main block anchor missing')
s=s.replace(old,new,1)
p.write_text(s)
