"""Integrity and upload policy for the GitHub Actions retention archive.

The retention archive is restored from the most recent successful workflow artifact.
To avoid uploading the full cumulative archive on every runner cycle, this module
requests an upload only when its content changes or its last upload is old enough
that the artifact needs a safe periodic refresh before GitHub's 90-day expiry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

STATE_SCHEMA_VERSION = 1
STATE_RECORD_TYPE = "RETENTION_ARCHIVE_CHAIN_STATE"
ARTIFACT_NAME = "zone-engine-retention-archive"
DEFAULT_REFRESH_DAYS = 45


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _archive_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError(f"retention archive path is not a directory: {root}")
    paths = sorted(root.rglob("*"), key=lambda p: p.as_posix())
    files: list[Path] = []
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in retention archive: {path}")
        if path.is_file():
            files.append(path)
    return files


def archive_fingerprint(root: Path) -> tuple[str, int]:
    """Hash relative file names and file bytes, streaming large archives safely."""
    digest = hashlib.sha256()
    files = _archive_files(root)
    digest.update(len(files).to_bytes(8, "big"))
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        path_stat_before = path.stat()
        expected_size = path_stat_before.st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(expected_size.to_bytes(8, "big"))
        read_size = 0

        def stat_signature(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                int(stat_result.st_dev),
                int(stat_result.st_ino),
                int(stat_result.st_size),
                int(stat_result.st_mtime_ns),
                int(stat_result.st_ctime_ns),
            )

        with path.open("rb") as handle:
            fd_stat_before = os.fstat(handle.fileno())
            if stat_signature(fd_stat_before) != stat_signature(path_stat_before):
                raise ValueError(f"retention archive file changed before fingerprinting: {path}")
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                read_size += len(chunk)
            fd_stat_after = os.fstat(handle.fileno())
        path_stat_after = path.stat()
        if (
            read_size != expected_size
            or stat_signature(fd_stat_before) != stat_signature(fd_stat_after)
            or stat_signature(fd_stat_before) != stat_signature(path_stat_after)
        ):
            raise ValueError(f"retention archive file changed while fingerprinting: {path}")
    return digest.hexdigest(), len(files)


def _read_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read retention archive-chain state {path}: {exc}") from exc
    valid = (
        isinstance(state, dict)
        and state.get("schema_version") == STATE_SCHEMA_VERSION
        and state.get("record_type") == STATE_RECORD_TYPE
        and state.get("artifact_name") == ARTIFACT_NAME
        and _parse_utc(state.get("initialized_at")) is not None
    )
    if not valid:
        raise ValueError(f"invalid retention archive-chain state; refusing unsafe archive operation: {path}")
    uploaded_sha = state.get("last_uploaded_archive_sha256")
    if uploaded_sha not in (None, "") and (not isinstance(uploaded_sha, str) or len(uploaded_sha) != 64 or any(c not in "0123456789abcdef" for c in uploaded_sha.lower())):
        raise ValueError(f"invalid last_uploaded_archive_sha256 in {path}")
    uploaded_at = state.get("last_uploaded_at")
    if uploaded_at not in (None, "") and _parse_utc(uploaded_at) is None:
        raise ValueError(f"invalid last_uploaded_at in {path}")
    return state


def decide_upload(
    archive_dir: Path,
    state_path: Path,
    *,
    refresh_days: int = DEFAULT_REFRESH_DAYS,
    now: datetime | None = None,
) -> dict[str, Any]:
    if refresh_days < 1 or refresh_days >= 90:
        raise ValueError("refresh_days must be between 1 and 89 to precede artifact expiration")
    fingerprint, file_count = archive_fingerprint(archive_dir)
    if file_count == 0:
        return {
            "archive_upload_required": False,
            "archive_sha256": "",
            "archive_file_count": 0,
            "archive_upload_reason": "empty_archive",
        }

    state = _read_state(state_path)
    if state is None:
        reason = "archive_state_missing"
    else:
        previous_sha = state.get("last_uploaded_archive_sha256")
        last_uploaded = _parse_utc(state.get("last_uploaded_at"))
        if not previous_sha or last_uploaded is None:
            reason = "upload_metadata_missing"
        elif previous_sha != fingerprint:
            reason = "archive_content_changed"
        else:
            current = (now or _utc_now()).astimezone(timezone.utc)
            if current - last_uploaded >= timedelta(days=refresh_days):
                reason = "periodic_refresh_due"
            else:
                reason = "archive_unchanged_recent_upload"
    return {
        "archive_upload_required": reason != "archive_unchanged_recent_upload",
        "archive_sha256": fingerprint,
        "archive_file_count": file_count,
        "archive_upload_reason": reason,
    }


def mark_uploaded(
    archive_dir: Path,
    state_path: Path,
    *,
    expected_sha256: str,
    run_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    fingerprint, file_count = archive_fingerprint(archive_dir)
    if file_count == 0:
        raise ValueError("cannot mark an empty retention archive as uploaded")
    if not expected_sha256 or fingerprint != expected_sha256:
        raise ValueError("retention archive changed between upload decision and marker update")

    timestamp = (now or _utc_now()).astimezone(timezone.utc).isoformat()
    old = _read_state(state_path)
    state: dict[str, Any] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "record_type": STATE_RECORD_TYPE,
        "artifact_name": ARTIFACT_NAME,
        "initialized_at": old.get("initialized_at") if old else timestamp,
        "initialized_by_run_id": old.get("initialized_by_run_id") if old else run_id,
        "last_uploaded_at": timestamp,
        "last_uploaded_archive_sha256": fingerprint,
        "last_uploaded_by_run_id": run_id,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{state_path.name}.", suffix=".tmp", dir=str(state_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, state_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return state


def verify_restored_archive(archive_dir: Path, state_path: Path) -> dict[str, Any]:
    """Validate downloaded archive bytes against the previously committed marker.

    Older markers without a recorded fingerprint are accepted for a one-time
    migration; the post-size-guard upload decision will force a verified upload.
    """
    fingerprint, file_count = archive_fingerprint(archive_dir)
    state = _read_state(state_path)
    if state is None:
        return {"archive_restore_verified": True, "archive_restore_status": "no_marker"}
    expected = state.get("last_uploaded_archive_sha256")
    if not expected:
        return {"archive_restore_verified": True, "archive_restore_status": "legacy_marker_without_fingerprint"}
    if file_count == 0:
        raise ValueError("marker records an uploaded archive, but the restored archive directory is empty")
    if fingerprint != expected:
        raise ValueError(
            "restored retention archive fingerprint mismatch "
            f"(marker={expected}, actual={fingerprint})"
        )
    return {
        "archive_restore_verified": True,
        "archive_restore_status": "fingerprint_match",
        "archive_sha256": fingerprint,
        "archive_file_count": file_count,
    }


def verify_uploaded_state(
    archive_dir: Path,
    state_path: Path,
    *,
    expected_sha256: str,
    refresh_days: int = DEFAULT_REFRESH_DAYS,
    now: datetime | None = None,
) -> None:
    fingerprint, file_count = archive_fingerprint(archive_dir)
    if file_count == 0:
        raise ValueError("retention archive is empty")
    if not expected_sha256 or fingerprint != expected_sha256:
        raise ValueError("retention archive content changed after upload decision")
    state = _read_state(state_path)
    if state is None:
        raise ValueError("retention archive-chain state marker is missing")
    if state.get("last_uploaded_archive_sha256") != fingerprint:
        raise ValueError("retention archive fingerprint does not match the last uploaded artifact")
    uploaded_at = _parse_utc(state.get("last_uploaded_at"))
    if uploaded_at is None:
        raise ValueError("retention archive last-upload time is missing or invalid")
    current = (now or _utc_now()).astimezone(timezone.utc)
    age = current - uploaded_at
    if age < timedelta(0) or age >= timedelta(days=refresh_days):
        raise ValueError("retention archive requires a fresh upload before commit")


def _write_github_output(path: str | None, result: dict[str, Any]) -> None:
    lines = []
    for key, value in result.items():
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        lines.append(f"{key}={rendered}\n")
    if path:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.writelines(lines)
    else:
        print("".join(lines), end="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--decide-upload", action="store_true")
    mode.add_argument("--mark-upload", action="store_true")
    mode.add_argument("--verify-upload", action="store_true")
    mode.add_argument("--verify-restored", action="store_true")
    parser.add_argument("--archive-dir", type=Path, default=Path("data/retention_archive"))
    parser.add_argument("--state-file", type=Path, default=Path("data/retention_archive_state.json"))
    parser.add_argument("--refresh-days", type=int, default=DEFAULT_REFRESH_DAYS)
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    args = parser.parse_args()

    try:
        if args.decide_upload:
            result = decide_upload(args.archive_dir, args.state_file, refresh_days=args.refresh_days)
            _write_github_output(args.github_output, result)
        elif args.mark_upload:
            state = mark_uploaded(
                args.archive_dir,
                args.state_file,
                expected_sha256=args.expected_sha256,
                run_id=args.run_id,
            )
            _write_github_output(args.github_output, {
                "marker_updated": True,
                "archive_sha256": state["last_uploaded_archive_sha256"],
            })
        elif args.verify_upload:
            verify_uploaded_state(
                args.archive_dir,
                args.state_file,
                expected_sha256=args.expected_sha256,
                refresh_days=args.refresh_days,
            )
            _write_github_output(args.github_output, {"archive_state_verified": True})
        elif args.verify_restored:
            _write_github_output(args.github_output, verify_restored_archive(args.archive_dir, args.state_file))
    except (OSError, ValueError) as exc:
        print(f"retention archive policy error: {exc}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
