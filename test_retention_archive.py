from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from event_engine.retention_archive import (
    ARTIFACT_NAME,
    STATE_RECORD_TYPE,
    STATE_SCHEMA_VERSION,
    archive_fingerprint,
    decide_upload,
    mark_uploaded,
    verify_uploaded_state,
    verify_restored_archive,
)

UTC = timezone.utc
NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


def _seed_archive(root: Path) -> None:
    (root / "generation-1").mkdir(parents=True)
    (root / "generation-1" / "delta.jsonl.gz").write_bytes(b"compressed-archive-payload")


def _write_state(path: Path, *, uploaded_at: datetime, sha: str) -> None:
    path.write_text(json.dumps({
        "schema_version": STATE_SCHEMA_VERSION,
        "record_type": STATE_RECORD_TYPE,
        "artifact_name": ARTIFACT_NAME,
        "initialized_at": NOW.isoformat(),
        "initialized_by_run_id": "100",
        "last_uploaded_at": uploaded_at.isoformat(),
        "last_uploaded_archive_sha256": sha,
        "last_uploaded_by_run_id": "100",
    }), encoding="utf-8")


def test_archive_fingerprint_tracks_paths_and_bytes(tmp_path):
    archive = tmp_path / "retention_archive"
    _seed_archive(archive)
    first_sha, first_count = archive_fingerprint(archive)
    same_sha, same_count = archive_fingerprint(archive)
    assert first_sha == same_sha
    assert first_count == same_count == 1

    payload = archive / "generation-1" / "delta.jsonl.gz"
    payload.write_bytes(b"changed-payload")
    second_sha, _ = archive_fingerprint(archive)
    assert second_sha != first_sha
    payload.rename(payload.with_name("delta-renamed.jsonl.gz"))
    third_sha, _ = archive_fingerprint(archive)
    assert third_sha != second_sha


def test_empty_archive_does_not_request_upload(tmp_path):
    result = decide_upload(tmp_path / "empty", tmp_path / "state.json", now=NOW)
    assert result["archive_upload_required"] is False
    assert result["archive_file_count"] == 0
    assert result["archive_upload_reason"] == "empty_archive"


def test_first_nonempty_archive_requires_upload_and_mark_is_atomic_state(tmp_path):
    archive = tmp_path / "retention_archive"
    state = tmp_path / "data" / "retention_archive_state.json"
    _seed_archive(archive)
    decision = decide_upload(archive, state, now=NOW)
    assert decision["archive_upload_required"] is True
    assert decision["archive_upload_reason"] == "archive_state_missing"

    marker = mark_uploaded(archive, state, expected_sha256=decision["archive_sha256"], run_id="123", now=NOW)
    assert marker["last_uploaded_archive_sha256"] == decision["archive_sha256"]
    assert marker["last_uploaded_by_run_id"] == "123"
    assert json.loads(state.read_text(encoding="utf-8"))["record_type"] == STATE_RECORD_TYPE
    verify_uploaded_state(archive, state, expected_sha256=decision["archive_sha256"], now=NOW + timedelta(days=44))


def test_unchanged_archive_skips_upload_until_safe_refresh_window(tmp_path):
    archive = tmp_path / "retention_archive"
    state = tmp_path / "state.json"
    _seed_archive(archive)
    fingerprint, _ = archive_fingerprint(archive)
    _write_state(state, uploaded_at=NOW, sha=fingerprint)

    recent = decide_upload(archive, state, now=NOW + timedelta(days=44))
    assert recent["archive_upload_required"] is False
    assert recent["archive_upload_reason"] == "archive_unchanged_recent_upload"

    due = decide_upload(archive, state, now=NOW + timedelta(days=45))
    assert due["archive_upload_required"] is True
    assert due["archive_upload_reason"] == "periodic_refresh_due"


def test_changed_archive_requires_upload_even_when_previous_upload_is_recent(tmp_path):
    archive = tmp_path / "retention_archive"
    state = tmp_path / "state.json"
    _seed_archive(archive)
    fingerprint, _ = archive_fingerprint(archive)
    _write_state(state, uploaded_at=NOW, sha=fingerprint)
    (archive / "generation-2").mkdir()
    (archive / "generation-2" / "new-delta.gz").write_bytes(b"new")

    decision = decide_upload(archive, state, now=NOW + timedelta(days=1))
    assert decision["archive_upload_required"] is True
    assert decision["archive_upload_reason"] == "archive_content_changed"


def test_marker_update_rejects_archive_changed_after_decision(tmp_path):
    archive = tmp_path / "retention_archive"
    state = tmp_path / "state.json"
    _seed_archive(archive)
    decision = decide_upload(archive, state, now=NOW)
    (archive / "generation-1" / "delta.jsonl.gz").write_bytes(b"changed after decision")
    with pytest.raises(ValueError, match="changed between upload decision and marker update"):
        mark_uploaded(archive, state, expected_sha256=decision["archive_sha256"], now=NOW)


def test_corrupt_marker_fails_closed(tmp_path):
    archive = tmp_path / "retention_archive"
    state = tmp_path / "state.json"
    _seed_archive(archive)
    state.write_text('{"record_type":"wrong"}', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid retention archive-chain state"):
        decide_upload(archive, state, now=NOW)


def test_legacy_marker_without_upload_fingerprint_forces_one_refresh(tmp_path):
    archive = tmp_path / "retention_archive"
    state = tmp_path / "state.json"
    _seed_archive(archive)
    state.write_text(json.dumps({
        "schema_version": STATE_SCHEMA_VERSION,
        "record_type": STATE_RECORD_TYPE,
        "artifact_name": ARTIFACT_NAME,
        "initialized_at": NOW.isoformat(),
        "initialized_by_run_id": "100",
    }), encoding="utf-8")
    decision = decide_upload(archive, state, now=NOW + timedelta(days=1))
    assert decision["archive_upload_required"] is True
    assert decision["archive_upload_reason"] == "upload_metadata_missing"


def test_verify_upload_rejects_stale_marker(tmp_path):
    archive = tmp_path / "retention_archive"
    state = tmp_path / "state.json"
    _seed_archive(archive)
    fingerprint, _ = archive_fingerprint(archive)
    _write_state(state, uploaded_at=NOW, sha=fingerprint)
    with pytest.raises(ValueError, match="requires a fresh upload"):
        verify_uploaded_state(
            archive, state, expected_sha256=fingerprint, now=NOW + timedelta(days=45)
        )


def test_verify_restored_archive_without_marker_is_allowed_for_initial_migration(tmp_path):
    archive = tmp_path / "retention_archive"
    _seed_archive(archive)
    result = verify_restored_archive(archive, tmp_path / "missing-state.json")
    assert result["archive_restore_verified"] is True
    assert result["archive_restore_status"] == "no_marker"


def test_verify_restored_archive_accepts_matching_fingerprint(tmp_path):
    archive = tmp_path / "retention_archive"
    state = tmp_path / "state.json"
    _seed_archive(archive)
    fingerprint, file_count = archive_fingerprint(archive)
    _write_state(state, uploaded_at=NOW, sha=fingerprint)

    result = verify_restored_archive(archive, state)
    assert result["archive_restore_verified"] is True
    assert result["archive_restore_status"] == "fingerprint_match"
    assert result["archive_sha256"] == fingerprint
    assert result["archive_file_count"] == file_count == 1


def test_verify_restored_archive_rejects_mismatch_and_empty_archive(tmp_path):
    archive = tmp_path / "retention_archive"
    state = tmp_path / "state.json"
    _seed_archive(archive)
    fingerprint, _ = archive_fingerprint(archive)
    _write_state(state, uploaded_at=NOW, sha=fingerprint)

    (archive / "generation-1" / "delta.jsonl.gz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_restored_archive(archive, state)

    (archive / "generation-1" / "delta.jsonl.gz").unlink()
    with pytest.raises(ValueError, match="directory is empty"):
        verify_restored_archive(archive, state)


def test_archive_fingerprint_rejects_same_size_file_rewrite_during_hash(monkeypatch, tmp_path):
    archive = tmp_path / "retention_archive"
    _seed_archive(archive)
    target = archive / "generation-1" / "delta.jsonl.gz"
    original_open = Path.open
    mutated = False

    def open_with_concurrent_rewrite(path, *args, **kwargs):
        nonlocal mutated
        handle = original_open(path, *args, **kwargs)
        if path == target and args and args[0] == "rb" and not mutated:
            mutated = True
            target.write_bytes(b"X" * target.stat().st_size)
        return handle

    monkeypatch.setattr(Path, "open", open_with_concurrent_rewrite)
    with pytest.raises(ValueError, match="changed before fingerprinting|changed while fingerprinting"):
        archive_fingerprint(archive)
