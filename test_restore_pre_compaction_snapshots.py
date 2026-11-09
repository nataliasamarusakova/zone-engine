from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from event_engine import data_retention as retention


def _write_snapshot(root: Path, filename: str, payload: bytes) -> dict:
    directory = root / filename
    directory.mkdir(parents=True)
    chunk = directory / f"{filename}.part-0001.jsonl.gz"
    with gzip.open(chunk, "wb") as gz:
        gz.write(payload)
    return {
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "source_bytes": len(payload),
        "source_rows": sum(1 for line in payload.splitlines(keepends=True) if line.strip()),
        "chunks": [{"path": str(chunk), "compressed_bytes": chunk.stat().st_size}],
    }


def _setup(monkeypatch, tmp_path: Path):
    project = tmp_path / "repo"
    data = project / "data"
    archive = data / "retention_archive"
    data.mkdir(parents=True)
    archive.mkdir(parents=True)
    monkeypatch.setattr(retention, "PROJECT_ROOT", project)
    monkeypatch.setattr(retention, "DATA_DIR", data)
    return project, data, archive


def test_restore_pre_compaction_snapshots_restores_exact_bytes_and_records_audit(monkeypatch, tmp_path):
    project, data, archive = _setup(monkeypatch, tmp_path)
    original = b'{"row":1}\n{ "row" : 2 }\n\n'
    snapshot = _write_snapshot(archive, "generation-1", original)
    target = data / "market_bars_5m.jsonl"
    target.write_bytes(b'{"row":2}\n')
    manifest_path = data / "data_retention_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "mode": "size_guard",
        "workflow_run_id": "current-run-id",
        "workflow_run_attempt": "4",
        "files": {
            "market_bars_5m.jsonl": {
                "path": "data/market_bars_5m.jsonl",
                "triggered": True,
                "bytes_before": len(original),
                "bytes_after": target.stat().st_size,
                "snapshot": snapshot,
            },
            "market_bars_1h.jsonl": {
                "path": "data/market_bars_1h.jsonl",
                "triggered": False,
            },
        },
    }), encoding="utf-8")
    github_output = project / "github_output.txt"

    result = retention.restore_pre_compaction_snapshots(
        manifest_path=manifest_path, archive_dir=archive, github_output=github_output,
        expected_workflow_run_id="current-run-id",
        expected_workflow_run_attempt="4",
    )

    assert target.read_bytes() == original
    assert result["restored_any"] is True
    assert result["restored_files"][0]["name"] == "market_bars_5m.jsonl"
    assert result["restored_bytes"] == len(original)
    assert "restored_any=true" in github_output.read_text(encoding="utf-8")
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["pre_compaction_rollback"]["status"] == "restored"


def test_bad_later_snapshot_does_not_replace_any_target(monkeypatch, tmp_path):
    project, data, archive = _setup(monkeypatch, tmp_path)
    original_a = b'{"a":1}\n'
    original_b = b'{"b":2}\n'
    snapshot_a = _write_snapshot(archive, "generation-a", original_a)
    snapshot_b = _write_snapshot(archive, "generation-b", original_b)
    target_a = data / "market_bars_5m.jsonl"
    target_b = data / "market_bars_1h.jsonl"
    target_a.write_bytes(b'{"a":"compacted"}\n')
    target_b.write_bytes(b'{"b":"compacted"}\n')
    bad_chunk = Path(snapshot_b["chunks"][0]["path"])
    bad_chunk.write_bytes(b"corrupt gzip")
    manifest_path = data / "data_retention_manifest.json"
    manifest_path.write_text(json.dumps({
        "mode": "size_guard",
        "files": {
            "market_bars_5m.jsonl": {"path": "data/market_bars_5m.jsonl", "triggered": True, "snapshot": snapshot_a},
            "market_bars_1h.jsonl": {"path": "data/market_bars_1h.jsonl", "triggered": True, "snapshot": snapshot_b},
        },
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="compressed snapshot size mismatch"):
        retention.restore_pre_compaction_snapshots(manifest_path=manifest_path, archive_dir=archive)

    assert target_a.read_bytes() == b'{"a":"compacted"}\n'
    assert target_b.read_bytes() == b'{"b":"compacted"}\n'


def test_restore_rejects_snapshot_chunk_outside_archive(monkeypatch, tmp_path):
    project, data, archive = _setup(monkeypatch, tmp_path)
    outside = project / "outside.jsonl.gz"
    with gzip.open(outside, "wb") as gz:
        gz.write(b'{"x":1}\n')
    target = data / "market_bars_5m.jsonl"
    compacted = b'{"x":2}\n'
    target.write_bytes(compacted)
    payload = b'{"x":1}\n'
    manifest_path = data / "data_retention_manifest.json"
    manifest_path.write_text(json.dumps({
        "mode": "size_guard",
        "files": {
            "market_bars_5m.jsonl": {
                "path": "data/market_bars_5m.jsonl",
                "triggered": True,
                "snapshot": {
                    "source_sha256": hashlib.sha256(payload).hexdigest(),
                    "source_bytes": len(payload),
                    "source_rows": 1,
                    "chunks": [{"path": str(outside), "compressed_bytes": outside.stat().st_size}],
                },
            },
        },
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="outside retention archive directory"):
        retention.restore_pre_compaction_snapshots(manifest_path=manifest_path, archive_dir=archive)
    assert target.read_bytes() == compacted


def test_no_compacted_files_is_a_safe_noop(monkeypatch, tmp_path):
    project, data, archive = _setup(monkeypatch, tmp_path)
    manifest_path = data / "data_retention_manifest.json"
    manifest_path.write_text(json.dumps({
        "mode": "size_guard",
        "files": {"market_bars_5m.jsonl": {"path": "data/market_bars_5m.jsonl", "triggered": False}},
    }), encoding="utf-8")
    result = retention.restore_pre_compaction_snapshots(manifest_path=manifest_path, archive_dir=archive)
    assert result["status"] == "no_compacted_files"
    assert result["restored_any"] is False


def test_staged_guard_hard_limit_is_strictly_less_than_100m(monkeypatch):
    import subprocess

    from event_engine import data_retention as retention

    monkeypatch.setattr(retention, "_iter_git_staged_paths", lambda: ["data/market_bars_5m.jsonl"])
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "99999999")
    result = retention.check_staged_size_guard(limit_bytes=100_000_000)
    assert result["ok"] is True

    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "100000000")
    with pytest.raises(RuntimeError, match="100000000 bytes"):
        retention.check_staged_size_guard(limit_bytes=100_000_000)


@pytest.mark.parametrize(
    ("previous_run_id", "previous_run_attempt"),
    [("previous-run-id", "1"), ("current-run-id", "1")],
)
def test_preparation_failure_never_reuses_previous_workflow_attempt_snapshot(
    monkeypatch, tmp_path, previous_run_id, previous_run_attempt,
):
    """A failed new size-guard pass must not roll back with an older run's manifest."""
    from datetime import datetime, timezone

    project, data, archive = _setup(monkeypatch, tmp_path)
    live = data / "market_bars_5m.jsonl"
    current_payload = b'{"version":"CURRENT_RUN"}\n'
    previous_payload = b'{"version":"PREVIOUS_RUN"}\n'
    live.write_bytes(current_payload)
    snapshot = _write_snapshot(archive, "previous-run", previous_payload)
    manifest_path = data / "data_retention_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "mode": "size_guard",
        "status": "complete",
        "workflow_run_id": previous_run_id,
        "workflow_run_attempt": previous_run_attempt,
        "files": {
            "market_bars_5m.jsonl": {
                "path": "data/market_bars_5m.jsonl",
                "triggered": True,
                "snapshot": snapshot,
            },
        },
    }), encoding="utf-8")

    monkeypatch.setenv("GITHUB_RUN_ID", "current-run-id")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setattr(retention, "_validate_state_before", lambda: {})

    def fail_preparation():
        raise ValueError("corrupt research_outcome_state.json")

    monkeypatch.setattr(retention, "_pending_observations", fail_preparation)
    with pytest.raises(ValueError, match="corrupt research_outcome_state"):
        retention.apply_size_guard(
            now=datetime(2026, 11, 9, tzinfo=timezone.utc),
            archive_root=archive,
        )

    # This is the same rollback call the workflow makes after a size-guard failure.
    with pytest.raises(RuntimeError, match="manifest.*current GitHub Actions run"):
        retention.restore_pre_compaction_snapshots(
            manifest_path=manifest_path, archive_dir=archive,
            expected_workflow_run_id="current-run-id",
            expected_workflow_run_attempt="2",
        )
    assert live.read_bytes() == current_payload


def test_size_guard_manifest_records_current_workflow_run_and_attempt(monkeypatch, tmp_path):
    from datetime import datetime, timezone

    project, data, archive = _setup(monkeypatch, tmp_path)
    (data / "market_bars_5m.jsonl").write_text("original\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_RUN_ID", "current-run-id")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "3")
    monkeypatch.setattr(retention, "_validate_state_before", lambda: {})
    monkeypatch.setattr(retention, "_validate_state_unchanged", lambda before: None)
    monkeypatch.setattr(retention, "_pending_observations", lambda: [])
    monkeypatch.setattr(retention, "_pending_bar_cutoffs", lambda **kwargs: {})
    monkeypatch.setattr(retention, "_protected_audit_refs", lambda **kwargs: (set(), set()))
    monkeypatch.setattr(
        retention, "_apply_size_guard_to_file",
        lambda path, **kwargs: {"path": f"data/{path.name}", "triggered": False, "failed": False},
    )

    retention.apply_size_guard(
        now=datetime(2026, 11, 9, tzinfo=timezone.utc), archive_root=archive,
    )
    manifest = json.loads((data / "data_retention_manifest.json").read_text(encoding="utf-8"))
    assert manifest["workflow_run_id"] == "current-run-id"
    assert manifest["workflow_run_attempt"] == "3"
