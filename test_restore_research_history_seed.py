from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from restore_research_history_seed import restore_research_outcomes_seed


def _write_seed(tmp_path: Path, seed_rows: list[dict], live_rows: list[dict]):
    repo = tmp_path
    data = repo / "data"
    data.mkdir(parents=True, exist_ok=True)
    seed_dir = repo / "bootstrap" / "research_history_seed" / "2026-10-16"
    seed_dir.mkdir(parents=True)
    seed_raw = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in seed_rows
    )
    archive = seed_dir / "research_outcomes.jsonl.before_2026-10-16.jsonl.gz"
    archive.write_bytes(gzip.compress(seed_raw, mtime=0))
    manifest = {
        "schema_version": 1,
        "record_type": "RESEARCH_HISTORY_RECOVERY_SEED",
        "files": {
            "research_outcomes.jsonl": {
                "archive_path": str(archive.relative_to(repo)).replace("\\\\", "/"),
                "source_bytes": len(seed_raw),
                "source_rows": len(seed_rows),
                "source_sha256": hashlib.sha256(seed_raw).hexdigest(),
                "compressed_bytes": archive.stat().st_size,
                "verification": "sha256_and_byte_count_match",
            }
        },
    }
    (seed_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    target = data / "research_outcomes.jsonl"
    target.write_bytes(
        b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in live_rows)
    )
    return data, target, seed_dir


def test_restores_missing_seed_rows_and_is_idempotent_after_compaction(tmp_path: Path):
    seed_rows = [
        {"outcome_id": "old-1", "observation_id": "obs-1", "result": "A"},
        {"outcome_id": "old-2", "observation_id": "obs-2", "result": "B"},
    ]
    live_rows = [{"outcome_id": "new-1", "observation_id": "obs-3", "result": "C"}]
    data, target, seed_dir = _write_seed(tmp_path, seed_rows, live_rows)

    first = restore_research_outcomes_seed(data)
    assert first["status"] == "ok"
    assert first["merged_rows"] == 2
    restored = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [row["outcome_id"] for row in restored] == ["old-1", "old-2", "new-1"]

    marker = data / "research_outcomes_restore_state.json"
    assert marker.exists()

    # Simulate a later legitimate size-guard compaction. The restore seed is a
    # one-time source; it must not be appended again on every fresh runner.
    target.write_text(json.dumps(live_rows[0], sort_keys=True) + "\n", encoding="utf-8")
    second = restore_research_outcomes_seed(data)
    assert second["status"] == "ok"
    assert second["merged_rows"] == 0
    assert [json.loads(line)["outcome_id"] for line in target.read_text(encoding="utf-8").splitlines()] == ["new-1"]
    assert second["sources"][0]["status"] == "already_marked_restored"


def test_already_merged_seed_is_not_rewritten_but_marker_is_created(tmp_path: Path):
    seed_rows = [{"outcome_id": "old-1", "observation_id": "obs-1"}]
    live_rows = seed_rows + [{"outcome_id": "new-1", "observation_id": "obs-2"}]
    data, target, seed_dir = _write_seed(tmp_path, seed_rows, live_rows)
    before = target.read_bytes()

    result = restore_research_outcomes_seed(data)

    assert result["merged_rows"] == 0
    assert target.read_bytes() == before
    assert (data / "research_outcomes_restore_state.json").exists()


def test_conflicting_outcome_id_fails_without_modifying_live_journal(tmp_path: Path):
    seed_rows = [{"outcome_id": "same-id", "observation_id": "obs-1", "value": 1}]
    live_rows = [{"outcome_id": "same-id", "observation_id": "obs-1", "value": 2}]
    data, target, seed_dir = _write_seed(tmp_path, seed_rows, live_rows)
    before = target.read_bytes()

    with pytest.raises(ValueError, match="conflicting contents"):
        restore_research_outcomes_seed(data)

    assert target.read_bytes() == before
    assert not (data / "research_outcomes_restore_state.json").exists()


def test_seed_checksum_mismatch_fails_without_modifying_live_journal(tmp_path: Path):
    seed_rows = [{"outcome_id": "old-1", "observation_id": "obs-1"}]
    data, target, seed_dir = _write_seed(tmp_path, seed_rows, [{"outcome_id": "new-1", "observation_id": "obs-2"}])
    archive = next(seed_dir.glob("*.gz"))
    archive.write_bytes(gzip.compress(b'{"outcome_id":"tampered"}\n', mtime=0))
    before = target.read_bytes()

    with pytest.raises(ValueError, match="mismatch"):
        restore_research_outcomes_seed(data)

    assert target.read_bytes() == before
    assert not (data / "research_outcomes_restore_state.json").exists()


def test_missing_seed_is_reported_as_not_restored(tmp_path: Path, capsys):
    data = tmp_path / "data"
    result = restore_research_outcomes_seed(data)
    assert result["status"] == "no_seed_directory"

    from restore_research_history_seed import main

    assert main(["--data-dir", str(data)]) == 3
    assert '"status": "no_seed_directory"' in capsys.readouterr().out
