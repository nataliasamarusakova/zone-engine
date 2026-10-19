from __future__ import annotations

import json
from pathlib import Path

import pytest

from event_engine.runtime_state_bundle import (
    MANIFEST_NAME,
    REQUIRED_FILES,
    create_bundle,
    restore_bundle,
    verify_bundle,
    verify_data_unchanged,
    write_data_snapshot,
)


def _make_complete_state(root: Path) -> tuple[Path, Path]:
    data = root / "data"
    seed = root / "bootstrap" / "research_history_seed"
    for rel in REQUIRED_FILES:
        if rel.startswith("data/"):
            path = data / rel.removeprefix("data/")
        else:
            path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".gz":
            path.write_bytes(b"tiny gz placeholder for state-bundle unit test")
        else:
            path.write_text(json.dumps({"path": rel, "version": 1}) + "\n", encoding="utf-8")
    (data / "retention_archive").mkdir(parents=True, exist_ok=True)
    (data / "retention_archive" / "large-history-delta.jsonl.gz").write_bytes(b"separately-managed-archive")
    (data / "scan_history.jsonl").write_text("ephemeral\n", encoding="utf-8")
    (data / "zone_engine_diagnostic.log").write_text("ephemeral log\n", encoding="utf-8")
    return data, seed


def test_runtime_state_bundle_roundtrip_is_verified_and_excludes_ephemeral_files(tmp_path):
    data, seed = _make_complete_state(tmp_path / "source")
    bundle = tmp_path / "runtime-state-bundle"
    create_bundle(data_dir=data, seed_dir=seed, output_dir=bundle, source_commit="abc", run_id="123")
    manifest = verify_bundle(bundle)

    paths = {item["path"] for item in manifest["files"]}
    assert "data/active_trades.json" in paths
    assert "bootstrap/research_history_seed/2026-10-16/manifest.json" in paths
    assert not any("retention_archive/" in path for path in paths)
    assert "data/scan_history.jsonl" not in paths
    assert "data/zone_engine_diagnostic.log" not in paths

    fresh_data = tmp_path / "fresh" / "data"
    fresh_seed = tmp_path / "fresh" / "bootstrap" / "research_history_seed"
    fresh_data.mkdir(parents=True)
    (fresh_data / "stale-state.json").write_text("stale\n", encoding="utf-8")
    restore_bundle(bundle_dir=bundle, data_dir=fresh_data, seed_dir=fresh_seed)

    assert (fresh_data / "active_trades.json").read_text(encoding="utf-8") == (data / "active_trades.json").read_text(encoding="utf-8")
    assert (fresh_data / "stale-state.json").exists() is False
    assert (fresh_data / "retention_archive").exists() is False
    assert (fresh_data / "scan_history.jsonl").exists() is False
    assert (fresh_seed / "2026-10-16" / "market_bars_5m.jsonl.before_2026-10-16.jsonl.gz").read_bytes() == (
        seed / "2026-10-16" / "market_bars_5m.jsonl.before_2026-10-16.jsonl.gz"
    ).read_bytes()


def test_runtime_state_bundle_rejects_corrupt_or_missing_state(tmp_path):
    data, seed = _make_complete_state(tmp_path / "source")
    bundle = tmp_path / "runtime-state-bundle"
    create_bundle(data_dir=data, seed_dir=seed, output_dir=bundle)

    active_path = bundle / "data" / "active_trades.json"
    original = active_path.read_text(encoding="utf-8")
    active_path.write_text(("x" if original[0] != "x" else "y") + original[1:], encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_bundle(bundle)

    # A valid manifest cannot authorize a bundle with a missing required journal.
    create_bundle(data_dir=data, seed_dir=seed, output_dir=bundle)
    (bundle / "data" / "execution_ledger.jsonl").unlink()
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        verify_bundle(bundle)


def test_runtime_data_snapshot_detects_test_mutation_but_ignores_ephemeral_logs(tmp_path):
    data, _ = _make_complete_state(tmp_path / "source")
    snapshot = tmp_path / "snapshot.json"
    write_data_snapshot(data_dir=data, output_path=snapshot)

    # These are explicitly ephemeral and are allowed to change during a test run.
    (data / "zone_engine_diagnostic.log").write_text("new log\n", encoding="utf-8")
    (data / "scan_history.jsonl").write_text("new scan\n", encoding="utf-8")
    (data / "retention_archive" / "large-history-delta.jsonl.gz").write_bytes(b"refreshed separately")
    verify_data_unchanged(data_dir=data, snapshot_path=snapshot)

    (data / "trades.jsonl").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Tests mutated production runtime data"):
        verify_data_unchanged(data_dir=data, snapshot_path=snapshot)
