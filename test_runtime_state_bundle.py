from __future__ import annotations

import json
from pathlib import Path

import pytest

from event_engine.runtime_state_bundle import (
    REQUIRED_FILES,
    create_bundle,
    restore_bundle,
    verify_bundle,
    verify_data_unchanged,
    write_data_snapshot,
)


def _make_complete_state(root: Path) -> Path:
    data = root / "data"
    for rel in REQUIRED_FILES:
        path = data / Path(rel).relative_to("data")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            path.write_text("", encoding="utf-8")
        elif path.name.endswith(".txt"):
            path.write_text("empty scan\n", encoding="utf-8")
        elif path.name in {"active_trades.json", "event_execution_claims.json"}:
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text(json.dumps({"path": rel, "version": 1}) + "\n", encoding="utf-8")
    (data / "retention_archive").mkdir(parents=True, exist_ok=True)
    (data / "retention_archive" / "large-history-delta.jsonl.gz").write_bytes(b"separately-managed-archive")
    (data / ".gitkeep").write_text("", encoding="utf-8")
    (data / "scan_history.jsonl").write_text("ephemeral\n", encoding="utf-8")
    (data / "zone_engine_diagnostic.log").write_text("ephemeral log\n", encoding="utf-8")
    return data


def test_runtime_state_bundle_roundtrip_is_verified_and_excludes_ephemeral_files(tmp_path):
    data = _make_complete_state(tmp_path / "source")
    bundle = tmp_path / "runtime-state-bundle"
    create_bundle(data_dir=data, output_dir=bundle, source_commit="abc", run_id="123")
    manifest = verify_bundle(bundle)

    paths = {item["path"] for item in manifest["files"]}
    assert "data/active_trades.json" in paths
    assert not any(path.startswith("bootstrap/") for path in paths)
    assert not any("retention_archive/" in path for path in paths)
    assert "data/scan_history.jsonl" not in paths
    assert "data/zone_engine_diagnostic.log" not in paths
    assert "data/.gitkeep" not in paths

    fresh_data = tmp_path / "fresh" / "data"
    fresh_data.mkdir(parents=True)
    (fresh_data / "stale-state.json").write_text("stale\n", encoding="utf-8")
    restore_bundle(bundle_dir=bundle, data_dir=fresh_data)

    assert (fresh_data / "active_trades.json").read_text(encoding="utf-8") == (data / "active_trades.json").read_text(encoding="utf-8")
    assert not (fresh_data / "stale-state.json").exists()
    assert not (fresh_data / "retention_archive").exists()
    assert not (fresh_data / "scan_history.jsonl").exists()


def test_runtime_state_bundle_rejects_pre_clean_start_schema(tmp_path):
    data = _make_complete_state(tmp_path / "source")
    bundle = tmp_path / "bundle"
    create_bundle(data_dir=data, output_dir=bundle)
    manifest_path = bundle / "runtime-state-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="format/schema is unsupported"):
        verify_bundle(bundle)


def test_runtime_state_bundle_rejects_legacy_recovery_seed_directory(tmp_path):
    data = _make_complete_state(tmp_path / "source")
    legacy = data / "research_history_seed"
    legacy.mkdir()
    (legacy / "old.jsonl.gz").write_bytes(b"legacy")
    with pytest.raises(RuntimeError, match="Legacy research recovery seed is not allowed"):
        create_bundle(data_dir=data, output_dir=tmp_path / "bundle")


def test_runtime_state_bundle_rejects_corrupt_or_missing_state(tmp_path):
    data = _make_complete_state(tmp_path / "source")
    bundle = tmp_path / "runtime-state-bundle"
    create_bundle(data_dir=data, output_dir=bundle)

    active_path = bundle / "data" / "active_trades.json"
    original = active_path.read_text(encoding="utf-8")
    active_path.write_text(("x" if original[0] != "x" else "y") + original[1:], encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_bundle(bundle)

    create_bundle(data_dir=data, output_dir=bundle)
    (bundle / "data" / "execution_ledger.jsonl").unlink()
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        verify_bundle(bundle)


def test_runtime_data_snapshot_detects_test_mutation_but_ignores_ephemeral_logs(tmp_path):
    data = _make_complete_state(tmp_path / "source")
    snapshot = tmp_path / "snapshot.json"
    write_data_snapshot(data_dir=data, output_path=snapshot)

    (data / "zone_engine_diagnostic.log").write_text("new log\n", encoding="utf-8")
    (data / "scan_history.jsonl").write_text("new scan\n", encoding="utf-8")
    (data / "retention_archive" / "large-history-delta.jsonl.gz").write_bytes(b"refreshed separately")
    verify_data_unchanged(data_dir=data, snapshot_path=snapshot)

    (data / "trades.jsonl").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Tests mutated production runtime data"):
        verify_data_unchanged(data_dir=data, snapshot_path=snapshot)


def test_runtime_state_bundle_rejects_symlinked_roots(tmp_path):
    data = _make_complete_state(tmp_path / "source")
    bundle = tmp_path / "bundle"
    create_bundle(data_dir=data, output_dir=bundle)
    bundle_link = tmp_path / "bundle-link"
    bundle_link.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinked runtime-state bundle"):
        verify_bundle(bundle_link)

    data_link = tmp_path / "data-link"
    data_link.symlink_to(data, target_is_directory=True)
    with pytest.raises(RuntimeError, match="must not be symlinks"):
        restore_bundle(bundle_dir=bundle, data_dir=data_link)
    with pytest.raises(RuntimeError, match="must not be symlinks"):
        create_bundle(data_dir=data_link, output_dir=tmp_path / "other-bundle")


def test_runtime_state_bundle_rejects_checksumming_corrupt_active_trade_json(tmp_path):
    import hashlib

    data = _make_complete_state(tmp_path / "source")
    bundle = tmp_path / "bundle"
    create_bundle(data_dir=data, output_dir=bundle)
    active_path = bundle / "data" / "active_trades.json"
    active_path.write_text("[]", encoding="utf-8")
    manifest_path = bundle / "runtime-state-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["files"]:
        if record["path"] == "data/active_trades.json":
            record["sha256"] = hashlib.sha256(active_path.read_bytes()).hexdigest()
            record["size_bytes"] = active_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="active_trades.json must have root type dict"):
        verify_bundle(bundle)
