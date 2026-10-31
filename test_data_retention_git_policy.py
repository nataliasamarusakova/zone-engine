import pytest
from pathlib import Path


def test_persistent_runtime_data_is_artifact_backed_and_ephemeral_data_is_ignored():
    lines = {line.strip() for line in Path(".gitignore").read_text(encoding="utf-8").splitlines()}
    assert "data/*" in lines
    assert "!data/.gitkeep" in lines
    # Source checkouts may keep the placeholder, while distributable archives
    # intentionally omit the runtime-data directory completely.
    assert not Path("data").exists() or (Path("data") / ".gitkeep").is_file()
    assert "bootstrap/research_history_seed/" not in lines
    assert "data/scan_history.jsonl" in lines
    assert "data/scan_history.jsonl.*" in lines
    assert "data/zone_engine_diagnostic.log" in lines
    assert "data/retention_archive/" in lines


def test_runbook_documents_artifact_backed_runtime_state_migration():
    text = Path("DATA_RETENTION_RUNBOOK.md").read_text(encoding="utf-8")
    assert "zone-engine-runtime-state" in text
    assert "guarded clean-start" in text
    assert "trading remains disabled" in text
    assert "self-verifying runtime-state artifact" in text
    assert "completed, successful event-engine run" in text
    assert "Runtime-state artifacts from failed runs are not selected" in text
    assert "last_uploaded_by_run_id" in text
    assert "archive-upload and marker-update steps both completed successfully" in text


def test_pytest_has_global_repository_data_isolation():
    text = Path("conftest.py").read_text(encoding="utf-8")
    assert "autouse=True" in text
    assert "Never allow pytest research tests to mutate the repository's production data" in text
    assert '"DATA_DIR": runtime_data' in text
    assert 'COUNTERFACTUAL_EXPERIMENTS_PATH' in text


def test_workflow_runs_size_guard_before_runtime_state_artifact_upload():
    text = Path(".github/workflows/event-engine.yml").read_text(encoding="utf-8")
    assert "Pre-commit data size guard" in text
    assert "--size-guard" in text
    assert "runtime-state-manifest.json" in Path("event_engine/runtime_state_bundle.py").read_text(encoding="utf-8")
    assert 'DATA_RETENTION_SIZE_GUARD_BYTES: "90000000"' in text
    assert 'DATA_RETENTION_SIZE_TARGET_BYTES: "80000000"' in text
    assert "Verify tests did not mutate runtime data" in text
    assert "runtime_state_bundle.py verify-unchanged" in text
    assert "Upload durable runtime-state artifact" in text
    assert text.index("Pre-commit data size guard") < text.index("Upload durable runtime-state artifact")
    assert 'EXECUTION_REQUIRE_EXCHANGE_TIMESTAMP: "true"' in text
    assert 'EXECUTION_UNKNOWN_QUOTE_AGE_POLICY: "block"' in text
    assert "Restore fingerprint-matched cumulative retention archive" in text
    assert "Upload cumulative retention archive for next clean runner" in text
    assert 'repos/${GITHUB_REPOSITORY}/actions/artifacts?name=zone-engine-retention-archive&per_page=100' in text
    assert 'gh run download "$run_id" --name zone-engine-retention-archive' in text
    assert "retention-days: 90" in text


def test_size_guard_constants_have_headroom():
    from event_engine import data_retention
    assert data_retention.GITHUB_SIZE_GUARD_BYTES == 90_000_000
    assert data_retention.GITHUB_SIZE_TARGET_BYTES == 80_000_000
    assert data_retention.GITHUB_SIZE_TARGET_BYTES < data_retention.GITHUB_SIZE_GUARD_BYTES < data_retention.GITHUB_MAX_FILE_BYTES


def test_git_policy_externalizes_engine_state_but_keeps_retention_archive_separate():
    lines = {line.strip() for line in Path(".gitignore").read_text(encoding="utf-8").splitlines()}
    assert "data/*" in lines
    assert "bootstrap/research_history_seed/" not in lines
    assert "data/retention_archive/" in lines


def test_size_guard_archives_before_compaction(monkeypatch, tmp_path):
    from event_engine import data_retention
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "market_bars_1h.jsonl"
    now = data_retention.datetime.now(data_retention.timezone.utc)
    rows = []
    for i in range(400):
        rows.append({
            "timestamp": int((now - data_retention.timedelta(hours=200 + i)).timestamp() * 1000),
            "symbol": "TEST-USDT",
            "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
            "payload": "x" * 80,
        })
    path.write_text("".join(__import__("json").dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(data_retention, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_retention, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(data_retention, "REPO_RETENTION_ARCHIVE_ROOT", data_dir / "retention_archive")
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_GUARD_BYTES", 20_000)
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_TARGET_BYTES", 10_000)
    result = data_retention.apply_size_guard(now=now, archive_root=data_dir / "retention_archive")
    assert result["files"]["market_bars_1h.jsonl"]["triggered"] is True
    assert path.stat().st_size < 20_000
    snapshots = list((data_dir / "retention_archive").rglob("*.before.part-*.jsonl.gz"))
    assert snapshots


def test_check_staged_guard_blocks_large_data_file(monkeypatch, tmp_path):
    import subprocess
    from event_engine import data_retention
    repo = tmp_path
    (repo / "data").mkdir()
    payload = "x" * 100
    (repo / "data" / "example.jsonl").write_text(payload + "\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "data/example.jsonl"], cwd=repo, check=True)
    monkeypatch.setattr(data_retention, "PROJECT_ROOT", repo)
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_GUARD_BYTES", 50)
    try:
        data_retention.check_staged_size_guard(limit_bytes=50)
    except RuntimeError as exc:
        assert "data/example.jsonl" in str(exc)
    else:
        raise AssertionError("staged size guard failed to block an oversized staged file")


def test_size_guard_zone_observations_can_archive_old_non_nearest_rows_without_touching_protected(monkeypatch, tmp_path):
    import json
    from event_engine import data_retention

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    now = data_retention.datetime.now(data_retention.timezone.utc)
    old = (now - data_retention.timedelta(hours=400)).isoformat()
    recent = (now - data_retention.timedelta(hours=1)).isoformat()
    protected = {
        "record_type": "ZONE_OBSERVATION",
        "observation_id": "OBS_PROTECTED",
        "event_type": "SIGNAL_CREATED",
        "scan_id": "SCAN_TRADE",
        "observation_ts": old,
    }
    rows = [protected] + [
        {
            "record_type": "ZONE_OBSERVATION",
            "observation_id": f"OBS_OLD_{i}",
            "event_type": "REARM" if i % 2 else "TOUCH_BLOCKED",
            "scan_id": f"SCAN_OLD_{i}",
            "observation_ts": old,
            "payload": "x" * 120,
        }
        for i in range(400)
    ] + [{
        "record_type": "ZONE_OBSERVATION",
        "observation_id": "OBS_RECENT",
        "event_type": "TOUCH_BLOCKED",
        "scan_id": "SCAN_RECENT",
        "observation_ts": recent,
    }]
    path = data_dir / "zone_observations.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (data_dir / "active_trades.json").write_text(json.dumps({}), encoding="utf-8")
    (data_dir / "research_outcome_state.json").write_text(
        json.dumps({"processed_observation_ids": [f"OBS_OLD_{i}" for i in range(400)] + ["OBS_RECENT", "OBS_PROTECTED"]}),
        encoding="utf-8",
    )
    (data_dir / "trades.jsonl").write_text(json.dumps({"record_type": "TRADE_OPEN", "event_id": "E1", "scan_id": "SCAN_TRADE", "observation_id": "OBS_PROTECTED"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(data_retention, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_retention, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(data_retention, "REPO_RETENTION_ARCHIVE_ROOT", data_dir / "retention_archive")
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_GUARD_BYTES", 20_000)
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_TARGET_BYTES", 10_000)

    result = data_retention.apply_size_guard(now=now, archive_root=data_dir / "retention_archive")
    meta = result["files"]["zone_observations.jsonl"]
    assert meta["bytes_after"] < 20_000
    kept = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept_ids = {row["observation_id"] for row in kept}
    assert "OBS_PROTECTED" not in kept_ids
    assert "OBS_RECENT" in kept_ids
    assert any(x.get("pruned", 0) > 0 for x in meta["policies"])
    snapshot_parts = list((data_dir / "retention_archive").rglob("zone_observations.jsonl.before.part-*.jsonl.gz"))
    assert snapshot_parts
    import gzip
    restored_ids = set()
    for part in snapshot_parts:
        with gzip.open(part, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    restored_ids.add(json.loads(line)["observation_id"])
    assert "OBS_PROTECTED" in restored_ids
    assert "OBS_OLD_0" in restored_ids


def test_size_guard_continues_safe_policies_below_guard_until_target(monkeypatch, tmp_path):
    import json
    from event_engine import data_retention

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "market_bars_5m.jsonl"
    rows = [
        {"id": "discard-first", "payload": "a" * 5_000},
        {"id": "keep-first-pass", "payload": "b" * 5_000},
        {"id": "discard-second", "payload": "c" * 5_000},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(data_retention, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_GUARD_BYTES", 10_000)
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_TARGET_BYTES", 1_000)
    monkeypatch.setattr(data_retention, "_size_guard_plan", lambda *args, **kwargs: [
        ("first-policy", lambda row: row.get("id") == "keep-first-pass"),
        ("second-policy", lambda row: False),
    ])

    result = data_retention._apply_size_guard_to_file(
        path, now=data_retention.datetime.now(data_retention.timezone.utc),
        archive_root=data_dir / "retention_archive", pending_ids=set(),
        pending_bar_cutoffs={}, protected_observation_ids=set(), protected_refs=set(),
    )

    assert result["bytes_after"] <= data_retention.GITHUB_SIZE_TARGET_BYTES
    assert [row["policy"] for row in result["policies"]] == ["first-policy", "second-policy"]
    assert list(path.read_text(encoding="utf-8").splitlines()) == []
    assert list((data_dir / "retention_archive").rglob("*.before.part-*.jsonl.gz"))
    assert list((data_dir / "retention_archive").rglob("*.pruned.jsonl.gz"))


def test_size_guard_archives_old_bars_instead_of_blocking_on_pending_observation(monkeypatch, tmp_path):
    """Pending outcomes must remain replayable from archive deltas without pinning every live 5m bar."""
    import gzip
    import json
    from datetime import timedelta
    from event_engine import data_retention

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    now = data_retention.datetime(2026, 10, 19, 12, 0, tzinfo=data_retention.timezone.utc)
    path = data_dir / "market_bars_5m.jsonl"
    rows = []
    for index in range(72):
        ts = now - timedelta(minutes=5 * index)
        rows.append({
            "record_type": "MARKET_BAR", "symbol": "TEST-USDT", "provider": "binance",
            "timestamp": ts.isoformat(), "close_time": (ts + timedelta(minutes=5)).isoformat(),
            "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1,
            "retention_test_id": f"BAR-{index:03d}", "payload": "x" * 2_000,
        })
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    (data_dir / "research_outcome_state.json").write_text(json.dumps({
        "version": 2,
        "pending_observations_v1": {
            "OBS_OLD": {
                "observation_id": "OBS_OLD", "symbol": "TEST-USDT", "direction": "LONG",
                "observation_ts": (now - timedelta(days=3)).isoformat(), "reference_price": 100,
                "event_type": "SIGNAL_CREATED",
            }
        },
    }), encoding="utf-8")

    monkeypatch.setattr(data_retention, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_retention, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(data_retention, "REPO_RETENTION_ARCHIVE_ROOT", data_dir / "retention_archive")
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_GUARD_BYTES", 50_000)
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_TARGET_BYTES", 30_000)

    result = data_retention.apply_size_guard(now=now, archive_root=data_dir / "retention_archive")
    file_result = result["files"]["market_bars_5m.jsonl"]
    assert file_result["triggered"] is True
    assert file_result["bytes_after"] <= data_retention.GITHUB_SIZE_TARGET_BYTES

    # Every original bar remains available either in the live file or a pruned
    # delta. The archive path is what makes pruning safe for pending research.
    retained_ids = {
        json.loads(line)["retention_test_id"]
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    archived_ids = set()
    for delta in (data_dir / "retention_archive").rglob("market_bars_5m.jsonl.pruned*.jsonl.gz"):
        with gzip.open(delta, "rt", encoding="utf-8") as fh:
            archived_ids.update(json.loads(line)["retention_test_id"] for line in fh if line.strip())
    original_ids = {row["retention_test_id"] for row in rows}
    assert retained_ids.isdisjoint(archived_ids)
    assert retained_ids | archived_ids == original_ids
    assert "OBS_OLD" in json.loads((data_dir / "research_outcome_state.json").read_text(encoding="utf-8"))["pending_observations_v1"]


def test_size_guard_has_lossless_backstop_for_unparseable_or_excess_recent_5m_rows(monkeypatch, tmp_path):
    """Oversized bars are bounded even if conservative timestamp policies cannot prune them."""
    import gzip
    import json
    from datetime import timedelta
    from event_engine import data_retention

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    now = data_retention.datetime(2026, 10, 19, 12, 0, tzinfo=data_retention.timezone.utc)
    path = data_dir / "market_bars_5m.jsonl"
    rows = []
    # These rows deliberately have timestamps the normal age policies cannot parse.
    for index in range(80):
        rows.append({
            "record_type": "MARKET_BAR", "symbol": f"BAD{index % 8}-USDT", "provider": "binance",
            "timestamp": "not-a-timestamp", "bar_id": f"BAD-{index:03d}",
            "payload": "u" * 1_400,
        })
    # Valid recent rows alone are also large enough to require the adaptive fallback.
    for index in range(50):
        ts = now - timedelta(minutes=index * 1.2)
        rows.append({
            "record_type": "MARKET_BAR", "symbol": f"GOOD{index % 10}-USDT", "provider": "binance",
            "timestamp": ts.isoformat(), "bar_id": f"GOOD-{index:03d}",
            "payload": "v" * 1_400,
        })
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")

    monkeypatch.setattr(data_retention, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_GUARD_BYTES", 80_000)
    monkeypatch.setattr(data_retention, "GITHUB_SIZE_TARGET_BYTES", 50_000)

    result = data_retention._apply_size_guard_to_file(
        path, now=now, archive_root=data_dir / "retention_archive", pending_ids=set(),
        pending_bar_cutoffs={}, protected_observation_ids=set(), protected_refs=set(),
    )

    assert result["triggered"] is True
    assert result["bytes_after"] < data_retention.GITHUB_SIZE_GUARD_BYTES
    assert result["bytes_after"] <= data_retention.GITHUB_SIZE_TARGET_BYTES
    assert result["policies"][-1]["policy"] == "keep_newest_valid_5m_rows_within_size_budget_archiving_unknown_or_excess_rows"

    kept_ids = {
        json.loads(line)["bar_id"]
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    archived_ids = set()
    for delta in (data_dir / "retention_archive").rglob("market_bars_5m.jsonl.pruned*.jsonl.gz"):
        with gzip.open(delta, "rt", encoding="utf-8") as fh:
            archived_ids.update(json.loads(line)["bar_id"] for line in fh if line.strip())
    original_ids = {row["bar_id"] for row in rows}
    assert kept_ids.isdisjoint(archived_ids)
    assert kept_ids | archived_ids == original_ids

def test_clean_checkout_can_restore_runtime_state_from_verified_bundle(tmp_path):
    """A clean checkout restores required state without any legacy seed directory."""
    import json

    from event_engine.runtime_state_bundle import create_bundle, restore_bundle, REQUIRED_FILES

    root = tmp_path / "source"
    data = root / "data"
    for relative in REQUIRED_FILES:
        path = data / relative.removeprefix("data/")
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".jsonl"):
            path.write_text("", encoding="utf-8")
        elif relative in {"data/active_trades.json", "data/event_execution_claims.json"}:
            path.write_text("{}\n", encoding="utf-8")
        elif relative == "data/zone_visit_state.json":
            path.write_text(json.dumps({"version": 1, "trigger_mode": "zone", "symbols": {}}) + "\n", encoding="utf-8")
        else:
            path.write_text(json.dumps({"path": relative}) + "\n", encoding="utf-8")
    (data / "active_trades.json").write_text(
        '{"E1": {"event_id": "E1", "symbol": "TEST-USDT", "direction": "LONG"}}\n', encoding="utf-8"
    )
    (data / "research_outcome_state.json").write_text(
        '{"schema_version": 1, "processed_observation_ids": ["OBS1"]}\n', encoding="utf-8"
    )
    (data / "trades.jsonl").write_text('{"record_type":"TRADE_OPEN","event_id":"E1"}\n', encoding="utf-8")
    (data / "scan_history.jsonl").write_text('{"ephemeral":true}\n', encoding="utf-8")
    (data / "zone_engine_diagnostic.log").write_text("diagnostic\n", encoding="utf-8")
    (data / "retention_archive").mkdir()
    (data / "retention_archive" / "snapshot.jsonl.gz").write_bytes(b"separately managed")

    bundle = tmp_path / "runtime-state-bundle"
    create_bundle(data_dir=data, output_dir=bundle)

    clean = tmp_path / "clean-checkout"
    clean_data = clean / "data"
    clean_data.mkdir(parents=True)
    (clean_data / "stale-state.json").write_text("stale\n", encoding="utf-8")
    restore_bundle(bundle_dir=bundle, data_dir=clean_data)

    assert json.loads((clean_data / "active_trades.json").read_text(encoding="utf-8"))["E1"]["symbol"] == "TEST-USDT"
    assert json.loads((clean_data / "research_outcome_state.json").read_text(encoding="utf-8"))["processed_observation_ids"] == ["OBS1"]
    assert (clean_data / "trades.jsonl").exists()
    assert not (clean_data / "scan_history.jsonl").exists()
    assert not (clean_data / "zone_engine_diagnostic.log").exists()
    assert not (clean_data / "retention_archive").exists()
    assert not (clean_data / "stale-state.json").exists()
    assert not (clean / "bootstrap").exists()


def test_size_guard_persists_failure_manifest_before_raising(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone

    from event_engine import data_retention as retention

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    target = data_dir / "market_bars_5m.jsonl"
    target.write_text("original\n", encoding="utf-8")
    monkeypatch.setattr(retention, "DATA_DIR", data_dir)
    monkeypatch.setattr(retention, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(retention, "REPO_RETENTION_ARCHIVE_ROOT", tmp_path / "retention_archive")
    monkeypatch.setattr(retention, "_validate_state_before", lambda: {})
    monkeypatch.setattr(retention, "_validate_state_unchanged", lambda before: None)
    monkeypatch.setattr(retention, "_pending_observations", lambda: [])
    monkeypatch.setattr(retention, "_pending_bar_cutoffs", lambda **kwargs: {})
    monkeypatch.setattr(retention, "_protected_audit_refs", lambda: (set(), set()))

    def fake_apply(path, **kwargs):
        path.write_text("compacted-but-unarchived\n", encoding="utf-8")
        return {
            "path": f"data/{path.name}", "triggered": True, "failed": True,
            "error": "cannot reduce below test guard", "snapshot": {"source_sha256": "before"},
        }

    monkeypatch.setattr(retention, "_apply_size_guard_to_file", fake_apply)
    with pytest.raises(RuntimeError, match="cannot reduce below test guard"):
        retention.apply_size_guard(now=datetime.now(timezone.utc), archive_root=tmp_path / "archive")
    manifest = json.loads((data_dir / "data_retention_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["files"]["market_bars_5m.jsonl"]["triggered"] is True
    assert manifest["files"]["market_bars_5m.jsonl"]["snapshot"]["source_sha256"] == "before"


def test_chunked_snapshot_can_restore_file_after_mid_policy_failure(tmp_path):
    from event_engine import data_retention as retention

    archive_root = tmp_path / "archive"
    path = tmp_path / "journal.jsonl"
    original = b'{"id":1}\n{"id":2}\n'
    path.write_bytes(original)
    snapshot = retention._snapshot_jsonl_chunked_locked(path, archive_root / "snapshots")
    path.write_bytes(b'{"partial":"compaction"}\n')
    retention._restore_chunked_snapshot(path, snapshot, archive_root)
    assert path.read_bytes() == original


def test_data_placeholder_is_visible_but_never_migrated_as_runtime_state():
    workflow = Path(".github/workflows/event-engine.yml").read_text(encoding="utf-8")
    detect = workflow.split("- name: Detect legacy Git-backed runtime state", 1)[1].split(
        "- name: Restore latest durable runtime-state artifact", 1
    )[0]
    externalize = workflow.split("- name: Remove legacy runtime data from Git history tip", 1)[1]
    assert "grep -vx 'data/.gitkeep'" in detect
    assert "touch data/.gitkeep" in externalize
    assert "git add -f data/.gitkeep" in externalize
