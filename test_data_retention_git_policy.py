from pathlib import Path


def test_data_directory_remains_tracked_by_policy():
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "data/*" not in text
    assert "data/retention_archive" not in text


def test_runbook_requires_history_rebuild_for_large_unpushed_blobs():
    text = Path("DATA_RETENTION_RUNBOOK.md").read_text(encoding="utf-8")
    assert "git reset --soft origin/main" in text
    assert "git branch backup-before-data-git-cleanup" in text


def test_workflow_runs_size_guard_before_commit():
    text = Path(".github/workflows/event-engine.yml").read_text(encoding="utf-8")
    assert "Pre-commit data size guard" in text
    assert "--size-guard" in text
    assert "--check-staged" in text
    assert 'DATA_RETENTION_SIZE_GUARD_BYTES: "90000000"' in text
    assert 'DATA_RETENTION_SIZE_TARGET_BYTES: "80000000"' in text


def test_size_guard_constants_have_headroom():
    from event_engine import data_retention
    assert data_retention.GITHUB_SIZE_GUARD_BYTES == 90_000_000
    assert data_retention.GITHUB_SIZE_TARGET_BYTES == 80_000_000
    assert data_retention.GITHUB_SIZE_TARGET_BYTES < data_retention.GITHUB_SIZE_GUARD_BYTES < data_retention.GITHUB_MAX_FILE_BYTES


def test_git_policy_keeps_data_tracked():
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "data/*" not in text
    assert "data/" not in [line.strip() for line in text.splitlines()]


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
