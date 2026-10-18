from pathlib import Path


def test_workflow_restores_snapshots_on_retention_archive_upload_failure():
    workflow = (Path(__file__).parent / ".github" / "workflows" / "event-engine.yml").read_text(encoding="utf-8")
    assert "id: restore_pre_compaction_snapshots" in workflow
    assert "--restore-pre-compaction-snapshots" in workflow
    assert "steps.upload_retention_archive.outcome != 'success'" in workflow
    assert 'steps.restore_pre_compaction_snapshots.outputs.restored_any' in workflow
    assert "--check-staged --limit-bytes 100000000" in workflow


def test_staged_guard_exposes_configurable_hard_limit():
    source = (Path(__file__).parent / "event_engine" / "data_retention.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--limit-bytes", type=int' in source
    assert "check_staged_size_guard(limit_bytes=args.limit_bytes)" in source
    assert "parser.add_argument(\"--restore-pre-compaction-snapshots\"" in source


def test_workflow_blocks_commit_if_snapshot_restore_fails_even_when_archive_directory_is_empty():
    workflow = (Path(__file__).parent / ".github" / "workflows" / "event-engine.yml").read_text(encoding="utf-8")
    gate = workflow.split("id: retention_archive_commit_gate", 1)[1].split("- name: Upload diagnostic log", 1)[0]
    assert 'rollback_outcome="${{ steps.restore_pre_compaction_snapshots.outcome }}"' in gate
    assert 'if [ "$rollback_outcome" = "failure" ]; then' in gate
    assert "Pre-compaction snapshot restoration failed; refusing to commit state." in gate
    assert "The size guard compacted journals, but rollback reported no restored files; refusing to commit." in gate
    assert gate.index('if [ "$rollback_outcome" = "failure" ]; then') < gate.index('if find data/retention_archive -type f -print -quit')


def test_commit_gate_blocks_compaction_when_archive_is_empty_and_rollback_was_skipped(tmp_path):
    import json
    import subprocess

    import yaml

    workflow_path = Path(__file__).parent / ".github" / "workflows" / "event-engine.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["run"]["steps"]
    gate = next(step for step in steps if step.get("id") == "retention_archive_commit_gate")
    script = gate["run"]
    substitutions = {
        "${{ steps.restore_pre_compaction_snapshots.outcome }}": "skipped",
        "${{ steps.restore_pre_compaction_snapshots.outputs.restored_any }}": "",
        "${{ steps.size_guard.outcome }}": "success",
        "${{ steps.retention_archive_restore_verification.outcome }}": "success",
        "${{ steps.size_guard.outcome }}": "success",
        "${{ steps.restore_retention_archive.outputs.recovery_safe }}": "true",
        "${{ steps.retention_archive_upload_decision.outcome }}": "success",
        "${{ steps.retention_archive_upload_decision.outputs.archive_upload_required }}": "false",
        "${{ steps.upload_retention_archive.outcome }}": "skipped",
        "${{ steps.mark_retention_archive.outcome }}": "skipped",
        "${{ steps.retention_archive_upload_decision.outputs.archive_sha256 }}": "",
    }
    for source, target in substitutions.items():
        script = script.replace(source, target)

    data_dir = tmp_path / "data"
    (data_dir / "retention_archive").mkdir(parents=True)
    (data_dir / "data_retention_manifest.json").write_text(json.dumps({
        "mode": "size_guard",
        "files": {"research_outcomes.jsonl": {"triggered": True}},
    }), encoding="utf-8")

    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=tmp_path,
                            text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "no retention archive is available" in result.stdout + result.stderr


def test_commit_gate_allows_empty_archive_when_size_guard_did_not_compact(tmp_path):
    import json
    import subprocess

    import yaml

    workflow_path = Path(__file__).parent / ".github" / "workflows" / "event-engine.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["run"]["steps"]
    gate = next(step for step in steps if step.get("id") == "retention_archive_commit_gate")
    script = gate["run"]
    substitutions = {
        "${{ steps.restore_pre_compaction_snapshots.outcome }}": "skipped",
        "${{ steps.restore_pre_compaction_snapshots.outputs.restored_any }}": "",
        "${{ steps.size_guard.outcome }}": "success",
        "${{ steps.retention_archive_restore_verification.outcome }}": "success",
        "${{ steps.restore_retention_archive.outputs.recovery_safe }}": "true",
        "${{ steps.retention_archive_upload_decision.outcome }}": "success",
        "${{ steps.retention_archive_upload_decision.outputs.archive_upload_required }}": "false",
        "${{ steps.upload_retention_archive.outcome }}": "skipped",
        "${{ steps.mark_retention_archive.outcome }}": "skipped",
        "${{ steps.retention_archive_upload_decision.outputs.archive_sha256 }}": "",
    }
    for source, target in substitutions.items():
        script = script.replace(source, target)

    data_dir = tmp_path / "data"
    (data_dir / "retention_archive").mkdir(parents=True)
    (data_dir / "data_retention_manifest.json").write_text(json.dumps({
        "mode": "size_guard",
        "files": {"research_outcomes.jsonl": {"triggered": False}},
    }), encoding="utf-8")

    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=tmp_path,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Retention archive commit gate passed." in result.stdout
