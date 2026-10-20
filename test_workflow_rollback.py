from pathlib import Path


def test_workflow_restores_snapshots_on_retention_archive_upload_failure():
    workflow = (Path(__file__).parent / ".github" / "workflows" / "event-engine.yml").read_text(encoding="utf-8")
    assert "id: restore_pre_compaction_snapshots" in workflow
    assert "--restore-pre-compaction-snapshots" in workflow
    assert "steps.upload_retention_archive.outcome != 'success'" in workflow
    assert 'steps.restore_pre_compaction_snapshots.outputs.restored_any' in workflow
    assert "Upload durable runtime-state artifact" in workflow
    assert "Retain only the two newest runtime-state artifacts" in workflow


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
        "${{ steps.quality_gate.outcome }}": "success",
        "${{ steps.verify_tests_data.outcome }}": "success",
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
        "${{ steps.quality_gate.outcome }}": "success",
        "${{ steps.verify_tests_data.outcome }}": "success",
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


def test_workflow_does_not_write_research_or_commit_state_after_test_failure(tmp_path):
    import subprocess

    import yaml

    workflow_path = Path(__file__).parent / ".github" / "workflows" / "event-engine.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["run"]["steps"]
    by_id = {step.get("id"): step for step in steps if step.get("id")}

    assert by_id["tests"]["name"] == "Run tests"
    assert "restore_research_history_seed" not in by_id
    assert "research_history_seed" not in (Path(__file__).parent / ".github" / "workflows" / "event-engine.yml").read_text(encoding="utf-8")
    assert by_id["clean_start_bootstrap"]["if"] == "${{ steps.restore_runtime_state.outputs.mode == 'bootstrap' }}"
    assert "steps.restore_runtime_state.outputs.mode == 'artifact' && 'true' || 'false'" in by_id["run_engine"]["env"]["EXECUTION_ENABLED"]
    assert "steps.quality_gate.outcome == 'success'" in by_id["research_outcomes"]["if"]
    assert '${{ steps.quality_gate.outcome }}' in by_id["retention_archive_commit_gate"]["run"]
    assert by_id["test_policy"]["id"] == "test_policy"
    assert "data/source_test_fingerprint.json" in by_id["test_policy"]["run"]
    assert "required=true" in by_id["test_policy"]["run"]
    assert by_id["source_fingerprint"]["id"] == "source_fingerprint"
    assert 'subprocess.check_output(["git", "ls-files", "-z"])' in by_id["source_fingerprint"]["run"]
    # requirements.txt allows compatible version ranges; test cache identity must
    # change if Python or any installed dependency changes, not only source files.
    assert '"-m", "pip", "freeze"' in by_id["source_fingerprint"]["run"]
    assert "PYTHON_VERSION" in by_id["source_fingerprint"]["run"]
    assert "steps.record_source_fingerprint.outcome" not in by_id["retention_archive_commit_gate"]["run"]
    assert "steps.tests.outcome == 'success'" in by_id["record_source_fingerprint"]["if"]
    persist_step = next(step for step in steps if step.get("id") == "externalize_runtime_state")
    assert "steps.quality_gate.outcome == 'success'" in persist_step["if"]
    assert "steps.verify_runtime_state_artifact.outcome == 'success'" in persist_step["if"]
    assert "git rm --cached -r -f --ignore-unmatch data/" in persist_step["run"]

    script = by_id["retention_archive_commit_gate"]["run"]
    script = script.replace("${{ steps.quality_gate.outcome }}", "failure")
    script = script.replace("${{ steps.verify_tests_data.outcome }}", "success")
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Quality gate did not pass (outcome=failure); refusing to commit persistent state." in result.stdout + result.stderr


def test_workflow_blocks_state_commit_when_runtime_data_isolation_check_is_skipped(tmp_path):
    import subprocess

    import yaml

    workflow_path = Path(__file__).parent / ".github" / "workflows" / "event-engine.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["run"]["steps"]
    gate = next(step for step in steps if step.get("id") == "retention_archive_commit_gate")
    script = gate["run"]
    script = script.replace("${{ steps.quality_gate.outcome }}", "success")
    script = script.replace("${{ steps.verify_tests_data.outcome }}", "skipped")
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Runtime-data isolation verification did not pass (outcome=skipped); refusing to commit persistent state." in result.stdout + result.stderr


def test_full_test_suite_is_skipped_only_when_source_fingerprint_matches(tmp_path):
    import json
    import os
    import subprocess
    import yaml

    workflow_path = Path(__file__).parent / ".github" / "workflows" / "event-engine.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["run"]["steps"]
    policy = next(step for step in steps if step.get("id") == "test_policy")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    marker = data_dir / "source_test_fingerprint.json"
    output = tmp_path / "github_output.txt"

    def run_policy(fingerprint: str) -> tuple[str, subprocess.CompletedProcess[str]]:
        output.write_text("", encoding="utf-8")
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", policy["run"]],
            cwd=tmp_path, text=True, capture_output=True, check=False,
            env={**os.environ, "CURRENT_SOURCE_FINGERPRINT": fingerprint, "GITHUB_OUTPUT": str(output)},
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return output.read_text(encoding="utf-8"), result

    marker.write_text(json.dumps({"schema_version": 1, "sha256": "same"}), encoding="utf-8")
    out, result = run_policy("same")
    assert "run_tests=false" in out
    assert "skipping redundant tests" in result.stdout

    out, _ = run_policy("changed")
    assert "run_tests=true" in out

    marker.write_text("{}", encoding="utf-8")
    out, _ = run_policy("same")
    assert "run_tests=true" in out


def test_runtime_state_is_not_externalized_until_uploaded_bundle_and_retention_archive_are_verified():
    import yaml

    workflow_path = Path(__file__).parent / ".github" / "workflows" / "event-engine.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["run"]["steps"]
    by_id = {step.get("id"): step for step in steps if step.get("id")}

    detect_legacy = by_id["legacy_runtime_state"]
    verify_bundle = by_id["verify_uploaded_runtime_state_bundle"]
    externalize = by_id["externalize_runtime_state"]
    assert "git ls-files -- data/" in detect_legacy["run"]
    assert "bootstrap/research_history_seed" not in detect_legacy["run"]
    assert "steps.verify_runtime_state_artifact.outcome == 'success'" in verify_bundle["if"]
    assert "steps.legacy_runtime_state.outputs.tracked == 'true'" not in verify_bundle["if"]
    assert 'runtime_state_bundle.py verify --bundle-dir "$tmp_dir"' in verify_bundle["run"]
    assert "steps.legacy_runtime_state.outputs.tracked == 'true'" in externalize["if"]
    assert "steps.verify_uploaded_runtime_state_bundle.outcome == 'success'" in externalize["if"]
    assert "steps.restore_retention_archive.outputs.recovery_safe == 'true'" in externalize["if"]
    assert "steps.retention_archive_restore_verification.outcome == 'success'" in externalize["if"]
    assert "steps.restore_retention_archive.outputs.recovery_safe != 'true'" not in externalize["if"]
    assert steps.index(detect_legacy) < steps.index(by_id["restore_runtime_state"])
    assert steps.index(verify_bundle) < steps.index(by_id["prune_runtime_state_artifacts"])
    assert steps.index(verify_bundle) < steps.index(externalize)
