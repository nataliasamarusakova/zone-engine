from __future__ import annotations

from pathlib import Path

import yaml

from event_engine.runtime_state_bundle import BUNDLE_SCHEMA_VERSION


def test_clean_start_is_guarded_and_cannot_open_trades_on_bootstrap_run():
    workflow_path = Path(__file__).parent / ".github" / "workflows" / "event-engine.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["run"]["steps"]
    by_id = {step.get("id"): step for step in steps if step.get("id")}

    assert BUNDLE_SCHEMA_VERSION == 2
    assert by_id["clean_start_bootstrap"]["if"] == "${{ steps.restore_runtime_state.outputs.mode == 'bootstrap' }}"
    assert "python -m event_engine.clean_start --data-dir data" == by_id["clean_start_bootstrap"]["run"]
    assert by_id["run_engine"]["env"]["EXECUTION_ENABLED"] == "${{ steps.restore_runtime_state.outputs.mode == 'artifact' && 'true' || 'false' }}"
    assert "steps.verify_runtime_state_artifact.outcome == 'success'" in by_id["verify_uploaded_runtime_state_bundle"]["if"]
    assert "steps.restore_runtime_state.outputs.mode != 'bootstrap' || steps.run_engine.outcome == 'success'" in by_id["prepare_runtime_state_bundle"]["if"]
    assert "steps.restore_pre_compaction_snapshots.outcome == 'success'" in by_id["prepare_runtime_state_bundle"]["if"]
    assert "steps.retention_archive_commit_gate.outcome == 'success'" in by_id["prepare_runtime_state_bundle"]["if"]
    assert "steps.legacy_runtime_state.outputs.tracked == 'true'" not in by_id["verify_uploaded_runtime_state_bundle"]["if"]


def test_self_verifying_completed_event_engine_runs_are_candidates_and_old_schema_is_not_restored():
    workflow = (Path(__file__).parent / ".github" / "workflows" / "event-engine.yml").read_text(encoding="utf-8")
    restore_script = workflow.split("- name: Restore latest durable runtime-state artifact", 1)[1].split("- name: Set up Python", 1)[0]
    assert '.status == "completed"' in restore_script
    assert '.conclusion == "success"' in restore_script
    assert '.event == "repository_dispatch"' in restore_script
    assert 'select(.name == "zone-engine-runtime-state")] | length' in restore_script
    assert "possibly expired/ineligible artifacts" in restore_script
    assert 'if [ "$bundle_schema" -lt 2 ]; then' in workflow
    assert 'echo "mode=bootstrap" >> "$GITHUB_OUTPUT"' in workflow
    assert 'if [ "$bundle_schema" -ne 2 ]; then' in workflow
    assert "check-legacy" not in workflow
    assert "research_history_seed" not in workflow
