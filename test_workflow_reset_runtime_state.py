from pathlib import Path

import yaml


def test_manual_runtime_reset_is_guarded_and_serialized_with_engine():
    path = Path(__file__).parent / ".github" / "workflows" / "reset-runtime-state.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert workflow["jobs"]["reset"]["if"] == "${{ inputs.confirmation == 'DELETE_PERSISTED_RUNTIME_STATE' }}"
    assert workflow["concurrency"]["group"] == "zone-engine"
    assert workflow["concurrency"]["cancel-in-progress"] is False
    text = path.read_text(encoding="utf-8")
    assert 'if [ "$GITHUB_REF" != "refs/heads/main" ]' in text
    assert "python -m event_engine.clean_start --data-dir data" in text
    assert "zone-engine-runtime-state" in text
    assert "zone-engine-retention-archive" in text
    assert "remaining" in text
    assert "zone-engine-diagnostic-" not in text


def test_reset_deletes_state_only_after_guarded_vst_clean_start():
    path = Path(__file__).parent / ".github" / "workflows" / "reset-runtime-state.yml"
    text = path.read_text(encoding="utf-8")
    guard = text.index("Verify VST account is flat and initialize empty local state")
    deletion = text.index("Delete all persisted runtime-state and retention-archive artifacts")
    assert guard < deletion
    assert "BINGX_ENV: vst" in text
    assert "https://open-api-vst.bingx.com" in text
