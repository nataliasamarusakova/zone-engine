from pathlib import Path


def test_retention_archive_search_processes_all_artifact_api_pages():
    workflow = (
        Path(__file__).parent / ".github" / "workflows" / "event-engine.yml"
    ).read_text(encoding="utf-8")

    assert "gh api --paginate --slurp" in workflow
    assert "all(.[]; .artifacts | type == \"array\")" in workflow
    assert "[.[].artifacts[] | select(.name == \"zone-engine-retention-archive\")]" in workflow
    assert "[ .[].artifacts[]" in workflow


def test_retention_restore_links_archive_to_persisted_marker_and_trusts_only_durable_partial_run():
    import yaml

    workflow = yaml.safe_load(
        (Path(__file__).parent / ".github" / "workflows" / "event-engine.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["run"]["steps"]
    restore = next(step for step in steps if step.get("id") == "restore_retention_archive")
    script = restore["run"]

    assert ".last_uploaded_by_run_id" in script
    assert 'select((.workflow_run.id | tostring) == $expected_run_id)' in script
    assert '"Upload cumulative retention archive for next clean runner" and .conclusion == "success"' in script
    assert '"Mark retention archive upload durable" and .conclusion == "success"' in script
    assert "--verify-restored" in script
    assert script.index("--verify-restored") < script.index('found_archive=true')
    assert "trying the next eligible candidate" in script
