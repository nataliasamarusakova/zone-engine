from pathlib import Path


def test_retention_archive_search_processes_all_artifact_api_pages():
    workflow = (
        Path(__file__).parent / ".github" / "workflows" / "event-engine.yml"
    ).read_text(encoding="utf-8")

    assert "gh api --paginate --slurp" in workflow
    assert "all(.[]; .artifacts | type == \"array\")" in workflow
    assert "[.[].artifacts[] | select(.name == \"zone-engine-retention-archive\")]" in workflow
    assert "[ .[].artifacts[]" in workflow
