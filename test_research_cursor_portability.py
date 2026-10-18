from pathlib import Path

import research_forward


def _cursor_for(path: Path, offset: int) -> dict:
    sig = research_forward._journal_signature(path, offset)
    return {
        "offset": offset,
        "size": sig["size"],
        "inode": sig["inode"],
        "prefix_sha256": sig["prefix_sha256"],
        "window_start": sig["window_start"],
        "window_sha256": sig["window_sha256"],
    }


def test_cursor_valid_when_inode_differs_but_content_signature_matches(tmp_path, monkeypatch):
    path = tmp_path / "zone_observations.jsonl"
    payload = b'{"observation_id":"OBS1"}\n{"observation_id":"OBS2"}\n'
    path.write_bytes(payload)
    cursor = _cursor_for(path, len(payload))
    # A different inode simulates the same persisted journal in a fresh runner.
    cursor["inode"] = cursor["inode"] + 100000
    assert research_forward._cursor_is_valid(path, cursor)


def test_cursor_invalid_when_content_prefix_changes(tmp_path):
    path = tmp_path / "zone_observations.jsonl"
    payload = b'{"observation_id":"OBS1"}\n{"observation_id":"OBS2"}\n'
    path.write_bytes(payload)
    cursor = _cursor_for(path, len(payload))
    path.write_bytes(b'{"observation_id":"CHANGED"}\n{"observation_id":"OBS2"}\n')
    assert not research_forward._cursor_is_valid(path, cursor)
