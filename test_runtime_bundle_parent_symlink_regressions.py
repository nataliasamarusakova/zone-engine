import json
from pathlib import Path

import pytest

from event_engine import runtime_state_bundle as bundle


def test_create_bundle_refuses_symlinked_output_parent_without_deleting_external_data(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "_validate_required_paths", lambda _paths, **_kwargs: None)
    monkeypatch.setattr(bundle, "_validate_critical_json_state", lambda _data_dir: None)
    source = tmp_path / "source-data"
    source.mkdir()
    (source / "state.json").write_text("{}", encoding="utf-8")

    outside = tmp_path / "outside"
    existing = outside / "bundle"
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    link_parent = tmp_path / "linked-parent"
    link_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        bundle.create_bundle(data_dir=source, output_dir=link_parent / "bundle")
    assert marker.read_text(encoding="utf-8") == "keep"


def test_restore_bundle_refuses_symlinked_target_parent_without_replacing_external_state(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "_validate_required_paths", lambda _paths, **_kwargs: None)
    monkeypatch.setattr(bundle, "_validate_critical_json_state", lambda _data_dir: None)
    source = tmp_path / "source-data"
    source.mkdir()
    (source / "state.json").write_text(json.dumps({"source": True}), encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    bundle.create_bundle(data_dir=source, output_dir=bundle_dir)

    outside = tmp_path / "outside"
    target = outside / "runtime"
    target.mkdir(parents=True)
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    link_parent = tmp_path / "linked-parent"
    link_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        bundle.restore_bundle(bundle_dir=bundle_dir, data_dir=link_parent / "runtime")
    assert marker.read_text(encoding="utf-8") == "keep"
