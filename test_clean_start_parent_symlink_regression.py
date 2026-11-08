from pathlib import Path

import pytest

from event_engine import clean_start


def test_clean_start_refuses_symlinked_parent_without_deleting_target(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    runtime = outside / "runtime"
    runtime.mkdir(parents=True)
    marker = runtime / "keep.txt"
    marker.write_text("do not delete", encoding="utf-8")

    link_parent = tmp_path / "checkout-data-link"
    link_parent.symlink_to(outside, target_is_directory=True)
    requested = link_parent / "runtime"

    monkeypatch.setattr(clean_start, "REQUIRED_FILES", ("data/active_trades.json",))
    with pytest.raises(RuntimeError, match="symlink"):
        clean_start.initialize_empty_runtime(requested)

    assert marker.read_text(encoding="utf-8") == "do not delete"
