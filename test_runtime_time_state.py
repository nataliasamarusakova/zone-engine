from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_provider_epoch_close_time_is_normalized_to_aware_utc():
    import run_once

    raw_epoch_ms = 1790697600000
    ts = run_once._coerce_utc_timestamp(raw_epoch_ms)
    assert ts is not None
    assert ts.tzinfo is not None
    assert str(ts.tz) == "UTC"

    now = pd.Timestamp.now(tz="UTC")
    age_hours = (now - ts).total_seconds() / 3600.0
    assert isinstance(age_hours, float)


def test_fresh_runtime_bootstraps_empty_active_state(monkeypatch, tmp_path):
    import run_once

    path = tmp_path / "data" / "active_trades.json"
    monkeypatch.setattr(run_once, "DATA", tmp_path / "data")
    state = run_once._load_active_trades_file(fresh_runtime_start=True)
    assert state == {}
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {}


def test_missing_state_is_not_bootstrapped_on_existing_runtime(tmp_path, monkeypatch):
    import run_once

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(run_once, "DATA", data_dir)
    try:
        run_once._load_active_trades_file(fresh_runtime_start=False)
    except RuntimeError as exc:
        assert "active_trades.json is untrusted" in str(exc)
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing active state on an existing runtime must block")
