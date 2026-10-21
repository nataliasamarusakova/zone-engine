from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from event_engine import research


def _bar(ts: pd.Timestamp) -> dict:
    return {
        "timestamp": ts.isoformat(),
        "open": 100.0,
        "high": 102.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 10.0,
        "close_time": int((ts + pd.Timedelta(minutes=5)).timestamp() * 1000),
    }


def _isolate_paths(tmp_path: Path, monkeypatch) -> None:
    for name in [
        "ZONE_OBSERVATIONS_PATH", "MARKET_BARS_1H_PATH", "MARKET_BARS_5M_PATH",
        "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH",
    ]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    monkeypatch.setattr(research, "_OBSERVATION_SEEN_IDS", None)
    monkeypatch.setattr(research, "_OBSERVATION_SEEN_IDS_PATH", None)
    monkeypatch.setattr(research, "_PERSISTENCE_BATCH", None)


def test_batch_persistence_flushes_observations_bars_cursors_and_counters(tmp_path, monkeypatch):
    _isolate_paths(tmp_path, monkeypatch)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    bar5 = _bar(now - pd.Timedelta(minutes=15))
    bar1 = {**_bar(now - pd.Timedelta(hours=2)), "close_time": int((now - pd.Timedelta(hours=1)).timestamp() * 1000)}

    with research.batch_persistence() as persistence_state:
        assert research.record_zone_observations([{"observation_id": "OBS_BATCH_1", "symbol": "AAA-USDT", "event_type": "TEST"}]) == 1
        assert research.record_zone_observations([{"observation_id": "OBS_BATCH_1", "symbol": "AAA-USDT", "event_type": "TEST"}]) == 0
        assert research.persist_market_bars("AAA-USDT", "1h", [bar1], provider="binance", source="test", scan_id="S1") == 1
        assert research.persist_market_bars("AAA-USDT", "5m", [bar5], provider="binance", source="test", scan_id="S1") == 1
        # Writes are buffered inside the context; no cursor may advance early.
        assert not (tmp_path / "zone_observations.jsonl").exists()
        assert not (tmp_path / "research_bar_cursors.json").exists()

    observations = [json.loads(x) for x in (tmp_path / "zone_observations.jsonl").read_text().splitlines()]
    bars1 = [json.loads(x) for x in (tmp_path / "market_bars_1h.jsonl").read_text().splitlines()]
    bars5 = [json.loads(x) for x in (tmp_path / "market_bars_5m.jsonl").read_text().splitlines()]
    cursors = json.loads((tmp_path / "research_bar_cursors.json").read_text())
    manifest = json.loads((tmp_path / "research_manifest.json").read_text())

    assert [x["observation_id"] for x in observations] == ["OBS_BATCH_1"]
    assert len(bars1) == len(bars5) == 1
    assert set(cursors) == {"AAA-USDT|1h|binance", "AAA-USDT|5m|binance"}
    assert manifest["zone_observations_written"] == 1
    assert manifest["market_bars_1h_written"] == 1
    assert manifest["market_bars_5m_written"] == 1
    assert manifest["observation_duplicates_skipped"] == 1
    assert persistence_state["flush_result"]["observations_written"] == 1
    assert persistence_state["flush_result"]["bars_1h_written"] == 1
    assert persistence_state["flush_result"]["bars_5m_written"] == 1
    assert persistence_state["flush_result"]["persistence_errors"] == 0


def test_batch_cursor_does_not_advance_when_bar_append_fails(tmp_path, monkeypatch):
    _isolate_paths(tmp_path, monkeypatch)
    now = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=2)
    bar = {**_bar(now), "close_time": int((now + pd.Timedelta(hours=1)).timestamp() * 1000)}
    original_append = research._append_market_bars_idempotent_locked

    def fail_hourly(path, rows):
        if path == research.MARKET_BARS_1H_PATH:
            raise OSError("simulated disk write failure")
        return original_append(path, rows)

    monkeypatch.setattr(research, "_append_market_bars_idempotent_locked", fail_hourly)
    with research.batch_persistence():
        assert research.persist_market_bars("AAA-USDT", "1h", [bar], provider="binance", source="test", scan_id="S1") == 1

    assert not (tmp_path / "market_bars_1h.jsonl").exists()
    cursors = json.loads((tmp_path / "research_bar_cursors.json").read_text()) if (tmp_path / "research_bar_cursors.json").exists() else {}
    assert "AAA-USDT|1h|binance" not in cursors
    errors = [json.loads(x) for x in (tmp_path / "research_persistence_errors.jsonl").read_text().splitlines()]
    assert any(row.get("kind") == "market_bar_batch_write" for row in errors)


def test_batch_keeps_existing_cursor_keys_for_other_symbols(tmp_path, monkeypatch):
    _isolate_paths(tmp_path, monkeypatch)
    now = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=3)
    old = {"last_timestamp": (now - pd.Timedelta(hours=1)).isoformat(), "last_bar_id": "OLD", "updated_at": "old", "scan_id": "old"}
    (tmp_path / "research_bar_cursors.json").write_text(json.dumps({"BBB-USDT|1h|binance": old}))
    bar = {**_bar(now), "close_time": int((now + pd.Timedelta(hours=1)).timestamp() * 1000)}

    with research.batch_persistence():
        research.persist_market_bars("AAA-USDT", "1h", [bar], provider="binance", source="test", scan_id="S1")

    cursors = json.loads((tmp_path / "research_bar_cursors.json").read_text())
    assert "AAA-USDT|1h|binance" in cursors
    assert cursors["BBB-USDT|1h|binance"] == old


def test_pending_forward_scan_seeds_id_cache_for_later_dedup(tmp_path, monkeypatch):
    _isolate_paths(tmp_path, monkeypatch)
    current = pd.Timestamp.now(tz="UTC")
    (tmp_path / "zone_observations.jsonl").write_text(
        json.dumps({
            "observation_id": "OBS_FROM_PRIOR_RUN",
            "symbol": "AAA-USDT",
            "observation_ts": current.isoformat(),
        }) + "\n",
        encoding="utf-8",
    )

    assert research.pending_forward_symbols(horizon_hours=24) == {"AAA-USDT"}
    assert "OBS_FROM_PRIOR_RUN" in research._OBSERVATION_SEEN_IDS

    def should_not_reload_journal():
        raise AssertionError("observation journal was redundantly re-read")

    monkeypatch.setattr(research, "_load_existing_observation_ids", should_not_reload_journal)
    assert research.record_zone_observations([{
        "observation_id": "OBS_NEW",
        "symbol": "BBB-USDT",
        "event_type": "TEST",
    }]) == 1


def test_batch_observation_write_failure_does_not_commit_bars_or_cursors(tmp_path, monkeypatch):
    _isolate_paths(tmp_path, monkeypatch)
    now = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=2)
    bar = {**_bar(now), "close_time": int((now + pd.Timedelta(hours=1)).timestamp() * 1000)}
    original_append = research._append_jsonl_locked

    def fail_observations(path, rows):
        if path == research.ZONE_OBSERVATIONS_PATH:
            raise OSError("simulated observation journal failure")
        return original_append(path, rows)

    monkeypatch.setattr(research, "_append_jsonl_locked", fail_observations)
    with research.batch_persistence() as persistence_state:
        assert research.record_zone_observations([{"observation_id": "OBS_FAIL", "symbol": "AAA-USDT", "event_type": "TEST"}]) == 1
        assert research.persist_market_bars("AAA-USDT", "1h", [bar], provider="binance", source="test", scan_id="S1") == 1

    assert not (tmp_path / "market_bars_1h.jsonl").exists()
    assert not (tmp_path / "research_bar_cursors.json").exists()
    result = persistence_state["flush_result"]
    assert result["observations_written"] == 0
    assert result["bars_1h_written"] == 0
    assert result["cursor_updates_written"] == 0
    assert result["bar_cursor_commit_skipped_observation_failure"] is True
    assert result["persistence_errors"] >= 1


def test_pending_forward_symbols_returns_partial_symbols_on_read_error(tmp_path, monkeypatch):
    from pathlib import Path

    _isolate_paths(tmp_path, monkeypatch)
    path = research.ZONE_OBSERVATIONS_PATH
    path.write_text("placeholder\n", encoding="utf-8")
    now = pd.Timestamp.now(tz="UTC")
    partial = json.dumps({"observation_id": "OBS_PARTIAL", "symbol": "AAA-USDT", "observation_ts": now.isoformat()})

    class BrokenReader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield partial + "\n"
            raise OSError("simulated mid-read failure")

    real_open = Path.open

    def flaky_open(self, *args, **kwargs):
        if self == path:
            return BrokenReader()
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)
    assert research.pending_forward_symbols(horizon_hours=24) == {"AAA-USDT"}
    assert research._OBSERVATION_SEEN_IDS is None


def test_research_persistence_batch_size_has_safe_fallback():
    import run_once

    assert run_once._research_persistence_batch_size("12") == 12
    assert run_once._research_persistence_batch_size("1") == 4
    assert run_once._research_persistence_batch_size("999") == 32
    assert run_once._research_persistence_batch_size("not-an-int") == 32


def test_research_persistence_batch_size_defaults_to_32(monkeypatch):
    import run_once

    monkeypatch.delenv("RESEARCH_PERSISTENCE_BATCH_SIZE", raising=False)
    assert run_once._research_persistence_batch_size() == 32


def test_tracker_position_snapshot_handles_hedge_and_one_way_rows():
    from event_engine import tracker

    snapshot = [
        {"symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "0.25", "avgPrice": "60000"},
        {"symbol": "ETH-USDT", "positionSide": "BOTH", "positionAmt": "-2", "entryPrice": "2500"},
        {"symbol": "SOL-USDT", "positionSide": "BOTH", "positionAmt": "3", "avgPrice": "150"},
    ]

    long_btc = tracker._position_from_snapshot(snapshot, "BTCUSDT", "LONG")
    short_eth = tracker._position_from_snapshot(snapshot, "ETH-USDT", "SHORT")
    long_sol = tracker._position_from_snapshot(snapshot, "SOL-USDT", "LONG")
    absent = tracker._position_from_snapshot(snapshot, "XRP-USDT", "SHORT")

    assert long_btc["status"] == "found" and long_btc["positionAmt"] == 0.25
    assert short_eth["status"] == "found" and short_eth["positionAmt"] == 2.0
    assert long_sol["status"] == "found" and long_sol["avgPrice"] == 150.0
    assert absent["status"] == "not_found"


def test_tracker_snapshot_wrapper_passes_successful_bulk_snapshot(monkeypatch):
    import run_once

    snapshot = [{"symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "1", "avgPrice": "60000"}]
    calls = []
    monkeypatch.setattr(run_once, "get_positions", lambda **kwargs: snapshot)
    monkeypatch.setattr(run_once, "update_active_trades", lambda **kwargs: calls.append(kwargs))

    run_once._update_active_trades_with_snapshot()

    assert calls == [{"position_snapshot": snapshot}]


def test_tracker_snapshot_wrapper_falls_back_when_bulk_fetch_fails(monkeypatch):
    import run_once

    calls = []

    def fail_bulk(**kwargs):
        raise RuntimeError("bulk endpoint unavailable")

    monkeypatch.setattr(run_once, "get_positions", fail_bulk)
    monkeypatch.setattr(run_once, "update_active_trades", lambda: calls.append("per_symbol_fallback"))

    run_once._update_active_trades_with_snapshot()

    assert calls == ["per_symbol_fallback"]


def test_tracker_snapshot_wrapper_supports_legacy_zero_arg_callable(monkeypatch):
    import run_once

    calls = []
    monkeypatch.setattr(run_once, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(run_once, "update_active_trades", lambda: calls.append("legacy_call"))

    run_once._update_active_trades_with_snapshot()

    assert calls == ["legacy_call"]
