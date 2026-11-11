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
    monkeypatch.setattr(research, "_MARKET_BAR_INDEX_CACHE", {})


def test_market_bar_idempotency_index_reuses_scan_and_invalidates_after_external_append(tmp_path, monkeypatch):
    research._MARKET_BAR_INDEX_CACHE.clear()
    path = tmp_path / "market_bars_5m.jsonl"
    original_reader = research._read_market_bar_identity_index
    read_calls = []

    def counted_reader(target):
        read_calls.append(target)
        return original_reader(target)

    monkeypatch.setattr(research, "_read_market_bar_identity_index", counted_reader)
    start = pd.Timestamp("2026-10-01T00:00:00Z")
    rows = [_bar(start + pd.Timedelta(minutes=5 * index)) for index in range(4)]
    rows = [
        {**row, "symbol": "AAA-USDT", "timeframe": "5m", "provider": "binance"}
        for row in rows
    ]

    assert research._append_market_bars_idempotent_locked(path, rows[:1]) == 1
    # A second normal batch must reuse the in-memory identity index.
    assert research._append_market_bars_idempotent_locked(path, rows[1:2]) == 1
    assert research._append_market_bars_idempotent_locked(path, rows[:1]) == 0
    assert len(read_calls) == 1

    # Simulate a durable writer outside this process/cache. The file signature
    # must invalidate it and preserve deduplication after the external append.
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rows[2], ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()

    assert research._append_market_bars_idempotent_locked(path, rows[2:3]) == 0
    assert len(read_calls) == 2

    # Retention restoration/compaction may atomically replace the journal. A new
    # inode with the same rows must trigger a rebuild rather than trust stale IDs.
    replacement = tmp_path / "market_bars_5m.replacement.jsonl"
    replacement.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows[:3]),
        encoding="utf-8",
    )
    replacement.replace(path)
    assert research._append_market_bars_idempotent_locked(path, rows[2:3]) == 0
    assert len(read_calls) == 3

    assert research._append_market_bars_idempotent_locked(path, rows[3:]) == 1
    assert len(read_calls) == 3

    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["timestamp"] for row in saved] == [row["timestamp"] for row in rows]
    research._MARKET_BAR_INDEX_CACHE.clear()


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


def test_load_market_bar_streams_normalizes_then_orders_by_utc_timestamp_key(tmp_path, monkeypatch):
    _isolate_paths(tmp_path, monkeypatch)
    rows = [
        {
            "symbol": "AAA-USDT", "timeframe": "5m", "provider": "binance",
            "timestamp": "2026-01-01T00:30:00+02:00", "open": 1, "high": 2,
            "low": 0.5, "close": 1.5, "volume": 10,
        },
        {
            "symbol": "AAA-USDT", "timeframe": "5m", "provider": "binance",
            "timestamp": "2025-12-31T23:00:00Z", "open": 1, "high": 2,
            "low": 0.5, "close": 1.6, "volume": 10,
        },
        {
            "symbol": "AAA-USDT", "timeframe": "5m", "provider": "binance",
            "timestamp": "2026-01-01T02:00:00+03:00", "open": 1, "high": 2,
            "low": 0.5, "close": 1.7, "volume": 10,
        },
        {
            "symbol": "AAA-USDT", "timeframe": "5m", "provider": "binance",
            "timestamp": "2026-01-01T03:00:00Z", "open": 1, "high": 2,
            "low": 0.5, "close": 1.8, "volume": 10,
        },
    ]
    research.MARKET_BARS_5M_PATH.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )

    loaded = research.load_market_bar_streams([("AAA-USDT", "5m", "binance")])
    stream = loaded[("AAA-USDT", "5m", "binance")]

    assert [row["timestamp"] for row in stream] == [
        "2025-12-31T22:30:00+00:00",
        "2025-12-31T23:00:00+00:00",
        "2026-01-01T03:00:00+00:00",
    ]
    # Two source representations identify the same UTC instant; the last row wins.
    assert [row["close"] for row in stream] == [1.5, 1.7, 1.8]


def test_merge_market_bar_history_normalizes_offsets_and_current_values_win():
    persisted = [
        {"timestamp": "2026-01-01T00:00:00+02:00", "close": 100.0, "quote_volume": 50.0},
        {"timestamp": "2026-01-01T03:00:00Z", "close": 103.0},
    ]
    current = [
        {"timestamp": "2025-12-31T22:00:00Z", "close": 101.0, "volume": 5.0},
        {"timestamp": "2026-01-01T04:00:00+00:00", "close": 104.0},
    ]

    merged = research.merge_market_bar_history(persisted, current, timeframe="5m")

    assert [row["timestamp"] for row in merged] == [
        "2025-12-31T22:00:00+00:00",
        "2026-01-01T03:00:00+00:00",
        "2026-01-01T04:00:00+00:00",
    ]
    assert merged[0]["close"] == 101.0
    assert merged[0]["quote_volume"] == 50.0
    assert merged[0]["volume"] == 5.0


def test_research_observation_predicate_distinguishes_bar_only_rows():
    assert research.requires_observation_build([], {"touch_events": [], "rearm_events": [], "zones": {}}) is False
    assert research.requires_observation_build([], {"touch_events": [{"reason": "signal_created"}], "rearm_events": [], "zones": {}}) is False
    assert research.requires_observation_build([], {"touch_events": [{"reason": "directional_candle_required"}]}) is True
    assert research.requires_observation_build([{"event_id": "EVT_1"}], {}) is True
    assert research.requires_observation_build([], {"rearm_events": [{"timestamp": "2026-01-01T00:00:00Z"}]}) is True
    assert research.requires_observation_build([], {
        "zones": {"DEMAND:Z1": {
            "closest_midpoint_bar": {"timestamp": "2026-01-01T00:00:00Z"},
            "closest_midpoint_distance_pct": 1.0,
        }}
    }) is True
    assert research.requires_observation_build([], {
        "zones": {"DEMAND:Z1": {
            "closest_midpoint_bar": {"timestamp": "2026-01-01T00:00:00Z"},
            "closest_midpoint_distance_pct": 1.01,
        }}
    }) is False


def test_bar_only_deferred_payload_does_not_merge_persisted_history():
    import run_once

    current_bars = [{"timestamp": 1794400000000 + i * 300000} for i in range(144)]
    persisted_bars = [{"timestamp": f"2026-01-01T00:{i:02d}:00+00:00"} for i in range(30)]
    payload = {
        "symbol": "AAA-USDT", "provider": "binance", "bars_5m": current_bars,
        "persist_bars_for_forward": True, "signals": [],
        "diagnostics": {"touch_events": [], "rearm_events": [], "zones": {}},
    }
    history = {("AAA-USDT", "5m", "binance"): persisted_bars}

    prepared = run_once._prepare_deferred_research_5m_bars(history, payload, None)

    assert prepared == current_bars
    assert len(prepared) == 144


def test_deferred_history_streams_include_only_payloads_that_build_observations():
    import run_once

    payloads = [
        {"symbol": "AAA-USDT", "provider": "binance", "signals": [], "diagnostics": {"zones": {}}},
        {"symbol": "BBB-USDT", "provider": "bingx", "signals": [{"event_id": "E1"}], "diagnostics": {}},
        {"symbol": "CCC-USDT", "provider": "binance", "signals": [], "diagnostics": {"rearm_events": [{"timestamp": "2026-01-01T00:00:00Z"}]}},
    ]

    assert list(run_once._deferred_research_history_stream_keys(payloads)) == [
        ("BBB-USDT", "5m", "bingx"),
        ("CCC-USDT", "5m", "binance"),
    ]


def test_persist_market_bars_skips_reparsing_valid_bars_at_or_before_cursor(tmp_path, monkeypatch):
    _isolate_paths(tmp_path, monkeypatch)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    cursor_ts = now - pd.Timedelta(minutes=15)
    key = "AAA-USDT|5m|binance"
    (tmp_path / "research_bar_cursors.json").write_text(json.dumps({
        key: {"last_timestamp": cursor_ts.isoformat(), "last_bar_id": "CURSOR"},
    }), encoding="utf-8")
    bars = []
    for index in range(60):
        ts = cursor_ts - pd.Timedelta(minutes=5 * (59 - index))
        bars.append({
            "timestamp": int(ts.timestamp() * 1000),
            "close_time": int((ts + pd.Timedelta(minutes=5)).timestamp() * 1000),
            "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 10.0,
        })

    original_parse = research._parse_timestamp
    parse_calls = []

    def counted_parse(value):
        parse_calls.append(value)
        return original_parse(value)

    monkeypatch.setattr(research, "_parse_timestamp", counted_parse)
    result = research.persist_market_bars(
        "AAA-USDT", "5m", bars, provider="binance", source="test", scan_id="S_NOOP",
    )

    assert result == 0
    assert len(parse_calls) == 1  # Only the saved cursor is parsed, not every old bar.
    assert not (tmp_path / "market_bars_5m.jsonl").exists()
    saved_cursor = json.loads((tmp_path / "research_bar_cursors.json").read_text())[key]
    assert saved_cursor["last_timestamp"] == cursor_ts.isoformat()


def test_persist_market_bars_fast_path_keeps_only_new_rows_even_when_input_is_unsorted(tmp_path, monkeypatch):
    _isolate_paths(tmp_path, monkeypatch)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    cursor_ts = now - pd.Timedelta(minutes=15)
    key = "AAA-USDT|5m|binance"
    (tmp_path / "research_bar_cursors.json").write_text(json.dumps({
        key: {"last_timestamp": cursor_ts.isoformat(), "last_bar_id": "CURSOR"},
    }), encoding="utf-8")

    def row(ts):
        return {
            "timestamp": int(ts.timestamp() * 1000),
            "close_time": int((ts + pd.Timedelta(minutes=5)).timestamp() * 1000),
            "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 10.0,
        }

    old_1 = row(cursor_ts - pd.Timedelta(minutes=10))
    old_2 = row(cursor_ts - pd.Timedelta(minutes=5))
    new_1 = row(cursor_ts + pd.Timedelta(minutes=5))
    new_2 = row(cursor_ts + pd.Timedelta(minutes=10))
    result = research.persist_market_bars(
        "AAA-USDT", "5m", [old_2, new_2, old_1, new_1],
        provider="binance", source="test", scan_id="S_PARTIAL",
    )

    stored = [json.loads(line) for line in (tmp_path / "market_bars_5m.jsonl").read_text().splitlines()]
    assert result == 2
    assert [row["timestamp"] for row in stored] == [
        (cursor_ts + pd.Timedelta(minutes=5)).isoformat(),
        (cursor_ts + pd.Timedelta(minutes=10)).isoformat(),
    ]
    cursors = json.loads((tmp_path / "research_bar_cursors.json").read_text())
    assert cursors[key]["last_timestamp"] == (cursor_ts + pd.Timedelta(minutes=10)).isoformat()


def test_persist_market_bars_fast_path_falls_back_for_invalid_old_rows(tmp_path, monkeypatch):
    _isolate_paths(tmp_path, monkeypatch)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    cursor_ts = now - pd.Timedelta(minutes=15)
    key = "AAA-USDT|5m|binance"
    (tmp_path / "research_bar_cursors.json").write_text(json.dumps({
        key: {"last_timestamp": cursor_ts.isoformat(), "last_bar_id": "CURSOR"},
    }), encoding="utf-8")
    invalid_old = {
        "timestamp": int((cursor_ts - pd.Timedelta(minutes=5)).timestamp() * 1000),
        "close_time": int(cursor_ts.timestamp() * 1000),
        "open": 100.0, "high": 98.0, "low": 99.0, "close": 101.0, "volume": 10.0,
    }

    assert research.persist_market_bars(
        "AAA-USDT", "5m", [invalid_old], provider="binance", source="test", scan_id="S_INVALID",
    ) == 0
    manifest = json.loads((tmp_path / "research_manifest.json").read_text())
    assert manifest["market_bars_invalid"] == 1
    assert not (tmp_path / "market_bars_5m.jsonl").exists()


def test_persist_market_bars_seed_history_bypasses_cursor_fast_path(tmp_path, monkeypatch):
    _isolate_paths(tmp_path, monkeypatch)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    cursor_ts = now - pd.Timedelta(minutes=10)
    key = "AAA-USDT|5m|binance"
    (tmp_path / "research_bar_cursors.json").write_text(json.dumps({
        key: {"last_timestamp": cursor_ts.isoformat(), "last_bar_id": "CURSOR"},
    }), encoding="utf-8")
    bars = []
    for index in range(3):
        ts = cursor_ts - pd.Timedelta(minutes=5 * (2 - index))
        bars.append({
            "timestamp": int(ts.timestamp() * 1000),
            "close_time": int((ts + pd.Timedelta(minutes=5)).timestamp() * 1000),
            "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 10.0,
        })

    assert research.persist_market_bars(
        "AAA-USDT", "5m", bars, provider="binance", source="test", scan_id="S_SEED",
        seed_history=True,
    ) == 3
    stored = [json.loads(line) for line in (tmp_path / "market_bars_5m.jsonl").read_text().splitlines()]
    assert len(stored) == 3
    saved_cursor = json.loads((tmp_path / "research_bar_cursors.json").read_text())[key]
    assert saved_cursor["history_seeded"] is True
    assert saved_cursor["history_seeded_count"] == 3

