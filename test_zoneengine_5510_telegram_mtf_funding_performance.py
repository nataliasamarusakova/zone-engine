from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


def _hourly_bars(count: int, start: pd.Timestamp | None = None) -> pd.DataFrame:
    start = start or pd.Timestamp("2026-10-01T00:00:00Z")
    rows = []
    for index in range(count):
        ts = start + pd.Timedelta(hours=index)
        base = 100.0 + (index % 7) * 0.05
        rows.append({
            "timestamp": ts,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.2,
            "volume": 10.0 + index,
        })
    return pd.DataFrame(rows)


def test_telegram_compact_format_uses_requested_layout_and_zone_proximity():
    from event_engine.telegram import format_signal

    msg = format_signal(
        {
            "type": "SHORT", "symbol": "G-USDT", "score": 60.0,
            "entry": 0.004174, "sl": 0.0045914, "tp1": 0.00404878,
            "tp2": 0.00392356, "risk_pct": 10.0, "tp1_rr": 0.3, "tp2_rr": 0.6,
            "zone": {"kind": "SUPPLY"},
            "target": {"tp1_pct": 3.0, "tp2_pct": 6.0, "tp1_close_fraction": 0.5, "tp2_close_fraction": 0.5},
            "confirmation": {
                "volume_ratio": 0.554,
                "mtf_zone_confluence": {
                    "status": "confluent", "kind": "DEMAND", "overlap": False,
                    "one_hour": {"poi": 77300.0}, "three_hour": {"poi": 77130.0},
                    "poi_distance_pct": abs(77300.0 - 77130.0) / 77300.0 * 100,
                },
            },
        },
        execution={
            "status": "opened_protected",
            "order": {"order_id": "2116810510872612864"},
            "protection": {"status": "PROTECTED"},
        },
        funding={"status": "ok", "funding_rate_pct": 0.01},
    )
    expected = "\n".join([
        '<a href="https://web.telegram.org/k/assets/img/emoji/1f534.png">🔴</a> SHORT · <code>GUSDT</code>',
        "",
        "Score: <code>60.0</code>",
        "Zone: <code>SUPPLY</code>",
        "Entry reference: <code>0.004174</code>",
        "SL: <code>0.0045914</code>",
        "TP1: <code>0.00404878 (3% / 0.3R / 50%)</code>",
        "TP2: <code>0.00392356 (6% / 0.6R / 50%)</code>",
        "Risk: <code>10%</code>",
        "",
        "CONFIRMATION",
        "Volume / SMA20: <code>0.554</code>",
        "Zone 1H: <code>DEMAND 77,300</code>",
        "Zone 3H: <code>DEMAND 77,130</code>",
        "Calculation: <code>NEAR · 0.22%</code>",
        "Funding: <code>+0.0100%</code>",
        "",
        "EXECUTION",
        "Status: <code>opened_protected</code>",
        "Order: <code>2116810510872612864</code>",
        "Protection: <code>PROTECTED</code>",
    ])
    assert msg == expected
    assert "1H / 3H zones:" not in msg
    assert "Funding (last):" not in msg
    assert "<b>" not in msg
    assert "<code>0.004174</code>" in msg
    assert "<code>NEAR · 0.22%</code>" in msg
    assert "<code>2116810510872612864</code>" in msg
    assert "Signal:" not in msg
    assert "Demand zones:" not in msg
    assert "Candle body / ATR:" not in msg
    assert "Age:" not in msg


def test_telegram_zone_proximity_reports_overlap_and_far_separately():
    from event_engine.telegram import _fmt_mtf_zone_lines

    base = {
        "kind": "SUPPLY",
        "one_hour": {"poi": 100.0},
        "three_hour": {"poi": 100.2},
        "poi_distance_pct": 0.2,
    }
    assert _fmt_mtf_zone_lines({**base, "status": "confluent", "overlap": True}) == [
        "Zone 1H: SUPPLY 100",
        "Zone 3H: SUPPLY 100.2",
        "Calculation: OVERLAP · 0.2%",
    ]
    far = {**base, "status": "not_confluent", "overlap": False, "poi_distance_pct": 0.8}
    assert _fmt_mtf_zone_lines(far)[2] == "Calculation: FAR · 0.8%"
    tiny = {**base, "status": "confluent", "overlap": False, "poi_distance_pct": 0.000001}
    assert _fmt_mtf_zone_lines(tiny)[2] == "Calculation: NEAR · 0.000001%"


def test_telegram_zone_proximity_handles_unavailable_mtf_data():
    from event_engine.telegram import _fmt_mtf_zone_lines

    result = _fmt_mtf_zone_lines({
        "status": "no_3h_zone",
        "kind": "DEMAND",
        "one_hour": {"poi": 77300.0},
    })
    assert result == [
        "Zone 1H: DEMAND 77,300",
        "Zone 3H: —",
        "Calculation: —",
    ]


def test_telegram_long_header_uses_green_emoji_link():
    from event_engine.telegram import format_signal

    message = format_signal({"type": "LONG", "symbol": "BTC-USDT", "confirmation": {}})
    assert message.startswith('<a href="https://web.telegram.org/k/assets/img/emoji/1f7e2.png">🟢</a> LONG · <code>BTCUSDT</code>')


def test_telegram_price_formatting_preserves_exchange_precision_without_scientific_notation():
    from event_engine.telegram import _fmt_price, _fmt_price_grouped, format_signal

    assert _fmt_price(0.000123456789) == "0.000123456789"
    assert _fmt_price(77300.123456789) == "77300.123456789"
    assert _fmt_price_grouped(77300.123456789) == "77,300.123456789"
    assert _fmt_price(1.23456789e-7) == "0.000000123456789"

    message = format_signal({
        "type": "LONG", "symbol": "TINY-USDT", "entry": 0.000123456789,
        "sl": 0.000111111119, "tp1": 0.000127160493,
        "tp2": 0.000130864197, "tp3": 0.000134567876, "risk_pct": 10.0,
        "tp1_rr": 0.3, "tp2_rr": 0.6, "tp3_rr": 0.9,
        "zone": {"kind": "DEMAND"},
        "target": {"tp1_pct": 3.0, "tp2_pct": 6.0, "tp3_pct": 9.0, "tp1_close_fraction": 0.5, "tp2_close_fraction": 0.25, "tp3_close_fraction": 0.25},
        "confirmation": {},
    })
    assert "Entry reference: <code>0.000123456789</code>" in message
    assert "SL: <code>0.000111111119</code>" in message
    assert "TP1: <code>0.000127160493 (3% / 0.3R / 50%)</code>" in message
    assert "TP2: <code>0.000130864197 (6% / 0.6R / 25%)</code>" in message
    assert "TP3: <code>0.000134567876 (9% / 0.9R / 25%)</code>" in message


def test_telegram_never_displays_stale_or_unavailable_funding_as_a_rate():
    from event_engine.telegram import _fmt_funding

    assert _fmt_funding({"status": "stale", "funding_rate_pct": 0.02}) == "STALE"
    assert _fmt_funding({"status": "unavailable", "funding_rate_pct": 0.02}) == "UNAVAILABLE"
    assert _fmt_funding({"status": "ok", "funding_rate_pct": -0.01234}) == "-0.0123%"


def test_bingx_funding_snapshot_converts_decimal_rate_to_percent(monkeypatch):
    from event_engine import bingx

    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(
        bingx,
        "_research_public_get",
        lambda *args, **kwargs: {
            "code": 0,
            "data": [{"symbol": "BTC-USDT", "lastFundingRate": "0.0001", "time": now_ms, "nextFundingTime": now_ms + 3_600_000}],
        },
    )

    result = bingx.fetch_funding_snapshot("BTC-USDT")
    assert result["status"] == "ok"
    assert result["funding_rate"] == 0.0001
    assert result["funding_rate_pct"] == 0.01
    assert result["timestamp_source"] == "exchange"


def test_bingx_funding_snapshot_rejects_missing_exchange_timestamp(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(
        bingx,
        "_research_public_get",
        lambda *args, **kwargs: {
            "code": 0,
            "data": {"symbol": "BTC-USDT", "lastFundingRate": "0.0001"},
        },
    )
    result = bingx.fetch_funding_snapshot("BTC-USDT")
    assert result["status"] == "unverified"
    assert "funding_rate_pct" not in result


def test_bingx_funding_snapshot_rejects_stale_exchange_data(monkeypatch):
    from event_engine import bingx

    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(
        bingx,
        "_research_public_get",
        lambda *args, **kwargs: {
            "code": 0,
            "data": {"symbol": "BTC-USDT", "lastFundingRate": "0.0001", "time": now_ms - 3_600_000},
        },
    )
    result = bingx.fetch_funding_snapshot("BTC-USDT")
    assert result["status"] == "stale"
    assert "funding_rate_pct" not in result


def test_aggregate_complete_3h_bars_excludes_partial_and_gapped_buckets():
    from event_engine.signals import _aggregate_complete_3h_bars

    bars = _hourly_bars(78)
    # Remove one constituent from a bucket to prove it is not synthesized.
    bars = bars.loc[bars["timestamp"] != pd.Timestamp("2026-10-02T06:00:00Z")].reset_index(drop=True)
    boundary = pd.Timestamp("2026-10-04T06:00:00Z")
    aggregated = _aggregate_complete_3h_bars(bars, boundary)

    # 78 source candles could form 26 buckets, but bucket 06:00 is gapped and
    # the decision boundary excludes any 1h candle that has not yet closed.
    assert not (aggregated["timestamp"] == pd.Timestamp("2026-10-02T06:00:00Z")).any()
    assert (aggregated["timestamp"] + pd.Timedelta(hours=3) <= boundary).all()
    assert (aggregated["timestamp"].dt.hour % 3 == 0).all()


def test_3h_zone_confluence_uses_same_side_zones_and_is_informational(monkeypatch):
    from event_engine import signals

    start = pd.Timestamp("2026-10-01T00:00:00Z")
    bars = _hourly_bars(240, start)
    captured = {}

    def fake_zone_walk(df):
        captured["count"] = len(df)
        captured["last_timestamp"] = df["timestamp"].max()
        demand = [{"top": 100.5, "btm": 100.3, "poi": 100.4, "start": 20}]
        supply = [{"top": 120.0, "btm": 119.0, "poi": 119.5, "start": 10}]
        return supply, demand, [], [], []

    monkeypatch.setattr(signals, "_pine_zone_walk", fake_zone_walk)
    signal = {
        "type": "LONG",
        "decision_boundary_ts": (start + pd.Timedelta(hours=240)).isoformat(),
        "zone": {"kind": "DEMAND", "top": 100.1, "btm": 99.9, "poi": 100.0},
        "confirmation": {},
    }
    before = dict(signal)
    result = signals.add_3h_zone_confluence(signal, bars, max_distance_pct=0.5)

    assert result is signal
    assert captured["count"] == 80
    assert captured["last_timestamp"] == start + pd.Timedelta(hours=237)
    data = signal["confirmation"]["mtf_zone_confluence"]
    assert data["status"] == "confluent"
    assert data["kind"] == "DEMAND"
    assert data["poi_distance_pct"] == pytest.approx(0.4)
    # No trade direction, entry, target, stop, or score is changed by this metadata.
    assert signal["type"] == before["type"]
    assert signal["zone"] == before["zone"]


def test_3h_zone_confluence_uses_configured_three_percent_threshold(monkeypatch):
    from event_engine import signals

    start = pd.Timestamp("2026-10-01T00:00:00Z")
    bars = _hourly_bars(240, start)
    monkeypatch.setenv("MTF_ZONE_CONFLUENCE_MAX_DISTANCE_PCT", "3.0")
    monkeypatch.setattr(
        signals,
        "_pine_zone_walk",
        lambda df: ([], [{"top": 103.1, "btm": 102.9, "poi": 103.0, "start": 10}], [], [], []),
    )
    signal = {
        "type": "LONG",
        "decision_boundary_ts": (start + pd.Timedelta(hours=240)).isoformat(),
        "zone": {"kind": "DEMAND", "top": 100.1, "btm": 99.9, "poi": 100.0},
        "confirmation": {},
    }

    signals.add_3h_zone_confluence(signal, bars)
    data = signal["confirmation"]["mtf_zone_confluence"]
    assert data["poi_distance_pct"] == pytest.approx(3.0)
    assert data["max_distance_pct"] == pytest.approx(3.0)
    assert data["status"] == "confluent"


def test_3h_zone_confluence_does_not_match_opposite_side(monkeypatch):
    from event_engine import signals

    start = pd.Timestamp("2026-10-01T00:00:00Z")
    bars = _hourly_bars(240, start)
    monkeypatch.setattr(
        signals,
        "_pine_zone_walk",
        lambda df: ([{"top": 101.0, "btm": 99.0, "poi": 100.0, "start": 10}], [], [], [], []),
    )
    signal = {
        "type": "LONG",
        "decision_boundary_ts": (start + pd.Timedelta(hours=240)).isoformat(),
        "zone": {"kind": "DEMAND", "top": 101.0, "btm": 99.0, "poi": 100.0},
        "confirmation": {},
    }
    signals.add_3h_zone_confluence(signal, bars)
    assert signal["confirmation"]["mtf_zone_confluence"]["status"] == "no_3h_zone"


def _isolated_research_paths(research, tmp_path: Path, monkeypatch) -> None:
    for name in ["MARKET_BARS_5M_PATH", "RESEARCH_BAR_CURSORS_PATH", "RESEARCH_MANIFEST_PATH", "RESEARCH_ERRORS_PATH"]:
        monkeypatch.setattr(research, name, tmp_path / getattr(research, name).name)
    monkeypatch.setattr(research, "_PERSISTENCE_BATCH", None)
    monkeypatch.setattr(research, "_MARKET_BAR_INDEX_CACHE", {})


def test_research_history_seed_backfills_older_bars_behind_existing_cursor(tmp_path, monkeypatch):
    from event_engine import research

    _isolated_research_paths(research, tmp_path, monkeypatch)
    now = pd.Timestamp.now(tz="UTC").floor("5min")
    rows = []
    for index in range(12):
        ts = now - pd.Timedelta(minutes=(11 - index) * 5 + 5)
        rows.append({
            "timestamp": ts.isoformat(), "open": 100.0, "high": 102.0,
            "low": 99.0, "close": 101.0, "volume": 10.0,
            "close_time": int((ts + pd.Timedelta(minutes=5)).timestamp() * 1000),
        })
    key = "AAA-USDT|5m|binance"
    cursors_path = research.RESEARCH_BAR_CURSORS_PATH
    cursors_path.write_text(json.dumps({key: {
        "last_timestamp": rows[-1]["timestamp"], "last_bar_id": "ALREADY_AT_TAIL",
        "updated_at": "prior", "scan_id": "prior",
    }}), encoding="utf-8")

    with research.batch_persistence() as batch:
        queued = research.persist_market_bars(
            "AAA-USDT", "5m", rows, provider="binance", source="test",
            scan_id="SEED_1", seed_history=True,
        )
        assert queued == len(rows)
        # Cursor seed marker is not durable until the bar batch itself flushes.
        on_disk = json.loads(cursors_path.read_text())
        assert on_disk[key].get("history_seeded") is not True

    saved = [json.loads(line) for line in research.MARKET_BARS_5M_PATH.read_text().splitlines()]
    cursor = json.loads(cursors_path.read_text())[key]
    assert len(saved) == len(rows)
    assert cursor["last_timestamp"] == rows[-1]["timestamp"]
    assert cursor["history_seeded"] is True
    assert cursor["history_seeded_count"] == len(rows)
    assert research.market_bar_history_seeded("AAA-USDT") is True
    assert batch["flush_result"]["persistence_errors"] == 0


def test_research_history_seed_is_idempotent(tmp_path, monkeypatch):
    from event_engine import research

    _isolated_research_paths(research, tmp_path, monkeypatch)
    now = pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(minutes=5)
    rows = [{
        "timestamp": (now - pd.Timedelta(minutes=5 * offset)).isoformat(),
        "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0,
        "volume": 10.0,
        "close_time": int((now - pd.Timedelta(minutes=5 * offset) + pd.Timedelta(minutes=5)).timestamp() * 1000),
    } for offset in reversed(range(8))]

    first = research.persist_market_bars("AAA-USDT", "5m", rows, provider="binance", source="test", scan_id="S1", seed_history=True)
    second = research.persist_market_bars("AAA-USDT", "5m", rows, provider="binance", source="test", scan_id="S2", seed_history=True)
    saved = [json.loads(line) for line in research.MARKET_BARS_5M_PATH.read_text().splitlines()]
    assert first == len(rows)
    assert second == 0
    assert len(saved) == len(rows)
    assert research.market_bar_history_seeded("AAA-USDT") is True


def test_research_history_seed_marker_is_not_committed_when_bar_append_fails(tmp_path, monkeypatch):
    from event_engine import research

    _isolated_research_paths(research, tmp_path, monkeypatch)
    now = pd.Timestamp.now(tz="UTC").floor("5min") - pd.Timedelta(minutes=5)
    bar = {
        "timestamp": now.isoformat(), "open": 100.0, "high": 102.0,
        "low": 99.0, "close": 101.0, "volume": 10.0,
        "close_time": int((now + pd.Timedelta(minutes=5)).timestamp() * 1000),
    }
    monkeypatch.setattr(research, "_append_market_bars_idempotent_locked", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failed")))
    with research.batch_persistence():
        research.persist_market_bars("AAA-USDT", "5m", [bar], provider="binance", source="test", scan_id="S1", seed_history=True)
    assert research.market_bar_history_seeded("AAA-USDT") is False


def test_deferred_prefetch_skips_a_symbol_after_history_seed(tmp_path, monkeypatch):
    import run_once

    monkeypatch.setattr(run_once.research, "_load_cursors", lambda: {})
    monkeypatch.setattr(run_once.research, "market_bar_history_seeded", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        run_once, "_fetch_missing_research_5m_bars",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("seeded symbol must not fetch 300 bars again")),
    )
    payload = {"symbol": "AAA-USDT", "provider": "binance", "bars_5m": [], "persist_bars_for_forward": True}
    assert run_once._prefetch_missing_research_5m_bars([payload], {}, limit=300, max_workers=4) == {}


def test_3h_zone_confluence_reuses_snapshot_within_symbol_scan(monkeypatch):
    from event_engine import signals

    start = pd.Timestamp("2026-10-01T00:00:00Z")
    bars = _hourly_bars(240, start)
    walk_calls = []

    def fake_zone_walk(df):
        walk_calls.append(len(df))
        return ([], [{"top": 100.5, "btm": 100.3, "poi": 100.4, "start": 20}], [], [], [])

    monkeypatch.setattr(signals, "_pine_zone_walk", fake_zone_walk)
    cache = {}
    boundary = (start + pd.Timedelta(hours=240)).isoformat()
    for index, poi in enumerate((100.0, 100.1)):
        decision_boundary = (
            pd.Timestamp(boundary) + pd.Timedelta(minutes=5 * index)
        ).isoformat()
        signal = {
            "type": "LONG", "decision_boundary_ts": decision_boundary,
            "zone": {"kind": "DEMAND", "top": poi + 0.1, "btm": poi - 0.1, "poi": poi},
            "confirmation": {},
        }
        signals.add_3h_zone_confluence(signal, bars, snapshot_cache=cache)
        assert signal["confirmation"]["mtf_zone_confluence"]["status"] == "confluent"

    assert walk_calls == [80]
    assert len(cache) == 1
