from __future__ import annotations

import pytest

from event_engine import binance
import run_once


def test_binance_book_ticker_returns_local_observation_window_without_claiming_exchange_time(monkeypatch):
    ticks = iter([1_000, 1_120])
    monkeypatch.setattr(binance.time, "time", lambda: next(ticks) / 1000.0)
    captured = {}

    def fake_get(path, params, *, timeout, retryable):
        captured.update(path=path, params=params, timeout=timeout, retryable=retryable)
        return {
            "symbol": "BTCUSDT",
            "bidPrice": "100.0",
            "bidQty": "2.0",
            "askPrice": "100.2",
            "askQty": "3.0",
        }

    monkeypatch.setattr(binance, "_get", fake_get)
    quote = binance.fetch_book_ticker("BTC-USDT", timeout=0.4, retryable=False)

    assert captured == {
        "path": binance.BOOK_TICKER_PATH,
        "params": {"symbol": "BTCUSDT"},
        "timeout": 0.4,
        "retryable": False,
    }
    assert quote["mid"] == pytest.approx(100.1)
    assert quote["bid_qty"] == pytest.approx(2.0)
    assert quote["ask_qty"] == pytest.approx(3.0)
    assert quote["observed_at_ms"] == 1_060
    assert quote["request_duration_ms"] == 120
    assert quote["timestamp_quality"] == "LOCAL_OBSERVATION_MIDPOINT_ONLY"
    assert quote["exchange_timestamp_available"] is False


def test_binance_book_ticker_rejects_invalid_or_cross_symbol_quote(monkeypatch):
    monkeypatch.setattr(binance, "_get", lambda *_args, **_kwargs: {
        "symbol": "ETHUSDT", "bidPrice": "100", "askPrice": "99",
        "bidQty": "1", "askQty": "1",
    })
    assert binance.fetch_book_ticker("BTCUSDT") is None

    monkeypatch.setattr(binance, "_get", lambda *_args, **_kwargs: {
        "symbol": "BTCUSDT", "bidPrice": "NaN", "askPrice": "100",
        "bidQty": "1", "askQty": "1",
    })
    assert binance.fetch_book_ticker("BTCUSDT") is None


def test_directional_basis_shadow_uses_executable_side_and_is_not_a_gate():
    binance_quote = {
        "bid": 99.9, "ask": 100.1, "observed_at_ms": 10_000,
        "request_duration_ms": 80, "timestamp_quality": "LOCAL_OBSERVATION_MIDPOINT_ONLY",
        "exchange_timestamp_available": False,
    }
    bingx_quote = {
        "bid": 100.2, "ask": 100.4, "quote_observed_at_ms": 10_250,
        "quote_source": "bookTicker", "quote_exchange_time_ms": None,
    }

    long_basis = run_once._build_cross_venue_basis_shadow("LONG", binance_quote, bingx_quote)
    short_basis = run_once._build_cross_venue_basis_shadow("SHORT", binance_quote, bingx_quote)

    assert long_basis["status"] == short_basis["status"] == "PRESENT"
    assert long_basis["basis_pct"] == pytest.approx((100.4 - 100.0) / 100.0 * 100.0)
    assert short_basis["basis_pct"] == pytest.approx((100.2 - 100.0) / 100.0 * 100.0)
    assert long_basis["directional_adverse_basis_pct"] > 0
    assert short_basis["directional_adverse_basis_pct"] < 0
    assert long_basis["observation_skew_ms"] == 250
    assert long_basis["basis_bucket"] == "DIRECTIONALLY_ADVERSE_GT_0_20"
    assert long_basis["binance_quote_time_ms"] == 10_000
    assert long_basis["bingx_quote_time_ms"] == 10_250
    assert long_basis["timestamp_quality"] == "LOCAL_OBSERVATION_WINDOW_ONLY"
    assert long_basis["exchange_timestamps_verified"] is False
    assert long_basis["shadow_only"] is True
    assert long_basis["production_gate_applied"] is False
    assert long_basis["score_changed"] is False
    assert long_basis["selection_order_changed"] is False


def test_basis_shadow_marks_unsynchronized_quotes_without_exposing_usable_basis():
    result = run_once._build_cross_venue_basis_shadow(
        "LONG",
        {"bid": 99.9, "ask": 100.1, "observed_at_ms": 1_000},
        {"bid": 100.0, "ask": 100.2, "quote_observed_at_ms": 3_000},
        max_observation_skew_ms=500,
    )
    assert result["status"] == "UNSYNCHRONIZED"
    assert result["basis_usable"] is False
    assert result["basis_pct"] is None
    assert result["basis_pct_candidate"] is not None  # Diagnostic candidate only; not a synchronized result.
    assert result["production_gate_applied"] is False


def test_basis_shadow_is_explicitly_unavailable_when_quotes_are_missing():
    result = run_once._build_cross_venue_basis_shadow("SHORT", None, None)
    assert result["status"] == "QUOTE_UNAVAILABLE"
    assert result["production_gate_applied"] is False
    assert result["shadow_only"] is True


def test_pre_entry_telemetry_marks_only_synchronized_basis_as_available(tmp_path, monkeypatch):
    from event_engine import telemetry

    monkeypatch.setattr(telemetry, "PRE_ENTRY_FEATURES_PATH", tmp_path / "pre_entry_features.jsonl")
    signal = {
        "cross_venue_basis": {
            "status": "PRESENT", "basis_usable": True, "basis_pct": 0.3,
            "binance_quote_time_ms": 10_000, "bingx_quote_time_ms": 10_200,
            "timestamp_quality": "LOCAL_OBSERVATION_WINDOW_ONLY",
            "exchange_timestamps_verified": False,
            "collection_mode": "OPTIONAL_BINANCE_BOOK_TICKER_REQUEST",
        }
    }
    telemetry.record_pre_entry_feature_snapshot(
        event_id="EVT_BASIS_PRESENT", attempt_id="ATT_BASIS_PRESENT",
        symbol="AAA-USDT", direction="LONG", signal=signal, quote={},
    )
    row = __import__("json").loads((tmp_path / "pre_entry_features.jsonl").read_text().splitlines()[0])
    assert row["feature_availability"]["cross_venue_basis"] == "PRESENT"
    assert row["supplemental_context"]["cross_venue_basis"]["exchange_timestamps_verified"] is False

    signal["cross_venue_basis"] = {
        "status": "UNSYNCHRONIZED", "basis_usable": False,
        "basis_pct": None, "basis_pct_candidate": 0.3,
        "binance_quote_time_ms": 10_000, "bingx_quote_time_ms": 20_000,
    }
    telemetry.record_pre_entry_feature_snapshot(
        event_id="EVT_BASIS_UNSYNC", attempt_id="ATT_BASIS_UNSYNC",
        symbol="AAA-USDT", direction="LONG", signal=signal, quote={},
    )
    rows = [__import__("json").loads(line) for line in (tmp_path / "pre_entry_features.jsonl").read_text().splitlines()]
    assert rows[1]["feature_availability"]["cross_venue_basis"] == "NOT_CAPTURED"


def test_directional_basis_favorable_and_neutral_buckets_are_correct_for_both_sides():
    binance_quote = {"bid": 99.9, "ask": 100.1, "observed_at_ms": 5_000}
    # BingX LONG ask below Binance mid is directionally favourable for a long.
    long_favourable = run_once._build_cross_venue_basis_shadow(
        "LONG", binance_quote,
        {"bid": 99.5, "ask": 99.7, "quote_observed_at_ms": 5_100},
    )
    assert long_favourable["basis_bucket"] == "DIRECTIONALLY_FAVORABLE_GT_0_20"
    assert long_favourable["directional_adverse_basis_pct"] < -0.20

    # For SHORT, a BingX bid above Binance mid is directionally favourable.
    short_favourable = run_once._build_cross_venue_basis_shadow(
        "SHORT", binance_quote,
        {"bid": 100.3, "ask": 100.5, "quote_observed_at_ms": 5_100},
    )
    assert short_favourable["basis_bucket"] == "DIRECTIONALLY_FAVORABLE_GT_0_20"
    assert short_favourable["directional_adverse_basis_pct"] < -0.20

    neutral = run_once._build_cross_venue_basis_shadow(
        "LONG", binance_quote,
        {"bid": 99.98, "ask": 100.02, "quote_observed_at_ms": 5_100},
    )
    assert neutral["basis_bucket"] == "NEUTRAL_WITHIN_PLUS_MINUS_0_10"


def test_basis_shadow_uses_mapped_spot_symbol_only_for_binance_crypto_candidates():
    assert run_once._cross_venue_basis_symbol({
        "market_snapshot": {
            "analysis_provider": "binance",
            "asset_class": "CRYPTO",
            "binance_symbol": "XBTUSDT",
        }
    }) == "XBTUSDT"
    assert run_once._cross_venue_basis_symbol({
        "market_snapshot": {
            "analysis_provider": "binance",
            "asset_class": "EQUITY",
            "binance_symbol": "ABCUSDT",
        }
    }) is None
    assert run_once._cross_venue_basis_symbol({
        "market_snapshot": {
            "analysis_provider": "bingx",
            "asset_class": "CRYPTO",
            "binance_symbol": "AAAUSDT",
        }
    }) is None


def test_cross_venue_basis_shadow_collection_is_disabled_by_default():
    # The optional Binance request must not add latency to ordinary execution
    # unless the operator explicitly enables the Shadow observation.
    assert run_once.CROSS_VENUE_BASIS_SHADOW_ENABLED is False
