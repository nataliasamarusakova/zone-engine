"""Append-only execution telemetry for the trading engine.

This module is deliberately dependency-light and has no trading logic.  Telemetry
writes are best-effort and must never change a trading decision.  Each journal is
an immutable JSONL event stream with a stable schema version.
"""
from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import re
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from event_engine.version import ENGINE_VERSION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
SCHEMA_VERSION = 1

QUOTE_SNAPSHOTS_PATH = DATA / "quote_snapshots.jsonl"
ORDER_LIFECYCLE_PATH = DATA / "order_lifecycle.jsonl"
PROTECTION_LIFECYCLE_PATH = DATA / "protection_lifecycle.jsonl"
POSITION_RECONCILIATION_PATH = DATA / "position_reconciliation.jsonl"
EXCHANGE_ERRORS_PATH = DATA / "exchange_errors.jsonl"
PRE_ENTRY_FEATURES_PATH = DATA / "pre_entry_features.jsonl"
EXECUTION_QUALITY_PATH = DATA / "execution_quality.jsonl"
EMERGENCY_EXITS_PATH = DATA / "emergency_exits.jsonl"

_WRITE_LOCK = threading.Lock()
_TELEMETRY_FAILURE_COUNT = 0
_TELEMETRY_LAST_FAILURE_TS_MS: int | None = None
_TELEMETRY_LAST_FAILURE_TYPE: str | None = None
_TELEMETRY_FAILURE_LOG_EVERY = max(1, int(os.environ.get("TELEMETRY_FAILURE_LOG_EVERY", "100")))
_LOG = logging.getLogger("event_engine.telemetry")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=str) + "\n")
                fh.flush()
                if os.environ.get("TELEMETRY_FSYNC", "false").lower() == "true":
                    os.fsync(fh.fileno())
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def emit(path: Path, record_type: str, *, event_id: str | None = None,
         attempt_id: str | None = None, position_id: str | None = None,
         order_id: str | None = None, **payload: Any) -> None:
    """Write one telemetry event; swallow telemetry-only failures."""
    now_ms = int(time.time() * 1000)
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "record_type": record_type,
        "ts_ms": now_ms,
        "ts": datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).isoformat(),
    }
    for key, value in (
        ("event_id", event_id),
        ("attempt_id", attempt_id),
        ("position_id", position_id),
        ("order_id", order_id),
    ):
        if value not in (None, ""):
            record[key] = str(value)
    record.update(payload)
    try:
        with _WRITE_LOCK:
            _append_jsonl(path, record)
    except Exception as exc:
        # Never let observability failures influence execution, but do not make
        # them completely silent: expose a process-local health counter and a
        # rate-limited warning for operators.
        global _TELEMETRY_FAILURE_COUNT, _TELEMETRY_LAST_FAILURE_TS_MS, _TELEMETRY_LAST_FAILURE_TYPE
        _TELEMETRY_FAILURE_COUNT += 1
        _TELEMETRY_LAST_FAILURE_TS_MS = int(time.time() * 1000)
        _TELEMETRY_LAST_FAILURE_TYPE = type(exc).__name__
        if _TELEMETRY_FAILURE_COUNT == 1 or _TELEMETRY_FAILURE_COUNT % _TELEMETRY_FAILURE_LOG_EVERY == 0:
            _LOG.warning(
                "Telemetry write failed count=%d record_type=%s path=%s error=%s",
                _TELEMETRY_FAILURE_COUNT, record_type, path.name, exc,
            )
        return


def health_snapshot() -> dict[str, Any]:
    """Return lightweight process-local telemetry health state.

    This function performs no I/O and therefore remains available even when
    telemetry storage itself is unavailable.
    """
    return {
        "write_failure_count": int(_TELEMETRY_FAILURE_COUNT),
        "last_failure_ts_ms": _TELEMETRY_LAST_FAILURE_TS_MS,
        "last_failure_type": _TELEMETRY_LAST_FAILURE_TYPE,
    }


def record_quote_snapshot(*, event_id: str | None, attempt_id: str | None,
                          symbol: str, direction: str | None,
                          quote: dict[str, Any],
                          signal_price: float | None = None,
                          stage: str = "PRE_ENTRY_QUOTE") -> None:
    emit(
        QUOTE_SNAPSHOTS_PATH,
        "QUOTE_SNAPSHOT",
        event_id=event_id,
        attempt_id=attempt_id,
        symbol=symbol,
        direction=direction,
        stage=stage,
        bid=quote.get("bid"),
        ask=quote.get("ask"),
        last_price=quote.get("last_price"),
        spread_pct=quote.get("spread_pct"),
        quote_source=quote.get("quote_source"),
        quote_sources_attempted=quote.get("quote_sources_attempted"),
        quote_fallback_reason=quote.get("quote_fallback_reason"),
        quote_time=quote.get("time"),
        quote_exchange_time_ms=quote.get("quote_exchange_time_ms"),
        quote_observed_at_ms=quote.get("quote_observed_at_ms"),
        quote_local_age_sec=quote.get("quote_local_age_sec"),
        quote_exchange_age_sec=quote.get("quote_exchange_age_sec"),
        quote_local_age_sec_at_post=quote.get("quote_local_age_sec_at_post"),
        quote_exchange_age_sec_at_post=quote.get("quote_exchange_age_sec_at_post"),
        quote_freshness_source=quote.get("quote_freshness_source"),
        quote_freshness_status=quote.get("quote_freshness_status"),
        quote_unknown_age_policy=quote.get("quote_unknown_age_policy"),
        execution_reference_quality=quote.get("execution_reference_quality"),
        quote_freshness_source_at_post=quote.get("quote_freshness_source_at_post"),
        signal_price=signal_price,
        execution_reference_price=quote.get("execution_reference_price"),
        signal_drift_pct=quote.get("signal_drift_pct"),
        order_submit_at_ms=quote.get("order_submit_at_ms"),
    )


def record_order_event(*, event_id: str | None, attempt_id: str | None,
                       order_id: str | None, symbol: str,
                       direction: str | None, leg: str, status: str,
                       **payload: Any) -> None:
    emit(
        ORDER_LIFECYCLE_PATH,
        "ORDER_EVENT",
        event_id=event_id,
        attempt_id=attempt_id,
        order_id=order_id,
        symbol=symbol,
        direction=direction,
        leg=leg,
        status=status,
        **payload,
    )


def record_protection_event(*, event_id: str | None, attempt_id: str | None,
                            position_id: str | None, order_id: str | None,
                            symbol: str, direction: str, leg: str,
                            status: str, **payload: Any) -> None:
    emit(
        PROTECTION_LIFECYCLE_PATH,
        "PROTECTION_EVENT",
        event_id=event_id,
        attempt_id=attempt_id,
        position_id=position_id,
        order_id=order_id,
        symbol=symbol,
        direction=direction,
        leg=leg,
        status=status,
        **payload,
    )


def record_position_reconciliation(*, event_id: str | None,
                                   attempt_id: str | None,
                                   position_id: str | None,
                                   symbol: str, direction: str,
                                   status: str, **payload: Any) -> None:
    emit(
        POSITION_RECONCILIATION_PATH,
        "POSITION_RECON",
        event_id=event_id,
        attempt_id=attempt_id,
        position_id=position_id,
        symbol=symbol,
        direction=direction,
        reconciliation_status=status,
        **payload,
    )


def parse_exchange_error(value: Any, *, default_code: Any = None) -> tuple[Any, str]:
    """Normalize an exchange/API error object or message for telemetry."""
    code = default_code
    if isinstance(value, dict):
        code = value.get("code", code)
        message = value.get("msg") or value.get("message") or value.get("error") or str(value)
    else:
        message = str(value)
    if code in (None, "", 0, "0"):
        match = re.search(r"\bcode[=:]\s*([A-Za-z0-9_-]+)", message, flags=re.IGNORECASE)
        if match:
            code = match.group(1)
    return code, message


def record_state_conflict(*, event_id: str | None, attempt_id: str | None,
                         position_id: str | None, order_id: str | None,
                         symbol: str, direction: str,
                         conflict_type: str, message: str,
                         owner_event_ids: list[str] | None = None, **payload: Any) -> None:
    """Record an internal ownership/state-integrity conflict.

    This is deliberately separate from exchange errors: the exchange may have
    behaved correctly while the local state model linked the object incorrectly.
    """
    emit(
        POSITION_RECONCILIATION_PATH,
        "STATE_CONFLICT",
        event_id=event_id,
        attempt_id=attempt_id,
        position_id=position_id,
        order_id=order_id,
        symbol=symbol,
        direction=direction,
        reconciliation_status="STATE_CONFLICT",
        conflict_type=conflict_type,
        message=message,
        owner_event_ids=owner_event_ids or [],
        **payload,
    )


def record_exchange_error(*, event_id: str | None, attempt_id: str | None,
                          position_id: str | None, order_id: str | None,
                          symbol: str | None, endpoint: str,
                          method: str, error_code: Any,
                          message: str, **payload: Any) -> None:
    emit(
        EXCHANGE_ERRORS_PATH,
        "EXCHANGE_ERROR",
        event_id=event_id,
        attempt_id=attempt_id,
        position_id=position_id,
        order_id=order_id,
        symbol=symbol,
        endpoint=endpoint,
        method=method,
        error_code=error_code,
        message=str(message),
        **payload,
    )

def _snapshot_safe(value: Any, *, depth: int = 0) -> Any:
    """Bound and sanitize feature snapshots before they reach JSONL storage."""
    if depth > 8:
        return "<max_depth>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    # Support NumPy scalar-like values without importing NumPy into telemetry.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = value
        if scalar is not value:
            return _snapshot_safe(scalar, depth=depth + 1)
    if isinstance(value, dict):
        return {
            str(key)[:120]: _snapshot_safe(item_value, depth=depth + 1)
            for index, (key, item_value) in enumerate(value.items())
            if index < 200
        }
    if isinstance(value, (list, tuple, set)):
        return [_snapshot_safe(item_value, depth=depth + 1) for item_value in list(value)[:200]]
    # Datetimes and other scalar metadata are represented, never evaluated.
    return str(value)[:500]


def record_pre_entry_feature_snapshot(*, event_id: str | None, attempt_id: str | None,
                                      symbol: str, direction: str,
                                      signal: dict[str, Any], quote: dict[str, Any],
                                      code_commit_sha: str | None = None) -> None:
    """Persist already-available decision features immediately before the final freshness check.

    This function performs local serialization only: it must not fetch market data,
    sleep, or make any trading decision. Optional features remain explicitly absent
    when their source was not collected for this execution.
    """
    feature_keys = (
        "strategy", "strategy_version", "trigger_timeframe", "time",
        "trigger_bar_time", "trigger_bar_close_time", "decision_ts",
        "decision_boundary_ts", "decision_boundary_kind", "decision_boundary_source",
        "observation_ts", "execution_gate_ts", "execution_started_ts",
        "execution_age_bars", "execution_age_minutes", "execution_age_minutes_at_order_ready",
        "trigger_to_execution_gate_minutes", "signal_drift_pct",
        "idx", "latest_closed_idx", "latest_closed_time",
        "entry", "sl", "tp1", "tp2", "risk_pct", "risk_abs", "atr",
        "tp1_rr", "tp2_rr", "entry_bar", "previous_bar", "trigger",
        "zone", "zone_counts", "zone_midpoint", "zone_width_abs", "zone_width_pct_of_entry",
        "target", "risk_model", "confirmation", "score", "volume_ratio_1h20",
        "signal_forensics", "zone_visit", "source_bar_close",
        "market_snapshot", "trend_filter", "shadow_experiments", "pre_entry_market_features",
        "pre_execution_market_context", "btc_context", "vwap_context",
        "cross_venue_basis", "portfolio_context", "open_positions_snapshot",
        "margin_snapshot",
    )
    features = {key: _snapshot_safe(signal.get(key)) for key in feature_keys if key in signal}
    context = signal.get("pre_execution_market_context")
    context = context if isinstance(context, dict) else {}
    order_book = context.get("order_book") if isinstance(context.get("order_book"), dict) else {}
    premium_available = context.get("premium_index_ts") is not None or context.get("funding_rate") is not None
    confirmation = signal.get("confirmation") if isinstance(signal.get("confirmation"), dict) else {}
    mtf_context = confirmation.get("mtf_zone_confluence")
    trend = signal.get("trend_filter") if isinstance(signal.get("trend_filter"), dict) else {}
    btc_context = signal.get("btc_context")
    if not isinstance(btc_context, dict):
        btc_fields = (
            "btc_regime", "btc_bar_close_ts", "btc_chg_1h_pct",
            "btc_chg_4h_pct", "btc_bars", "btc_context_veto",
        )
        btc_context = {key: trend[key] for key in btc_fields if key in trend}
    market_snapshot = signal.get("market_snapshot") if isinstance(signal.get("market_snapshot"), dict) else {}
    cross_venue_basis = signal.get("cross_venue_basis")
    if not isinstance(cross_venue_basis, dict):
        cross_venue_basis = None
    deviation_fields = (
        "binance_live_price", "bingx_live_price", "market_spread_pct",
        "venue_price_deviation_pct", "market_spread_metric", "scan_time",
    )
    cross_venue_price_deviation = {
        key: market_snapshot[key] for key in deviation_fields if key in market_snapshot
    }
    portfolio_context = signal.get("portfolio_context")
    if not isinstance(portfolio_context, dict):
        portfolio_context = context.get("portfolio_context") if isinstance(context.get("portfolio_context"), dict) else None
    open_positions_snapshot = signal.get("open_positions_snapshot")
    margin_snapshot = signal.get("margin_snapshot")
    if margin_snapshot is None:
        margin_snapshot = context.get("margin_snapshot")
    if open_positions_snapshot is None:
        open_positions_snapshot = context.get("open_positions_snapshot")
    vwap_context = signal.get("vwap_context")
    if not isinstance(vwap_context, dict):
        vwap_fields = ("session_vwap_approx", "session_vwap_distance_pct", "vwap", "vwap_distance_pct")
        vwap_context = {key: signal[key] for key in vwap_fields if key in signal}
    btc_available = isinstance(btc_context, dict) and any(
        value is not None for key, value in btc_context.items()
        if key not in {"status", "source", "as_of_decision_ts"}
    )
    vwap_available = isinstance(vwap_context, dict) and any(
        vwap_context.get(key) is not None
        for key in ("session_vwap_approx", "session_vwap_distance_pct", "vwap", "vwap_distance_pct")
    )
    basis_available = (
        isinstance(cross_venue_basis, dict)
        and cross_venue_basis.get("status") == "PRESENT"
        and cross_venue_basis.get("basis_usable") is True
        and cross_venue_basis.get("basis_pct") is not None
        and cross_venue_basis.get("binance_quote_time_ms") is not None
        and cross_venue_basis.get("bingx_quote_time_ms") is not None
    )
    cross_venue_deviation_available = any(
        market_snapshot.get(key) is not None
        for key in ("binance_live_price", "bingx_live_price", "market_spread_pct", "venue_price_deviation_pct")
    )
    margin_available = isinstance(margin_snapshot, dict) and any(
        margin_snapshot.get(key) is not None
        for key in ("equity", "balance", "available_margin", "used_margin", "freezed_margin")
    )
    availability = {
        "signal_forensics": "PRESENT" if isinstance(signal.get("signal_forensics"), dict) else "NOT_CAPTURED",
        "zone_geometry": "PRESENT" if isinstance(signal.get("zone"), dict) else "NOT_CAPTURED",
        "trend_filter_context": "PRESENT" if isinstance(signal.get("trend_filter"), dict) else "NOT_CAPTURED",
        "market_snapshot": "PRESENT" if isinstance(signal.get("market_snapshot"), dict) else "NOT_CAPTURED",
        "signal_score": "PRESENT" if signal.get("score") is not None else "NOT_CAPTURED",
        "volume_ratio_1h20": "PRESENT" if signal.get("volume_ratio_1h20") is not None else "NOT_CAPTURED",
        "confirmation_features": "PRESENT" if isinstance(signal.get("confirmation"), dict) else "NOT_CAPTURED",
        "mtf_zone_confluence": str(mtf_context.get("status") or "PRESENT") if isinstance(mtf_context, dict) else "NOT_CAPTURED",
        "btc_context": "PRESENT" if btc_available else "NOT_CAPTURED",
        "vwap_context": "PRESENT" if vwap_available else "NOT_CAPTURED",
        "cross_venue_basis": "PRESENT" if basis_available else "NOT_CAPTURED",
        "cross_venue_price_deviation": "PRESENT" if cross_venue_deviation_available else "NOT_CAPTURED",
        "portfolio_context": "PRESENT" if portfolio_context else "NOT_CAPTURED",
        "open_positions_snapshot": "PRESENT" if open_positions_snapshot is not None else "NOT_CAPTURED",
        "margin_snapshot": "PRESENT" if margin_available else "NOT_CAPTURED",
        "pre_execution_context": str(context.get("status") or "NOT_CAPTURED"),
        "funding_context": "PRESENT" if premium_available else "NOT_CAPTURED",
        "order_book_depth": str(order_book.get("status") or "NOT_CAPTURED"),
        "recent_trades": str((context.get("recent_trades") or {}).get("status") or "NOT_CAPTURED")
            if isinstance(context.get("recent_trades"), dict) else "NOT_CAPTURED",
    }
    captured_at_ms = int(time.time() * 1000)
    basis_collection_attempted = (
        isinstance(cross_venue_basis, dict)
        and cross_venue_basis.get("collection_mode") == "OPTIONAL_BINANCE_BOOK_TICKER_REQUEST"
    )
    feature_collection_mode = (
        "LOCAL_SNAPSHOT_PLUS_OPTIONAL_SHADOW_BINANCE_BOOK_TICKER"
        if basis_collection_attempted
        else "LOCAL_SNAPSHOT_NO_ADDITIONAL_MARKET_REQUESTS"
    )
    emit(
        PRE_ENTRY_FEATURES_PATH, "PRE_ENTRY_FEATURE_SNAPSHOT",
        event_id=event_id, attempt_id=attempt_id,
        symbol=symbol, direction=direction,
        capture_phase="IMMEDIATELY_BEFORE_FINAL_FRESHNESS_CHECK",
        captured_at_ms=captured_at_ms,
        engine_version=ENGINE_VERSION, code_commit_sha=code_commit_sha,
        signal_features=features,
        supplemental_context=_snapshot_safe({
            "btc_context": btc_context or None,
            "vwap_context": vwap_context or None,
            "cross_venue_basis": cross_venue_basis,
            "cross_venue_price_deviation": cross_venue_price_deviation or None,
            "portfolio_context": portfolio_context,
            "open_positions_snapshot": open_positions_snapshot,
            "margin_snapshot": margin_snapshot,
        }),
        feature_availability=availability,
        quote=_snapshot_safe({
            key: quote.get(key) for key in (
                "bid", "ask", "last_price", "spread_pct", "quote_source",
                "quote_sources_attempted", "quote_fallback_reason", "time",
                "quote_exchange_time_ms", "quote_observed_at_ms",
                "quote_local_age_sec", "quote_exchange_age_sec",
                "quote_freshness_source", "quote_freshness_status",
                "execution_reference_price", "execution_reference_quality",
            ) if key in quote
        }),
        feature_collection_mode=feature_collection_mode,
    )


def record_execution_quality_snapshot(*, event_id: str | None, attempt_id: str | None,
                                      symbol: str, direction: str, stage: str,
                                      quote: dict[str, Any], signal_price: float | None = None,
                                      execution_reference_price: float | None = None,
                                      fill_price: float | None = None,
                                      signed_slippage_pct: float | None = None,
                                      adverse_slippage_pct: float | None = None,
                                      market_context: dict[str, Any] | None = None) -> None:
    """Append a linked pre/post execution quality observation without affecting orders."""
    context = market_context if isinstance(market_context, dict) else {}
    order_book = context.get("order_book") if isinstance(context.get("order_book"), dict) else {}
    emit(
        EXECUTION_QUALITY_PATH, "EXECUTION_QUALITY_SNAPSHOT",
        event_id=event_id, attempt_id=attempt_id, symbol=symbol, direction=direction,
        stage=str(stage).upper(), signal_price=_snapshot_safe(signal_price),
        bid=_snapshot_safe(quote.get("bid")), ask=_snapshot_safe(quote.get("ask")),
        bid_ask_spread_pct=_snapshot_safe(quote.get("spread_pct")),
        quote_source=quote.get("quote_source"),
        quote_exchange_time_ms=_snapshot_safe(quote.get("quote_exchange_time_ms")),
        quote_observed_at_ms=_snapshot_safe(quote.get("quote_observed_at_ms")),
        quote_local_age_sec=_snapshot_safe(quote.get("quote_local_age_sec")),
        quote_exchange_age_sec=_snapshot_safe(quote.get("quote_exchange_age_sec")),
        quote_freshness_status=quote.get("quote_freshness_status"),
        execution_reference_price=_snapshot_safe(execution_reference_price),
        actual_fill_price=_snapshot_safe(fill_price),
        signed_slippage_pct=_snapshot_safe(signed_slippage_pct),
        adverse_slippage_pct=_snapshot_safe(adverse_slippage_pct),
        depth_metrics=_snapshot_safe(order_book) if order_book else None,
        depth_availability=str(order_book.get("status") or "NOT_CAPTURED"),
        context_capture_status=context.get("status", "NOT_CAPTURED"),
    )


def _emergency_attempt_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    attempts = result.get("attempts") if isinstance(result.get("attempts"), list) else []
    for item in attempts[:25]:
        if not isinstance(item, dict):
            summaries.append({"kind": type(item).__name__})
            continue
        response = item.get("response") if isinstance(item.get("response"), dict) else {}
        data = response.get("data") if isinstance(response.get("data"), dict) else {}
        order = data.get("order") if isinstance(data.get("order"), dict) else data
        summaries.append({
            "status": item.get("status"),
            "verification": item.get("verification"),
            "order_id": item.get("order_id") or order.get("orderId"),
            "order_status": item.get("order_status") or item.get("orderStatus"),
            "exchange_code": response.get("code"),
            "error": str(item.get("error") or response.get("msg") or "")[:500] or None,
        })
    return summaries


def record_emergency_protection_cleanup(*, event_id: str | None, attempt_id: str | None,
                                        position_id: str | None, symbol: str,
                                        direction: str, source: str,
                                        close_event_ts_ms: int | None,
                                        retry_number: int,
                                        position_check: dict[str, Any] | None,
                                        result: dict[str, Any]) -> None:
    """Persist each post-flat protection cleanup attempt for restart recovery."""
    emit(
        EMERGENCY_EXITS_PATH,
        "EMERGENCY_PROTECTION_CLEANUP",
        event_id=event_id,
        attempt_id=attempt_id,
        position_id=position_id,
        symbol=symbol,
        direction=direction,
        source=str(source or "UNKNOWN"),
        close_event_ts_ms=close_event_ts_ms,
        retry_number=max(0, int(retry_number)),
        position_check=_snapshot_safe(position_check),
        status=result.get("status"),
        verified=result.get("verified") is True,
        cleanup_result=_snapshot_safe(result),
        remaining_order_ids=list(result.get("remaining_order_ids") or []),
        error="; ".join(str(item) for item in (result.get("errors") or []) if item)[:1000] or None,
    )


def record_emergency_exit(*, event_id: str | None, attempt_id: str | None,
                          position_id: str | None, symbol: str, direction: str,
                          reason: str, requested_qty: float | None,
                          result: dict[str, Any]) -> None:
    """Append a compact, linked record for every emergency-close attempt/result."""
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    emit(
        EMERGENCY_EXITS_PATH, "EMERGENCY_EXIT",
        event_id=event_id, attempt_id=attempt_id, position_id=position_id,
        symbol=symbol, direction=direction, reason=str(reason or "UNSPECIFIED"),
        status=result.get("status"), requested_qty=_snapshot_safe(requested_qty),
        remaining_qty=_snapshot_safe(result.get("remaining_qty")),
        last_known_qty=_snapshot_safe(result.get("last_known_qty")),
        verified_flat=str(result.get("status", "")).lower() in {"closed_verified", "already_closed"},
        verification_status=verification.get("status"),
        verify_poll=result.get("verify_poll"),
        attempt_count=len(result.get("attempts", [])) if isinstance(result.get("attempts"), list) else 0,
        attempts=_emergency_attempt_summary(result),
        cleanup_after_flat_verified=_snapshot_safe(result.get("cleanup_after_flat_verified")),
        error=str(result.get("error") or "")[:1000] or None,
    )

