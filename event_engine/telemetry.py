"""Append-only execution telemetry for the trading engine.

This module is deliberately dependency-light and has no trading logic.  Telemetry
writes are best-effort and must never change a trading decision.  Each journal is
an immutable JSONL event stream with a stable schema version.
"""
from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
SCHEMA_VERSION = 1

QUOTE_SNAPSHOTS_PATH = DATA / "quote_snapshots.jsonl"
ORDER_LIFECYCLE_PATH = DATA / "order_lifecycle.jsonl"
PROTECTION_LIFECYCLE_PATH = DATA / "protection_lifecycle.jsonl"
POSITION_RECONCILIATION_PATH = DATA / "position_reconciliation.jsonl"
EXCHANGE_ERRORS_PATH = DATA / "exchange_errors.jsonl"

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
