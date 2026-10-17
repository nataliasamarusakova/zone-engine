"""Low-overhead, in-memory HTTP and operation timing for one engine run.

This module is diagnostic only: it never retries requests, changes request arguments,
controls trading decisions, or writes runtime data files. Timings are grouped by
service/phase/method/endpoint so slow API calls can be separated from local work.
"""
from __future__ import annotations

import math
import os
from functools import wraps
import threading
import time
from collections import Counter
from contextlib import contextmanager
from typing import Any, Callable, Iterator

_ENABLED = os.environ.get("REQUEST_TIMING_METRICS", "true").strip().lower() not in {"0", "false", "no", "off"}
_MAX_SAMPLES_PER_KEY = 10_000
_LOCK = threading.RLock()
_LOCAL = threading.local()
_REQUESTS: dict[tuple[str, str, str, str], dict[str, Any]] = {}
_HTTP_DETAILS: list[dict[str, Any]] = []
_OPERATIONS: dict[str, dict[str, Any]] = {}


def enabled() -> bool:
    return _ENABLED


def current_phase() -> str:
    return str(getattr(_LOCAL, "phase", "main") or "main")


def current_symbol() -> str | None:
    value = getattr(_LOCAL, "symbol", None)
    return str(value) if value not in (None, "") else None


@contextmanager
def symbol_context(symbol: str) -> Iterator[None]:
    """Associate HTTP calls in the current scan thread with one symbol."""
    previous = getattr(_LOCAL, "symbol", None)
    _LOCAL.symbol = str(symbol)
    try:
        yield
    finally:
        if previous is None:
            try:
                delattr(_LOCAL, "symbol")
            except AttributeError:
                pass
        else:
            _LOCAL.symbol = previous


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Temporarily label HTTP calls made by the current thread."""
    previous = getattr(_LOCAL, "phase", None)
    _LOCAL.phase = str(name or "main")
    try:
        yield
    finally:
        if previous is None:
            try:
                delattr(_LOCAL, "phase")
            except AttributeError:
                pass
        else:
            _LOCAL.phase = previous


def reset() -> None:
    """Start a fresh timing window, normally at the beginning of run_once.main()."""
    with _LOCK:
        _REQUESTS.clear()
        _HTTP_DETAILS.clear()
        _OPERATIONS.clear()


def timed_request(
    service: str,
    endpoint: str,
    http_method: str,
    request_callable: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Time one existing HTTP call, preserving its return value and exceptions."""
    if not _ENABLED:
        return request_callable(*args, **kwargs)

    started = time.perf_counter()
    outcome = "exception"
    failed = True
    try:
        response = request_callable(*args, **kwargs)
        status_code = getattr(response, "status_code", None)
        if status_code is None:
            outcome = "response"
            failed = False
        else:
            outcome = str(int(status_code))
            failed = int(status_code) >= 400
        return response
    except Exception as exc:
        outcome = f"EXC:{type(exc).__name__}"
        raise
    finally:
        _record_request(
            phase=current_phase(),
            service=str(service).lower(),
            endpoint=str(endpoint).split("?", 1)[0],
            method=str(http_method).upper(),
            seconds=max(0.0, time.perf_counter() - started),
            outcome=outcome,
            failed=failed,
        )


def instrument_scan_symbol(function: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate scan_one while preserving its AST-visible body for existing audits/tests."""
    @wraps(function)
    def wrapped(symbol: str, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        outcome = "exception"
        try:
            with symbol_context(str(symbol)):
                with phase("scan_symbol"):
                    result = function(symbol, *args, **kwargs)
            outcome = str(result.get("price_position") or "unknown").lower() if isinstance(result, dict) else "unknown"
            return result
        finally:
            record_operation(
                "scan_symbol", str(symbol), time.perf_counter() - started, outcome=outcome,
            )
    return wrapped


def record_operation(name: str, label: str, seconds: float, *, outcome: str = "ok") -> None:
    """Record a local operation such as a complete per-symbol scan."""
    if not _ENABLED:
        return
    entry = {
        "label": str(label),
        "seconds": max(0.0, float(seconds)),
        "outcome": str(outcome),
    }
    with _LOCK:
        stat = _OPERATIONS.setdefault(str(name), {"count": 0, "samples": [], "outcomes": Counter(), "items": []})
        stat["count"] += 1
        stat["outcomes"][entry["outcome"]] += 1
        if len(stat["samples"]) < _MAX_SAMPLES_PER_KEY:
            stat["samples"].append(entry["seconds"])
        # Keep bounded detail for slowest-item reporting.
        if len(stat["items"]) < _MAX_SAMPLES_PER_KEY:
            stat["items"].append(entry)


def _record_request(
    *, phase: str, service: str, endpoint: str, method: str,
    seconds: float, outcome: str, failed: bool,
) -> None:
    key = (phase, service, method, endpoint)
    with _LOCK:
        stat = _REQUESTS.setdefault(
            key,
            {"count": 0, "errors": 0, "total_seconds": 0.0, "max_seconds": 0.0, "samples": [], "outcomes": Counter()},
        )
        stat["count"] += 1
        stat["errors"] += int(failed)
        stat["total_seconds"] += seconds
        stat["max_seconds"] = max(stat["max_seconds"], seconds)
        stat["outcomes"][outcome] += 1
        if len(stat["samples"]) < _MAX_SAMPLES_PER_KEY:
            stat["samples"].append(seconds)
        if len(_HTTP_DETAILS) < _MAX_SAMPLES_PER_KEY:
            _HTTP_DETAILS.append({
                "phase": phase,
                "symbol": current_symbol(),
                "service": service,
                "method": method,
                "endpoint": endpoint,
                "seconds": seconds,
                "outcome": outcome,
                "failed": failed,
            })


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def snapshot() -> dict[str, Any]:
    """Return a detached, serializable view, mainly for tests and diagnostics."""
    with _LOCK:
        requests: list[dict[str, Any]] = []
        for (phase_name, service, method, endpoint), value in _REQUESTS.items():
            requests.append({
                "phase": phase_name,
                "service": service,
                "method": method,
                "endpoint": endpoint,
                "count": value["count"],
                "errors": value["errors"],
                "total_seconds": value["total_seconds"],
                "max_seconds": value["max_seconds"],
                "samples": list(value["samples"]),
                "outcomes": dict(value["outcomes"]),
            })
        operations: dict[str, Any] = {}
        for name, value in _OPERATIONS.items():
            operations[name] = {
                "count": value["count"],
                "samples": list(value["samples"]),
                "outcomes": dict(value["outcomes"]),
                "items": [dict(item) for item in value["items"]],
            }
        http_details = [dict(item) for item in _HTTP_DETAILS]
    return {"enabled": _ENABLED, "requests": requests, "operations": operations, "http_details": http_details}


def log_summary(logger: Any, *, max_request_groups: int = 40, max_slowest_items: int = 10) -> None:
    """Emit endpoint-level latency distributions and slowest local operations."""
    if not _ENABLED:
        logger.info("[REQUEST_METRICS] disabled=true reason=REQUEST_TIMING_METRICS")
        return

    data = snapshot()
    request_rows = sorted(data["requests"], key=lambda row: row["total_seconds"], reverse=True)
    logger.info(
        "[REQUEST_METRICS_SUMMARY] groups=%d total_calls=%d max_groups=%d",
        len(request_rows), sum(row["count"] for row in request_rows), max_request_groups,
    )
    for row in request_rows[:max_request_groups]:
        samples = row["samples"]
        avg = row["total_seconds"] / row["count"] if row["count"] else 0.0
        logger.info(
            "[HTTP_TIMING] phase=%s service=%s method=%s endpoint=%s calls=%d errors=%d "
            "total_s=%.3f avg_s=%.3f p50_s=%.3f p95_s=%.3f max_s=%.3f outcomes=%s",
            row["phase"], row["service"], row["method"], row["endpoint"],
            row["count"], row["errors"], row["total_seconds"], avg,
            _percentile(samples, 0.50), _percentile(samples, 0.95), row["max_seconds"],
            dict(sorted(row["outcomes"].items())),
        )
    if len(request_rows) > max_request_groups:
        logger.info("[HTTP_TIMING] omitted_groups=%d", len(request_rows) - max_request_groups)

    slow_http = sorted(data["http_details"], key=lambda item: item["seconds"], reverse=True)[:max_slowest_items]
    for item in slow_http:
        logger.info(
            "[HTTP_SLOWEST] phase=%s symbol=%s service=%s method=%s endpoint=%s seconds=%.3f outcome=%s",
            item["phase"], item["symbol"] or "-", item["service"], item["method"],
            item["endpoint"], item["seconds"], item["outcome"],
        )

    for name, stat in sorted(data["operations"].items()):
        samples = stat["samples"]
        logger.info(
            "[OP_TIMING] operation=%s count=%d avg_s=%.3f p50_s=%.3f p95_s=%.3f outcomes=%s",
            name, stat["count"], sum(samples) / len(samples) if samples else 0.0,
            _percentile(samples, 0.50), _percentile(samples, 0.95),
            dict(sorted(stat["outcomes"].items())),
        )
        slowest = sorted(stat["items"], key=lambda item: item["seconds"], reverse=True)[:max_slowest_items]
        for item in slowest:
            logger.info(
                "[OP_SLOWEST] operation=%s label=%s seconds=%.3f outcome=%s",
                name, item["label"], item["seconds"], item["outcome"],
            )
