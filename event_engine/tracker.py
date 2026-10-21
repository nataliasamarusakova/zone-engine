from __future__ import annotations

import copy
import json
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import logging
import math
import os
import shutil
import time
import uuid
try:
    import fcntl
except ImportError:
    fcntl = None
import threading
from pathlib import Path
from typing import Any

from event_engine.bingx import (
    get_position_directional,
    get_positions,
    get_order,
    credentials_available,
    get_all_orders,
    get_fill_orders,
    get_open_protection_directional,
    cancel_order,
    fetch_klines,
    to_bx_symbol,
    get_contract,
    close_position_market,
    _format_price,
    _format_qty,
    _request,
    ORDER_PATH,
    POSITION_PATH,
    position_side_param,
    _validate_sl_order_for_position,
    _repair_client_order_id,
    _client_order_id_matches_base,
    _add_reduce_only_for_one_way,
    _cancel_engine_protection_orders_verified,
    thread_local_retry_session,
)
from event_engine.telegram import send as send_tg
from event_engine import request_metrics, telemetry
from event_engine.state_validation import validate_active_trade_record

log = logging.getLogger("event_engine.tracker")
_TRACKER_TRADE_CLOSED_LOG_FORMAT = (
    "[TRACKER_TRADE_CLOSED] %s %s (%s) | PnL: %+.2f%% | "
    "Realized R:R: %s | Planned R:R: %.2f | Exit: %.8g (%s) | Duration: %.1f min"
)
_TRACKER_TRADE_CLOSED_UNCONFIRMED_LOG_FORMAT = (
    "[TRACKER_TRADE_CLOSED] %s %s (%s) | PnL: %s | "
    "Realized R:R: %s | Planned R:R: %.2f | Exit: %.8g (%s) | Duration: %.1f min"
)


def _pnl_log_value(value: float | None) -> str:
    """Render unknown/unconfirmed PnL without passing None to numeric logging formats."""
    return f"{value:+.2f}%" if value is not None else "unconfirmed"


_POSITION_ABSENCE_UNVERIFIED_KEYS = (
    "position_absence_unverified",
    "position_absence_unverified_ts",
    "position_absence_unverified_last_seen_ts",
    "position_absence_unverified_qty",
    "position_absence_unverified_reason",
    "position_absence_unverified_alerted",
    "position_absence_unverified_alert_ts",
)


def _clear_position_absence_unverified_if_position_found(
    trade: dict[str, Any], *, exchange_qty: float, now_ms: int, qty_tolerance: float,
) -> bool:
    """Clear a stale absence marker only after an authoritative positive position read.

    This deliberately does not close a trade or unblock reconciliation when the exchange
    still reports no position; that case needs confirmed exit evidence or operator review.
    """
    if exchange_qty <= max(0.0, qty_tolerance) or not trade.get("position_absence_unverified"):
        return False
    first_seen = trade.get("position_absence_unverified_ts")
    if first_seen is not None:
        trade["position_absence_unverified_first_seen_ts"] = first_seen
    trade["position_absence_unverified_resolved_ts"] = int(now_ms)
    trade["position_absence_unverified_resolution_reason"] = "exchange_position_reappeared"
    for key in _POSITION_ABSENCE_UNVERIFIED_KEYS:
        trade.pop(key, None)
    return True

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
ACTIVE_TRADES_PATH = DATA / "active_trades.json"
TRADES_PATH = DATA / "trades.jsonl"
ACTIONS_PATH = DATA / "actions.jsonl"
DEFAULT_PLANNED_WEIGHTED_RR = 0.45

# The executor is scoped to one update_active_trades call so it is always shut
# down cleanly and never leaves a worker alive after the engine run completes.

def _tracker_parallel_first_tp_read_enabled() -> bool:
    return os.environ.get("TRACKER_PARALLEL_FIRST_TP_READ", "true").strip().lower() not in {"0", "false", "no", "off"}


def _record_tracker_elapsed(timings: dict[str, dict[str, float]], name: str, elapsed: float) -> None:
    if not request_metrics.enabled():
        return
    elapsed = max(0.0, float(elapsed))
    bucket = timings.setdefault(name, {"calls": 0.0, "seconds": 0.0, "max_seconds": 0.0})
    bucket["calls"] += 1
    bucket["seconds"] += elapsed
    bucket["max_seconds"] = max(bucket["max_seconds"], elapsed)


def _fetch_tp_evidence_timed(
    symbol: str, order_id: str, entry_ts: int, phase_name: str,
) -> tuple[dict | None, float, Exception | None]:
    """Fetch one read-only TP status in a worker with isolated retry/session context."""
    started = time.perf_counter() if request_metrics.enabled() else None
    try:
        with request_metrics.phase(phase_name), request_metrics.symbol_context(symbol), thread_local_retry_session():
            value = _get_order_execution_evidence(symbol, order_id, entry_ts=entry_ts)
        elapsed = max(0.0, time.perf_counter() - started) if started is not None else 0.0
        return value, elapsed, None
    except Exception as exc:
        elapsed = max(0.0, time.perf_counter() - started) if started is not None else 0.0
        return None, elapsed, exc


def _timed_tracker_call(
    timings: dict[str, dict[str, float]],
    name: str,
    function,
    *args,
    **kwargs,
):
    """Measure one tracker operation only when request timing diagnostics are enabled."""
    if not request_metrics.enabled():
        return function(*args, **kwargs)
    started = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        _record_tracker_elapsed(timings, name, time.perf_counter() - started)


def _log_tracker_timings(timings: dict[str, dict[str, float]], total_seconds: float, active_trades: int) -> None:
    """Emit one compact diagnostic line only in explicit timing-diagnostic mode."""
    if not request_metrics.enabled():
        return
    pieces = [
        f"total_s={max(0.0, total_seconds):.3f}",
        f"active_trades={max(0, int(active_trades))}",
    ]
    for name in (
        "kline_1m",
        "tp_order_evidence",
        "tp_position_recheck",
        "final_position_recheck",
        "be_retry",
        "be_after_tp1",
        "exit_sl_order",
        "historical_exit_lookup",
        "close_fill_accounting",
        "exit_order_evidence",
        "active_state_save",
    ):
        item = timings.get(name)
        if not item:
            continue
        pieces.extend((
            f"{name}_calls={int(item['calls'])}",
            f"{name}_s={item['seconds']:.3f}",
            f"{name}_max_s={item['max_seconds']:.3f}",
        ))
    log.info("[TRACKER_PERF_DETAIL] %s", " ".join(pieces))


_ACTIVE_TRADES_THREAD_LOCK = threading.RLock()


@contextmanager
def _active_trades_lock():
    """Serialize active-state read/validate/mutate/write transactions across threads/processes."""
    _ACTIVE_TRADES_THREAD_LOCK.acquire()
    lockf = None
    try:
        ACTIVE_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
        lock_path = ACTIVE_TRADES_PATH.with_suffix(ACTIVE_TRADES_PATH.suffix + ".lock")
        lockf = lock_path.open("a+")
        if fcntl is not None:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if lockf is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
            finally:
                lockf.close()
        _ACTIVE_TRADES_THREAD_LOCK.release()


class ActiveTradeStateCorrupt(RuntimeError):
    """The persisted active-trade state cannot be trusted for trading decisions."""


def _quarantine_state_copy(path: Path, reason: str) -> None:
    """Copy, but never remove, a corrupt state file for forensic recovery."""
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    quarantine = path.with_name(f"{path.name}.quarantine.{stamp}")
    suffix = 1
    while quarantine.exists():
        quarantine = path.with_name(f"{path.name}.quarantine.{stamp}.{suffix}")
        suffix += 1
    try:
        shutil.copy2(path, quarantine)
        log.critical("[TRACKER_STATE_CORRUPT] quarantined copy=%s reason=%s", quarantine, reason)
    except Exception as exc:
        log.critical("[TRACKER_STATE_CORRUPT] could not quarantine %s: %s", path, exc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if result != result:
            return default
        if result in (float("inf"), float("-inf")):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    """Parse an exchange numeric field without converting unavailable data to zero."""
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _all_present_numeric(rows: list[dict], key: str) -> bool:
    return bool(rows) and all(_optional_float(row.get(key)) is not None for row in rows)


def _adverse_exit_slippage_pct(direction: str, actual_fill: float, trigger_price: float) -> float | None:
    """Return positive adverse stop/BE slippage; favorable fills are zero."""
    actual = _safe_float(actual_fill, 0.0)
    trigger = _safe_float(trigger_price, 0.0)
    if actual <= 0 or trigger <= 0:
        return None
    raw = (actual - trigger) / trigger * 100.0
    return max(0.0, -raw) if str(direction).upper() == "LONG" else max(0.0, raw)


def _is_full_tp_close(position_gone: bool, realized_qty: float, initial_qty: float, hit_legs: set[str]) -> bool:
    """A full TP close requires the whole initial quantity to be confirmed as realized."""
    if not position_gone or not hit_legs:
        return False
    init = max(0.0, _safe_float(initial_qty, 0.0))
    realized = max(0.0, _safe_float(realized_qty, 0.0))
    if init <= 0:
        return False
    tolerance = max(1e-12, init * 1e-8)
    return realized >= init - tolerance and realized <= init + tolerance


def _load_active_trades(*, allow_missing: bool = True) -> dict[str, dict]:
    """Load and validate active state.

    Missing state is tolerated only for non-trading helpers that explicitly use the
    default ``allow_missing=True``. Mutation paths that can own a live exchange
    position must pass ``allow_missing=False`` so a missing state cannot silently
    become an empty owner set after an entry has filled.
    """
    if not ACTIVE_TRADES_PATH.exists():
        if allow_missing:
            return {}
        raise ActiveTradeStateCorrupt("active_trades.json is missing")
    try:
        data = json.loads(ACTIVE_TRADES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            normalized = {}
            for event_id, trade in data.items():
                try:
                    validate_active_trade_record(event_id, trade)
                except ValueError as exc:
                    reason = str(exc)
                    _quarantine_state_copy(ACTIVE_TRADES_PATH, reason)
                    raise ActiveTradeStateCorrupt(reason) from exc
                t = dict(trade)
                t.setdefault("mae_pct", 0.0)
                t.setdefault("max_drawdown_pct", 0.0)
                t.setdefault("be_required", False)
                t.setdefault("be_last_error", None)
                t.setdefault("be_trigger_rule", "after_tp1_filled")
                t.setdefault("strategy_version", None)
                t.setdefault("signal_snapshot", {})
                t.setdefault("entry_bar", {})
                t.setdefault("previous_bar", {})
                t.setdefault("entry_order", {})
                t.setdefault("fill_position", {})
                t.setdefault("execution_snapshot", {})
                t.setdefault("tp_fill_events", [])
                t.setdefault("tp_fill_classification", {})
                t.setdefault("tp_mode", "single_tp" if len(t.get("tp_orders", [])) == 1 else "multi_tp")
                t.setdefault("effective_tp_levels", t.get("tp_levels", []))
                t.setdefault("effective_weighted_rr", t.get("planned_weighted_rr", DEFAULT_PLANNED_WEIGHTED_RR))
                t.setdefault("realized_weighted_rr", None)
                t.setdefault("remaining_weighted_rr", None)
                normalized[str(event_id)] = t
            return normalized
        reason = f"state root is {type(data).__name__}, expected object"
        _quarantine_state_copy(ACTIVE_TRADES_PATH, reason)
        raise ActiveTradeStateCorrupt(reason)
    except Exception as exc:
        log.error("[TRACKER] Corrupt state in %s: %s", ACTIVE_TRADES_PATH, exc)
        if isinstance(exc, ActiveTradeStateCorrupt):
            raise
        reason = f"{type(exc).__name__}: {exc}"
        _quarantine_state_copy(ACTIVE_TRADES_PATH, reason)
        raise ActiveTradeStateCorrupt(reason) from exc


def _load_active_trades_required() -> dict[str, dict]:
    """Load active state when a live owner is expected; missing state is fatal."""
    if not ACTIVE_TRADES_PATH.exists():
        raise ActiveTradeStateCorrupt("active_trades.json is missing")
    return _load_active_trades()


def _write_active_trades_unlocked(trades: dict[str, dict]) -> None:
    """Atomically replace active state; caller must hold _active_trades_lock()."""
    ACTIVE_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ACTIVE_TRADES_PATH.with_name(ACTIVE_TRADES_PATH.name + ".tmp")
    payload = json.dumps(trades, ensure_ascii=False, indent=2, allow_nan=False)
    with tmp_path.open("w", encoding="utf-8") as tf:
        tf.write(payload)
        tf.flush()
        os.fsync(tf.fileno())
    os.replace(tmp_path, ACTIVE_TRADES_PATH)
    if hasattr(os, "O_DIRECTORY"):
        try:
            dir_fd = os.open(str(ACTIVE_TRADES_PATH.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass


def _save_active_trades(trades: dict[str, dict]) -> None:
    """Atomically persist active-trade state and serialize concurrent writers."""
    with _active_trades_lock():
        _write_active_trades_unlocked(trades)


def record_verified_emergency_close(event_id: str, close_result: dict, *, reason: str) -> bool:
    """Persist verified flatness for an existing owner so tracker can close it honestly."""
    if not isinstance(close_result, dict) or str(close_result.get("status", "")).lower() != "closed_verified":
        return False
    with _active_trades_lock():
        trades = _load_active_trades_required()
        trade = trades.get(str(event_id))
        if not isinstance(trade, dict) or trade.get("closed") is True:
            return False
        trade["verified_emergency_close"] = {
            "status": "closed_verified",
            "reason": str(reason),
            "verified_at_ms": int(time.time() * 1000),
            "evidence": close_result,
        }
        _write_active_trades_unlocked(trades)
    return True


def _save_active_trades_after_reconciliation(
    original_trades: dict[str, dict],
    updated_trades: dict[str, dict],
) -> None:
    """Apply reconciliation changes only when an event was not concurrently modified.

    Reconciliation can take seconds.  Before persisting its snapshot, compare each
    event with the exact state that reconciliation started from.  A concurrent run
    that changed or closed the same event wins; the next cycle can reconcile it again.
    This prevents both new-event loss and same-event stale-state clobbering.
    """
    with _active_trades_lock():
        # Reconciliation writes are only valid against an already-trusted state file.
        # A state file disappearing after preflight is an integrity failure, not a
        # fresh-start condition, because this function is called only when it had
        # a real active-state snapshot to reconcile. Never replace that missing state
        # with an empty object.
        latest = _load_active_trades_required()
        for raw_event_id, original in original_trades.items():
            key = str(raw_event_id)
            if key not in latest:
                # Another process already removed/closed it. Never resurrect stale state.
                continue
            if latest[key] != original:
                log.warning("[TRACKER_STATE_RACE] skip stale reconciliation write event=%s", key)
                continue
            if key in updated_trades:
                latest[key] = updated_trades[key]
            else:
                # This event was closed successfully by this reconciliation cycle.
                latest.pop(key, None)
        _write_active_trades_unlocked(latest)


def backfill_active_trade_provenance(trades: dict[str, dict] | None = None) -> int:
    """Backfill missing code_commit_sha on open trades from their immutable TRADE_OPEN record."""
    current = trades if trades is not None else _load_active_trades()
    if not current or not TRADES_PATH.exists():
        return 0
    missing_ids = {str(eid) for eid, trade in current.items()
                   if isinstance(trade, dict) and not trade.get("closed", False) and not trade.get("code_commit_sha")}
    if not missing_ids:
        return 0
    found: dict[str, str] = {}
    try:
        with TRADES_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("record_type") != "TRADE_OPEN":
                    continue
                eid = str(row.get("event_id") or "")
                sha = str(row.get("code_commit_sha") or "")
                if eid in missing_ids and sha:
                    found[eid] = sha
        changed = 0
        for eid, sha in found.items():
            current[eid]["code_commit_sha"] = sha
            current[eid]["provenance_backfilled_ts"] = int(time.time() * 1000)
            changed += 1
        if changed:
            _save_active_trades(current)
        return changed
    except OSError as exc:
        log.warning("[TRACKER_PROVENANCE] backfill failed: %s", exc)
        return 0


def _close_record_exists(event_id: str) -> bool:
    if not TRADES_PATH.exists():
        return False
    try:
        with TRADES_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("record_type") == "TRADE_CLOSE" and str(obj.get("event_id")) == str(event_id):
                    return True
    except OSError as exc:
        log.warning("[TRACKER] Could not inspect close journal: %s", exc)
    return False


_JOURNAL_LOCK = threading.Lock()

def _append_journal_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with _JOURNAL_LOCK:
        with lock_path.open("a+", encoding="utf-8") as lockf:
            if fcntl is not None:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _append_trade_record(record: dict) -> None:
    _append_journal_record(TRADES_PATH, record)


def _append_action_record(record: dict) -> None:
    _append_journal_record(ACTIONS_PATH, record)


def _send_tracker_notification(kind: str, event_id: str, text: str, *, symbol: str, leg: str | None = None) -> bool:
    """Send a non-trading notification and record the delivery outcome.

    Telegram delivery is deliberately not retried here: a client-side timeout
    can mean Telegram accepted the message, and an automatic resend could create
    a duplicate notification. The caller must treat False as a delivery failure
    for observability, never as a trading-state failure.
    """
    try:
        ok = bool(send_tg(text))
    except Exception as exc:
        ok = False
        log.error("[TG_%s_FAILED] event=%s symbol=%s leg=%s error=%s", kind, event_id, symbol, leg or "", exc)
        try:
            _append_action_record({
                "ts": int(time.time() * 1000),
                "action": kind,
                "event_id": str(event_id),
                "symbol": symbol,
                "leg": leg,
                "telegram_ok": False,
                "error": str(exc),
            })
        except Exception as journal_exc:
            log.error("[TRACKER_ACTION_JOURNAL_FAILED] event=%s kind=%s error=%s", event_id, kind, journal_exc)
        return False

    try:
        _append_action_record({
            "ts": int(time.time() * 1000),
            "action": kind,
            "event_id": str(event_id),
            "symbol": symbol,
            "leg": leg,
            "telegram_ok": ok,
        })
    except Exception as journal_exc:
        log.error("[TRACKER_ACTION_JOURNAL_FAILED] event=%s kind=%s error=%s", event_id, kind, journal_exc)
    if not ok:
        log.error("[TG_%s_FAILED] event=%s symbol=%s leg=%s send() returned False", kind, event_id, symbol, leg or "")
    else:
        log.debug("[TG_%s_SENT] event=%s symbol=%s leg=%s", kind, event_id, symbol, leg or "")
    return ok

def _append_trade_close_once(record: dict) -> bool:
    """Atomically avoid duplicate TRADE_CLOSE records across overlapping runs."""
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = TRADES_PATH.with_suffix(TRADES_PATH.suffix + ".lock")
    with _JOURNAL_LOCK:
        with lock_path.open("a+", encoding="utf-8") as lockf:
            if fcntl is not None:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                event_id = str(record.get("event_id", ""))
                if TRADES_PATH.exists():
                    with TRADES_PATH.open("r", encoding="utf-8") as rf:
                        for line in rf:
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            if obj.get("record_type") == "TRADE_CLOSE" and str(obj.get("event_id")) == event_id:
                                return False
                with TRADES_PATH.open("a", encoding="utf-8") as wf:
                    wf.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    wf.flush()
                    os.fsync(wf.fileno())
                return True
            finally:
                if fcntl is not None:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _normalize_direction(direction: str) -> str:
    d = str(direction or "").upper()
    if d not in {"LONG", "SHORT"}:
        raise ValueError(f"Invalid direction={direction}")
    return d


def _position_snapshot_symbol_key(symbol: Any) -> str:
    """Normalize exchange/local symbol spellings to the same position key."""
    return str(symbol or "").upper().replace("-", "")


def _position_from_snapshot(positions: list[dict], symbol: str, direction: str) -> dict:
    """Look up a position from a successful all-positions response.

    The caller may only use this result when the bulk endpoint completed
    successfully. An absent matching position is therefore authoritative for
    this first snapshot; later tracker checks still query the exchange again.
    """
    target_symbol = _position_snapshot_symbol_key(symbol)
    target_direction = _normalize_direction(direction)
    for row in positions:
        if not isinstance(row, dict):
            continue
        if _position_snapshot_symbol_key(row.get("symbol")) != target_symbol:
            continue

        raw_side = str(row.get("positionSide", "")).upper()
        amount = _safe_float(row.get("positionAmt"), 0.0)
        if raw_side == "BOTH":
            row_direction = "LONG" if amount > 0 else "SHORT" if amount < 0 else None
        elif raw_side in {"LONG", "SHORT"}:
            row_direction = raw_side
        else:
            continue
        if row_direction != target_direction:
            continue

        qty = abs(amount)
        avg_price = _safe_float(row.get("avgPrice") or row.get("entryPrice"), 0.0)
        if qty <= 0 or avg_price <= 0:
            continue
        return {
            "status": "found",
            "symbol": row.get("symbol", symbol),
            "positionSide": target_direction,
            "avgPrice": avg_price,
            "positionAmt": qty,
            "entryPrice": _safe_float(row.get("entryPrice"), avg_price) or avg_price,
        }

    return {"status": "not_found", "symbol": symbol, "positionSide": target_direction}


def _normalized_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace("-USDT", "")


def _display_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace("-", "")


def _active_trade_conflicts(
    trades: dict[str, dict], symbol: str, direction: str,
    *, exclude_event_id: str | None = None,
) -> list[tuple[str, dict]]:
    """Return active local trades for one exchange position identity.

    BingX position state is unique at the (symbol, direction) level for the
    engine's configured position mode. Local state must therefore never silently
    allow two live owners for that same exchange position.
    """
    want_bx = _normalized_symbol(symbol)
    want_direction = str(direction).upper()
    conflicts: list[tuple[str, dict]] = []
    for event_id, trade in trades.items():
        if str(event_id) == str(exclude_event_id or ""):
            continue
        if not isinstance(trade, dict) or trade.get("closed", False):
            continue
        trade_bx = _normalized_symbol(trade.get("symbol", ""))
        if trade_bx == want_bx and str(trade.get("direction", "")).upper() == want_direction:
            conflicts.append((str(event_id), trade))
    return conflicts


def has_active_trade_conflict(symbol: str, direction: str, *, exclude_event_id: str | None = None) -> bool:
    # This helper is called only inside live-execution admission after state
    # preflight. A missing/corrupt file at this point is an integrity failure,
    # not an empty owner set. Fail closed instead of allowing an order through a
    # state race.
    return bool(_active_trade_conflicts(_load_active_trades_required(), symbol, direction, exclude_event_id=exclude_event_id))


def _collect_order_ids(trade: dict) -> set[str]:
    ids: set[str] = set()
    entry_order = trade.get("entry_order") if isinstance(trade.get("entry_order"), dict) else {}
    for key in ("order_id", "orderId"):
        if entry_order.get(key):
            ids.add(str(entry_order[key]))
    sl = trade.get("sl_order") if isinstance(trade.get("sl_order"), dict) else {}
    for key in ("order_id", "orderId"):
        if sl.get(key):
            ids.add(str(sl[key]))
    for tp in trade.get("tp_orders", []) if isinstance(trade.get("tp_orders"), list) else []:
        if not isinstance(tp, dict):
            continue
        for key in ("order_id", "orderId"):
            if tp.get(key):
                ids.add(str(tp[key]))
    for key in ("be_order_id", "exit_order_id"):
        if trade.get(key):
            ids.add(str(trade[key]))
    for event in trade.get("tp_fill_events", []) if isinstance(trade.get("tp_fill_events"), list) else []:
        if isinstance(event, dict) and event.get("order_id"):
            ids.add(str(event["order_id"]))
    return ids


def _protection_order_ids(tp_orders: list[dict] | None, sl_result: dict | None) -> list[str]:
    ids: list[str] = []
    for container in ([sl_result] if isinstance(sl_result, dict) else []):
        for key in ("order_id", "orderId"):
            if container.get(key):
                ids.append(str(container[key]))
    for tp in tp_orders if isinstance(tp_orders, list) else []:
        if not isinstance(tp, dict):
            continue
        for key in ("order_id", "orderId"):
            if tp.get(key):
                ids.append(str(tp[key]))
    return ids


def _order_ids_owned_by_other_trades(trades: dict[str, dict], order_ids: set[str], *, exclude_event_id: str) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    if not order_ids:
        return owners
    for other_event_id, other_trade in trades.items():
        if str(other_event_id) == str(exclude_event_id) or not isinstance(other_trade, dict) or other_trade.get("closed", False):
            continue
        overlap = _collect_order_ids(other_trade) & order_ids
        for order_id in overlap:
            owners.setdefault(order_id, []).append(str(other_event_id))
    return owners


def _validate_protection_order_ownership(
    trades: dict[str, dict], event_id: str, symbol: str, direction: str,
    tp_orders: list[dict], sl_result: dict,
) -> tuple[bool, str, list[str]]:
    ids = _protection_order_ids(tp_orders, sl_result)
    duplicate_ids = sorted({oid for oid in ids if ids.count(oid) > 1})
    if duplicate_ids:
        return False, "one protection payload references the same exchange order more than once", duplicate_ids
    owners = _order_ids_owned_by_other_trades(trades, set(ids), exclude_event_id=event_id)
    if owners:
        conflicting = sorted(owners)
        return False, f"exchange order id already belongs to another active event: {owners}", conflicting
    return True, "", []


def update_active_trade_protection(
    symbol: str,
    direction: str,
    tp_orders: list[dict],
    sl_result: dict,
    effective_tp_levels: list[dict] | None = None,
    tp_mode: str | None = None,
    effective_weighted_rr: float | None = None,
    *,
    event_id: str | None = None,
) -> bool:
    with _active_trades_lock():
        return _update_active_trade_protection_locked(
            symbol, direction, tp_orders, sl_result, effective_tp_levels,
            tp_mode, effective_weighted_rr, event_id=event_id,
        )


def _update_active_trade_protection_locked(
    symbol: str,
    direction: str,
    tp_orders: list[dict],
    sl_result: dict,
    effective_tp_levels: list[dict] | None = None,
    tp_mode: str | None = None,
    effective_weighted_rr: float | None = None,
    *,
    event_id: str | None = None,
) -> bool:
    trades = _load_active_trades_required()
    want_bx = _normalized_symbol(symbol)
    want_direction = str(direction).upper()

    candidates: list[tuple[str, dict]] = []
    if event_id:
        trade = trades.get(str(event_id))
        if isinstance(trade, dict) and not trade.get("closed", False):
            candidates.append((str(event_id), trade))
    else:
        candidates = _active_trade_conflicts(trades, symbol, direction)

    # Without an explicit event_id there must be exactly one unambiguous local owner.
    if len(candidates) != 1:
        owner_ids = [eid for eid, _ in candidates]
        telemetry.record_state_conflict(
            event_id=event_id, attempt_id=None, position_id=event_id, order_id=None,
            symbol=symbol, direction=direction, conflict_type="PROTECTION_OWNER_AMBIGUOUS",
            message=f"cannot update protection without exactly one local owner; candidates={owner_ids}",
            owner_event_ids=owner_ids,
        )
        return False

    owner_event_id, trade = candidates[0]
    trade_bx = _normalized_symbol(trade.get("symbol", ""))
    trade_direction = str(trade.get("direction", "")).upper()
    if trade_bx != want_bx or trade_direction != want_direction:
        telemetry.record_state_conflict(
            event_id=owner_event_id, attempt_id=trade.get("attempt_id"), position_id=trade.get("position_id") or owner_event_id, order_id=None,
            symbol=symbol, direction=direction, conflict_type="PROTECTION_OWNER_MISMATCH",
            message=f"requested={want_bx}:{want_direction} owner={trade_bx}:{trade_direction}",
            owner_event_ids=[owner_event_id],
        )
        return False

    ownership_ok, ownership_message, conflicting_order_ids = _validate_protection_order_ownership(
        trades, owner_event_id, symbol, direction, tp_orders if isinstance(tp_orders, list) else [], sl_result if isinstance(sl_result, dict) else {}
    )
    if not ownership_ok:
        telemetry.record_state_conflict(
            event_id=owner_event_id, attempt_id=trade.get("attempt_id"), position_id=trade.get("position_id") or owner_event_id,
            order_id=conflicting_order_ids[0] if conflicting_order_ids else None, symbol=symbol, direction=direction,
            conflict_type="ORDER_OWNERSHIP_CONFLICT", message=ownership_message,
            owner_event_ids=[owner_event_id], conflicting_order_ids=conflicting_order_ids,
        )
        log.error("[TRACKER_STATE_CONFLICT] refusing protection ownership update event=%s orders=%s", owner_event_id, conflicting_order_ids)
        return False

    # Keep execution-bearing historical TP identities when protection is rebuilt.
    # Replacing the list wholesale can erase a filled TP1 whose ID only survives
    # in setup.protection_result, preventing TP1 accounting and BE activation.
    new_tp_orders = tp_orders if isinstance(tp_orders, list) else []
    new_ids = {str(x.get("order_id")) for x in new_tp_orders if isinstance(x, dict) and x.get("order_id")}
    preserved_tp_orders = []
    for old_tp in trade.get("tp_orders", []) if isinstance(trade.get("tp_orders"), list) else []:
        if not isinstance(old_tp, dict):
            continue
        old_id = str(old_tp.get("order_id") or "")
        old_leg = str(old_tp.get("leg") or "")
        if old_id and old_id not in new_ids and (
            trade.get("tp_recovery_pending")
            or old_tp.get("recovered_execution_pending")
            or old_leg in set(trade.get("hit_legs", []))
            or str(old_tp.get("status", "")).upper() == "FILLED"
        ):
            preserved_tp_orders.append(old_tp)
    trade["tp_orders"] = preserved_tp_orders + new_tp_orders
    trade["sl_order"] = sl_result if isinstance(sl_result, dict) else {}

    # The SL saved after reconciliation is the authoritative currently-verified
    # stop. If the previously recorded BE order disappeared and a normal engine
    # SL was rebuilt, do not leave be_activated=True: that would suppress the
    # TP1->BE retry and misrepresent the protection actually on the exchange.
    new_sl_order_id = str(trade["sl_order"].get("order_id") or trade["sl_order"].get("orderId") or "")
    new_sl_client_id = str(
        trade["sl_order"].get("client_order_id") or trade["sl_order"].get("clientOrderId") or ""
    ).upper()
    old_be_order_id = str(trade.get("be_order_id") or "")
    recorded_be_client_id = str(trade.get("be_client_order_id") or "").upper()
    same_recorded_be_identity = bool(
        recorded_be_client_id
        and new_sl_client_id.startswith("EVT_BE_")
        and _client_order_id_matches_base(new_sl_client_id, recorded_be_client_id)
    )
    current_sl_is_recorded_be = bool(
        new_sl_order_id
        and new_sl_client_id.startswith("EVT_BE_")
        and (new_sl_order_id == old_be_order_id or same_recorded_be_identity)
        and trade.get("sl_order", {}).get("protection_order_verified", True) is not False
    )
    if current_sl_is_recorded_be:
        # The exchange order ID may change while a verified BE is repaired under
        # its recorded client ID. Keep the BE state, but refresh the concrete ID
        # used by exit classification/accounting.
        trade["be_order_id"] = new_sl_order_id
        trade["be_client_order_id"] = new_sl_client_id
    if trade.get("be_activated") and new_sl_order_id and not current_sl_is_recorded_be:
        trade["be_previous_order_id"] = old_be_order_id or None
        trade["be_previous_activation_ts"] = trade.get("be_activation_ts")
        trade["be_activated"] = False
        trade["be_order_id"] = None
        trade["be_client_order_id"] = None
        trade["be_activation_ts"] = None
        tp1_was_hit = any(str(leg).lower() == "tp1" for leg in (trade.get("hit_legs") or []))
        remaining_qty = max(0.0, _safe_float(trade.get("remaining_qty"), 0.0))
        trade["be_required"] = bool(tp1_was_hit and remaining_qty > max(1e-12, _safe_float(trade.get("initial_qty"), 0.0) * 1e-9))
        trade["be_last_error"] = (
            "reconciliation replaced missing break-even stop with a non-BE stop; retry required"
            if trade["be_required"] else None
        )
        log.warning(
            "[TRACKER_BE_STATE_RECONCILED] event=%s symbol=%s previous_be_order=%s new_sl=%s be_required=%s",
            owner_event_id, symbol, old_be_order_id or "unknown", new_sl_order_id, trade["be_required"],
        )
    if effective_tp_levels is not None:
        trade["effective_tp_levels"] = effective_tp_levels
    if tp_mode:
        trade["tp_mode"] = tp_mode
    # Recompute weighted RR from the whole original position. The caller may provide
    # a remaining-leg RR (for example 0.60 after TP1), but the persisted
    # effective_weighted_rr must include the realized TP1 leg as well.
    rr_effective, rr_realized, rr_remaining = _weighted_rr_snapshot(trade)
    if rr_effective is not None:
        trade["effective_weighted_rr"] = rr_effective
        trade["realized_weighted_rr"] = rr_realized
        trade["remaining_weighted_rr"] = rr_remaining
    elif effective_weighted_rr is not None:
        trade["effective_weighted_rr"] = _safe_float(effective_weighted_rr, DEFAULT_PLANNED_WEIGHTED_RR)
    trade["protection_last_updated_ts"] = int(time.time() * 1000)
    _write_active_trades_unlocked(trades)
    return True


def merge_reconciled_tp_orders(event_id: str, recovered_orders: list[dict]) -> bool:
    """Persist original TP order identities recovered from exchange order history.

    This deliberately does not mark a leg filled or book PnL. The normal tracker
    must independently re-query the order and apply its existing execution-evidence
    path before mutating hit_legs, realized PnL, or break-even state.
    """
    if not recovered_orders:
        return True
    with _active_trades_lock():
        trades = _load_active_trades_required()
        trade = trades.get(str(event_id))
        if not isinstance(trade, dict) or trade.get("closed"):
            return False
        current = trade.get("tp_orders") if isinstance(trade.get("tp_orders"), list) else []
        by_id = {str(x.get("order_id")): x for x in current if isinstance(x, dict) and x.get("order_id")}
        changed = False
        valid_recovery_seen = False
        for order in recovered_orders:
            if not isinstance(order, dict) or not order.get("order_id"):
                continue
            valid_recovery_seen = True
            oid = str(order["order_id"])
            item = by_id.get(oid)
            if item is None:
                item = dict(order)
                current.append(item)
                by_id[oid] = item
                changed = True
            # Idempotently restore the marker even if another tracker/reconcile
            # pass inserted the same order between our read and this locked write.
            if not item.get("recovered_execution_pending"):
                item["recovered_execution_pending"] = True
                changed = True
            # Keep the exchange-confirmed execution facts if this ID already
            # existed, without overwriting unrelated local fields.
            for key in ("status", "executed_qty", "avg_price", "exchange_update_time_ms"):
                value = order.get(key)
                if value is not None and item.get(key) != value:
                    item[key] = value
                    changed = True
        if valid_recovery_seen and not trade.get("tp_recovery_pending"):
            trade["tp_recovery_pending"] = True
            changed = True
        if changed:
            trade["tp_orders"] = current
            _write_active_trades_unlocked(trades)
        return valid_recovery_seen


def _extract_setup_metrics(setup: dict | None) -> dict[str, Any]:
    if not isinstance(setup, dict):
        return {
            "planned_risk_pct": None,
            "planned_target_rr": None,
            "planned_weighted_rr": DEFAULT_PLANNED_WEIGHTED_RR,
            "entry_reference": None,
            "invalidation_price": None,
            "target_price": None,
            "tp_levels": [],
            "effective_tp_levels": [],
            "effective_weighted_rr": DEFAULT_PLANNED_WEIGHTED_RR,
            "tp_mode": "multi_tp",
            "strategy_version": None,
            "code_commit_sha": None,
            "signal_snapshot": {},
            "entry_bar": {},
            "previous_bar": {},
            "entry_order": {},
            "fill_position": {},
            "execution_snapshot": {},
            "tp_fill_events": [],
        }

    return {
        "planned_risk_pct": _safe_float(setup.get("risk_pct"), 0.0) if setup.get("risk_pct") is not None else None,
        "planned_target_rr": _safe_float(setup.get("target_rr"), 0.0) if setup.get("target_rr") is not None else None,
        "planned_weighted_rr": _safe_float(setup.get("planned_weighted_rr", DEFAULT_PLANNED_WEIGHTED_RR), DEFAULT_PLANNED_WEIGHTED_RR),
        "effective_tp_levels": setup.get("effective_tp_levels") if isinstance(setup.get("effective_tp_levels"), list) else [],
        "effective_weighted_rr": _safe_float(setup.get("effective_weighted_rr", setup.get("planned_weighted_rr", DEFAULT_PLANNED_WEIGHTED_RR)), DEFAULT_PLANNED_WEIGHTED_RR),
        "tp_mode": str(setup.get("tp_mode", "multi_tp")),
        "strategy_version": str(setup.get("strategy_version", "")) or None,
        "code_commit_sha": str(setup.get("code_commit_sha", "")) or None,
        "signal_snapshot": setup.get("signal_snapshot") if isinstance(setup.get("signal_snapshot"), dict) else {},
        "entry_bar": setup.get("entry_bar") if isinstance(setup.get("entry_bar"), dict) else {},
        "previous_bar": setup.get("previous_bar") if isinstance(setup.get("previous_bar"), dict) else {},
        "entry_order": setup.get("entry_order") if isinstance(setup.get("entry_order"), dict) else {},
        "fill_position": setup.get("fill_position") if isinstance(setup.get("fill_position"), dict) else {},
        "execution_snapshot": setup.get("execution_snapshot") if isinstance(setup.get("execution_snapshot"), dict) else {},
        "entry_reference": _safe_float(setup.get("entry_reference"), 0.0) if setup.get("entry_reference") is not None else None,
        "invalidation_price": _safe_float(setup.get("invalidation_price"), 0.0) if setup.get("invalidation_price") is not None else None,
        "target_price": _safe_float(setup.get("target_price"), 0.0) if setup.get("target_price") is not None else None,
        "tp_levels": setup.get("tp_levels") if isinstance(setup.get("tp_levels"), list) else [],
    }


def register_active_trade(
    event_id: str,
    symbol: str,
    name: str,
    direction: str,
    entry_price: float,
    qty: float,
    tp_orders: list[dict],
    sl_result: dict,
    event_type: str,
    timeframe: str | None = None,
    score: float = 50.0,
    setup: dict | None = None,
    requested_entry_price: float | None = None,
    entry_ts_ms: int | None = None,
) -> bool:
    with _active_trades_lock():
        return _register_active_trade_locked(
            event_id, symbol, name, direction, entry_price, qty, tp_orders, sl_result,
            event_type, timeframe, score, setup, requested_entry_price, entry_ts_ms,
        )


def _register_active_trade_locked(
    event_id: str,
    symbol: str,
    name: str,
    direction: str,
    entry_price: float,
    qty: float,
    tp_orders: list[dict],
    sl_result: dict,
    event_type: str,
    timeframe: str | None = None,
    score: float = 50.0,
    setup: dict | None = None,
    requested_entry_price: float | None = None,
    entry_ts_ms: int | None = None,
) -> bool:
    direction = _normalize_direction(direction)
    # Registration happens only after a real exchange position exists and
    # protection has been verified. Missing local state here is therefore not a
    # fresh-start condition: it is an ownership-integrity failure that must force
    # the caller's verified rollback path.
    trades = _load_active_trades_required()
    now_ms = int(time.time() * 1000)
    actual_entry_ts = int(entry_ts_ms) if entry_ts_ms is not None and int(entry_ts_ms) > 0 else now_ms

    actual_entry_price = _safe_float(entry_price)
    actual_qty = abs(_safe_float(qty))

    if actual_entry_price <= 0 or actual_qty <= 0:
        raise ValueError(f"Cannot register invalid position: entry_price={actual_entry_price} qty={actual_qty}")

    existing_same_event = trades.get(str(event_id))
    if isinstance(existing_same_event, dict) and not existing_same_event.get("closed", False):
        telemetry.record_state_conflict(
            event_id=event_id, attempt_id=(setup or {}).get("attempt_id") if isinstance(setup, dict) else None,
            position_id=event_id, order_id=None, symbol=symbol, direction=direction,
            conflict_type="LOCAL_EVENT_ID_DUPLICATE",
            message="active local event_id already exists; refusing overwrite",
            owner_event_ids=[str(event_id)],
        )
        log.error("[TRACKER_STATE_CONFLICT] refusing overwrite of active event=%s", event_id)
        return False

    conflicts = _active_trade_conflicts(trades, symbol, direction, exclude_event_id=event_id)
    if conflicts:
        conflict_ids = [eid for eid, _ in conflicts]
        telemetry.record_state_conflict(
            event_id=event_id, attempt_id=(setup or {}).get("attempt_id") if isinstance(setup, dict) else None,
            position_id=event_id, order_id=None, symbol=symbol, direction=direction,
            conflict_type="LOCAL_ACTIVE_TRADE_DUPLICATE",
            message=f"active local trade already owns {symbol}:{direction}; conflicts={conflict_ids}",
            owner_event_ids=conflict_ids,
        )
        log.error("[TRACKER_STATE_CONFLICT] refusing duplicate active owner event=%s conflicts=%s %s %s", event_id, conflict_ids, symbol, direction)
        return False

    setup_metrics = _extract_setup_metrics(setup)
    signal_snapshot = setup_metrics.get("signal_snapshot") if isinstance(setup_metrics.get("signal_snapshot"), dict) else {}
    signal_source = signal_snapshot.get("market_snapshot", {}) if isinstance(signal_snapshot.get("market_snapshot"), dict) else {}
    setup_source = (setup or {}) if isinstance(setup, dict) else {}
    research_source = setup_source.get("analysis_source") or signal_source.get("analysis_source") or setup_source.get("market_source")
    research: dict[str, Any] = {
        "source": str(research_source or "runtime_market_data"),
        "analysis_provider": signal_source.get("analysis_provider"),
        "analysis_source": signal_source.get("analysis_source"),
    }
    if isinstance(setup, dict) and isinstance(setup.get("signal_forensics"), dict):
        research["signal_forensics"] = dict(setup.get("signal_forensics") or {})

    requested_price = _safe_float(requested_entry_price, 0.0) if requested_entry_price is not None else setup_metrics["entry_reference"]
    signal_reference_price = _safe_float((setup or {}).get("signal_price"), 0.0) if isinstance(setup, dict) else 0.0
    if signal_reference_price <= 0:
        signal_reference_price = _safe_float(setup_metrics.get("entry_reference"), 0.0)
    pre_order_reference_price = _safe_float((setup or {}).get("execution_reference_price"), 0.0) if isinstance(setup, dict) else 0.0
    signal_drift_pct = _safe_float((setup or {}).get("signal_drift_pct"), 0.0) if isinstance(setup, dict) else 0.0
    execution_slippage_pct = (setup or {}).get("execution_slippage_pct") if isinstance(setup, dict) else None
    execution_slippage_pct = _safe_float(execution_slippage_pct, 0.0) if execution_slippage_pct is not None else None

    # From this version onward, entry_slippage_pct means actual MARKET fill
    # versus the top-of-book price captured immediately before the order. The
    # old implementation compared the fill to the stale signal price and mixed
    # signal drift with execution slippage.
    entry_slippage_pct = None
    adverse_entry_slippage_pct = None
    if pre_order_reference_price > 0:
        entry_slippage_pct = (actual_entry_price - pre_order_reference_price) / pre_order_reference_price * 100.0
        if direction == "LONG":
            adverse_entry_slippage_pct = max(0.0, entry_slippage_pct)
        else:
            adverse_entry_slippage_pct = max(0.0, -entry_slippage_pct)
    if execution_slippage_pct is not None:
        adverse_entry_slippage_pct = execution_slippage_pct

    signal_snapshot = setup_metrics.get("signal_snapshot") if isinstance(setup_metrics.get("signal_snapshot"), dict) else {}
    entry_bar = setup_metrics.get("entry_bar") if isinstance(setup_metrics.get("entry_bar"), dict) else {}
    previous_bar = setup_metrics.get("previous_bar") if isinstance(setup_metrics.get("previous_bar"), dict) else {}
    trades[event_id] = {
        "event_id": event_id,
        "attempt_id": (setup or {}).get("attempt_id") if isinstance(setup, dict) else None,
        "position_id": event_id,
        "symbol": symbol,
        "name": name or symbol,
        "direction": direction,
        "entry_price": actual_entry_price,
        "actual_entry_price": actual_entry_price,
        "requested_entry_price": requested_price,
        "signal_reference_price": signal_reference_price if signal_reference_price > 0 else None,
        "pre_order_reference_price": pre_order_reference_price if pre_order_reference_price > 0 else requested_price,
        "signal_drift_pct": signal_drift_pct,
        "execution_slippage_pct": execution_slippage_pct,
        "entry_slippage_pct": entry_slippage_pct,
        "signed_entry_slippage_pct": entry_slippage_pct,
        "adverse_entry_slippage_pct": adverse_entry_slippage_pct,
        "initial_qty": actual_qty,
        "remaining_qty": actual_qty,
        "entry_ts": actual_entry_ts,
        "tp_orders": tp_orders if isinstance(tp_orders, list) else [],
        "sl_order": sl_result if isinstance(sl_result, dict) else {},
        "hit_legs": [],
        "be_activated": False,
        "be_activation_ts": None,
        "be_trigger_ts": None,
        "be_trigger_rule": "after_tp1_filled",
        "be_trigger_peak_r": None,
        "be_order_id": None,
        "be_trigger_price": None,
        "be_fill_price": None,
        "be_execution_slippage_pct": None,
        "mfe_milestones_r": {},
        "be_required": False,
        "be_last_error": None,
        "peak_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "current_pnl_pct": 0.0,
        "score": _safe_float(score, 50.0),
        "strategy_version": setup_metrics.get("strategy_version"),
        "code_commit_sha": setup_metrics.get("code_commit_sha"),
        "entry_bar": entry_bar,
        "previous_bar": previous_bar,
        "signal_snapshot": signal_snapshot,
        "entry_order": setup_metrics.get("entry_order", {}),
        "fill_position": setup_metrics.get("fill_position", {}),
        "execution_snapshot": setup_metrics.get("execution_snapshot", {}),
        "protection_status": (
            ((setup or {}).get("execution_snapshot") or {}).get("protection_status")
            if isinstance((setup or {}).get("execution_snapshot"), dict)
            else None
        ),
        "tp_fill_events": [],
        "event_type": event_type,
        "timeframe": str(timeframe or (setup or {}).get("event_timeframe") or (setup or {}).get("timeframe") or "1h").lower(),
        "research": research,
        "setup": setup.copy() if isinstance(setup, dict) else {},
        "planned_risk_pct": setup_metrics["planned_risk_pct"],
        "planned_target_rr": setup_metrics["planned_target_rr"],
        "planned_weighted_rr": setup_metrics["planned_weighted_rr"],
        "planned_entry_reference": setup_metrics["entry_reference"],
        "planned_invalidation_price": setup_metrics["invalidation_price"],
        "planned_target_price": setup_metrics["target_price"],
        "tp_levels": setup_metrics["tp_levels"],
        "effective_tp_levels": setup_metrics["effective_tp_levels"],
        "tp_mode": setup_metrics["tp_mode"],
        "effective_weighted_rr": setup_metrics["effective_weighted_rr"],
        "tp_filled_qty": {},
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "last_tp_exec_price": None,
        "last_close_exec_price": None,
        "realized_pnl_pct": None,
        "realized_rr": None,
        "exit_price": None,
        "exit_reason": None,
        "closed_ts": None,
        "duration_min": None,
        "closed": False,
        "last_observation_ts": now_ms,
    }

    new_order_ids = _collect_order_ids(trades[event_id])
    raw_new_order_ids: list[str] = []
    for container in (
        trades[event_id].get("entry_order") if isinstance(trades[event_id].get("entry_order"), dict) else {},
        trades[event_id].get("sl_order") if isinstance(trades[event_id].get("sl_order"), dict) else {},
    ):
        for key in ("order_id", "orderId"):
            if container.get(key):
                raw_new_order_ids.append(str(container[key]))
    for tp in trades[event_id].get("tp_orders", []) if isinstance(trades[event_id].get("tp_orders"), list) else []:
        if not isinstance(tp, dict):
            continue
        for key in ("order_id", "orderId"):
            if tp.get(key):
                raw_new_order_ids.append(str(tp[key]))
    duplicate_new_ids = sorted({oid for oid in raw_new_order_ids if raw_new_order_ids.count(oid) > 1})
    if duplicate_new_ids:
        telemetry.record_state_conflict(
            event_id=event_id, attempt_id=trades[event_id].get("attempt_id"), position_id=event_id,
            order_id=duplicate_new_ids[0], symbol=symbol, direction=direction,
            conflict_type="ORDER_OWNERSHIP_CONFLICT",
            message=f"one local event references the same exchange order more than once: {duplicate_new_ids}",
            owner_event_ids=[str(event_id)], conflicting_order_ids=duplicate_new_ids,
        )
        del trades[event_id]
        log.error("[TRACKER_STATE_CONFLICT] duplicate order ids inside event=%s order_ids=%s", event_id, duplicate_new_ids)
        return False

    for existing_event_id, existing_trade in trades.items():
        if existing_event_id == event_id or existing_trade.get("closed", False):
            continue
        existing_ids = _collect_order_ids(existing_trade)
        overlap = sorted(new_order_ids & existing_ids)
        if overlap:
            telemetry.record_state_conflict(
                event_id=event_id, attempt_id=trades[event_id].get("attempt_id"), position_id=event_id,
                order_id=overlap[0], symbol=symbol, direction=direction,
                conflict_type="ORDER_OWNERSHIP_CONFLICT",
                message=f"exchange order id already belongs to another local event: {overlap}",
                owner_event_ids=[str(existing_event_id), str(event_id)],
                conflicting_order_ids=overlap,
            )
            del trades[event_id]
            log.error("[TRACKER_STATE_CONFLICT] order ownership conflict event=%s other=%s order_ids=%s", event_id, existing_event_id, overlap)
            return False

    _write_active_trades_unlocked(trades)
    return True


def format_tp_hit_message(
    name: str, symbol: str, leg: str, pnl_pct: float,
    exec_price: float, closed_qty: float, remaining_qty: float, remaining_pct: float,
) -> str:
    return (
        f"💰 <b>{_display_symbol(symbol)}</b>\n\n"
        f"Leg: <b>{leg}</b>\n"
        f"PnL TP: <b>+{pnl_pct:.2f}%</b>\n"
        f"Цена исполнения: <code>{exec_price:.8g}</code>\n"
        f"Закрыто: <code>{closed_qty:.8f}</code>\n"
        f"Осталось: <code>{remaining_qty:.8f} ({remaining_pct:.1f}%)</code>"
    )


def format_trade_closed_message(
    name: str, symbol: str, direction: str, entry_price: float, exit_price: float,
    pnl_pct: float | None, realized_rr: float | None, planned_rr: float | None,
    duration_min: float, peak_pnl: float, max_drawdown: float,
    exit_reason: str, event_type: str, timeframe: str = "1h",
) -> str:
    is_confirmed = pnl_pct is not None
    is_win = is_confirmed and pnl_pct >= 0.0
    emoji = "💚" if is_win else "💔" if is_confirmed else "⚪"
    pnl_sign = "+" if is_confirmed and pnl_pct > 0 else ""
    pnl_text = f"{pnl_sign}{pnl_pct:.2f}%" if is_confirmed else "не подтверждён"
    realized_rr_text = f"{realized_rr:.3f}" if realized_rr is not None else "—"
    planned_rr_text = f"{planned_rr:.3f}" if planned_rr is not None else "—"

    lines = [
        f"{emoji} <b>{_display_symbol(symbol)} — сделка закрыта</b>",
        "",
        f"Вход <code>{entry_price:.8g}</code> → Выход <code>{exit_price:.8g}</code>   <b>{pnl_text}</b>",
        f"Realized R:R: <b>{realized_rr_text}</b> · Planned Weighted R:R: <b>{planned_rr_text}</b>",
        f"Держали <b>{duration_min:.1f} мин</b> · пик <b>+{peak_pnl:.2f}%</b> · просадка <b>{max_drawdown:.2f}%</b>",
        f"Вход: <code>{event_type}</code> · TF <b>{str(timeframe or '1h').lower()}</b>",
        f"Выход: <b>{exit_reason}</b>",
    ]

    return "\n".join(lines)


def _cancel_old_sl_verified(symbol: str, direction: str, order_id: str, max_attempts: int = 3, *, owner_event_id: str | None = None) -> tuple[bool, str]:
    """Cancel an SL order and verify it actually disappeared (audit fix B4).

    Returns (cancelled, message). After failing DELETE attempts we re-query
    open orders: some exchanges answer an already-executed/expired cancel with
    an error code while the order is in fact gone.
    """
    last_error = ""
    # Resolve ownership before DELETE. Never cancel a manual/user SL merely
    # because its ID was copied into local trade state by an older version.
    try:
        current = get_open_protection_directional(symbol, direction)
    except Exception as exc:
        return False, f"ownership preflight failed: {exc}"
    if current.get("status") != "ok":
        return False, f"ownership preflight unavailable: {current.get('error', 'openOrders unavailable')}"
    matched = next((o for o in current.get("sl_orders", []) if str(o.get("orderId", "")) == str(order_id)), None)
    if matched is None:
        return True, "already_gone_from_open_orders"
    if not str(matched.get("clientOrderId", "")).upper().startswith("EVT_"):
        return False, "refusing to cancel non-engine/manual SL"
    try:
        active = _load_active_trades_required()
        owners = _order_ids_owned_by_other_trades(active, {str(order_id)}, exclude_event_id=str(owner_event_id or ""))
    except Exception as exc:
        return False, f"ownership state unavailable: {exc}"
    if owners:
        return False, f"refusing to cancel SL owned by other active trade: {owners}"
    for attempt in range(max_attempts):
        try:
            resp = cancel_order(symbol, order_id)
        except Exception as exc:
            resp = {"code": -1, "msg": str(exc)}
        if isinstance(resp, dict) and resp.get("code") in (0, "0"):
            try:
                after = get_open_protection_directional(symbol, direction)
            except Exception as exc:
                after = {"status": "error", "error": str(exc)}
            if after.get("status") == "ok":
                open_ids = {str(o.get("orderId", "")) for o in (after.get("sl_orders", []) + after.get("tp_orders", []))}
                if str(order_id) not in open_ids:
                    return True, "cancelled_verified_absent"
                last_error = "cancel acknowledged but SL remains visible in openOrders"
            else:
                last_error = f"cancel acknowledged but openOrders verification failed: {after.get('error', 'unknown')}"
        else:
            last_error = str(resp.get("msg", resp)) if isinstance(resp, dict) else str(resp)
        if attempt + 1 < max_attempts:
            time.sleep(0.3 * (attempt + 1))

    try:
        prot = get_open_protection_directional(symbol, direction)
        if prot.get("status") == "ok":
            open_ids = {str(o.get("orderId", "")) for o in (prot.get("sl_orders", []) + prot.get("tp_orders", []))}
            if str(order_id) not in open_ids:
                return True, "already_gone_from_open_orders"
    except Exception as exc:
        last_error = f"{last_error}; openOrders verify failed: {exc}"

    return False, last_error


def _notify_be_failure(symbol: str, direction: str, detail: str, *, event_id: str = "", rollback: dict | None = None) -> None:
    """Alert on BE failure with an exact, non-misleading rollback state."""
    rollback = rollback or {}
    status = str(rollback.get("status") or "").lower()
    if status in {"closed_verified", "already_closed"}:
        suffix = "Позиция подтверждённо закрыта биржей."
    elif status == "close_unverified":
        remaining = rollback.get("remaining_qty")
        last_known = rollback.get("last_known_qty")
        if remaining is not None:
            suffix = (
                f"Закрытие не подтверждено биржей; подтверждённый остаток позиции: <code>{_safe_float(remaining):.8f}</code>. "
                "Требуется следующее reconciliation."
            )
        elif last_known is not None:
            suffix = (
                "Закрытие не подтверждено биржей; текущий остаток неизвестен. "
                f"Последний известный объём: <code>{_safe_float(last_known):.8f}</code>. "
                "Требуется следующее reconciliation."
            )
        else:
            suffix = "Закрытие не подтверждено биржей; текущий остаток неизвестен. Требуется следующее reconciliation."
    else:
        suffix = "Состояние rollback требует reconciliation; закрытие не считается подтверждённым."
    text = (
        f"🛑 <b>BE move failed ({_display_symbol(symbol)} {direction})</b>\n"
        f"Status: <code>{detail}</code>\n"
        f"{suffix}"
    )
    _send_tracker_notification("BE_FAILURE", event_id, text, symbol=symbol)


def _cancel_engine_protection_before_emergency_close(symbol: str, direction: str) -> dict:
    """Cancel engine-owned SL/TP orders before a MARKET safety rollback."""
    result = {"status": "ok", "cancelled": [], "errors": []}
    try:
        existing = get_open_protection_directional(symbol, direction)
    except Exception as exc:
        return {"status": "error", "cancelled": [], "errors": [str(exc)]}
    if existing.get("status") != "ok":
        return {"status": "error", "cancelled": [], "errors": [existing.get("error", "openOrders unavailable")]}
    for order in list(existing.get("sl_orders", [])) + list(existing.get("tp_orders", [])):
        oid = str(order.get("orderId", ""))
        cid = str(order.get("clientOrderId", "")).upper()
        if not oid or not cid.startswith("EVT_"):
            continue
        try:
            resp = cancel_order(symbol, oid)
        except Exception as exc:
            result["errors"].append(f"{oid}: {exc}")
            continue
        if isinstance(resp, dict) and resp.get("code") in (0, "0"):
            result["cancelled"].append(oid)
        else:
            result["errors"].append(f"{oid}: {resp}")
    if result["errors"]:
        result["status"] = "partial" if result["cancelled"] else "error"
    return result


def _emergency_close_after_be_failure(symbol: str, direction: str, qty: float, trade_id: str | None) -> dict:
    """Close and verify a position after BE protection cannot be proven.

    The MARKET close POST is never blindly retried. After an accepted POST we
    reconcile the order and the live position. A transient verification error
    must never be reported as a numeric "remaining_qty": that number would only
    be the last known quantity, not proof that the position still exists.
    """
    attempts: list[dict] = []
    polls = max(4, int(os.environ.get("BE_FAILURE_CLOSE_POLLS", "12")))
    poll_delay = max(0.15, float(os.environ.get("BE_FAILURE_CLOSE_POLL_SEC", "0.5")))

    def _extract_directional_from_positions(rows: list[dict]) -> dict:
        wanted = str(direction).upper()
        # Rows returned by get_positions() already contain the BingX symbol. Do
        # not call to_bx_symbol() here: that helper can refresh contracts over the
        # network and would turn a supposedly local fallback reconciliation into
        # another network dependency during an emergency path.
        bx = str(symbol or "").strip().upper().replace("-", "").replace("_", "")
        for index, p in enumerate(rows or []):
            if not isinstance(p, dict):
                return {"status": "error", "error": f"full positions row {index} is not an object", "symbol": bx, "positionSide": wanted}
            row_symbol = str(p.get("symbol", "")).strip().upper().replace("-", "").replace("_", "")
            if not row_symbol:
                return {"status": "error", "error": f"full positions row {index} is missing symbol", "positionSide": wanted}
            if row_symbol != bx:
                continue
            side = str(p.get("positionSide", "")).upper()
            if side not in {"LONG", "SHORT", "BOTH"}:
                return {"status": "error", "error": f"full positions row for {bx} has unrecognized positionSide={side!r}", "symbol": bx, "positionSide": wanted}
            # A separately labelled opposite hedge leg cannot satisfy the
            # requested position. Validate the target leg strictly before ever
            # reporting it absent.
            if side in {"LONG", "SHORT"} and side != wanted:
                continue
            try:
                raw_value = p.get("positionAmt")
                if raw_value in (None, ""):
                    raise ValueError("positionAmt is missing")
                raw_amt = float(raw_value)
                raw_avg = p.get("avgPrice") or p.get("entryPrice")
                avg_price = float(raw_avg or 0)
            except (TypeError, ValueError) as exc:
                return {"status": "error", "error": f"malformed full-position row for {bx}: {exc}", "symbol": bx, "positionSide": wanted}
            if not math.isfinite(raw_amt) or not math.isfinite(avg_price):
                return {"status": "error", "error": f"non-finite full-position values for {bx}", "symbol": bx, "positionSide": wanted}
            if side == "BOTH":
                if wanted == "LONG" and raw_amt < 0:
                    continue
                if wanted == "SHORT" and raw_amt > 0:
                    continue
            qty_abs = abs(raw_amt)
            if qty_abs <= 0:
                continue
            if avg_price <= 0:
                return {"status": "error", "error": f"open full-position row for {bx} has no valid entry price", "symbol": bx, "positionSide": wanted}
            try:
                entry_price = float(p.get("entryPrice") or avg_price)
            except (TypeError, ValueError):
                entry_price = 0.0
            if not math.isfinite(entry_price) or entry_price <= 0:
                entry_price = avg_price
            return {
                "status": "found",
                "symbol": p.get("symbol", bx),
                "positionSide": wanted,
                "positionAmt": qty_abs,
                "avgPrice": avg_price,
                "entryPrice": entry_price,
            }
        return {"status": "not_found", "symbol": bx, "positionSide": wanted}

    def _authoritative_position_read() -> dict:
        """Use directional read first, then full position list as independent parser path."""
        try:
            direct = get_position_directional(symbol, direction)
        except Exception as exc:
            direct = {"status": "error", "error": str(exc)}
        if direct.get("status") in {"found", "not_found"}:
            return direct

        try:
            rows = get_positions(timeout_sec=float(os.environ.get("RECONCILIATION_HTTP_TIMEOUT_SEC", "5")))
            return _extract_directional_from_positions(rows)
        except Exception as exc:
            return {
                "status": "error",
                "error": f"directional={direct.get('error', 'unknown')}; full_positions={exc}",
            }

    def _finalize_verified_flat(payload: dict) -> dict:
        """Remove engine-owned residual orders only after exchange flatness is verified."""
        try:
            cleanup = _cancel_engine_protection_before_emergency_close(symbol, direction)
        except Exception as exc:
            cleanup = {"status": "error", "cancelled": [], "errors": [str(exc)]}
        payload["cleanup_after_flat_verified"] = cleanup
        if cleanup.get("status") != "ok":
            log.error("[TRACKER_BE_CLOSE_CLEANUP_PENDING] %s %s cleanup=%s", symbol, direction, cleanup)
        return payload

    initial = _authoritative_position_read()
    if initial.get("status") == "not_found":
        return _finalize_verified_flat({"status": "already_closed", "attempts": attempts, "remaining_qty": 0.0, "last_known_qty": 0.0})
    if initial.get("status") != "found":
        return {
            "status": "close_unverified",
            "error": initial.get("error", "position state unavailable"),
            "attempts": attempts,
            "remaining_qty": None,
            "last_known_qty": max(0.0, float(qty or 0.0)),
        }

    current_qty = abs(_safe_float(initial.get("positionAmt"), qty))
    if current_qty <= 0:
        return _finalize_verified_flat({"status": "already_closed", "attempts": attempts, "remaining_qty": 0.0, "last_known_qty": 0.0})

    # Keep the verified old SL/TP active while the emergency MARKET close is
    # in flight. Cancelling first creates an avoidable naked-position window if
    # the close POST or its reconciliation fails.
    try:
        close_result = close_position_market(symbol, direction, current_qty, trade_id=trade_id)
    except Exception as exc:
        close_result = {"status": "error", "error": str(exc)}
    attempts.append(close_result)

    accepted_order_id = None
    if isinstance(close_result, dict):
        response = close_result.get("response") or {}
        order = (response.get("data") or {}).get("order") or response.get("data") or {}
        accepted_order_id = order.get("orderId") or close_result.get("order_id")

    last_known_qty = current_qty
    last_verification: dict = {"status": "error", "error": "verification not attempted"}
    order_filled = False
    for poll in range(polls):
        if accepted_order_id:
            try:
                order_info = get_order(symbol, accepted_order_id)
            except Exception as exc:
                order_info = {"status": "error", "error": str(exc)}
            if order_info.get("status") == "ok" and str(order_info.get("order_status", "")).upper() in {"FILLED", "PARTIALLY_FILLED"}:
                executed_qty = _safe_float(order_info.get("executed_qty"), 0.0)
                order_filled = executed_qty >= max(0.0, current_qty - 1e-12)
                attempts.append({"verification": "order", "poll": poll + 1, "order": order_info})

        verification = _authoritative_position_read()
        last_verification = verification
        if verification.get("status") == "not_found":
            return _finalize_verified_flat({
                "status": "closed_verified",
                "attempts": attempts,
                "verification": verification,
                "verify_poll": poll + 1,
                "remaining_qty": 0.0,
                "last_known_qty": 0.0,
                "order_filled": order_filled,
            })
        if verification.get("status") == "found":
            remaining = abs(_safe_float(verification.get("positionAmt"), last_known_qty))
            last_known_qty = remaining
            if remaining <= 1e-12 and (order_filled or accepted_order_id):
                return _finalize_verified_flat({
                    "status": "closed_verified",
                    "attempts": attempts,
                    "verification": verification,
                    "verify_poll": poll + 1,
                    "remaining_qty": 0.0,
                    "last_known_qty": 0.0,
                    "order_filled": order_filled,
                })
        time.sleep(poll_delay)

    final_verification = _authoritative_position_read()
    if final_verification.get("status") == "not_found":
        return _finalize_verified_flat({
            "status": "closed_verified",
            "attempts": attempts,
            "verification": final_verification,
            "remaining_qty": 0.0,
            "last_known_qty": 0.0,
            "order_filled": order_filled,
        })
    if final_verification.get("status") == "found":
        confirmed_qty = abs(_safe_float(final_verification.get("positionAmt"), last_known_qty))
        return {
            "status": "close_unverified",
            "attempts": attempts,
            "verification": final_verification,
            "remaining_qty": confirmed_qty,
            "last_known_qty": confirmed_qty,
            "order_filled": order_filled,
        }

    return {
        "status": "close_unverified",
        "attempts": attempts,
        "verification": final_verification or last_verification,
        # Do not expose the stale quantity as if it were current exchange truth.
        "remaining_qty": None,
        "last_known_qty": last_known_qty,
        "order_filled": order_filled,
    }

def _is_break_even_stop_shape_valid(
    order: dict,
    direction: str,
    entry_price: float,
    qty: float,
    price_precision: int,
) -> bool:
    """Check whether an exchange stop has BE geometry, independent of ownership.

    This predicate is for conflict detection only. Passing it does not mean the
    engine may adopt or cancel the order; identity and owner checks remain
    mandatory in `_is_valid_break_even_stop` and in the reconciliation path.
    """
    if not isinstance(order, dict):
        return False
    order_id = str(order.get("orderId") or order.get("order_id") or "")
    if not order_id or not _validate_sl_order_for_position(order, direction, entry_price, qty):
        return False
    stop_price = _safe_float(order.get("stopPrice") or order.get("price"), 0.0)
    tolerance = max(0.5 * (10 ** (-max(0, int(price_precision)))), entry_price * 1e-8)
    if abs(stop_price - entry_price) > tolerance:
        return False
    try:
        expected_position_side = position_side_param(direction)
    except Exception:
        return False
    if str(order.get("positionSide", "")).upper() != expected_position_side:
        return False
    expected_side = "SELL" if str(direction).upper() == "LONG" else "BUY"
    if str(order.get("side", "")).upper() != expected_side:
        return False
    return True


def _is_valid_break_even_stop(
    order: dict,
    direction: str,
    entry_price: float,
    qty: float,
    price_precision: int,
    *,
    expected_client_order_id: str | None = None,
) -> bool:
    """Validate a BE stop's engine ownership, identity, trigger and quantity.

    A user-created stop at the entry price is not proof that the engine's BE
    transition completed. Only an EVT_BE_* order belonging to this event may
    be adopted as the engine's active BE protection.
    """
    if not isinstance(order, dict):
        return False
    client_order_id = str(order.get("clientOrderId") or order.get("client_order_id") or "").upper()
    if not client_order_id.startswith("EVT_BE_"):
        return False
    if expected_client_order_id and not _client_order_id_matches_base(
        client_order_id, expected_client_order_id
    ):
        return False
    return _is_break_even_stop_shape_valid(
        order, direction, entry_price, qty, price_precision
    )


def _cancel_conflicted_be_if_old_sl_verified(
    symbol: str,
    direction: str,
    entry_price: float,
    qty: float,
    old_sl_id: str | None,
    be_order_id: str,
    be_client_id: str,
    price_precision: int,
) -> dict[str, Any]:
    """Best-effort cleanup of this event's just-created BE order on owner conflict.

    The new BE is cancelled only when its exact order ID + unique client ID are
    visible AND a separate, quantity-covering old SL is verified both before
    and after cancellation. If that proof is unavailable, leave exchange-side
    protection untouched and let the normal BE retry/reconciliation surface it.
    """
    if not old_sl_id:
        return {"verified": False, "status": "deferred", "error": "old SL id unavailable"}

    def _find_protection(snapshot: dict[str, Any]) -> tuple[dict | None, dict | None]:
        all_orders = list(snapshot.get("sl_orders", [])) + list(snapshot.get("tp_orders", []))
        candidate = next((
            order for order in all_orders
            if str(order.get("orderId") or order.get("order_id") or "") == str(be_order_id)
            and str(order.get("clientOrderId") or "").upper() == str(be_client_id).upper()
        ), None)
        old = next((
            order for order in snapshot.get("sl_orders", [])
            if str(order.get("orderId") or order.get("order_id") or "") == str(old_sl_id)
        ), None)
        return candidate, old

    def _old_sl_covers(old: dict | None) -> bool:
        if not isinstance(old, dict) or not _validate_sl_order_for_position(old, direction, entry_price, None):
            return False
        old_qty = _safe_float(old.get("origQty") or old.get("quantity"), 0.0)
        return old_qty > 0 and old_qty + max(qty * 0.01, 1e-12) >= qty

    try:
        before = get_open_protection_directional(symbol, direction)
    except Exception as exc:
        return {"verified": False, "status": "deferred", "error": f"pre-cancel openOrders failed: {exc}"}
    if before.get("status") != "ok":
        return {"verified": False, "status": "deferred", "error": f"pre-cancel openOrders unavailable: {before.get('error', 'unknown')}"}
    new_order, old_sl = _find_protection(before)
    if not _old_sl_covers(old_sl):
        return {"verified": False, "status": "deferred", "error": "old SL not independently verified to cover the live residual"}
    if new_order is None:
        return {"verified": True, "status": "new_BE_already_absent", "error": None}
    if not _is_valid_break_even_stop(
        new_order, direction, entry_price, qty, price_precision,
        expected_client_order_id=be_client_id,
    ):
        return {"verified": False, "status": "deferred", "error": "matched client ID did not pass BE validation"}

    try:
        cancel_order(symbol, str(be_order_id))
    except Exception as exc:
        cancel_error = f"cancel request raised: {type(exc).__name__}: {exc}"
    else:
        cancel_error = ""

    try:
        after = get_open_protection_directional(symbol, direction)
    except Exception as exc:
        return {"verified": False, "status": "cleanup_pending", "error": f"post-cancel verification failed: {exc}"}
    if after.get("status") != "ok":
        return {"verified": False, "status": "cleanup_pending", "error": f"post-cancel openOrders unavailable: {after.get('error', 'unknown')}"}
    remaining_be, old_sl_after = _find_protection(after)
    if remaining_be is None and _old_sl_covers(old_sl_after):
        return {"verified": True, "status": "cancelled_verified", "error": cancel_error or None}
    return {
        "verified": False,
        "status": "cleanup_pending",
        "error": cancel_error or "BE order remains visible or old SL no longer passes verification",
    }


def _move_sl_to_break_even(
    symbol: str, direction: str, entry_price: float, qty: float, old_sl_id: str | None, trade_id: str | None = None,
    old_sl_price: float | None = None, owner_event_id: str | None = None,
) -> dict:
    """Move the stop-loss to break-even without leaving an unprotected window.

    A verified replacement may briefly coexist with the old stop while cleanup
    is pending; avoiding an unprotected position takes priority over temporary
    duplicate protective orders. The exchange-visible sequence is:

      Step 1: if a BE SL already exists -> keep it, then remove any old SL.
      Step 2: create the new BE STOP_MARKET while the old SL still protects the
              position.
      Step 3: verify the new SL on the exchange.
      Step 4: cancel the old SL and verify that cancellation.
      Step 5: if old-SL cancellation cannot be proven, keep the verified BE SL
              (position remains protected) and report a cleanup issue; do not
              emergency-close a position merely because two valid protective
              stops briefly coexist.
      Step 6: if BE creation/verification fails, the old SL remains in place;
              only then use the emergency-close path if restoration cannot be
              proven.
    """
    direction = str(direction).upper()
    bx = to_bx_symbol(symbol)
    contract = get_contract(symbol) or {}

    try:
        precision = int(contract.get("quantityPrecision") or 0)
        price_precision = int(contract.get("pricePrecision") or 4)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc), "order_id": "", "stop_price": entry_price}

    if qty <= 0 or entry_price <= 0 or not bx:
        return {"status": "error", "error": "invalid BE parameters", "order_id": "", "stop_price": entry_price}

    trade_token = str(trade_id) if trade_id else uuid.uuid4().hex.upper()[:16]
    base_be_client_id = f"EVT_BE_{trade_token}"
    be_client_id = base_be_client_id

    verified = get_open_protection_directional(symbol, direction)
    if not isinstance(verified, dict) or verified.get("status") != "ok":
        # Do not post while the exchange order book cannot be inspected: the
        # canonical BE clientOrderId might still be occupied by a stale order.
        return {
            "status": "error",
            "error": f"cannot inspect existing protection before BE repair: {verified.get('error', 'openOrders unavailable') if isinstance(verified, dict) else 'invalid response'}",
            "order_id": "",
            "stop_price": entry_price,
            "safety_action": "existing_protection_unchanged",
        }
    # Detect BE-shaped engine stops already owned by another active event before
    # attempting a new POST. They must not be adopted or cancelled, but ignoring
    # them and posting another protection order can create overlapping stops and
    # violates the existing ownership-conflict guard.
    current_trades_for_conflict_check = None
    current_owner_id = str(owner_event_id or trade_id or "")
    for candidate in verified.get("sl_orders", []):
        candidate_id = str(candidate.get("orderId") or candidate.get("order_id") or "")
        candidate_cid = str(candidate.get("clientOrderId") or candidate.get("client_order_id") or "").upper()
        if (
            not candidate_id
            or not candidate_cid.startswith("EVT_")
            or _client_order_id_matches_base(candidate_cid, base_be_client_id)
            or not _is_break_even_stop_shape_valid(candidate, direction, entry_price, qty, price_precision)
        ):
            continue
        if current_trades_for_conflict_check is None:
            current_trades_for_conflict_check = _load_active_trades_required()
        foreign_owners = _order_ids_owned_by_other_trades(
            current_trades_for_conflict_check, {candidate_id}, exclude_event_id=current_owner_id
        )
        if not foreign_owners:
            continue
        telemetry.record_state_conflict(
            event_id=owner_event_id or trade_id, attempt_id=None, position_id=owner_event_id or trade_id,
            order_id=candidate_id, symbol=symbol, direction=direction,
            conflict_type="BE_ORDER_OWNERSHIP_CONFLICT",
            message=f"BE-shaped protection order is owned by another active event: {foreign_owners}; existing SL is kept",
            owner_event_ids=[current_owner_id] + sorted({eid for ids in foreign_owners.values() for eid in ids}),
            conflicting_order_ids=[candidate_id], leg="BE_SL",
        )
        return {
            "status": "error",
            "error": "BE-shaped protection order belongs to another active event",
            "order_id": candidate_id,
            "client_order_id": candidate_cid,
            "stop_price": entry_price,
            "safety_action": "old_sl_kept",
        }

    invalid_be_order_ids = [
        str(order.get("orderId", "")) for order in verified.get("sl_orders", [])
        if str(order.get("orderId", ""))
        and _client_order_id_matches_base(order.get("clientOrderId", ""), base_be_client_id)
        and not _is_valid_break_even_stop(
            order, direction, entry_price, qty, price_precision,
            expected_client_order_id=base_be_client_id,
        )
    ]
    if verified.get("status") == "ok":
        for order in verified.get("sl_orders", []):
            cid = str(order.get("clientOrderId", "")).upper()
            # A client ID is only an identity hint, not proof that this is a valid
            # BE stop. Price, order type, side/position-side and quantity must match.
            if _is_valid_break_even_stop(
                order, direction, entry_price, qty, price_precision,
                expected_client_order_id=base_be_client_id,
            ):
                existing_id = str(order.get("orderId", ""))
                if existing_id:
                    current_trades = _load_active_trades_required()
                    owners = _order_ids_owned_by_other_trades(
                        current_trades, {existing_id}, exclude_event_id=str(owner_event_id or trade_id or "")
                    )
                    if owners:
                        cleanup = (
                            _cancel_conflicted_be_if_old_sl_verified(
                                symbol, direction, entry_price, qty, old_sl_id, existing_id,
                                be_client_id, price_precision,
                            )
                            if cid == be_client_id.upper()
                            else {"verified": False, "status": "deferred", "error": "conflicting order is not identified by this event's unique client ID"}
                        )
                        conflict_ids = sorted(owners)
                        telemetry.record_state_conflict(
                            event_id=owner_event_id or trade_id, attempt_id=None, position_id=owner_event_id or trade_id,
                            order_id=existing_id, symbol=symbol, direction=direction, conflict_type="BE_ORDER_OWNERSHIP_CONFLICT",
                            message=f"existing BE stop is already owned by another active event: {owners}; cleanup={cleanup.get('status')}; old SL is kept",
                            owner_event_ids=[str(owner_event_id or trade_id or "")] + sorted({eid for ids in owners.values() for eid in ids}),
                            conflicting_order_ids=conflict_ids, leg="BE_SL",
                        )
                        return {
                            "status": "error",
                            "error": "existing BE order belongs to another active event",
                            "order_id": existing_id,
                            "client_order_id": cid or be_client_id,
                            "stop_price": entry_price,
                            "safety_action": "old_sl_kept",
                            "orphaned_order_cleanup_pending": bool(cid == be_client_id.upper() and not cleanup.get("verified")),
                            "orphaned_order_cleanup_status": cleanup.get("status"),
                            "orphaned_order_cleanup_error": cleanup.get("error"),
                        }
                for stale_be_id in invalid_be_order_ids:
                    if str(stale_be_id) == existing_id:
                        continue
                    stale_cancelled, stale_note = _cancel_old_sl_verified(
                        symbol, direction, str(stale_be_id), owner_event_id=owner_event_id or trade_id,
                    )
                    if not stale_cancelled:
                        return {
                            "status": "created_cleanup_pending",
                            "error": f"valid BE exists but stale BE cleanup failed: {stale_note}",
                            "order_id": existing_id,
                            "client_order_id": cid or be_client_id,
                            "stop_price": entry_price,
                            "old_sl_cleanup_pending": True,
                            "old_sl_cleanup_pending_order_id": str(stale_be_id),
                        }
                old_cancelled = True
                cancel_note = ""
                if old_sl_id and existing_id and str(old_sl_id) != existing_id:
                    old_cancelled, cancel_note = _cancel_old_sl_verified(symbol, direction, str(old_sl_id), owner_event_id=owner_event_id or trade_id)
                if not old_cancelled:
                    log.warning(
                        "[TRACKER] BE already in place for %s; old SL %s cleanup remains pending: %s. "
                        "The verified BE stays active while cleanup is retried.",
                        symbol, old_sl_id, cancel_note,
                    )
                    return {
                        "status": "created_cleanup_pending",
                        "error": f"BE exists but old SL cleanup is pending: {cancel_note}",
                        "order_id": existing_id,
                        "client_order_id": cid or be_client_id,
                        "stop_price": entry_price,
                        "old_sl_cleanup_pending": True,
                        "old_sl_cleanup_pending_order_id": str(old_sl_id),
                    }
                return {
                    "status": "created",
                    "order_id": existing_id,
                    "client_order_id": cid or be_client_id,
                    "stop_price": entry_price,
                }

    def _restore_old_sl() -> bool:
        """Best-effort restore of protection when the new SL could not be placed."""
        if not old_sl_price or old_sl_price <= 0 or abs(old_sl_price - entry_price) / max(entry_price, 1e-12) < 1e-9:
            return False
        restore_side = "SELL" if direction == "LONG" else "BUY"
        try:
            restore_params = {
                "symbol": bx,
                "side": restore_side,
                "positionSide": position_side_param(direction),
                "type": "STOP_MARKET",
                "stopPrice": _format_price(old_sl_price, price_precision),
                "quantity": _format_qty(qty, precision),
                "clientOrderId": f"EVT_BE_RST_{trade_token}",
            }
            _add_reduce_only_for_one_way(restore_params)
            restore_resp = _request("POST", ORDER_PATH, restore_params)
        except Exception:
            return False
        if not isinstance(restore_resp, dict) or restore_resp.get("code") != 0:
            return False
        try:
            verification = get_open_protection_directional(symbol, direction)
        except Exception:
            return False
        if verification.get("status") != "ok":
            return False
        restored = [
            order for order in verification.get("sl_orders", [])
            if str(order.get("orderId", "")) == str((restore_resp.get("data") or {}).get("order", {}).get("orderId", ""))
            or str(order.get("clientOrderId", "")).upper() == str(restore_params["clientOrderId"]).upper()
        ]
        return bool(restored) and _validate_sl_order_for_position(restored[0], direction, entry_price, qty)

    def _fail(error: str) -> dict:
        # With the safe create-new-first sequence the old SL was never cancelled
        # before a BE verification failure. First prove whether that old SL is
        # still present. If it is, keep it: posting another copy would create a
        # duplicate protection order and is unnecessary.
        old_sl_still_protected = False
        if old_sl_id:
            try:
                current_protection = get_open_protection_directional(symbol, direction)
            except Exception as exc:
                current_protection = {"status": "error", "error": str(exc)}
            if current_protection.get("status") == "ok":
                for sl in current_protection.get("sl_orders", []):
                    if str(sl.get("orderId", "")) != str(old_sl_id):
                        continue
                    sl_qty = _safe_float(sl.get("origQty") or sl.get("quantity"), 0.0)
                    # After TP1 the live position may be smaller than the original
                    # SL order quantity. The original SL still protects the residual
                    # position as long as it covers the residual size and is on the
                    # correct side of the market.
                    qty_covers_position = sl_qty > 0 and sl_qty + max(qty * 0.01, 1e-12) >= qty
                    if qty_covers_position and _validate_sl_order_for_position(sl, direction, entry_price, None):
                        old_sl_still_protected = True
                        break

        if old_sl_still_protected:
            log.error(
                "[TRACKER] BE move failed for %s %s: %s; original SL remains verified.",
                direction, symbol, error,
            )
            _notify_be_failure(symbol, direction, f"{error}; old_sl=verified", event_id=trade_id or "")
            return {
                "status": "error",
                "error": error,
                "order_id": "",
                "stop_price": entry_price,
                "old_sl_restored": True,
                "safety_action": "old_sl_kept",
            }

        # Old protection cannot be proven. Only now attempt restoration.
        restored = _restore_old_sl()
        log.error(
            "[TRACKER] BE move failed for %s %s: %s; old SL restore %s.",
            direction, symbol, error, "succeeded" if restored else "FAILED",
        )
        if restored:
            _notify_be_failure(symbol, direction, f"{error}; restore=ok", event_id=trade_id or "")
            return {
                "status": "error",
                "error": error,
                "order_id": "",
                "stop_price": entry_price,
                "old_sl_restored": True,
                "safety_action": "old_sl_restored",
            }

        # At this point the original SL is not proven to exist. The position is
        # therefore not allowed to remain live in an unverified protection state.
        rollback = _emergency_close_after_be_failure(symbol, direction, qty, trade_token)
        _notify_be_failure(symbol, direction, f"{error}; restore=failed; rollback={rollback.get('status')}", event_id=trade_id or "", rollback=rollback)
        return {
            "status": "error",
            "error": error,
            "order_id": "",
            "stop_price": entry_price,
            "old_sl_restored": False,
            "safety_action": "emergency_close",
            "rollback": rollback,
        }

    # Never retry the canonical BE ID while a malformed order using it remains
    # open. Keep the stale stop until a uniquely identified replacement is
    # verified on the exchange.
    be_client_id = _repair_client_order_id(base_be_client_id) if invalid_be_order_ids else base_be_client_id
    sl_side = "SELL" if direction == "LONG" else "BUY"
    params = {
        "symbol": bx,
        "side": sl_side,
        "positionSide": position_side_param(direction),
        "type": "STOP_MARKET",
        "stopPrice": _format_price(entry_price, price_precision),
        "quantity": _format_qty(qty, precision),
        "clientOrderId": be_client_id,
    }
    _add_reduce_only_for_one_way(params)

    try:
        resp = _request("POST", ORDER_PATH, params)
    except Exception as exc:
        return _fail(f"BE stop request exception: {exc}")

    if not isinstance(resp, dict) or resp.get("code") != 0:
        return _fail(f"BE stop failed: {resp}")

    order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
    new_order_id = str(order.get("orderId", ""))
    if not new_order_id:
        return _fail("BE stop response has no orderId")

    # BingX may acknowledge the order before it appears in open-orders. Poll
    # GET/openOrders rather than treating the first read as a definitive failure.
    found = False
    polls = max(3, int(os.environ.get("BE_VERIFY_POLLS", "5")))
    delay = max(0.10, float(os.environ.get("BE_VERIFY_POLL_SEC", "0.30")))
    for attempt in range(polls):
        try:
            verified_after = get_open_protection_directional(symbol, direction)
        except Exception as exc:
            verified_after = {"status": "error", "error": str(exc)}
        if verified_after.get("status") == "ok":
            if any(
                str(o.get("orderId", "")) == new_order_id
                and _is_valid_break_even_stop(
                    o, direction, entry_price, qty, price_precision,
                    expected_client_order_id=be_client_id,
                )
                and str(o.get("clientOrderId") or be_client_id).upper() == be_client_id.upper()
                for o in verified_after.get("sl_orders", [])
            ):
                found = True
                break
        if attempt + 1 < polls:
            time.sleep(delay * (attempt + 1))

    if not found:
        return _fail("BE stop not visible on exchange")

    # The BE stop is now proven to exist. Before touching the old SL, prove the
    # new exchange order is not already attributed to another active local event.
    # If ownership is ambiguous, keep the old SL and surface a state conflict.
    current_trades = _load_active_trades_required()
    owners = _order_ids_owned_by_other_trades(
        current_trades, {new_order_id}, exclude_event_id=str(owner_event_id or trade_id or "")
    )
    if owners:
        cleanup = _cancel_conflicted_be_if_old_sl_verified(
            symbol, direction, entry_price, qty, old_sl_id, new_order_id, be_client_id, price_precision,
        )
        conflict_ids = sorted(owners)
        telemetry.record_state_conflict(
            event_id=owner_event_id or trade_id, attempt_id=None, position_id=owner_event_id or trade_id,
            order_id=new_order_id, symbol=symbol, direction=direction, conflict_type="BE_ORDER_OWNERSHIP_CONFLICT",
            message=f"new BE order id collides with another active event: {owners}; cleanup={cleanup.get('status')}; old SL is kept",
            owner_event_ids=[str(owner_event_id or trade_id or "")] + sorted({eid for ids in owners.values() for eid in ids}),
            conflicting_order_ids=conflict_ids, leg="BE_SL",
        )
        return {
            "status": "error",
            "error": "new BE order id collides with another active event",
            "order_id": new_order_id,
            "client_order_id": order.get("clientOrderId") or be_client_id,
            "stop_price": entry_price,
            "safety_action": "old_sl_kept",
            "orphaned_order_cleanup_pending": not bool(cleanup.get("verified")),
            "orphaned_order_cleanup_status": cleanup.get("status"),
            "orphaned_order_cleanup_error": cleanup.get("error"),
        }

    # The BE stop is now proven to exist, so the position never becomes naked.
    # Clean malformed BE orders for this event only after the replacement has
    # been verified. An unresolved duplicate stays explicitly pending; it must
    # not trigger an emergency close while the new BE is already confirmed.
    for stale_be_id in invalid_be_order_ids:
        if str(stale_be_id) == new_order_id:
            continue
        stale_cancelled, stale_note = _cancel_old_sl_verified(
            symbol, direction, str(stale_be_id), owner_event_id=owner_event_id or trade_id,
        )
        if not stale_cancelled:
            return {
                "status": "created_cleanup_pending",
                "order_id": new_order_id,
                "client_order_id": order.get("clientOrderId") or be_client_id,
                "stop_price": entry_price,
                "error": f"new BE verified but stale BE cleanup failed: {stale_note}",
                "old_sl_cleanup_pending": True,
                "old_sl_cleanup_pending_order_id": str(stale_be_id),
            }

    # Remove the obsolete original SL only after the verified BE is in place.
    old_cleanup_error = None
    if old_sl_id and str(old_sl_id) != new_order_id:
        old_cancelled, cancel_note = _cancel_old_sl_verified(symbol, direction, str(old_sl_id), owner_event_id=owner_event_id or trade_id)
        if not old_cancelled:
            old_cleanup_error = f"old SL cleanup failed: {cancel_note}"
            log.error("[TRACKER] %s %s: BE verified but old SL cleanup failed: %s", direction, symbol, cancel_note)

    result = {
        "status": "created" if old_cleanup_error is None else "created_cleanup_pending",
        "order_id": new_order_id,
        "client_order_id": order.get("clientOrderId") or be_client_id,
        "stop_price": entry_price,
    }
    if old_cleanup_error:
        result["error"] = old_cleanup_error
        result["old_sl_cleanup_pending"] = True
        result["old_sl_cleanup_pending_order_id"] = str(old_sl_id) if old_sl_id else None
    return result


def _exit_outcome_category(exit_reason: str) -> str:
    """Separate strategy exits from technical/emergency outcomes in analytics."""
    reason = str(exit_reason or "").upper()
    if reason in {"STOP_LOSS", "TAKE_PROFIT_FULL", "BREAK_EVEN"}:
        return "STRATEGY_EXIT"
    if reason in {"POSITION_CLOSED_UNVERIFIED", "UNVERIFIED_CLOSE"}:
        return "UNVERIFIED_CLOSE"
    if reason in {"MANUAL_CLOSE_RECONCILED"}:
        return "POSITION_CLOSED_RECONCILED"
    if "EMERGENCY" in reason:
        return "EMERGENCY_EXIT"
    if "PROTECTION" in reason:
        return "PROTECTION_FAILURE"
    return "TECHNICAL_EXIT"


def _calc_trade_pnl_pct(entry_price: float, exit_price: float, direction: str) -> float:
    if entry_price <= 0 or exit_price <= 0:
        return 0.0
    if str(direction).upper() == "LONG":
        return (exit_price - entry_price) / entry_price * 100.0
    return (entry_price - exit_price) / entry_price * 100.0


def _weighted_rr_snapshot(trade: dict) -> tuple[float | None, float | None, float | None]:
    """Return (effective_total_rr, realized_rr_component, remaining_rr_component).

    effective_total_rr is the projected RR for the whole original position: realized
    legs use actual fills, while remaining legs use their planned target. This avoids
    reporting only the remaining leg after a partial TP.
    """
    init_qty = max(_safe_float(trade.get("initial_qty"), 0.0), 0.0)
    risk_pct = _derive_planned_risk_pct(trade)
    if init_qty <= 0 or risk_pct is None or risk_pct <= 0:
        return None, None, None
    realized_qty = max(_safe_float(trade.get("realized_pnl_qty"), 0.0), 0.0)
    realized_weighted = _safe_float(trade.get("realized_pnl_weighted_sum"), 0.0)
    realized_rr_component = (realized_weighted / init_qty) / risk_pct if realized_qty > 0 else 0.0

    hit_legs = {str(x) for x in (trade.get("hit_legs") or [])}
    completed_legs = {str(x) for x in (trade.get("completed_tp_legs") or [])}
    consumed_legs = hit_legs | completed_legs
    planned_tp_levels = trade.get("tp_levels") if isinstance(trade.get("tp_levels"), list) else []
    setup = trade.get("setup") if isinstance(trade.get("setup"), dict) else {}
    protection_result = setup.get("protection_result") if isinstance(setup.get("protection_result"), dict) else {}
    # Exchange-rounded quantities/fractions are authoritative for the remaining
    # targets. Legacy migration may seed effective_tp_levels with planned 50/50
    # targets, so prefer candidates that carry actual order quantities first.
    candidates = [
        trade.get("effective_tp_levels"),
        setup.get("effective_tp_levels"),
        protection_result.get("effective_tp_levels"),
    ]
    effective_candidates = [candidate for candidate in candidates if isinstance(candidate, list) and candidate]
    tp_levels = next((
        candidate for candidate in effective_candidates
        if any(isinstance(level, dict) and _safe_float(level.get("qty"), 0.0) > 0 for level in candidate)
    ), None)
    if tp_levels is None and effective_candidates:
        planned_fractions = {
            str(level.get("leg", "")): _safe_float(level.get("close_fraction"), 0.0)
            for level in planned_tp_levels if isinstance(level, dict)
        }
        tp_levels = next((
            candidate for candidate in effective_candidates
            if any(
                isinstance(level, dict)
                and str(level.get("leg", "")) in planned_fractions
                and abs(_safe_float(level.get("close_fraction"), 0.0) - planned_fractions[str(level.get("leg", ""))]) > 1e-9
                for level in candidate
            )
        ), None)
    if tp_levels is None:
        tp_levels = planned_tp_levels
    remaining_qty = max(_safe_float(trade.get("remaining_qty"), 0.0), 0.0)
    remaining_rr_component = 0.0
    if remaining_qty > 0 and tp_levels:
        remaining = [x for x in tp_levels if isinstance(x, dict) and str(x.get("leg", "")) not in consumed_legs and _safe_float(x.get("pnl_pct"), 0.0) > 0]
        if remaining:
            raw_weights = [
                max(0.0, _safe_float(x.get("qty"), 0.0))
                if _safe_float(x.get("qty"), 0.0) > 0
                else max(0.0, _safe_float(x.get("close_fraction"), 0.0))
                for x in remaining
            ]
            weight_sum = sum(raw_weights)
            if weight_sum <= 0:
                raw_weights = [1.0] * len(remaining)
                weight_sum = float(len(remaining))
            for level, raw_weight in zip(remaining, raw_weights):
                pnl_pct = _safe_float(level.get("pnl_pct"), 0.0)
                qty = remaining_qty * raw_weight / weight_sum
                remaining_rr_component += (qty / init_qty) * (pnl_pct / risk_pct)
    effective = realized_rr_component + remaining_rr_component
    remaining_rr = None
    if remaining_qty > 0:
        remaining_rr = remaining_rr_component / (remaining_qty / init_qty) if init_qty > 0 else None
    return effective, realized_rr_component, remaining_rr


def _derive_planned_risk_pct(trade: dict) -> float | None:
    direct = trade.get("planned_risk_pct")
    if direct is not None:
        val = _safe_float(direct, 0.0)
        if val > 0:
            return val

    setup = trade.get("setup")
    if isinstance(setup, dict):
        val = _safe_float(setup.get("risk_pct"), 0.0)
        if val > 0:
            return val

    return None


def _tp1_fill_timestamp_ms(trade: dict, fallback_ms: int) -> int:
    """Use confirmed TP1 fill/update time consistently in both BE paths."""
    events = trade.get("tp_fill_events") if isinstance(trade.get("tp_fill_events"), list) else []
    event = next((
        item for item in reversed(events)
        if isinstance(item, dict) and str(item.get("leg", "")).upper() == "TP1"
    ), None)
    for key in ("fill_time_ms", "order_update_time_ms", "ts"):
        value = _optional_float((event or {}).get(key))
        if value is not None and value > 0:
            return int(value)
    return int(fallback_ms)


def _calc_realized_rr(pnl_pct: float, risk_pct: float | None) -> float | None:
    if risk_pct is None or risk_pct <= 0:
        return None
    return pnl_pct / risk_pct


def _update_mfe_mae(trade: dict, candles: list[dict]) -> None:
    if not candles:
        return
    entry_price = _safe_float(trade.get("entry_price"))
    direction = str(trade.get("direction", "LONG")).upper()
    entry_ts = int(_safe_float(trade.get("entry_ts"), 0.0))
    if entry_price <= 0:
        return
    peak = _safe_float(trade.get("peak_pnl_pct", 0.0))
    mae = _safe_float(trade.get("mae_pct", 0.0))
    drawdown = _safe_float(trade.get("max_drawdown_pct", 0.0))
    for candle in sorted(candles, key=lambda x: int(_safe_float(x.get("open_time"), x.get("close_time", 0)))):
        open_ts = int(_safe_float(candle.get("open_time"), 0.0))
        close_ts = int(_safe_float(candle.get("close_time"), 0.0))
        close = _safe_float(candle.get("close"), 0.0)
        # Ignore candles that definitively closed before entry. If entry overlaps
        # the candle, OHLC cannot tell whether its extremes were pre- or post-fill.
        # The close is a known post-entry observation, so use it for this candle
        # instead of discarding all movement or attributing unknown extremes.
        overlapping_entry_candle = bool(entry_ts and open_ts and open_ts < entry_ts and close_ts > entry_ts)
        timestamp_ambiguous_candle = bool(entry_ts and not open_ts and close_ts > entry_ts)
        if entry_ts and ((close_ts and close_ts <= entry_ts) or ((open_ts and open_ts < entry_ts) and close_ts <= 0)):
            continue
        if overlapping_entry_candle or timestamp_ambiguous_candle:
            if close <= 0:
                continue
            high = low = close
        else:
            high = _safe_float(candle.get("high"), 0.0)
            low = _safe_float(candle.get("low"), 0.0)
        if high <= 0 or low <= 0:
            continue
        if direction == "LONG":
            favorable = (high - entry_price) / entry_price * 100.0
            adverse = (low - entry_price) / entry_price * 100.0
        else:
            favorable = (entry_price - low) / entry_price * 100.0
            adverse = (entry_price - high) / entry_price * 100.0
        prior_peak = peak
        peak = max(peak, favorable)
        mae = min(mae, adverse)
        drawdown = min(drawdown, adverse - max(prior_peak, favorable))

        risk_pct = _derive_planned_risk_pct(trade)
        if risk_pct and risk_pct > 0:
            milestones = trade.setdefault("mfe_milestones_r", {})
            favorable_r = favorable / risk_pct
            observed_ts = close_ts or open_ts
            for threshold in (0.25, 0.50, 1.00, 2.00):
                key = f"{threshold:.2f}"
                if favorable_r >= threshold and key not in milestones:
                    milestones[key] = observed_ts
    trade["peak_pnl_pct"] = peak
    trade["mae_pct"] = mae
    trade["max_drawdown_pct"] = drawdown

def _get_filled_order(symbol: str, order_id: str | None) -> dict | None:
    info = _get_order_execution_evidence(symbol, order_id)
    if not info:
        return None
    if info.get("status") != "ok" or str(info.get("order_status", "")).upper() != "FILLED":
        return None
    if _safe_float(info.get("executed_qty"), 0.0) <= 0 or _safe_float(info.get("avg_price"), 0.0) <= 0:
        return None
    return info


def _get_order_execution_evidence(symbol: str, order_id: str | None, *, entry_ts: int | None = None) -> dict | None:
    """Return order status enriched with actual fill-history evidence when needed.

    A FILLED status with executedQty=0 is not enough evidence of execution.  In that
    inconsistent case, query the exchange fill ledger by orderId and use actual fill
    quantity/weighted price.  origQty is deliberately never treated as a fill.

    Exchange financial fields are kept tri-state: a real zero is valid, while a missing
    or unparseable fee/realizedPnl remains ``None`` and is never manufactured as zero.
    """
    if not order_id:
        return None
    try:
        info = get_order(symbol, order_id)
    except Exception as exc:
        log.warning("[TRACKER] order query error for %s/%s: %s", symbol, order_id, exc)
        info = {"status": "error", "error": str(exc)}

    if not isinstance(info, dict):
        info = {"status": "error", "error": "invalid order response"}

    status = str(info.get("order_status", "")).upper()
    executed_qty = max(0.0, _safe_float(info.get("executed_qty"), 0.0))
    avg_price = max(0.0, _safe_float(info.get("avg_price"), 0.0))
    if info.get("status") == "ok" and executed_qty > 0 and avg_price > 0:
        info["execution_evidence_source"] = "ORDER_STATUS"
        return info

    if info.get("status") == "error":
        # Preserve exchange/API errors for the caller's telemetry path.
        return info
    if status not in {"FILLED", "PARTIALLY_FILLED"}:
        return info

    now_ms = int(time.time() * 1000)
    start_ms = max(0, int(entry_ts or info.get("time_ms") or now_ms) - 60_000)
    end_ms = max(start_ms, now_ms + 60_000)
    try:
        fills = get_fill_orders(symbol, start_ms, end_ms, order_id=str(order_id), limit=1000)
    except Exception as exc:
        log.warning("[TRACKER] fill-history query error for %s/%s: %s", symbol, order_id, exc)
        fills = []

    fills = [f for f in fills if str(f.get("order_id", "")) == str(order_id)]
    info["fill_history"] = fills
    qty_complete = _all_present_numeric(fills, "qty")
    price_complete = _all_present_numeric(fills, "price")
    priced_fills = [(float(_optional_float(f.get("qty"))), float(_optional_float(f.get("price")))) for f in fills if _optional_float(f.get("qty")) is not None and _optional_float(f.get("price")) is not None]
    total_qty = sum(q for q, _ in priced_fills)
    priced_qty = sum(q * p for q, p in priced_fills)
    weighted_price = priced_qty / total_qty if total_qty > 0 and qty_complete and price_complete else 0.0

    if total_qty > 0 and weighted_price > 0:
        info["executed_qty"] = total_qty
        info["avg_price"] = weighted_price
        info["execution_evidence_source"] = "ALL_FILL_ORDERS" if qty_complete and price_complete else "ALL_FILL_ORDERS_PARTIAL"
        if _all_present_numeric(fills, "realized_pnl"):
            info["fill_realized_pnl_abs"] = sum(float(_optional_float(f.get("realized_pnl"))) for f in fills)
            info["fill_realized_pnl_status"] = "CONFIRMED"
        else:
            info["fill_realized_pnl_abs"] = None
            info["fill_realized_pnl_status"] = "UNAVAILABLE_MISSING_FIELD"
        if _all_present_numeric(fills, "fee"):
            info["fill_fee_raw"] = sum(float(_optional_float(f.get("fee"))) for f in fills)
            info["fill_fee_status"] = "CONFIRMED"
        else:
            info["fill_fee_raw"] = None
            info["fill_fee_status"] = "UNAVAILABLE_MISSING_FIELD"
        info["fill_trade_ids"] = [str(f.get("trade_id")) for f in fills if f.get("trade_id")]
        return info

    # Keep the order response intact, but do not manufacture execution from origQty.
    info["execution_evidence_source"] = "UNVERIFIED"
    info["fill_realized_pnl_abs"] = None
    info["fill_fee_raw"] = None
    info["fill_realized_pnl_status"] = "UNAVAILABLE"
    info["fill_fee_status"] = "UNAVAILABLE"
    return info if info.get("status") == "ok" else None


def _collect_trade_fill_accounting(trade: dict, now_ms: int) -> dict[str, Any]:
    """Collect exchange execution/fee/PnL evidence for one logical trade.

    Priority is strict and explicit:
    1. exchange fill-history rows are the authoritative fill ledger;
    2. when the aggregate fill query does not return a known order, retry that
       order by ``orderId``;
    3. when fill history is still unavailable, an exchange order response with
       ``executedQty > 0`` and ``avgPrice > 0`` is retained as *execution
       evidence*, but it is never treated as fill-level fee/realized-PnL evidence.

    This lets us reconstruct gross price PnL and execution timestamps without
    manufacturing account fees or exchange realized PnL. Missing financial fields
    remain ``None``.
    """
    symbol = str(trade.get("symbol", ""))
    direction = _normalize_direction(trade.get("direction", ""))
    entry_ts = int(_safe_float(trade.get("entry_ts"), 0.0))
    order_roles: dict[str, str] = {}

    entry_order = trade.get("entry_order") if isinstance(trade.get("entry_order"), dict) else {}
    for key in ("order_id", "orderId", "orderID"):
        if entry_order.get(key):
            order_roles[str(entry_order[key])] = "ENTRY"

    for tp in trade.get("tp_orders", []) if isinstance(trade.get("tp_orders"), list) else []:
        if not isinstance(tp, dict):
            continue
        for key in ("order_id", "orderId", "orderID"):
            if tp.get(key):
                order_roles[str(tp[key])] = str(tp.get("leg") or "TP").upper()

    # Filled TP orders may be removed from ``tp_orders`` once the tracker sees
    # them complete. Keep their IDs from the immutable TP-fill event ledger so the
    # exchange fill-history query can still account for every closed leg.
    for tp_event in trade.get("tp_fill_events", []) if isinstance(trade.get("tp_fill_events"), list) else []:
        if not isinstance(tp_event, dict):
            continue
        order_id = str(tp_event.get("order_id") or tp_event.get("orderId") or "")
        if order_id:
            order_roles.setdefault(order_id, str(tp_event.get("leg") or "TP").upper())

    sl_order = trade.get("sl_order") if isinstance(trade.get("sl_order"), dict) else {}
    for key in ("order_id", "orderId", "orderID"):
        if sl_order.get(key):
            order_roles[str(sl_order[key])] = "SL"

    if trade.get("be_order_id"):
        order_roles[str(trade["be_order_id"])] = "BE_SL"
    if trade.get("exit_order_id"):
        order_roles[str(trade["exit_order_id"])] = order_roles.get(str(trade["exit_order_id"]), "EXIT")

    # Emergency rollback MARKET orders are nested in durable evidence. Include
    # every known order ID so their actual fills can complete the whole-trade PnL.
    def _collect_emergency_order_ids(node: Any) -> None:
        if isinstance(node, dict):
            for key in ("order_id", "orderId", "orderID"):
                value = node.get(key)
                if value not in (None, ""):
                    order_roles.setdefault(str(value), "EXIT")
            for value in node.values():
                if isinstance(value, (dict, list)):
                    _collect_emergency_order_ids(value)
        elif isinstance(node, list):
            for value in node:
                _collect_emergency_order_ids(value)

    _collect_emergency_order_ids(trade.get("be_failure_rollback"))
    _collect_emergency_order_ids(trade.get("verified_emergency_close"))

    base = {
        "fills": [],
        "order_execution_evidence": [],
        "order_ids": sorted(order_roles),
        "source": "ALL_FILL_ORDERS",
        "fill_query_mode": "AGGREGATE",
        "exact_order_queries_attempted": [],
        "exact_order_query_errors": {},
        "order_execution_evidence_errors": {},
        "incomplete_order_fill_ids": [],
    }
    if not symbol or not order_roles:
        return {**base, "status": "NO_ORDER_IDS"}

    start_ms = max(0, entry_ts - 60_000)
    end_ms = max(entry_ts, now_ms) + 60_000
    aggregate_error = None
    try:
        fills = get_fill_orders(symbol, start_ms, end_ms, limit=1000)
    except Exception as exc:
        fills = []
        aggregate_error = f"{type(exc).__name__}:{exc}"

    def _fill_key(fill: dict[str, Any]) -> tuple[Any, ...]:
        return (
            str(fill.get("order_id", "")),
            str(fill.get("trade_id", "")),
            int(_optional_float(fill.get("time_ms")) or 0),
            float(_optional_float(fill.get("qty")) or 0.0),
            float(_optional_float(fill.get("price")) or 0.0),
        )

    relevant_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    for fill in fills if isinstance(fills, list) else []:
        if str(fill.get("order_id", "")) not in order_roles:
            continue
        row = dict(fill)
        row["role"] = order_roles.get(str(row.get("order_id", "")), "UNKNOWN")
        relevant_map[_fill_key(row)] = row

    # The aggregate endpoint is not guaranteed to surface every order for a
    # busy account. Retry absent orders, and also retry orders whose observed fill
    # quantity is lower than exchange executedQty (partial/truncated aggregate).
    aggregate_order_ids = {str(row.get("order_id", "")) for row in relevant_map.values()}
    exact_attempts: list[str] = []
    exact_errors: dict[str, str] = {}
    order_execution_evidence_errors: dict[str, str] = {}
    incomplete_order_fill_ids: list[str] = []
    expected_qty_by_order: dict[str, float] = {}
    order_evidence_cache: dict[str, dict[str, Any]] = {}
    for order_id in sorted(order_roles):
        rows_for_order = [row for row in relevant_map.values() if str(row.get("order_id", "")) == order_id]
        had_any_fill_rows = bool(rows_for_order)
        observed_qty = sum(abs(_safe_float(row.get("qty"), 0.0)) for row in rows_for_order)
        needs_exact = order_id not in aggregate_order_ids or order_id in order_execution_evidence_errors
        # Order status gives us the authoritative executedQty needed to detect a
        # truncated aggregate response. Cache it so missing-fill fallback below
        # doesn't issue a duplicate request for the same order.
        execution_evidence = None
        if credentials_available():
            try:
                execution_evidence = _get_order_execution_evidence(symbol, order_id, entry_ts=entry_ts)
            except Exception as exc:
                execution_evidence = {"status": "error", "error": f"{type(exc).__name__}:{exc}"}
            if isinstance(execution_evidence, dict):
                order_evidence_cache[order_id] = execution_evidence
                if execution_evidence.get("status") == "error":
                    order_execution_evidence_errors[order_id] = str(
                        execution_evidence.get("error") or "order execution evidence query failed"
                    )
                    # An aggregate row is not authoritative proof of order-level
                    # completeness when the status query failed. Attempt the exact
                    # order-specific fill query as an independent evidence path.
                    needs_exact = True
            elif execution_evidence is None:
                order_execution_evidence_errors[order_id] = "order execution evidence unavailable"
                needs_exact = True
        if isinstance(execution_evidence, dict) and execution_evidence.get("status") == "ok":
            expected_qty = _optional_float(execution_evidence.get("executed_qty"))
            if expected_qty is not None:
                expected_qty_by_order[order_id] = expected_qty
                if expected_qty > observed_qty + max(1e-10, expected_qty * 1e-6):
                    needs_exact = True
        if not needs_exact:
            continue
        exact_attempts.append(order_id)
        try:
            exact_rows = get_fill_orders(symbol, start_ms, end_ms, order_id=order_id, limit=1000)
        except Exception as exc:
            exact_rows = []
            exact_errors[order_id] = f"{type(exc).__name__}:{exc}"
        exact_rows_added = 0
        for fill in exact_rows if isinstance(exact_rows, list) else []:
            if str(fill.get("order_id", "")) != order_id:
                continue
            row = dict(fill)
            row["role"] = order_roles.get(order_id, "UNKNOWN")
            relevant_map[_fill_key(row)] = row
            exact_rows_added += 1
        expected_qty = expected_qty_by_order.get(order_id)
        if expected_qty is not None:
            observed_after_exact = sum(
                abs(_safe_float(row.get("qty"), 0.0))
                for row in relevant_map.values() if str(row.get("order_id", "")) == order_id
            )
            if (had_any_fill_rows or exact_rows_added > 0) and expected_qty > observed_after_exact + max(1e-10, expected_qty * 1e-6):
                incomplete_order_fill_ids.append(order_id)

    relevant = list(relevant_map.values())

    # If fill history still lacks a known order, retain exchange order-level
    # execution evidence. This is actual executedQty/avgPrice evidence, not an
    # invented fill and never supplies fee/realizedPnl values.
    fill_order_ids = {str(row.get("order_id", "")) for row in relevant}
    order_execution_evidence: list[dict[str, Any]] = []
    for order_id in sorted(order_roles):
        if order_id in fill_order_ids:
            continue
        info = order_evidence_cache.get(order_id)
        if info is None and not credentials_available():
            continue
        if info is None:
            try:
                info = _get_order_execution_evidence(symbol, order_id, entry_ts=entry_ts)
            except Exception as exc:
                info = {"status": "error", "error": f"{type(exc).__name__}:{exc}"}
        if not isinstance(info, dict) or info.get("status") != "ok":
            continue
        executed_qty = _safe_float(info.get("executed_qty"), 0.0)
        avg_price = _safe_float(info.get("avg_price"), 0.0)
        if executed_qty <= 0 or avg_price <= 0:
            continue
        order_execution_evidence.append({
            "order_id": str(info.get("order_id") or order_id),
            "role": order_roles.get(order_id, "UNKNOWN"),
            "order_status": str(info.get("order_status") or "").upper(),
            "executed_qty": executed_qty,
            "avg_price": avg_price,
            "trigger_price": _safe_float(info.get("trigger_price"), 0.0) or None,
            "time_ms": info.get("time_ms"),
            "update_time_ms": info.get("update_time_ms"),
            "execution_evidence_source": "ORDER_STATUS_EXECUTION_EVIDENCE",
            "fill_realized_pnl_abs": None,
            "fill_fee_raw": None,
            "fill_realized_pnl_status": "UNAVAILABLE_ORDER_LEVEL_ONLY",
            "fill_fee_status": "UNAVAILABLE_ORDER_LEVEL_ONLY",
        })

    relevant = sorted(relevant, key=lambda row: (
        int(_optional_float(row.get("time_ms")) or 0),
        str(row.get("order_id", "")),
        str(row.get("trade_id", "")),
    ))
    for row in relevant:
        row.setdefault("execution_evidence_source", "ALL_FILL_ORDERS")

    entry_fills = [f for f in relevant if f.get("role") == "ENTRY"]
    exit_fills = [f for f in relevant if f.get("role") != "ENTRY"]
    entry_exec = [
        {"qty": _optional_float(f.get("qty")), "price": _optional_float(f.get("price")), "source": "FILL"}
        for f in entry_fills
    ] + [
        {"qty": x["executed_qty"], "price": x["avg_price"], "source": "ORDER"}
        for x in order_execution_evidence if x.get("role") == "ENTRY"
    ]
    exit_exec = [
        {"qty": _optional_float(f.get("qty")), "price": _optional_float(f.get("price")), "source": "FILL", "time_ms": f.get("time_ms")}
        for f in exit_fills
    ] + [
        {"qty": x["executed_qty"], "price": x["avg_price"], "source": "ORDER", "time_ms": x.get("update_time_ms") or x.get("time_ms")}
        for x in order_execution_evidence if x.get("role") != "ENTRY"
    ]

    def _complete_exec(rows: list[dict[str, Any]]) -> bool:
        return bool(rows) and all(
            isinstance(row.get("qty"), (int, float)) and math.isfinite(float(row["qty"])) and float(row["qty"]) > 0
            and isinstance(row.get("price"), (int, float)) and math.isfinite(float(row["price"])) and float(row["price"]) > 0
            for row in rows
        )

    entry_execution_complete = _complete_exec(entry_exec)
    exit_execution_complete = _complete_exec(exit_exec)
    # If order-status lookup failed, aggregate rows alone cannot establish that
    # the fill history covers the order's full executedQty. Preserve observed
    # rows for forensic review, but do not label that order's accounting complete.
    if order_execution_evidence_errors:
        if any(order_roles.get(order_id) == "ENTRY" for order_id in order_execution_evidence_errors):
            entry_execution_complete = False
        if any(order_roles.get(order_id) != "ENTRY" for order_id in order_execution_evidence_errors):
            exit_execution_complete = False
    if incomplete_order_fill_ids:
        if any(order_roles.get(order_id) == "ENTRY" for order_id in incomplete_order_fill_ids):
            entry_execution_complete = False
        if any(order_roles.get(order_id) != "ENTRY" for order_id in incomplete_order_fill_ids):
            exit_execution_complete = False
    entry_qty = sum(float(row["qty"]) for row in entry_exec if row.get("qty") is not None)
    exit_qty = sum(float(row["qty"]) for row in exit_exec if row.get("qty") is not None)
    qty_tolerance = max(1e-12, abs(entry_qty) * 1e-8) if entry_qty > 0 else 1e-12
    exit_qty_covers_entry = entry_qty > 0 and exit_qty >= entry_qty - qty_tolerance
    entry_vwap = (
        sum(float(row["qty"]) * float(row["price"]) for row in entry_exec) / entry_qty
        if entry_execution_complete and entry_qty > 0 else None
    )

    gross_realized_pnl_abs = None
    if (
        entry_vwap is not None and exit_execution_complete and direction in {"LONG", "SHORT"}
        and not incomplete_order_fill_ids and not order_execution_evidence_errors and exit_qty_covers_entry
    ):
        gross_realized_pnl_abs = 0.0
        for row in exit_exec:
            q = float(row["qty"])
            px = float(row["price"])
            gross_realized_pnl_abs += (px - entry_vwap) * q if direction == "LONG" else (entry_vwap - px) * q

    # Financial totals remain fill-ledger-only. An order-level execution response
    # has no authoritative fee/realizedPnl fields and must not be treated as one.
    fee_complete = (
        bool(relevant) and _all_present_numeric(relevant, "fee") and not order_execution_evidence
        and not incomplete_order_fill_ids and not order_execution_evidence_errors and exit_qty_covers_entry
    )
    realized_complete = (
        bool(relevant) and _all_present_numeric(relevant, "realized_pnl") and not order_execution_evidence
        and not incomplete_order_fill_ids and not order_execution_evidence_errors and exit_qty_covers_entry
    )
    exchange_realized_pnl_abs = sum(float(_optional_float(f.get("realized_pnl"))) for f in relevant) if realized_complete else None
    entry_fee_raw = exit_fee_raw = fee_raw_total = fees_paid_abs = None
    if fee_complete:
        entry_fee_raw = sum(float(_optional_float(f.get("fee"))) for f in entry_fills)
        exit_fee_raw = sum(float(_optional_float(f.get("fee"))) for f in exit_fills)
        fee_raw_total = entry_fee_raw + exit_fee_raw
        fees_paid_abs = max(0.0, -fee_raw_total)

    net_realized_pnl_abs = None
    if gross_realized_pnl_abs is not None and fee_raw_total is not None:
        net_realized_pnl_abs = gross_realized_pnl_abs + fee_raw_total

    fill_exit_ts = [
        int(_optional_float(f.get("time_ms")))
        for f in exit_fills
        if _optional_float(f.get("time_ms")) is not None and int(_optional_float(f.get("time_ms"))) > 0
    ]
    order_exit_ts = [
        int(_optional_float(x.get("update_time_ms") or x.get("time_ms")))
        for x in order_execution_evidence
        if x.get("role") != "ENTRY" and _optional_float(x.get("update_time_ms") or x.get("time_ms")) is not None
        and int(_optional_float(x.get("update_time_ms") or x.get("time_ms"))) > 0
    ]
    # Keep true fill time authoritative when fills exist. A later order update is
    # lifecycle metadata, not proof that execution happened at that later time;
    # use order update time only as an explicitly labeled fallback when no fills exist.
    if fill_exit_ts:
        last_exit_fill_ts_ms = max(fill_exit_ts)
        last_exit_execution_timestamp_source = "FILL_TIME"
    elif order_exit_ts:
        last_exit_fill_ts_ms = max(order_exit_ts)
        last_exit_execution_timestamp_source = "ORDER_UPDATE_OR_TIME"
    else:
        last_exit_fill_ts_ms = None
        last_exit_execution_timestamp_source = "UNAVAILABLE"

    execution_source = "ALL_FILL_ORDERS"
    if order_execution_evidence and relevant:
        execution_source = "ALL_FILL_ORDERS+ORDER_STATUS"
    elif order_execution_evidence:
        execution_source = "ORDER_STATUS_EXECUTION_EVIDENCE"
    elif aggregate_error:
        execution_source = "ALL_FILL_ORDERS_ERROR"
    if order_execution_evidence_errors:
        execution_source += "+ORDER_STATUS_ERROR"

    exit_vwap_confirmed = (
        sum(float(row["qty"]) * float(row["price"]) for row in exit_exec) / exit_qty
        if exit_execution_complete and exit_qty > 0 and not incomplete_order_fill_ids else None
    )
    exit_sources = {str(row.get("source") or "").upper() for row in exit_exec}
    if exit_vwap_confirmed is None:
        exit_vwap_source = "UNAVAILABLE"
    elif exit_sources == {"FILL"}:
        exit_vwap_source = "CONFIRMED_FILL_HISTORY"
    elif exit_sources and exit_sources.issubset({"FILL", "ORDER"}):
        exit_vwap_source = "CONFIRMED_EXECUTION_EVIDENCE"
    elif exit_sources == {"ORDER"}:
        exit_vwap_source = "CONFIRMED_ORDER_EXECUTION_EVIDENCE"
    else:
        exit_vwap_source = "UNAVAILABLE"
    execution_qty_status = (
        "VERIFIED" if entry_execution_complete and exit_execution_complete and exit_qty_covers_entry
        and not incomplete_order_fill_ids and not order_execution_evidence_errors
        else "PARTIAL_OR_UNVERIFIED"
    )
    full_execution_accounting = (
        entry_execution_complete and exit_execution_complete and exit_qty_covers_entry
        and not incomplete_order_fill_ids and not order_execution_evidence_errors
    )
    status = (
        "VERIFIED" if full_execution_accounting and not order_execution_evidence
        else "VERIFIED_ORDER_EXECUTION" if full_execution_accounting
        else "PARTIAL_ACCOUNTING"
    )
    return {
        "status": status,
        "fills": relevant,
        "order_execution_evidence": order_execution_evidence,
        "order_ids": sorted(order_roles),
        "entry_qty_confirmed": entry_qty if entry_execution_complete else None,
        "confirmed_exit_qty": exit_qty if exit_execution_complete else None,
        "observed_entry_qty": entry_qty,
        "observed_exit_qty": exit_qty,
        "confirmed_exit_qty_source": "FILL_HISTORY" if exit_fills and not order_execution_evidence else "ORDER_STATUS_EXECUTION_EVIDENCE" if order_execution_evidence else "UNAVAILABLE",
        "exit_vwap_confirmed": exit_vwap_confirmed,
        "exit_vwap_source": exit_vwap_source,
        "entry_vwap_confirmed": entry_vwap,
        "entry_fee_raw": entry_fee_raw,
        "exit_fee_raw": exit_fee_raw,
        "gross_realized_pnl_abs": gross_realized_pnl_abs,
        "gross_realized_pnl_source": "CONFIRMED_EXECUTION_EVIDENCE" if gross_realized_pnl_abs is not None else "UNAVAILABLE",
        "exchange_realized_pnl_abs": exchange_realized_pnl_abs,
        "exchange_realized_pnl_source": "ALL_FILL_ORDERS" if exchange_realized_pnl_abs is not None else "UNAVAILABLE_FILL_LEVEL_EVIDENCE",
        "fee_raw_total": fee_raw_total,
        "fees_paid_abs": fees_paid_abs,
        "fee_source": "ALL_FILL_ORDERS" if fee_raw_total is not None else "UNAVAILABLE_FILL_LEVEL_EVIDENCE",
        "fees_status": "CONFIRMED" if fee_raw_total is not None else "UNAVAILABLE",
        "net_realized_pnl_abs": net_realized_pnl_abs,
        "net_realized_pnl_source": "CONFIRMED_EXECUTION_EVIDENCE_PLUS_SIGNED_FEES" if net_realized_pnl_abs is not None else "UNAVAILABLE",
        "last_exit_fill_ts_ms": last_exit_fill_ts_ms,
        "last_exit_execution_timestamp_source": last_exit_execution_timestamp_source,
        "quantity_reconciliation": {
            "status": execution_qty_status,
            "initial_qty": entry_qty if entry_execution_complete else None,
            "confirmed_exit_qty": exit_qty if exit_execution_complete else None,
            "difference_qty": (exit_qty - entry_qty) if entry_execution_complete and exit_execution_complete else None,
            "tolerance": qty_tolerance,
        },
        "source": execution_source,
        "aggregate_query_error": aggregate_error,
        "exact_order_queries_attempted": exact_attempts,
        "exact_order_query_errors": exact_errors,
        "order_execution_evidence_errors": order_execution_evidence_errors,
        "incomplete_order_fill_ids": sorted(set(incomplete_order_fill_ids)),
    }


def _close_execution_evidence(trade: dict, fill_accounting: dict[str, Any]) -> dict[str, Any]:
    """Return the strongest confirmed exit-leg evidence available for a close.

    Fill-ledger rows are preferred.  When the exchange fill ledger is unavailable,
    confirmed order execution evidence is used.  As a final tracker-local fallback
    for TP closes, previously recorded ``tp_fill_events`` are retained as evidence
    but are clearly labelled as TP-event evidence.
    """
    reason = str(trade.get("exit_reason") or "").upper()
    def _include_leg(leg: str) -> bool:
        role = str(leg or "").upper()
        if role == "ENTRY":
            return False
        if reason == "TAKE_PROFIT_FULL":
            return role.startswith("TP")
        if reason in {"STOP_LOSS", "BREAK_EVEN"}:
            # A final SL/BE close only exits the remaining quantity. Earlier TP
            # fills are part of the same logical trade's exit accounting and must
            # be included in its total realized exit quantity/VWAP.
            return role in {"SL", "BE_SL", "EXIT"} or role.startswith("BE") or role.startswith("TP")
        return True

    rows: list[dict[str, Any]] = []
    tp_event_by_order: dict[str, dict[str, Any]] = {}
    for event in trade.get("tp_fill_events", []) if isinstance(trade.get("tp_fill_events"), list) else []:
        oid = str(event.get("order_id") or "")
        if oid:
            tp_event_by_order[oid] = event

    for fill in fill_accounting.get("fills", []) if isinstance(fill_accounting.get("fills"), list) else []:
        role = str(fill.get("role") or "EXIT").upper()
        if not _include_leg(role):
            continue
        qty = _optional_float(fill.get("qty"))
        px = _optional_float(fill.get("price"))
        if qty is None or px is None or qty <= 0 or px <= 0:
            continue
        event = tp_event_by_order.get(str(fill.get("order_id") or "")) or {}
        rows.append({
            "order_id": str(fill.get("order_id") or ""),
            "leg": role,
            "status": "FILLED",
            "executed_qty": float(qty),
            "avg_price": float(px),
            "trigger_price": event.get("trigger_price"),
            "fill_time_ms": _optional_float(fill.get("time_ms")),
            "order_update_time_ms": event.get("order_update_time_ms"),
            "evidence_source": "ALL_FILL_ORDERS",
            "trade_id": fill.get("trade_id"),
        })
    for row in fill_accounting.get("order_execution_evidence", []) if isinstance(fill_accounting.get("order_execution_evidence"), list) else []:
        role = str(row.get("role") or "EXIT").upper()
        if not _include_leg(role):
            continue
        qty = _optional_float(row.get("executed_qty"))
        px = _optional_float(row.get("avg_price"))
        if qty is None or px is None or qty <= 0 or px <= 0:
            continue
        rows.append({
            "order_id": str(row.get("order_id") or ""),
            "leg": role,
            "status": str(row.get("order_status") or "FILLED").upper(),
            "executed_qty": float(qty),
            "avg_price": float(px),
            "trigger_price": row.get("trigger_price"),
            "fill_time_ms": None,
            "order_update_time_ms": row.get("update_time_ms"),
            "evidence_source": row.get("execution_evidence_source") or "ORDER_STATUS_EXECUTION_EVIDENCE",
            "trade_id": None,
        })

    # If the exchange fill/order query did not return a TP order, retain its
    # previously recorded tracker TP-fill event as lower-authority evidence.
    # Never add an event whose order ID is already represented by exchange rows;
    # that would double-count a leg that the aggregate endpoint already returned.
    represented_order_ids = {str(row.get("order_id") or "") for row in rows if row.get("order_id")}
    # Tracker snapshots can emit several events for one partially-filled TP order.
    # Those events contain cumulative ``executed_qty_total`` as well as the latest
    # incremental delta. Collapse to the highest cumulative quantity/latest event
    # per order before using local fallback evidence.
    tp_events_by_order: dict[str, dict[str, Any]] = {}
    for event in trade.get("tp_fill_events", []) if isinstance(trade.get("tp_fill_events"), list) else []:
        if not isinstance(event, dict):
            continue
        order_id = str(event.get("order_id") or "")
        if not order_id:
            continue
        prior = tp_events_by_order.get(order_id)
        current_total = _optional_float(event.get("executed_qty_total")) or _optional_float(event.get("delta_qty")) or 0.0
        prior_total = (_optional_float(prior.get("executed_qty_total")) or _optional_float(prior.get("delta_qty")) or 0.0) if prior else -1.0
        current_ts = _optional_float(event.get("fill_time_ms") or event.get("order_update_time_ms") or event.get("ts")) or 0.0
        prior_ts = (_optional_float(prior.get("fill_time_ms") or prior.get("order_update_time_ms") or prior.get("ts")) or 0.0) if prior else -1.0
        if prior is None or current_total > prior_total or (current_total == prior_total and current_ts >= prior_ts):
            tp_events_by_order[order_id] = event

    for order_id, event in tp_events_by_order.items():
        if order_id in represented_order_ids:
            continue
        qty = _optional_float(event.get("executed_qty_total")) or _optional_float(event.get("delta_qty"))
        px = _optional_float(event.get("avg_price"))
        # An orderless event cannot be deduplicated against exchange execution
        # evidence, so do not use it for cross-leg quantity reconciliation.
        if qty is None or px is None or qty <= 0 or px <= 0:
            continue
        rows.append({
            "order_id": order_id,
            "leg": str(event.get("leg") or "TP").upper(),
            "status": str(event.get("order_status") or "FILLED").upper(),
            "executed_qty": float(qty),
            "avg_price": float(px),
            "trigger_price": event.get("trigger_price"),
            "fill_time_ms": event.get("fill_time_ms"),
            "order_update_time_ms": event.get("order_update_time_ms"),
            "evidence_source": "TP_FILL_EVENT",
            "trade_id": (event.get("fill_trade_ids") or [None])[0],
        })
        represented_order_ids.add(order_id)

    if not rows:
        return {
            "status": "UNAVAILABLE",
            "authority": "UNAVAILABLE",
            "rows": [],
            "confirmed_exit_qty": None,
            "exit_vwap_confirmed": None,
            "exit_vwap_source": "UNAVAILABLE",
            "latest": None,
        }

    total_qty = sum(float(r["executed_qty"]) for r in rows)
    exit_vwap = sum(float(r["executed_qty"]) * float(r["avg_price"]) for r in rows) / total_qty if total_qty > 0 else None

    def _ts(row: dict[str, Any]) -> int:
        value = row.get("fill_time_ms") or row.get("order_update_time_ms")
        parsed = _optional_float(value)
        return int(parsed) if parsed is not None and parsed > 0 else 0

    latest = max(rows, key=_ts)
    evidence_sources = {str(row.get("evidence_source") or "").upper() for row in rows}
    if evidence_sources and evidence_sources.issubset({"ALL_FILL_ORDERS"}):
        authority = "EXCHANGE_FILL_HISTORY"
        exit_vwap_source = "CONFIRMED_FILL_HISTORY"
    elif evidence_sources and evidence_sources.issubset({"ALL_FILL_ORDERS", "ORDER_STATUS_EXECUTION_EVIDENCE"}):
        authority = "EXCHANGE_EXECUTION_EVIDENCE"
        exit_vwap_source = "CONFIRMED_EXECUTION_EVIDENCE"
    elif evidence_sources == {"ORDER_STATUS_EXECUTION_EVIDENCE"}:
        authority = "EXCHANGE_ORDER_EXECUTION"
        exit_vwap_source = "CONFIRMED_ORDER_EXECUTION_EVIDENCE"
    else:
        authority = "TRACKER_LOCAL_EVENT"
        exit_vwap_source = "TRACKER_RECORDED_TP_EVENT"
    return {
        "status": "CONFIRMED",
        "authority": authority,
        "rows": rows,
        "confirmed_exit_qty": total_qty,
        "exit_vwap_confirmed": exit_vwap,
        "exit_vwap_source": exit_vwap_source,
        "latest": latest,
        "latest_execution_ts_ms": _ts(latest) or None,
        "latest_execution_timestamp_source": (
            "FILL_TIME" if _optional_float(latest.get("fill_time_ms")) is not None and _optional_float(latest.get("fill_time_ms")) > 0
            else "ORDER_UPDATE_TIME" if _optional_float(latest.get("order_update_time_ms")) is not None and _optional_float(latest.get("order_update_time_ms")) > 0
            else "UNAVAILABLE"
        ),
    }


def _classify_confirmed_stop_exit_reason(trade: dict[str, Any], sl_order_id: str | None, sl_order: dict[str, Any] | None = None) -> str:
    """Classify a confirmed SL/BE execution from order identity, not price tolerance."""
    if bool(trade.get("be_activated")):
        be_order_id = trade.get("be_order_id")
        if be_order_id and sl_order_id and str(be_order_id) == str(sl_order_id):
            return "BREAK_EVEN"
        order = sl_order if isinstance(sl_order, dict) else {}
        client_id = str(order.get("client_order_id") or order.get("clientOrderId") or "").upper()
        be_client_id = str(trade.get("be_client_order_id") or "").upper()
        if be_client_id and client_id == be_client_id:
            return "BREAK_EVEN"
        if client_id.startswith("EVT_BE_"):
            return "BREAK_EVEN"
    return "STOP_LOSS"


def _get_exit_from_sl(symbol: str, sl_order_id: str | None) -> tuple[float | None, str | None]:
    info = _get_filled_order(symbol, sl_order_id)
    if not info:
        return None, None
    exit_price = _safe_float(info.get("avg_price"), 0.0)
    return (exit_price if exit_price > 0 else None, "SL_FILLED")


def _reconcile_historical_exit_order(
    symbol: str,
    direction: str,
    entry_ts: int,
    remaining_qty: float,
    tp_orders: list[dict],
    sl_order: dict | None,
    *,
    trade: dict | None = None,
) -> tuple[float | None, str | None, str | None, dict | None]:
    """Recover an exchange-verified exit for the still-open residual quantity.

    A previously accounted TP fill must never be reused as evidence that the
    residual position closed. Candidates are therefore filtered by order
    identity, HEDGE position side (when available), execution chronology, and a
    near-exact residual-quantity match. Ambiguous evidence fails closed.
    """
    # First ask the currently-owned stop order directly.
    sl_id = sl_order.get("order_id") if isinstance(sl_order, dict) else None
    sl_info = _get_filled_order(symbol, sl_id)
    if sl_info:
        px = _safe_float(sl_info.get("avg_price"), 0.0)
        if px > 0:
            return px, "STOP_LOSS", "historical_sl_order", sl_info

    residual = max(0.0, float(remaining_qty))
    if residual <= 0:
        return None, None, None, None

    tp_ids = {str(x.get("order_id")) for x in tp_orders if x.get("order_id")}
    trade_state = trade if isinstance(trade, dict) else {}
    tp_fill_events = trade_state.get("tp_fill_events")
    tp_fill_events = tp_fill_events if isinstance(tp_fill_events, list) else []
    tp_filled_qty = trade_state.get("tp_filled_qty")
    tp_filled_qty = tp_filled_qty if isinstance(tp_filled_qty, dict) else {}
    hit_legs = {str(x).lower() for x in (trade_state.get("hit_legs") or [])}
    prior_tp_fill_known = bool(tp_fill_events) or bool(hit_legs) or any(
        _safe_float(qty, 0.0) > 0 for qty in tp_filled_qty.values()
    )

    # Use exchange execution/order timestamps, not the tracker observation time.
    tp_execution_times: list[int] = []
    for event in tp_fill_events:
        if not isinstance(event, dict):
            continue
        for field in ("fill_time_ms", "order_update_time_ms", "order_time_ms"):
            stamp = _optional_float(event.get(field))
            if stamp is not None and stamp > 0:
                tp_execution_times.append(int(stamp))
    latest_tp_execution_ms = max(tp_execution_times) if tp_execution_times else None

    start = max(0, int(entry_ts) - 30_000)
    end = int(time.time() * 1000) + 30_000
    try:
        orders = get_all_orders(symbol, start, end, limit=1000)
    except Exception as exc:
        log.warning("[TRACKER] historical order reconciliation failed for %s: %s", symbol, exc)
        return None, None, None, None

    wanted_side = "SELL" if direction == "LONG" else "BUY"
    candidates: list[dict] = []
    for order in orders:
        oid = str(order.get("orderId", order.get("orderID", "")))
        if oid and oid in tp_ids:
            continue
        status = str(order.get("status", order.get("orderStatus", ""))).upper()
        if status != "FILLED":
            continue
        side = str(order.get("side", "")).upper()
        if side != wanted_side:
            continue
        order_position_side = str(
            order.get("positionSide", order.get("position_side", "")) or ""
        ).upper()
        if order_position_side and order_position_side not in {direction, "BOTH"}:
            continue
        # In HEDGE mode, a side-only order is ambiguous after a partial TP.
        if prior_tp_fill_known and order_position_side not in {direction, "BOTH"}:
            continue
        try:
            created = int(float(order.get("updateTime") or order.get("time") or order.get("createTime") or 0))
            qty = abs(float(order.get("executedQty") or order.get("cumQty") or 0))
            px = float(order.get("avgPrice") or order.get("price") or order.get("stopPrice") or 0)
        except (TypeError, ValueError):
            continue
        if entry_ts and created and created < entry_ts:
            continue
        # A residual exit must happen strictly after the last already-accounted
        # TP execution. Equal timestamps are ambiguous and fail closed.
        if prior_tp_fill_known:
            if not created or latest_tp_execution_ms is None or created <= latest_tp_execution_ms:
                continue
        if px <= 0 or qty <= 0:
            continue
        # Requiring only >=95% of the residual wrongly accepted another half-size
        # TP1. A separate order must closely match the residual quantity.
        qty_tolerance = max(1e-8, residual * 1e-4)
        if abs(qty - residual) > qty_tolerance:
            continue
        candidates.append({**order, "_ts": created, "_qty": qty, "_px": px})

    if not candidates:
        return None, None, None, None

    # Prefer an explicitly identified stop over a manual/external close.
    stop_candidates = [
        o for o in candidates
        if str(o.get("type", "")).upper() in {"STOP", "STOP_MARKET"}
    ]
    pool = stop_candidates or candidates
    pool.sort(key=lambda o: int(o.get("_ts", 0)), reverse=True)

    for order in pool:
        order_id = str(order.get("orderId") or order.get("orderID") or "")
        client_id = str(order.get("clientOrderId") or order.get("client_order_id") or "").upper()
        trade_be_order_id = str(trade_state.get("be_order_id") or "")
        trade_be_client_id = str(trade_state.get("be_client_order_id") or "").upper()
        # Preserve BE semantics using durable order identity.
        if trade_be_order_id and order_id and trade_be_order_id == order_id:
            return order["_px"], "BREAK_EVEN", "historical_all_orders", order
        if trade_be_client_id and client_id and client_id == trade_be_client_id:
            return order["_px"], "BREAK_EVEN", "historical_all_orders", order
        if client_id.startswith("EVT_BE_"):
            return order["_px"], "BREAK_EVEN", "historical_all_orders", order
        reason = "STOP_LOSS" if str(order.get("type", "")).upper() in {"STOP", "STOP_MARKET"} else "MANUAL_CLOSE_RECONCILED"
        return order["_px"], reason, "historical_all_orders", order

    return None, None, None, None

def _classify_tp_fill(direction: str, entry_price: float, exec_price: float) -> tuple[bool, str]:
    """Return whether an observed TP execution is economically favorable.

    Trigger slippage may worsen the fill, but a TP fill for a LONG must not cross
    through entry (and vice versa). Crossing entry is classified as an execution
    anomaly and must never activate BE.
    """
    try:
        tolerance_pct = max(0.0, float(os.environ.get("TP_FILL_ECONOMIC_TOLERANCE_PCT", "0.10")))
    except (TypeError, ValueError):
        tolerance_pct = 0.10
    tol = tolerance_pct / 100.0
    if direction == "LONG":
        if exec_price < entry_price * (1.0 - tol):
            return False, "ANOMALOUS_ADVERSE_TP_FILL"
    elif direction == "SHORT":
        if exec_price > entry_price * (1.0 + tol):
            return False, "ANOMALOUS_ADVERSE_TP_FILL"
    else:
        return False, "INVALID_DIRECTION"
    return True, "FAVORABLE_OR_TOLERATED_TP_FILL"


def _reconciliation_status(local_qty: float, exchange_qty: float, base_status: str) -> str:
    """Return a truthful reconciliation status including quantity mismatches."""
    if base_status != "FOUND":
        return base_status
    try:
        local = abs(float(local_qty))
        exchange = abs(float(exchange_qty))
    except (TypeError, ValueError):
        return "FOUND_QTY_MISMATCH"
    try:
        rel_tol = max(1e-12, float(os.environ.get("POSITION_QTY_RECON_REL_TOL", "1e-9")))
        abs_tol = max(1e-12, float(os.environ.get("POSITION_QTY_RECON_ABS_TOL", "1e-8")))
    except (TypeError, ValueError):
        rel_tol, abs_tol = 1e-9, 1e-8
    return "FOUND" if math.isclose(local, exchange, rel_tol=rel_tol, abs_tol=abs_tol) else "FOUND_QTY_MISMATCH"


def _clear_completed_tp_recovery_markers(trade: dict[str, Any]) -> bool:
    """Clear recovery blockers once recovered fills are authoritatively booked.

    A completed but adverse TP fill consumes a target without becoming a
    validated hit (which remains the gate for TP1 -> BE). Therefore recovery
    completion must use completed_tp_legs ∪ hit_legs, not hit_legs alone.
    Unfilled or unverified recovered orders remain pending.
    """
    if not isinstance(trade, dict):
        return False
    tp_orders = trade.get("tp_orders") if isinstance(trade.get("tp_orders"), list) else []
    pending_legs = {
        str(item.get("leg", "")) for item in tp_orders
        if isinstance(item, dict) and item.get("recovered_execution_pending")
    }
    if not pending_legs:
        return False
    completed_legs = {str(value) for value in (trade.get("completed_tp_legs") or [])}
    completed_legs.update(str(value) for value in (trade.get("hit_legs") or []))
    if not pending_legs.issubset(completed_legs):
        return False
    trade["tp_recovery_pending"] = False
    for item in tp_orders:
        if isinstance(item, dict) and str(item.get("leg", "")) in pending_legs:
            item.pop("recovered_execution_pending", None)
    return True


def update_active_trades(*, position_snapshot: list[dict] | None = None) -> None:
    tracker_perf_started = time.perf_counter() if request_metrics.enabled() else None
    tracker_timings: dict[str, dict[str, float]] = {}
    # Reconciliation is a live-state operation. Once the engine has reached this
    # stage, missing state must fail closed rather than be interpreted as an empty
    # portfolio. Fresh-start bootstrap happens only in the top-level run preflight.
    trades = _load_active_trades_required()
    if not trades:
        return

    # One bounded worker overlaps a read-only TP status lookup with the existing
    # 1m kline request. The worker is shut down before this function returns.
    tp_read_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracker-tp-read")

    original_trades = copy.deepcopy(trades)
    now_ms = int(time.time() * 1000)
    updated_trades: dict[str, dict] = {}
    # Only consume a bulk snapshot when its caller confirms it came from a
    # successful get_positions request. Malformed inputs fall back to the
    # original symbol-by-symbol query path instead of being treated as flat.
    if position_snapshot is not None and not isinstance(position_snapshot, list):
        log.warning("[TRACKER_POSITION_SNAPSHOT] invalid payload type=%s; falling back to per-symbol queries", type(position_snapshot).__name__)
        position_snapshot = None
    elif position_snapshot is not None and any(not isinstance(row, dict) for row in position_snapshot):
        log.warning("[TRACKER_POSITION_SNAPSHOT] malformed rows; falling back to per-symbol queries")
        position_snapshot = None
    if position_snapshot is not None:
        log.info("[TRACKER_POSITION_SNAPSHOT] active_trades=%d exchange_positions=%d initial_symbol_queries=0", len(trades), len(position_snapshot))

    for event_id, trade in trades.items():
        if trade.get("closed", False):
            continue

        try:
            symbol = str(trade.get("symbol", ""))
            direction = _normalize_direction(trade.get("direction", ""))
            entry_price = _safe_float(trade.get("entry_price"))
            init_qty = abs(_safe_float(trade.get("initial_qty")))
            rem_qty = max(0.0, _safe_float(trade.get("remaining_qty")))
            entry_ts = int(_safe_float(trade.get("entry_ts")))

            if not symbol or entry_price <= 0 or init_qty <= 0:
                updated_trades[event_id] = trade
                continue

            pending_cleanup = trade.get("sl_cleanup_pending")
            if isinstance(pending_cleanup, dict) and pending_cleanup.get("old_order_id"):
                try:
                    cleanup_ok, cleanup_note = _cancel_old_sl_verified(
                        symbol, direction, str(pending_cleanup["old_order_id"]), owner_event_id=str(event_id),
                    )
                except Exception as cleanup_exc:
                    cleanup_ok, cleanup_note = False, str(cleanup_exc)
                if cleanup_ok:
                    trade.pop("sl_cleanup_pending", None)
                    log.info("[TRACKER_SL_CLEANUP] event=%s old_sl=%s cancellation verified", event_id, pending_cleanup.get("old_order_id"))
                else:
                    pending_cleanup["last_attempt_ts"] = now_ms
                    pending_cleanup["last_error"] = str(cleanup_note)
                    trade["sl_cleanup_pending"] = pending_cleanup
                    log.error("[TRACKER_SL_CLEANUP_PENDING] event=%s old_sl=%s error=%s", event_id, pending_cleanup.get("old_order_id"), cleanup_note)

            hit_legs = set(trade.get("hit_legs", []))
            filled_by_leg = {str(k): max(0.0, _safe_float(v)) for k, v in (trade.get("tp_filled_qty", {}) or {}).items()}
            realized_qty = max(0.0, _safe_float(trade.get("realized_pnl_qty", 0.0)))
            realized_weighted = _safe_float(trade.get("realized_pnl_weighted_sum", 0.0))

            pos = (
                _position_from_snapshot(position_snapshot, symbol, direction)
                if position_snapshot is not None
                else get_position_directional(symbol, direction)
            )
            pos_status = str(pos.get("status", "")).lower()

            if pos_status not in {"found", "not_found"}:
                trade["last_observation_ts"] = now_ms
                updated_trades[event_id] = trade
                continue

            local_remaining_qty_before = rem_qty
            pos_amt = abs(_safe_float(pos.get("positionAmt"))) if pos_status == "found" else 0.0
            telemetry.record_position_reconciliation(
                event_id=event_id,
                attempt_id=trade.get("attempt_id") or ((trade.get("setup") or {}).get("attempt_id") if isinstance(trade.get("setup"), dict) else None),
                position_id=trade.get("position_id") or event_id,
                symbol=symbol,
                direction=direction,
                status=_reconciliation_status(
                    local_remaining_qty_before,
                    pos_amt if pos_status in {"found", "not_found"} else 0.0,
                    "FOUND" if pos_status == "found" else "NOT_FOUND" if pos_status == "not_found" else "ERROR",
                ),
                internal_remaining_qty=local_remaining_qty_before,
                exchange_position_qty=pos_amt if pos_status in {"found", "not_found"} else None,
                exchange_avg_price=pos.get("avgPrice") if pos_status == "found" else None,
                local_be_activated=bool(trade.get("be_activated")),
                local_tp_filled_qty=dict(trade.get("tp_filled_qty", {}) or {}),
                local_sl_order_id=((trade.get("sl_order") or {}).get("order_id") if isinstance(trade.get("sl_order"), dict) else None),
                exchange_error=pos.get("error") if pos_status not in {"found", "not_found"} else None,
            )
            if pos_status == "found":
                # The exchange is authoritative for the live position quantity.
                # Local remaining_qty may be stale after a partial fill, manual
                # close, or an external reduction. Keep TP fill accounting
                # separate from the actual residual position size.
                rem_qty = pos_amt
                trade["remaining_qty"] = pos_amt
                exchange_avg = _safe_float(pos.get("avgPrice"))
                if exchange_avg > 0:
                    trade["last_exchange_avg_price"] = exchange_avg
            else:
                # The position is already gone. Keep the last locally tracked
                # residual quantity available for historical exit/PnL recovery.
                # The final exchange lookup below will set the live quantity to
                # zero after reconciliation.
                rem_qty = max(0.0, rem_qty)

            # Start only the first known TP status read alongside the existing
            # 1m kline read. The result is still consumed at its original position
            # in the TP loop, before any later TP leg is queried/processed.
            tp_orders_for_evidence = trade.get("tp_orders", [])
            first_tp_evidence_index = next((
                idx for idx, candidate in enumerate(tp_orders_for_evidence)
                if isinstance(candidate, dict) and candidate.get("leg") and candidate.get("order_id")
            ), None)
            first_tp_evidence_future: Future | None = None
            first_tp_evidence_result: tuple[dict | None, float, Exception | None] | None = None
            if _tracker_parallel_first_tp_read_enabled() and first_tp_evidence_index is not None:
                first_tp = tp_orders_for_evidence[first_tp_evidence_index]
                first_tp_evidence_future = tp_read_executor.submit(
                    _fetch_tp_evidence_timed,
                    symbol,
                    str(first_tp.get("order_id")),
                    entry_ts,
                    request_metrics.current_phase(),
                )

            cur_price = entry_price
            try:
                k1m = _timed_tracker_call(tracker_timings, "kline_1m", fetch_klines, symbol, "1m", limit=60)
                if k1m:
                    cur_price = _safe_float(k1m[-1].get("close"), entry_price)
                    _update_mfe_mae(trade, k1m)
            except Exception as exc:
                log.warning("[TRACKER] Kline fetch error for %s: %s", symbol, exc)

            current_pnl = _calc_trade_pnl_pct(entry_price, cur_price, direction)
            trade["current_pnl_pct"] = current_pnl
            trade["current_position_qty"] = pos_amt
            trade["last_observation_ts"] = now_ms

            # Join the read-only worker before any possible protection mutation below.
            # Its result is consumed in the original TP-loop position after the BE retry.
            if first_tp_evidence_future is not None:
                first_tp_evidence_result = first_tp_evidence_future.result()

            # Break-even is intentionally activated ONLY after TP1 is fully filled.
            # There is no earlier +0.50R price-triggered BE in this strategy.
            planned_risk_pct = _derive_planned_risk_pct(trade)
            peak_r = (float(trade.get("peak_pnl_pct", 0.0)) / planned_risk_pct) if planned_risk_pct and planned_risk_pct > 0 else 0.0

            # Retry a failed TP1 -> BE transition while the position remains open.
            previous_be_rollback = trade.get("be_failure_rollback") if isinstance(trade.get("be_failure_rollback"), dict) else {}
            previous_be_close_verified = str(previous_be_rollback.get("status", "")).lower() in {"closed_verified", "already_closed"}
            if "tp1" in set(trade.get("hit_legs", [])) and not trade.get("be_activated") and rem_qty > 0 and not previous_be_close_verified:
                old_sl = trade.get("sl_order", {}) if isinstance(trade.get("sl_order"), dict) else {}
                old_sl_id = old_sl.get("order_id")
                old_sl_price = _safe_float(old_sl.get("stop_price"), 0.0) or None
                retry = _timed_tracker_call(
                    tracker_timings, "be_retry", _move_sl_to_break_even,
                    symbol, direction, entry_price, rem_qty, old_sl_id,
                    str(event_id).replace("EVT_", ""), old_sl_price=old_sl_price,
                    owner_event_id=str(event_id),
                )
                retry_status = str(retry.get("status", "")).lower()
                if retry_status in {"created", "created_cleanup_pending"}:
                    trade["sl_order"] = retry
                    trade["be_activated"] = True
                    trade["be_required"] = False
                    trade["be_last_error"] = retry.get("error") if retry_status == "created_cleanup_pending" else None
                    trade["be_trigger_ts"] = trade.get("be_trigger_ts") or _tp1_fill_timestamp_ms(trade, now_ms)
                    trade["be_trigger_peak_r"] = trade.get("be_trigger_peak_r") if trade.get("be_trigger_peak_r") is not None else peak_r
                    trade["be_trigger_rule"] = trade.get("be_trigger_rule") or "tp1_filled"
                    trade["be_trigger_price"] = trade.get("be_trigger_price") or _safe_float(retry.get("stop_price"), entry_price)
                    trade["be_activation_ts"] = trade.get("be_activation_ts") or now_ms
                    trade["be_order_id"] = retry.get("order_id")
                    trade["be_client_order_id"] = retry.get("client_order_id")
                    pending_old_sl_id = retry.get("old_sl_cleanup_pending_order_id")
                    if pending_old_sl_id:
                        trade["sl_cleanup_pending"] = {
                            "old_order_id": str(pending_old_sl_id),
                            "since_ts": now_ms,
                            "last_error": retry.get("error"),
                        }
                    else:
                        trade.pop("sl_cleanup_pending", None)
                else:
                    trade["be_required"] = True
                    trade["be_last_error"] = retry.get("error")
                    rollback = retry.get("rollback") if isinstance(retry.get("rollback"), dict) else None
                    if rollback and str(rollback.get("status", "")).lower() in {"closed_verified", "already_closed"}:
                        trade["be_failure_rollback"] = rollback
                        trade["be_failure_rollback_verified_ts"] = now_ms

            # Проверка исполнения Тейк-Профитов
            for tp_index, tp in enumerate(tp_orders_for_evidence):
                leg = str(tp.get("leg", ""))
                order_id = tp.get("order_id")
                if not leg or not order_id:
                    continue

                try:
                    if tp_index == first_tp_evidence_index and first_tp_evidence_result is not None:
                        order_info, elapsed, query_error = first_tp_evidence_result
                        _record_tracker_elapsed(tracker_timings, "tp_order_evidence", elapsed)
                        if query_error is not None:
                            raise query_error
                    else:
                        order_info = _timed_tracker_call(
                            tracker_timings, "tp_order_evidence", _get_order_execution_evidence,
                            symbol, str(order_id), entry_ts=entry_ts,
                        )
                except Exception as exc:
                    telemetry.record_exchange_error(
                        event_id=event_id,
                        attempt_id=trade.get("attempt_id") or ((trade.get("setup") or {}).get("attempt_id") if isinstance(trade.get("setup"), dict) else None),
                        position_id=trade.get("position_id") or event_id,
                        order_id=str(order_id),
                        symbol=symbol,
                        endpoint=ORDER_PATH,
                        method="GET",
                        error_code=None,
                        message=str(exc),
                        error_class="TP_ORDER_QUERY",
                        blocking=False,
                        business_impact="TP_STATE_UNKNOWN",
                        leg=leg.upper(),
                    )
                    continue

                if order_info is None:
                    continue

                if order_info.get("status") == "error":
                    error_code, error_message = telemetry.parse_exchange_error(order_info, default_code=order_info.get("code"))
                    telemetry.record_exchange_error(
                        event_id=event_id,
                        attempt_id=trade.get("attempt_id") or ((trade.get("setup") or {}).get("attempt_id") if isinstance(trade.get("setup"), dict) else None),
                        position_id=trade.get("position_id") or event_id,
                        order_id=str(order_id),
                        symbol=symbol,
                        endpoint=ORDER_PATH,
                        method="GET",
                        error_code=error_code,
                        message=error_message,
                        error_class="TP_ORDER_QUERY",
                        blocking=False,
                        business_impact="TP_STATE_UNKNOWN",
                        leg=leg.upper(),
                    )
                    continue

                order_status = str(order_info.get("order_status", "")).upper()
                if order_status not in {"PARTIALLY_FILLED", "FILLED"}:
                    continue
                leg_already_hit = leg in hit_legs

                executed_qty = max(0.0, _safe_float(order_info.get("executed_qty", 0.0)))
                previous_qty = max(0.0, _safe_float(filled_by_leg.get(leg, 0.0)))
                delta_qty = max(0.0, executed_qty - previous_qty)

                if delta_qty <= 0:
                    # A status=FILLED response with zero execution remains
                    # unverified.  Do not infer a fill from origQty or from the
                    # disappearance of the position; the fill-history fallback
                    # above must provide real execution evidence first.
                    continue

                exec_price = _safe_float(order_info.get("avg_price"), 0.0)
                if exec_price <= 0:
                    log.warning("[TRACKER_TP] %s %s %s has no actual avgPrice; deferring realized PnL", symbol, direction, leg)
                    continue
                pnl_tp = _calc_trade_pnl_pct(entry_price, exec_price, direction)
                tp_ok, tp_classification = _classify_tp_fill(direction, entry_price, exec_price)
                trade.setdefault("tp_fill_classification", {})[leg] = tp_classification

                # The position snapshot taken before this TP query is already exchange-authoritative.
                # Do not subtract the TP delta from it again: the exchange position may already
                # reflect the fill. Reconcile the live quantity after observing the TP execution.
                try:
                    tp_pos = _timed_tracker_call(
                        tracker_timings, "tp_position_recheck", get_position_directional, symbol, direction,
                    )
                except Exception as tp_pos_exc:
                    tp_pos = {"status": "error", "error": str(tp_pos_exc)}
                    telemetry.record_exchange_error(
                        event_id=event_id,
                        attempt_id=trade.get("attempt_id") or ((trade.get("setup") or {}).get("attempt_id") if isinstance(trade.get("setup"), dict) else None),
                        position_id=trade.get("position_id") or event_id,
                        order_id=str(order_id),
                        symbol=symbol,
                        endpoint=POSITION_PATH,
                        method="GET",
                        error_code=None,
                        message=str(tp_pos_exc),
                        error_class="TP_POSITION_RECONCILIATION",
                        blocking=False,
                        business_impact="TP_STATE_UNKNOWN",
                    )
                if tp_pos.get("status") == "found":
                    rem_qty = abs(_safe_float(tp_pos.get("positionAmt")))
                    trade["current_position_qty"] = rem_qty
                elif tp_pos.get("status") == "not_found":
                    rem_qty = 0.0
                    trade["current_position_qty"] = 0.0
                else:
                    # Keep the last authoritative snapshot if the immediate TP reconciliation failed.
                    log.warning("[TRACKER_TP] %s %s %s immediate position reconcile failed: %s", symbol, direction, leg, tp_pos.get("error"))

                realized_qty += delta_qty
                realized_weighted += delta_qty * pnl_tp
                filled_by_leg[leg] = executed_qty
                if order_status == "FILLED" and not leg_already_hit:
                    # Execution status consumes the target for RR projections even
                    # if its fill is adverse. Keep `hit_legs` reserved for validated
                    # TP hits because it also gates TP1 -> break-even activation.
                    completed_legs = {str(value) for value in (trade.get("completed_tp_legs") or [])}
                    completed_legs.add(leg)
                    trade["completed_tp_legs"] = sorted(completed_legs)
                    if tp_ok:
                        hit_legs.add(leg)
                        trade["hit_legs"] = sorted(hit_legs)
                    trade["tp_filled_qty"] = dict(filled_by_leg)

                trade["remaining_qty"] = rem_qty
                trade["realized_pnl_qty"] = realized_qty
                trade["realized_pnl_weighted_sum"] = realized_weighted
                rr_effective, rr_realized, rr_remaining = _weighted_rr_snapshot(trade)
                trade["effective_weighted_rr"] = rr_effective
                trade["realized_weighted_rr"] = rr_realized
                trade["remaining_weighted_rr"] = rr_remaining
                trade["last_tp_exec_price"] = exec_price
                trade.setdefault("tp_fill_events", []).append({
                    "ts": now_ms,
                    "leg": leg,
                    "order_id": order_id,
                    "order_status": order_status,
                    "executed_qty_total": executed_qty,
                    "delta_qty": delta_qty,
                    "avg_price": exec_price,
                    "pnl_pct": pnl_tp,
                    "remaining_qty": rem_qty,
                    "classification": tp_classification,
                    "execution_evidence_source": order_info.get("execution_evidence_source"),
                    "fill_evidence_status": "CONFIRMED",
                    "fill_time_ms": max((int(_optional_float(f.get("time_ms"))) for f in (order_info.get("fill_history") or []) if _optional_float(f.get("time_ms")) is not None), default=None),
                    "order_update_time_ms": order_info.get("update_time_ms"),
                    "order_time_ms": order_info.get("time_ms"),
                    "trigger_price": _safe_float(order_info.get("trigger_price"), 0.0) or None,
                    "fill_realized_pnl_abs": order_info.get("fill_realized_pnl_abs"),
                    "fill_fee_raw": order_info.get("fill_fee_raw"),
                    "fill_trade_ids": order_info.get("fill_trade_ids", []),
                })
                telemetry.record_protection_event(
                    event_id=event_id,
                    attempt_id=trade.get("attempt_id") or ((trade.get("setup") or {}).get("attempt_id") if isinstance(trade.get("setup"), dict) else None),
                    position_id=trade.get("position_id") or event_id,
                    order_id=str(order_id),
                    symbol=symbol,
                    direction=direction,
                    leg=leg.upper(),
                    status="FILLED" if order_status == "FILLED" else "PARTIALLY_FILLED",
                    executed_qty_total=executed_qty,
                    delta_qty=delta_qty,
                    avg_price=exec_price,
                    pnl_pct=pnl_tp,
                    remaining_qty=rem_qty,
                    order_update_time_ms=order_info.get("update_time_ms"),
                    classification=tp_classification,
                    fill_evidence_status="CONFIRMED",
                    economic_validation_passed=tp_ok,
                )

                if not tp_ok:
                    telemetry.record_state_conflict(
                        event_id=event_id, attempt_id=trade.get("attempt_id"), position_id=trade.get("position_id") or event_id,
                        order_id=str(order_id), symbol=symbol, direction=direction,
                        conflict_type="ADVERSE_TP_FILL",
                        message=f"TP {leg} fill crossed entry: entry={entry_price} fill={exec_price}",
                        owner_event_ids=[event_id], leg=leg, entry_price=entry_price, fill_price=exec_price, pnl_pct=pnl_tp,
                    )

                if order_status == "FILLED" and not leg_already_hit and tp_ok:
                    hit_legs.add(leg)
                    # Persist idempotency markers before side effects that may fail.
                    trade["hit_legs"] = sorted(hit_legs)
                    trade["tp_filled_qty"] = dict(filled_by_leg)
                    rr_effective_now, rr_realized_now, rr_remaining_now = _weighted_rr_snapshot(trade)
                    trade["effective_weighted_rr"] = rr_effective_now
                    trade["realized_weighted_rr"] = rr_realized_now
                    trade["remaining_weighted_rr"] = rr_remaining_now
                    rem_pct = rem_qty / init_qty * 100.0 if init_qty > 0 else 0.0

                    log.info(
                        "[TRACKER_TP_HIT] 💰 %s (%s) Leg: %s | PnL: +%.2f%% | Exec: %.8g | Remaining: %.8f (%.1f%%)",
                        trade.get("name", symbol), symbol, leg, pnl_tp, exec_price, rem_qty, rem_pct
                    )

                    _send_tracker_notification(
                        "TP_HIT",
                        event_id,
                        format_tp_hit_message(
                            name=trade.get("name", symbol),
                            symbol=symbol,
                            leg=leg,
                            pnl_pct=pnl_tp,
                            exec_price=exec_price,
                            # The notification describes the cumulative execution of this TP leg,
                    # not only the increment observed in the latest tracker cycle.
                    closed_qty=executed_qty,
                            remaining_qty=rem_qty,
                            remaining_pct=rem_pct,
                        ),
                        symbol=symbol,
                        leg=leg,
                    )

                    # ПЕРЕНОС В БЕЗУБЫТОК ПОСЛЕ TP1 (audit fix B4: cancel-first swap)
                    if leg == "tp1" and not trade.get("be_activated") and rem_qty > 0:
                        old_sl = trade.get("sl_order", {}) if isinstance(trade.get("sl_order"), dict) else {}
                        old_sl_id = old_sl.get("order_id")
                        old_sl_price = _safe_float(old_sl.get("stop_price"), 0.0) or None
                        new_sl = _timed_tracker_call(
                            tracker_timings, "be_after_tp1", _move_sl_to_break_even,
                            symbol=symbol,
                            direction=direction,
                            entry_price=entry_price,
                            qty=rem_qty,
                            old_sl_id=old_sl_id,
                            trade_id=str(event_id).replace("EVT_", ""),
                            old_sl_price=old_sl_price,
                            owner_event_id=str(event_id),
                        )
                        if new_sl.get("status") in {"created", "created_cleanup_pending"}:
                            be_order_ids = _protection_order_ids([], new_sl)
                            current_trades = _load_active_trades_required()
                            owners = _order_ids_owned_by_other_trades(current_trades, set(be_order_ids), exclude_event_id=event_id)
                            if owners:
                                conflict_ids = sorted(owners)
                                telemetry.record_state_conflict(
                                    event_id=event_id, attempt_id=trade.get("attempt_id"), position_id=trade.get("position_id") or event_id,
                                    order_id=conflict_ids[0] if conflict_ids else None, symbol=symbol, direction=direction,
                                    conflict_type="BE_ORDER_OWNERSHIP_CONFLICT",
                                    message=f"BE order already owned by another active event: {owners}",
                                    owner_event_ids=[event_id] + sorted({eid for ids in owners.values() for eid in ids}),
                                    conflicting_order_ids=conflict_ids, leg="BE_SL",
                                )
                                trade["be_required"] = True
                                trade["be_last_error"] = "BE order ownership conflict"
                            else:
                                trade["sl_order"] = new_sl
                                pending_old_sl_id = new_sl.get("old_sl_cleanup_pending_order_id")
                                if pending_old_sl_id:
                                    trade["sl_cleanup_pending"] = {
                                        "old_order_id": str(pending_old_sl_id),
                                        "since_ts": now_ms,
                                        "last_error": new_sl.get("error"),
                                    }
                                else:
                                    trade.pop("sl_cleanup_pending", None)
                                trade["be_activated"] = True
                                trade["be_trigger_ts"] = trade.get("be_trigger_ts") or _tp1_fill_timestamp_ms(trade, now_ms)
                                trade["be_trigger_peak_r"] = peak_r
                                trade["be_trigger_rule"] = "tp1_filled"
                                trade["be_activation_ts"] = now_ms
                                trade["be_order_id"] = new_sl.get("order_id")
                                trade["be_client_order_id"] = new_sl.get("client_order_id")
                                trade["be_trigger_price"] = _safe_float(new_sl.get("stop_price"), entry_price)
                                telemetry.record_protection_event(
                                    event_id=event_id,
                                    attempt_id=trade.get("attempt_id") or ((trade.get("setup") or {}).get("attempt_id") if isinstance(trade.get("setup"), dict) else None),
                                    position_id=trade.get("position_id") or event_id,
                                    order_id=new_sl.get("order_id"),
                                    symbol=symbol,
                                    direction=direction,
                                    leg="BE_SL",
                                    status="ACTIVATED",
                                    stop_price=_safe_float(new_sl.get("stop_price"), entry_price),
                                    qty=rem_qty,
                                    trigger_ts_ms=now_ms,
                                    trigger_rule="tp1_filled",
                                    trigger_peak_r=peak_r,
                                )
                                log.info(
                                    "[TRACKER_BE_ACTIVATED] %s (%s) TP1 taken. Stop-loss moved to Break-Even: %.8g (Risk: 0.00%%)",
                                    trade.get("name", symbol), symbol, entry_price
                                )
                        else:
                            trade["be_required"] = True
                            trade["be_last_error"] = new_sl.get("error")
                            rollback = new_sl.get("rollback") if isinstance(new_sl.get("rollback"), dict) else None
                            if rollback and str(rollback.get("status", "")).lower() in {"closed_verified", "already_closed"}:
                                trade["be_failure_rollback"] = rollback
                                trade["be_failure_rollback_verified_ts"] = now_ms
                            telemetry.record_protection_event(
                                event_id=event_id,
                                attempt_id=trade.get("attempt_id") or ((trade.get("setup") or {}).get("attempt_id") if isinstance(trade.get("setup"), dict) else None),
                                position_id=trade.get("position_id") or event_id,
                                order_id=new_sl.get("order_id"),
                                symbol=symbol,
                                direction=direction,
                                leg="BE_SL",
                                status="FAILED",
                                stop_price=_safe_float(new_sl.get("stop_price"), entry_price),
                                qty=rem_qty,
                                trigger_ts_ms=now_ms,
                                trigger_rule="tp1_filled",
                                error=new_sl.get("error"),
                                safety_action=new_sl.get("safety_action"),
                            )
                            log.error(
                                "[TRACKER_BE_FAILED] %s (%s) Failed to move SL to Break-Even: %s",
                                trade.get("name", symbol), symbol, new_sl.get("error")
                            )

            trade["hit_legs"] = sorted(hit_legs)
            trade["tp_filled_qty"] = filled_by_leg
            # Recompute after hit_legs is persisted; the earlier per-fill snapshot
            # ran before a newly FILLED leg was added and could count its target twice.
            rr_effective, rr_realized, rr_remaining = _weighted_rr_snapshot(trade)
            trade["effective_weighted_rr"] = rr_effective
            trade["realized_weighted_rr"] = rr_realized
            trade["remaining_weighted_rr"] = rr_remaining
            _clear_completed_tp_recovery_markers(trade)

            # Re-read the position after processing TP order status. A TP can
            # fill between the first position snapshot and the order query loop;
            # the exchange position remains the source of truth for the final
            # residual quantity and for deciding whether the position is gone.
            # Preserve the residual quantity that was still outstanding before
            # the final exchange lookup says the position has disappeared. It is
            # required to reconstruct the PnL of the residual close.
            residual_qty_before_position_disappeared = max(0.0, rem_qty)
            final_pos = _timed_tracker_call(
                tracker_timings, "final_position_recheck", get_position_directional, symbol, direction,
            )
            final_pos_status = str(final_pos.get("status", "")).lower()
            if final_pos_status == "found":
                rem_qty = abs(_safe_float(final_pos.get("positionAmt")))
                trade["current_position_qty"] = rem_qty
                trade["remaining_qty"] = rem_qty
                _clear_position_absence_unverified_if_position_found(
                    trade,
                    exchange_qty=rem_qty,
                    now_ms=now_ms,
                    qty_tolerance=max(1e-12, init_qty * 1e-8),
                )
                final_avg = _safe_float(final_pos.get("avgPrice"))
                if final_avg > 0:
                    trade["last_exchange_avg_price"] = final_avg
                position_gone = False
            elif final_pos_status == "not_found":
                rem_qty = 0.0
                trade["current_position_qty"] = 0.0
                trade["remaining_qty"] = 0.0
                position_gone = True
            else:
                # Do not manufacture a close from an exchange error. Preserve
                # the authoritative last-known quantity and keep the trade open
                # for the next reconciliation cycle.
                position_gone = False
                trade["position_reconcile_error"] = final_pos.get("error")
                telemetry.record_exchange_error(
                    event_id=event_id,
                    attempt_id=trade.get("attempt_id") or ((trade.get("setup") or {}).get("attempt_id") if isinstance(trade.get("setup"), dict) else None),
                    position_id=trade.get("position_id") or event_id,
                    order_id=None,
                    symbol=symbol,
                    endpoint=POSITION_PATH,
                    method="GET",
                    error_code=final_pos.get("code"),
                    message=str(final_pos.get("error") or "position reconciliation error"),
                    error_class="POSITION_RECONCILIATION",
                    blocking=False,
                    business_impact="POSITION_STATE_UNKNOWN",
                )

            telemetry.record_position_reconciliation(
                event_id=event_id,
                attempt_id=trade.get("attempt_id") or ((trade.get("setup") or {}).get("attempt_id") if isinstance(trade.get("setup"), dict) else None),
                position_id=trade.get("position_id") or event_id,
                symbol=symbol,
                direction=direction,
                status=_reconciliation_status(
                    residual_qty_before_position_disappeared,
                    rem_qty if final_pos_status in {"found", "not_found"} else 0.0,
                    "NOT_FOUND" if final_pos_status == "not_found" else "FOUND" if final_pos_status == "found" else "ERROR",
                ),
                internal_remaining_qty=residual_qty_before_position_disappeared,
                exchange_position_qty=rem_qty if final_pos_status in {"found", "not_found"} else None,
                exchange_avg_price=final_pos.get("avgPrice") if final_pos_status == "found" else None,
                local_realized_qty=realized_qty,
                local_tp_filled_qty=dict(filled_by_leg),
                local_be_activated=bool(trade.get("be_activated")),
                local_exit_reason=trade.get("exit_reason"),
                exchange_error=final_pos.get("error") if final_pos_status not in {"found", "not_found"} else None,
                position_gone=position_gone,
            )

            qty_tolerance = max(1e-12, init_qty * 1e-8)
            closed_by_tp = _is_full_tp_close(position_gone, realized_qty, init_qty, hit_legs)
            be_rollback = trade.get("be_failure_rollback") if isinstance(trade.get("be_failure_rollback"), dict) else {}
            verified_emergency_close = trade.get("verified_emergency_close") if isinstance(trade.get("verified_emergency_close"), dict) else {}
            be_rollback_verified = str(be_rollback.get("status", "")).lower() in {"closed_verified", "already_closed"}
            emergency_close_verified = str(verified_emergency_close.get("status", "")).lower() == "closed_verified"
            verified_flatness = be_rollback_verified or emergency_close_verified

            if not position_gone and not closed_by_tp:
                updated_trades[event_id] = trade
                continue

            # Фиксация выхода и закрытие
            duration_min = max(0.0, (now_ms - entry_ts) / 60000.0)
            exit_price = _safe_float(trade.get("last_tp_exec_price"), cur_price)
            sl_order = trade.get("sl_order", {}) if isinstance(trade.get("sl_order"), dict) else {}
            sl_order_id = sl_order.get("order_id")

            sl_exit_price, _ = _timed_tracker_call(
                tracker_timings, "exit_sl_order", _get_exit_from_sl, symbol, sl_order_id,
            )
            # Default the historical source for every exit path. Verified BE
            # rollback and verified emergency-close branches intentionally skip
            # historical-order lookup, but downstream accounting still reads this
            # variable to distinguish a confirmed fill from an estimate.
            hist_source = None
            historical_order = None
            if sl_exit_price is not None:
                exit_price = sl_exit_price
                exit_reason = _classify_confirmed_stop_exit_reason(trade, sl_order_id, sl_order)
            elif closed_by_tp:
                exit_reason = "TAKE_PROFIT_FULL"
            elif position_gone and be_rollback_verified:
                exit_reason = "EMERGENCY_CLOSE_AFTER_BE_FAILURE"
                trade["exit_evidence_source"] = "BE_FAILURE_ROLLBACK_POSITION_ABSENCE_VERIFIED"
            elif position_gone and emergency_close_verified:
                exit_reason = str(verified_emergency_close.get("reason") or "VERIFIED_EMERGENCY_CLOSE")
                trade["exit_evidence_source"] = "EMERGENCY_CLOSE_POSITION_ABSENCE_VERIFIED"
            elif position_gone:
                hist_px, hist_reason, hist_source, historical_order = _timed_tracker_call(
                    tracker_timings, "historical_exit_lookup", _reconcile_historical_exit_order,
                    symbol, direction, entry_ts, residual_qty_before_position_disappeared, trade.get("tp_orders", []), sl_order,
                    trade=trade,
                )
                if hist_px is not None:
                    exit_price = hist_px
                    exit_reason = hist_reason or "MANUAL_CLOSE_RECONCILED"
                    if historical_order:
                        historical_order_id = historical_order.get("orderId") or historical_order.get("orderID")
                        if historical_order_id:
                            trade["exit_order_id"] = str(historical_order_id)
                else:
                    exit_reason = "POSITION_CLOSED_UNVERIFIED"
            else:
                exit_reason = "POSITION_CLOSED_UNVERIFIED"

            # A missing position response is not enough to close a local trade
            # when a confirmed residual quantity remains and no unique residual
            # exit order was found. Keep ownership and protection; do not cancel
            # TP2/BE or emit a false TRADE_CLOSE. The next tracker cycle can
            # reconcile again when the exchange position snapshot recovers.
            expected_residual_qty = max(0.0, init_qty - realized_qty)
            residual_qty_to_preserve = max(
                0.0,
                residual_qty_before_position_disappeared,
                expected_residual_qty,
            )
            qty_tolerance_for_close = max(1e-12, init_qty * 1e-8)
            if (
                position_gone
                and sl_exit_price is None
                and not closed_by_tp
                and not verified_flatness
                and historical_order is None
                and residual_qty_to_preserve > qty_tolerance_for_close
            ):
                first_pending_observation = not bool(trade.get("position_absence_unverified"))
                trade["remaining_qty"] = residual_qty_to_preserve
                trade["current_position_qty"] = residual_qty_to_preserve
                trade["position_absence_unverified"] = True
                trade.setdefault("position_absence_unverified_ts", now_ms)
                trade["position_absence_unverified_last_seen_ts"] = now_ms
                trade["position_absence_unverified_qty"] = residual_qty_to_preserve
                trade["position_absence_unverified_reason"] = "no_confirmed_residual_exit_order"
                if first_pending_observation:
                    telemetry.record_state_conflict(
                        event_id=event_id,
                        attempt_id=trade.get("attempt_id") or ((trade.get("setup") or {}).get("attempt_id") if isinstance(trade.get("setup"), dict) else None),
                        position_id=trade.get("position_id") or event_id,
                        order_id=None,
                        symbol=symbol,
                        direction=direction,
                        conflict_type="POSITION_ABSENCE_WITHOUT_EXIT_EVIDENCE",
                        message=(
                            "Exchange position lookup returned not_found, but a residual quantity remains after "
                            "confirmed fills and no unique post-TP residual exit order was verified; retaining "
                            "active state and protection rather than declaring the trade closed."
                        ),
                        owner_event_ids=[event_id],
                        residual_qty=residual_qty_to_preserve,
                    )
                telemetry.record_position_reconciliation(
                    event_id=event_id,
                    attempt_id=trade.get("attempt_id") or ((trade.get("setup") or {}).get("attempt_id") if isinstance(trade.get("setup"), dict) else None),
                    position_id=trade.get("position_id") or event_id,
                    symbol=symbol,
                    direction=direction,
                    status="POSITION_ABSENCE_UNVERIFIED",
                    internal_remaining_qty=residual_qty_to_preserve,
                    exchange_position_qty=0.0,
                    exchange_avg_price=None,
                    local_tp_filled_qty=dict(filled_by_leg),
                    local_be_activated=bool(trade.get("be_activated")),
                    local_exit_reason=None,
                    position_gone=False,
                    close_evidence_status="MISSING",
                )
                log.error(
                    "[TRACKER_POSITION_ABSENCE_UNVERIFIED] %s (%s) residual=%.12g; keeping active owner and protection",
                    trade.get("name", symbol), symbol, residual_qty_to_preserve,
                )
                updated_trades[event_id] = trade
                continue

            if position_gone:
                telemetry.record_position_reconciliation(
                    event_id=event_id,
                    attempt_id=trade.get("attempt_id") or ((trade.get("setup") or {}).get("attempt_id") if isinstance(trade.get("setup"), dict) else None),
                    position_id=trade.get("position_id") or event_id,
                    symbol=symbol,
                    direction=direction,
                    status="CLOSED",
                    internal_remaining_qty=residual_qty_before_position_disappeared,
                    exchange_position_qty=0.0,
                    exchange_avg_price=None,
                    local_realized_qty=realized_qty,
                    local_tp_filled_qty=dict(filled_by_leg),
                    local_be_activated=bool(trade.get("be_activated")),
                    local_exit_reason=exit_reason,
                    position_gone=True,
                    close_evidence_status=("CONFIRMED" if sl_exit_price is not None or closed_by_tp or historical_order is not None or verified_flatness else "NO_RESIDUAL_QTY"),
                )

            if exit_price <= 0:
                exit_price = cur_price

            residual_exit_verified = not position_gone or residual_qty_before_position_disappeared <= 0
            if sl_exit_price is not None or closed_by_tp:
                residual_exit_verified = True
            if position_gone and residual_qty_before_position_disappeared > 0 and init_qty > 0:
                if closed_by_tp:
                    # Full TP accounting already reconciled the entire initial quantity.
                    # A stale pre-disappearance rem_qty must not be booked a second time.
                    trade["remaining_qty"] = 0.0
                    residual_exit_verified = True
                elif verified_flatness:
                    # Closure is confirmed, but no verified fill price was recovered.
                    # Clear lifecycle ownership without inventing residual realized PnL.
                    trade["remaining_qty"] = 0.0
                    residual_exit_verified = False
                elif historical_order is not None or sl_exit_price is not None:
                    residual_pnl = _calc_trade_pnl_pct(entry_price, exit_price, direction)
                    realized_weighted += residual_qty_before_position_disappeared * residual_pnl
                    realized_qty += residual_qty_before_position_disappeared
                    trade["remaining_qty"] = 0.0
                    residual_exit_verified = True
                else:
                    # Position disappearance is evidence that the position is no longer
                    # open, but it is NOT evidence of which exit leg executed or at
                    # which fill price. Keep the local accounting unmodified and mark
                    # the close as unverified rather than manufacturing realized PnL.
                    residual_exit_verified = False

            # Reconcile actual exchange fills once more at close time.  This is
            # diagnostic/accounting I/O only; it does not affect live entry logic.
            fill_accounting = _timed_tracker_call(
                tracker_timings, "close_fill_accounting", _collect_trade_fill_accounting, trade, now_ms,
            )
            trade["fill_accounting"] = fill_accounting
            close_execution_evidence = _close_execution_evidence(trade, fill_accounting)
            trade["close_execution_evidence"] = close_execution_evidence
            trade["entry_fee_raw"] = fill_accounting.get("entry_fee_raw")
            trade["exit_fee_raw"] = fill_accounting.get("exit_fee_raw")
            trade["fee_raw_total"] = fill_accounting.get("fee_raw_total")
            trade["fees_paid_abs"] = fill_accounting.get("fees_paid_abs")
            trade["exchange_realized_pnl_abs"] = fill_accounting.get("exchange_realized_pnl_abs")
            trade["exit_vwap_confirmed"] = fill_accounting.get("exit_vwap_confirmed")
            trade["exit_vwap_source"] = fill_accounting.get("exit_vwap_source")
            trade["exit_leg_vwap_confirmed"] = close_execution_evidence.get("exit_vwap_confirmed")
            trade["exit_leg_vwap_source"] = close_execution_evidence.get("exit_vwap_source")
            trade["fee_source"] = fill_accounting.get("fee_source")
            if fill_accounting.get("last_exit_fill_ts_ms"):
                trade["exchange_close_ts"] = int(fill_accounting["last_exit_fill_ts_ms"])

            accounting_diff_qty = realized_qty - init_qty
            confirmed_exit_fill_qty = fill_accounting.get("confirmed_exit_qty")
            close_fill_qty_verified = (
                confirmed_exit_fill_qty is not None
                and abs(float(confirmed_exit_fill_qty) - init_qty) <= qty_tolerance
            )
            # A missing residual price at the position snapshot must not keep the
            # whole trade permanently unverified when the close-time execution
            # ledger now proves every entry and exit fill.  Upgrade only on exact
            # quantity coverage, complete per-order evidence, and a confirmed
            # entry VWAP + gross realized PnL; otherwise remain fail-closed.
            confirmed_entry_qty = _optional_float(fill_accounting.get("entry_qty_confirmed"))
            confirmed_entry_vwap = _optional_float(fill_accounting.get("entry_vwap_confirmed"))
            confirmed_exit_vwap = _optional_float(fill_accounting.get("exit_vwap_confirmed"))
            confirmed_gross_pnl_abs = _optional_float(fill_accounting.get("gross_realized_pnl_abs"))
            incomplete_order_ids = fill_accounting.get("incomplete_order_fill_ids")
            fill_evidence_status = str(fill_accounting.get("status") or "").upper()
            complete_fill_based_close = bool(
                position_gone
                and residual_qty_before_position_disappeared > 0
                and not residual_exit_verified
                and fill_evidence_status in {"VERIFIED", "VERIFIED_ORDER_EXECUTION"}
                and confirmed_entry_qty is not None
                and abs(confirmed_entry_qty - init_qty) <= qty_tolerance
                and confirmed_entry_vwap is not None and confirmed_entry_vwap > 0
                and close_fill_qty_verified
                and confirmed_exit_vwap is not None and confirmed_exit_vwap > 0
                and confirmed_gross_pnl_abs is not None
                and not (isinstance(incomplete_order_ids, list) and incomplete_order_ids)
            )
            if complete_fill_based_close:
                verified_full_trade_pnl_pct = (
                    confirmed_gross_pnl_abs / (confirmed_entry_vwap * confirmed_entry_qty)
                ) * 100.0
                if math.isfinite(verified_full_trade_pnl_pct):
                    residual_exit_verified = True
                    realized_qty = init_qty
                    realized_weighted = verified_full_trade_pnl_pct * init_qty
                    accounting_diff_qty = 0.0
                    trade["remaining_qty"] = 0.0
                    exit_price = confirmed_exit_vwap
                    close_execution_evidence["authority"] = "EXCHANGE_CONFIRMED_COMPLETE_FILL_ACCOUNTING"
                    close_execution_evidence["confirmed_exit_qty"] = float(confirmed_exit_fill_qty)
                    close_execution_evidence["exit_vwap_confirmed"] = float(confirmed_exit_vwap)
                    close_execution_evidence["exit_vwap_source"] = fill_accounting.get("exit_vwap_source")
            if not residual_exit_verified:
                accounting_status = "UNVERIFIED_RESIDUAL_EXIT"
            elif close_fill_qty_verified and abs(accounting_diff_qty) <= qty_tolerance:
                accounting_status = "VERIFIED"
            elif close_fill_qty_verified and abs(accounting_diff_qty) > qty_tolerance:
                accounting_status = "LOCAL_VS_EXCHANGE_QTY_MISMATCH"
            elif realized_qty > 0:
                accounting_status = "PARTIAL_REALIZED_QTY"
            else:
                accounting_status = "NO_CONFIRMED_REALIZED_QTY"
            trade["realized_pnl_reconciliation"] = {
                "status": accounting_status,
                "initial_qty": init_qty,
                "confirmed_realized_qty": realized_qty,
                "confirmed_exit_fill_qty": confirmed_exit_fill_qty,
                "difference_qty": accounting_diff_qty,
                "exit_fill_difference_qty": (float(confirmed_exit_fill_qty) - init_qty) if confirmed_exit_fill_qty is not None else None,
                "quantity_tolerance": qty_tolerance,
                "weighted_pnl_pct": (realized_weighted / realized_qty) if realized_qty > 0 else None,
            }

            partial_pnl_contribution = (realized_weighted / init_qty) if (init_qty > 0 and realized_qty > 0) else None
            final_pnl = (
                partial_pnl_contribution if residual_exit_verified and partial_pnl_contribution is not None
                else current_pnl if residual_exit_verified else None
            )
            realized_pnl_source = (
                "UNVERIFIED_RESIDUAL_EXIT" if not residual_exit_verified
                else "confirmed_fill_accounting" if complete_fill_based_close
                else "executed_tp_or_sl" if (closed_by_tp or sl_exit_price is not None)
                else (hist_source or "observation_price_estimate")
            )
            exit_price_semantics = "UNAVAILABLE"
            if closed_by_tp and realized_qty > 0 and sl_exit_price is None:
                confirmed_vwap = close_execution_evidence.get("exit_vwap_confirmed")
                if confirmed_vwap is not None and close_execution_evidence.get("confirmed_exit_qty") is not None:
                    exit_price = float(confirmed_vwap)
                    exit_price_semantics = (
                        "CONFIRMED_EXIT_VWAP"
                        if str(close_execution_evidence.get("authority")) != "TRACKER_LOCAL_EVENT"
                        else "TRACKER_RECORDED_TP_EVENT_VWAP"
                    )
                else:
                    # Backward-compatible price for consumers that still expect a scalar
                    # exit_price, but explicitly mark it as a tracker-weighted estimate.
                    exit_price = entry_price * (1.0 + final_pnl / 100.0) if direction == "LONG" else entry_price * (1.0 - final_pnl / 100.0)
                    exit_price_semantics = "TRACKER_WEIGHTED_EXIT_ESTIMATE"
            elif complete_fill_based_close:
                exit_price = float(fill_accounting["exit_vwap_confirmed"])
                exit_price_semantics = "CONFIRMED_EXIT_VWAP"
            elif sl_exit_price is not None:
                exit_price_semantics = "SINGLE_CONFIRMED_EXIT_FILL"
            elif historical_order is not None:
                exit_price_semantics = "SINGLE_CONFIRMED_HISTORICAL_ORDER"
            else:
                exit_price_semantics = "OBSERVED_PRICE_ESTIMATE"
            planned_risk_pct = _derive_planned_risk_pct(trade)
            realized_rr = _calc_realized_rr(final_pnl, planned_risk_pct) if final_pnl is not None and realized_pnl_source in {"executed_tp_or_sl", "confirmed_fill_accounting"} else None
            final_weighted_rr = _calc_realized_rr(final_pnl, planned_risk_pct) if final_pnl is not None and planned_risk_pct and planned_risk_pct > 0 else None
            if final_pnl is not None:
                projected_rr, realized_rr_component, remaining_rr_component = _weighted_rr_snapshot({**trade, "remaining_qty": 0.0, "realized_pnl_qty": init_qty, "realized_pnl_weighted_sum": final_pnl * init_qty, "hit_legs": hit_legs})
            else:
                projected_rr = realized_rr_component = remaining_rr_component = None
            effective_rr_at_close = final_weighted_rr if final_weighted_rr is not None else projected_rr
            trade["effective_weighted_rr"] = effective_rr_at_close
            trade["realized_weighted_rr"] = final_weighted_rr
            trade["remaining_weighted_rr"] = None
            # Keep the planned ratio separate from realized/effective R:R.
            # The old assignment displayed the effective result as "Planned".
            planned_rr = _safe_float(trade.get("planned_weighted_rr"), DEFAULT_PLANNED_WEIGHTED_RR)

            # For BE/SL exits, retain exchange-confirmed trigger/fill telemetry
            # so the next audit can measure actual stop degradation rather than
            # infer it from PnL.
            exit_order_info = (
                _timed_tracker_call(tracker_timings, "exit_order_evidence", _get_filled_order, symbol, sl_order_id)
                if sl_order_id else None
            )
            if exit_order_info:
                actual_exit_fill = _safe_float(exit_order_info.get("avg_price"), 0.0)
                trigger_px = _safe_float(
                    exit_order_info.get("trigger_price")
                    or exit_order_info.get("stop_price")
                    or sl_order.get("trigger_price")
                    or sl_order.get("stop_price"),
                    0.0,
                )
                if actual_exit_fill > 0:
                    trade["exit_order_id"] = exit_order_info.get("order_id") or sl_order_id
                    trade["exit_order_avg_price"] = actual_exit_fill
                    trade["exit_order_trigger_price"] = trigger_px if trigger_px > 0 else None
                    if trigger_px > 0:
                        trade["exit_order_adverse_slippage_pct"] = _adverse_exit_slippage_pct(
                            direction, actual_exit_fill, trigger_px
                        )
                    if trade.get("be_activated"):
                        trade["be_order_id"] = trade.get("be_order_id") or sl_order_id
                        trade["be_trigger_price"] = trade.get("be_trigger_price") or (trigger_px if trigger_px > 0 else entry_price)
                        trade["be_fill_price"] = actual_exit_fill
                        be_trigger_px = _safe_float(trade.get("be_trigger_price"), entry_price)
                        if be_trigger_px > 0:
                            trade["be_execution_slippage_pct"] = _adverse_exit_slippage_pct(
                                direction, actual_exit_fill, be_trigger_px
                            )

            # For TP closes, surface the latest confirmed TP leg as the canonical exit-order
            # evidence while keeping all legs in tp_fill_events.  For historical closes,
            # surface the normalized exchange order used to reconcile the residual.
            if close_execution_evidence.get("latest"):
                latest_exit = close_execution_evidence["latest"]
                trade["exit_order_id"] = latest_exit.get("order_id") or trade.get("exit_order_id")
                trade["exit_order_avg_price"] = _safe_float(latest_exit.get("avg_price"), 0.0) or None
                trade["exit_order_trigger_price"] = _safe_float(latest_exit.get("trigger_price"), 0.0) or None
                if trade.get("exit_order_trigger_price") is not None and trade.get("exit_order_avg_price") is not None:
                    trade["exit_order_adverse_slippage_pct"] = _adverse_exit_slippage_pct(
                        direction, float(trade["exit_order_avg_price"]), float(trade["exit_order_trigger_price"])
                    )
            elif historical_order is not None:
                trade["exit_order_evidence"] = {
                    "order_id": str(historical_order.get("orderId") or historical_order.get("orderID") or "") or None,
                    "status": str(historical_order.get("status") or historical_order.get("orderStatus") or "FILLED").upper(),
                    "type": str(historical_order.get("type") or "").upper() or None,
                    "side": str(historical_order.get("side") or "").upper() or None,
                    "position_side": str(historical_order.get("positionSide") or "").upper() or None,
                    "executed_qty": _optional_float(historical_order.get("executedQty") or historical_order.get("cumQty") or historical_order.get("_qty")),
                    "avg_price": _optional_float(historical_order.get("avgPrice") or historical_order.get("_px")),
                    "trigger_price": _optional_float(historical_order.get("stopPrice")),
                    "update_time_ms": _optional_float(historical_order.get("updateTime") or historical_order.get("time") or historical_order.get("createTime") or historical_order.get("_ts")),
                    "source": "HISTORICAL_ALL_ORDERS",
                    "fill_evidence_status": "ORDER_EXECUTION_CONFIRMED",
                }

            tp_actual_close_fractions: dict[str, float] = {}
            if init_qty > 0:
                for tp_event in trade.get("tp_fill_events", []) if isinstance(trade.get("tp_fill_events"), list) else []:
                    leg = str(tp_event.get("leg") or "").lower()
                    delta = _optional_float(tp_event.get("delta_qty"))
                    if leg and delta is not None and delta > 0:
                        tp_actual_close_fractions[leg] = tp_actual_close_fractions.get(leg, 0.0) + float(delta) / init_qty
            trade["tp_actual_close_fractions"] = tp_actual_close_fractions
            if trade.get("exit_order_evidence") is None and close_execution_evidence.get("latest"):
                latest_exit = close_execution_evidence["latest"]
                trade["exit_order_evidence"] = {
                    "order_id": latest_exit.get("order_id"),
                    "status": latest_exit.get("status"),
                    "leg": latest_exit.get("leg"),
                    "executed_qty": latest_exit.get("executed_qty"),
                    "avg_price": latest_exit.get("avg_price"),
                    "trigger_price": latest_exit.get("trigger_price"),
                    "fill_time_ms": latest_exit.get("fill_time_ms"),
                    "order_update_time_ms": latest_exit.get("order_update_time_ms"),
                    "source": latest_exit.get("evidence_source"),
                    "fill_evidence_status": "CONFIRMED",
                }
            elif trade.get("exit_order_evidence") is None:
                trade["exit_order_evidence"] = None

            if trade.get("be_trigger_ts") and entry_ts:
                trade["time_to_be_trigger_min"] = max(0.0, (int(trade["be_trigger_ts"]) - entry_ts) / 60000.0)
            if trade.get("be_activation_ts") and entry_ts:
                trade["time_to_be_activation_min"] = max(0.0, (int(trade["be_activation_ts"]) - entry_ts) / 60000.0)

            trade["remaining_qty"] = 0.0
            trade["realized_pnl_pct_confirmed_partial"] = partial_pnl_contribution if not residual_exit_verified else None
            trade["realized_pnl_pct"] = final_pnl
            trade["realized_pnl_qty"] = realized_qty
            trade["realized_pnl_weighted_sum"] = realized_weighted
            trade["realized_rr"] = realized_rr
            trade["realized_pnl_source"] = realized_pnl_source
            trade["exit_price"] = exit_price
            trade["exit_reason"] = exit_reason

            # Canonical absolute accounting is fill-evidence based only.  An exchange
            # position disappearing without an identified exit fill can still close
            # the tracker state, but it cannot produce a canonical account PnL number.
            canonical_net = (
                fill_accounting.get("net_realized_pnl_abs")
                if close_fill_qty_verified and residual_exit_verified
                else None
            )
            trade["gross_realized_pnl_abs"] = (
                fill_accounting.get("gross_realized_pnl_abs")
                if close_fill_qty_verified and residual_exit_verified
                else None
            )
            trade["gross_realized_pnl_source"] = (
                fill_accounting.get("gross_realized_pnl_source")
                if trade["gross_realized_pnl_abs"] is not None
                else "UNAVAILABLE"
            )
            trade["exchange_realized_pnl_abs"] = (
                fill_accounting.get("exchange_realized_pnl_abs")
                if close_fill_qty_verified and residual_exit_verified
                else None
            )
            trade["exchange_realized_pnl_source"] = (
                fill_accounting.get("exchange_realized_pnl_source")
                if trade["exchange_realized_pnl_abs"] is not None
                else "UNAVAILABLE"
            )
            trade["fees_abs"] = (
                fill_accounting.get("fees_paid_abs")
                if fill_accounting.get("fee_raw_total") is not None and close_fill_qty_verified and residual_exit_verified
                else None
            )
            trade["fees_source"] = fill_accounting.get("fee_source") if trade["fees_abs"] is not None else "UNAVAILABLE"
            trade["net_realized_pnl_abs"] = canonical_net
            trade["net_realized_pnl_source"] = (
                fill_accounting.get("net_realized_pnl_source")
                if canonical_net is not None
                else "UNAVAILABLE"
            )
            trade["realized_pnl_abs"] = canonical_net
            trade["realized_pnl_abs_source"] = "NET_CONFIRMED_FILL_ACCOUNTING" if canonical_net is not None else "UNAVAILABLE"
            trade["realized_pnl_abs_status"] = "CONFIRMED" if canonical_net is not None else "UNAVAILABLE"

            observed_closed_ts = now_ms
            execution_close_candidates: list[tuple[int, str]] = []
            fill_ts = _optional_float(fill_accounting.get("last_exit_fill_ts_ms"))
            if fill_ts is not None and int(fill_ts) > 0:
                execution_close_candidates.append((int(fill_ts), "EXCHANGE_FILL_TIME"))
            elif close_execution_evidence.get("latest_execution_ts_ms"):
                evidence_ts = int(close_execution_evidence["latest_execution_ts_ms"])
                evidence_source = str(close_execution_evidence.get("latest_execution_timestamp_source") or "UNAVAILABLE")
                execution_close_candidates.append((
                    evidence_ts,
                    "EXCHANGE_FILL_TIME" if evidence_source == "FILL_TIME"
                    else "EXCHANGE_ORDER_UPDATE_TIME" if evidence_source == "ORDER_UPDATE_TIME"
                    else "TP_FILL_EVENT_EXCHANGE_TIME",
                ))
            if exit_order_info and exit_order_info.get("update_time_ms"):
                execution_close_candidates.append((int(exit_order_info["update_time_ms"]), "EXCHANGE_ORDER_UPDATE_TIME"))
            if historical_order:
                hist_ts = historical_order.get("updateTime") or historical_order.get("time") or historical_order.get("createTime")
                hist_ts_num = _optional_float(hist_ts)
                if hist_ts_num is not None and int(hist_ts_num) > 0:
                    execution_close_candidates.append((int(hist_ts_num), "EXCHANGE_HISTORICAL_ORDER_TIME"))
            execution_closed_ts, execution_closed_ts_source = (
                max(execution_close_candidates, key=lambda item: item[0])
                if execution_close_candidates else (None, "UNAVAILABLE")
            )
            trade["execution_close_ts"] = execution_closed_ts
            trade["observed_closed_ts"] = observed_closed_ts
            trade["close_timestamp_source"] = execution_closed_ts_source
            # Keep the legacy field only when execution-time evidence exists. Never
            # substitute local observation time into canonical close time.
            trade["closed_ts"] = execution_closed_ts
            trade["duration_min"] = (
                max(0.0, (execution_closed_ts - entry_ts) / 60000.0)
                if execution_closed_ts is not None else None
            )
            closed_ts = execution_closed_ts
            trade["observed_duration_min"] = max(0.0, (observed_closed_ts - entry_ts) / 60000.0)
            trade["observed_pnl_pct_estimate"] = (
                _calc_trade_pnl_pct(entry_price, cur_price, direction)
                if not residual_exit_verified else None
            )
            emoji = "💚" if final_pnl is not None and final_pnl >= 0.0 else "💔" if final_pnl is not None else "⚪"
            pending_cancel_ids = []
            for tp in (trade.get("tp_orders") if isinstance(trade.get("tp_orders"), list) else []):
                if not isinstance(tp, dict):
                    continue
                if str(tp.get("leg", "")) not in hit_legs and tp.get("order_id"):
                    pending_cancel_ids.append(str(tp["order_id"]))
            if sl_order_id:
                pending_cancel_ids.append(str(sl_order_id))
            if pending_cancel_ids:
                cleanup_ok, cleanup_error = _cancel_engine_protection_orders_verified(
                    symbol, direction, pending_cancel_ids,
                )
                if not cleanup_ok:
                    trade["protection_cleanup_pending"] = {
                        "order_ids": list(dict.fromkeys(pending_cancel_ids)),
                        "error": str(cleanup_error or "protection cancellation could not be verified"),
                        "last_attempt_ts": int(time.time() * 1000),
                    }
                    updated_trades[event_id] = trade
                    log.critical(
                        "[TRACKER_CLOSE_CLEANUP_PENDING] event=%s %s %s; retaining active local state: %s",
                        event_id, direction, symbol, cleanup_error,
                    )
                    continue
            trade.pop("protection_cleanup_pending", None)

            close_record = {
                "record_type": "TRADE_CLOSE",
                "event_id": event_id,
                "symbol": symbol,
                "direction": direction,
                "event_type": trade.get("event_type"),
                "closed_ts": closed_ts,
                "execution_close_ts": trade.get("execution_close_ts"),
                "observed_closed_ts": observed_closed_ts,
                "close_timestamp_source": trade.get("close_timestamp_source"),
                "exit_reason": exit_reason,
                "outcome_category": _exit_outcome_category(exit_reason),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_price_semantics": exit_price_semantics,
                "exit_vwap_confirmed": fill_accounting.get("exit_vwap_confirmed"),
                "exit_vwap_source": fill_accounting.get("exit_vwap_source"),
                "exit_leg_vwap_confirmed": close_execution_evidence.get("exit_vwap_confirmed"),
                "exit_leg_vwap_source": close_execution_evidence.get("exit_vwap_source"),
                "realized_pnl_pct": final_pnl,
                "realized_pnl_pct_confirmed_partial": partial_pnl_contribution if not residual_exit_verified else None,
                "realized_rr": realized_rr,
                "realized_pnl_source": realized_pnl_source,
                "effective_weighted_rr": _safe_float(trade.get("effective_weighted_rr"), planned_rr),
                "planned_weighted_rr": _safe_float(trade.get("planned_weighted_rr"), planned_rr),
                "realized_weighted_rr": final_weighted_rr,
                "remaining_weighted_rr": None,
                "tp_mode": trade.get("tp_mode", "multi_tp"),
                "strategy_version": trade.get("strategy_version"),
                "code_commit_sha": trade.get("code_commit_sha"),
                "be_trigger_rule": trade.get("be_trigger_rule", "after_tp1_filled"),
                "entry_bar": trade.get("entry_bar", {}),
                "previous_bar": trade.get("previous_bar", {}),
                "signal_snapshot": trade.get("signal_snapshot", {}),
                "entry_order": trade.get("entry_order", {}),
                "fill_position": trade.get("fill_position", {}),
                "execution_snapshot": trade.get("execution_snapshot", {}),
                "tp_fill_events": trade.get("tp_fill_events", []),
                "effective_tp_levels": trade.get("effective_tp_levels", []),
                "peak_pnl_pct": _safe_float(trade.get("peak_pnl_pct")),
                "mae_pct": _safe_float(trade.get("mae_pct")),
                "max_drawdown_pct": _safe_float(trade.get("max_drawdown_pct")),
                "duration_min": trade.get("duration_min"),
                "observed_duration_min": trade.get("observed_duration_min"),
                "observed_pnl_pct_estimate": trade.get("observed_pnl_pct_estimate"),
                "hit_legs": sorted(hit_legs),
                "tp_filled_qty": filled_by_leg,
                "be_activated": bool(trade.get("be_activated")),
                "be_trigger_ts": trade.get("be_trigger_ts"),
                "be_trigger_peak_r": trade.get("be_trigger_peak_r"),
                "be_order_id": trade.get("be_order_id"),
                "be_trigger_price": trade.get("be_trigger_price"),
                "be_fill_price": trade.get("be_fill_price"),
                "be_execution_slippage_pct": trade.get("be_execution_slippage_pct"),
                "time_to_be_trigger_min": trade.get("time_to_be_trigger_min"),
                "time_to_be_activation_min": trade.get("time_to_be_activation_min"),
                "mfe_milestones_r": trade.get("mfe_milestones_r", {}),
                "exit_order_id": trade.get("exit_order_id"),
                "exit_order_avg_price": trade.get("exit_order_avg_price"),
                "exit_order_trigger_price": trade.get("exit_order_trigger_price"),
                "exit_order_adverse_slippage_pct": trade.get("exit_order_adverse_slippage_pct"),
                "exit_order_evidence": trade.get("exit_order_evidence"),
                "tp_actual_close_fractions": trade.get("tp_actual_close_fractions", {}),
                "close_execution_evidence": close_execution_evidence,
                "fill_accounting": fill_accounting,
                "realized_pnl_reconciliation": trade.get("realized_pnl_reconciliation"),
                "entry_fee_raw": trade.get("entry_fee_raw"),
                "exit_fee_raw": trade.get("exit_fee_raw"),
                "fee_raw_total": trade.get("fee_raw_total"),
                "fees_paid_abs": trade.get("fees_paid_abs"),
                "fees_abs": trade.get("fees_abs"),
                "fees_source": trade.get("fees_source"),
                "gross_realized_pnl_abs": trade.get("gross_realized_pnl_abs"),
                "gross_realized_pnl_source": trade.get("gross_realized_pnl_source"),
                "exchange_realized_pnl_abs": trade.get("exchange_realized_pnl_abs"),
                "exchange_realized_pnl_source": trade.get("exchange_realized_pnl_source"),
                "net_realized_pnl_abs": trade.get("net_realized_pnl_abs"),
                "net_realized_pnl_source": trade.get("net_realized_pnl_source"),
                "realized_pnl_abs": trade.get("realized_pnl_abs"),
                "realized_pnl_abs_source": trade.get("realized_pnl_abs_source"),
                "realized_pnl_abs_status": trade.get("realized_pnl_abs_status"),
                "research": trade.get("research", {}),
                "setup": trade.get("setup", {}),
            }
            # Journal first.  If serialization or disk I/O fails, the trade remains
            # open in local state and the next cycle can retry without losing the
            # close event.  If another process already wrote the close, treating the
            # idempotent duplicate as success is safe.
            _append_trade_close_once(close_record)
            trade["closed"] = True

            duration_min = trade["duration_min"] if trade["duration_min"] is not None else trade["observed_duration_min"]
            realized_rr_log = f"{realized_rr:.3f}" if realized_rr is not None else "—"
            if final_pnl is None:
                log.info(
                    _TRACKER_TRADE_CLOSED_UNCONFIRMED_LOG_FORMAT,
                    emoji, trade.get("name", symbol), symbol, _pnl_log_value(final_pnl),
                    realized_rr_log, planned_rr, exit_price, exit_reason, duration_min,
                )
            else:
                log.info(
                    _TRACKER_TRADE_CLOSED_LOG_FORMAT,
                    emoji, trade.get("name", symbol), symbol, final_pnl,
                    realized_rr_log, planned_rr, exit_price, exit_reason, duration_min,
                )

            _send_tracker_notification(
                "TRADE_CLOSE",
                event_id,
                format_trade_closed_message(
                    name=trade.get("name", symbol),
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pct=final_pnl,
                    realized_rr=realized_rr,
                    planned_rr=planned_rr,
                    duration_min=duration_min,
                    peak_pnl=_safe_float(trade.get("peak_pnl_pct")),
                    max_drawdown=_safe_float(trade.get("max_drawdown_pct")),
                    exit_reason=exit_reason,
                    event_type=trade.get("event_type", "DIVERGENCE"),
                    timeframe=trade.get("timeframe") or (trade.get("setup") or {}).get("event_timeframe") or "1h",
                ),
                symbol=symbol,
            )

        except Exception as exc:
            log.exception("[TRACKER] Fatal trade error for event %s: %s", event_id, exc)
            updated_trades[event_id] = trade

    try:
        _timed_tracker_call(
            tracker_timings, "active_state_save", _save_active_trades_after_reconciliation,
            original_trades, updated_trades,
        )
    finally:
        tp_read_executor.shutdown(wait=True)
        if tracker_perf_started is not None:
            _log_tracker_timings(
                tracker_timings, time.perf_counter() - tracker_perf_started,
                sum(1 for trade in trades.values() if isinstance(trade, dict) and not trade.get("closed", False)),
            )
