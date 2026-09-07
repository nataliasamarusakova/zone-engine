from __future__ import annotations

import json
import logging
import os
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
    get_order,
    get_all_orders,
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
    position_side_param,
    _validate_sl_order_for_position,
)
from event_engine.telegram import send as send_tg

log = logging.getLogger("event_engine.tracker")

DATA = Path("data")
ACTIVE_TRADES_PATH = DATA / "active_trades.json"
TRADES_PATH = DATA / "trades.jsonl"
ACTIONS_PATH = DATA / "actions.jsonl"


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


def _load_active_trades() -> dict[str, dict]:
    if not ACTIVE_TRADES_PATH.exists():
        return {}
    try:
        data = json.loads(ACTIVE_TRADES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            normalized = {}
            for event_id, trade in data.items():
                if not isinstance(trade, dict):
                    continue
                t = dict(trade)
                t.setdefault("mae_pct", 0.0)
                t.setdefault("max_drawdown_pct", 0.0)
                t.setdefault("be_required", False)
                t.setdefault("be_last_error", None)
                t.setdefault("tp_mode", "single_tp" if len(t.get("tp_orders", [])) == 1 else "multi_tp")
                t.setdefault("effective_tp_levels", t.get("tp_levels", []))
                t.setdefault("effective_weighted_rr", t.get("planned_weighted_rr", 0.75))
                normalized[str(event_id)] = t
            return normalized
        log.error("[TRACKER] Invalid state: %s is not a JSON object", ACTIVE_TRADES_PATH)
        return {}
    except Exception as exc:
        log.error("[TRACKER] Corrupt state in %s: %s", ACTIVE_TRADES_PATH, exc)
        return {}


def _save_active_trades(trades: dict[str, dict]) -> None:
    """Atomically persist active-trade state and serialize concurrent writers."""
    ACTIVE_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ACTIVE_TRADES_PATH.with_name(ACTIVE_TRADES_PATH.name + ".tmp")
    lock_path = ACTIVE_TRADES_PATH.with_suffix(ACTIVE_TRADES_PATH.suffix + ".lock")
    payload = json.dumps(trades, ensure_ascii=False, indent=2)
    with lock_path.open("a+") as lockf:
        if fcntl is not None:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, ACTIVE_TRADES_PATH)
        finally:
            if fcntl is not None:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


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

def _append_trade_record(record: dict) -> None:
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _JOURNAL_LOCK:
        with TRADES_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_action_record(record: dict) -> None:
    ACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _JOURNAL_LOCK:
        with ACTIONS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
        _append_action_record({
            "ts": int(time.time() * 1000),
            "action": kind,
            "event_id": str(event_id),
            "symbol": symbol,
            "leg": leg,
            "telegram_ok": False,
            "error": str(exc),
        })
        return False

    _append_action_record({
        "ts": int(time.time() * 1000),
        "action": kind,
        "event_id": str(event_id),
        "symbol": symbol,
        "leg": leg,
        "telegram_ok": ok,
    })
    if not ok:
        log.error("[TG_%s_FAILED] event=%s symbol=%s leg=%s send() returned False", kind, event_id, symbol, leg or "")
    else:
        log.info("[TG_%s_SENT] event=%s symbol=%s leg=%s", kind, event_id, symbol, leg or "")
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
                    wf.write(json.dumps(record, ensure_ascii=False) + "\n")
                return True
            finally:
                if fcntl is not None:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _normalize_direction(direction: str) -> str:
    d = str(direction or "").upper()
    if d not in {"LONG", "SHORT"}:
        raise ValueError(f"Invalid direction={direction}")
    return d


def _normalized_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace("-USDT", "")


def update_active_trade_protection(
    symbol: str,
    direction: str,
    tp_orders: list[dict],
    sl_result: dict,
    effective_tp_levels: list[dict] | None = None,
    tp_mode: str | None = None,
    effective_weighted_rr: float | None = None,
) -> bool:
    trades = _load_active_trades()
    want_bx = to_bx_symbol(symbol) or _normalized_symbol(symbol)
    want_direction = str(direction).upper()

    for trade in trades.values():
        if trade.get("closed", False):
            continue
        trade_bx = to_bx_symbol(trade.get("symbol", "")) or _normalized_symbol(trade.get("symbol", ""))
        trade_direction = str(trade.get("direction", "")).upper()

        if trade_bx == want_bx and trade_direction == want_direction:
            trade["tp_orders"] = tp_orders if isinstance(tp_orders, list) else []
            trade["sl_order"] = sl_result if isinstance(sl_result, dict) else {}
            if effective_tp_levels is not None:
                trade["effective_tp_levels"] = effective_tp_levels
            if tp_mode:
                trade["tp_mode"] = tp_mode
            if effective_weighted_rr is not None:
                trade["effective_weighted_rr"] = _safe_float(effective_weighted_rr, 0.75)
            trade["protection_last_updated_ts"] = int(time.time() * 1000)
            _save_active_trades(trades)
            return True

    return False


def _extract_setup_metrics(setup: dict | None) -> dict[str, Any]:
    if not isinstance(setup, dict):
        return {
            "planned_risk_pct": None,
            "planned_target_rr": None,
            "planned_weighted_rr": 0.75,
            "entry_reference": None,
            "invalidation_price": None,
            "target_price": None,
            "tp_levels": [],
            "effective_tp_levels": [],
            "effective_weighted_rr": 0.75,
            "tp_mode": "multi_tp",
        }

    return {
        "planned_risk_pct": _safe_float(setup.get("risk_pct"), 0.0) if setup.get("risk_pct") is not None else None,
        "planned_target_rr": _safe_float(setup.get("target_rr"), 0.0) if setup.get("target_rr") is not None else None,
        "planned_weighted_rr": _safe_float(setup.get("planned_weighted_rr", 0.75), 0.75),
        "effective_tp_levels": setup.get("effective_tp_levels") if isinstance(setup.get("effective_tp_levels"), list) else [],
        "effective_weighted_rr": _safe_float(setup.get("effective_weighted_rr", setup.get("planned_weighted_rr", 0.75)), 0.75),
        "tp_mode": str(setup.get("tp_mode", "multi_tp")),
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
) -> None:
    direction = _normalize_direction(direction)
    trades = _load_active_trades()
    now_ms = int(time.time() * 1000)
    actual_entry_ts = int(entry_ts_ms) if entry_ts_ms is not None and int(entry_ts_ms) > 0 else now_ms

    actual_entry_price = _safe_float(entry_price)
    actual_qty = abs(_safe_float(qty))

    if actual_entry_price <= 0 or actual_qty <= 0:
        raise ValueError(f"Cannot register invalid position: entry_price={actual_entry_price} qty={actual_qty}")

    setup_metrics = _extract_setup_metrics(setup)
    research: dict[str, Any] = {"source": "BingX 1H demand_supply zone engine"}

    requested_price = _safe_float(requested_entry_price, 0.0) if requested_entry_price is not None else setup_metrics["entry_reference"]
    signal_reference_price = _safe_float((setup or {}).get("signal_price"), 0.0) if isinstance(setup, dict) else 0.0
    if signal_reference_price <= 0:
        signal_reference_price = _safe_float(setup_metrics.get("entry_reference"), 0.0)
    pre_order_reference_price = _safe_float((setup or {}).get("pre_order_reference_price"), 0.0) if isinstance(setup, dict) else 0.0
    entry_slippage_pct = None
    adverse_entry_slippage_pct = None

    if requested_price is not None and requested_price > 0:
        entry_slippage_pct = (actual_entry_price - requested_price) / requested_price * 100.0
        if direction == "LONG":
            adverse_entry_slippage_pct = max(0.0, entry_slippage_pct)
        else:
            adverse_entry_slippage_pct = max(0.0, -entry_slippage_pct)

    trades[event_id] = {
        "event_id": event_id,
        "symbol": symbol,
        "name": name or symbol,
        "direction": direction,
        "entry_price": actual_entry_price,
        "actual_entry_price": actual_entry_price,
        "requested_entry_price": requested_price,
        "signal_reference_price": signal_reference_price if signal_reference_price > 0 else None,
        "pre_order_reference_price": pre_order_reference_price if pre_order_reference_price > 0 else requested_price,
        "entry_slippage_pct": entry_slippage_pct,
        "adverse_entry_slippage_pct": adverse_entry_slippage_pct,
        "initial_qty": actual_qty,
        "remaining_qty": actual_qty,
        "entry_ts": actual_entry_ts,
        "tp_orders": tp_orders if isinstance(tp_orders, list) else [],
        "sl_order": sl_result if isinstance(sl_result, dict) else {},
        "hit_legs": [],
        "be_activated": False,
        "be_activation_ts": None,
        "be_required": False,
        "be_last_error": None,
        "peak_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "current_pnl_pct": 0.0,
        "score": _safe_float(score, 50.0),
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

    _save_active_trades(trades)


def format_tp_hit_message(
    name: str, symbol: str, leg: str, pnl_pct: float,
    exec_price: float, closed_qty: float, remaining_qty: float, remaining_pct: float,
) -> str:
    return (
        f"💰 <b>{name} ({symbol})</b>\n\n"
        f"Leg: <b>{leg}</b>\n"
        f"PnL TP: <b>+{pnl_pct:.2f}%</b>\n"
        f"Цена исполнения: <code>{exec_price:.8g}</code>\n"
        f"Закрыто: <code>{closed_qty:.8f}</code>\n"
        f"Осталось: <code>{remaining_qty:.8f} ({remaining_pct:.1f}%)</code>"
    )


def format_trade_closed_message(
    name: str, symbol: str, direction: str, entry_price: float, exit_price: float,
    pnl_pct: float, realized_rr: float | None, planned_rr: float | None,
    duration_min: float, peak_pnl: float, max_drawdown: float,
    exit_reason: str, event_type: str, timeframe: str = "1h",
) -> str:
    is_win = pnl_pct >= 0.0
    emoji = "💚" if is_win else "💔"
    pnl_sign = "+" if pnl_pct > 0 else ""
    realized_rr_text = f"{realized_rr:.3f}" if realized_rr is not None else "—"
    planned_rr_text = f"{planned_rr:.3f}" if planned_rr is not None else "—"

    lines = [
        f"{emoji} <b>{name} ({symbol}) — сделка закрыта</b>",
        "",
        f"Вход <code>{entry_price:.8g}</code> → Выход <code>{exit_price:.8g}</code>   <b>{pnl_sign}{pnl_pct:.2f}%</b>",
        f"Realized R:R: <b>{realized_rr_text}</b> · Planned Weighted R:R: <b>{planned_rr_text}</b>",
        f"Держали <b>{duration_min:.1f} мин</b> · пик <b>+{peak_pnl:.2f}%</b> · просадка <b>{max_drawdown:.2f}%</b>",
        f"Вход: <code>{event_type}</code> · TF <b>{str(timeframe or '1h').lower()}</b>",
        f"Выход: <b>{exit_reason}</b>",
    ]

    return "\n".join(lines)


def _cancel_old_sl_verified(symbol: str, direction: str, order_id: str, max_attempts: int = 3) -> tuple[bool, str]:
    """Cancel an SL order and verify it actually disappeared (audit fix B4).

    Returns (cancelled, message). After failing DELETE attempts we re-query
    open orders: some exchanges answer an already-executed/expired cancel with
    an error code while the order is in fact gone.
    """
    last_error = ""
    for attempt in range(max_attempts):
        try:
            resp = cancel_order(symbol, order_id)
        except Exception as exc:
            resp = {"code": -1, "msg": str(exc)}
        if isinstance(resp, dict) and resp.get("code") in (0, "0"):
            return True, "cancelled"
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
        remaining = _safe_float(rollback.get("remaining_qty"), 0.0)
        suffix = (
            f"Закрытие не подтверждено биржей; остаток позиции: <code>{remaining:.8f}</code>. "
            "Требуется следующее reconciliation."
        )
    else:
        suffix = "Состояние rollback требует reconciliation; закрытие не считается подтверждённым."
    text = (
        f"🛑 <b>BE move failed ({symbol} {direction})</b>\n"
        f"Status: <code>{detail}</code>\n"
        f"{suffix}"
    )
    _send_tracker_notification("BE_FAILURE", event_id, text, symbol=symbol)


def _emergency_close_after_be_failure(symbol: str, direction: str, qty: float, trade_id: str | None) -> dict:
    """Close and verify a position after BE protection cannot be proven.

    The MARKET close POST is never blindly retried. After an accepted POST we
    reconcile both the order and the live position for a bounded period so a
    normal exchange propagation delay does not get misclassified as
    ``close_unverified``.
    """
    attempts: list[dict] = []
    polls = max(4, int(os.environ.get("BE_FAILURE_CLOSE_POLLS", "12")))
    poll_delay = max(0.15, float(os.environ.get("BE_FAILURE_CLOSE_POLL_SEC", "0.5")))

    try:
        pos = get_position_directional(symbol, direction)
    except Exception as exc:
        pos = {"status": "error", "error": str(exc)}
    if pos.get("status") == "not_found":
        return {"status": "already_closed", "attempts": attempts, "remaining_qty": 0.0}
    if pos.get("status") != "found":
        return {"status": "close_unverified", "error": pos.get("error", "position state unavailable"), "attempts": attempts}

    current_qty = abs(_safe_float(pos.get("positionAmt"), qty))
    if current_qty <= 0:
        return {"status": "already_closed", "attempts": attempts, "remaining_qty": 0.0}

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

    last_qty = current_qty
    last_verification: dict = {"status": "error", "error": "verification not attempted"}
    order_filled = False
    for poll in range(polls):
        # Verify the accepted order first when an order id is available. This
        # distinguishes 'POST accepted, position endpoint lagging' from a real
        # close failure without retransmitting the POST.
        if accepted_order_id:
            try:
                order_info = get_order(symbol, accepted_order_id)
            except Exception as exc:
                order_info = {"status": "error", "error": str(exc)}
            if order_info.get("status") == "ok" and str(order_info.get("order_status", "")).upper() in {"FILLED", "PARTIALLY_FILLED"}:
                order_filled = float(order_info.get("executed_qty", 0.0) or 0.0) > 0
                attempts.append({"verification": "order", "poll": poll + 1, "order": order_info})

        try:
            verification = get_position_directional(symbol, direction)
        except Exception as exc:
            verification = {"status": "error", "error": str(exc)}
        last_verification = verification
        if verification.get("status") == "not_found":
            return {
                "status": "closed_verified",
                "attempts": attempts,
                "verification": verification,
                "verify_poll": poll + 1,
                "remaining_qty": 0.0,
                "order_filled": order_filled,
            }
        if verification.get("status") == "found":
            remaining = abs(_safe_float(verification.get("positionAmt"), last_qty))
            last_qty = remaining
            if remaining <= 1e-12 and (order_filled or accepted_order_id):
                return {
                    "status": "closed_verified",
                    "attempts": attempts,
                    "verification": verification,
                    "verify_poll": poll + 1,
                    "remaining_qty": 0.0,
                    "order_filled": order_filled,
                }
        time.sleep(poll_delay)

    # One final exchange read after the polling window. Never claim success on
    # a stale/errored exchange response.
    try:
        final_verification = get_position_directional(symbol, direction)
    except Exception as exc:
        final_verification = {"status": "error", "error": str(exc)}
    if final_verification.get("status") == "not_found":
        return {"status": "closed_verified", "attempts": attempts, "verification": final_verification, "remaining_qty": 0.0, "order_filled": order_filled}
    if final_verification.get("status") == "found":
        last_qty = abs(_safe_float(final_verification.get("positionAmt"), last_qty))

    return {
        "status": "close_unverified",
        "attempts": attempts,
        "verification": final_verification or last_verification,
        "remaining_qty": last_qty,
        "order_filled": order_filled,
    }

def _move_sl_to_break_even(
    symbol: str, direction: str, entry_price: float, qty: float, old_sl_id: str | None, trade_id: str | None = None,
    old_sl_price: float | None = None,
) -> dict:
    """Move the stop-loss to break-even without ever holding two SL orders.

    Safety rule: never create an unprotected window. The exchange-visible
    sequence is:

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
    be_client_id = f"EVT_BE_{trade_token}"

    verified = get_open_protection_directional(symbol, direction)
    if verified.get("status") == "ok":
        for order in verified.get("sl_orders", []):
            cid = str(order.get("clientOrderId", "")).upper()
            order_price = _safe_float(order.get("stopPrice") or order.get("price"), 0.0)
            price_matches = order_price > 0 and abs(order_price - entry_price) / max(entry_price, 1e-12) < 0.002

            if be_client_id.upper() in cid or price_matches:
                existing_id = str(order.get("orderId", ""))
                old_cancelled = True
                cancel_note = ""
                if old_sl_id and existing_id and str(old_sl_id) != existing_id:
                    old_cancelled, cancel_note = _cancel_old_sl_verified(symbol, direction, str(old_sl_id))
                if not old_cancelled:
                    log.warning(
                        "[TRACKER] BE already in place for %s but old SL %s cancel failed: %s; "
                        "BE is NOT considered active while multiple SL states may coexist.",
                        symbol, old_sl_id, cancel_note,
                    )
                    return {
                        "status": "error",
                        "error": f"BE exists but old SL cancel failed: {cancel_note}",
                        "order_id": existing_id,
                        "client_order_id": cid or be_client_id,
                        "stop_price": entry_price,
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
        restore_params = {
            "symbol": bx,
            "side": restore_side,
            "positionSide": position_side_param(direction),
            "type": "STOP_MARKET",
            "stopPrice": _format_price(old_sl_price, price_precision),
            "quantity": _format_qty(qty, precision),
            "clientOrderId": f"EVT_BE_RST_{trade_token}",
        }
        try:
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
                    if str(sl.get("orderId", "")) == str(old_sl_id) and _validate_sl_order_for_position(sl, direction, entry_price, qty):
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

    verified_after = get_open_protection_directional(symbol, direction)
    if verified_after.get("status") != "ok":
        return _fail("BE stop verification failed")

    found = any(str(o.get("orderId", "")) == new_order_id for o in verified_after.get("sl_orders", []))
    if not found:
        return _fail("BE stop not visible on exchange")

    # The BE stop is now proven to exist, so the position never becomes naked.
    # Remove the obsolete engine SL afterwards. If cancellation cannot be
    # proven, retain the verified BE stop and surface cleanup as an error rather
    # than closing a safely protected position.
    old_cleanup_error = None
    if old_sl_id and str(old_sl_id) != new_order_id:
        old_cancelled, cancel_note = _cancel_old_sl_verified(symbol, direction, str(old_sl_id))
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
        if entry_ts and ((open_ts and open_ts < entry_ts) or (not open_ts and close_ts <= entry_ts)):
            continue
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
    trade["peak_pnl_pct"] = peak
    trade["mae_pct"] = mae
    trade["max_drawdown_pct"] = drawdown

def _get_filled_order(symbol: str, order_id: str | None) -> dict | None:
    if not order_id:
        return None
    try:
        info = get_order(symbol, order_id)
    except Exception as exc:
        log.warning("[TRACKER] order query error for %s/%s: %s", symbol, order_id, exc)
        return None
    if info.get("status") != "ok" or str(info.get("order_status", "")).upper() != "FILLED":
        return None
    return info


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
) -> tuple[float | None, str | None, str | None, dict | None]:
    """Recover an exchange-verified residual exit after a position disappears.

    TP fills already accounted for by the tracker are excluded. A residual
    position must be closed by a protective SL or a separate closing order; a
    previous TP1 fill must never be reused as the residual exit price.
    """
    # First ask the known SL order directly.
    sl_id = sl_order.get("order_id") if isinstance(sl_order, dict) else None
    sl_info = _get_filled_order(symbol, sl_id)
    if sl_info:
        px = _safe_float(sl_info.get("avg_price"), 0.0)
        if px > 0:
            return px, "STOP_LOSS", "historical_sl_order", sl_info

    tp_ids = {str(x.get("order_id")) for x in tp_orders if x.get("order_id")}
    start = max(0, int(entry_ts) - 30_000)
    end = int(time.time() * 1000) + 30_000
    try:
        orders = get_all_orders(symbol, start, end, limit=100)
    except Exception as exc:
        log.warning("[TRACKER] historical order reconciliation failed for %s: %s", symbol, exc)
        return None, None, None, None

    wanted_side = "SELL" if direction == "LONG" else "BUY"
    candidates: list[dict] = []
    for order in orders:
        oid = str(order.get("orderId", ""))
        if oid and oid in tp_ids:
            continue
        status = str(order.get("status", order.get("orderStatus", ""))).upper()
        if status != "FILLED":
            continue
        side = str(order.get("side", "")).upper()
        if side != wanted_side:
            continue
        try:
            created = int(float(order.get("updateTime") or order.get("time") or order.get("createTime") or 0))
            qty = abs(float(order.get("executedQty") or order.get("cumQty") or order.get("origQty") or order.get("quantity") or 0))
            px = float(order.get("avgPrice") or order.get("price") or order.get("stopPrice") or 0)
        except (TypeError, ValueError):
            continue
        if entry_ts and created and created < entry_ts:
            continue
        if px <= 0 or qty <= 0:
            continue
        candidates.append({**order, "_ts": created, "_qty": qty, "_px": px})

    if not candidates:
        return None, None, None, None

    # Prefer protective stop orders for a disappearing residual position.
    stop_candidates = [
        o for o in candidates
        if str(o.get("type", "")).upper() in {"STOP", "STOP_MARKET"}
    ]
    pool = stop_candidates or candidates
    pool.sort(key=lambda o: int(o.get("_ts", 0)), reverse=True)

    # Avoid mistaking a small unrelated close for the residual position.
    residual = max(0.0, float(remaining_qty))
    for order in pool:
        if residual <= 0 or order["_qty"] >= residual * 0.95:
            reason = "STOP_LOSS" if str(order.get("type", "")).upper() in {"STOP", "STOP_MARKET"} else "MANUAL_CLOSE_RECONCILED"
            return order["_px"], reason, "historical_all_orders", order

    return None, None, None, None


def update_active_trades() -> None:
    trades = _load_active_trades()
    if not trades:
        return

    now_ms = int(time.time() * 1000)
    updated_trades: dict[str, dict] = {}

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

            hit_legs = set(trade.get("hit_legs", []))
            filled_by_leg = {str(k): max(0.0, _safe_float(v)) for k, v in (trade.get("tp_filled_qty", {}) or {}).items()}
            realized_qty = max(0.0, _safe_float(trade.get("realized_pnl_qty", 0.0)))
            realized_weighted = _safe_float(trade.get("realized_pnl_weighted_sum", 0.0))

            pos = get_position_directional(symbol, direction)
            pos_status = str(pos.get("status", "")).lower()

            if pos_status not in {"found", "not_found"}:
                trade["last_observation_ts"] = now_ms
                updated_trades[event_id] = trade
                continue

            pos_amt = abs(_safe_float(pos.get("positionAmt"))) if pos_status == "found" else 0.0
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

            cur_price = entry_price
            try:
                k1m = fetch_klines(symbol, "1m", limit=60)
                if k1m:
                    cur_price = _safe_float(k1m[-1].get("close"), entry_price)
                    _update_mfe_mae(trade, k1m)
            except Exception as exc:
                log.warning("[TRACKER] Kline fetch error for %s: %s", symbol, exc)

            current_pnl = _calc_trade_pnl_pct(entry_price, cur_price, direction)
            trade["current_pnl_pct"] = current_pnl
            trade["current_position_qty"] = pos_amt
            trade["last_observation_ts"] = now_ms

            # Audit fix: BE is triggered by PRICE reaching +0.50R, not by TP1
            # order status. MFE is updated from the fetched 1m candles above, so
            # an intrabar touch that later retraces is still detected.
            planned_risk_pct = _derive_planned_risk_pct(trade)
            be_trigger_r = float(os.environ.get("BE_TRIGGER_R", "0.50"))
            peak_r = (float(trade.get("peak_pnl_pct", 0.0)) / planned_risk_pct) if planned_risk_pct and planned_risk_pct > 0 else 0.0
            if pos_status == "found" and rem_qty > 0 and not trade.get("be_activated") and peak_r >= be_trigger_r:
                old_sl = trade.get("sl_order", {}) if isinstance(trade.get("sl_order"), dict) else {}
                old_sl_id = old_sl.get("order_id")
                old_sl_price = _safe_float(old_sl.get("stop_price"), 0.0) or None
                be_result = _move_sl_to_break_even(symbol, direction, entry_price, rem_qty, old_sl_id, str(event_id).replace("EVT_", ""), old_sl_price=old_sl_price)
                if be_result.get("status") == "created":
                    trade["sl_order"] = be_result
                    trade["be_activated"] = True
                    trade["be_required"] = False
                    trade["be_trigger_r"] = be_trigger_r
                    trade["be_trigger_peak_r"] = peak_r
                    trade["be_activation_ts"] = trade.get("be_activation_ts") or now_ms
                    log.info("[TRACKER_BE_ACTIVATED] %s (%s) Price reached +%.2fR. Stop-loss moved to Break-Even: %.8g", trade.get("name", symbol), symbol, peak_r, entry_price)
                else:
                    trade["be_required"] = True
                    trade["be_last_error"] = be_result.get("error")
                    trade["be_trigger_r"] = be_trigger_r
                    trade["be_trigger_peak_r"] = peak_r
                    log.error("[TRACKER_BE_FAILED] %s (%s) Price reached +%.2fR but BE failed: %s", trade.get("name", symbol), symbol, peak_r, be_result.get("error"))

            # Retry a failed TP1 -> BE transition while the position remains open.
            if "tp1" in set(trade.get("hit_legs", [])) and not trade.get("be_activated") and rem_qty > 0:
                old_sl = trade.get("sl_order", {}) if isinstance(trade.get("sl_order"), dict) else {}
                old_sl_id = old_sl.get("order_id")
                old_sl_price = _safe_float(old_sl.get("stop_price"), 0.0) or None
                retry = _move_sl_to_break_even(symbol, direction, entry_price, rem_qty, old_sl_id, str(event_id).replace("EVT_", ""), old_sl_price=old_sl_price)
                if retry.get("status") == "created":
                    trade["sl_order"] = retry
                    trade["be_activated"] = True
                    trade["be_required"] = False
                    trade["be_last_error"] = None
                    trade["be_activation_ts"] = trade.get("be_activation_ts") or now_ms
                else:
                    trade["be_required"] = True
                    trade["be_last_error"] = retry.get("error")

            # Проверка исполнения Тейк-Профитов
            for tp in trade.get("tp_orders", []):
                leg = str(tp.get("leg", ""))
                order_id = tp.get("order_id")
                if not leg or not order_id:
                    continue

                try:
                    order_info = get_order(symbol, order_id)
                except Exception:
                    continue

                if order_info.get("status") == "error":
                    continue

                order_status = str(order_info.get("order_status", "")).upper()
                if order_status not in {"PARTIALLY_FILLED", "FILLED"}:
                    continue

                executed_qty = max(0.0, _safe_float(order_info.get("executed_qty", 0.0)))
                previous_qty = max(0.0, _safe_float(filled_by_leg.get(leg, 0.0)))
                delta_qty = max(0.0, executed_qty - previous_qty)

                if delta_qty <= 0:
                    if order_status == "FILLED":
                        hit_legs.add(leg)
                    continue

                exec_price = _safe_float(order_info.get("avg_price"), 0.0)
                if exec_price <= 0:
                    log.warning("[TRACKER_TP] %s %s %s has no actual avgPrice; deferring realized PnL", symbol, direction, leg)
                    continue
                pnl_tp = _calc_trade_pnl_pct(entry_price, exec_price, direction)

                # The position snapshot taken before this TP query is already exchange-authoritative.
                # Do not subtract the TP delta from it again: the exchange position may already
                # reflect the fill. Reconcile the live quantity after observing the TP execution.
                try:
                    tp_pos = get_position_directional(symbol, direction)
                except Exception as tp_pos_exc:
                    tp_pos = {"status": "error", "error": str(tp_pos_exc)}
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

                trade["remaining_qty"] = rem_qty
                trade["realized_pnl_qty"] = realized_qty
                trade["realized_pnl_weighted_sum"] = realized_weighted
                trade["last_tp_exec_price"] = exec_price

                if order_status == "FILLED" and leg not in hit_legs:
                    hit_legs.add(leg)
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
                            closed_qty=delta_qty,
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
                        new_sl = _move_sl_to_break_even(
                            symbol=symbol,
                            direction=direction,
                            entry_price=entry_price,
                            qty=rem_qty,
                            old_sl_id=old_sl_id,
                            trade_id=str(event_id).replace("EVT_", ""),
                            old_sl_price=old_sl_price,
                        )
                        if new_sl.get("status") in {"created", "created_cleanup_pending"}:
                            trade["sl_order"] = new_sl
                            trade["be_activated"] = True
                            trade["be_activation_ts"] = now_ms
                            log.info(
                                "[TRACKER_BE_ACTIVATED] %s (%s) TP1 taken. Stop-loss moved to Break-Even: %.8g (Risk: 0.00%%)",
                                trade.get("name", symbol), symbol, entry_price
                            )                             
                        else:
                            trade["be_required"] = True
                            trade["be_last_error"] = new_sl.get("error")
                            log.error(
                                "[TRACKER_BE_FAILED] %s (%s) Failed to move SL to Break-Even: %s",
                                trade.get("name", symbol), symbol, new_sl.get("error")
                            )

            trade["hit_legs"] = sorted(hit_legs)
            trade["tp_filled_qty"] = filled_by_leg

            # Re-read the position after processing TP order status. A TP can
            # fill between the first position snapshot and the order query loop;
            # the exchange position remains the source of truth for the final
            # residual quantity and for deciding whether the position is gone.
            # Preserve the residual quantity that was still outstanding before
            # the final exchange lookup says the position has disappeared. It is
            # required to reconstruct the PnL of the residual close.
            residual_qty_before_position_disappeared = max(0.0, rem_qty)
            final_pos = get_position_directional(symbol, direction)
            final_pos_status = str(final_pos.get("status", "")).lower()
            if final_pos_status == "found":
                rem_qty = abs(_safe_float(final_pos.get("positionAmt")))
                trade["current_position_qty"] = rem_qty
                trade["remaining_qty"] = rem_qty
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

            closed_by_tp = rem_qty <= 1e-12 and realized_qty > 0 and position_gone

            if not position_gone and not closed_by_tp:
                updated_trades[event_id] = trade
                continue

            # Фиксация выхода и закрытие
            duration_min = (now_ms - entry_ts) / 60000.0
            exit_price = _safe_float(trade.get("last_tp_exec_price"), cur_price)
            sl_order = trade.get("sl_order", {}) if isinstance(trade.get("sl_order"), dict) else {}
            sl_order_id = sl_order.get("order_id")

            sl_exit_price, _ = _get_exit_from_sl(symbol, sl_order_id)
            historical_source = None
            historical_order = None
            if sl_exit_price is not None:
                exit_price = sl_exit_price
                if trade.get("be_activated") and abs(exit_price - entry_price) / max(entry_price, 1e-12) < 0.003:
                    exit_reason = "BREAK_EVEN"
                else:
                    exit_reason = "STOP_LOSS"
            elif closed_by_tp:
                exit_reason = "TAKE_PROFIT_FULL"
            elif position_gone:
                hist_px, hist_reason, hist_source, historical_order = _reconcile_historical_exit_order(
                    symbol, direction, entry_ts, residual_qty_before_position_disappeared, trade.get("tp_orders", []), sl_order
                )
                if hist_px is not None:
                    exit_price = hist_px
                    exit_reason = hist_reason or "MANUAL_CLOSE_RECONCILED"
                else:
                    exit_reason = "POSITION_CLOSED_UNVERIFIED"
            else:
                exit_reason = "POSITION_CLOSED_UNVERIFIED"

            if exit_price <= 0:
                exit_price = cur_price

            if position_gone and residual_qty_before_position_disappeared > 0 and init_qty > 0:
                residual_pnl = _calc_trade_pnl_pct(entry_price, exit_price, direction)
                realized_weighted += residual_qty_before_position_disappeared * residual_pnl
                realized_qty += residual_qty_before_position_disappeared
                trade["remaining_qty"] = 0.0

            final_pnl = (realized_weighted / init_qty) if (init_qty > 0 and realized_qty > 0) else current_pnl
            realized_pnl_source = (
                "executed_tp_or_sl" if (closed_by_tp or sl_exit_price is not None)
                else (hist_source or "observation_price_estimate")
            )
            if closed_by_tp and realized_qty > 0 and sl_exit_price is None:
                exit_price = entry_price * (1.0 + final_pnl / 100.0) if direction == "LONG" else entry_price * (1.0 - final_pnl / 100.0)
            planned_risk_pct = _derive_planned_risk_pct(trade)
            realized_rr = _calc_realized_rr(final_pnl, planned_risk_pct) if realized_pnl_source == "executed_tp_or_sl" else None
            planned_rr = _safe_float(trade.get("effective_weighted_rr", trade.get("planned_weighted_rr", 1.05)), 1.05)

            trade["remaining_qty"] = 0.0
            trade["realized_pnl_pct"] = final_pnl
            trade["realized_pnl_qty"] = realized_qty
            trade["realized_pnl_weighted_sum"] = realized_weighted
            trade["realized_rr"] = realized_rr
            trade["realized_pnl_source"] = realized_pnl_source
            trade["exit_price"] = exit_price
            trade["exit_reason"] = exit_reason
            trade["closed_ts"] = now_ms
            trade["duration_min"] = duration_min
            trade["closed"] = True
            emoji = "💚" if final_pnl >= 0.0 else "💔"
            _append_trade_close_once({
                "record_type": "TRADE_CLOSE",
                "event_id": event_id,
                "symbol": symbol,
                "direction": direction,
                "event_type": trade.get("event_type"),
                "closed_ts": now_ms,
                "exit_reason": exit_reason,
                "outcome_category": _exit_outcome_category(exit_reason),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "realized_pnl_pct": final_pnl,
                "realized_rr": realized_rr,
                "realized_pnl_source": realized_pnl_source,
                "effective_weighted_rr": planned_rr,
                "tp_mode": trade.get("tp_mode", "multi_tp"),
                "effective_tp_levels": trade.get("effective_tp_levels", []),
                "peak_pnl_pct": _safe_float(trade.get("peak_pnl_pct")),
                "mae_pct": _safe_float(trade.get("mae_pct")),
                "max_drawdown_pct": _safe_float(trade.get("max_drawdown_pct")),
                "duration_min": duration_min,
                "hit_legs": sorted(hit_legs),
                "tp_filled_qty": filled_by_leg,
                "be_activated": bool(trade.get("be_activated")),
                "research": trade.get("research", {}),
                "setup": trade.get("setup", {}),
                })

            log.info("[TRACKER_TRADE_CLOSED] %s (%s) | PnL: %+.2f%% | Realized R:R: %s | Planned R:R: %.2f | Exit: %.8g (%s) | Duration: %.1f min", emoji, trade.get("name", symbol), symbol, final_pnl, (f"{realized_rr:.3f}" if realized_rr is not None else "—"), planned_rr, exit_price, exit_reason, duration_min)

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

            for tp in trade.get("tp_orders", []):
                if tp.get("leg") not in hit_legs and tp.get("order_id"):
                    try:
                        cancel_order(symbol, tp["order_id"])
                    except Exception:
                        pass

            if sl_order_id:
                try:
                    cancel_order(symbol, sl_order_id)
                except Exception:
                    pass

        except Exception as exc:
            log.exception("[TRACKER] Fatal trade error for event %s: %s", event_id, exc)
            updated_trades[event_id] = trade

    _save_active_trades(updated_trades)
