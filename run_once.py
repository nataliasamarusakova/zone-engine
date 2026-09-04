from __future__ import annotations

import json
import math
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from event_engine.analytics import save_scan
from event_engine.binance import analysis_symbols_for_bingx, classify_bingx_contract, fetch_klines as fetch_binance_klines
from event_engine.bingx import (
    contracts,
    credentials_available,
    ensure_directional_protection,
    fetch_klines as fetch_bingx_klines,
    get_contract,
    get_positions,
    get_position_mode,
    get_position_directional,
    get_open_protection_directional,
    cancel_order,
    close_position_market,
    open_market,
    wait_for_position_fill_directional,
)
from event_engine.signals import SWING_LEN, TP1_R, TP2_R, generate_zone_signals, score_zone_signal
from event_engine.telegram import format_signal, send as send_tg
from event_engine.tracker import register_active_trade, update_active_trades, update_active_trade_protection

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("zone_engine")

DATA = Path("data")
TRADES_PATH = DATA / "trades.jsonl"
ACTIONS_PATH = DATA / "actions.jsonl"

EXECUTION_ENABLED = os.environ.get("EXECUTION_ENABLED", "true").lower() == "true"
MARGIN_USDT = float(os.environ.get("BINGX_MARGIN_USDT", "1"))
MAX_TRADES_PER_CYCLE = int(os.environ.get("MAX_TRADES_PER_CYCLE", "5"))
MAX_SCAN_SYMBOLS = int(os.environ.get("MAX_SCAN_SYMBOLS", "0"))
WATCHLIST_ONLY = os.environ.get("WATCHLIST_ONLY", "false").lower() == "true"
WATCHLIST_SYMBOLS = tuple(x.strip().upper() for x in os.environ.get(
    "WATCHLIST_SYMBOLS",
    "BTC-USDT,ETH-USDT,SOL-USDT,BNB-USDT,TAO-USDT,LTC-USDT,BCH-USDT,AVAX-USDT,LINK-USDT,ETC-USDT,ADA-USDT,UNI-USDT,XRP-USDT,ICP-USDT,HYPE-USDT,DOGE-USDT,HBAR-USDT,ARB-USDT,POL-USDT,SUI-USDT",
).split(",") if x.strip())
KLINE_LIMIT_1H = int(os.environ.get("KLINE_LIMIT_1H", "120"))
MAX_SIGNAL_AGE_BARS = int(os.environ.get("MAX_SIGNAL_AGE_BARS", "0"))
MAX_PRODUCTION_RISK_PCT = min(float(os.environ.get("MAX_SIGNAL_RISK_PCT", "1.50")), 1.50)
# Production execution is strict by default: only the latest closed 1H bar may open a trade.
EXECUTION_MAX_SIGNAL_AGE_BARS = int(os.environ.get("EXECUTION_MAX_SIGNAL_AGE_BARS", "0"))
DIAGNOSTICS_MODE = os.environ.get("DIAGNOSTICS_MODE", "historical").strip().lower()
if DIAGNOSTICS_MODE not in {"historical", "live"}:
    raise ValueError("DIAGNOSTICS_MODE must be historical or live")
SCAN_WORKERS = max(1, int(os.environ.get("SCAN_WORKERS", "12")))
SCAN_BATCH_SIZE = max(SCAN_WORKERS, int(os.environ.get("SCAN_BATCH_SIZE", "48")))
SCAN_BATCH_PAUSE_SEC = max(0.0, float(os.environ.get("SCAN_BATCH_PAUSE_SEC", "0.10")))
BINANCE_ASSET_CLASSES = {x.strip().upper() for x in os.environ.get("BINANCE_ASSET_CLASSES", "CRYPTO,EQUITY").split(",") if x.strip()}
MAX_MARKET_SPREAD_PCT = float(os.environ.get("MAX_MARKET_SPREAD_PCT", "1.50"))
RECONCILIATION_MAX_SECONDS = float(os.environ.get("RECONCILIATION_MAX_SECONDS", "45"))
# Live execution requires the signal timestamp to be exactly the latest closed 1H bar.
# Also reject stale market data so a symbol with an old/delisted Binance series cannot
# masquerade as a fresh signal merely because its DataFrame index is zero-based.
MAX_DATA_STALENESS_HOURS = float(os.environ.get("MAX_DATA_STALENESS_HOURS", "2.0"))


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _load_successful_trade_ids() -> set[str]:
    if not TRADES_PATH.exists():
        return set()
    out: set[str] = set()
    for line in TRADES_PATH.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        result = row.get("result", {}) if isinstance(row.get("result"), dict) else {}
        status = str(result.get("status", row.get("status", ""))).lower()
        if status in {"opened_protected", "opened", "already_executed", "existing_position"} and row.get("event_id"):
            out.add(str(row["event_id"]))
    return out


def _symbol_from_contract(c: dict[str, Any]) -> str | None:
    if str(c.get("status", "1")) not in {"1", "1.0", "true", "True"} and c.get("status") is not None:
        return None
    if str(c.get("apiStateOpen", "true")).lower() not in {"true", "1"}:
        return None
    symbol = str(c.get("symbol", "")).strip().upper()
    display = str(c.get("displayName", "")).strip().upper()
    candidate = symbol or display
    if not candidate:
        return None
    if "-USDT" in candidate:
        return candidate
    if candidate.endswith("USDT"):
        return candidate[:-4] + "-USDT"
    return None


def get_scan_symbols() -> list[str]:
    all_contracts = contracts()
    available = set()
    for c in all_contracts.values():
        symbol = _symbol_from_contract(c)
        if symbol:
            available.add(symbol)

    if WATCHLIST_ONLY:
        # Keep the order from the configured watchlist so the runtime log is
        # deterministic and manual checking is straightforward.
        symbols = [s for s in WATCHLIST_SYMBOLS if s in available]
        missing = [s for s in WATCHLIST_SYMBOLS if s not in available]
        if missing:
            log.warning("[WATCHLIST_MISSING] symbols_not_active=%s", ",".join(missing))
        if MAX_SCAN_SYMBOLS > 0:
            symbols = symbols[:MAX_SCAN_SYMBOLS]
        return symbols

    symbols = sorted(available)
    if MAX_SCAN_SYMBOLS > 0:
        symbols = symbols[:MAX_SCAN_SYMBOLS]
    return symbols


def _log_coin_skip(symbol: str, reason: str) -> None:
    log.warning("[COIN_SKIP] %s | %s", symbol, reason)


def _select_latest_signal(signals: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the newest signal bar; score is only a same-bar tie-breaker."""
    if not signals:
        return None
    return max(
        signals,
        key=lambda s: (int(s["idx"]), float(s.get("score", 0.0))),
    )



def _signal_matches_latest_bar(signal: dict[str, Any], latest_closed_idx: int, latest_closed_time: str | None) -> tuple[bool, str]:
    """Strict latest-bar identity check for the zone-only strategy.

    ZONE_ONLY deliberately has no Pine/ALMA execution gate. A zone signal is
    executable when the signal bar is exactly the latest closed 1H bar.
    """
    try:
        signal_idx = int(signal["idx"])
    except (KeyError, TypeError, ValueError):
        return False, "invalid_signal_idx"
    if signal_idx != int(latest_closed_idx):
        return False, "signal_idx_not_latest"
    signal_time = signal.get("time")
    if signal_time is None or latest_closed_time is None:
        return False, "missing_signal_or_latest_time"
    try:
        if pd.Timestamp(signal_time) != pd.Timestamp(latest_closed_time):
            return False, "signal_time_not_latest"
    except Exception:
        return False, "invalid_signal_time"
    return True, "ok"

def _private_layer_ready() -> bool:
    ready = credentials_available()
    if not ready:
        log.warning(
            "[AUTH] BingX private credentials are missing. "
            "Public market scan will continue; positions, reconciliation and execution are disabled for this run."
        )
    return ready


def _price_position(price: float, demand: list[dict], supply: list[dict]) -> str:
    # Display status must reflect the literal zone boundaries. Do not add
    # percentage padding here: a price below Supply or above Demand is not
    # "in the zone" merely because it is close to it.
    in_dem = any(float(z["btm"]) <= price <= float(z["top"]) for z in demand)
    in_sup = any(float(z["btm"]) <= price <= float(z["top"]) for z in supply)
    if in_dem:
        return "🟢 В зоне DEMAND"
    if in_sup:
        return "🔴 В зоне SUPPLY"
    return "⚪ Вне зон (Ждать)"


def _bingx_last_price(contract: dict[str, Any]) -> float | None:
    for key in ("lastPrice", "last", "price", "markPrice"):
        try:
            value = float(contract.get(key))
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return None


def _market_spread_pct(binance_price: float | None, bingx_price: float | None) -> float | None:
    if not binance_price or not bingx_price or binance_price <= 0:
        return None
    return round(abs(bingx_price - binance_price) / binance_price * 100.0, 4)


def _position_keys(positions: list[dict]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for p in positions:
        symbol = str(p.get("symbol", "")).upper()
        side = str(p.get("positionSide", "")).upper()
        try:
            qty = abs(float(p.get("positionAmt", 0) or 0))
        except Exception:
            qty = 0.0
        if qty <= 0:
            continue
        if side in {"LONG", "SHORT"}:
            out.add((symbol, side))
    return out


def reconcile_all_open_positions() -> None:
    started = time.time()
    try:
        positions = get_positions(timeout_sec=float(os.environ.get("RECONCILIATION_HTTP_TIMEOUT_SEC", "5")), retryable=False)
    except Exception as exc:
        log.error("[RECON] positions fetch failed: %s", exc)
        return

    # Protection repair is driven from actual entry price and a stored setup when available.
    active = _load_active_trades_file()
    for p in positions:
        if time.time() - started >= RECONCILIATION_MAX_SECONDS:
            break
        symbol = str(p.get("symbol", "")).upper()
        side = str(p.get("positionSide", "")).upper()
        if side not in {"LONG", "SHORT"}:
            continue
        qty = abs(float(p.get("positionAmt", 0) or 0))
        avg = float(p.get("avgPrice", 0) or p.get("entryPrice", 0) or 0)
        if qty <= 0 or avg <= 0:
            continue
        key = f"{symbol}:{side}"
        trade = active.get(key)
        stop_loss_pct = float((trade or {}).get("planned_risk_pct") or 1.0)
        setup = (trade or {}).get("setup", {}) if isinstance(trade, dict) else {}
        tp_levels = setup.get("tp_levels") if isinstance(setup.get("tp_levels"), list) else None
        if not tp_levels:
            risk_pct = max(stop_loss_pct, 0.05)
            tp_levels = [
                {"leg": "tp1", "pnl_pct": risk_pct * TP1_R, "close_fraction": 0.50},
                {"leg": "tp2", "pnl_pct": risk_pct * TP2_R, "close_fraction": 0.50},
            ]

        # Never recreate a TP leg already confirmed as executed. After TP1 the
        # remaining TP2 becomes 100% of the remaining position.
        hit_legs = set((trade or {}).get("hit_legs", []))
        remaining_levels = [x for x in tp_levels if str(x.get("leg", "")) not in hit_legs]
        if remaining_levels:
            share = 1.0 / len(remaining_levels)
            tp_levels = [
                {"leg": str(x.get("leg")), "pnl_pct": float(x.get("pnl_pct", 0.0)), "close_fraction": share}
                for x in remaining_levels
            ]
        else:
            tp_levels = []

        try:
            result = ensure_directional_protection(symbol, side, avg, qty, stop_loss_pct, tp_levels, trade_id=(trade or {}).get("event_id") or key)
            if result.get("status") in {"PROTECTED", "SL_ONLY"}:
                if trade:
                    update_active_trade_protection(symbol, side, result.get("tp_orders", []), result.get("sl_result", {}), result.get("effective_tp_levels", []), result.get("tp_mode"), result.get("effective_weighted_rr"))
        except Exception as exc:
            log.exception("[RECON] protection repair failed for %s %s: %s", symbol, side, exc)


def _load_active_trades_file() -> dict[str, dict]:
    path = DATA / "active_trades.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    if isinstance(raw, dict):
        for trade in raw.values():
            if not isinstance(trade, dict) or trade.get("closed"):
                continue
            sym = str(trade.get("symbol", "")).upper()
            side = str(trade.get("direction", "")).upper()
            if sym and side:
                out[f"{sym}:{side}"] = trade
    return out


def _build_setup(signal: dict[str, Any]) -> dict[str, Any]:
    risk_pct = float(signal["risk_pct"])
    return {
        "strategy": str(signal.get("strategy", "Demand/Supply Zone First")),
        "signal_price": float(signal["entry"]),
        "entry_reference": float(signal["entry"]),
        "invalidation_price": float(signal["sl"]),
        "risk_pct": risk_pct,
        "target_rr": float(signal.get("tp2_rr", TP2_R)),
        "planned_weighted_rr": float(signal.get("tp1_rr", TP1_R)) * 0.50 + float(signal.get("tp2_rr", TP2_R)) * 0.50,
        "tp_levels": [
            {"leg": "tp1", "pnl_pct": float(signal.get("tp1_rr", TP1_R)) * risk_pct, "close_fraction": 0.50, "price": float(signal["tp1"])},
            {"leg": "tp2", "pnl_pct": float(signal.get("tp2_rr", TP2_R)) * risk_pct, "close_fraction": 0.50, "price": float(signal["tp2"])},
        ],
        "target_price": float(signal["tp2"]),
        "zone": signal.get("zone", {}),
        "confirmation": signal.get("confirmation", {}),
        "score": float(signal.get("score", 0.0)),
        "event_time": signal.get("time"),
    }



def _validate_trade_geometry(signal: dict[str, Any]) -> tuple[bool, str]:
    """Reject mathematically invalid setups before any MARKET order is sent."""
    try:
        direction = str(signal["type"]).upper()
        entry = float(signal["entry"])
        sl = float(signal["sl"])
        tp1 = float(signal["tp1"])
        tp2 = float(signal["tp2"])
        risk_pct = float(signal["risk_pct"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"invalid_numeric_setup: {exc}"
    values = {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "risk_pct": risk_pct}
    if any(not math.isfinite(v) for v in values.values()):
        return False, "non_finite_setup"
    if min(entry, sl, tp1, tp2) <= 0:
        return False, "non_positive_price"
    if risk_pct <= 0:
        return False, f"non_positive_risk_pct={risk_pct}"
    if risk_pct > MAX_PRODUCTION_RISK_PCT:
        return False, f"risk_pct_above_limit={risk_pct}"
    target = signal.get("target") if isinstance(signal.get("target"), dict) else {}
    obstacle_price = target.get("obstacle_price")
    try:
        obstacle_price = float(obstacle_price) if obstacle_price is not None else None
    except (TypeError, ValueError):
        obstacle_price = None
    min_room_r = float(os.environ.get("MIN_STRUCTURE_ROOM_R", "1.20"))
    risk_abs = abs(entry - sl)
    if obstacle_price is None and os.environ.get("REQUIRE_STRUCTURE_OBSTACLE", "true").lower() == "true":
        return False, "missing_structural_obstacle"
    if obstacle_price is not None and risk_abs > 0:
        room = (obstacle_price - entry) if direction == "LONG" else (entry - obstacle_price)
        if room <= 0 or (room / risk_abs) < min_room_r:
            return False, f"insufficient_structure_room={room / risk_abs:.3f}R < {min_room_r:.3f}R"
    if direction == "LONG":
        if not sl < entry:
            return False, f"LONG invalid SL: sl={sl} entry={entry}"
        if not (tp1 > entry and tp2 > tp1):
            return False, f"LONG invalid TP geometry: entry={entry} tp1={tp1} tp2={tp2}"
    elif direction == "SHORT":
        if not sl > entry:
            return False, f"SHORT invalid SL: sl={sl} entry={entry}"
        if not (tp1 < entry and tp2 < tp1):
            return False, f"SHORT invalid TP geometry: entry={entry} tp1={tp1} tp2={tp2}"
    else:
        return False, f"invalid_direction={direction}"
    return True, "ok"


def _protection_geometry_from_fill(direction: str, avg_price: float, risk_pct: float) -> tuple[float, float, float]:
    risk = avg_price * risk_pct / 100.0
    if direction == "LONG":
        return avg_price - risk, avg_price + TP1_R * risk, avg_price + TP2_R * risk
    return avg_price + risk, avg_price - TP1_R * risk, avg_price - TP2_R * risk


def _cleanup_engine_protection(symbol: str, direction: str) -> dict[str, Any]:
    """Cancel only this engine's outstanding SL/TP orders after an emergency close."""
    result = {"status": "ok", "cancelled": [], "errors": []}
    try:
        existing = get_open_protection_directional(symbol, direction)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "cancelled": [], "errors": []}
    if existing.get("status") != "ok":
        return {"status": "error", "error": existing.get("error", "openOrders unavailable"), "cancelled": [], "errors": []}
    for order in list(existing.get("sl_orders", [])) + list(existing.get("tp_orders", [])):
        cid = str(order.get("clientOrderId", "")).upper()
        oid = str(order.get("orderId", ""))
        if not oid or not cid.startswith("EVT_"):
            continue
        try:
            resp = cancel_order(symbol, oid)
            if isinstance(resp, dict) and resp.get("code") in (0, "0"):
                result["cancelled"].append(oid)
            else:
                result["errors"].append(f"{oid}: code={resp.get('code') if isinstance(resp, dict) else None} msg={resp.get('msg') if isinstance(resp, dict) else resp}")
        except Exception as exc:
            result["errors"].append(f"{oid}: {exc}")
    if result["errors"]:
        result["status"] = "partial" if result["cancelled"] else "error"
    return result

def _rebase_protection_after_fill(signal: dict[str, Any], avg_price: float) -> dict[str, Any]:
    """Recalculate zone-based SL/TP from the *actual* market fill.

    A market order can fill materially away from the signal/reference candle close.
    Never submit stale absolute targets derived from the pre-fill reference price.
    """
    direction = str(signal["type"]).upper()
    entry = float(avg_price)
    zone = signal.get("zone") if isinstance(signal.get("zone"), dict) else {}
    target = signal.get("target") if isinstance(signal.get("target"), dict) else {}
    atr = float(signal.get("atr", 0.0) or 0.0)
    if entry <= 0:
        raise ValueError("actual fill price must be positive")

    zone_top = float(zone.get("top")) if zone.get("top") is not None else None
    zone_bottom = float(zone.get("btm")) if zone.get("btm") is not None else None
    if zone_top is None or zone_bottom is None:
        raise ValueError("zone boundaries unavailable for post-fill protection")

    sl_buffer = max(atr * float(os.environ.get("ZONE_SL_ATR_BUFFER", "0.10")), entry * 0.0002)
    if direction == "LONG":
        zone_stop = zone_bottom - sl_buffer
        sl = zone_stop if zone_stop < entry else entry - sl_buffer
    else:
        zone_stop = zone_top + sl_buffer
        sl = zone_stop if zone_stop > entry else entry + sl_buffer

    risk = abs(entry - sl)
    if risk <= 0:
        raise ValueError("post-fill risk is non-positive")

    obstacle = target.get("obstacle_price")
    try:
        obstacle = float(obstacle) if obstacle is not None else None
    except (TypeError, ValueError):
        obstacle = None

    obstacle_buffer = max(atr * float(os.environ.get("TP_OBSTACLE_BUFFER_ATR", "0.10")), entry * 0.0002)
    tp1_r = float(os.environ.get("TP1_R", "0.5"))
    tp2_r = float(os.environ.get("TP2_R", "1.0"))
    tp_min_r = float(os.environ.get("TP_MIN_R", "0.50"))
    tp_max_r = float(os.environ.get("TP_MAX_R", "1.50"))
    tp1_fraction = float(os.environ.get("TP1_OBSTACLE_FRACTION", "0.50"))
    tp2_fraction = float(os.environ.get("TP2_OBSTACLE_FRACTION", "0.90"))

    target_source = "atr_rr_fallback_after_fill"
    if obstacle is None and os.environ.get("REQUIRE_STRUCTURE_OBSTACLE", "true").lower() == "true":
        raise ValueError("structural obstacle unavailable after fill")
    if obstacle is not None:
        if direction == "LONG" and obstacle > entry + obstacle_buffer:
            usable = obstacle - obstacle_buffer - entry
        elif direction == "SHORT" and obstacle < entry - obstacle_buffer:
            usable = entry - obstacle_buffer - obstacle
        else:
            usable = -1.0
        if usable > 0:
            tp2_distance = min(usable * tp2_fraction, tp_max_r * risk)
            tp1_distance = tp2_distance * tp1_fraction
            if tp2_distance / risk >= tp_min_r and tp1_distance > 0 and tp2_distance > tp1_distance:
                target_source = "nearest_opposing_structure_after_fill"
            else:
                tp2_distance = 0.0
        else:
            tp2_distance = 0.0
    else:
        tp2_distance = 0.0

    if tp2_distance <= 0:
        tp1_distance = tp1_r * risk
        tp2_distance = tp2_r * risk
        if tp2_distance <= tp1_distance:
            tp2_distance = max(tp1_distance * 2.0, risk)

    if direction == "LONG":
        tp1 = entry + tp1_distance
        tp2 = entry + tp2_distance
    else:
        tp1 = entry - tp1_distance
        tp2 = entry - tp2_distance

    return {
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_abs": risk,
        "risk_pct": (risk / entry) * 100.0,
        "tp1_rr": abs(tp1 - entry) / risk,
        "tp2_rr": abs(tp2 - entry) / risk,
        "target_source": target_source,
        "obstacle_price": obstacle,
    }


def _emergency_close_and_verify(symbol: str, direction: str, qty: float, trade_id: str) -> dict[str, Any]:
    """Close a safety-rollback position and verify that it is actually gone.

    A rollback is not considered successful merely because the POST returned 0.
    We re-read the directional position and retry once with the currently
    reported quantity. This keeps an exchange/API failure from turning a
    protection failure into an untracked naked position.
    """
    attempts = []
    last_qty = max(0.0, float(qty or 0.0))
    for attempt in range(2):
        try:
            current = get_position_directional(symbol, direction)
        except Exception as exc:
            current = {"status": "error", "error": str(exc)}
        if current.get("status") == "not_found":
            return {"status": "closed_verified", "attempts": attempts, "verification": current}
        if current.get("status") == "found":
            last_qty = abs(float(current.get("positionAmt", last_qty) or last_qty))
        if last_qty <= 0:
            return {"status": "closed_verified", "attempts": attempts, "verification": current}
        try:
            close_result = close_position_market(symbol, direction, last_qty, trade_id=trade_id)
        except Exception as exc:
            close_result = {"status": "error", "error": str(exc)}
        attempts.append(close_result)
        time.sleep(0.35 * (attempt + 1))

    try:
        verification = get_position_directional(symbol, direction)
    except Exception as exc:
        verification = {"status": "verification_error", "error": str(exc)}
    if verification.get("status") == "not_found":
        return {"status": "closed_verified", "attempts": attempts, "verification": verification}
    return {"status": "close_unverified", "attempts": attempts, "verification": verification}

def execute_new_position(signal: dict[str, Any]) -> dict[str, Any]:
    symbol = str(signal["symbol"])
    direction = str(signal["type"]).upper()
    event_id = str(signal["event_id"])
    entry_price = float(signal["entry"])

    # Validate the planned setup before making any network call or opening a position.
    valid, reason = _validate_trade_geometry(signal)
    if not valid:
        log.warning("[EXEC_SKIPPED] %s %s | invalid_setup | %s", symbol, direction, reason)
        return {"status": "skipped_invalid_setup", "error": reason, "symbol": symbol, "direction": direction}

    # Do not open first and discover that the trigger-order endpoint is unavailable.
    # BingX can temporarily disable this endpoint under its trigger-frequency rule;
    # in that state there must be NO market entry because mandatory protection cannot
    # be installed/verified safely.
    try:
        protection_preflight = get_open_protection_directional(symbol, direction)
    except Exception as exc:
        protection_preflight = {"status": "error", "error": str(exc)}
    if protection_preflight.get("status") != "ok":
        reason = str(protection_preflight.get("error", "protection endpoint unavailable"))
        log.error("[EXEC_BLOCKED_PROTECTION_PRECHECK] %s %s | %s", symbol, direction, reason)
        return {"status": "blocked_protection_preflight", "symbol": symbol, "direction": direction, "error": reason}

    setup = _build_setup(signal)
    order = open_market(symbol, direction, entry_price, event_id)
    if order.get("status") == "skipped_min_qty":
        return order
    if order.get("status") != "opened":
        return {"status": str(order.get("status", "error")).upper(), "error": order.get("error"), "order": order}

    position = wait_for_position_fill_directional(symbol, direction, timeout_sec=int(os.environ.get("POSITION_FILL_TIMEOUT_SEC", "30")), poll_interval=0.5)
    if position.get("status") != "found":
        return {"status": "OPENED_UNCONFIRMED", "order": order, "position": position, "error": "position fill could not be confirmed"}

    avg_price = float(position["avgPrice"])
    qty = abs(float(position["positionAmt"]))
    # Recalculate ALL absolute protection levels from the real market fill.
    # Never submit targets computed from the stale signal/reference close.
    try:
        rebased = _rebase_protection_after_fill(signal, avg_price)
    except Exception as exc:
        reason = f"post-fill protection rebase failed: {exc}"
        log.critical("[SAFETY_CLOSE] %s %s | %s", symbol, direction, reason)
        close_result = _emergency_close_and_verify(symbol, direction, qty, event_id)
        cleanup = _cleanup_engine_protection(symbol, direction)
        return {"status": "opened_then_emergency_closed", "error": reason, "order": order, "position": position, "close": close_result, "protection_cleanup": cleanup, "executed_signal": dict(signal)}

    sl_price = float(rebased["sl"])
    tp1_price = float(rebased["tp1"])
    tp2_price = float(rebased["tp2"])
    actual_risk_abs = float(rebased["risk_abs"])
    actual_risk_pct = float(rebased["risk_pct"])
    tp1_pnl_pct = (abs(tp1_price - avg_price) / avg_price) * 100.0
    tp2_pnl_pct = (abs(tp2_price - avg_price) / avg_price) * 100.0
    actual_signal = dict(signal)
    actual_signal.update({
        "entry": avg_price,
        "sl": sl_price,
        "tp1": tp1_price,
        "tp2": tp2_price,
        "risk_pct": actual_risk_pct,
        "risk_abs": actual_risk_abs,
        "tp1_rr": float(rebased["tp1_rr"]),
        "tp2_rr": float(rebased["tp2_rr"]),
        "target": {
            **(signal.get("target") if isinstance(signal.get("target"), dict) else {}),
            "source": rebased["target_source"],
            "obstacle_price": rebased.get("obstacle_price"),
        },
    })
    valid, reason = _validate_trade_geometry(actual_signal)
    if not valid:
        log.critical("[SAFETY_CLOSE] %s %s | invalid post-fill protection geometry | %s", symbol, direction, reason)
        close_result = _emergency_close_and_verify(symbol, direction, qty, event_id)
        cleanup = _cleanup_engine_protection(symbol, direction)
        return {"status": "opened_then_emergency_closed", "error": reason, "order": order, "position": position, "close": close_result, "protection_cleanup": cleanup, "executed_signal": actual_signal}

    setup["entry_reference"] = avg_price
    setup["invalidation_price"] = sl_price
    setup["risk_pct"] = actual_risk_pct
    setup["target_rr"] = actual_signal["tp2_rr"]
    setup["planned_weighted_rr"] = actual_signal["tp1_rr"] * 0.50 + actual_signal["tp2_rr"] * 0.50
    setup["tp_levels"] = [
        {"leg": "tp1", "pnl_pct": tp1_pnl_pct, "close_fraction": 0.50, "price": tp1_price},
        {"leg": "tp2", "pnl_pct": tp2_pnl_pct, "close_fraction": 0.50, "price": tp2_price},
    ]
    setup["target_price"] = tp2_price

    log.info(
        "[EXEC_POST_FILL_REBASED] %s %s | fill=%s sl=%s tp1=%s tp2=%s tp1_rr=%.3f tp2_rr=%.3f target_source=%s",
        symbol, direction, avg_price, sl_price, tp1_price, tp2_price,
        actual_signal["tp1_rr"], actual_signal["tp2_rr"], rebased["target_source"],
    )

    protection = ensure_directional_protection(
        symbol, direction, avg_price, qty,
        actual_risk_pct, setup["tp_levels"], trade_id=event_id,
    )
    if protection.get("status") != "PROTECTED":
        log.critical("[SAFETY_CLOSE] %s %s | mandatory protection incomplete | %s", symbol, direction, protection)
        # Mandatory rule: never leave a newly-opened position live without BOTH
        # a verified SL and both TP legs. Attempt an immediate market rollback.
        close_result = _emergency_close_and_verify(symbol, direction, qty, event_id)
        cleanup = _cleanup_engine_protection(symbol, direction)
        try:
            time.sleep(0.25)
            verify_closed = get_position_directional(symbol, direction)
        except Exception as exc:
            verify_closed = {"status": "verification_error", "error": str(exc)}
        return {
            "status": "opened_then_emergency_closed",
            "error": protection.get("error") or protection.get("status"),
            "order": order,
            "position": position,
            "protection": protection,
            "close": close_result,
            "protection_cleanup": cleanup,
            "close_verification": verify_closed,
            "executed_signal": actual_signal,
        }

    register_active_trade(
        event_id=event_id,
        symbol=symbol,
        name=symbol,
        direction=direction,
        entry_price=avg_price,
        qty=qty,
        tp_orders=protection.get("tp_orders", []),
        sl_result=protection.get("sl_result", {}),
        event_type=f"{setup['zone'].get('kind', 'ZONE')}_ZONE_TOUCH",
        timeframe="1h",
        score=float(signal.get("score", 0.0)),
        setup={**setup, "protection_status": protection.get("status"), "protection_result": protection},
        requested_entry_price=entry_price,
    )

    return {
        "status": "opened_protected",
        "order": order,
        "position": position,
        "protection": protection,
        "executed_signal": actual_signal,
    }

def _send_signal(signal: dict[str, Any], execution: dict[str, Any] | None = None) -> None:
    try:
        display_signal = signal
        if isinstance(execution, dict) and isinstance(execution.get("executed_signal"), dict):
            display_signal = execution["executed_signal"]
        text = format_signal(display_signal, setup=_build_setup(display_signal), execution=execution)
        ok = send_tg(text)
        _append_jsonl(ACTIONS_PATH, {"ts": int(time.time() * 1000), "action": "SIGNAL", "event_id": signal["event_id"], "telegram_ok": ok})
    except Exception as exc:
        log.warning("[TG] signal send failed: %s", exc)


def main() -> None:
    started = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    scan_id = f"SCAN_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8].upper()}"

    # 1) Private account layer is optional for a scan. Never let missing credentials
    #    prevent public market analysis from running.
    private_ready = _private_layer_ready()
    if private_ready:
        try:
            mode = get_position_mode(timeout_sec=float(os.environ.get("PRIVATE_PREFLIGHT_TIMEOUT_SEC", "5")))
            log.info("[AUTH] BingX private preflight OK | position_mode=%s", mode)
        except Exception as exc:
            log.error("[AUTH] BingX private preflight failed: %s; execution/reconciliation disabled for this run", exc)
            private_ready = False
        if private_ready:
            try:
                update_active_trades()
            except Exception as exc:
                log.exception("[TRACKER] active trade update failed: %s", exc)
            try:
                reconcile_all_open_positions()
            except Exception as exc:
                log.exception("[RECON] reconciliation failed: %s", exc)
    else:
        log.info("[TRACKER] skipped: private BingX layer unavailable")
        log.info("[RECON] skipped: private BingX layer unavailable")

    # 2) Dynamic universe. Crypto uses Binance public Spot/Vision candles.
    # TradFi/equity contracts are analyzed from BingX candles because Binance
    # public Spot market data does not expose the corresponding stock universe.
    bingx_symbols = get_scan_symbols()
    log.info("[SCAN] %s symbols selected from BingX | count=%d", "WATCHLIST" if WATCHLIST_ONLY else "FULL_UNIVERSE", len(bingx_symbols))
    bingx_contract_map = contracts()
    try:
        mapped = analysis_symbols_for_bingx(bingx_symbols)
    except Exception as exc:
        log.exception("[SCAN] Binance public exchangeInfo failed: %s", exc)
        mapped = []

    # If Binance exchangeInfo itself is unreachable, still build the equity branch
    # from BingX contracts; crypto will simply be skipped rather than dropping all
    # assets or aborting the complete scan.
    analysis_universe: list[dict[str, Any]] = []
    mapped_by_symbol = {str(x.get("bingx_symbol", "")).upper(): x for x in mapped}
    for symbol in bingx_symbols:
        bx = bingx_contract_map.get(symbol) or get_contract(symbol)
        asset_class = classify_bingx_contract(bx)
        if asset_class not in BINANCE_ASSET_CLASSES:
            continue
        item = dict(mapped_by_symbol.get(symbol, {}))
        item["bingx_symbol"] = symbol
        item["asset_class"] = asset_class
        item["binance_symbol"] = item.get("binance_symbol") or symbol.replace("-", "")
        if asset_class == "CRYPTO":
            if not item.get("binance_available"):
                continue
            item["market_provider"] = "binance"
        elif asset_class == "EQUITY":
            item["market_provider"] = "bingx"
        analysis_universe.append(item)

    symbols = [str(item["bingx_symbol"]) for item in analysis_universe]
    analysis_meta = {str(item["bingx_symbol"]): item for item in analysis_universe}
    crypto_n = sum(1 for x in analysis_universe if str(x.get("asset_class")).upper() == "CRYPTO")
    equity_n = sum(1 for x in analysis_universe if str(x.get("asset_class")).upper() == "EQUITY")
    log.info("[SCAN] Eligible symbols: %d | crypto=%d equity=%d | strategy_mode=ZONE_ONLY | diagnostics_mode=%s", len(symbols), crypto_n, equity_n, DIAGNOSTICS_MODE)
    if not symbols:
        log.error("[SCAN] No eligible symbols for signal scan")

    successful_ids = _load_successful_trade_ids()
    if private_ready:
        try:
            open_keys = _position_keys(get_positions())
        except Exception as exc:
            log.exception("[POSITIONS] initial positions fetch failed; no new entries will be executed: %s", exc)
            open_keys = set()
            private_ready = False
    else:
        open_keys = set()
    scan_rows: list[dict[str, Any]] = []
    fresh_signals: list[dict[str, Any]] = []

    def scan_one(symbol: str) -> dict[str, Any]:
        """Public-market scan for one symbol. Safe to run concurrently."""
        source_name = "unknown"
        try:
            contract = get_contract(symbol)
            meta = analysis_meta.get(symbol, {})
            if not contract:
                return {
                    "symbol": symbol, "current_price": None, "price_position": "CONTRACT_NOT_FOUND",
                    "fresh_signal": "—", "active_demand": 0, "active_supply": 0,
                    "zones": {"demand": [], "supply": []}, "last_signal_count": 0,
                    "error": "contract_not_found", "signals": [],
                }

            binance_symbol = str(meta.get("binance_symbol") or "")
            provider = str(meta.get("market_provider") or "binance").lower()
            if provider == "bingx":
                bars = fetch_bingx_klines(symbol, "1h", limit=KLINE_LIMIT_1H, retryable=False)
                source_name = "bingx"
            else:
                bars = fetch_binance_klines(binance_symbol, "1h", limit=KLINE_LIMIT_1H, retryable=False)
                source_name = "binance_spot"
            min_bars = SWING_LEN * 2 + 10
            if len(bars) < min_bars:
                return {
                    "symbol": symbol, "current_price": None, "price_position": "INSUFFICIENT_DATA",
                    "fresh_signal": "—", "active_demand": 0, "active_supply": 0,
                    "zones": {"demand": [], "supply": []}, "last_signal_count": 0,
                    "error": f"insufficient_1h_candles:{len(bars)}<{min_bars}", "signals": [],
                }

            df, supply, demand, signals = generate_zone_signals(pd.DataFrame(bars), symbol=symbol, mode=DIAGNOSTICS_MODE)
            latest_price = float(df["close"].iloc[-1])
            bingx_price = _bingx_last_price(contract)
            latest_closed_idx = len(df) - 1
            latest_closed_time = pd.Timestamp(df["timestamp"].iloc[-1])
            now_utc = pd.Timestamp.now(tz="UTC")
            data_age_hours = max(0.0, (now_utc - latest_closed_time).total_seconds() / 3600.0)

            # Never treat an old Binance series as current just because its last
            # row happens to have index len(df)-1. This specifically protects against
            # stale/delisted symbols such as historical-only series.
            if data_age_hours > MAX_DATA_STALENESS_HOURS:
                log.warning(
                    "[DATA_STALE_REJECT] %s | latest_closed_time=%s age_hours=%.2f allowed_hours=%.2f",
                    symbol, latest_closed_time.isoformat(), data_age_hours, MAX_DATA_STALENESS_HOURS,
                )
                return {
                    "symbol": symbol,
                    "current_price": latest_price,
                    "binance_price": latest_price,
                    "bingx_price": bingx_price,
                    "market_spread_pct": None,
                    "market_source": source_name,
                    "binance_symbol": binance_symbol,
                    "asset_class": meta.get("asset_class", "UNKNOWN"),
                    "price_position": _price_position(latest_price, demand, supply),
                    "fresh_signal": "—",
                    "active_demand": len(demand),
                    "active_supply": len(supply),
                    "zones": {"demand": demand, "supply": supply},
                    "last_signal_count": 0,
                    "latest_closed_idx": int(latest_closed_idx),
                    "latest_closed_time": latest_closed_time.isoformat(),
                    "signals": [],
                    "error": f"stale_1h_data:{data_age_hours:.2f}h>{MAX_DATA_STALENESS_HOURS:.2f}h",
                }

            # ZONE_ONLY trading decisions come only from a fresh Demand/Supply
            # touch on the latest closed 1H candle. ALMA/Pine diagnostics may still
            # be attached to the dataframe, but they never gate execution.
            recent = [
                s for s in signals
                if int(s.get("idx", -1)) == int(latest_closed_idx)
                and pd.Timestamp(s.get("time")) == latest_closed_time
            ]
            for sig in recent:
                sig["score"] = score_zone_signal(sig)

            # Only validate BingX live price when a fresh signal exists. This keeps
            # the full-market scan on Binance while spending a small number of extra
            # BingX public requests only on actionable candidates.
            if recent and bingx_price is None:
                try:
                    bx_live = fetch_bingx_klines(symbol, "1m", limit=1, retryable=False)
                    if bx_live:
                        bingx_price = float(bx_live[-1]["close"])
                except Exception as bx_exc:
                    log.warning("[MARKET_CHECK] %s | BingX price validation failed: %s", symbol, bx_exc)
            spread_pct = _market_spread_pct(latest_price, bingx_price) if provider == "binance" else None
            if spread_pct is not None and spread_pct > MAX_MARKET_SPREAD_PCT:
                log.warning("[MARKET_SPREAD] %s | Binance=%.12g | BingX=%.12g | spread=%.4f%% > %.4f%%", symbol, latest_price, bingx_price, spread_pct, MAX_MARKET_SPREAD_PCT)
                recent = []
            # Signal freshness is controlled by MAX_SIGNAL_AGE_BARS, but the
            # executable candidate must always be the newest signal bar. Score is
            # only a tie-breaker for multiple signals on the same bar.
            latest_signal = _select_latest_signal(recent)
            if not latest_signal and spread_pct is not None and spread_pct > MAX_MARKET_SPREAD_PCT:
                fresh_text = f"BLOCKED_SPREAD>{MAX_MARKET_SPREAD_PCT:.2f}%"
            else:
                fresh_text = None
            price_position = _price_position(latest_price, demand, supply)
            if fresh_text is None:
                fresh_text = (
                    f"{latest_signal['type']} @ {latest_signal['entry']} score={latest_signal.get('score', 0):.1f}"
                    if latest_signal else "—"
                )
            return {
                "symbol": symbol,
                "current_price": latest_price,
                "binance_price": latest_price,
                "bingx_price": bingx_price,
                "market_spread_pct": spread_pct,
                "market_source": source_name,
                "binance_symbol": binance_symbol,
                "asset_class": meta.get("asset_class", "UNKNOWN"),
                "price_position": price_position,
                "fresh_signal": fresh_text,
                "active_demand": len(demand),
                "active_supply": len(supply),
                "zones": {"demand": demand, "supply": supply},
                "last_signal_count": len(recent),
                "latest_closed_idx": int(latest_closed_idx),
                "latest_closed_time": latest_closed_time.isoformat(),
                "signals": recent,
            }
        except Exception as exc:
            return {
                "symbol": symbol, "current_price": None, "binance_price": None, "bingx_price": None, "market_spread_pct": None,
                "market_source": source_name, "binance_symbol": analysis_meta.get(symbol, {}).get("binance_symbol"),
                "asset_class": analysis_meta.get(symbol, {}).get("asset_class", "UNKNOWN"), "price_position": "ERROR",
                "fresh_signal": "—", "active_demand": 0, "active_supply": 0,
                "zones": {"demand": [], "supply": []}, "last_signal_count": 0,
                "error": f"{type(exc).__name__}: {exc}", "signals": [],
                "exception": exc,
            }

    # Parallel public-market scan. Work is submitted in bounded batches so the
    # engine is much faster than serial I/O without opening hundreds of sockets at once.
    total = len(symbols)
    for batch_start in range(0, total, SCAN_BATCH_SIZE):
        batch = symbols[batch_start: batch_start + SCAN_BATCH_SIZE]
        batch_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(SCAN_WORKERS, len(batch))) as executor:
            future_to_symbol = {executor.submit(scan_one, symbol): symbol for symbol in batch}
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                except Exception as exc:  # defensive: scan_one already catches errors
                    result = {
                        "symbol": symbol, "current_price": None, "binance_price": None, "bingx_price": None, "market_spread_pct": None,
                "market_source": source_name, "binance_symbol": analysis_meta.get(symbol, {}).get("binance_symbol"),
                "asset_class": analysis_meta.get(symbol, {}).get("asset_class", "UNKNOWN"), "price_position": "ERROR",
                        "fresh_signal": "—", "active_demand": 0, "active_supply": 0,
                        "zones": {"demand": [], "supply": []}, "last_signal_count": 0,
                        "error": f"{type(exc).__name__}: {exc}", "signals": [], "exception": exc,
                    }
                batch_results[symbol] = result

        for symbol in batch:
            result = batch_results[symbol]
            result.pop("exception", None)
            scan_rows.append({k: v for k, v in result.items() if k != "signals"})
            fresh_signals.extend(result.get("signals", []))

            if result.get("price_position") == "ERROR":
                log.error("[COIN_ERROR] %s | %s", symbol, result.get("error", "unknown error"))
            elif result.get("price_position") == "INSUFFICIENT_DATA":
                log.warning("[COIN_SKIP] %s | %s", symbol, result.get("error", "insufficient data"))
            elif result.get("price_position") == "CONTRACT_NOT_FOUND":
                log.warning("[COIN_SKIP] %s | contract not found in BingX cache", symbol)
            else:
                # Log only active symbols: a current Pine signal or a price inside
                # an active Demand/Supply zone. Inactive "outside zones" symbols
                # are intentionally omitted from runtime logs.
                is_active = (
                    result.get("fresh_signal") not in {None, "—"}
                    or result.get("price_position") in {"🟢 В зоне DEMAND", "🔴 В зоне SUPPLY"}
                )
                if is_active:
                    log.info(
                        "[ACTIVE] %s | price=%s | %s | signal=%s | demand=%d | supply=%d",
                        symbol, result["current_price"], result["price_position"], result["fresh_signal"],
                        result["active_demand"], result["active_supply"],
                    )

        scanned = min(batch_start + len(batch), total)
        log.info("[SCAN_PROGRESS] %d/%d symbols | batch=%d | workers=%d", scanned, total, len(batch), min(SCAN_WORKERS, len(batch)))
        if scanned < total and SCAN_BATCH_PAUSE_SEC > 0:
            time.sleep(SCAN_BATCH_PAUSE_SEC)

    scan_errors = sum(1 for row in scan_rows if row.get("price_position") == "ERROR")
    scan_skips = sum(1 for row in scan_rows if row.get("price_position") in {"INSUFFICIENT_DATA", "CONTRACT_NOT_FOUND"})
    log.info(
        "[SCAN_DONE] symbols=%d errors=%d skipped=%d fresh_signals=%d duration=%.1fs",
        total, scan_errors, scan_skips, len(fresh_signals), time.time() - started,
    )

    # Execution safety: choose exactly ONE zone signal per symbol, namely the
    # newest fresh-touch bar. Never allow an older setup to compete with a newer
    # setup because of score sorting.
    latest_by_symbol: dict[str, dict[str, Any]] = {}
    for signal in fresh_signals:
        symbol_key = str(signal["symbol"]).upper()
        previous = latest_by_symbol.get(symbol_key)
        candidate_key = (int(signal["idx"]), float(signal.get("score", 0.0)))
        previous_key = (int(previous["idx"]), float(previous.get("score", 0.0))) if previous else None
        if previous is None or candidate_key > previous_key:
            latest_by_symbol[symbol_key] = signal

    executable: list[dict[str, Any]] = []
    latest_index_by_symbol = {
        str(r.get("symbol", "")).upper(): r.get("latest_closed_idx")
        for r in scan_rows
        if r.get("latest_closed_idx") is not None
    }
    latest_time_by_symbol = {
        str(r.get("symbol", "")).upper(): r.get("latest_closed_time")
        for r in scan_rows
        if r.get("latest_closed_time") is not None
    }
    for signal in latest_by_symbol.values():
        symbol_key = str(signal["symbol"]).upper()
        latest_closed_idx = latest_index_by_symbol.get(symbol_key)
        if latest_closed_idx is None:
            log.warning("[EXEC_REJECT_NO_LATEST] %s %s | latest_closed_idx unavailable", signal["symbol"], signal["type"])
            continue
        latest_closed_time = latest_time_by_symbol.get(symbol_key)
        signal_age = int(latest_closed_idx) - int(signal["idx"])
        signal["execution_age_bars"] = int(signal_age)
        signal["latest_closed_idx"] = int(latest_closed_idx)
        signal["latest_closed_time"] = latest_closed_time
        if signal_age > EXECUTION_MAX_SIGNAL_AGE_BARS:
            log.info("[EXEC_REJECT_AGE] %s %s | signal_idx=%s signal_time=%s latest_closed_idx=%s latest_closed_time=%s age_bars=%s allowed=%s", signal["symbol"], signal["type"], signal.get("idx"), signal.get("time"), latest_closed_idx, latest_closed_time, signal_age, EXECUTION_MAX_SIGNAL_AGE_BARS)
            continue
        matches_latest, reject_reason = _signal_matches_latest_bar(signal, latest_closed_idx, latest_closed_time)
        if not matches_latest:
            log.warning(
                "[EXEC_REJECT_LATEST_BAR] %s %s | reason=%s signal_idx=%s signal_time=%s latest_closed_idx=%s latest_closed_time=%s age_bars=%s",
                signal["symbol"], signal["type"], reject_reason, signal.get("idx"), signal.get("time"),
                latest_closed_idx, latest_closed_time, signal_age,
            )
            continue
        bx = get_contract(signal["symbol"])
        bx_symbol = str((bx or {}).get("symbol", signal["symbol"])).upper()
        key = (bx_symbol, signal["type"])
        opposite = (bx_symbol, "SHORT" if signal["type"] == "LONG" else "LONG")
        if signal["event_id"] in successful_ids:
            continue
        if key in open_keys or opposite in open_keys:
            continue
        executable.append(signal)

    # Safety ordering: newest signal bar first; score only breaks ties.
    executable.sort(key=lambda x: (-int(x["idx"]), -float(x.get("score", 0.0))))

    executed = 0
    for signal in executable[:MAX_TRADES_PER_CYCLE]:
        if not EXECUTION_ENABLED:
            _send_signal(signal, {"status": "DISABLED"})
            continue
        log.info(
            "[EXEC_SIGNAL] symbol=%s direction=%s signal_idx=%s signal_time=%s age_bars=%s "
            "zone=%s zone_low=%s zone_high=%s target_source=%s obstacle=%s tp1=%s tp2=%s event_id=%s",
            signal.get("symbol"), signal.get("type"), signal.get("idx"), signal.get("time"),
            signal.get("execution_age_bars", 0),
            (signal.get("zone") or {}).get("kind"),
            (signal.get("zone") or {}).get("btm"),
            (signal.get("zone") or {}).get("top"),
            (signal.get("target") or {}).get("source"),
            (signal.get("target") or {}).get("obstacle_price"),
            signal.get("tp1"), signal.get("tp2"), signal.get("event_id"),
        )
        if not private_ready:
            blocked = {"status": "BLOCKED_MISSING_CREDENTIALS", "error": "BingX private credentials are unavailable"}
            log.error("[EXEC_BLOCKED] %s %s: missing BingX private credentials", signal["symbol"], signal["type"])
            _append_jsonl(TRADES_PATH, {
                "record_type": "TRADE_BLOCKED",
                "event_id": signal["event_id"],
                "symbol": signal["symbol"],
                "direction": signal["type"],
                "score": signal["score"],
                "signal": signal,
                "result": blocked,
            })
            _send_signal(signal, blocked)
            continue
        execution = execute_new_position(signal)
        execution_status = str(execution.get("status", ""))
        if execution_status in {"skipped_min_qty", "skipped_tp_min_qty", "skipped_invalid_setup"}:
            log.warning(
                "[EXEC_SKIPPED] %s %s | status=%s | reason=%s | error=%s | qty=%s min_qty=%s required_margin=%.4f configured_margin=%.4f",
                signal["symbol"], signal["type"], execution_status, execution.get("reason", "invalid_setup"), execution.get("error"), execution.get("qty"), execution.get("min_qty"),
                float(execution.get("required_margin_usdt", 0.0) or 0.0), float(execution.get("configured_margin_usdt", MARGIN_USDT) or MARGIN_USDT),
            )
        elif not execution_status.startswith("opened"):
            log.error("[EXEC_FAILED] %s %s | status=%s | error=%s | order=%s", signal["symbol"], signal["type"], execution_status, execution.get("error"), execution.get("order"))
        _append_jsonl(TRADES_PATH, {"record_type": "TRADE_OPEN", "event_id": signal["event_id"], "symbol": signal["symbol"], "direction": signal["type"], "score": signal["score"], "signal": signal, "result": execution})
        _send_signal(signal, execution)
        if str(execution.get("status")) == "opened_protected":
            executed += 1

    # Persist the complete scan for analytics/backtesting, but do not dump the
    # full inactive-symbol table into runtime logs. Runtime logs contain active
    # symbols only.
    save_scan(scan_rows, fresh_signals, duration_sec=time.time() - started, scan_id=scan_id)
    active_log_rows = [
        r for r in scan_rows
        if r.get("fresh_signal") not in {None, "—"}
        or r.get("price_position") in {"🟢 В зоне DEMAND", "🔴 В зоне SUPPLY"}
    ]
    log.info("[ACTIVE_SUMMARY] active_symbols=%d signals=%d", len(active_log_rows), len(fresh_signals))
    log.info("[DONE] symbols=%d fresh_signals=%d executed=%d duration=%.1fs", len(scan_rows), len(fresh_signals), executed, time.time() - started)


if __name__ == "__main__":
    main()
