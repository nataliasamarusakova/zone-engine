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
KLINE_LIMIT_1H = int(os.environ.get("KLINE_LIMIT_1H", "120"))
MAX_SIGNAL_AGE_BARS = int(os.environ.get("MAX_SIGNAL_AGE_BARS", "0"))
# Production execution is strict by default: only the latest closed 1H bar may open a trade.
EXECUTION_MAX_SIGNAL_AGE_BARS = int(os.environ.get("EXECUTION_MAX_SIGNAL_AGE_BARS", "0"))
PINE_SIGNAL_MODE = os.environ.get("PINE_SIGNAL_MODE", "historical").strip().lower()
if PINE_SIGNAL_MODE not in {"historical", "live"}:
    raise ValueError("PINE_SIGNAL_MODE must be historical or live")
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
    symbols = []
    for c in all_contracts.values():
        symbol = _symbol_from_contract(c)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    symbols.sort()
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
    """Strict live-execution identity check for the signal bar.

    A signal is executable only when its index AND timestamp are exactly the latest
    closed 1H bar. This prevents a stale/incorrect timestamp from masquerading as
    a fresh signal merely because the DataFrame index is the last row.
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
    trigger = signal.get("trigger") or {}
    pine_buy = bool(trigger.get("buy"))
    pine_sell = bool(trigger.get("sell"))
    expected = "LONG" if pine_buy and not pine_sell else "SHORT" if pine_sell and not pine_buy else None
    if expected != str(signal.get("type", "")).upper():
        return False, "direction_not_equal_to_pine_trigger"
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
    in_dem = any(price <= float(z["top"]) * 1.005 and price >= float(z["btm"]) for z in demand)
    in_sup = any(price >= float(z["btm"]) * 0.995 and price <= float(z["top"]) for z in supply)
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
        "strategy": "Ajay R5.41",
        "signal_price": float(signal["entry"]),
        "entry_reference": float(signal["entry"]),
        "invalidation_price": float(signal["sl"]),
        "risk_pct": risk_pct,
        "target_rr": TP2_R,
        "planned_weighted_rr": TP1_R * 0.50 + TP2_R * 0.50,
        "tp_levels": [
            {"leg": "tp1", "pnl_pct": TP1_R * risk_pct, "close_fraction": 0.50},
            {"leg": "tp2", "pnl_pct": TP2_R * risk_pct, "close_fraction": 0.50},
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
    if risk_pct > float(os.environ.get("MAX_SIGNAL_RISK_PCT", "25")):
        return False, f"risk_pct_above_limit={risk_pct}"
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

def execute_new_position(signal: dict[str, Any]) -> dict[str, Any]:
    symbol = str(signal["symbol"])
    direction = str(signal["type"]).upper()
    event_id = str(signal["event_id"])
    entry_price = float(signal["entry"])

    valid, reason = _validate_trade_geometry(signal)
    if not valid:
        log.warning("[EXEC_SKIPPED] %s %s | invalid_setup | %s", symbol, direction, reason)
        return {"status": "skipped_invalid_setup", "error": reason, "symbol": symbol, "direction": direction}

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
    # Rebuild mandatory protection from the ACTUAL fill, not the reference price.
    sl_price, tp1_price, tp2_price = _protection_geometry_from_fill(direction, avg_price, float(signal["risk_pct"]))
    actual_signal = dict(signal)
    actual_signal.update({"entry": avg_price, "sl": sl_price, "tp1": tp1_price, "tp2": tp2_price})
    valid, reason = _validate_trade_geometry(actual_signal)
    if not valid:
        log.critical("[SAFETY_CLOSE] %s %s | invalid post-fill protection geometry | %s", symbol, direction, reason)
        close_result = close_position_market(symbol, direction, qty, trade_id=event_id)
        cleanup = _cleanup_engine_protection(symbol, direction)
        return {"status": "opened_then_emergency_closed", "error": reason, "order": order, "position": position, "close": close_result, "protection_cleanup": cleanup}

    setup["entry_reference"] = avg_price
    setup["invalidation_price"] = sl_price
    setup["tp_levels"] = [
        {"leg": "tp1", "pnl_pct": TP1_R * float(signal["risk_pct"]), "close_fraction": 0.50, "price": tp1_price},
        {"leg": "tp2", "pnl_pct": TP2_R * float(signal["risk_pct"]), "close_fraction": 0.50, "price": tp2_price},
    ]
    setup["target_price"] = tp2_price

    protection = ensure_directional_protection(
        symbol, direction, avg_price, qty,
        float(signal["risk_pct"]), setup["tp_levels"], trade_id=event_id,
    )
    if protection.get("status") != "PROTECTED":
        log.critical("[SAFETY_CLOSE] %s %s | mandatory protection incomplete | %s", symbol, direction, protection)
        # Mandatory rule: never leave a newly-opened position live without BOTH
        # a verified SL and both TP legs. Attempt an immediate market rollback.
        try:
            current = get_position_directional(symbol, direction)
            remaining_qty = abs(float(current.get("positionAmt", qty) or qty)) if current.get("status") == "found" else qty
        except Exception:
            remaining_qty = qty
        close_result = close_position_market(symbol, direction, remaining_qty, trade_id=event_id)
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
        event_type=f"{setup['zone'].get('kind', 'ZONE')}_ALMA_CROSS",
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
    }

def _send_signal(signal: dict[str, Any], execution: dict[str, Any] | None = None) -> None:
    try:
        text = format_signal(signal, setup=_build_setup(signal), execution=execution)
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
    log.info("[SCAN] BingX active USDT symbols: %d", len(bingx_symbols))
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
    log.info("[SCAN] Eligible symbols: %d | crypto=%d equity=%d | pine_mode=%s", len(symbols), crypto_n, equity_n, PINE_SIGNAL_MODE)
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

            df, supply, demand, signals = generate_zone_signals(pd.DataFrame(bars), symbol=symbol, mode=PINE_SIGNAL_MODE)
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

            # Trading decisions are derived ONLY from the Pine trigger on the
            # latest closed 1H candle. We deliberately do not backfill the live
            # developing-8H model across historical bars: those historical
            # developing states are not what exists on the currently running
            # TradingView bar and must never become execution candidates.
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

    # Execution safety: choose exactly ONE signal per symbol, namely the newest
    # Pine ALMA crossover bar. Never allow an older SHORT to compete with a newer
    # LONG (or vice versa) because of score sorting.
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
                "[EXEC_REJECT_LATEST_BAR] %s %s | reason=%s signal_idx=%s signal_time=%s latest_closed_idx=%s latest_closed_time=%s age_bars=%s pine_buy=%s pine_sell=%s",
                signal["symbol"], signal["type"], reject_reason, signal.get("idx"), signal.get("time"),
                latest_closed_idx, latest_closed_time, signal_age,
                (signal.get("trigger") or {}).get("buy"), (signal.get("trigger") or {}).get("sell"),
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
        log.warning(
            "[EXEC_SIGNAL] symbol=%s direction=%s signal_idx=%s signal_time=%s age_bars=%s "
            "pine_buy=%s pine_sell=%s alma_close=%s alma_open=%s event_id=%s",
            signal.get("symbol"), signal.get("type"), signal.get("idx"), signal.get("time"),
            signal.get("execution_age_bars", 0),
            (signal.get("trigger") or {}).get("buy"), (signal.get("trigger") or {}).get("sell"),
            (signal.get("trigger") or {}).get("alma_close_alt"), (signal.get("trigger") or {}).get("alma_open_alt"),
            signal.get("event_id"),
        )
        execution = execute_new_position(signal)
        execution_status = str(execution.get("status", ""))
        if execution_status in {"skipped_min_qty", "skipped_invalid_setup"}:
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
