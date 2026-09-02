from __future__ import annotations

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from event_engine.analytics import save_scan
from event_engine.bingx import (
    contracts,
    credentials_available,
    ensure_directional_protection,
    fetch_klines,
    get_contract,
    get_positions,
    get_position_directional,
    get_open_protection_directional,
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
MAX_SIGNAL_AGE_BARS = int(os.environ.get("MAX_SIGNAL_AGE_BARS", "3"))
MIN_SIGNAL_SCORE = float(os.environ.get("MIN_SIGNAL_SCORE", "70"))
SCAN_WORKERS = max(1, int(os.environ.get("SCAN_WORKERS", "12")))
SCAN_BATCH_SIZE = max(SCAN_WORKERS, int(os.environ.get("SCAN_BATCH_SIZE", "48")))
SCAN_BATCH_PAUSE_SEC = max(0.0, float(os.environ.get("SCAN_BATCH_PAUSE_SEC", "0.10")))
RECONCILIATION_MAX_SECONDS = float(os.environ.get("RECONCILIATION_MAX_SECONDS", "45"))


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
                {"leg": "tp1", "pnl_pct": risk_pct * 1.5, "close_fraction": 0.50},
                {"leg": "tp2", "pnl_pct": risk_pct * 3.0, "close_fraction": 0.50},
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
        "strategy": "DEMAND_SUPPLY_SMART_MONEY",
        "signal_price": float(signal["entry"]),
        "entry_reference": float(signal["entry"]),
        "invalidation_price": float(signal["sl"]),
        "risk_pct": risk_pct,
        "target_rr": 3.0,
        "planned_weighted_rr": 2.25,
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


def execute_new_position(signal: dict[str, Any]) -> dict[str, Any]:
    symbol = str(signal["symbol"])
    direction = str(signal["type"]).upper()
    event_id = str(signal["event_id"])
    entry_price = float(signal["entry"])
    setup = _build_setup(signal)

    order = open_market(symbol, direction, entry_price, event_id)
    if order.get("status") != "opened":
        return {"status": str(order.get("status", "error")).upper(), "error": order.get("error"), "order": order}

    position = wait_for_position_fill_directional(symbol, direction, timeout_sec=int(os.environ.get("POSITION_FILL_TIMEOUT_SEC", "30")), poll_interval=0.5)
    if position.get("status") != "found":
        return {"status": "OPENED_UNCONFIRMED", "order": order, "position": position}

    avg_price = float(position["avgPrice"])
    qty = abs(float(position["positionAmt"]))
    setup["entry_reference"] = avg_price
    setup["invalidation_price"] = avg_price * (1.0 - setup["risk_pct"] / 100.0) if direction == "LONG" else avg_price * (1.0 + setup["risk_pct"] / 100.0)
    setup["tp_levels"] = [
        {"leg": "tp1", "pnl_pct": TP1_R * setup["risk_pct"], "close_fraction": 0.50},
        {"leg": "tp2", "pnl_pct": TP2_R * setup["risk_pct"], "close_fraction": 0.50},
    ]
    setup["target_price"] = avg_price * (1.0 + TP2_R * setup["risk_pct"] / 100.0) if direction == "LONG" else avg_price * (1.0 - TP2_R * setup["risk_pct"] / 100.0)

    protection = ensure_directional_protection(
        symbol, direction, avg_price, qty,
        setup["risk_pct"], setup["tp_levels"], trade_id=event_id,
    )
    if protection.get("status") not in {"PROTECTED"}:
        # Entry remains live; tracker/reconciliation will repair missing protection on next cycle.
        log.error("[EXEC] protection incomplete for %s %s: %s", symbol, direction, protection)

    register_active_trade(
        event_id=event_id,
        symbol=symbol,
        name=symbol,
        direction=direction,
        entry_price=avg_price,
        qty=qty,
        tp_orders=protection.get("tp_orders", []),
        sl_result=protection.get("sl_result", {}),
        event_type=f"{setup['zone'].get('kind', 'ZONE')}_HMA_CROSS",
        timeframe="1h",
        score=float(signal.get("score", 0.0)),
        setup={**setup, "protection_status": protection.get("status"), "protection_result": protection},
        requested_entry_price=entry_price,
    )

    result = {
        "status": "opened_protected" if protection.get("status") == "PROTECTED" else "opened_protection_check_required",
        "order": order,
        "position": position,
        "protection": protection,
    }
    return result


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

    # 2) Dynamic universe from public BingX contracts.
    symbols = get_scan_symbols()
    log.info("[SCAN] BingX active USDT symbols: %d", len(symbols))
    if not symbols:
        log.error("[SCAN] No active USDT symbols returned by BingX; no market scan was performed")

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
        try:
            contract = get_contract(symbol)
            if not contract:
                return {
                    "symbol": symbol, "current_price": None, "price_position": "CONTRACT_NOT_FOUND",
                    "fresh_signal": "—", "active_demand": 0, "active_supply": 0,
                    "zones": {"demand": [], "supply": []}, "last_signal_count": 0,
                    "error": "contract_not_found", "signals": [],
                }

            bars = fetch_klines(symbol, "1h", limit=KLINE_LIMIT_1H, retryable=False)
            min_bars = SWING_LEN * 2 + 10
            if len(bars) < min_bars:
                return {
                    "symbol": symbol, "current_price": None, "price_position": "INSUFFICIENT_DATA",
                    "fresh_signal": "—", "active_demand": 0, "active_supply": 0,
                    "zones": {"demand": [], "supply": []}, "last_signal_count": 0,
                    "error": f"insufficient_1h_candles:{len(bars)}<{min_bars}", "signals": [],
                }

            df, supply, demand, signals = generate_zone_signals(pd.DataFrame(bars), symbol=symbol)
            latest_price = float(df["close"].iloc[-1])
            recent = [s for s in signals if int(s["idx"]) >= len(df) - MAX_SIGNAL_AGE_BARS]
            for sig in recent:
                sig["score"] = score_zone_signal(sig)
            recent = [s for s in recent if float(s["score"]) >= MIN_SIGNAL_SCORE]
            latest_signal = max(recent, key=lambda s: (float(s.get("score", 0)), int(s.get("idx", 0)))) if recent else None
            price_position = _price_position(latest_price, demand, supply)
            fresh_text = (
                f"{latest_signal['type']} @ {latest_signal['entry']} score={latest_signal.get('score', 0):.1f}"
                if latest_signal else "—"
            )
            return {
                "symbol": symbol,
                "current_price": latest_price,
                "price_position": price_position,
                "fresh_signal": fresh_text,
                "active_demand": len(demand),
                "active_supply": len(supply),
                "zones": {"demand": demand, "supply": supply},
                "last_signal_count": len(recent),
                "signals": recent,
            }
        except Exception as exc:
            return {
                "symbol": symbol, "current_price": None, "price_position": "ERROR",
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
                        "symbol": symbol, "current_price": None, "price_position": "ERROR",
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
                log.debug(
                    "[COIN] %s | price=%s | %s | fresh=%s | demand=%d | supply=%d",
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

    # Keep one best fresh signal per symbol/direction and block existing exposure.
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for signal in fresh_signals:
        key = (signal["symbol"], signal["type"])
        previous = by_key.get(key)
        if previous is None or (float(signal["score"]), int(signal["idx"])) > (float(previous["score"]), int(previous["idx"])):
            by_key[key] = signal

    executable: list[dict[str, Any]] = []
    for signal in by_key.values():
        bx = get_contract(signal["symbol"])
        bx_symbol = str((bx or {}).get("symbol", signal["symbol"])).upper()
        key = (bx_symbol, signal["type"])
        opposite = (bx_symbol, "SHORT" if signal["type"] == "LONG" else "LONG")
        if signal["event_id"] in successful_ids:
            continue
        if key in open_keys or opposite in open_keys:
            continue
        executable.append(signal)
    executable.sort(key=lambda x: (-float(x["score"]), -int(x["idx"])))

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
        execution = execute_new_position(signal)
        _append_jsonl(TRADES_PATH, {"record_type": "TRADE_OPEN", "event_id": signal["event_id"], "symbol": signal["symbol"], "direction": signal["type"], "score": signal["score"], "signal": signal, "result": execution})
        _send_signal(signal, execution)
        if str(execution.get("status")).startswith("opened"):
            executed += 1

    # Save the exact table + detailed rows for later tuning/backtesting.
    table = save_scan(scan_rows, fresh_signals, duration_sec=time.time() - started, scan_id=scan_id)
    log.info("\n%s", table)
    log.info("[DONE] symbols=%d fresh_signals=%d executed=%d duration=%.1fs", len(scan_rows), len(fresh_signals), executed, time.time() - started)


if __name__ == "__main__":
    main()
