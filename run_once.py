from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from event_engine.analytics import save_scan
from event_engine.bingx import (
    contracts,
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
KLINE_LIMIT_1H = int(os.environ.get("KLINE_LIMIT_1H", "220"))
MAX_SIGNAL_AGE_BARS = int(os.environ.get("MAX_SIGNAL_AGE_BARS", "3"))
MIN_SIGNAL_SCORE = float(os.environ.get("MIN_SIGNAL_SCORE", "70"))
SCAN_INTERVAL_SEC = float(os.environ.get("SCAN_INTERVAL_SEC", "0.15"))
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

    # 1) Existing positions first: TP/SL/BE lifecycle + missing protection repair.
    update_active_trades()
    reconcile_all_open_positions()

    # 2) Dynamic universe from BingX.
    symbols = get_scan_symbols()
    log.info("[SCAN] BingX active USDT symbols: %d", len(symbols))

    successful_ids = _load_successful_trade_ids()
    open_keys = _position_keys(get_positions())
    scan_rows: list[dict[str, Any]] = []
    fresh_signals: list[dict[str, Any]] = []

    for symbol in symbols:
        try:
            contract = get_contract(symbol)
            if not contract:
                continue
            bars = fetch_klines(symbol, "1h", limit=KLINE_LIMIT_1H, retryable=False)
            if len(bars) < SWING_LEN * 2 + 10:
                continue
            df, supply, demand, signals = generate_zone_signals(pd.DataFrame(bars), symbol=symbol)
            latest_price = float(df["close"].iloc[-1])
            recent = [s for s in signals if int(s["idx"]) >= len(df) - MAX_SIGNAL_AGE_BARS]
            for sig in recent:
                sig["score"] = score_zone_signal(sig)
            recent = [s for s in recent if float(s["score"]) >= MIN_SIGNAL_SCORE]
            latest_signal = recent[-1] if recent else None
            price_position = _price_position(latest_price, demand, supply)
            fresh_text = f"{latest_signal['type']} @ {latest_signal['entry']} score={latest_signal.get('score', 0):.1f}" if latest_signal else "—"
            scan_row = {
                "symbol": symbol,
                "current_price": latest_price,
                "price_position": price_position,
                "fresh_signal": fresh_text,
                "active_demand": len(demand),
                "active_supply": len(supply),
                "zones": {"demand": demand, "supply": supply},
                "last_signal_count": len(recent),
            }
            scan_rows.append(scan_row)
            log.info(
                "[COIN] %s | price=%s | %s | fresh=%s | demand=%d | supply=%d",
                symbol, latest_price, price_position, fresh_text, len(demand), len(supply)
            )
            fresh_signals.extend(recent)
            if SCAN_INTERVAL_SEC > 0:
                time.sleep(SCAN_INTERVAL_SEC)
        except Exception as exc:
            log.exception("[COIN_ERROR] %s scan failed: %s", symbol, exc)
            scan_rows.append({
                "symbol": symbol,
                "current_price": None,
                "price_position": "ERROR",
                "fresh_signal": "—",
                "active_demand": 0,
                "active_supply": 0,
                "zones": {"demand": [], "supply": []},
                "last_signal_count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            })

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
