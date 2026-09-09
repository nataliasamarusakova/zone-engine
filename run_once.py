from __future__ import annotations

import json
import math
import logging
import os
import time
import uuid
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows development fallback
    fcntl = None
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from event_engine.analytics import save_scan
from event_engine.binance import analysis_symbols_for_bingx, classify_bingx_contract, fetch_24h_ticker, fetch_klines as fetch_binance_klines
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
    LEVERAGE,
    cancel_order,
    close_position_market,
    open_market,
    wait_for_position_fill_directional,
)
from event_engine.signals import SWING_LEN, TP1_R, TP2_R, generate_zone_signals, score_zone_signal
from event_engine.levels import protective_price, select_opposing_levels, select_protective_level
from event_engine.telegram import format_signal, send as send_tg
from event_engine.tracker import get_active_trade_id, process_structural_exits, recover_exchange_position, register_active_trade, update_active_trades, update_active_trade_protection

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("zone_engine")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data"
TRADES_PATH = DATA / "trades.jsonl"
FAILED_SIGNALS_PATH = DATA / "failed_signals.json"
FAILED_SIGNAL_TTL_SEC = int(os.environ.get("FAILED_SIGNAL_TTL_SEC", str(24 * 3600)))
FAILED_SIGNAL_MAX_RETRIES = max(1, int(os.environ.get("FAILED_SIGNAL_MAX_RETRIES", "3")))
FAILED_SIGNAL_RETRY_BASE_SEC = max(1, int(os.environ.get("FAILED_SIGNAL_RETRY_BASE_SEC", "300")))
FAILED_SIGNAL_RETRY_MAX_SEC = max(FAILED_SIGNAL_RETRY_BASE_SEC, int(os.environ.get("FAILED_SIGNAL_RETRY_MAX_SEC", str(3600))))
ACTIONS_PATH = DATA / "actions.jsonl"
EXECUTION_LOCK_PATH = DATA / "execution.lock"

EXECUTION_ENABLED = os.environ.get("EXECUTION_ENABLED", "true").lower() == "true"
MARGIN_USDT = float(os.environ.get("BINGX_MARGIN_USDT", "1"))
MAX_TRADES_PER_CYCLE = int(os.environ.get("MAX_TRADES_PER_CYCLE", "5"))
STRONG_ZONE_MARGIN_USDT = float(os.environ.get("STRONG_ZONE_MARGIN_USDT", "1.00"))
MEDIUM_ZONE_MARGIN_USDT = float(os.environ.get("MEDIUM_ZONE_MARGIN_USDT", "0.50"))
WEAK_ZONE_MARGIN_USDT = float(os.environ.get("WEAK_ZONE_MARGIN_USDT", "0.25"))
MAX_POSITION_MARGIN_USDT = float(os.environ.get("MAX_POSITION_MARGIN_USDT", "2.50"))
MAX_SCAN_SYMBOLS = int(os.environ.get("MAX_SCAN_SYMBOLS", "0"))
WATCHLIST_ONLY = os.environ.get("WATCHLIST_ONLY", "true").lower() == "true"
WATCHLIST_SYMBOLS = tuple(x.strip().upper() for x in os.environ.get(
    "WATCHLIST_SYMBOLS",
    "BTC-USDT,ETH-USDT,SOL-USDT,BNB-USDT,XRP-USDT,DOGE-USDT,TRX-USDT,HYPE-USDT,XMR-USDT,ZEC-USDT,LINK-USDT,ADA-USDT,XLM-USDT,BCH-USDT,UNI-USDT,LTC-USDT,AVAX-USDT,SUI-USDT,HBAR-USDT,TAO-USDT,ICP-USDT,ARB-USDT,POL-USDT,ETC-USDT",
).split(",") if x.strip())
KLINE_LIMIT_1H = int(os.environ.get("KLINE_LIMIT_1H", "1000"))
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
MAX_ENTRY_SLIPPAGE_PCT = max(0.0, float(os.environ.get("MAX_ENTRY_SLIPPAGE_PCT", "1.00")))
RECONCILIATION_MAX_SECONDS = float(os.environ.get("RECONCILIATION_MAX_SECONDS", "45"))
# Live execution requires the signal timestamp to be exactly the latest closed 1H bar.
# Also reject stale market data so a symbol with an old/delisted Binance series cannot
# masquerade as a fresh signal merely because its DataFrame index is zero-based.
MAX_DATA_STALENESS_HOURS = float(os.environ.get("MAX_DATA_STALENESS_HOURS", "2.0"))


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")



def _fmt_num(value: Any, digits: int = 8) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _fmt_level_human(level: dict[str, Any]) -> str:
    lid = str(level.get("level_id") or "?")[-8:]
    kind = str(level.get("kind") or "LEVEL").upper()
    sources = [str(x).upper() for x in (level.get("member_kinds") or [])]
    if not sources and level.get("source"):
        sources = [str(level.get("source")).upper()]
    lo = level.get("lower", level.get("price"))
    hi = level.get("upper", level.get("price"))
    lo_s, hi_s = _fmt_num(lo), _fmt_num(hi)
    if lo_s == "—":
        return f"{kind}#{lid}"
    price_text = lo_s if hi_s == "—" or hi_s == lo_s else f"{lo_s}–{hi_s}"
    meta = []
    if level.get("strength") is not None:
        meta.append(f"strength={int(level['strength'])}")
    if level.get("age_bars") is not None:
        meta.append(f"age={int(level['age_bars'])}")
    if sources:
        meta.append(f"sources={'+'.join(dict.fromkeys(sources))}")
    return f"{kind}#{lid} {price_text}" + (f" [{', '.join(meta)}]" if meta else "")


def _fmt_zone_price(zone: dict[str, Any], semantic: str | None = None) -> str:
    """Return exactly one price for a TradingView-comparable zone/level."""
    semantic_set = {str(semantic or "").upper()}
    semantic_set |= {str(x).upper() for x in (zone.get("member_kinds") or [])}
    semantic_set.discard("")

    if "DEMAND" in semantic_set:
        price = zone.get("btm", zone.get("lower", zone.get("price")))
    elif "SUPPLY" in semantic_set:
        price = zone.get("top", zone.get("upper", zone.get("price")))
    elif "PIVOT_LOW" in semantic_set:
        price = zone.get("price", zone.get("lower"))
    elif "PIVOT_HIGH" in semantic_set:
        price = zone.get("price", zone.get("upper"))
    else:
        price = zone.get("price", zone.get("lower", zone.get("upper")))

    try:
        x = float(price)
    except (TypeError, ValueError):
        return "INVALID_LEVEL"
    if not math.isfinite(x):
        return "INVALID_LEVEL"
    return f"{x:,.8f}".rstrip("0").rstrip(".")


def _zone_descriptor(level: dict[str, Any]) -> tuple[str, str]:
    kinds = [str(x).upper() for x in (level.get("member_kinds") or [])]
    if not kinds:
        kinds = [str(level.get("kind") or level.get("source") or "LEVEL").upper()]
    if "DEMAND" in kinds:
        return "🔵", "DEMAND"
    if "SUPPLY" in kinds:
        return "🔴", "SUPPLY"
    order = ["SUPPORT", "RESISTANCE", "PIVOT_LOW", "PIVOT_HIGH"]
    ordered = [x for x in order if x in kinds]
    label = "+".join(ordered or kinds)
    blue = {"SUPPORT", "PIVOT_LOW"}
    icon = "🔵" if all(x in blue for x in (ordered or kinds)) else "🔴"
    return icon, label


def _log_human_level_map(symbol: str, result: dict[str, Any]) -> None:
    levels = result.get("levels") or {}
    blue = levels.get("blue") or []
    red = levels.get("red") or []
    invalid = levels.get("invalid") or []
    zones = levels.get("active_zones") or {}
    demand = zones.get("demand") or []
    supply = zones.get("supply") or []

    # Human-readable comparison output: one exact price per displayed item.
    valid_demand = [z for z in demand if _fmt_zone_price(z, "DEMAND") != "INVALID_LEVEL"]
    valid_supply = [z for z in supply if _fmt_zone_price(z, "SUPPLY") != "INVALID_LEVEL"]

    if valid_demand:
        for idx, zone in enumerate(valid_demand, 1):
            log.info("[ZONES] %s | 🔵 DEMAND #%d | %s", symbol, idx, _fmt_zone_price(zone, "DEMAND"))
    else:
        log.info("[ZONES] %s | 🔵 DEMAND | —", symbol)

    if valid_supply:
        for idx, zone in enumerate(valid_supply, 1):
            log.info("[ZONES] %s | 🔴 SUPPLY #%d | %s", symbol, idx, _fmt_zone_price(zone, "SUPPLY"))
    else:
        log.info("[ZONES] %s | 🔴 SUPPLY | —", symbol)

    # Every other active structural level is printed in exactly the same
    # one-price format. Demand/Supply clusters are skipped because they were
    # already printed above.
    other = []
    for level in [*blue, *red]:
        kinds = {str(x).upper() for x in (level.get("member_kinds") or [])}
        if "DEMAND" in kinds or "SUPPLY" in kinds:
            continue
        if _fmt_zone_price(level) == "INVALID_LEVEL":
            continue
        other.append(level)
    other.sort(key=lambda x: float(x.get("price", x.get("lower", 0.0))))

    counters: dict[str, int] = {}
    for level in other:
        icon, label = _zone_descriptor(level)
        counters[label] = counters.get(label, 0) + 1
        log.info(
            "[ZONES] %s | %s %s #%d | %s",
            symbol, icon, label, counters[label], _fmt_zone_price(level),
        )

    high = levels.get("high_level") or {}
    low = levels.get("low_level") or {}
    high_price = _fmt_num(high.get("price"))
    low_price = _fmt_num(low.get("price"))
    if high_price != "—":
        log.info("[ZONES] %s | 🔴 HIGH LEVEL | %s", symbol, high_price)
    if low_price != "—":
        log.info("[ZONES] %s | 🔵 LOW LEVEL | %s", symbol, low_price)

    # Keep the existing diagnostics below; the [ZONES] lines above are the
    # human-facing TradingView comparison format.
    log.info(
        "[LEVEL_MAP] %s | EXTREMES | PINE_HIGH=%s | PINE_LOW=%s | INVALID=%d",
        symbol, high_price, low_price, len(invalid),
    )

    try:
        cp = float(result.get("current_price"))
    except (TypeError, ValueError):
        cp = None
    if cp is not None and math.isfinite(cp):
        below = sorted(
            [x for x in blue if float(x.get("upper", x.get("price", 0))) < cp],
            key=lambda x: float(x.get("upper", x.get("price", 0))), reverse=True,
        )
        above = sorted(
            [x for x in red if float(x.get("lower", x.get("price", 0))) > cp],
            key=lambda x: float(x.get("lower", x.get("price", 0))),
        )
        log.info("[LEVEL_MAP] %s | NEAREST_BLUE_BELOW | %s", symbol, _fmt_level_human(below[0]) if below else "—")
        log.info("[LEVEL_MAP] %s | NEXT_BLUE | %s", symbol, _fmt_level_human(below[1]) if len(below) > 1 else "—")
        log.info("[LEVEL_MAP] %s | NEAREST_RED_ABOVE | %s", symbol, _fmt_level_human(above[0]) if above else "—")
        log.info("[LEVEL_MAP] %s | NEXT_RED | %s", symbol, _fmt_level_human(above[1]) if len(above) > 1 else "—")

    signal = (result.get("signals") or [None])[0]
    if isinstance(signal, dict):
        direction = str(signal.get("type") or "?").upper()
        entry_level = signal.get("entry_level") or {}
        protection_level = signal.get("protection_level") or {}
        target_levels = ((signal.get("target") or {}).get("target_levels") or [])
        log.info("[TRADE_MAP] %s | DIRECTION=%s", symbol, direction)
        log.info("[TRADE_MAP] %s | ENTRY      | %s", symbol, _fmt_level_human(entry_level) if entry_level else "—")
        log.info("[TRADE_MAP] %s | PROTECTION | %s | SL=%s", symbol, _fmt_level_human(protection_level) if protection_level else "—", _fmt_num(signal.get("sl")))
        log.info("[TRADE_MAP] %s | TP1        | %s | price=%s", symbol, _fmt_level_human(target_levels[0]) if target_levels else "—", _fmt_num(signal.get("tp1")))
        log.info("[TRADE_MAP] %s | TP2        | %s | price=%s", symbol, _fmt_level_human(target_levels[1]) if len(target_levels) > 1 else "—", _fmt_num(signal.get("tp2")))
        log.info("[TRADE_MAP] %s | RISK       | risk=%s%% | TP2_R=%s", symbol, _fmt_num(signal.get("risk_pct")), _fmt_num(signal.get("tp2_rr"), 4))

def _load_failed_signal_ids() -> dict[str, dict[str, Any]]:
    try:
        if not FAILED_SIGNALS_PATH.exists():
            return {}
        raw = json.loads(FAILED_SIGNALS_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        now = int(time.time())
        cleaned = {}
        for event_id, record in raw.items():
            if not isinstance(record, dict):
                continue
            ts = int(record.get("ts", 0) or 0)
            if ts > 0 and now - ts <= FAILED_SIGNAL_TTL_SEC:
                cleaned[str(event_id)] = record
        if cleaned != raw:
            tmp = FAILED_SIGNALS_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, FAILED_SIGNALS_PATH)
        return cleaned
    except Exception as exc:
        log.warning("[FAILED_SIGNALS] load failed: %s", exc)
        return {}


def _failed_signal_is_blocked(record: dict[str, Any] | None, now: int | None = None) -> bool:
    """Return whether a failed signal should currently be suppressed.

    Failed execution is retriable, but bounded: after MAX retries the record is
    terminal and must not create an infinite execution loop.
    """
    if not isinstance(record, dict):
        return False
    now = int(time.time()) if now is None else int(now)
    if bool(record.get("terminal")):
        return True
    next_retry_at = int(record.get("next_retry_at", 0) or 0)
    return next_retry_at <= 0 or now < next_retry_at


def _mark_failed_signal(event_id: str, status: str, error: str = "") -> None:
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        raw = _load_failed_signal_ids()
        previous = raw.get(str(event_id), {}) if isinstance(raw.get(str(event_id)), dict) else {}
        retry_count = int(previous.get("retry_count", 0) or 0) + 1
        now = int(time.time())
        delay = min(FAILED_SIGNAL_RETRY_MAX_SEC, FAILED_SIGNAL_RETRY_BASE_SEC * (2 ** max(0, retry_count - 1)))
        terminal = retry_count >= FAILED_SIGNAL_MAX_RETRIES
        raw[str(event_id)] = {
            "ts": now,
            "status": str(status),
            "error": str(error)[:500],
            "retry_count": retry_count,
            "last_failed_at": now,
            "next_retry_at": 0 if terminal else now + delay,
            "terminal": terminal,
        }
        tmp = FAILED_SIGNALS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, FAILED_SIGNALS_PATH)
    except Exception as exc:
        log.warning("[FAILED_SIGNALS] mark failed for %s: %s", event_id, exc)


def _execution_outcome_category(status: str) -> str:
    """Classify execution outcomes without conflating them with strategy exits."""
    value = str(status or "").upper()
    if value == "OPENED_PROTECTED":
        return "PROTECTED_ENTRY"
    if "EMERGENCY" in value:
        return "EMERGENCY_EXIT"
    if "PROTECTION" in value or "SL_UNVERIFIED" in value:
        return "PROTECTION_FAILURE"
    if "UNVERIFIED" in value or "TIMEOUT" in value:
        return "EXECUTION_UNVERIFIED"
    if value in {"ENTRY_NOT_FILLED", "SKIPPED_MIN_QTY", "SKIPPED_TP_MIN_QTY", "SKIPPED_INVALID_SETUP", "BLOCKED_PROTECTION_PREFLIGHT"}:
        return "EXECUTION_BLOCKED"
    if value in {"ERROR", "OPENED", "DISABLED", "BLOCKED_MISSING_CREDENTIALS"} or value.endswith("_FAILED") or value.startswith("FAILED"):
        return "EXECUTION_FAILURE"
    return "EXECUTION_OTHER"


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

def _log_latest_trigger_check(symbol: str, df: pd.DataFrame, demand: list[dict], supply: list[dict], levels: dict[str, Any] | None) -> None:
    """Explain whether the latest closed 1H candle actually touched a trade zone.

    This is diagnostics only. It never changes the entry rules. In particular,
    a red S/R resistance touch is intentionally distinguished from a Pine
    Supply-zone touch because ZONE_ONLY entries are Demand/Supply based.
    """
    try:
        if df is None or len(df) < 2:
            return
        i = len(df) - 1
        o = float(df.loc[i, "open"])
        h = float(df.loc[i, "high"])
        l = float(df.loc[i, "low"])
        c = float(df.loc[i, "close"])
        prev_c = float(df.loc[i - 1, "close"])
        log.info(
            "[TRIGGER_CHECK] %s | CLOSED_1H O=%s H=%s L=%s C=%s | prevC=%s",
            symbol, _fmt_num(o), _fmt_num(h), _fmt_num(l), _fmt_num(c), _fmt_num(prev_c),
        )

        def zone_touch(z: dict[str, Any], direction: str) -> tuple[bool, bool]:
            lo = float(z.get("btm", z.get("lower")))
            hi = float(z.get("top", z.get("upper")))
            if direction == "LONG":
                literal = l <= hi and h >= lo
                fresh = prev_c > hi and l <= hi and c >= lo
            else:
                literal = h >= lo and l <= hi
                fresh = prev_c < lo and h >= lo and c <= hi
            return literal, fresh

        for idx, z in enumerate(demand, 1):
            literal, fresh = zone_touch(z, "LONG")
            price = _fmt_zone_price(z, "DEMAND")
            if literal or fresh:
                log.info(
                    "[TRIGGER_CHECK] %s | 🔵 DEMAND #%d | price=%s | touch=%s | fresh_touch=%s | directional=%s",
                    symbol, idx, price, "YES" if literal else "NO", "YES" if fresh else "NO",
                    "YES" if c > o else "NO",
                )

        for idx, z in enumerate(supply, 1):
            literal, fresh = zone_touch(z, "SHORT")
            price = _fmt_zone_price(z, "SUPPLY")
            if literal or fresh:
                log.info(
                    "[TRIGGER_CHECK] %s | 🔴 SUPPLY #%d | price=%s | touch=%s | fresh_touch=%s | directional=%s",
                    symbol, idx, price, "YES" if literal else "NO", "YES" if fresh else "NO",
                    "YES" if c < o else "NO",
                )

        # Structural S/R is useful to explain visually plausible touches such as
        # BTC 79,485 on TradingView. It is diagnostic only and is NOT itself an
        # entry trigger in the current ZONE_ONLY strategy.
        touched_structures: list[tuple[float, str, str]] = []
        snap = levels if isinstance(levels, dict) else {}
        for level in [*(snap.get("blue") or []), *(snap.get("red") or [])]:
            try:
                lo = float(level.get("lower", level.get("price")))
                hi = float(level.get("upper", level.get("price")))
                if l <= hi and h >= lo:
                    icon, label = _zone_descriptor(level)
                    touched_structures.append((lo, icon, label))
            except (TypeError, ValueError):
                continue
        touched_structures.sort(key=lambda x: abs(x[0] - c))
        for price, icon, label in touched_structures[:3]:
            log.info(
                "[TRIGGER_CHECK] %s | %s %s | candle_touch=YES | price=%s | ENTRY_TRIGGER=NO (%s is diagnostic only)",
                symbol, icon, label, _fmt_num(price), label,
            )

        any_fresh = False
        for z in demand:
            _, fresh = zone_touch(z, "LONG")
            any_fresh = any_fresh or fresh
        for z in supply:
            _, fresh = zone_touch(z, "SHORT")
            any_fresh = any_fresh or fresh

        if any_fresh:
            log.info("[TRIGGER_CHECK] %s | FRESH_DEMAND_SUPPLY_TOUCH=YES | continue_to_setup_validation", symbol)
        else:
            log.info(
                "[TRIGGER_CHECK] %s | FRESH_DEMAND_SUPPLY_TOUCH=NO | ACTION=WAIT | reason=latest_closed_1H_did_not_fresh-touch_active_demand_or_supply",
                symbol,
            )
    except Exception as exc:
        log.warning("[TRIGGER_CHECK_ERROR] %s | %s: %s", symbol, type(exc).__name__, exc)


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


def _fetch_analysis_bars(symbol: str, binance_symbol: str, provider: str) -> tuple[list[dict[str, Any]], str]:
    """Load fresh 1H analysis bars, falling back to BingX when Binance history is stale.

    The primary provider remains unchanged; fallback only activates for stale Binance
    history so a listed BingX asset is not rejected merely because Binance Spot lacks
    current candles for it.
    """
    source = "bingx" if provider == "bingx" else "binance_spot"
    bars = (fetch_bingx_klines(symbol, "1h", limit=KLINE_LIMIT_1H, retryable=False)
            if provider == "bingx"
            else fetch_binance_klines(binance_symbol, "1h", limit=KLINE_LIMIT_1H, retryable=False))
    min_bars = SWING_LEN * 2 + 10
    if len(bars) < min_bars:
        return bars, source
    latest_ts = pd.to_datetime(bars[-1]["timestamp"], unit="ms", utc=True)
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.tz_localize("UTC")
    age_h = max(0.0, (pd.Timestamp.now(tz="UTC") - latest_ts).total_seconds() / 3600.0)
    if provider == "binance" and age_h > MAX_DATA_STALENESS_HOURS:
        try:
            bx_bars = fetch_bingx_klines(symbol, "1h", limit=KLINE_LIMIT_1H, retryable=False)
            if len(bx_bars) >= min_bars:
                bx_latest = pd.to_datetime(bx_bars[-1]["timestamp"], unit="ms", utc=True)
                if bx_latest.tzinfo is None:
                    bx_latest = bx_latest.tz_localize("UTC")
                bx_age_h = max(0.0, (pd.Timestamp.now(tz="UTC") - bx_latest).total_seconds() / 3600.0)
                if bx_age_h <= MAX_DATA_STALENESS_HOURS:
                    return bx_bars, "bingx_fallback"
        except Exception as exc:
            log.warning("[DATA_FALLBACK_FAILED] %s | %s", symbol, exc)
    return bars, source


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


def _normalize_position_direction(position: dict[str, Any]) -> str | None:
    """Normalize BingX position-side semantics into LONG/SHORT.

    In HEDGE mode BingX returns positionSide=LONG/SHORT. In ONE_WAY mode it
    commonly returns positionSide=BOTH and the sign of positionAmt carries the
    direction. Reconciliation must understand both representations.
    """
    side = str(position.get("positionSide", "")).upper()
    if side in {"LONG", "SHORT"}:
        return side
    if side != "BOTH":
        return None
    try:
        raw_amt = float(position.get("positionAmt", 0) or 0)
    except (TypeError, ValueError):
        return None
    if raw_amt > 0:
        return "LONG"
    if raw_amt < 0:
        return "SHORT"
    return None


def _position_keys(positions: list[dict]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for p in positions:
        symbol = str(p.get("symbol", "")).upper()
        direction = _normalize_position_direction(p)
        try:
            qty = abs(float(p.get("positionAmt", 0) or 0))
        except (TypeError, ValueError):
            qty = 0.0
        if not symbol or direction is None or qty <= 0:
            continue
        out.add((symbol, direction))
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
        side = _normalize_position_direction(p)
        if side is None:
            continue
        qty = abs(float(p.get("positionAmt", 0) or 0))
        avg = float(p.get("avgPrice", 0) or p.get("entryPrice", 0) or 0)
        if qty <= 0 or avg <= 0:
            continue
        key = f"{symbol}:{side}"
        trade = active.get(key)
        stop_loss_pct = float((trade or {}).get("planned_risk_pct") or 1.0)
        setup = (trade or {}).get("setup", {}) if isinstance(trade, dict) else {}
        if trade is None:
            try:
                existing_protection = get_open_protection_directional(symbol, side)
            except Exception as exc:
                existing_protection = {"status": "error", "error": str(exc), "sl_orders": [], "tp_orders": []}
            structural_sl = None
            engine_sl = None
            if existing_protection.get("status") == "ok":
                for sl_order in existing_protection.get("sl_orders", []):
                    cid = str(sl_order.get("clientOrderId", "")).upper()
                    try:
                        candidate = float(sl_order.get("stopPrice", 0) or sl_order.get("price", 0) or 0)
                    except (TypeError, ValueError):
                        candidate = 0.0
                    if cid.startswith("EVT_") and candidate > 0 and ((side == "LONG" and candidate < avg) or (side == "SHORT" and candidate > avg)):
                        structural_sl = candidate
                        engine_sl = sl_order
                        break
            recovered_event_id = f"RECON_{symbol.replace('-', '')}_{side}"
            setup = {
                "strategy": "Zone/Structure First",
                "event_time": pd.Timestamp.now(tz="UTC").isoformat(),
                "entry_reference": avg,
                "signal_price": avg,
                "invalidation_price": structural_sl,
                "risk_pct": stop_loss_pct,
                "tp_levels": [],
                "recovery": True,
            }
            recover_exchange_position(
                recovered_event_id, symbol, side, avg, qty,
                sl_order={
                    "status": "recovered_existing",
                    "order_id": str((engine_sl or {}).get("orderId", "")),
                    "client_order_id": str((engine_sl or {}).get("clientOrderId", "")),
                    "stop_price": structural_sl,
                    "qty": abs(float((engine_sl or {}).get("origQty", qty) or qty)),
                } if engine_sl else {},
                tp_orders=list(existing_protection.get("tp_orders", [])) if existing_protection.get("status") == "ok" else [],
            )
            active = _load_active_trades_file()
            trade = active.get(key)
            log.warning("[RECON_RECOVERED] %s %s | exchange position restored to local aggregate state qty=%.12g avg=%.12g", symbol, side, qty, avg)

        setup = (trade or {}).get("setup", {}) if isinstance(trade, dict) else setup
        stop_loss_pct = float((trade or {}).get("planned_risk_pct") or stop_loss_pct or 1.0)
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
            stored_sl = (setup.get("invalidation_price") if isinstance(setup, dict) else None)
            result = ensure_directional_protection(
                symbol, side, avg, qty, stop_loss_pct, tp_levels,
                trade_id=(trade or {}).get("event_id") or key,
                requested_sl_price=stored_sl,
            )
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
        "levels": signal.get("levels", {}),
        "entry_level": signal.get("entry_level"),
        "protection_level": signal.get("protection_level"),
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
    """Recalculate structural SL/TP from the *actual* market fill.

    A market order can fill materially away from the signal/reference candle close.
    Never submit stale absolute targets derived from the pre-fill reference price.
    """
    direction = str(signal["type"]).upper()
    entry = float(avg_price)
    target = signal.get("target") if isinstance(signal.get("target"), dict) else {}
    levels = signal.get("levels") if isinstance(signal.get("levels"), dict) else {}
    atr = float(signal.get("atr", 0.0) or 0.0)
    if entry <= 0:
        raise ValueError("actual fill price must be positive")

    blue = list(levels.get("blue", [])) if isinstance(levels.get("blue"), list) else []
    red = list(levels.get("red", [])) if isinstance(levels.get("red"), list) else []
    protective_level = select_protective_level(direction, entry, blue, red)
    if protective_level is None:
        raise ValueError("no valid same-color protective level at actual fill")
    sl_buffer = max(atr * float(os.environ.get("ZONE_SL_ATR_BUFFER", "0.10")), entry * 0.0002)
    sl = protective_price(protective_level, direction, sl_buffer)
    risk = abs(entry - sl)
    if risk <= 0:
        raise ValueError("post-fill risk is non-positive")

    opposing = select_opposing_levels(direction, entry, blue, red, limit=2)
    obstacle = None
    if opposing:
        first = opposing[0]
        obstacle = float(first["lower"]) if direction == "LONG" else float(first["upper"])

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
        "protection_level": protective_level,
        "target_levels": opposing,
    }


def _cancel_engine_protection_before_emergency_close(symbol: str, direction: str) -> dict[str, Any]:
    """Best-effort cleanup of this engine's SL/TP orders before a MARKET rollback."""
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
            if isinstance(resp, dict) and resp.get("code") in (0, "0"):
                result["cancelled"].append(oid)
            else:
                result["errors"].append(f"{oid}: {resp}")
        except Exception as exc:
            result["errors"].append(f"{oid}: {exc}")
    if result["errors"]:
        result["status"] = "partial" if result["cancelled"] else "error"
    return result


def _emergency_close_and_verify(symbol: str, direction: str, qty: float, trade_id: str) -> dict[str, Any]:
    """Close a safety-rollback position and verify that it is actually gone."""
    attempts = []
    configured_attempts = max(2, int(os.environ.get("EMERGENCY_CLOSE_ATTEMPTS", "4")))
    verify_polls = max(2, int(os.environ.get("EMERGENCY_CLOSE_VERIFY_POLLS", "6")))
    last_qty = max(0.0, float(qty or 0.0))

    for attempt in range(configured_attempts):
        try:
            current = get_position_directional(symbol, direction)
        except Exception as exc:
            current = {"status": "error", "error": str(exc)}
        if current.get("status") == "not_found":
            return {"status": "closed_verified", "attempts": attempts, "verification": current}
        if current.get("status") == "error":
            attempts.append({"status": "position_check_error", "error": current.get("error")})
            time.sleep(0.4 * (attempt + 1))
            continue
        if current.get("status") == "found":
            try:
                last_qty = abs(float(current.get("positionAmt", last_qty) or last_qty))
            except (TypeError, ValueError):
                pass
        if last_qty <= 0:
            return {"status": "closed_verified", "attempts": attempts, "verification": current}
        try:
            close_result = close_position_market(symbol, direction, last_qty, trade_id=trade_id)
        except Exception as exc:
            close_result = {"status": "error", "error": str(exc)}
        attempts.append(close_result)
        time.sleep(0.35 * (attempt + 1))

    verification = {"status": "verification_error", "error": "no verification attempted"}
    for poll in range(verify_polls):
        try:
            verification = get_position_directional(symbol, direction)
        except Exception as exc:
            verification = {"status": "verification_error", "error": str(exc)}
        if verification.get("status") == "not_found":
            return {"status": "closed_verified", "attempts": attempts, "verification": verification, "verify_poll": poll + 1}
        if verification.get("status") == "found":
            try:
                last_qty = abs(float(verification.get("positionAmt", last_qty) or last_qty))
            except (TypeError, ValueError):
                pass
        time.sleep(0.35)

    return {"status": "close_unverified", "attempts": attempts, "verification": verification, "remaining_qty": last_qty}

def _zone_strength_tier(signal: dict[str, Any]) -> tuple[str, float]:
    zone = signal.get("zone") if isinstance(signal.get("zone"), dict) else {}
    kinds = {str(x).upper() for x in (zone.get("member_kinds") or [])}
    if not kinds:
        kinds = {str(zone.get("kind") or "").upper()}
    if "DEMAND" in kinds or "SUPPLY" in kinds:
        return "STRONG", STRONG_ZONE_MARGIN_USDT
    if "SUPPORT" in kinds or "RESISTANCE" in kinds:
        return "MEDIUM", MEDIUM_ZONE_MARGIN_USDT
    return "WEAK", WEAK_ZONE_MARGIN_USDT


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
    zone_tier, entry_margin = _zone_strength_tier(signal)
    setup["entry_margin_usdt"] = entry_margin
    setup["zone_strength_tier"] = zone_tier
    setup["max_position_margin_usdt"] = MAX_POSITION_MARGIN_USDT
    log.info(
        "[POSITION_SIZING] %s %s | zone=%s | tier=%s | entry_margin=%.2f | max_position_margin=%.2f",
        symbol, direction, (signal.get("zone") or {}).get("kind"), zone_tier, entry_margin, MAX_POSITION_MARGIN_USDT,
    )
    order = open_market(
        symbol, direction, entry_price, event_id,
        margin_usdt=entry_margin,
        max_position_margin_usdt=MAX_POSITION_MARGIN_USDT,
    )
    if order.get("status") in {"skipped_min_qty", "skipped_tp_min_qty", "skipped_position_margin_cap"}:
        return order
    if order.get("status") != "opened":
        return {"status": str(order.get("status", "error")).upper(), "error": order.get("error"), "order": order}

    position = wait_for_position_fill_directional(
        symbol,
        direction,
        timeout_sec=int(os.environ.get("POSITION_FILL_TIMEOUT_SEC", "30")),
        poll_interval=0.5,
    )
    if position.get("status") != "found":
        # A MARKET response plus a fill-poll timeout is an ambiguous exchange
        # state, not proof that the position does not exist. Never leave a
        # potentially-open position unmanaged and never blindly repost the entry.
        try:
            reconciled_position = get_position_directional(symbol, direction)
        except Exception as exc:
            reconciled_position = {"status": "error", "error": str(exc)}

        if reconciled_position.get("status") == "found":
            log.warning(
                "[EXEC_ENTRY_RECONCILED] %s %s | fill polling did not confirm entry, "
                "but authoritative position reconciliation found the position; continuing to protection.",
                symbol, direction,
            )
            position = reconciled_position
        elif reconciled_position.get("status") == "not_found":
            log.warning(
                "[EXEC_ENTRY_NOT_FILLED] %s %s | market order acknowledged but no position exists after authoritative reconciliation.",
                symbol, direction,
            )
            return {
                "status": "entry_not_filled",
                "order": order,
                "position": reconciled_position,
                "fill_poll": position,
                "error": "market order acknowledged but authoritative position reconciliation found no open position",
            }
        else:
            error_detail = reconciled_position.get("error") or position.get("error") or position.get("last_poll_error")
            log.critical(
                "[EXEC_ENTRY_UNVERIFIED] %s %s | position state remains ambiguous after fill timeout: %s",
                symbol, direction, error_detail,
            )
            return {
                "status": "entry_state_unverified",
                "order": order,
                "position": reconciled_position,
                "fill_poll": position,
                "error": f"entry state could not be authoritatively reconciled: {error_detail}",
            }

    avg_price = float(position["avgPrice"])
    qty = abs(float(position["positionAmt"]))
    pre_qty = abs(float(order.get("pre_position_qty", 0.0) or 0.0))
    added_qty = qty - pre_qty
    if pre_qty > 0 and added_qty <= max(1e-12, pre_qty * 1e-9):
        # A confirmed existing position without an observable increase is not a
        # proof that this signal's MARKET order filled. Never register it again.
        return {
            "status": "ENTRY_STATE_UNVERIFIED",
            "error": f"position did not increase after entry order: pre={pre_qty:.12g} post={qty:.12g}",
            "order": order,
            "position": position,
        }
    added_qty = qty if pre_qty <= 0 else added_qty

    # The requested margin cap is checked before POST, but market fills can move
    # against the requested sizing price. Re-apply the hard cap to the authoritative
    # post-fill position before installing protection or persisting the trade.
    max_position_margin = order.get("max_position_margin_usdt")
    if max_position_margin is not None:
        try:
            max_position_margin = float(max_position_margin)
        except (TypeError, ValueError):
            max_position_margin = MAX_POSITION_MARGIN_USDT
        try:
            mult = float(order.get("contract_multiplier", 1.0) or 1.0)
            leverage_used = float(order.get("leverage", LEVERAGE) or LEVERAGE)
            actual_total_margin = (qty * avg_price * mult) / max(leverage_used, 1.0)
        except (TypeError, ValueError, ZeroDivisionError):
            actual_total_margin = float("inf")
    else:
        actual_total_margin = 0.0
    if max_position_margin is not None and actual_total_margin > max_position_margin + 1e-9:
        reason = (
            f"actual_position_margin={actual_total_margin:.8f} > max={max_position_margin:.8f}"
        )
        log.critical("[SAFETY_CLOSE] %s %s | post-fill margin cap exceeded | %s", symbol, direction, reason)
        cleanup = _cancel_engine_protection_before_emergency_close(symbol, direction)
        rollback_qty = added_qty if added_qty > 0 else qty
        close_result = _emergency_close_and_verify(symbol, direction, rollback_qty, event_id)
        return {
            "status": "opened_then_margin_cap_rollback",
            "error": reason,
            "actual_position_margin_usdt": actual_total_margin,
            "max_position_margin_usdt": max_position_margin,
            "order": order,
            "position": position,
            "close": close_result,
            "protection_cleanup": cleanup,
            "executed_signal": dict(signal),
        }

    # Abort on materially adverse market-entry slippage. Once a market order is
    # filled, accepting a severely worse price can invalidate the signal geometry
    # before protection is even submitted. Roll back safely instead of widening risk.
    try:
        adverse_slippage_pct = (
            max(0.0, avg_price - entry_price) / entry_price * 100.0
            if direction == "LONG"
            else max(0.0, entry_price - avg_price) / entry_price * 100.0
        )
    except (TypeError, ValueError, ZeroDivisionError):
        adverse_slippage_pct = float("inf")
    if adverse_slippage_pct > MAX_ENTRY_SLIPPAGE_PCT:
        reason = f"adverse_entry_slippage={adverse_slippage_pct:.4f}% > {MAX_ENTRY_SLIPPAGE_PCT:.4f}%"
        log.critical("[SAFETY_CLOSE] %s %s | %s", symbol, direction, reason)
        cleanup = _cancel_engine_protection_before_emergency_close(symbol, direction)
        close_result = _emergency_close_and_verify(symbol, direction, qty, event_id)
        return {"status": "opened_then_emergency_closed", "error": reason, "order": order, "position": position, "close": close_result, "protection_cleanup": cleanup, "executed_signal": dict(signal)}

    # Recalculate ALL absolute protection levels from the real market fill.
    # Never submit targets computed from the stale signal/reference close.
    try:
        rebased = _rebase_protection_after_fill(signal, avg_price)
    except Exception as exc:
        reason = f"post-fill protection rebase failed: {exc}"
        log.critical("[SAFETY_CLOSE] %s %s | %s", symbol, direction, reason)
        cleanup = _cancel_engine_protection_before_emergency_close(symbol, direction)
        close_result = _emergency_close_and_verify(symbol, direction, qty, event_id)
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
            "target_levels": rebased.get("target_levels", []),
        },
        "protection_level": rebased.get("protection_level"),
        "level_selection_after_fill": {
            "protection_level_id": (rebased.get("protection_level") or {}).get("level_id"),
            "target_level_ids": [x.get("level_id") for x in rebased.get("target_levels", [])],
        },
    })
    valid, reason = _validate_trade_geometry(actual_signal)
    if not valid:
        log.critical("[SAFETY_CLOSE] %s %s | invalid post-fill protection geometry | %s", symbol, direction, reason)
        cleanup = _cancel_engine_protection_before_emergency_close(symbol, direction)
        close_result = _emergency_close_and_verify(symbol, direction, qty, event_id)
        return {"status": "opened_then_emergency_closed", "error": reason, "order": order, "position": position, "close": close_result, "protection_cleanup": cleanup, "executed_signal": actual_signal}

    setup["entry_reference"] = avg_price
    setup["invalidation_price"] = sl_price
    setup["risk_pct"] = actual_risk_pct
    setup["protection_level"] = rebased.get("protection_level")
    setup["target_levels"] = rebased.get("target_levels", [])
    setup["level_selection_after_fill"] = actual_signal.get("level_selection_after_fill")
    setup["target_rr"] = actual_signal["tp2_rr"]
    setup["planned_weighted_rr"] = actual_signal["tp1_rr"] * 0.50 + actual_signal["tp2_rr"] * 0.50
    setup["tp_levels"] = [
        {"leg": "tp1", "pnl_pct": tp1_pnl_pct, "close_fraction": 0.50, "price": tp1_price},
        {"leg": "tp2", "pnl_pct": tp2_pnl_pct, "close_fraction": 0.50, "price": tp2_price},
    ]
    setup["target_price"] = tp2_price

    log.info(
        "[EXEC_POST_FILL_REBASED] %s %s | fill=%s sl=%s tp1=%s tp2=%s tp1_rr=%.3f tp2_rr=%.3f target_source=%s "
        "protective_level=%s entry_level=%s target_levels=%s",
        symbol, direction, avg_price, sl_price, tp1_price, tp2_price,
        actual_signal["tp1_rr"], actual_signal["tp2_rr"], rebased["target_source"],
        (rebased.get("protection_level") or {}).get("level_id"),
        (actual_signal.get("entry_level") or {}).get("level_id"),
        [x.get("level_id") for x in rebased.get("target_levels", [])],
    )

    aggregate_trade_id = get_active_trade_id(symbol, direction) or event_id
    protection = ensure_directional_protection(
        symbol, direction, avg_price, qty,
        actual_risk_pct, setup["tp_levels"], trade_id=aggregate_trade_id,
        requested_sl_price=sl_price,
    )
    if protection.get("status") != "PROTECTED":
        log.critical("[SAFETY_CLOSE] %s %s | mandatory protection incomplete | %s", symbol, direction, protection)
        # Mandatory rule: never leave a newly-opened position live without BOTH
        # a verified SL and both TP legs. Attempt an immediate market rollback.
        cleanup = _cancel_engine_protection_before_emergency_close(symbol, direction)
        close_result = _emergency_close_and_verify(symbol, direction, qty, event_id)
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
        entry_margin_usdt=entry_margin,
        zone_id=str(signal.get("zone_id") or (signal.get("zone") or {}).get("level_id") or ""),
        entry_leg_qty=added_qty,
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
    failed_ids = _load_failed_signal_ids()
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
            bars, source_name = _fetch_analysis_bars(symbol, binance_symbol, provider)
            min_bars = SWING_LEN * 2 + 10
            if len(bars) < min_bars:
                return {
                    "symbol": symbol, "current_price": None, "price_position": "INSUFFICIENT_DATA",
                    "fresh_signal": "—", "active_demand": 0, "active_supply": 0,
                    "zones": {"demand": [], "supply": []}, "last_signal_count": 0,
                    "error": f"insufficient_1h_candles:{len(bars)}<{min_bars}", "signals": [],
                }

            df, supply, demand, signals = generate_zone_signals(pd.DataFrame(bars), symbol=symbol, mode=DIAGNOSTICS_MODE)
            level_snapshot = df.attrs.get("level_snapshot") if isinstance(df.attrs.get("level_snapshot"), dict) else {}
            _log_latest_trigger_check(symbol, df, demand, supply, level_snapshot)
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
                    "binance_price": latest_price if source_name == "binance_spot" else None,
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
                    "levels": level_snapshot,
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
            # For executable spread validation, compare two current venue prices.
            # The closed 1H candle remains the strategy reference; it must not be
            # used as the Binance side of a live cross-venue spread check.
            binance_live_price = None
            if recent and provider == "binance":
                try:
                    ticker = fetch_24h_ticker(binance_symbol)
                    if isinstance(ticker, dict):
                        raw_last = ticker.get("lastPrice") or ticker.get("last") or ticker.get("price")
                        if raw_last is not None:
                            candidate = float(raw_last)
                            if candidate > 0:
                                binance_live_price = candidate
                except Exception as ticker_exc:
                    log.warning("[MARKET_CHECK] %s | Binance live ticker validation failed: %s", symbol, ticker_exc)
            spread_pct = _market_spread_pct(
                binance_live_price if binance_live_price is not None else latest_price,
                bingx_price,
            ) if provider == "binance" and source_name == "binance_spot" else None
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
                "levels": level_snapshot,
                "last_signal_count": len(recent),
                "latest_closed_idx": int(latest_closed_idx),
                "latest_closed_time": latest_closed_time.isoformat(),
                "latest_ohlc": {
                    "open": float(df.loc[latest_closed_idx, "open"]),
                    "high": float(df.loc[latest_closed_idx, "high"]),
                    "low": float(df.loc[latest_closed_idx, "low"]),
                    "close": float(df.loc[latest_closed_idx, "close"]),
                    "prev_close": float(df.loc[latest_closed_idx - 1, "close"]) if latest_closed_idx > 0 else float(df.loc[latest_closed_idx, "close"]),
                },
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
                        "market_source": "unknown", "binance_symbol": analysis_meta.get(symbol, {}).get("binance_symbol"),
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

            # Visual separator: every coin gets its own clearly delimited log block.
            log.info("[COIN_START] %s | ==============================", symbol)

            if result.get("price_position") == "ERROR":
                log.error("[COIN_ERROR] %s | %s", symbol, result.get("error", "unknown error"))
            elif result.get("price_position") == "INSUFFICIENT_DATA":
                log.warning("[COIN_SKIP] %s | %s", symbol, result.get("error", "insufficient data"))
            elif result.get("price_position") == "CONTRACT_NOT_FOUND":
                log.warning("[COIN_SKIP] %s | contract not found in BingX cache", symbol)
            else:
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
                _log_human_level_map(symbol, result)

            log.info("[COIN_END] %s | ================================", symbol)

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

    if private_ready:
        try:
            exit_result = process_structural_exits(scan_rows)
            log.info("[STRUCTURE_EXIT_SUMMARY] %s", exit_result)
        except Exception as exc:
            log.exception("[STRUCTURE_EXIT] processing failed: %s", exc)

    # Execution safety: keep one candidate per (symbol, side, structural cluster),
    # not one candidate per symbol. Independent structures may legitimately add
    # to the same directional position on one closed candle.
    latest_by_zone: dict[tuple[str, str, str], dict[str, Any]] = {}
    for signal in fresh_signals:
        symbol_key = str(signal["symbol"]).upper()
        zone_id = str(signal.get("zone_id") or (signal.get("zone") or {}).get("level_id") or "")
        key = (symbol_key, str(signal["type"]).upper(), zone_id)
        previous = latest_by_zone.get(key)
        candidate_key = (int(signal["idx"]), float(signal.get("score", 0.0)))
        previous_key = (int(previous["idx"]), float(previous.get("score", 0.0))) if previous else None
        if previous is None or candidate_key > previous_key:
            latest_by_zone[key] = signal

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
    for signal in latest_by_zone.values():
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
        failed_record = failed_ids.get(signal["event_id"])
        if signal["event_id"] in successful_ids or _failed_signal_is_blocked(failed_record):
            continue
        # Same-direction additions are allowed; open_market enforces the total
        # directional margin cap. Opposite-side positions remain blocked by the
        # strategy to avoid accidental reversal/netting even in HEDGE mode.
        if opposite in open_keys:
            log.info("[REJECT] %s | %s | reason=opposite_position_exists", signal["symbol"], signal["type"])
            continue
        executable.append(signal)

    # Deterministic same-candle order: nearest structural price first, then
    # stronger cluster. This is a defined rule; never rely on dict/list order.
    executable.sort(key=lambda x: (
        -int(x["idx"]),
        0 if str(x.get("type", "")).upper() == "LONG" else 1,
        -float((x.get("zone") or {}).get("price", x.get("entry", 0.0))),
        -float(x.get("zone_strength", x.get("score", 0.0))),
        str(x.get("zone_id") or ""),
    ))

    executed = 0
    for signal in executable[:MAX_TRADES_PER_CYCLE]:
        if not EXECUTION_ENABLED:
            _send_signal(signal, {"status": "DISABLED"})
            continue
        log.info(
            "[EXEC_SIGNAL] symbol=%s direction=%s signal_idx=%s signal_time=%s age_bars=%s "
            "zone=%s zone_low=%s zone_high=%s entry_level=%s protection_level=%s "
            "target_levels=%s target_source=%s obstacle=%s tp1=%s tp2=%s event_id=%s",
            signal.get("symbol"), signal.get("type"), signal.get("idx"), signal.get("time"),
            signal.get("execution_age_bars", 0),
            (signal.get("zone") or {}).get("kind"),
            (signal.get("zone") or {}).get("btm"),
            (signal.get("zone") or {}).get("top"),
            (signal.get("entry_level") or {}).get("level_id"),
            (signal.get("protection_level") or {}).get("level_id"),
            [x.get("level_id") for x in (signal.get("target", {}).get("target_levels") or [])],
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
        # Cross-process serialization covers the check -> MARKET POST critical
        # section. GitHub Actions also serializes this workflow, but the engine
        # must remain safe when invoked by another scheduler/process.
        DATA.mkdir(parents=True, exist_ok=True)
        lockf = EXECUTION_LOCK_PATH.open("a+", encoding="utf-8")
        try:
            if fcntl is not None:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            refreshed_successful = _load_successful_trade_ids()
            if signal["event_id"] in refreshed_successful:
                execution = {"status": "ALREADY_EXECUTED", "event_id": signal["event_id"]}
            else:
                current_positions = get_positions(timeout_sec=float(os.environ.get("RECONCILIATION_HTTP_TIMEOUT_SEC", "5")), retryable=False)
                current_keys = _position_keys(current_positions)
                bx_key = (str(get_contract(signal["symbol"])["symbol"]).upper(), str(signal["type"]).upper()) if get_contract(signal["symbol"]) else (str(signal["symbol"]).upper(), str(signal["type"]).upper())
                opp_key = (bx_key[0], "SHORT" if bx_key[1] == "LONG" else "LONG")
                if opp_key in current_keys:
                    execution = {"status": "OPPOSITE_POSITION_EXISTS", "symbol": signal["symbol"], "direction": signal["type"]}
                else:
                    execution = execute_new_position(signal)
        except Exception as exc:
            log.exception("[EXEC_TECHNICAL_ERROR] %s %s | %s", signal["symbol"], signal["type"], exc)
            execution = {"status": "EXECUTION_EXCEPTION", "error": str(exc)}
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            lockf.close()
        execution_status = str(execution.get("status", ""))
        if execution_status in {"skipped_min_qty", "skipped_tp_min_qty", "skipped_invalid_setup"}:
            log.warning(
                "[EXEC_SKIPPED] %s %s | status=%s | reason=%s | error=%s | qty=%s min_qty=%s required_margin=%.4f configured_margin=%.4f",
                signal["symbol"], signal["type"], execution_status, execution.get("reason", "invalid_setup"), execution.get("error"), execution.get("qty"), execution.get("min_qty"),
                float(execution.get("required_margin_usdt", 0.0) or 0.0), float(execution.get("configured_margin_usdt", MARGIN_USDT) or MARGIN_USDT),
            )
        elif execution_status != "opened_protected":
            log.error("[EXEC_FAILED] %s %s | status=%s | error=%s | order=%s", signal["symbol"], signal["type"], execution_status, execution.get("error"), execution.get("order"))
            if execution_status not in {"DISABLED", "BLOCKED_MISSING_CREDENTIALS"}:
                _mark_failed_signal(signal["event_id"], execution_status, execution.get("error", ""))
        _append_jsonl(TRADES_PATH, {
            "record_type": "TRADE_OPEN",
            "event_id": signal["event_id"],
            "symbol": signal["symbol"],
            "direction": signal["type"],
            "score": signal["score"],
            "signal": signal,
            "outcome_category": _execution_outcome_category(execution_status),
            "result": execution,
        })
        _send_signal(signal, execution)
        if str(execution.get("status")) == "opened_protected":
            executed += 1

    # Persist the complete scan for analytics/backtesting, but do not dump the
    # full inactive-symbol table into runtime logs. Runtime logs contain active
    # symbols only.
    save_scan(scan_rows, fresh_signals, duration_sec=time.time() - started, scan_id=scan_id)
    # Final tracking checkpoint: orders/positions can change during the scan or
    # execution phase. Reconcile once more before this process exits so a TP,
    # close, or BE transition that happened during this run is not deferred to
    # the next invocation. Long-lived monitoring is provided by the scheduled
    # tracker workflow; this checkpoint is deliberately only one pass.
    if private_ready:
        try:
            update_active_trades()
        except Exception as exc:
            log.exception("[TRACKER_FINAL] active trade update failed: %s", exc)
        try:
            reconcile_all_open_positions()
        except Exception as exc:
            log.exception("[RECON_FINAL] reconciliation failed: %s", exc)

    active_log_rows = [
        r for r in scan_rows
        if r.get("fresh_signal") not in {None, "—"}
        or r.get("price_position") in {"🟢 В зоне DEMAND", "🔴 В зоне SUPPLY"}
    ]
    log.info("[ACTIVE_SUMMARY] active_symbols=%d signals=%d", len(active_log_rows), len(fresh_signals))
    log.info("[DONE] symbols=%d fresh_signals=%d executed=%d duration=%.1fs", len(scan_rows), len(fresh_signals), executed, time.time() - started)


if __name__ == "__main__":
    main()
