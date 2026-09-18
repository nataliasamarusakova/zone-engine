from __future__ import annotations

import json
import math
import logging
import os
import time
import uuid
import fcntl
import hashlib
import re
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
    prepare_protection_capacity,
    get_execution_quote,
    fetch_research_market_context,
    fetch_research_account_snapshot,
    cancel_order,
    close_position_market,
    open_market,
    wait_for_position_fill_directional,
)
from event_engine.signals import STRATEGY_VERSION, SWING_LEN, TP1_PCT, TP2_PCT, generate_zone_signals, score_zone_signal, _nearest_opposing_level, _signal_forensics
from event_engine.fundamental_assets import FUNDAMENTAL_ASSET_SYMBOLS
from event_engine.telegram import format_signal, send as send_tg
from event_engine.tracker import register_active_trade, update_active_trades, update_active_trade_protection
from event_engine import research

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("zone_engine")

_PROJECT_SYMBOL_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z0-9]+)-USDT\b", re.IGNORECASE)
_DIAGNOSTIC_FILE_HANDLER_NAME = "zone_engine_diagnostic_file"

class _CompactSymbolFilter(logging.Filter):
    """Keep human-readable logs compact by rendering BTC-USDT as BTCUSDT.

    Machine-readable JSON/state retains canonical BingX symbols with '-' because
    those values are used as stable keys across modules. Only log rendering changes.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
            compact = _PROJECT_SYMBOL_RE.sub(lambda m: f"{m.group(1).upper()}USDT", message)
            record.msg = compact
            record.args = ()
        except Exception:
            pass
        return True

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data"
TRADES_PATH = DATA / "trades.jsonl"
FAILED_SIGNALS_PATH = DATA / "failed_signals.json"
FAILED_SIGNAL_TTL_SEC = int(os.environ.get("FAILED_SIGNAL_TTL_SEC", str(24 * 3600)))
FAILED_SIGNAL_MAX_RETRIES = max(1, int(os.environ.get("FAILED_SIGNAL_MAX_RETRIES", "3")))
FAILED_SIGNAL_RETRY_BASE_SEC = max(1, int(os.environ.get("FAILED_SIGNAL_RETRY_BASE_SEC", "300")))
FAILED_SIGNAL_RETRY_MAX_SEC = max(FAILED_SIGNAL_RETRY_BASE_SEC, int(os.environ.get("FAILED_SIGNAL_RETRY_MAX_SEC", str(3600))))
ACTIONS_PATH = DATA / "actions.jsonl"
ENTRY_DECISIONS_PATH = DATA / "entry_decisions.jsonl"
EVENT_CLAIMS_PATH = DATA / "event_execution_claims.json"
EVENT_CLAIMS_LOCK_PATH = DATA / "event_execution_claims.json.lock"
EVENT_CLAIM_LEASE_SEC = max(60, int(os.environ.get("EVENT_CLAIM_LEASE_SEC", "900")))

EXECUTION_ENABLED = os.environ.get("EXECUTION_ENABLED", "true").lower() == "true"
MARGIN_USDT = float(os.environ.get("BINGX_MARGIN_USDT", "1"))
MAX_TRADES_PER_CYCLE = int(os.environ.get("MAX_TRADES_PER_CYCLE", "5"))
MAX_SCAN_SYMBOLS = int(os.environ.get("MAX_SCAN_SYMBOLS", "0"))
WATCHLIST_ONLY = os.environ.get("WATCHLIST_ONLY", "false").lower() == "true"
# Temporary test switches. Normal mode keeps the curated 150-asset whitelist and
# midpoint trigger. The test workflow can disable either without changing the
# underlying 1H Demand/Supply zone-construction algorithm.
FUNDAMENTAL_WHITELIST_ENABLED = os.environ.get("FUNDAMENTAL_WHITELIST_ENABLED", "true").lower() == "true"
ZONE_TRIGGER_MODE = os.environ.get("ZONE_TRIGGER_MODE", "midpoint").strip().lower()
if ZONE_TRIGGER_MODE not in {"midpoint", "zone"}:
    raise ValueError("ZONE_TRIGGER_MODE must be midpoint or zone")
WATCHLIST_SYMBOLS = tuple(x.strip().upper() for x in os.environ.get(
    "WATCHLIST_SYMBOLS",
    "BTC-USDT,ETH-USDT,SOL-USDT,BNB-USDT,TAO-USDT,LTC-USDT,BCH-USDT,AVAX-USDT,LINK-USDT,ETC-USDT,ADA-USDT,UNI-USDT,XRP-USDT,ICP-USDT,HYPE-USDT,DOGE-USDT,HBAR-USDT,ARB-USDT,POL-USDT,SUI-USDT",
).split(",") if x.strip())
KLINE_LIMIT_1H = int(os.environ.get("KLINE_LIMIT_1H", "120"))
MAX_SIGNAL_AGE_BARS = int(os.environ.get("MAX_SIGNAL_AGE_BARS", "0"))
FIXED_STOP_PCT = float(os.environ.get("FIXED_STOP_PCT", "10.00"))
MAX_PRODUCTION_RISK_PCT = min(float(os.environ.get("MAX_SIGNAL_RISK_PCT", str(FIXED_STOP_PCT))), FIXED_STOP_PCT)
MIN_STRUCTURE_ROOM_R = float(os.environ.get("MIN_STRUCTURE_ROOM_R", "1.20"))
REQUIRE_STRUCTURE_OBSTACLE = os.environ.get("REQUIRE_STRUCTURE_OBSTACLE", "false").lower() == "true"
REQUIRE_DIRECTIONAL_CANDLE = os.environ.get("REQUIRE_DIRECTIONAL_CANDLE", "false").lower() == "true"
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
# Zone geometry remains 1H; trigger detection runs on closed 5m bars so a short
# midpoint visit cannot be missed merely because the 1H candle later closes elsewhere.
KLINE_LIMIT_5M = int(os.environ.get("KLINE_LIMIT_5M", "144"))  # 12h of 5m bars
MAX_5M_TRIGGER_AGE_MINUTES = float(os.environ.get("MAX_5M_TRIGGER_AGE_MINUTES", "15"))
INITIAL_5M_TRIGGER_LOOKBACK_MINUTES = float(os.environ.get("INITIAL_5M_TRIGGER_LOOKBACK_MINUTES", "15"))
ZONE_VISIT_STATE_PATH = DATA / "zone_visit_state.json"
ZONE_VISIT_STATE_LOCK_PATH = DATA / "zone_visit_state.json.lock"
ZONE_VISIT_STATE_VERSION = 1
RESET_ZONE_VISIT_STATE_ON_START = os.environ.get("RESET_ZONE_VISIT_STATE_ON_START", "false").lower() == "true"
DIAGNOSTIC_LOG_PATH = DATA / "zone_engine_diagnostic.log"


def _effective_strategy_version() -> str:
    """Return the exact strategy variant that is actually running."""
    return STRATEGY_VERSION if ZONE_TRIGGER_MODE == "midpoint" else f"{STRATEGY_VERSION}-zone-touch-test"


def _code_commit_sha() -> str | None:
    """Best-effort repository revision for immutable trade/decision lineage."""
    value = str(os.environ.get("GITHUB_SHA", "")).strip()
    if value:
        return value
    if not (PROJECT_ROOT / ".git").exists():
        return None
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        value = result.stdout.strip()
        return value or None
    except Exception:
        return None


CODE_COMMIT_SHA = _code_commit_sha()


def _init_diagnostic_log() -> None:
    """Create one complete text log for the current engine run.

    This is intentionally runtime-only: the GitHub workflow uploads it as an
    artifact and excludes it from persistent git state. It captures logs from
    zone_engine and child event_engine loggers for post-run diagnosis.
    """
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        handler = None
        for existing in logging.getLogger().handlers:
            if getattr(existing, "name", "") == _DIAGNOSTIC_FILE_HANDLER_NAME:
                handler = existing
                break
        if handler is None:
            handler = logging.FileHandler(DIAGNOSTIC_LOG_PATH, mode="w", encoding="utf-8")
            handler.name = _DIAGNOSTIC_FILE_HANDLER_NAME
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            logging.getLogger().addHandler(handler)
        else:
            handler.acquire()
            try:
                stream = getattr(handler, "stream", None)
                if stream is not None:
                    stream.seek(0)
                    stream.truncate(0)
            finally:
                handler.release()
        compact_filter = _CompactSymbolFilter()
        for h in logging.getLogger().handlers:
            h.addFilter(compact_filter)
    except Exception as exc:
        log.warning("[DIAGNOSTIC_LOG] initialization failed: %s", exc)


def _display_symbol(symbol: Any) -> str:
    value = str(symbol or "").strip().upper()
    return value.replace("-", "") if value else value


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    """Append one JSONL record with a cross-process lock and durable flush."""
    DATA.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(obj, ensure_ascii=False, default=str, allow_nan=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _record_entry_decision(
    scan_id: str,
    signal: dict[str, Any],
    stage: str,
    reason: str,
    *,
    selection_rank: int | None = None,
    attempt_id: str | None = None,
    execution_status: str | None = None,
    terminal: bool | None = None,
) -> None:
    """Persist a compact, append-only decision lineage record.

    This is intentionally separate from ``actions.jsonl`` (notifications) and
    ``trades.jsonl`` (execution records). It closes the audit gap between a
    generated signal and its eventual execution outcome.
    """
    event_id = str(signal.get("event_id", ""))
    decision_ts = pd.Timestamp.now(tz="UTC")
    payload = {
        "decision_id": hashlib.sha256(
            f"{scan_id}:{event_id}:{stage}:{reason}:{attempt_id or ''}".encode("utf-8")
        ).hexdigest().upper()[:24],
        "scan_id": str(scan_id),
        "ts": int(decision_ts.timestamp() * 1000),
        "stage": str(stage),
        "reason": str(reason),
        "event_id": event_id,
        "symbol": str(signal.get("symbol", "")),
        "direction": str(signal.get("type", "")),
        "strategy_version": _effective_strategy_version(),
        "code_commit_sha": CODE_COMMIT_SHA,
        "trigger_bar_time": signal.get("trigger_bar_time") or signal.get("time"),
        "selection_rank": selection_rank,
        "attempt_id": attempt_id,
        "execution_status": execution_status,
        "terminal": terminal,
        "decision_ts": decision_ts.isoformat(),
        "execution_gate_ts": signal.get("execution_gate_ts"),
        "execution_age_minutes": signal.get("execution_age_minutes"),
        "execution_age_semantics": "trigger_to_execution_gate_when_available",
        "trigger_to_decision_minutes": _elapsed_minutes(signal.get("trigger_bar_time") or signal.get("time"), decision_ts),
    }
    try:
        ok = research.record_entry_decision(payload, path=ENTRY_DECISIONS_PATH)
        if not ok:
            log.error("[ENTRY_DECISION_WRITE_FAILED] event_id=%s stage=%s error=persistence_returned_false", event_id, stage)
    except Exception as exc:
        # Decision telemetry is fail-open by design, but the loss is visible in logs.
        log.error("[ENTRY_DECISION_WRITE_FAILED] event_id=%s stage=%s error=%s", event_id, stage, exc)



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


def _load_event_claims_unlocked() -> dict[str, dict[str, Any]]:
    if not EVENT_CLAIMS_PATH.exists():
        return {}
    try:
        raw = json.loads(EVENT_CLAIMS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _write_event_claims_unlocked(raw: dict[str, dict[str, Any]]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    tmp = EVENT_CLAIMS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, EVENT_CLAIMS_PATH)


def _claim_event_for_execution(event_id: str) -> tuple[bool, str, str]:
    """Atomically claim an event so one setup cannot execute twice concurrently.

    Returns (claimed, reason, attempt_id). Retryable failures release the claim;
    terminal failures/successes retain it. A stale in-flight claim expires via lease.
    """
    event_id = str(event_id)
    attempt_id = uuid.uuid4().hex.upper()[:16]
    now = int(time.time())
    lock_fh = None
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        with EVENT_CLAIMS_LOCK_PATH.open("a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            raw = _load_event_claims_unlocked()
            existing = raw.get(event_id)
            if isinstance(existing, dict):
                state = str(existing.get("state", "")).lower()
                if state in {"terminal", "completed"}:
                    return False, state, ""
                lease_until = int(existing.get("lease_until", 0) or 0)
                if state == "in_flight" and lease_until > now:
                    return False, "in_flight", ""
            raw[event_id] = {
                "state": "in_flight",
                "attempt_id": attempt_id,
                "claimed_at": now,
                "lease_until": now + EVENT_CLAIM_LEASE_SEC,
            }
            _write_event_claims_unlocked(raw)
            return True, "claimed", attempt_id
    except Exception as exc:
        log.error("[EVENT_CLAIM] failed for %s: %s", event_id, exc)
        return False, "claim_error", ""
    finally:
        if lock_fh is not None:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _finalize_event_claim(event_id: str, attempt_id: str, *, terminal: bool, status: str) -> None:
    """Finalize or release an execution claim without overwriting another attempt."""
    event_id = str(event_id)
    lock_fh = None
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        with EVENT_CLAIMS_LOCK_PATH.open("a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            raw = _load_event_claims_unlocked()
            existing = raw.get(event_id)
            if not isinstance(existing, dict) or str(existing.get("attempt_id")) != str(attempt_id):
                return
            if terminal:
                raw[event_id] = {
                    "state": "terminal",
                    "attempt_id": attempt_id,
                    "finalized_at": int(time.time()),
                    "status": str(status),
                }
            else:
                raw.pop(event_id, None)
            _write_event_claims_unlocked(raw)
    except Exception as exc:
        log.warning("[EVENT_CLAIM] finalize failed for %s: %s", event_id, exc)
    finally:
        if lock_fh is not None:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _load_terminal_event_ids() -> set[str]:
    """Recover terminal execution identities from the append-only trade journal."""
    if not TRADES_PATH.exists():
        return set()
    out: set[str] = set()
    terminal_statuses = {
        "skipped_stale_signal",
        "skipped_tp_min_qty",
        "skipped_min_qty",
        "skipped_invalid_setup",
        "opened_then_emergency_closed",
    }
    for line in TRADES_PATH.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not row.get("event_id"):
            continue
        result = row.get("result", {}) if isinstance(row.get("result"), dict) else {}
        status = str(result.get("status", row.get("status", ""))).lower()
        if status in terminal_statuses:
            out.add(str(row["event_id"]))
    return out


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


def _mark_failed_signal(event_id: str, status: str, error: str = "", *, force_terminal: bool = False) -> None:
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        raw = _load_failed_signal_ids()
        previous = raw.get(str(event_id), {}) if isinstance(raw.get(str(event_id)), dict) else {}
        retry_count = int(previous.get("retry_count", 0) or 0) + 1
        now = int(time.time())
        delay = min(FAILED_SIGNAL_RETRY_MAX_SEC, FAILED_SIGNAL_RETRY_BASE_SEC * (2 ** max(0, retry_count - 1)))
        terminal = bool(force_terminal) or retry_count >= FAILED_SIGNAL_MAX_RETRIES
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
    if value in {"ENTRY_NOT_FILLED", "SKIPPED_MIN_QTY", "SKIPPED_TP_MIN_QTY", "SKIPPED_INVALID_SETUP", "SKIPPED_STALE_SIGNAL", "EXECUTION_QUOTE_UNAVAILABLE", "BLOCKED_PROTECTION_PREFLIGHT"}:
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

    if FUNDAMENTAL_WHITELIST_ENABLED:
        universe_available = available.intersection(FUNDAMENTAL_ASSET_SYMBOLS)
    else:
        universe_available = available
    if WATCHLIST_ONLY:
        symbols = [s for s in WATCHLIST_SYMBOLS if s in universe_available]
        missing = [s for s in WATCHLIST_SYMBOLS if s not in universe_available]
        if missing:
            log.warning("[WATCHLIST_MISSING] symbols_not_active_or_unavailable=%s", ",".join(_display_symbol(x) for x in missing))
    else:
        symbols = sorted(universe_available)
    if MAX_SCAN_SYMBOLS > 0:
        symbols = symbols[:MAX_SCAN_SYMBOLS]
    return symbols



def _zone_visit_key(zone: dict[str, Any], kind: str) -> str:
    """Stable identity for one 1H zone across repeated scans."""
    zone_id = str(zone.get("zone_id", "")).strip()
    if zone_id:
        return f"{kind.upper()}:{zone_id}"
    origin_ts_ms = int(zone.get("origin_ts_ms", -1) or -1)
    if origin_ts_ms > 0:
        return f"{kind.upper()}:ORIGIN:{origin_ts_ms}:{float(zone.get('top', 0.0)):.12f}:{float(zone.get('btm', 0.0)):.12f}"
    # Backward-compatible fallback for pre-fix in-memory callers only. Newly
    # generated zones always carry origin_ts_ms/zone_id.
    return f"{kind.upper()}:{int(zone.get('start', -1))}:{float(zone.get('top', 0.0)):.12f}:{float(zone.get('btm', 0.0)):.12f}"


def _load_zone_visit_state() -> dict[str, Any]:
    """Load only the new zone-visit state schema; no legacy migration is supported."""
    if not ZONE_VISIT_STATE_PATH.exists():
        return {"version": ZONE_VISIT_STATE_VERSION, "trigger_mode": ZONE_TRIGGER_MODE, "symbols": {}}
    try:
        raw = json.loads(ZONE_VISIT_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("[ZONE_STATE] load failed; starting empty state: %s", exc)
        return {"version": ZONE_VISIT_STATE_VERSION, "trigger_mode": ZONE_TRIGGER_MODE, "symbols": {}}
    if not isinstance(raw, dict) or int(raw.get("version", -1)) != ZONE_VISIT_STATE_VERSION or not isinstance(raw.get("symbols"), dict):
        log.warning("[ZONE_STATE] invalid/newer schema; starting empty state")
        return {"version": ZONE_VISIT_STATE_VERSION, "trigger_mode": ZONE_TRIGGER_MODE, "symbols": {}}
    stored_mode = str(raw.get("trigger_mode", "")).strip().lower()
    if stored_mode and stored_mode != ZONE_TRIGGER_MODE:
        log.warning("[ZONE_STATE] trigger_mode changed %s -> %s; starting empty visit state", stored_mode, ZONE_TRIGGER_MODE)
        return {"version": ZONE_VISIT_STATE_VERSION, "trigger_mode": ZONE_TRIGGER_MODE, "symbols": {}}
    raw["trigger_mode"] = ZONE_TRIGGER_MODE
    return raw


def _save_zone_visit_state(state: dict[str, Any]) -> None:
    """Atomically persist zone-visit state after a complete scan batch."""
    DATA.mkdir(parents=True, exist_ok=True)
    payload = {"version": ZONE_VISIT_STATE_VERSION, "trigger_mode": ZONE_TRIGGER_MODE, "symbols": state.get("symbols", {})}
    lock_path = ZONE_VISIT_STATE_LOCK_PATH
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        tmp = ZONE_VISIT_STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, ZONE_VISIT_STATE_PATH)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _make_5m_event_id(symbol: str, direction: str, trigger_ts_ms: int, zone: dict[str, Any]) -> str:
    zone_id = str(zone.get("zone_id", "")).strip()
    if zone_id:
        zone_component = zone_id
    else:
        origin_ts_ms = int(zone.get("origin_ts_ms", -1) or -1)
        zone_component = f"ORIGIN:{origin_ts_ms}:{float(zone.get('top', 0.0)):.12f}:{float(zone.get('btm', 0.0)):.12f}"
    raw = (
        f"ZONE5M:{symbol.upper()}:{direction.upper()}:{int(trigger_ts_ms)}:"
        f"{zone_component}"
    )
    return "ZONE_" + hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()[:24]


def _normalize_closed_5m(bars: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize provider 5m bars and retain only fully closed candles."""
    if not bars:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    x = pd.DataFrame(bars).copy()
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError(f"5m bars missing columns: {missing}")
    raw_ts = x["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(raw_ts):
        x["timestamp"] = pd.to_datetime(raw_ts, utc=True, errors="coerce")
    else:
        numeric = pd.to_numeric(raw_ts, errors="coerce")
        magnitude = float(numeric.dropna().abs().median()) if not numeric.dropna().empty else 0.0
        unit = "ms" if magnitude >= 1e11 else "s" if magnitude >= 1e8 else None
        x["timestamp"] = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce") if unit else pd.to_datetime(raw_ts, utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=required).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if x.empty:
        return x
    now = pd.Timestamp.now(tz="UTC")
    close_time = x["timestamp"] + pd.Timedelta(minutes=5)
    x = x.loc[close_time <= now].copy().reset_index(drop=True)
    return x


def _build_5m_zone_signal(
    symbol: str,
    direction: str,
    zone: dict[str, Any],
    bar: pd.Series,
    prev_bar: pd.Series | None,
    df_1h: pd.DataFrame,
    demand: list[dict[str, Any]],
    supply: list[dict[str, Any]],
    zone_state: dict[str, Any],
) -> dict[str, Any]:
    """Build an execution-ready signal from the configured 5m touch of an existing 1H zone."""
    top = float(zone["top"])
    bottom = float(zone["btm"])
    midpoint = (top + bottom) / 2.0
    midpoint_touch = float(bar["low"]) <= midpoint <= float(bar["high"])
    zone_touch = float(bar["low"]) <= top and float(bar["high"]) >= bottom
    if not zone_touch:
        raise ValueError("zone_touch_required")
    entry = midpoint if ZONE_TRIGGER_MODE == "midpoint" else float(bar["close"])
    fixed_stop_pct = FIXED_STOP_PCT
    risk = entry * fixed_stop_pct / 100.0
    stop = entry - risk if direction == "LONG" else entry + risk
    current_idx = len(df_1h) - 1
    obstacle = _nearest_opposing_level(direction, entry, demand, supply, df_1h, current_idx)
    if obstacle is not None:
        obstacle_price = float(obstacle["price"])
        structural_distance = obstacle_price - entry if direction == "LONG" else entry - obstacle_price
        if structural_distance <= 0 or structural_distance / risk < MIN_STRUCTURE_ROOM_R:
            raise ValueError(f"insufficient_structure_room={structural_distance / risk if risk else 0.0:.3f}R < {MIN_STRUCTURE_ROOM_R:.3f}R")
    elif REQUIRE_STRUCTURE_OBSTACLE:
        raise ValueError("missing_structural_obstacle")

    tp1 = entry * (1.0 + TP1_PCT / 100.0) if direction == "LONG" else entry * (1.0 - TP1_PCT / 100.0)
    tp2 = entry * (1.0 + TP2_PCT / 100.0) if direction == "LONG" else entry * (1.0 - TP2_PCT / 100.0)
    ts = pd.Timestamp(bar["timestamp"])
    trigger_ts_ms = int(ts.timestamp() * 1000)
    zone_copy = {**zone, "kind": "DEMAND" if direction == "LONG" else "SUPPLY"}
    prev_bar_dict = {
        "timestamp": pd.Timestamp(prev_bar["timestamp"]).isoformat() if prev_bar is not None else None,
        "open": float(prev_bar["open"]) if prev_bar is not None else None,
        "high": float(prev_bar["high"]) if prev_bar is not None else None,
        "low": float(prev_bar["low"]) if prev_bar is not None else None,
        "close": float(prev_bar["close"]) if prev_bar is not None else None,
        "volume": float(prev_bar["volume"]) if prev_bar is not None else None,
    }
    event_id = _make_5m_event_id(symbol, direction, trigger_ts_ms, zone)
    atr_1h = float(df_1h.loc[current_idx, "atr50"]) if "atr50" in df_1h.columns else 0.0
    volume_window = df_1h["volume"].rolling(20, min_periods=20).mean() if "volume" in df_1h.columns else pd.Series(dtype=float)
    avg_vol = float(volume_window.iloc[-1]) if not volume_window.empty and pd.notna(volume_window.iloc[-1]) else 0.0
    trigger_volume = float(bar["volume"])
    vol_ratio = trigger_volume / avg_vol if avg_vol > 0 else None
    # Zone age is diagnostic metadata only. An active zone may be traded
    # regardless of age; invalidation is determined by the zone engine itself.
    zone_age_bars = max(0, int(current_idx - int(zone.get("start", current_idx))))
    directional_ok = (float(bar["close"]) >= float(bar["open"])) if direction == "LONG" else (float(bar["close"]) <= float(bar["open"]))
    if REQUIRE_DIRECTIONAL_CANDLE and not directional_ok:
        raise ValueError("directional_candle_required")
    entry_bar = {k: (pd.Timestamp(bar[k]).isoformat() if k == "timestamp" else float(bar[k])) for k in ["timestamp", "open", "high", "low", "close", "volume"]}
    setup_zone = {**zone_copy, "age_bars": zone_age_bars}
    signal_forensics = _signal_forensics(direction, float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"]), max(atr_1h, 1e-12), setup_zone)
    return {
        "event_id": event_id,
        "idx": trigger_ts_ms,
        "time": ts.isoformat(),
        "trigger_bar_time": ts.isoformat(),
        "trigger_timeframe": "5m",
        "type": direction,
        "symbol": symbol.upper(),
        "entry": entry,
        "sl": stop,
        "tp1": tp1,
        "tp2": tp2,
        "risk_pct": fixed_stop_pct,
        "risk_abs": risk,
        "atr": atr_1h,
        "tp1_rr": TP1_PCT / fixed_stop_pct,
        "tp2_rr": TP2_PCT / fixed_stop_pct,
        "strategy": "Demand/Supply Zone First",
        "strategy_version": _effective_strategy_version(),
        "entry_bar": entry_bar,
        "previous_bar": prev_bar_dict,
        "trigger": {
            "type": ("ZONE_MIDPOINT_TOUCH_5M" if ZONE_TRIGGER_MODE == "midpoint" else "ZONE_TOUCH_5M"),
            "alma_required": False,
            "alternate_timeframe": "8h",
            "zone_touch": zone_touch,
            "zone_entry_rule": ("fresh_midpoint_touch_5m" if ZONE_TRIGGER_MODE == "midpoint" else "fresh_zone_touch_5m"),
            "zone_trigger_mode": ZONE_TRIGGER_MODE,
            "zone_midpoint": midpoint,
            "zone_midpoint_pct": 50.0,
            "midpoint_touched_diagnostic": midpoint_touch,
            "previous_bar_midpoint_touch": bool(zone_state.get("previous_midpoint_touch", False)),
            "previous_bar_zone_touch": bool(zone_state.get("previous_zone_touch", False)),
            "trigger_entry_reference": ("midpoint" if ZONE_TRIGGER_MODE == "midpoint" else "trigger_bar_close"),
            "zone_visit_id": zone_state.get("visit_id"),
            "zone_visit_state": "TRIGGERED",
        },
        "zone": setup_zone,
        "zone_counts": {"demand": len(demand), "supply": len(supply)},
        "target": {
            "source": "fixed_entry_percentage",
            "obstacle_source": obstacle.get("source") if obstacle else None,
            "obstacle_price": float(obstacle["price"]) if obstacle else None,
            "tp1_pct": TP1_PCT,
            "tp2_pct": TP2_PCT,
            "tp1_close_fraction": 0.50,
            "tp2_close_fraction": 0.50,
            "be_rule": "after_tp1_filled",
        },
        "risk_model": {
            "sl_source": "fixed_percent_from_entry",
            "fixed_stop_pct": FIXED_STOP_PCT,
            "max_signal_risk_pct": MAX_PRODUCTION_RISK_PCT,
            "initial_risk_pct": fixed_stop_pct,
        },
        "confirmation": {
            "alma_cross": False,
            "directional_candle_required": REQUIRE_DIRECTIONAL_CANDLE,
            "directional_candle_ok": (float(bar["close"]) >= float(bar["open"])) if direction == "LONG" else (float(bar["close"]) <= float(bar["open"])),
            "zone_age_bars": zone_age_bars,
            "minimum_structure_room_r": MIN_STRUCTURE_ROOM_R,
            "zone_touch": zone_touch,
            "midpoint_touch": midpoint_touch,
            "trigger_timeframe": "5m",
            "zone_trigger_mode": ZONE_TRIGGER_MODE,
            "volume_ratio": vol_ratio,
            "bullish_candle": float(bar["close"]) >= float(bar["open"]),
            "bearish_candle": float(bar["close"]) <= float(bar["open"]),
        },
        "source_bar_close": float(bar["close"]),
        "zone_midpoint": midpoint,
        "zone_width_abs": max(0.0, top - bottom),
        "zone_width_pct_of_entry": (max(0.0, top - bottom) / entry) * 100.0 if entry > 0 else None,
        "signal_forensics": signal_forensics,
        "zone_visit": {
            "visit_id": zone_state.get("visit_id"),
            "first_touch_ts": zone_state.get("first_touch_ts"),
            "touch_count_before_trigger": zone_state.get("touch_count", 0),
            "state": "LOCKED",
        },
    }


def _process_5m_zone_visits(
    symbol: str,
    bars: list[dict[str, Any]],
    demand: list[dict[str, Any]],
    supply: list[dict[str, Any]],
    df_1h: pd.DataFrame,
    state_for_symbol: dict[str, Any] | None,
    successful_ids: set[str],
    terminal_event_ids: set[str],
    diagnostics: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """Process closed 5m bars with one locked visit per 1H zone and durable cursor state."""
    x = _normalize_closed_5m(bars)
    now = pd.Timestamp.now(tz="UTC")
    symbol_state = dict(state_for_symbol or {})
    if diagnostics is not None:
        preserved_current_1h_idx = diagnostics.get("current_1h_idx")
        diagnostics.clear()
        diagnostics.update({
            "bars_received": len(bars),
            "bars_closed": len(x),
            "processed_bars": 0,
            "last_processed_before": symbol_state.get("last_processed_5m_ts"),
            "last_processed_after": None,
            "initial_lookback_minutes": INITIAL_5M_TRIGGER_LOOKBACK_MINUTES,
            "trigger_max_age_minutes": MAX_5M_TRIGGER_AGE_MINUTES,
            "zones": {},
            "touch_events": [],
            "rearm_events": [],
            "pending_retries": 0,
            "current_1h_idx": preserved_current_1h_idx,
            "decision": None,
        })
    symbol_state.setdefault("version", ZONE_VISIT_STATE_VERSION)
    symbol_state.setdefault("zones", {})

    def _set_decision(status: str, *, zone_key: str | None = None, direction: str | None = None,
                      reason: str | None = None, timestamp: str | None = None) -> None:
        if diagnostics is None:
            return
        diagnostics["decision"] = {
            "status": status,
            "zone_key": zone_key,
            "direction": direction,
            "reason": reason,
            "timestamp": timestamp,
        }

    active_zone_map: dict[str, tuple[str, dict[str, Any]]] = {}
    for z in demand:
        active_zone_map[_zone_visit_key(z, "DEMAND")] = ("LONG", z)
    for z in supply:
        active_zone_map[_zone_visit_key(z, "SUPPLY")] = ("SHORT", z)

    if diagnostics is not None:
        for zone_key, (direction, zone) in active_zone_map.items():
            top = float(zone["top"]); bottom = float(zone["btm"]); midpoint = (top + bottom) / 2.0
            diagnostics["zones"][zone_key] = {
                "direction": direction,
                "kind": "DEMAND" if direction == "LONG" else "SUPPLY",
                "start_idx": int(zone.get("start", -1)),
                "top": top,
                "bottom": bottom,
                "midpoint": midpoint,
                "width": max(0.0, top - bottom),
                "state_before": None,
                "state_after": None,
                "bars_in_zone": 0,
                "zone_touches": 0,
                "midpoint_touches": 0,
                "same_visit_blocks": 0,
                "ambiguous_blocks": 0,
                "stale_touches": 0,
                "activation_blocks": 0,
                "structure_rejects": 0,
                "directional_rejects": 0,
                "other_rejects": 0,
                "signals_created": 0,
                "rearms": 0,
                "touch_timestamps": [],
                "closest_midpoint_distance_pct": None,
                "closest_midpoint_bar": None,
                "window_bars_in_zone": 0,
                "window_zone_touches": 0,
                "window_midpoint_touches": 0,
                "window_closest_midpoint_distance_pct": None,
                "window_closest_midpoint_bar": None,
                "window_last_midpoint_touch": None,
                "window_last_zone_touch": None,
            }

    # Diagnostic-only pass over all fetched closed 5m bars. This does not alter
    # the durable cursor/state machine; it tells the operator whether the market
    # touched a midpoint in the fetched window even if that touch was already
    # processed on an earlier cron run.
    if diagnostics is not None and not x.empty:
        for zone_key, (direction, zone) in active_zone_map.items():
            zdiag = diagnostics["zones"][zone_key]
            midpoint = float(zdiag["midpoint"])
            activation_idx = int(zone.get("start", -1)) + SWING_LEN
            activation_ts = pd.Timestamp(df_1h.loc[activation_idx, "timestamp"]) if 0 <= activation_idx < len(df_1h) else None
            for _, wbar in x.iterrows():
                low = float(wbar["low"]); high = float(wbar["high"]); ts = pd.Timestamp(wbar["timestamp"])
                if activation_ts is not None and ts < activation_ts:
                    continue
                zone_touched = low <= float(zone["top"]) and high >= float(zone["btm"])
                if zone_touched:
                    zdiag["window_bars_in_zone"] += 1
                    zdiag["window_zone_touches"] += 1
                    zdiag["window_last_zone_touch"] = ts.isoformat()
                touched = low <= midpoint <= high
                distance_pct = 0.0 if touched else ((midpoint - high) if high < midpoint else (low - midpoint)) / midpoint * 100.0 if midpoint else None
                if distance_pct is not None and (zdiag["window_closest_midpoint_distance_pct"] is None or distance_pct < zdiag["window_closest_midpoint_distance_pct"]):
                    zdiag["window_closest_midpoint_distance_pct"] = distance_pct
                    zdiag["window_closest_midpoint_bar"] = {"timestamp": ts.isoformat(), "open": float(wbar["open"]), "high": high, "low": low, "close": float(wbar["close"]), "distance_pct": distance_pct}
                if touched:
                    zdiag["window_midpoint_touches"] += 1
                    zdiag["window_last_midpoint_touch"] = ts.isoformat()

    # Drop stale state entries for zones that no longer exist; a broken 1H zone
    # cannot be resurrected by an old 5m visit.
    symbol_state["zones"] = {k: v for k, v in symbol_state["zones"].items() if k in active_zone_map and isinstance(v, dict)}
    last_processed_raw = symbol_state.get("last_processed_5m_ts")
    if last_processed_raw:
        last_processed = pd.Timestamp(last_processed_raw)
        start_mask = x["timestamp"] > last_processed
    else:
        # Align the initial lookback boundary to the 5m bar clock. Without this,
        # cron seconds can make a bar exactly at the configured lookback edge fall
        # just outside the >= cutoff (e.g. 10:40:00 vs 10:40:47), causing the first
        # actionable 5m touch to be skipped on a fresh state.
        initial_cutoff = (now - pd.Timedelta(minutes=INITIAL_5M_TRIGGER_LOOKBACK_MINUTES)).floor("5min")
        start_mask = x["timestamp"] >= initial_cutoff

    signals: list[dict[str, Any]] = []
    trigger_text: str | None = None
    processed_rows = x.loc[start_mask].copy()
    if diagnostics is not None:
        diagnostics["processed_bars"] = int(len(processed_rows))
        if not x.empty:
            diagnostics["latest_closed_5m_ts"] = pd.Timestamp(x["timestamp"].iloc[-1]).isoformat()
    for _, bar in processed_rows.iterrows():
        bar_ts = pd.Timestamp(bar["timestamp"])
        prev_bar = x.loc[x["timestamp"] < bar_ts].tail(1)
        prev = prev_bar.iloc[0] if not prev_bar.empty else None
        # First update re-arm status for every currently active zone. Re-arm only
        # after a closed 5m candle finishes fully beyond the far boundary of the zone.
        for zone_key, (direction, zone) in active_zone_map.items():
            zs = symbol_state["zones"].setdefault(zone_key, {"state": "ARMED", "visit_id": f"{symbol}:{zone_key}", "touch_count": 0})
            close = float(bar["close"])
            if diagnostics is not None:
                zdiag = diagnostics["zones"][zone_key]
                if zdiag["state_before"] is None:
                    zdiag["state_before"] = zs.get("state", "ARMED")
            if zs.get("state") == "LOCKED":
                rearmed = (close > float(zone["top"])) if direction == "LONG" else (close < float(zone["btm"]))
                # A bar that both exits beyond the far edge and crosses the midpoint
                # is an exit/breakout bar, not a fresh return. Re-arm only after it.
                if rearmed and zs.get("first_touch_ts") and bar_ts > pd.Timestamp(zs["first_touch_ts"]):
                    zs.update({"state": "ARMED", "rearm_ts": bar_ts.isoformat(), "pending_signal": None, "trigger_event_id": None, "touch_count": 0, "first_touch_ts": None, "last_touch_ts": None})
                    if diagnostics is not None:
                        zdiag = diagnostics["zones"][zone_key]
                        zdiag["rearms"] += 1
                        diagnostics["rearm_events"].append({
                            "zone_key": zone_key, "direction": direction, "timestamp": bar_ts.isoformat(), "close": close,
                            "open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "volume": float(bar["volume"]),
                            "visit_id": zs.get("visit_id")
                        })
                        zdiag["state_after"] = "ARMED"
                    continue
            if zs.get("state") != "ARMED":
                continue
            activation_idx = int(zone.get("start", -1)) + SWING_LEN
            if activation_idx >= 0 and activation_idx < len(df_1h):
                activation_ts = pd.Timestamp(df_1h.loc[activation_idx, "timestamp"])
                if bar_ts < activation_ts:
                    if diagnostics is not None:
                        diagnostics["zones"][zone_key]["activation_blocks"] += 1
                        diagnostics["touch_events"].append({
                            "timestamp": bar_ts.isoformat(), "zone_key": zone_key, "direction": direction,
                            "midpoint": float((float(zone["top"]) + float(zone["btm"])) / 2.0),
                            "state_before": zs.get("state", "ARMED"), "reason": "zone_not_active",
                            "visit_id": zs.get("visit_id"),
                            "bar": {"open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"])}
                        })
                    _set_decision("BLOCKED_ZONE_NOT_ACTIVE", zone_key=zone_key, direction=direction, reason="zone_not_active", timestamp=bar_ts.isoformat())
                    continue
            midpoint = (float(zone["top"]) + float(zone["btm"])) / 2.0
            zone_overlap = float(bar["low"]) <= float(zone["top"]) and float(bar["high"]) >= float(zone["btm"])
            midpoint_touch = float(bar["low"]) <= midpoint <= float(bar["high"])
            if diagnostics is not None and zone_overlap:
                diagnostics["zones"][zone_key]["bars_in_zone"] += 1
            distance_pct = 0.0 if midpoint_touch else ((midpoint - float(bar["high"])) if float(bar["high"]) < midpoint else (float(bar["low"]) - midpoint)) / midpoint * 100.0 if midpoint else None
            if diagnostics is not None and distance_pct is not None:
                zdiag = diagnostics["zones"][zone_key]
                if zdiag["closest_midpoint_distance_pct"] is None or distance_pct < zdiag["closest_midpoint_distance_pct"]:
                    zdiag["closest_midpoint_distance_pct"] = distance_pct
                    zdiag["closest_midpoint_bar"] = {"timestamp": bar_ts.isoformat(), "open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "distance_pct": distance_pct}
            touch = midpoint_touch if ZONE_TRIGGER_MODE == "midpoint" else zone_overlap
            if not touch:
                continue
            if diagnostics is not None:
                zdiag = diagnostics["zones"][zone_key]
                zdiag["zone_touches"] += 1
                if midpoint_touch:
                    zdiag["midpoint_touches"] += 1
                zdiag["touch_timestamps"].append(bar_ts.isoformat())
            # Ambiguous overlap is a no-trade state; still record the visit so it
            # cannot create repeated directional attempts while price oscillates.
            if direction == "LONG":
                opposite_touch = any(float(bar["low"]) <= float(z["top"]) and float(bar["high"]) >= float(z["btm"]) for z in supply)
            else:
                opposite_touch = any(float(bar["low"]) <= float(z["top"]) and float(bar["high"]) >= float(z["btm"]) for z in demand)
            if opposite_touch:
                zs.update({"state": "LOCKED", "first_touch_ts": bar_ts.isoformat(), "last_touch_ts": bar_ts.isoformat(), "touch_count": int(zs.get("touch_count", 0)) + 1, "lock_reason": "ambiguous_overlap", "visit_id": f"{symbol}:{zone_key}:{int(bar_ts.timestamp()*1000)}"})
                if diagnostics is not None:
                    diagnostics["zones"][zone_key]["ambiguous_blocks"] += 1
                    diagnostics["touch_events"].append({"timestamp": bar_ts.isoformat(), "zone_key": zone_key, "direction": direction, "midpoint": midpoint, "state_before": "ARMED", "reason": "ambiguous_overlap", "visit_id": zs.get("visit_id"), "bar": {"open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"])}})
                    diagnostics["zones"][zone_key]["state_after"] = "LOCKED"
                _set_decision("BLOCKED_AMBIGUOUS", zone_key=zone_key, direction=direction, reason="ambiguous_overlap", timestamp=bar_ts.isoformat())
                continue
            state = dict(zs)
            # Only the previous bar that was actually processed in this scan batch
            # participates in the continuous-visit guard. The first processed bar
            # often has one older fetched bar before it; that bar may already belong
            # to the pre-run history and must not suppress the first actionable touch
            # after a fresh/empty durable state. Prior-run touches are represented by
            # the persisted zone state itself (LOCKED), so they do not need this guard.
            prev_was_processed_now = False
            if prev is not None:
                prev_ts = pd.Timestamp(prev["timestamp"])
                if last_processed_raw:
                    try:
                        prev_was_processed_now = prev_ts > last_processed
                    except Exception:
                        prev_was_processed_now = True
                else:
                    prev_was_processed_now = bool(prev_ts >= processed_rows["timestamp"].min())
            state["previous_midpoint_touch"] = bool(
                prev_was_processed_now and prev is not None and
                float(prev["low"]) <= midpoint <= float(prev["high"])
            )
            state["previous_zone_touch"] = bool(
                prev_was_processed_now and prev is not None and
                float(prev["low"]) <= float(zone["top"]) and float(prev["high"]) >= float(zone["btm"])
            )
            previous_touch = state["previous_midpoint_touch"] if ZONE_TRIGGER_MODE == "midpoint" else state["previous_zone_touch"]
            if previous_touch:
                # This is still the same continuous visit, not a new touch event.
                zs.update({"state": "LOCKED", "last_touch_ts": bar_ts.isoformat(), "touch_count": int(zs.get("touch_count", 0)) + 1})
                if diagnostics is not None:
                    diagnostics["zones"][zone_key]["same_visit_blocks"] += 1
                    diagnostics["touch_events"].append({"timestamp": bar_ts.isoformat(), "zone_key": zone_key, "direction": direction, "midpoint": midpoint, "state_before": "LOCKED_OR_CONTINUOUS", "reason": ("previous_5m_midpoint_touch" if ZONE_TRIGGER_MODE == "midpoint" else "previous_5m_zone_touch"), "visit_id": zs.get("visit_id"), "touch_count_before_trigger": int(zs.get("touch_count", 0)), "bar": {"open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"])}})
                _set_decision("BLOCKED_SAME_VISIT", zone_key=zone_key, direction=direction, reason=("previous_5m_midpoint_touch" if ZONE_TRIGGER_MODE == "midpoint" else "previous_5m_zone_touch"), timestamp=bar_ts.isoformat())
                continue
            if not zs.get("first_touch_ts"):
                visit_id = f"{symbol}:{zone_key}:{int(bar_ts.timestamp()*1000)}"
                zs.update({"state": "LOCKED", "first_touch_ts": bar_ts.isoformat(), "last_touch_ts": bar_ts.isoformat(), "touch_count": int(zs.get("touch_count", 0)), "visit_id": visit_id, "lock_reason": ("midpoint_touch" if ZONE_TRIGGER_MODE == "midpoint" else "zone_touch")})
            # Compute the structural obstacle independently before signal construction so
            # rejected touches retain the same research feature that would have been used
            # for an executable signal. This does not change the production decision.
            research_entry_ref = midpoint if ZONE_TRIGGER_MODE == "midpoint" else float(bar["close"])
            research_risk_abs = research_entry_ref * FIXED_STOP_PCT / 100.0 if research_entry_ref > 0 else None
            research_obstacle = _nearest_opposing_level(direction, research_entry_ref, demand, supply, df_1h, len(df_1h) - 1)
            if diagnostics is not None:
                touch_payload = {
                    "timestamp": bar_ts.isoformat(), "zone_key": zone_key, "direction": direction,
                    "midpoint": midpoint, "visit_id": zs.get("visit_id"),
                    "touch_count_before_trigger": int(zs.get("touch_count", 0)),
                    "entry_ref": research_entry_ref,
                    "bar": {"open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"])},
                }
                if research_obstacle is not None:
                    obstacle_price = float(research_obstacle.get("price"))
                    structural_distance = obstacle_price - research_entry_ref if direction == "LONG" else research_entry_ref - obstacle_price
                    touch_payload.update({
                        "obstacle_price": obstacle_price,
                        "obstacle_source": research_obstacle.get("source"),
                        "structural_distance": structural_distance,
                        "structure_room_R": (structural_distance / research_risk_abs) if research_risk_abs else None,
                    })
            try:
                signal = _build_5m_zone_signal(symbol, direction, zone, bar, prev, df_1h, demand, supply, zs)
            except ValueError as exc:
                reason = str(exc)
                zs.update({"state": "LOCKED", "lock_reason": reason, "trigger_event_id": None})
                if diagnostics is not None:
                    if reason.startswith("insufficient_structure_room"):
                        diagnostics["zones"][zone_key]["structure_rejects"] += 1
                        decision = "BLOCKED_STRUCTURE_ROOM"
                    elif reason == "directional_candle_required":
                        diagnostics["zones"][zone_key]["directional_rejects"] += 1
                        decision = "BLOCKED_DIRECTIONAL_CANDLE"
                    else:
                        diagnostics["zones"][zone_key]["other_rejects"] += 1
                        decision = "BLOCKED_SIGNAL_BUILD"
                    touch_payload["state_before"] = "ARMED"
                    touch_payload["reason"] = reason
                    diagnostics["touch_events"].append(touch_payload)
                    diagnostics["zones"][zone_key]["state_after"] = "LOCKED"
                    _set_decision(decision, zone_key=zone_key, direction=direction, reason=reason, timestamp=bar_ts.isoformat())
                continue
            event_id = str(signal["event_id"])
            age_min = max(0.0, (now - bar_ts).total_seconds() / 60.0)
            if age_min > MAX_5M_TRIGGER_AGE_MINUTES:
                # A stale touch is useful audit data but must not consume the zone visit.
                # Otherwise a delayed provider/API response could lock a zone until a
                # future full exit/re-entry, silently suppressing the next valid setup.
                zs.update({
                    "state": "ARMED",
                    "last_touch_ts": bar_ts.isoformat(),
                    "touch_count": int(zs.get("touch_count", 0)) + 1,
                    "lock_reason": ("stale_midpoint_touch_ignored" if ZONE_TRIGGER_MODE == "midpoint" else "stale_zone_touch_ignored"),
                    "trigger_event_id": None,
                    "pending_signal": None,
                })
                if diagnostics is not None:
                    diagnostics["zones"][zone_key]["stale_touches"] += 1
                    diagnostics["touch_events"].append({"timestamp": bar_ts.isoformat(), "zone_key": zone_key, "direction": direction, "midpoint": midpoint, "state_before": "ARMED", "reason": ("stale_midpoint_touch_ignored" if ZONE_TRIGGER_MODE == "midpoint" else "stale_zone_touch_ignored"), "age_min": age_min, "visit_id": zs.get("visit_id"), "touch_count_before_trigger": int(zs.get("touch_count", 0)), "bar": {"open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"])}})
                _set_decision("BLOCKED_STALE_TOUCH", zone_key=zone_key, direction=direction, reason=("stale_midpoint_touch_ignored" if ZONE_TRIGGER_MODE == "midpoint" else "stale_zone_touch_ignored"), timestamp=bar_ts.isoformat())
                continue
            zs["trigger_event_id"] = event_id
            zs["pending_signal"] = signal
            zs["state"] = "LOCKED"
            # Re-use the same event for a retryable execution failure.
            if event_id not in successful_ids and event_id not in terminal_event_ids:
                signals.append(signal)
                trigger_text = f"{direction} @ {float(signal.get('entry', midpoint)):.12g} 5m_{ZONE_TRIGGER_MODE}_touch"
                if diagnostics is not None:
                    diagnostics["zones"][zone_key]["signals_created"] += 1
                    diagnostics["touch_events"].append({
                        "timestamp": bar_ts.isoformat(), "zone_key": zone_key, "direction": direction,
                        "midpoint": midpoint, "state_before": "ARMED", "reason": "signal_created", "touch_mode": ZONE_TRIGGER_MODE,
                        "visit_id": zs.get("visit_id"), "touch_count_before_trigger": int(zs.get("touch_count", 0)),
                        "event_id": event_id, "age_min": age_min,
                        "entry_ref": float(signal.get("entry", midpoint) or midpoint),
                        "sl_ref": float(signal.get("sl", 0.0) or 0.0),
                        "tp1": float(signal.get("tp1", 0.0) or 0.0),
                        "tp2": float(signal.get("tp2", 0.0) or 0.0),
                        "obstacle_price": (signal.get("target") or {}).get("obstacle_price"),
                        "zone_age_bars": (signal.get("confirmation") or {}).get("zone_age_bars"),
                        "bar": {"open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"])},
                        "previous_bar": ({"timestamp": pd.Timestamp(prev["timestamp"]).isoformat(), "open": float(prev["open"]), "high": float(prev["high"]), "low": float(prev["low"]), "close": float(prev["close"]), "volume": float(prev["volume"])} if prev is not None else {}),
                    })
                _set_decision("SIGNAL_CREATED", zone_key=zone_key, direction=direction, reason="signal_created", timestamp=bar_ts.isoformat())

    if not processed_rows.empty:
        symbol_state["last_processed_5m_ts"] = pd.Timestamp(processed_rows["timestamp"].max()).isoformat()
    symbol_state["last_scan_ts"] = now.isoformat()

    if diagnostics is not None:
        diagnostics["last_processed_after"] = symbol_state.get("last_processed_5m_ts")
        for zone_key, zdiag in diagnostics["zones"].items():
            if zone_key in symbol_state["zones"]:
                zdiag["state_after"] = symbol_state["zones"][zone_key].get("state")
        if diagnostics.get("decision") is None:
            diagnostics["decision"] = {
                "status": "NO_5M_TOUCH" if not processed_rows.empty else "NO_NEW_5M_BARS",
                "zone_key": None,
                "direction": None,
                "reason": ("no_configured_touch_in_processed_5m_bars" if not processed_rows.empty else "no_new_closed_5m_bars"),
                "timestamp": diagnostics.get("last_processed_after"),
            }

    # Retry an already-created event only while it remains fresh. This keeps
    # transient exchange failures attached to the same visit/event id.
    for zs in symbol_state["zones"].values():
        pending = zs.get("pending_signal")
        if not isinstance(pending, dict):
            continue
        eid = str(pending.get("event_id", ""))
        if not eid or eid in successful_ids or eid in terminal_event_ids:
            continue
        try:
            age_min = max(0.0, (now - pd.Timestamp(pending["trigger_bar_time"])).total_seconds() / 60.0)
        except Exception:
            age_min = float("inf")
        if age_min <= MAX_5M_TRIGGER_AGE_MINUTES and not any(str(s.get("event_id")) == eid for s in signals):
            signals.append(pending)
            if diagnostics is not None:
                diagnostics["pending_retries"] += 1
            trigger_text = trigger_text or f"{pending.get('type')} @ {pending.get('entry')} 5m_{ZONE_TRIGGER_MODE}_touch_retry"
    return signals, symbol_state, trigger_text

def _signal_count_from_diag(diagnostics: dict[str, Any]) -> int:
    return sum(int(z.get("signals_created", 0)) for z in (diagnostics.get("zones") or {}).values())


def _format_zone_for_log(zone: dict[str, Any]) -> str:
    try:
        bottom = float(zone.get("btm"))
        top = float(zone.get("top"))
        return f"[{bottom:.12g},{top:.12g}]"
    except Exception:
        return "[invalid]"


def _active_log_zone(latest_price: float, demand: list[dict], supply: list[dict]) -> tuple[str, dict[str, Any]] | None:
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    for kind, zones in (("DEMAND", demand), ("SUPPLY", supply)):
        for zone in zones:
            try:
                bottom = float(zone["btm"])
                top = float(zone["top"])
                if bottom <= latest_price <= top:
                    midpoint = (bottom + top) / 2.0
                    candidates.append((abs(latest_price - midpoint), kind, zone))
            except (TypeError, ValueError, KeyError):
                continue
    if not candidates:
        return None
    _, kind, zone = min(candidates, key=lambda x: x[0])
    return kind, zone


def _log_5m_zone_diagnostics(
    symbol: str,
    latest_price: float,
    latest_closed_1h_time: pd.Timestamp,
    demand: list[dict[str, Any]],
    supply: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    symbol_state: dict[str, Any],
    pending_signal_count: int,
) -> None:
    """Log one actionable zone/5m outcome per symbol; detail remains in scan JSON."""
    del latest_closed_1h_time, symbol_state
    display = _display_symbol(symbol)
    decision = diagnostics.get("decision") or {}
    status = str(decision.get("status") or "UNKNOWN")
    reason = str(decision.get("reason") or "")
    zone_key = decision.get("zone_key")
    zones = diagnostics.get("zones") or {}
    selected = zones.get(zone_key) if zone_key else None
    active = _active_log_zone(latest_price, demand, supply)
    actionable_block = status in {
        "BLOCKED_SAME_VISIT", "BLOCKED_AMBIGUOUS", "BLOCKED_STRUCTURE_ROOM",
        "BLOCKED_DIRECTIONAL_CANDLE", "BLOCKED_SIGNAL_BUILD", "BLOCKED_STALE_TOUCH",
    }
    if not active and pending_signal_count == 0 and _signal_count_from_diag(diagnostics) == 0 and not actionable_block:
        return
    if selected:
        zone_label = f"{selected.get('kind','ZONE')} {_format_zone_for_log({'btm': selected.get('bottom'), 'top': selected.get('top')})}"
    elif active:
        zone_label = f"{active[0]} {_format_zone_for_log(active[1])}"
    else:
        zone_label = "none"
    latest_5m = diagnostics.get("latest_closed_5m_ts") or "None"
    processed = int(diagnostics.get("processed_bars", 0))
    age = "None"
    if latest_5m != "None":
        try:
            age = f"{max(0.0, (pd.Timestamp.now(tz='UTC') - pd.Timestamp(latest_5m)).total_seconds() / 60.0):.1f}m"
        except Exception:
            pass
    suffix = f" | reason={reason}" if reason else ""
    if pending_signal_count:
        suffix += f" | pending={pending_signal_count}"
    log.info(
        "[ZONE_STATUS] %s | 1h_close=%.12g | active=%s | 5m=%s | processed=%d latest_5m=%s age=%s | zone=%s%s",
        display, latest_price, bool(active), status, processed, latest_5m, age, zone_label, suffix,
    )


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
        tp_levels = setup.get("tp_levels") if isinstance(setup.get("tp_levels"), list) else None
        if not tp_levels:
            risk_pct = max(stop_loss_pct, 0.05)
            tp_levels = [
                {"leg": "tp1", "pnl_pct": TP1_PCT, "close_fraction": 0.50},
                {"leg": "tp2", "pnl_pct": TP2_PCT, "close_fraction": 0.50},
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
        "strategy_version": str(signal.get("strategy_version", STRATEGY_VERSION)),
        "entry_rule": str((signal.get("trigger") or {}).get("zone_entry_rule", "fresh_midpoint_touch")),
        "stop_rule": str((signal.get("risk_model") or {}).get("sl_source", "fixed_percent_from_entry")),
        "target_rule": str((signal.get("target") or {}).get("source", "fixed_entry_percentage")),
        "be_rule": str((signal.get("target") or {}).get("be_rule", "after_tp1_filled")),
        "signal_snapshot": dict(signal),
        "target": dict(signal.get("target", {})) if isinstance(signal.get("target"), dict) else {},
        "risk_model": dict(signal.get("risk_model", {})) if isinstance(signal.get("risk_model"), dict) else {},
        "trigger": dict(signal.get("trigger", {})) if isinstance(signal.get("trigger"), dict) else {},
        "entry_bar": dict(signal.get("entry_bar", {})) if isinstance(signal.get("entry_bar"), dict) else {},
        "previous_bar": dict(signal.get("previous_bar", {})) if isinstance(signal.get("previous_bar"), dict) else {},
        "signal_price": float(signal["entry"]),
        "entry_reference": float(signal["entry"]),
        "invalidation_price": float(signal["sl"]),
        "risk_pct": risk_pct,
        "target_rr": float(signal.get("tp2_rr", TP2_PCT / FIXED_STOP_PCT)),
        "planned_weighted_rr": float(signal.get("tp1_rr", TP1_PCT / FIXED_STOP_PCT)) * 0.50 + float(signal.get("tp2_rr", TP2_PCT / FIXED_STOP_PCT)) * 0.50,
        "tp_levels": [
            {"leg": "tp1", "pnl_pct": TP1_PCT, "close_fraction": 0.50, "price": float(signal["tp1"])},
            {"leg": "tp2", "pnl_pct": TP2_PCT, "close_fraction": 0.50, "price": float(signal["tp2"])},
        ],
        "target_price": float(signal["tp2"]),
        "zone": signal.get("zone", {}),
        "zone_counts": dict(signal.get("zone_counts", {})) if isinstance(signal.get("zone_counts"), dict) else {},
        "confirmation": signal.get("confirmation", {}),
        "signal_forensics": signal.get("signal_forensics", {}),
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
    risk_limit_epsilon = 1e-9
    if risk_pct > MAX_PRODUCTION_RISK_PCT + risk_limit_epsilon:
        return False, f"risk_pct_above_limit={risk_pct}"
    target = signal.get("target") if isinstance(signal.get("target"), dict) else {}
    obstacle_price = target.get("obstacle_price")
    try:
        obstacle_price = float(obstacle_price) if obstacle_price is not None else None
    except (TypeError, ValueError):
        obstacle_price = None
    min_room_r = MIN_STRUCTURE_ROOM_R
    risk_abs = abs(entry - sl)
    if obstacle_price is None and REQUIRE_STRUCTURE_OBSTACLE:
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
    # Kept as a small geometry helper for tests/diagnostics. Production uses the
    # same fixed stop percentage through _rebase_protection_after_fill().
    risk = avg_price * risk_pct / 100.0
    if direction == "LONG":
        return avg_price - risk, avg_price * (1.0 + TP1_PCT / 100.0), avg_price * (1.0 + TP2_PCT / 100.0)
    return avg_price + risk, avg_price * (1.0 - TP1_PCT / 100.0), avg_price * (1.0 - TP2_PCT / 100.0)


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

    fixed_stop_pct = float(os.environ.get("FIXED_STOP_PCT", "10.00"))
    if not (0.0 < fixed_stop_pct <= 10.0):
        raise ValueError(f"FIXED_STOP_PCT must be in (0, 10], got {fixed_stop_pct}")
    risk = entry * fixed_stop_pct / 100.0
    if direction == "LONG":
        sl = entry - risk
    else:
        sl = entry + risk

    if risk <= 0:
        raise ValueError("post-fill risk is non-positive")

    obstacle = target.get("obstacle_price")
    try:
        obstacle = float(obstacle) if obstacle is not None else None
    except (TypeError, ValueError):
        obstacle = None

    # Targets are fixed percentages from the actual fill. The obstacle is kept
    # only as metadata/diagnostic context; it no longer compresses TP distance.
    tp1_distance = entry * TP1_PCT / 100.0
    tp2_distance = entry * TP2_PCT / 100.0
    if direction == "LONG":
        tp1 = entry + tp1_distance
        tp2 = entry + tp2_distance
    else:
        tp1 = entry - tp1_distance
        tp2 = entry - tp2_distance
    target_source = "fixed_entry_percentage_after_fill"

    return {
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "risk_abs": risk,
        "risk_pct": fixed_stop_pct,
        "tp1_rr": abs(tp1 - entry) / risk,
        "tp2_rr": abs(tp2 - entry) / risk,
        "target_source": target_source,
        "obstacle_price": obstacle,
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

def _adverse_signal_drift_pct(signal_entry: float, executable_price: float, direction: str) -> float:
    """Distance the current executable price has moved against the original signal."""
    if signal_entry <= 0 or executable_price <= 0:
        return float("inf")
    if direction == "LONG":
        return max(0.0, executable_price - signal_entry) / signal_entry * 100.0
    return max(0.0, signal_entry - executable_price) / signal_entry * 100.0


def _build_actual_signal_from_rebase(signal: dict[str, Any], rebased: dict[str, Any]) -> dict[str, Any]:
    actual_signal = dict(signal)
    actual_signal.update({
        "entry": float(rebased.get("entry", signal.get("entry"))),
        "sl": float(rebased["sl"]),
        "tp1": float(rebased["tp1"]),
        "tp2": float(rebased["tp2"]),
        "risk_pct": float(rebased["risk_pct"]),
        "risk_abs": float(rebased["risk_abs"]),
        "tp1_rr": float(rebased["tp1_rr"]),
        "tp2_rr": float(rebased["tp2_rr"]),
        "target": {
            **(signal.get("target") if isinstance(signal.get("target"), dict) else {}),
            "source": rebased["target_source"],
            "obstacle_price": rebased.get("obstacle_price"),
        },
    })
    return actual_signal


def _validate_exchange_price_distinctness(signal: dict[str, Any]) -> tuple[bool, str]:
    """Ensure SL/TP survive exchange pricePrecision rounding as distinct prices."""
    try:
        contract = get_contract(signal["symbol"]) or {}
        price_precision = int(contract.get("pricePrecision") or 0)
        fmt = lambda x: f"{float(x):.{price_precision}f}"
        entry, sl, tp1, tp2 = (float(signal[k]) for k in ("entry", "sl", "tp1", "tp2"))
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"price_precision_validation_failed: {exc}"
    if fmt(entry) == fmt(sl):
        return False, f"SL collapses to entry at pricePrecision={price_precision}: entry={fmt(entry)} sl={fmt(sl)}"
    if fmt(entry) == fmt(tp1):
        return False, f"TP1 collapses to entry at pricePrecision={price_precision}: entry={fmt(entry)} tp1={fmt(tp1)}"
    if fmt(entry) == fmt(tp2) or fmt(tp1) == fmt(tp2):
        return False, f"TP levels collapse at pricePrecision={price_precision}: entry={fmt(entry)} tp1={fmt(tp1)} tp2={fmt(tp2)}"
    return True, "ok"


def _is_terminal_execution_failure(execution: dict[str, Any]) -> bool:
    status = str(execution.get("status", "")).lower()
    error = str(execution.get("error", "")).lower()
    if status in {"skipped_stale_signal", "skipped_tp_min_qty", "skipped_min_qty", "skipped_invalid_setup", "opened_then_emergency_closed"}:
        return True
    terminal_fragments = (
        "risk_pct_above_limit",
        "insufficient_structure_room",
        "signal_drift",
        "execution_slippage",
        "collapses to entry",
        "missing_structural_obstacle",
        "calculated quantity is <= 0",
    )
    return any(fragment in error for fragment in terminal_fragments)


def _elapsed_seconds(start_ts: Any, end_ts: Any) -> float | None:
    try:
        a = pd.Timestamp(start_ts); b = pd.Timestamp(end_ts)
        if a.tzinfo is None: a = a.tz_localize("UTC")
        if b.tzinfo is None: b = b.tz_localize("UTC")
        return max(0.0, (b - a).total_seconds())
    except Exception:
        return None


def _elapsed_minutes(start_ts: Any, end_ts: Any) -> float | None:
    sec = _elapsed_seconds(start_ts, end_ts)
    return sec / 60.0 if sec is not None else None


def execute_new_position(signal: dict[str, Any]) -> dict[str, Any]:
    symbol = str(signal["symbol"])
    direction = str(signal["type"]).upper()
    event_id = str(signal["event_id"])
    entry_price = float(signal["entry"])
    execution_started_ts = pd.Timestamp.now(tz="UTC")
    signal["execution_started_ts"] = execution_started_ts.isoformat()

    # Validate the planned setup before making any network call or opening a position.
    valid, reason = _validate_trade_geometry(signal)
    if not valid:
        log.warning("[EXEC_SKIPPED] %s %s | invalid_setup | %s", symbol, direction, reason)
        return {"status": "skipped_invalid_setup", "error": reason, "symbol": symbol, "direction": direction}

    # Do not open first and discover that the trigger-order endpoint is unavailable.
    # BingX can temporarily disable this endpoint under its trigger-frequency rule;
    # in that state there must be NO market entry because mandatory protection cannot
    # be installed/verified safely. Also clear any stale engine-owned protection on
    # a flat symbol before the market order so old TP/SLs cannot consume protection
    # capacity and cause an avoidable emergency rollback.
    try:
        protection_capacity = prepare_protection_capacity(symbol, direction)
    except Exception as exc:
        protection_capacity = {"status": "error", "error": str(exc)}
    if protection_capacity.get("status") not in {"ready"}:
        reason = str(protection_capacity.get("error", protection_capacity.get("status", "protection capacity unavailable")))
        log.error("[EXEC_BLOCKED_PROTECTION_PRECHECK] %s %s | %s", symbol, direction, reason)
        return {
            "status": "blocked_protection_preflight",
            "symbol": symbol,
            "direction": direction,
            "error": reason,
            "protection_preflight": protection_capacity,
        }

    # Revalidate against the actual BingX top-of-book immediately before MARKET.
    # The previous implementation validated the stale signal price and only
    # discovered changed risk/structure after the fill.
    execution_quote = get_execution_quote(symbol, reference_price=entry_price)
    if execution_quote.get("status") != "ok":
        reason = execution_quote.get("error", "execution quote unavailable")
        log.error("[EXEC_BLOCKED_QUOTE] %s %s | %s", symbol, direction, reason)
        return {"status": "execution_quote_unavailable", "error": reason, "symbol": symbol, "direction": direction}

    executable_price = float(execution_quote["ask"] if direction == "LONG" else execution_quote["bid"])
    signal_drift_pct = _adverse_signal_drift_pct(entry_price, executable_price, direction)
    if signal_drift_pct > MAX_ENTRY_SLIPPAGE_PCT:
        reason = f"signal_drift_pct={signal_drift_pct:.4f}% > {MAX_ENTRY_SLIPPAGE_PCT:.4f}%"
        log.warning("[EXEC_REJECT_STALE] %s %s | %s | signal=%s executable=%s", symbol, direction, reason, entry_price, executable_price)
        return {
            "status": "skipped_stale_signal",
            "error": reason,
            "symbol": symbol,
            "direction": direction,
            "signal_price": entry_price,
            "pre_entry_bid": execution_quote.get("bid"),
            "pre_entry_ask": execution_quote.get("ask"),
            "execution_reference_price": executable_price,
            "signal_drift_pct": signal_drift_pct,
        }

    try:
        preflight_rebased = _rebase_protection_after_fill(signal, executable_price)
        preflight_rebased.setdefault("entry", executable_price)
        preflight_signal = _build_actual_signal_from_rebase(signal, preflight_rebased)
    except Exception as exc:
        reason = f"pre-entry protection rebase failed: {exc}"
        log.warning("[EXEC_REJECT_GEOMETRY] %s %s | %s", symbol, direction, reason)
        return {"status": "skipped_invalid_setup", "error": reason, "symbol": symbol, "direction": direction}

    valid, reason = _validate_trade_geometry(preflight_signal)
    if not valid:
        log.warning("[EXEC_REJECT_GEOMETRY] %s %s | %s", symbol, direction, reason)
        return {
            "status": "skipped_invalid_setup",
            "error": f"pre_entry_current_price_geometry: {reason}",
            "symbol": symbol,
            "direction": direction,
            "signal_price": entry_price,
            "execution_reference_price": executable_price,
            "signal_drift_pct": signal_drift_pct,
        }
    distinct, reason = _validate_exchange_price_distinctness(preflight_signal)
    if not distinct:
        log.warning("[EXEC_REJECT_PRECISION] %s %s | %s", symbol, direction, reason)
        return {"status": "skipped_invalid_setup", "error": reason, "symbol": symbol, "direction": direction}

    setup = _build_setup(signal)
    order_submit_ts = pd.Timestamp.now(tz="UTC")
    signal["order_submit_ts"] = order_submit_ts.isoformat()
    order = open_market(symbol, direction, entry_price, event_id, execution_quote=execution_quote)
    if order.get("status") == "skipped_min_qty":
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

    fill_observed_ts = pd.Timestamp.now(tz="UTC")
    signal["fill_observed_ts"] = fill_observed_ts.isoformat()
    avg_price = float(position["avgPrice"])
    qty = abs(float(position["positionAmt"]))
    pre_entry_bid = float(execution_quote["bid"])
    pre_entry_ask = float(execution_quote["ask"])
    executable_reference_price = pre_entry_ask if direction == "LONG" else pre_entry_bid
    signed_entry_slippage_pct = None
    try:
        signed_entry_slippage_pct = (avg_price - executable_reference_price) / executable_reference_price * 100.0
        execution_slippage_pct = (
            max(0.0, avg_price - pre_entry_ask) / pre_entry_ask * 100.0
            if direction == "LONG"
            else max(0.0, pre_entry_bid - avg_price) / pre_entry_bid * 100.0
        )
    except (TypeError, ValueError, ZeroDivisionError):
        signed_entry_slippage_pct = float("inf")
        execution_slippage_pct = float("inf")

    if execution_slippage_pct > MAX_ENTRY_SLIPPAGE_PCT:
        reason = f"execution_slippage_pct={execution_slippage_pct:.4f}% > {MAX_ENTRY_SLIPPAGE_PCT:.4f}%"
        log.critical("[SAFETY_CLOSE] %s %s | %s", symbol, direction, reason)
        cleanup = _cancel_engine_protection_before_emergency_close(symbol, direction)
        close_result = _emergency_close_and_verify(symbol, direction, qty, event_id)
        return {
            "status": "opened_then_emergency_closed",
            "error": reason,
            "order": order,
            "position": position,
            "close": close_result,
            "protection_cleanup": cleanup,
            "signal_price": entry_price,
            "pre_entry_bid": pre_entry_bid,
            "pre_entry_ask": pre_entry_ask,
            "execution_reference_price": executable_reference_price,
            "signal_drift_pct": signal_drift_pct,
            "signed_entry_slippage_pct": signed_entry_slippage_pct,
            "execution_slippage_pct": execution_slippage_pct,
            "executed_signal": dict(signal),
        }

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
    actual_signal = _build_actual_signal_from_rebase(signal, rebased)
    distinct, precision_reason = _validate_exchange_price_distinctness(actual_signal)
    if not distinct:
        log.critical("[SAFETY_CLOSE] %s %s | invalid exchange-rounded protection geometry | %s", symbol, direction, precision_reason)
        cleanup = _cancel_engine_protection_before_emergency_close(symbol, direction)
        close_result = _emergency_close_and_verify(symbol, direction, qty, event_id)
        return {"status": "opened_then_emergency_closed", "error": precision_reason, "order": order, "position": position, "close": close_result, "protection_cleanup": cleanup, "executed_signal": actual_signal}
    valid, reason = _validate_trade_geometry(actual_signal)
    if not valid:
        log.critical("[SAFETY_CLOSE] %s %s | invalid post-fill protection geometry | %s", symbol, direction, reason)
        cleanup = _cancel_engine_protection_before_emergency_close(symbol, direction)
        close_result = _emergency_close_and_verify(symbol, direction, qty, event_id)
        return {"status": "opened_then_emergency_closed", "error": reason, "order": order, "position": position, "close": close_result, "protection_cleanup": cleanup, "executed_signal": actual_signal}

    setup["entry_reference"] = avg_price
    setup["signal_price"] = entry_price
    setup["pre_entry_bid"] = pre_entry_bid
    setup["pre_entry_ask"] = pre_entry_ask
    setup["execution_reference_price"] = executable_reference_price
    setup["signal_drift_pct"] = signal_drift_pct
    setup["execution_slippage_pct"] = execution_slippage_pct
    setup["invalidation_price"] = sl_price
    setup["risk_pct"] = actual_risk_pct
    setup["target_rr"] = actual_signal["tp2_rr"]
    setup["planned_weighted_rr"] = actual_signal["tp1_rr"] * 0.50 + actual_signal["tp2_rr"] * 0.50
    setup["tp_levels"] = [
        {"leg": "tp1", "pnl_pct": tp1_pnl_pct, "close_fraction": 0.50, "price": tp1_price},
        {"leg": "tp2", "pnl_pct": tp2_pnl_pct, "close_fraction": 0.50, "price": tp2_price},
    ]
    setup["target_price"] = tp2_price
    setup["entry_order"] = order if isinstance(order, dict) else {}
    setup["fill_position"] = position if isinstance(position, dict) else {}
    setup["execution_snapshot"] = {
        "strategy_version": _effective_strategy_version(),
        "signal_entry": entry_price,
        "requested_entry": entry_price,
        "pre_entry_bid": pre_entry_bid,
        "pre_entry_ask": pre_entry_ask,
        "execution_reference_price": executable_reference_price,
        "signal_drift_pct": signal_drift_pct,
        "fill_price": avg_price,
        "entry_slippage_pct": signed_entry_slippage_pct,
        "signed_entry_slippage_pct": signed_entry_slippage_pct,
        "execution_slippage_pct": execution_slippage_pct,
        "adverse_entry_slippage_pct": execution_slippage_pct,
        "sl_price": sl_price,
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        "stop_pct": actual_risk_pct,
        "tp1_pct": tp1_pnl_pct,
        "tp2_pct": tp2_pnl_pct,
        "tp1_fraction": 0.50,
        "tp2_fraction": 0.50,
        "be_rule": "after_tp1_filled",
        "quote_source": execution_quote.get("quote_source"),
        "quote_sources_attempted": execution_quote.get("quote_sources_attempted"),
        "quote_fallback_reason": execution_quote.get("quote_fallback_reason"),
        "quote_time": execution_quote.get("time"),
        "execution_started_ts": signal.get("execution_started_ts"),
        "order_submit_ts": signal.get("order_submit_ts"),
        "fill_observed_ts": signal.get("fill_observed_ts"),
        "protection_started_ts": signal.get("protection_started_ts"),
        "protection_finished_ts": signal.get("protection_finished_ts"),
        "trigger_to_order_minutes": _elapsed_minutes(signal.get("trigger_bar_time") or signal.get("time"), signal.get("order_submit_ts")),
        "order_to_fill_seconds": _elapsed_seconds(signal.get("order_submit_ts"), signal.get("fill_observed_ts")),
        "fill_to_protection_seconds": _elapsed_seconds(signal.get("fill_observed_ts"), signal.get("protection_started_ts")),
        "protection_seconds": _elapsed_seconds(signal.get("protection_started_ts"), signal.get("protection_finished_ts")),
    }
    setup["code_commit_sha"] = CODE_COMMIT_SHA

    log.info(
        "[EXEC_POST_FILL_REBASED] %s %s | fill=%s sl=%s tp1=%s tp2=%s tp1_rr=%.3f tp2_rr=%.3f target_source=%s",
        symbol, direction, avg_price, sl_price, tp1_price, tp2_price,
        actual_signal["tp1_rr"], actual_signal["tp2_rr"], rebased["target_source"],
    )

    protection_started_ts = pd.Timestamp.now(tz="UTC")
    signal["protection_started_ts"] = protection_started_ts.isoformat()
    protection = ensure_directional_protection(
        symbol, direction, avg_price, qty,
        actual_risk_pct, setup["tp_levels"], trade_id=event_id,
    )
    protection_finished_ts = pd.Timestamp.now(tz="UTC")
    signal["protection_finished_ts"] = protection_finished_ts.isoformat()
    setup["execution_snapshot"]["protection_status"] = protection.get("status")
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
        event_type=f"{setup['zone'].get('kind', 'ZONE')}_{'MIDPOINT_TOUCH_5M' if ZONE_TRIGGER_MODE == 'midpoint' else 'ZONE_TOUCH_5M'}",
        timeframe="5m",
        score=float(signal.get("score", 0.0)),
        setup={**setup, "protection_status": protection.get("status"), "protection_result": protection},
        requested_entry_price=entry_price,
    )

    return {
        "status": "opened_protected",
        "order": order,
        "position": position,
        "protection": protection,
        "execution_snapshot": {
            "strategy_version": _effective_strategy_version(),
            "signal_entry": entry_price,
            "requested_entry": entry_price,
            "pre_entry_bid": pre_entry_bid,
            "pre_entry_ask": pre_entry_ask,
            "execution_reference_price": executable_reference_price,
            "signal_drift_pct": signal_drift_pct,
            "fill_price": avg_price,
            "entry_slippage_pct": signed_entry_slippage_pct,
            "signed_entry_slippage_pct": signed_entry_slippage_pct,
            "execution_slippage_pct": execution_slippage_pct,
            "adverse_entry_slippage_pct": execution_slippage_pct,
            "sl_price": sl_price,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "stop_pct": actual_risk_pct,
            "tp1_pct": tp1_pnl_pct,
            "tp2_pct": tp2_pnl_pct,
            "tp1_fraction": 0.50,
            "tp2_fraction": 0.50,
            "be_rule": "after_tp1_filled",
            "quote_source": execution_quote.get("quote_source"),
            "quote_sources_attempted": execution_quote.get("quote_sources_attempted"),
            "quote_fallback_reason": execution_quote.get("quote_fallback_reason"),
            "quote_time": execution_quote.get("time"),
            "execution_started_ts": signal.get("execution_started_ts"),
            "order_submit_ts": signal.get("order_submit_ts"),
            "fill_observed_ts": signal.get("fill_observed_ts"),
            "protection_started_ts": signal.get("protection_started_ts"),
            "protection_finished_ts": signal.get("protection_finished_ts"),
            "trigger_to_order_minutes": _elapsed_minutes(signal.get("trigger_bar_time") or signal.get("time"), signal.get("order_submit_ts")),
            "order_to_fill_seconds": _elapsed_seconds(signal.get("order_submit_ts"), signal.get("fill_observed_ts")),
            "fill_to_protection_seconds": _elapsed_seconds(signal.get("fill_observed_ts"), signal.get("protection_started_ts")),
            "protection_seconds": _elapsed_seconds(signal.get("protection_started_ts"), signal.get("protection_finished_ts")),
        },
        "executed_signal": actual_signal,
    }

def _send_signal(signal: dict[str, Any], execution: dict[str, Any] | None = None) -> None:
    # Entry notifications are sent only after a position is actually opened
    # and mandatory SL/TP protection has been verified.
    if not isinstance(execution, dict) or str(execution.get("status", "")) != "opened_protected":
        return
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
    _init_diagnostic_log()
    log.info(
        "[RUN_DIAGNOSTIC] scan_id=%s strategy=%s | universe=%s | trigger=5m_%s | zone_tf=1h | stop=%.2f%% | tp1=%.2f%% | tp2=%.2f%% | be=after_tp1_filled | max_5m_age_min=%.1f | structure_min=%.2fR | data_stale_hours=%.2f",
        scan_id, (STRATEGY_VERSION if ZONE_TRIGGER_MODE == "midpoint" else f"{STRATEGY_VERSION}-zone-touch-test"), ("150" if FUNDAMENTAL_WHITELIST_ENABLED else "ALL_ACTIVE_BINGX"), ZONE_TRIGGER_MODE, FIXED_STOP_PCT, TP1_PCT, TP2_PCT,
        MAX_5M_TRIGGER_AGE_MINUTES, MIN_STRUCTURE_ROOM_R, MAX_DATA_STALENESS_HOURS,
    )
    research.update_manifest_run(
        scan_id=scan_id, code_commit_sha=CODE_COMMIT_SHA,
        effective_config={
            "execution_enabled": EXECUTION_ENABLED, "fundamental_whitelist_enabled": FUNDAMENTAL_WHITELIST_ENABLED,
            "zone_trigger_mode": ZONE_TRIGGER_MODE, "fixed_stop_pct": FIXED_STOP_PCT,
            "tp1_pct": TP1_PCT, "tp2_pct": TP2_PCT, "min_structure_room_r": MIN_STRUCTURE_ROOM_R,
            "require_directional_candle": REQUIRE_DIRECTIONAL_CANDLE, "require_structure_obstacle": REQUIRE_STRUCTURE_OBSTACLE,
            "max_trades_per_cycle": MAX_TRADES_PER_CYCLE, "max_entry_slippage_pct": MAX_ENTRY_SLIPPAGE_PCT,
            "max_market_spread_pct": MAX_MARKET_SPREAD_PCT, "max_5m_trigger_age_minutes": MAX_5M_TRIGGER_AGE_MINUTES,
        },
    )

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
        log.info("[PRIVATE] BingX private layer unavailable | reconciliation/execution disabled")

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

    # Research-only account context is fetched once per scan and reused by all observations.
    research_account_context = None
    if research.RESEARCH_ENABLED:
        try:
            research_account_context = fetch_research_account_snapshot()
            if research_account_context is not None:
                research_account_context["scan_id"] = scan_id
                research_account_context["strategy_version"] = _effective_strategy_version()
                research_account_context["code_commit_sha"] = CODE_COMMIT_SHA
                research_account_context["account_context_id"] = research_account_context.get("account_context_id") or research.stable_id(
                    "account-context-v1", scan_id, research_account_context.get("captured_at_ms", ""), prefix="AC_"
                )
                research_account_context["persisted"] = bool(research.record_account_context(research_account_context))
        except Exception as exc:
            log.warning("[RESEARCH_ACCOUNT] account context unavailable: %s", exc)

    # Research-only BTC context is fetched once per scan and reused by all symbol observations.
    btc_research_df_5m = pd.DataFrame()
    btc_research_df_1h = pd.DataFrame()
    if research.RESEARCH_ENABLED:
        try:
            btc_raw_5m = fetch_binance_klines("BTCUSDT", interval="5m", limit=288, retryable=False)
            if btc_raw_5m:
                btc_research_df_5m = pd.DataFrame(btc_raw_5m)
                research.persist_market_bars("BTC-USDT", "5m", btc_raw_5m, provider="binance", source="binance_spot_btc_context", scan_id=scan_id, code_commit_sha=CODE_COMMIT_SHA)
        except Exception as exc:
            log.warning("[RESEARCH_BTC] 5m context unavailable: %s", exc)
        try:
            btc_raw_1h = fetch_binance_klines("BTCUSDT", interval="1h", limit=300, retryable=False)
            if btc_raw_1h:
                btc_research_df_1h = pd.DataFrame(btc_raw_1h)
                research.persist_market_bars("BTC-USDT", "1h", btc_raw_1h, provider="binance", source="binance_spot_btc_context", scan_id=scan_id, code_commit_sha=CODE_COMMIT_SHA)
        except Exception as exc:
            log.warning("[RESEARCH_BTC] 1h context unavailable: %s", exc)

    research_pending_forward_symbols: set[str] = set()
    if research.RESEARCH_ENABLED:
        try:
            research_pending_forward_symbols = research.pending_forward_symbols(horizon_hours=24.0)
        except Exception as exc:
            log.warning("[RESEARCH_PENDING_SYMBOLS] unable to load pending symbols: %s", exc)

    symbols = [str(item["bingx_symbol"]) for item in analysis_universe]
    analysis_meta = {str(item["bingx_symbol"]): item for item in analysis_universe}
    crypto_n = sum(1 for x in analysis_universe if str(x.get("asset_class")).upper() == "CRYPTO")
    equity_n = sum(1 for x in analysis_universe if str(x.get("asset_class")).upper() == "EQUITY")
    log.info("[SCAN] Eligible symbols: %d | crypto=%d equity=%d | strategy_mode=ZONE_1H_5M_%s | universe_mode=%s | diagnostics_mode=%s", len(symbols), crypto_n, equity_n, ZONE_TRIGGER_MODE.upper(), "FUNDAMENTAL_150" if FUNDAMENTAL_WHITELIST_ENABLED else "ALL_ACTIVE_BINGX", DIAGNOSTICS_MODE)
    if not symbols:
        log.error("[SCAN] No eligible symbols for signal scan")

    successful_ids = _load_successful_trade_ids()
    terminal_event_ids = _load_terminal_event_ids()
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
    if open_keys:
        log.info("[POSITIONS] open=%s", json.dumps([f"{_display_symbol(sym)}:{direction}" for sym, direction in sorted(open_keys)], ensure_ascii=False))
    zone_visit_state = _load_zone_visit_state()
    if RESET_ZONE_VISIT_STATE_ON_START:
        log.warning("[ZONE_STATE] RESET_ZONE_VISIT_STATE_ON_START=true | starting the 5m visit state empty for this test run")
        zone_visit_state = {"version": ZONE_VISIT_STATE_VERSION, "trigger_mode": ZONE_TRIGGER_MODE, "symbols": {}}
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
            research.persist_market_bars(symbol, "1h", bars, provider=provider, source=source_name, scan_id=scan_id, code_commit_sha=CODE_COMMIT_SHA)
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

            # 1H remains the zone-construction timeframe. The actual execution trigger
            # is detected from CLOSED 5m candles so a short intrahour midpoint touch is
            # observable to a 5-minute cron. Zone-visit state prevents repeated entries
            # while price chops around the same zone.
            zone_signals: list[dict[str, Any]] = []
            trigger_bars_raw: list[dict[str, Any]] = []
            symbol_state = zone_visit_state.get("symbols", {}).get(symbol, {}) if isinstance(zone_visit_state.get("symbols"), dict) else {}
            five_min_diag: dict[str, Any] = {
                "decision": {
                    "status": "NO_ZONE_CHECK",
                    "zone_key": None,
                    "direction": None,
                    "reason": "no_active_zones",
                    "timestamp": None,
                },
                "processed_bars": 0,
                "latest_closed_5m_ts": None,
            }
            if demand or supply:
                trigger_limit = max(24, KLINE_LIMIT_5M)
                if provider == "bingx":
                    trigger_bars_raw = fetch_bingx_klines(symbol, "5m", limit=trigger_limit, retryable=False)
                else:
                    trigger_bars_raw = fetch_binance_klines(binance_symbol, "5m", limit=trigger_limit, retryable=False)
                five_min_diag["current_1h_idx"] = int(latest_closed_idx)
                zone_signals, symbol_state, trigger_text = _process_5m_zone_visits(
                    symbol=symbol,
                    bars=trigger_bars_raw,
                    demand=demand,
                    supply=supply,
                    df_1h=df,
                    state_for_symbol=symbol_state,
                    successful_ids=successful_ids,
                    terminal_event_ids=terminal_event_ids,
                    diagnostics=five_min_diag,
                )
                _log_5m_zone_diagnostics(
                    symbol=symbol, latest_price=latest_price, latest_closed_1h_time=latest_closed_time,
                    demand=demand, supply=supply, diagnostics=five_min_diag, symbol_state=symbol_state,
                    pending_signal_count=len(zone_signals),
                )
                zone_visit_state.setdefault("symbols", {})[symbol] = symbol_state
            else:
                # A symbol may have a still-maturing research observation even after
                # its zone disappears. Keep persisting new 5m bars for that symbol so
                # the 24h forward path can actually mature. This is research-only.
                if research.RESEARCH_ENABLED and str(symbol).upper() in research_pending_forward_symbols:
                    try:
                        pending_limit = max(12, int(os.environ.get("RESEARCH_PENDING_5M_FETCH_LIMIT", "24")))
                        if provider == "bingx":
                            trigger_bars_raw = fetch_bingx_klines(symbol, "5m", limit=pending_limit, retryable=False)
                        else:
                            trigger_bars_raw = fetch_binance_klines(binance_symbol, "5m", limit=pending_limit, retryable=False)
                    except Exception as pending_exc:
                        log.warning("[RESEARCH_PENDING_BARS] %s | 5m fetch unavailable: %s", _display_symbol(symbol), pending_exc)
                zone_visit_state.setdefault("symbols", {})[symbol] = {
                    "version": ZONE_VISIT_STATE_VERSION,
                    "zones": {},
                    "last_scan_ts": pd.Timestamp.now(tz="UTC").isoformat(),
                }
                trigger_text = None

            recent = zone_signals
            for sig in recent:
                sig["score"] = score_zone_signal(sig)
                sig["market_snapshot"] = {
                    "analysis_provider": provider,
                    "analysis_source": source_name,
                    "asset_class": meta.get("asset_class", "UNKNOWN"),
                    "binance_symbol": binance_symbol,
                    "analysis_last_price": latest_price,
                    "analysis_latest_closed_time": latest_closed_time.isoformat(),
                    "trigger_timeframe": "5m",
                    "trigger_bar_time": sig.get("trigger_bar_time"),
                    "zone_visit_id": (sig.get("zone_visit") or {}).get("visit_id"),
                }

            # Keep an immutable copy for research even when the later market gate rejects
            # the signal. Research must observe the full candidate population.
            research_signals = list(recent)

            # Only validate BingX live price when a fresh 5m trigger exists. This keeps
            # the full-market scan on Binance while spending extra venue requests only
            # on actionable candidates.
            if recent and bingx_price is None:
                try:
                    bx_live = fetch_bingx_klines(symbol, "1m", limit=1, retryable=False)
                    if bx_live:
                        bingx_price = float(bx_live[-1]["close"])
                except Exception as bx_exc:
                    log.warning("[MARKET_CHECK] %s | BingX price validation failed: %s", symbol, bx_exc)
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
            ) if provider == "binance" else None
            for sig in recent:
                market_snapshot = sig.setdefault("market_snapshot", {})
                trigger_ts = pd.Timestamp(sig.get("trigger_bar_time")) if sig.get("trigger_bar_time") else pd.Timestamp.now(tz="UTC")
                market_snapshot.update({
                    "bingx_live_price": bingx_price,
                    "binance_live_price": binance_live_price,
                    "market_spread_pct": spread_pct,
                    "venue_price_deviation_pct": spread_pct,
                    "market_spread_metric": "cross_venue_price_deviation",
                    "scan_time": pd.Timestamp.now(tz="UTC").isoformat(),
                    "trigger_age_minutes_at_scan": max(0.0, (pd.Timestamp.now(tz="UTC") - trigger_ts).total_seconds() / 60.0),
                })

            # Research market snapshot is captured only for symbols that reached at least one
            # observation boundary. It is best-effort and never gates production execution.
            # Raw market bars are persisted only for boundary symbols so a 5-minute universe scan
            # does not grow the journal with irrelevant 5m history.
            research_context = None
            research_bars_5m_raw = []
            research_bars_1h_raw = []
            touch_events = five_min_diag.get("touch_events") or []
            rearm_events = five_min_diag.get("rearm_events") or []
            near_approach_limit = max(0.0, float(os.environ.get("RESEARCH_NEAREST_APPROACH_MAX_PCT", "1.0")))
            has_nearest = any(
                isinstance(z, dict)
                and isinstance(z.get("closest_midpoint_bar"), dict)
                and z.get("closest_midpoint_distance_pct") is not None
                and 0.0 < float(z.get("closest_midpoint_distance_pct")) <= near_approach_limit
                for z in (five_min_diag.get("zones") or {}).values()
            )
            rich_nearest_context = os.environ.get("RESEARCH_CONTEXT_NEAREST_APPROACH", "false").lower() == "true"
            rich_rejection_context = os.environ.get("RESEARCH_CONTEXT_FOR_REJECTIONS", "false").lower() == "true"
            boundary_types: list[str] = []
            if research_signals:
                boundary_types.append("SIGNAL_CREATED")
            if touch_events:
                boundary_types.extend(sorted({str(e.get("reason", "TOUCH_EVENT")) for e in touch_events if isinstance(e, dict)}))
            if rearm_events:
                boundary_types.append("REARM")
            # Persist the 5m path for a qualifying nearest approach even when the
            # expensive order-book/funding/OI context is deliberately disabled.
            if has_nearest:
                boundary_types.append("NEAREST_APPROACH")
            has_research_boundary = bool(boundary_types)
            if research.RESEARCH_ENABLED and has_research_boundary:
                research_bars_1h_raw = bars
                research_bars_5m_raw = trigger_bars_raw
            collect_rich_context = bool(research_signals) or (rich_rejection_context and bool(touch_events or rearm_events))
            if research.RESEARCH_ENABLED and has_research_boundary and collect_rich_context:
                # Pull a larger 5m history only for research-boundary symbols. This gives
                # enough lookback for 12h/24h returns without increasing the production
                # trigger fetch for the entire universe.
                try:
                    research_limit = max(
                        len(trigger_bars_raw),
                        int(os.environ.get("RESEARCH_5M_HISTORY_LIMIT", "288")),
                    )
                    if research_limit > len(trigger_bars_raw):
                        if provider == "bingx":
                            research_bars_5m_raw = fetch_bingx_klines(symbol, "5m", limit=research_limit, retryable=False)
                        else:
                            research_bars_5m_raw = fetch_binance_klines(binance_symbol, "5m", limit=research_limit, retryable=False)
                except Exception as bar_exc:
                    log.warning("[RESEARCH_BARS] %s | extended 5m history unavailable: %s", _display_symbol(symbol), bar_exc)
                try:
                    research_context = fetch_research_market_context(
                        symbol,
                        depth_limit=max(5, int(os.environ.get("RESEARCH_DEPTH_LEVELS", "20"))),
                        trades_limit=max(10, int(os.environ.get("RESEARCH_RECENT_TRADES_LIMIT", "100"))),
                    )
                    if research_context is not None:
                        research_context.update({
                            "asset_class": meta.get("asset_class", "UNKNOWN"),
                            "analysis_provider": provider,
                            "analysis_source": source_name,
                            "context_provider": "bingx",
                            "context_source": "bingx_swap_public",
                            "binance_symbol": binance_symbol,
                            "cross_venue_binance_price": binance_live_price,
                            "cross_venue_bingx_price": bingx_price,
                            "cross_venue_deviation_pct": spread_pct,
                            "cross_venue_metric": "binance_bingx_last_price_deviation",
                            "boundary_types": sorted(set(boundary_types)),
                            "boundary_event_ids": [str(s.get("event_id")) for s in research_signals if s.get("event_id")],
                        })
                except Exception as context_exc:
                    log.warning("[RESEARCH_CONTEXT] %s | collection failed: %s", _display_symbol(symbol), context_exc)

            # Research is persisted after all pre-entry market-context fields are known,
            # but before any production gate can remove a candidate from the population.
            research.record_scan_symbol(
                scan_id=scan_id, symbol=symbol, strategy_version=_effective_strategy_version(),
                code_commit_sha=CODE_COMMIT_SHA, provider=provider, source=source_name,
                bars_1h=research_bars_1h_raw, bars_5m=research_bars_5m_raw, df_1h=df, demand=demand, supply=supply,
                diagnostics=five_min_diag, symbol_state=symbol_state, signals=research_signals,
                decision_ts=pd.Timestamp.now(tz="UTC").isoformat(), market_context=research_context,
                persist_bars_for_forward=(str(symbol).upper() in research_pending_forward_symbols),
                account_context=research_account_context, btc_df_5m=btc_research_df_5m, btc_df_1h=btc_research_df_1h,
            )

            if spread_pct is not None and spread_pct > MAX_MARKET_SPREAD_PCT:
                log.warning("[MARKET_SPREAD] %s | Binance=%s | BingX=%s | spread=%.4f%% > %.4f%%", _display_symbol(symbol), binance_live_price, bingx_price, spread_pct, MAX_MARKET_SPREAD_PCT)
                recent = []
                fresh_text = f"BLOCKED_SPREAD>{MAX_MARKET_SPREAD_PCT:.2f}%"
            else:
                latest_signal = _select_latest_signal(recent)
                fresh_text = trigger_text or (
                    f"{latest_signal['type']} @ {latest_signal['entry']} score={latest_signal.get('score', 0):.1f} 5m_{ZONE_TRIGGER_MODE}_touch"
                    if latest_signal else "—"
                )
            price_position = _price_position(latest_price, demand, supply)

            return {
                "symbol": symbol,
                "current_price": latest_price,
                "binance_price": latest_price,
                "bingx_price": bingx_price,
                "market_spread_pct": spread_pct,
                "venue_price_deviation_pct": spread_pct,
                "market_spread_metric": "cross_venue_price_deviation",
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
                "trigger_status": str((five_min_diag.get("decision") or {}).get("status") or "NO_ZONE_CHECK"),
                "trigger_reason": str((five_min_diag.get("decision") or {}).get("reason") or ""),
                "trigger_zone_key": (five_min_diag.get("decision") or {}).get("zone_key"),
                "trigger_direction": (five_min_diag.get("decision") or {}).get("direction"),
                "trigger_bar_time": (five_min_diag.get("decision") or {}).get("timestamp"),
                "signals_created_5m": len(zone_signals),
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

        scanned = min(batch_start + len(batch), total)
        if scanned < total and SCAN_BATCH_PAUSE_SEC > 0:
            time.sleep(SCAN_BATCH_PAUSE_SEC)

    scan_errors = sum(1 for row in scan_rows if row.get("price_position") == "ERROR")
    scan_insufficient_data = sum(1 for row in scan_rows if row.get("price_position") == "INSUFFICIENT_DATA")
    scan_contract_not_found = sum(1 for row in scan_rows if row.get("price_position") == "CONTRACT_NOT_FOUND")
    scan_stale_rejects = sum(
        1 for row in scan_rows
        if str(row.get("trigger_status", "")) in {"BLOCKED_STALE_TOUCH", "DATA_STALE_REJECT"}
        or "DATA_STALE_REJECT" in str(row.get("fresh_signal", ""))
    )
    scan_skips = scan_insufficient_data + scan_contract_not_found + scan_stale_rejects
    scan_duration_sec = time.time() - started
    log.info(
        "[SCAN_DONE] symbols=%d errors=%d skipped=%d insufficient_data=%d contract_not_found=%d stale_rejects=%d fresh_signals=%d duration=%.1fs",
        total, scan_errors, scan_skips, scan_insufficient_data, scan_contract_not_found, scan_stale_rejects, len(fresh_signals), scan_duration_sec,
    )

    # Normalize execution-facing metadata once at the runtime boundary. The
    # signal builder keeps the base strategy version for backwards-compatible
    # analytics, while the live workflow may select a trigger-mode variant.
    effective_version = _effective_strategy_version()
    for sig in fresh_signals:
        sig["strategy_version"] = effective_version
        sig["code_commit_sha"] = CODE_COMMIT_SHA

    # Persist visit locks before any exchange execution. A process crash after scanning
    # must not make the same midpoint visit eligible again on the next cron run.
    _save_zone_visit_state(zone_visit_state)

    # Execution safety: choose exactly ONE newest 5m visit event per symbol.
    latest_by_symbol: dict[str, dict[str, Any]] = {}
    for signal in fresh_signals:
        symbol_key = str(signal["symbol"]).upper()
        previous = latest_by_symbol.get(symbol_key)
        candidate_key = (int(signal.get("idx", 0)), float(signal.get("score", 0.0)))
        previous_key = (int(previous.get("idx", 0)), float(previous.get("score", 0.0))) if previous else None
        if previous is None or candidate_key > previous_key:
            latest_by_symbol[symbol_key] = signal
            if previous is not None:
                _record_entry_decision(
                    scan_id,
                    previous,
                    "SIGNAL_DEDUPLICATED",
                    "newer_signal_same_symbol_selected",
                )
        else:
            _record_entry_decision(
                scan_id,
                signal,
                "SIGNAL_DEDUPLICATED",
                "older_signal_same_symbol_not_selected",
            )

    executable: list[dict[str, Any]] = []
    exec_gate_stats: dict[str, int] = {}

    def _exec_gate(reason: str, signal: dict[str, Any]) -> None:
        exec_gate_stats[reason] = exec_gate_stats.get(reason, 0) + 1
        _record_entry_decision(scan_id, signal, "EXECUTION_GATE_REJECT", reason)
        log.info(
            "[EXEC_GATE] %s | direction=%s | event_id=%s | gate=%s",
            _display_symbol(signal.get("symbol")), signal.get("type"), signal.get("event_id"), reason,
        )

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
    now_exec = pd.Timestamp.now(tz="UTC")
    for signal in latest_by_symbol.values():
        symbol_key = str(signal["symbol"]).upper()
        bx = get_contract(signal["symbol"])
        bx_symbol = str((bx or {}).get("symbol", signal["symbol"])).upper()
        key = (bx_symbol, signal["type"])
        opposite = (bx_symbol, "SHORT" if signal["type"] == "LONG" else "LONG")
        event_id = str(signal["event_id"])
        failed_record = failed_ids.get(event_id)
        if event_id in successful_ids:
            _exec_gate("already_successful_event", signal)
            continue
        if event_id in terminal_event_ids:
            _exec_gate("terminal_event", signal)
            continue
        if _failed_signal_is_blocked(failed_record):
            _exec_gate("failed_signal_backoff_or_terminal", signal)
            continue
        if key in open_keys or opposite in open_keys:
            _exec_gate("existing_open_position_same_or_opposite", signal)
            continue

        trigger_tf = str(signal.get("trigger_timeframe", "1h")).lower()
        if trigger_tf == "5m":
            try:
                trigger_ts = pd.Timestamp(signal.get("trigger_bar_time") or signal.get("time"))
                if trigger_ts.tzinfo is None:
                    trigger_ts = trigger_ts.tz_localize("UTC")
                age_min = max(0.0, (now_exec - trigger_ts).total_seconds() / 60.0)
            except Exception:
                log.warning("[EXEC_REJECT_TRIGGER_TIME] %s %s | invalid 5m trigger time=%s", _display_symbol(signal.get("symbol")), signal.get("type"), signal.get("trigger_bar_time"))
                _exec_gate("invalid_5m_trigger_time", signal)
                continue
            signal["execution_age_minutes"] = age_min
            if age_min > MAX_5M_TRIGGER_AGE_MINUTES:
                log.info("[EXEC_REJECT_5M_AGE] %s %s | age_min=%.2f allowed=%.2f", _display_symbol(signal["symbol"]), signal["type"], age_min, MAX_5M_TRIGGER_AGE_MINUTES)
                _exec_gate("5m_trigger_too_old", signal)
                continue
        else:
            latest_closed_idx = latest_index_by_symbol.get(symbol_key)
            latest_closed_time = latest_time_by_symbol.get(symbol_key)
            if latest_closed_idx is None:
                _exec_gate("missing_latest_closed_1h_idx", signal)
                continue
            signal_age = int(latest_closed_idx) - int(signal["idx"])
            signal["execution_age_bars"] = int(signal_age)
            signal["latest_closed_idx"] = int(latest_closed_idx)
            signal["latest_closed_time"] = latest_closed_time
            if signal_age > EXECUTION_MAX_SIGNAL_AGE_BARS:
                _exec_gate("1h_signal_age_exceeded", signal)
                continue
            matches_latest, reject_reason = _signal_matches_latest_bar(signal, latest_closed_idx, latest_closed_time)
            if not matches_latest:
                log.warning("[EXEC_REJECT_LATEST_BAR] %s %s | reason=%s", _display_symbol(signal["symbol"]), signal["type"], reject_reason)
                _exec_gate(f"latest_bar_mismatch:{reject_reason}", signal)
                continue
        executable.append(signal)

    log.info("[EXEC_INPUT] fresh_signals=%d unique_symbols=%d open_positions=%d successful_event_ids=%d terminal_event_ids=%d failed_signal_records=%d", len(fresh_signals), len(latest_by_symbol), len(open_keys), len(successful_ids), len(terminal_event_ids), len(failed_ids))

    # Safety ordering: newest signal bar first; score only breaks ties.
    executable.sort(key=lambda x: (-int(x["idx"]), -float(x.get("score", 0.0))))
    log.info("[EXEC_GATE_SUMMARY] input_signals=%d executable=%d rejected=%d reasons=%s", len(latest_by_symbol), len(executable), sum(exec_gate_stats.values()), json.dumps(exec_gate_stats, ensure_ascii=False, sort_keys=True))

    executed = 0
    selected_signals = executable[:MAX_TRADES_PER_CYCLE]
    cycle_cap_signals = executable[MAX_TRADES_PER_CYCLE:]
    for rank, signal in enumerate(cycle_cap_signals, start=MAX_TRADES_PER_CYCLE + 1):
        _record_entry_decision(
            scan_id,
            signal,
            "CYCLE_CAP",
            "max_trades_per_cycle_reached",
            selection_rank=rank,
        )
        log.warning(
            "[EXEC_CYCLE_CAP] symbol=%s direction=%s event_id=%s rank=%d cap=%d score=%.2f trigger=%s",
            signal.get("symbol"), signal.get("type"), signal.get("event_id"), rank,
            MAX_TRADES_PER_CYCLE, signal.get("score", 0.0), signal.get("trigger_bar_time") or signal.get("time"),
        )

    for rank, signal in enumerate(selected_signals, start=1):
        execution_gate_ts = pd.Timestamp.now(tz="UTC")
        signal["execution_gate_ts"] = execution_gate_ts.isoformat()
        trigger_raw = signal.get("trigger_bar_time") or signal.get("time")
        try:
            trigger_ts = pd.Timestamp(trigger_raw)
            if trigger_ts.tzinfo is None:
                trigger_ts = trigger_ts.tz_localize("UTC")
            signal["trigger_to_execution_gate_minutes"] = max(0.0, (execution_gate_ts - trigger_ts).total_seconds() / 60.0)
        except Exception:
            signal["trigger_to_execution_gate_minutes"] = None
        _record_entry_decision(
            scan_id,
            signal,
            "EXECUTION_SELECTED",
            "selected_within_cycle_cap",
            selection_rank=rank,
        )
        if not EXECUTION_ENABLED:
            _send_signal(signal, {"status": "DISABLED"})
            _record_entry_decision(scan_id, signal, "EXECUTION_RESULT", "execution_disabled", selection_rank=rank, execution_status="DISABLED")
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
            _record_entry_decision(scan_id, signal, "EXECUTION_RESULT", "missing_private_credentials", selection_rank=rank, execution_status=blocked.get("status"), terminal=True)
            _send_signal(signal, blocked)
            continue
        event_id = str(signal["event_id"])
        claimed, claim_reason, attempt_id = _claim_event_for_execution(event_id)
        if not claimed:
            _record_entry_decision(scan_id, signal, "EXECUTION_CLAIM", claim_reason, selection_rank=rank, execution_status="CLAIM_REJECTED", terminal=(claim_reason in {"terminal", "completed", "claim_error"}))
            log.info("[EXEC_SKIP_CLAIM] %s %s | event_id=%s reason=%s", signal["symbol"], signal["type"], event_id, claim_reason)
            continue
        _record_entry_decision(scan_id, signal, "EXECUTION_CLAIM", "claimed", selection_rank=rank, attempt_id=attempt_id, execution_status="CLAIMED", terminal=False)
        execution_call_start_ts = pd.Timestamp.now(tz="UTC")
        execution = execute_new_position(signal)
        execution_finish_ts = pd.Timestamp.now(tz="UTC")
        execution["attempt_id"] = attempt_id
        execution["execution_call_start_ts"] = execution_call_start_ts.isoformat()
        execution["execution_finish_ts"] = execution_finish_ts.isoformat()
        execution["execution_wall_clock_seconds"] = max(0.0, (execution_finish_ts - execution_call_start_ts).total_seconds())
        execution["trigger_to_execution_finish_minutes"] = _elapsed_minutes(signal.get("trigger_bar_time") or signal.get("time"), execution_finish_ts.isoformat())
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
                terminal_failure = _is_terminal_execution_failure(execution)
                _mark_failed_signal(
                    signal["event_id"], execution_status, execution.get("error", ""),
                    force_terminal=terminal_failure,
                )
            else:
                terminal_failure = False
        else:
            terminal_failure = False
        _record_entry_decision(
            scan_id,
            signal,
            "EXECUTION_RESULT",
            execution_status or "empty_execution_status",
            selection_rank=rank,
            attempt_id=attempt_id,
            execution_status=execution_status,
            terminal=(True if execution_status == "opened_protected" else bool(_is_terminal_execution_failure(execution))),
        )
        if execution_status == "opened_protected":
            _finalize_event_claim(event_id, attempt_id, terminal=True, status=execution_status)
        elif _is_terminal_execution_failure(execution):
            terminal_event_ids.add(event_id)
            _finalize_event_claim(event_id, attempt_id, terminal=True, status=execution_status)
        else:
            _finalize_event_claim(event_id, attempt_id, terminal=False, status=execution_status)
        _append_jsonl(TRADES_PATH, {
            "record_type": "TRADE_OPEN",
            "event_id": signal["event_id"],
            "attempt_id": execution.get("attempt_id"),
            "symbol": signal["symbol"],
            "direction": signal["type"],
            "score": signal["score"],
            "strategy_version": signal.get("strategy_version", _effective_strategy_version()),
            "code_commit_sha": CODE_COMMIT_SHA,
            "entry_rule": (signal.get("trigger") or {}).get("zone_entry_rule"),
            "zone_midpoint": signal.get("zone_midpoint") or (signal.get("trigger") or {}).get("zone_midpoint"),
            "stop_pct": (signal.get("risk_model") or {}).get("fixed_stop_pct"),
            "tp1_pct": (signal.get("target") or {}).get("tp1_pct"),
            "tp2_pct": (signal.get("target") or {}).get("tp2_pct"),
            "execution_age_minutes": signal.get("execution_age_minutes"),
            "trigger_to_execution_gate_minutes": signal.get("trigger_to_execution_gate_minutes"),
            "trigger_to_execution_finish_minutes": execution.get("trigger_to_execution_finish_minutes"),
            "be_rule": (signal.get("target") or {}).get("be_rule", "after_tp1_filled"),
            "entry_bar": signal.get("entry_bar", {}),
            "previous_bar": signal.get("previous_bar", {}),
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

    total_duration_sec = time.time() - started
    log.info("[DONE] symbols=%d fresh_signals=%d executed=%d duration=%.1fs", len(scan_rows), len(fresh_signals), executed, total_duration_sec)


if __name__ == "__main__":
    main()
