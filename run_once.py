from __future__ import annotations

import json
import math
import logging
import os
import time
import uuid
import fcntl
import hashlib
import inspect
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from event_engine.analytics import save_scan
from event_engine.binance import analysis_symbols_for_bingx, classify_bingx_contract, fetch_24h_ticker, fetch_klines as fetch_binance_klines
from event_engine.bingx import (
    contracts,
    credentials_available,
    ensure_directional_protection,
    fetch_klines as fetch_bingx_klines,
    get_contract,
    get_positions,
    get_order,
    get_order_by_identity,
    get_position_mode,
    get_position_directional,
    get_open_protection_directional,
    prepare_protection_capacity,
    get_execution_quote,
    OPEN_ORDERS_PATH,
    ORDER_PATH,
    POSITION_PATH,
    fetch_research_market_context,
    fetch_research_account_snapshot,
    cancel_order,
    close_position_market,
    open_market,
    wait_for_position_fill_directional,
)
from event_engine.signals import STRATEGY_VERSION, SWING_LEN, MIN_BARS as SIGNALS_MIN_BARS, TP1_PCT, TP2_PCT, generate_zone_signals, score_zone_signal, _nearest_opposing_level, _signal_forensics
from event_engine.version import ENGINE_VERSION
from event_engine.state_validation import validate_active_trade_record
from event_engine.fundamental_assets import FUNDAMENTAL_ASSET_SYMBOLS
from event_engine.telegram import format_signal, send as send_tg
from event_engine.tracker import ActiveTradeStateCorrupt, register_active_trade, update_active_trades, update_active_trade_protection, update_active_trade_entry_reconciliation, merge_reconciled_tp_orders, has_active_trade_conflict, backfill_active_trade_provenance, record_verified_emergency_close
from event_engine import research, telemetry, shadow, request_metrics
from event_engine.trend_filter import evaluate_trend_filter

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
EXECUTION_LEDGER_PATH = DATA / "execution_ledger.jsonl"
EVENT_CLAIMS_PATH = DATA / "event_execution_claims.json"
EVENT_CLAIMS_LOCK_PATH = DATA / "event_execution_claims.json.lock"
EVENT_CLAIM_LEASE_SEC = max(60, int(os.environ.get("EVENT_CLAIM_LEASE_SEC", "900")))

EXECUTION_ENABLED = os.environ.get("EXECUTION_ENABLED", "true").lower() == "true"
MARGIN_USDT = float(os.environ.get("BINGX_MARGIN_USDT", "1"))
MAX_SCAN_SYMBOLS = int(os.environ.get("MAX_SCAN_SYMBOLS", "0"))
WATCHLIST_ONLY = os.environ.get("WATCHLIST_ONLY", "false").lower() == "true"
# Production defaults. The active workflow uses the full BingX universe and zone-touch
# trigger mode. These defaults keep a direct `python run_once.py` invocation aligned
# with production without changing the underlying 1H Demand/Supply zone construction.
FUNDAMENTAL_WHITELIST_ENABLED = os.environ.get("FUNDAMENTAL_WHITELIST_ENABLED", "false").lower() == "true"
ZONE_TRIGGER_MODE = os.environ.get("ZONE_TRIGGER_MODE", "zone").strip().lower()
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
# Absolute movement from the original signal price. Unlike adverse execution
# slippage, this deliberately rejects stale/broken ideas when price has moved
# too far in either direction before the order is submitted.
MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT = max(0.0, float(os.environ.get("MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT", "1.00")))
EXECUTION_QUOTE_MAX_AGE_SEC = max(0.0, float(os.environ.get("EXECUTION_QUOTE_MAX_AGE_SEC", "2.0")))
RESEARCH_RICH_CONTEXT_ENABLED = os.environ.get("RESEARCH_RICH_CONTEXT_ENABLED", "false").lower() == "true"

# Trend Filter V1 is diagnostic-only by default. It never changes zone construction
# or the existing closed-5m trigger. Enforcement is an explicit opt-in.
TREND_FILTER_ENABLED = os.environ.get("TREND_FILTER_ENABLED", "true").lower() == "true"
TREND_FILTER_MODE = os.environ.get("TREND_FILTER_MODE", "shadow").strip().lower()
if TREND_FILTER_MODE not in {"shadow", "enforce"}:
    raise ValueError("TREND_FILTER_MODE must be shadow or enforce")
TREND_FILTER_MIN_1H_BARS = max(200, int(os.environ.get("TREND_FILTER_MIN_1H_BARS", "400")))
TREND_FILTER_MIN_4H_BARS = max(200, int(os.environ.get("TREND_FILTER_MIN_4H_BARS", "400")))
TREND_FILTER_PERSISTENCE_LOOKBACK_1H = max(1, int(os.environ.get("TREND_FILTER_PERSISTENCE_LOOKBACK_1H", "6")))
TREND_FILTER_PERSISTENCE_LOOKBACK_4H = max(1, int(os.environ.get("TREND_FILTER_PERSISTENCE_LOOKBACK_4H", "3")))
TREND_FILTER_SLOPE_LOOKBACK_4H = max(1, int(os.environ.get("TREND_FILTER_SLOPE_LOOKBACK_4H", "6")))
TREND_FILTER_HISTORY_BUFFER_BARS = max(4, int(os.environ.get("TREND_FILTER_HISTORY_BUFFER_BARS", "12")))
TREND_FILTER_REQUIRE_PERSISTENCE = os.environ.get("TREND_FILTER_REQUIRE_PERSISTENCE", "false").lower() == "true"
ENTRY_CONTEXT_TELEMETRY_ENABLED = os.environ.get("ENTRY_CONTEXT_TELEMETRY_ENABLED", "false").strip().lower() == "true"
ENTRY_CONTEXT_DEPTH_LEVELS = max(5, int(os.environ.get("ENTRY_CONTEXT_DEPTH_LEVELS", "20")))
ENTRY_CONTEXT_TRADES_LIMIT = max(10, int(os.environ.get("ENTRY_CONTEXT_TRADES_LIMIT", "100")))
ENTRY_CONTEXT_HTTP_TIMEOUT_SEC = max(0.5, float(os.environ.get("ENTRY_CONTEXT_HTTP_TIMEOUT_SEC", "2.0")))
RESEARCH_DEFER_UNTIL_AFTER_EXECUTION = os.environ.get("RESEARCH_DEFER_UNTIL_AFTER_EXECUTION", "true").lower() == "true"
RECONCILIATION_MAX_SECONDS = float(os.environ.get("RECONCILIATION_MAX_SECONDS", "45"))
# Live execution requires the signal timestamp to be exactly the latest closed 1H bar.
# Also reject stale market data so a symbol with an old/delisted Binance series cannot
# masquerade as a fresh signal merely because its DataFrame index is zero-based.
MAX_DATA_STALENESS_HOURS = float(os.environ.get("MAX_DATA_STALENESS_HOURS", "2.0"))
# Zone geometry remains 1H; trigger detection runs on closed 5m bars so a short
# midpoint visit cannot be missed merely because the 1H candle later closes elsewhere.
KLINE_LIMIT_5M = int(os.environ.get("KLINE_LIMIT_5M", "144"))  # 12h of 5m bars
MAX_5M_TRIGGER_AGE_MINUTES = float(os.environ.get("MAX_5M_TRIGGER_AGE_MINUTES", "10"))
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


def _atomic_write_text_durable(path: Path, text: str) -> None:
    """Atomically replace a small JSON/state file and durably commit its directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
            dir_fd = os.open(str(path.parent), flags)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


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



def _append_execution_ledger(
    event_id: str,
    attempt_id: str | None,
    stage: str,
    payload: dict[str, Any] | None = None,
    *,
    raise_on_error: bool = False,
) -> None:
    """Append one immutable execution-lifecycle event.

    Ledger writes are normally observability-only. A narrow set of transitions
    (notably the pre-POST MARKET identity breadcrumb) can request fail-closed
    behavior because proceeding without that durable identity would make a
    later exchange order impossible to reconcile reliably after a process crash.
    """
    record = {
        "record_type": "EXECUTION_LEDGER",
        "ledger_ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "ts_ms": int(time.time() * 1000),
        "event_id": str(event_id),
        "attempt_id": attempt_id,
        "stage": str(stage),
        "strategy_version": _effective_strategy_version(),
        "engine_version": ENGINE_VERSION,
        "code_commit_sha": CODE_COMMIT_SHA,
    }
    if isinstance(payload, dict):
        record.update(payload)
    try:
        _append_jsonl(EXECUTION_LEDGER_PATH, record)
    except Exception as exc:
        log.error("[EXECUTION_LEDGER_WRITE_FAILED] event_id=%s attempt_id=%s stage=%s error=%s", event_id, attempt_id, stage, exc)
        if raise_on_error:
            raise RuntimeError(f"execution ledger write failed for stage={stage}: {exc}") from exc



def _trend_filter_enforce_reject(snapshot: dict[str, Any] | None, mode: str) -> bool:
    """Return True only when explicit enforce mode must veto the entry."""
    if str(mode or "shadow").strip().lower() != "enforce":
        return False
    if not isinstance(snapshot, dict):
        return True
    return str(snapshot.get("trend_decision", "REJECT")).upper() != "ALIGNED"


def _closed_count_for_decision(frame: pd.DataFrame | None, decision_ts_ms: int) -> int:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "close_time" not in frame.columns:
        return 0
    x = pd.to_numeric(frame["close_time"], errors="coerce").dropna()
    return int((x <= int(decision_ts_ms)).sum())


def _ensure_trend_history(*, symbol: str, timeframe: str, frame: pd.DataFrame | None, provider: str, binance_symbol: str, decision_ts_ms: int, min_bars: int) -> tuple[pd.DataFrame, bool]:
    """Return enough causal bars for Trend Filter without changing strategy data."""
    current = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    if _closed_count_for_decision(current, decision_ts_ms) >= int(min_bars):
        return current, False
    limit = int(min_bars) + TREND_FILTER_HISTORY_BUFFER_BARS
    try:
        rows = fetch_bingx_klines(symbol, timeframe, limit=limit, retryable=False) if provider == "bingx" else fetch_binance_klines(binance_symbol, timeframe, limit=limit, retryable=False)
        refreshed = pd.DataFrame(rows)
        if not refreshed.empty:
            return refreshed, True
    except Exception as exc:
        log.warning("[TREND_HISTORY] %s/%s supplemental history unavailable: %s", _display_symbol(symbol), timeframe, exc)
    return current, False


def _signal_decision_boundary_ts_ms(signal: dict[str, Any]) -> int | None:
    """Return the canonical causal boundary for the closed 5m trigger bar."""
    trigger_ts = _coerce_utc_timestamp(signal.get("trigger_bar_time") or signal.get("time"))
    canonical = trigger_ts + pd.Timedelta(minutes=5) if trigger_ts is not None else None
    for raw in (signal.get("decision_boundary_ts"), signal.get("trigger_bar_close_time")):
        if raw in (None, ""):
            continue
        parsed = _coerce_utc_timestamp(raw)
        if parsed is None:
            continue
        if canonical is None:
            return int(parsed.timestamp() * 1000)
        # Provider close should match the deterministic 5m trigger-close boundary.
        # A small tolerance covers millisecond/rounding differences without allowing
        # stale/future values to become a decision boundary.
        if parsed >= trigger_ts and abs((parsed - canonical).total_seconds()) <= 2.0:
            return int(parsed.timestamp() * 1000)
    return int(canonical.timestamp() * 1000) if canonical is not None else None


def _causal_1h_frame(df_1h: pd.DataFrame, boundary_ts: pd.Timestamp) -> pd.DataFrame:
    """Return only 1H candles whose close was known by the 5M decision boundary."""
    if df_1h is None or df_1h.empty or "timestamp" not in df_1h.columns:
        return pd.DataFrame()
    x = df_1h.copy()
    x["timestamp"] = x["timestamp"].map(_coerce_utc_timestamp)
    if "close_time" in x.columns:
        x["close_time"] = x["close_time"].map(_coerce_utc_timestamp)
    else:
        x["close_time"] = x["timestamp"] + pd.Timedelta(hours=1)
    x = x.dropna(subset=["timestamp", "close_time"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return x.loc[x["close_time"] <= boundary_ts].reset_index(drop=True)


def _causal_zone_lists(demand: list[dict[str, Any]], supply: list[dict[str, Any]], df_1h: pd.DataFrame, boundary_ts: pd.Timestamp) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter existing zones to those whose confirmation candle had closed by boundary."""
    def active(zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for zone in zones:
            activation_ts = _zone_activation_timestamp(zone, df_1h)
            if activation_ts is not None and activation_ts <= boundary_ts:
                out.append(zone)
        return out
    return active(demand), active(supply)


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
    forensics = signal.get("signal_forensics") if isinstance(signal.get("signal_forensics"), dict) else {}
    body_to_range = forensics.get("body_to_range")
    lower_wick_ratio = forensics.get("lower_wick_ratio")
    shadow_geometry_available = False
    shadow_geometry_bad = None
    try:
        if body_to_range is not None and lower_wick_ratio is not None:
            body_value = float(body_to_range)
            wick_value = float(lower_wick_ratio)
            if math.isfinite(body_value) and math.isfinite(wick_value):
                shadow_geometry_available = True
                shadow_geometry_bad = (
                    str(signal.get("type", "")).upper() == "SHORT"
                    and body_value > research.SHADOW_SHORT_BODY_TO_RANGE_GT
                    and wick_value < research.SHADOW_SHORT_LOWER_WICK_LT
                )
    except (TypeError, ValueError):
        shadow_geometry_available = False
        shadow_geometry_bad = None

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
        "trigger_to_decision_minutes": _elapsed_minutes(signal.get("trigger_bar_close_time") or signal.get("decision_boundary_ts") or signal.get("trigger_bar_time") or signal.get("time"), decision_ts),
        "shadow_short_filter_experiment": research.SHADOW_SHORT_FILTER_EXPERIMENT,
        "shadow_short_geometry_available": shadow_geometry_available,
        "shadow_short_geometry_bad": shadow_geometry_bad,
        "shadow_short_body_to_range_gt": research.SHADOW_SHORT_BODY_TO_RANGE_GT,
        "shadow_short_lower_wick_lt": research.SHADOW_SHORT_LOWER_WICK_LT,
        "counterfactual_experiment_version": shadow.SHADOW_EXPERIMENT_VERSION,
        "counterfactual_applied": False,
        "counterfactual_experiment_ids": sorted(list((signal.get("shadow_experiments") or {}).get("experiments", {}).keys())),
        "trend_filter_version": (signal.get("trend_filter") or {}).get("version"),
        "trend_filter_enabled": (signal.get("trend_filter") or {}).get("enabled"),
        "trend_filter_mode": (signal.get("trend_filter") or {}).get("mode"),
        "trend_decision": (signal.get("trend_filter") or {}).get("trend_decision"),
        "trend_reject_reason": (signal.get("trend_filter") or {}).get("trend_reject_reason"),
        "trend_4h": (signal.get("trend_filter") or {}).get("trend_4h"),
        "trend_1h": (signal.get("trend_filter") or {}).get("trend_1h"),
        "trend_persistence": (signal.get("trend_filter") or {}).get("trend_persistence"),
        "btc_regime": (signal.get("trend_filter") or {}).get("btc_regime"),
        "trend_decision_boundary_ts": signal.get("decision_boundary_ts"),
        "pre_execution_context_id": (signal.get("pre_execution_market_context") or {}).get("context_id"),
        "pre_execution_context_status": (signal.get("pre_execution_market_context") or {}).get("status"),
        "pre_execution_context_persisted": (signal.get("pre_execution_market_context") or {}).get("persisted"),
        "pre_execution_context_persisted_at_ms": (signal.get("pre_execution_market_context") or {}).get("persisted_at_ms"),
        "pre_execution_context_persist_write_started_at_ms": (signal.get("pre_execution_market_context") or {}).get("persist_write_started_at_ms"),
        "pre_execution_context_capture_ts": (signal.get("pre_execution_market_context") or {}).get("captured_at_ms"),
        "pre_execution_context_capture_span_ms": (signal.get("pre_execution_market_context") or {}).get("context_capture_span_ms"),
        "pre_execution_context_age_ms_at_order_ready": (signal.get("pre_execution_market_context") or {}).get("context_age_ms_at_order_ready"),
        "pre_execution_context_age_ms_at_execution_call_start": (signal.get("pre_execution_market_context") or {}).get("context_age_ms_at_execution_call_start"),
        "research_schema_version": research.RESEARCH_SCHEMA_VERSION,
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
            _atomic_write_text_durable(FAILED_SIGNALS_PATH, json.dumps(cleaned, ensure_ascii=False, indent=2))
        return cleaned
    except Exception as exc:
        log.warning("[FAILED_SIGNALS] load failed: %s", exc)
        return {}


def _load_event_claims_unlocked() -> dict[str, dict[str, Any]]:
    if not EVENT_CLAIMS_PATH.exists():
        raise RuntimeError("execution claims file is missing; refusing duplicate-prone reset")
    try:
        raw = json.loads(EVENT_CLAIMS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"execution claims file is unreadable; refusing duplicate-prone reset: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("execution claims file is not a JSON object; refusing duplicate-prone reset")
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def _write_event_claims_unlocked(raw: dict[str, dict[str, Any]]) -> None:
    _atomic_write_text_durable(EVENT_CLAIMS_PATH, json.dumps(raw, ensure_ascii=False, indent=2))


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
        "opened_emergency_close_unverified",
        "opened_registration_failed_close_unverified",
        "entry_state_unverified",
        "entry_state_unverified_protected",
        "entry_not_filled",
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
        _atomic_write_text_durable(FAILED_SIGNALS_PATH, json.dumps(raw, ensure_ascii=False, indent=2))
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
    if value in {"ENTRY_NOT_FILLED", "SKIPPED_MIN_QTY", "SKIPPED_TP_MIN_QTY", "SKIPPED_INVALID_SETUP", "SKIPPED_STALE_SIGNAL", "EXECUTION_QUOTE_UNAVAILABLE", "BLOCKED_PROTECTION_PREFLIGHT", "BLOCKED_PRIVATE_PREFLIGHT"}:
        return "EXECUTION_BLOCKED"
    if value in {"ERROR", "OPENED", "DISABLED", "BLOCKED_MISSING_CREDENTIALS"} or value.endswith("_FAILED") or value.startswith("FAILED"):
        return "EXECUTION_FAILURE"
    return "EXECUTION_OTHER"


def _trade_event_record_type(status: str) -> str:
    """Use TRADE_OPEN only when exchange execution confirms a position lifecycle."""
    return "TRADE_OPEN" if str(status or "").lower() in {
        "opened_protected",
        "opened_then_emergency_closed",
    } else "ENTRY_ATTEMPT"


def _emergency_trade_close_record(
    signal: dict[str, Any], execution: dict[str, Any], *, observed_closed_ts_ms: int | None = None
) -> dict[str, Any] | None:
    """Build a gross-only close row when rollback fills and flat position are confirmed."""
    if str(execution.get("status", "")).lower() != "opened_then_emergency_closed":
        return None
    close = execution.get("close") if isinstance(execution.get("close"), dict) else {}
    verification = close.get("verification") if isinstance(close.get("verification"), dict) else {}
    if str(close.get("status", "")).lower() != "closed_verified" or str(verification.get("status", "")).lower() != "not_found":
        return None
    position = execution.get("position") if isinstance(execution.get("position"), dict) else {}
    try:
        entry_price = float(position.get("avgPrice") or position.get("entryPrice"))
        entry_qty = abs(float(position.get("positionAmt") or position.get("availableAmt")))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(entry_price) or entry_price <= 0 or not math.isfinite(entry_qty) or entry_qty <= 0:
        return None

    confirmed_orders = []
    for attempt in close.get("attempts", []):
        if not isinstance(attempt, dict):
            continue
        response = attempt.get("response") if isinstance(attempt.get("response"), dict) else {}
        if response.get("code") not in (0, "0"):
            continue
        data = response.get("data") if isinstance(response.get("data"), dict) else {}
        order = data.get("order") if isinstance(data.get("order"), dict) else {}
        if str(order.get("status", "")).upper() != "FILLED":
            continue
        try:
            qty = abs(float(order.get("executedQty")))
            price = float(order.get("avgPrice"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(qty) or qty <= 0 or not math.isfinite(price) or price <= 0:
            continue
        confirmed_orders.append({
            "order_id": str(order.get("orderId") or order.get("orderID") or "") or None,
            "status": "FILLED", "qty": qty, "avg_price": price,
            "working_type": order.get("workingType"),
            "execution_ts_ms": next((int(order[key]) for key in ("updateTime", "time", "transactTime")
                                      if str(order.get(key, "")).isdigit() and int(order[key]) > 0), None),
        })
    exit_qty = sum(row["qty"] for row in confirmed_orders)
    tolerance = max(1e-10, entry_qty * 1e-6)
    if not confirmed_orders or abs(exit_qty - entry_qty) > tolerance:
        return None

    exit_vwap = sum(row["qty"] * row["avg_price"] for row in confirmed_orders) / exit_qty
    direction = str(signal.get("type") or "").upper()
    if direction not in {"LONG", "SHORT"}:
        return None
    signed_move = exit_vwap - entry_price if direction == "LONG" else entry_price - exit_vwap
    gross_pnl_abs = signed_move * entry_qty
    gross_pnl_pct = signed_move / entry_price * 100.0
    execution_times = [row["execution_ts_ms"] for row in confirmed_orders if row["execution_ts_ms"] is not None]
    observed_ts = observed_closed_ts_ms if observed_closed_ts_ms is not None else int(time.time() * 1000)
    order_ids = [row["order_id"] for row in confirmed_orders if row["order_id"]]
    evidence = {
        "status": "CONFIRMED", "authority": "EMERGENCY_ORDER_EXECUTION_RESPONSE",
        "confirmed_exit_qty": exit_qty, "exit_vwap_confirmed": exit_vwap,
        "exit_vwap_source": "FILLED_EMERGENCY_ORDER_AVG_PRICE",
        "position_verification": "not_found", "orders": confirmed_orders,
        "execution_timestamp_source": "EXCHANGE_ORDER_TIME" if execution_times else "UNAVAILABLE",
    }
    return {
        "record_type": "TRADE_CLOSE", "event_id": signal.get("event_id"),
        "attempt_id": signal.get("attempt_id") or execution.get("attempt_id"),
        "symbol": signal.get("symbol"), "direction": direction,
        "closed_ts": max(execution_times) if execution_times else None,
        "execution_close_ts": max(execution_times) if execution_times else None,
        "observed_closed_ts": observed_ts,
        "close_timestamp_source": "EXCHANGE_ORDER_TIME" if execution_times else "UNAVAILABLE",
        "exit_reason": "EMERGENCY_CLOSE", "outcome_category": "EMERGENCY_EXIT",
        "strategy_version": signal.get("strategy_version", _effective_strategy_version()),
        "code_commit_sha": CODE_COMMIT_SHA,
        "entry_bar": signal.get("entry_bar", {}),
        "previous_bar": signal.get("previous_bar", {}),
        "signal_snapshot": dict(signal),
        "entry_order": execution.get("order", {}) if isinstance(execution.get("order"), dict) else {},
        "fill_position": dict(position),
        "entry_price": entry_price, "exit_price": exit_vwap,
        "exit_price_semantics": "CONFIRMED_EMERGENCY_ORDER_AVG_PRICE",
        "exit_vwap_confirmed": exit_vwap, "exit_vwap_source": "FILLED_EMERGENCY_ORDER_AVG_PRICE",
        "realized_pnl_pct": gross_pnl_pct,
        "realized_pnl_source": "CONFIRMED_EMERGENCY_POSITION_AND_ORDER_FILL_GROSS",
        "realized_pnl_abs": None, "realized_pnl_abs_source": "UNAVAILABLE_NET_FEES_FUNDING",
        "realized_pnl_abs_status": "UNAVAILABLE", "gross_realized_pnl_abs": gross_pnl_abs,
        "gross_realized_pnl_source": "CONFIRMED_EMERGENCY_POSITION_AND_ORDER_FILL",
        "net_realized_pnl_abs": None, "net_realized_pnl_source": "UNAVAILABLE",
        "entry_qty": entry_qty, "exit_qty": exit_qty, "exit_order_ids": order_ids,
        "fees_paid_abs": None, "fee_raw_total": None, "fees_source": "UNAVAILABLE",
        "funding_paid_abs": None, "funding_source": "UNAVAILABLE",
        "close_execution_evidence": evidence,
    }


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
        if status in {"opened_protected", "entry_state_unverified_protected", "opened", "already_executed", "existing_position"} and row.get("event_id"):
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
    raw_version = raw.get("version") if isinstance(raw, dict) else None
    if (
        not isinstance(raw, dict)
        or isinstance(raw_version, bool)
        or not isinstance(raw_version, int)
        or raw_version != ZONE_VISIT_STATE_VERSION
        or not isinstance(raw.get("symbols"), dict)
    ):
        log.warning("[ZONE_STATE] invalid/newer schema; starting empty state")
        return {"version": ZONE_VISIT_STATE_VERSION, "trigger_mode": ZONE_TRIGGER_MODE, "symbols": {}}
    stored_mode = str(raw.get("trigger_mode", "")).strip().lower()
    if stored_mode and stored_mode != ZONE_TRIGGER_MODE:
        log.warning("[ZONE_STATE] trigger_mode changed %s -> %s; starting empty visit state", stored_mode, ZONE_TRIGGER_MODE)
        return {"version": ZONE_VISIT_STATE_VERSION, "trigger_mode": ZONE_TRIGGER_MODE, "symbols": {}}
    raw["trigger_mode"] = ZONE_TRIGGER_MODE
    return raw


def _save_zone_visit_state(state: dict[str, Any]) -> None:
    """Atomically and durably persist zone-visit state after a complete scan batch."""
    DATA.mkdir(parents=True, exist_ok=True)
    payload = {"version": ZONE_VISIT_STATE_VERSION, "trigger_mode": ZONE_TRIGGER_MODE, "symbols": state.get("symbols", {})}
    lock_path = ZONE_VISIT_STATE_LOCK_PATH
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _atomic_write_text_durable(ZONE_VISIT_STATE_PATH, json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _clear_stale_pending_zone_visit(zone_state: dict[str, Any], signal: dict[str, Any]) -> bool:
    """Release only the zone visit that owns a stale 5m event rejected pre-order."""
    symbol = str(signal.get("symbol", "")).upper()
    event_id = str(signal.get("event_id", ""))
    if not symbol or not event_id:
        return False
    symbols = zone_state.get("symbols") if isinstance(zone_state, dict) else None
    symbol_state = symbols.get(symbol) if isinstance(symbols, dict) else None
    zones = symbol_state.get("zones") if isinstance(symbol_state, dict) else None
    if not isinstance(zones, dict):
        return False
    for visit in zones.values():
        if not isinstance(visit, dict):
            continue
        pending = visit.get("pending_signal")
        if not isinstance(pending, dict) or str(pending.get("event_id", "")) != event_id:
            continue
        visit.update({
            "state": "ARMED",
            "pending_signal": None,
            "trigger_event_id": None,
            "first_touch_ts": None,
            "last_touch_ts": None,
            "touch_count": 0,
            "visit_id": None,
            "lock_reason": "stale_5m_trigger_rejected_before_order",
        })
        return True
    return False


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


def _coerce_utc_timestamp(value: Any) -> pd.Timestamp | None:
    """Normalize a scalar provider timestamp to timezone-aware UTC.

    BingX/Binance kline payloads expose epoch milliseconds, while tests and
    persisted telemetry may also provide ISO strings or datetime objects.
    Never call ``pd.Timestamp(raw_epoch_ms)`` directly: pandas interprets an
    integer without a unit as nanoseconds and returns a tz-naive timestamp.
    """
    if value is None:
        return None
    try:
        if isinstance(value, pd.Timestamp):
            ts = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if not math.isfinite(numeric):
                return None
            magnitude = abs(numeric)
            unit = ("ns" if magnitude >= 1e17 else
                    "us" if magnitude >= 1e14 else
                    "ms" if magnitude >= 1e11 else
                    "s" if magnitude >= 1e8 else None)
            if unit is None:
                ts = pd.to_datetime(value, utc=True)
            else:
                ts = pd.to_datetime(value, unit=unit, utc=True)
        else:
            text = str(value).strip()
            if not text:
                return None
            numeric = pd.to_numeric(text, errors="coerce")
            if pd.notna(numeric):
                numeric = float(numeric)
                magnitude = abs(numeric)
                unit = ("ns" if magnitude >= 1e17 else
                        "us" if magnitude >= 1e14 else
                        "ms" if magnitude >= 1e11 else
                        "s" if magnitude >= 1e8 else None)
                ts = pd.to_datetime(numeric, unit=unit, utc=True) if unit else pd.to_datetime(text, utc=True)
            else:
                ts = pd.to_datetime(text, utc=True)
        if pd.isna(ts):
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return None


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
    df_5m: pd.DataFrame | None = None,
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
    ts = _coerce_utc_timestamp(bar.get("timestamp") if hasattr(bar, "get") else None)
    if ts is None:
        raise ValueError("invalid_trigger_bar_timestamp")
    canonical_trigger_close_ts = ts + pd.Timedelta(minutes=5)
    provider_close_ts = _coerce_utc_timestamp(bar.get("close_time") if hasattr(bar, "get") else None)
    trigger_close_ts = provider_close_ts if provider_close_ts is not None and provider_close_ts >= ts and abs((provider_close_ts - canonical_trigger_close_ts).total_seconds()) <= 2.0 else canonical_trigger_close_ts
    causal_df_1h = _causal_1h_frame(df_1h, trigger_close_ts)
    if causal_df_1h.empty:
        raise ValueError("causal_1h_history_unavailable")
    causal_idx = len(causal_df_1h) - 1
    causal_demand, causal_supply = _causal_zone_lists(demand, supply, df_1h, trigger_close_ts)
    obstacle = _nearest_opposing_level(direction, entry, causal_demand, causal_supply, causal_df_1h, causal_idx)
    if obstacle is not None:
        obstacle_price = float(obstacle["price"])
        structural_distance = obstacle_price - entry if direction == "LONG" else entry - obstacle_price
        if structural_distance <= 0 or structural_distance / risk < MIN_STRUCTURE_ROOM_R:
            raise ValueError(f"insufficient_structure_room={structural_distance / risk if risk else 0.0:.3f}R < {MIN_STRUCTURE_ROOM_R:.3f}R")
    elif REQUIRE_STRUCTURE_OBSTACLE:
        raise ValueError("missing_structural_obstacle")

    tp1 = entry * (1.0 + TP1_PCT / 100.0) if direction == "LONG" else entry * (1.0 - TP1_PCT / 100.0)
    tp2 = entry * (1.0 + TP2_PCT / 100.0) if direction == "LONG" else entry * (1.0 - TP2_PCT / 100.0)
    trigger_ts_ms = int(ts.timestamp() * 1000)
    zone_copy = {**zone, "kind": "DEMAND" if direction == "LONG" else "SUPPLY"}
    prev_ts = _coerce_utc_timestamp(prev_bar.get("timestamp")) if prev_bar is not None and hasattr(prev_bar, "get") else None
    prev_bar_dict = {
        "timestamp": prev_ts.isoformat() if prev_ts is not None else None,
        "open": float(prev_bar["open"]) if prev_bar is not None else None,
        "high": float(prev_bar["high"]) if prev_bar is not None else None,
        "low": float(prev_bar["low"]) if prev_bar is not None else None,
        "close": float(prev_bar["close"]) if prev_bar is not None else None,
        "volume": float(prev_bar["volume"]) if prev_bar is not None else None,
    }
    event_id = _make_5m_event_id(symbol, direction, trigger_ts_ms, zone)
    atr_1h = float(causal_df_1h.loc[causal_idx, "atr50"]) if "atr50" in causal_df_1h.columns else 0.0
    trigger_volume = float(bar["volume"])
    vol_ratio = None
    vol_ratio_baseline = None
    vol_ratio_baseline_count = 0
    if df_5m is not None and not df_5m.empty and "timestamp" in df_5m.columns and "volume" in df_5m.columns:
        x5 = df_5m.copy()
        x5["timestamp"] = x5["timestamp"].map(_coerce_utc_timestamp)
        x5["volume"] = pd.to_numeric(x5["volume"], errors="coerce")
        trigger_ts_norm = ts
        prior = x5.loc[(x5["timestamp"] < trigger_ts_norm) & x5["volume"].notna(), "volume"].tail(20)
        if len(prior) == 20:
            baseline = float(prior.mean())
            if math.isfinite(baseline) and baseline > 0:
                vol_ratio_baseline = baseline
                vol_ratio_baseline_count = 20
                vol_ratio = trigger_volume / baseline
    # Zone age is diagnostic metadata only. An active zone may be traded
    # regardless of age; invalidation is determined by the zone engine itself.
    zone_age_bars = max(0, int(causal_idx - int(zone.get("start", causal_idx))))
    directional_ok = (float(bar["close"]) >= float(bar["open"])) if direction == "LONG" else (float(bar["close"]) <= float(bar["open"]))
    if REQUIRE_DIRECTIONAL_CANDLE and not directional_ok:
        raise ValueError("directional_candle_required")
    entry_bar = {k: (ts.isoformat() if k == "timestamp" else float(bar[k])) for k in ["timestamp", "open", "high", "low", "close", "volume"]}
    setup_zone = {**zone_copy, "age_bars": zone_age_bars}
    signal_forensics = _signal_forensics(direction, float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"]), max(atr_1h, 1e-12), setup_zone)
    return {
        "event_id": event_id,
        "idx": trigger_ts_ms,
        "time": ts.isoformat(),
        "trigger_bar_time": ts.isoformat(),
        "trigger_bar_close_time": trigger_close_ts.isoformat(),
        "decision_boundary_ts": trigger_close_ts.isoformat(),
        "decision_boundary_kind": "CLOSED_5M_TRIGGER_CLOSE",
        "decision_boundary_source": "PROVIDER_CLOSE_TIME" if provider_close_ts is not None and provider_close_ts >= ts and abs((provider_close_ts - canonical_trigger_close_ts).total_seconds()) <= 2.0 else "DERIVED_TRIGGER_OPEN_PLUS_5M",
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
            "volume_ratio_5m20": vol_ratio,
            "volume_ratio_semantics": "trigger_5m_volume_divided_by_mean_previous_20_closed_5m_volumes",
            "volume_ratio_source": "5M_PREVIOUS_20_CLOSED_BARS" if vol_ratio is not None else "UNAVAILABLE_TRIGGER_TIMESTAMP_OR_20_BAR_BASELINE",
            "volume_ratio_baseline": vol_ratio_baseline,
            "volume_ratio_baseline_count": vol_ratio_baseline_count,
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


def _zone_activation_timestamp(zone: dict[str, Any], df_1h: pd.DataFrame) -> pd.Timestamp | None:
    """Return the first timestamp at which a zone is live in real time.

    ``zone.start + SWING_LEN`` identifies the 1H confirmation candle. Its stored
    timestamp is the candle OPEN, so the zone cannot be known until that candle
    has CLOSED one hour later. Using the open timestamp contaminates historical
    candidate populations with information that was not yet available live.
    """
    try:
        activation_idx = int(zone.get("start", -1)) + SWING_LEN
    except (TypeError, ValueError):
        return None
    if activation_idx < 0 or activation_idx >= len(df_1h):
        return None
    ts = _coerce_utc_timestamp(df_1h.loc[activation_idx, "timestamp"])
    if ts is None:
        return None
    return ts + pd.Timedelta(hours=1)


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
        # This is a diagnostic-only historical-window pass. Cache normalized
        # columns once and use NumPy masks/arrays rather than allocating a pandas
        # Series per candle through iterrows(). The entry state machine below is
        # intentionally unchanged and still processes every unprocessed closed bar.
        window_timestamps = x["timestamp"].tolist()
        window_timestamp_ns = x["timestamp"].astype("int64").to_numpy(copy=False)
        window_opens = x["open"].to_numpy(dtype=float, copy=False)
        window_highs = x["high"].to_numpy(dtype=float, copy=False)
        window_lows = x["low"].to_numpy(dtype=float, copy=False)
        window_closes = x["close"].to_numpy(dtype=float, copy=False)
        window_volumes = x["volume"].to_numpy(dtype=float, copy=False)
        for zone_key, (direction, zone) in active_zone_map.items():
            zdiag = diagnostics["zones"][zone_key]
            midpoint = float(zdiag["midpoint"])
            activation_ts = _zone_activation_timestamp(zone, df_1h)
            if activation_ts is None:
                active_mask = np.ones(len(window_timestamps), dtype=bool)
            else:
                active_mask = window_timestamp_ns >= activation_ts.value
            active_indices = np.flatnonzero(active_mask)
            if active_indices.size == 0:
                continue

            top = float(zone["top"])
            bottom = float(zone["btm"])
            zone_touch_mask = active_mask & (window_lows <= top) & (window_highs >= bottom)
            zone_touch_indices = np.flatnonzero(zone_touch_mask)
            if zone_touch_indices.size:
                zdiag["window_bars_in_zone"] += int(zone_touch_indices.size)
                zdiag["window_zone_touches"] += int(zone_touch_indices.size)
                zdiag["window_last_zone_touch"] = window_timestamps[int(zone_touch_indices[-1])].isoformat()

            midpoint_touch_mask = active_mask & (window_lows <= midpoint) & (window_highs >= midpoint)
            midpoint_touch_indices = np.flatnonzero(midpoint_touch_mask)
            if midpoint_touch_indices.size:
                zdiag["window_midpoint_touches"] += int(midpoint_touch_indices.size)
                zdiag["window_last_midpoint_touch"] = window_timestamps[int(midpoint_touch_indices[-1])].isoformat()

            # Preserve the original chronological iteration and strict '<' tie
            # break for the closest bar: the earliest bar wins equal distances.
            # The source feeds validated finite OHLC values, so this loop operates
            # on numeric arrays and does not construct pandas row objects.
            closest_idx = None
            closest_distance = None
            if midpoint:
                for raw_idx in active_indices:
                    i = int(raw_idx)
                    low = float(window_lows[i])
                    high = float(window_highs[i])
                    touched = low <= midpoint <= high
                    distance_pct = 0.0 if touched else ((midpoint - high) if high < midpoint else (low - midpoint)) / midpoint * 100.0
                    if distance_pct is not None and (closest_distance is None or distance_pct < closest_distance):
                        closest_distance = distance_pct
                        closest_idx = i
            if closest_idx is not None and (
                zdiag["window_closest_midpoint_distance_pct"] is None
                or closest_distance < zdiag["window_closest_midpoint_distance_pct"]
            ):
                i = int(closest_idx)
                ts = window_timestamps[i]
                zdiag["window_closest_midpoint_distance_pct"] = closest_distance
                zdiag["window_closest_midpoint_bar"] = {
                    "timestamp": ts.isoformat(), "open": float(window_opens[i]),
                    "high": float(window_highs[i]), "low": float(window_lows[i]),
                    "close": float(window_closes[i]), "volume": float(window_volumes[i]),
                    "distance_pct": closest_distance,
                }

    # Drop stale state entries for zones that no longer exist; a broken 1H zone
    # cannot be resurrected by an old 5m visit.
    symbol_state["zones"] = {k: v for k, v in symbol_state["zones"].items() if k in active_zone_map and isinstance(v, dict)}
    last_processed_raw = symbol_state.get("last_processed_5m_ts")
    last_processed = _coerce_utc_timestamp(last_processed_raw) if last_processed_raw else None
    if last_processed is not None:
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
            _latest_ts = _coerce_utc_timestamp(x["timestamp"].iloc[-1])
            diagnostics["latest_closed_5m_ts"] = _latest_ts.isoformat() if _latest_ts is not None else None
    for _, bar in processed_rows.iterrows():
        bar_ts = _coerce_utc_timestamp(bar["timestamp"])
        if bar_ts is None:
            continue
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
                first_touch_ts = _coerce_utc_timestamp(zs.get("first_touch_ts"))
                if rearmed and first_touch_ts is not None and bar_ts > first_touch_ts:
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
            activation_ts = _zone_activation_timestamp(zone, df_1h)
            if activation_ts is not None:
                if bar_ts < activation_ts:
                    if diagnostics is not None:
                        diagnostics["zones"][zone_key]["activation_blocks"] += 1
                        diagnostics["touch_events"].append({
                            "timestamp": bar_ts.isoformat(), "zone_key": zone_key, "direction": direction,
                            "midpoint": float((float(zone["top"]) + float(zone["btm"])) / 2.0),
                            "state_before": zs.get("state", "ARMED"), "reason": "zone_not_active", "touch_mode": ZONE_TRIGGER_MODE,
                            "visit_id": zs.get("visit_id"),
                            "bar": {"timestamp": bar_ts.isoformat(), "open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"])}
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
                    zdiag["closest_midpoint_bar"] = {"timestamp": bar_ts.isoformat(), "open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"]), "distance_pct": distance_pct}
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
                    diagnostics["touch_events"].append({"timestamp": bar_ts.isoformat(), "zone_key": zone_key, "direction": direction, "midpoint": midpoint, "state_before": "ARMED", "reason": "ambiguous_overlap", "touch_mode": ZONE_TRIGGER_MODE, "visit_id": zs.get("visit_id"), "bar": {"timestamp": bar_ts.isoformat(), "open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"])}})
                    diagnostics["zones"][zone_key]["state_after"] = "LOCKED"
                _set_decision("BLOCKED_AMBIGUOUS", zone_key=zone_key, direction=direction, reason="ambiguous_overlap", timestamp=bar_ts.isoformat())
                continue
            state = dict(zs)
            # Only the previous bar that was actually processed in this scan batch
            # and was itself inside the trigger freshness window participates in the
            # continuous-visit guard. A stale touch is explicitly released below;
            # it must not suppress a following fresh touch merely because both bars
            # overlap the zone. Prior-run touches are represented by the persisted
            # zone state itself (LOCKED), so they do not need this guard.
            prev_was_processed_now = False
            prev_bar_is_fresh = False
            prev_ts = None
            if prev is not None:
                prev_ts = _coerce_utc_timestamp(prev["timestamp"])
                if prev_ts is not None:
                    # Candle timestamps mark the open; a 5m trigger becomes
                    # actionable at its close, matching the signal-age calculation.
                    prev_close_ts = prev_ts + pd.Timedelta(minutes=5)
                    prev_age_min = max(0.0, (now - prev_close_ts).total_seconds() / 60.0)
                    prev_bar_is_fresh = prev_age_min <= MAX_5M_TRIGGER_AGE_MINUTES
                if last_processed_raw:
                    try:
                        prev_was_processed_now = prev_ts is not None and prev_ts > last_processed
                    except Exception:
                        prev_was_processed_now = True
                else:
                    prev_was_processed_now = bool(
                        prev_ts is not None and prev_ts >= processed_rows["timestamp"].min()
                    )
            previous_bar_can_block_new_visit = prev_was_processed_now and prev_bar_is_fresh
            state["previous_midpoint_touch"] = bool(
                previous_bar_can_block_new_visit and prev is not None and
                float(prev["low"]) <= midpoint <= float(prev["high"])
            )
            state["previous_zone_touch"] = bool(
                previous_bar_can_block_new_visit and prev is not None and
                float(prev["low"]) <= float(zone["top"]) and float(prev["high"]) >= float(zone["btm"])
            )
            previous_touch = state["previous_midpoint_touch"] if ZONE_TRIGGER_MODE == "midpoint" else state["previous_zone_touch"]
            if previous_touch:
                # This is still a continuous touch, not a new signal. However, a
                # preceding breakout/re-arm may already have cleared first_touch_ts.
                # Re-locking without restoring visit metadata makes the zone
                # permanently LOCKED because the re-arm gate requires first_touch_ts.
                if not zs.get("first_touch_ts"):
                    zs["first_touch_ts"] = bar_ts.isoformat()
                    zs["visit_id"] = f"{symbol}:{zone_key}:{int(bar_ts.timestamp()*1000)}"
                    zs["lock_reason"] = "continuous_touch_after_rearm"
                zs.update({"state": "LOCKED", "last_touch_ts": bar_ts.isoformat(), "touch_count": int(zs.get("touch_count", 0)) + 1})
                if diagnostics is not None:
                    diagnostics["zones"][zone_key]["same_visit_blocks"] += 1
                    diagnostics["touch_events"].append({"timestamp": bar_ts.isoformat(), "zone_key": zone_key, "direction": direction, "midpoint": midpoint, "state_before": "LOCKED_OR_CONTINUOUS", "reason": ("previous_5m_midpoint_touch" if ZONE_TRIGGER_MODE == "midpoint" else "previous_5m_zone_touch"), "touch_mode": ZONE_TRIGGER_MODE, "visit_id": zs.get("visit_id"), "touch_count_before_trigger": int(zs.get("touch_count", 0)), "bar": {"timestamp": bar_ts.isoformat(), "open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"])}})
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
            event_boundary_ts = bar_ts + pd.Timedelta(minutes=5)
            causal_df_1h = _causal_1h_frame(df_1h, event_boundary_ts)
            causal_demand, causal_supply = _causal_zone_lists(demand, supply, df_1h, event_boundary_ts)
            research_obstacle = _nearest_opposing_level(direction, research_entry_ref, causal_demand, causal_supply, causal_df_1h, len(causal_df_1h) - 1) if not causal_df_1h.empty else None
            if diagnostics is not None:
                touch_payload = {
                    "timestamp": bar_ts.isoformat(), "zone_key": zone_key, "direction": direction,
                    "midpoint": midpoint, "visit_id": zs.get("visit_id"),
                    "touch_count_before_trigger": int(zs.get("touch_count", 0)),
                    "touch_mode": ZONE_TRIGGER_MODE,
                    "entry_ref": research_entry_ref,
                    "bar": {"timestamp": bar_ts.isoformat(), "open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"])},
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
                signal = _build_5m_zone_signal(symbol, direction, zone, bar, prev, df_1h, demand, supply, state, df_5m=x)
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
            # The 5m trigger becomes actionable when the candle closes. Measuring
            # from its open incorrectly removes five minutes from the configured
            # freshness window and rejects otherwise-current signals.
            bar_close_ts = _coerce_utc_timestamp(signal.get("trigger_bar_close_time") or signal.get("decision_boundary_ts"))
            if bar_close_ts is None:
                bar_close_ts = bar_ts + pd.Timedelta(minutes=5)
            age_min = max(0.0, (now - bar_close_ts).total_seconds() / 60.0)
            if age_min > MAX_5M_TRIGGER_AGE_MINUTES:
                # A stale touch is useful audit data but must not consume the zone visit.
                # Otherwise a delayed provider/API response could lock a zone until a
                # future full exit/re-entry, silently suppressing the next valid setup.
                stale_visit_id = zs.get("visit_id")
                stale_touch_count = int(zs.get("touch_count", 0))
                zs.update({
                    "state": "ARMED",
                    "first_touch_ts": None,
                    "last_touch_ts": None,
                    "touch_count": 0,
                    "visit_id": None,
                    "lock_reason": ("stale_midpoint_touch_ignored" if ZONE_TRIGGER_MODE == "midpoint" else "stale_zone_touch_ignored"),
                    "trigger_event_id": None,
                    "pending_signal": None,
                })
                if diagnostics is not None:
                    diagnostics["zones"][zone_key]["stale_touches"] += 1
                    diagnostics["touch_events"].append({"timestamp": bar_ts.isoformat(), "zone_key": zone_key, "direction": direction, "midpoint": midpoint, "state_before": "ARMED", "reason": ("stale_midpoint_touch_ignored" if ZONE_TRIGGER_MODE == "midpoint" else "stale_zone_touch_ignored"), "touch_mode": ZONE_TRIGGER_MODE, "age_min": age_min, "visit_id": stale_visit_id, "touch_count_before_trigger": stale_touch_count, "bar": {"timestamp": bar_ts.isoformat(), "open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"])}})
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
                        "bar": {"timestamp": bar_ts.isoformat(), "open": float(bar["open"]), "high": float(bar["high"]), "low": float(bar["low"]), "close": float(bar["close"]), "volume": float(bar["volume"])},
                        "previous_bar": ({"timestamp": (_coerce_utc_timestamp(prev["timestamp"]).isoformat() if _coerce_utc_timestamp(prev["timestamp"]) is not None else None), "open": float(prev["open"]), "high": float(prev["high"]), "low": float(prev["low"]), "close": float(prev["close"]), "volume": float(prev["volume"])} if prev is not None else {}),
                    })
                _set_decision("SIGNAL_CREATED", zone_key=zone_key, direction=direction, reason="signal_created", timestamp=bar_ts.isoformat())

    if not processed_rows.empty:
        _processed_ts = _coerce_utc_timestamp(processed_rows["timestamp"].max())
        symbol_state["last_processed_5m_ts"] = _processed_ts.isoformat() if _processed_ts is not None else symbol_state.get("last_processed_5m_ts")
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
    for zone_key, zs in symbol_state["zones"].items():
        pending = zs.get("pending_signal")
        if not isinstance(pending, dict):
            continue
        eid = str(pending.get("event_id", ""))
        if not eid or eid in successful_ids or eid in terminal_event_ids:
            continue
        try:
            _pending_ts = _coerce_utc_timestamp(pending.get("trigger_bar_close_time") or pending.get("decision_boundary_ts") or pending["trigger_bar_time"])
            age_min = max(0.0, (now - _pending_ts).total_seconds() / 60.0) if _pending_ts is not None else float("inf")
        except Exception:
            age_min = float("inf")
        if age_min <= MAX_5M_TRIGGER_AGE_MINUTES:
            if not any(str(s.get("event_id")) == eid for s in signals):
                signals.append(pending)
                if diagnostics is not None:
                    diagnostics["pending_retries"] += 1
                trigger_text = trigger_text or f"{pending.get('type')} @ {pending.get('entry')} 5m_{ZONE_TRIGGER_MODE}_touch_retry"
        else:
            # Only retire retryable events here. Successful/terminal events above
            # remain LOCKED until a real price re-arm, preventing duplicate entries.
            zs.update({
                "state": "ARMED", "pending_signal": None, "trigger_event_id": None,
                "first_touch_ts": None, "last_touch_ts": None, "touch_count": 0,
                "visit_id": None, "lock_reason": "stale_pending_5m_trigger_retired",
            })
            if diagnostics is not None:
                diagnostics.setdefault("stale_pending_retired", []).append({"zone_key": zone_key, "event_id": eid, "age_min": age_min})
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
        signal_ts = _coerce_utc_timestamp(signal_time)
        latest_ts = _coerce_utc_timestamp(latest_closed_time)
        if signal_ts is None or latest_ts is None or signal_ts != latest_ts:
            return False, "signal_time_not_latest"
    except Exception:
        return False, "invalid_signal_time"
    return True, "ok"

def _update_active_trades_with_snapshot() -> None:
    """Reduce per-position HTTP reads without weakening the later live checks.

    The bulk positions response is used only for each trade's first state read.
    TP-fill reconciliation and the final per-trade position read remain fresh
    targeted exchange queries inside the tracker. If bulk retrieval fails, keep
    the original per-symbol behavior.
    """
    try:
        snapshot = get_positions(
            timeout_sec=float(os.environ.get("TRACKER_POSITION_SNAPSHOT_TIMEOUT_SEC", "5")),
            retryable=False,
        )
        if not isinstance(snapshot, list) or any(not isinstance(row, dict) for row in snapshot):
            raise RuntimeError("bulk positions response has an invalid shape")
    except Exception as exc:
        log.warning("[TRACKER_POSITION_SNAPSHOT] bulk fetch failed; using per-symbol queries: %s", exc)
        update_active_trades()
        return

    # A few test/integration wrappers replace the tracker with a zero-argument
    # callable. Detect that signature explicitly rather than catching TypeError
    # around the call, which could otherwise hide a genuine tracker failure.
    try:
        parameters = inspect.signature(update_active_trades).parameters.values()
        supports_snapshot = any(
            parameter.name == "position_snapshot" or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        supports_snapshot = True
    if supports_snapshot:
        update_active_trades(position_snapshot=snapshot)
    else:
        update_active_trades()


def _blocked_private_execution_result() -> dict[str, str]:
    """Describe why execution is blocked without importing the BingX module alias."""
    if not credentials_available():
        return {"status": "BLOCKED_MISSING_CREDENTIALS", "error": "BingX private credentials are unavailable"}
    return {
        "status": "BLOCKED_PRIVATE_PREFLIGHT",
        "error": "BingX private execution is blocked by state integrity or reconciliation preflight; inspect [LIVE_BLOCKED_*] logs",
    }


def _private_block_decision_reason(blocked: dict[str, Any]) -> str:
    """Map the actual private execution block to an accurate telemetry reason."""
    status = str(blocked.get("status", "")).strip().upper()
    if status == "BLOCKED_MISSING_CREDENTIALS":
        return "missing_private_credentials"
    if status == "BLOCKED_PRIVATE_PREFLIGHT":
        return "private_preflight_unresolved"
    return "private_execution_blocked"


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


def _find_recent_unresolved_engine_entry(symbol: str, direction: str, *, max_age_seconds: float = 7200.0) -> dict[str, Any] | None:
    """Find a recent engine entry explicitly recorded as unresolved, never a manual position."""
    # The trade journal may not exist if the process dies on the very first
    # entry before TRADE_OPEN is written. Still inspect the execution ledger;
    # otherwise that exact crash window would bypass the identity recovery below.
    # Normalize to the exchange's compact contract symbol (AAA-USDT -> AAAUSDT).
    # Do not strip the quote asset here: BingX order lookups return the compact
    # symbol without a hyphen, so stripping USDT on one side breaks reconciliation.
    wanted_symbol = _display_symbol(symbol)
    wanted_direction = str(direction).upper()
    now = pd.Timestamp.now(tz="UTC")
    eligible = {"opened_emergency_close_unverified", "opened_registration_failed_close_unverified", "entry_state_unverified"}
    rows: list[dict[str, Any]] = []
    if TRADES_PATH.exists():
        try:
            with TRADES_PATH.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        log.error("[RECON_ORPHAN_LOOKUP] skipping malformed trade journal line")
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
        except OSError as exc:
            log.error("[RECON_ORPHAN_LOOKUP] cannot inspect trade journal: %s", exc)
            return None
    for row in reversed(rows):
        if not isinstance(row, dict):
            continue
        if _display_symbol(row.get("symbol", "")) != wanted_symbol:
            continue
        if str(row.get("direction", "")).upper() != wanted_direction:
            continue
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        if str(result.get("status", "")).lower() not in eligible:
            continue
        stamp = _coerce_utc_timestamp(result.get("execution_call_start_ts") or result.get("order_submit_ts"))
        if stamp is None or (now - stamp).total_seconds() < 0 or (now - stamp).total_seconds() > max_age_seconds:
            continue
        return {"event_id": str(row.get("event_id") or ""), "row": row, "result": result, "source": "trades_journal"}

    # The process can die after ORDER_ACK/POSITION_FILL but before writing the
    # result row in trades.jsonl. The append-only execution ledger is the second
    # recovery source for a recent entry with no local owner.
    ledger_path = EXECUTION_LEDGER_PATH
    if not ledger_path.exists():
        return None
    try:
        ledger_rows = []
        with ledger_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    ledger_rows.append(record)
    except OSError as exc:
        log.error("[RECON_ORPHAN_LOOKUP] cannot inspect execution ledger: %s", exc)
        return None
    closed_event_ids = {
        str(row.get("event_id")) for row in rows
        if row.get("event_id") and (
            row.get("record_type") == "TRADE_CLOSE"
            or (row.get("record_type") == "TRADE_ROLLBACK" and str(row.get("status", "")).lower() == "opened_then_emergency_closed")
        )
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in ledger_rows:
        event_id = str(record.get("event_id") or "")
        if event_id:
            grouped.setdefault(event_id, []).append(record)
    for event_id, records in reversed(list(grouped.items())):
        if event_id in closed_event_ids:
            continue
        start = next((r for r in records if r.get("stage") == "EXECUTION_START"), None)
        ack = next((r for r in reversed(records) if r.get("stage") == "ORDER_ACK"), None)
        identity = next((r for r in reversed(records) if r.get("stage") == "ENTRY_ORDER_IDENTITY"), None)
        if not start:
            continue
        if _display_symbol(start.get("symbol", "")) != wanted_symbol:
            continue
        if str(start.get("direction", "")).upper() != wanted_direction:
            continue
        if any(r.get("stage") == "TRADE_CLOSE" for r in records):
            continue
        recovery = next((r for r in reversed(records) if r.get("stage") == "ORPHAN_ENGINE_POSITION_RECOVERY"), None)
        if recovery and str(recovery.get("status", "")).lower() == "closed_verified":
            continue

        # A crash can occur after the exchange accepted MARKET but before the
        # ORDER_ACK ledger row was appended. Recover only if the pre-POST identity
        # breadcrumb exists and the exchange confirms that exact MARKET order.
        if ack is None:
            client_order_id = str((identity or {}).get("client_order_id") or "")
            if not client_order_id:
                continue
            try:
                observed_order = get_order(symbol, client_order_id=client_order_id)
            except Exception as exc:
                log.warning("[RECON_ORPHAN_LOOKUP] entry clientOrderId query failed for %s: %s", client_order_id, exc)
                continue
            observed_status = str(observed_order.get("order_status", "")).upper() if isinstance(observed_order, dict) else ""
            observed_side = str(observed_order.get("side", "")).upper() if isinstance(observed_order, dict) else ""
            expected_side = "BUY" if wanted_direction == "LONG" else "SELL"
            observed_symbol = _display_symbol(observed_order.get("symbol", "")) if isinstance(observed_order, dict) else ""
            if not (
                isinstance(observed_order, dict)
                and observed_order.get("status") == "ok"
                and observed_status in {"PARTIALLY_FILLED", "FILLED"}
                and observed_symbol == wanted_symbol
                and observed_side == expected_side
                and str(observed_order.get("order_type", "")).upper() == "MARKET"
                and str(observed_order.get("client_order_id", client_order_id)) == client_order_id
            ):
                continue
            ack = {
                "stage": "ORDER_ACK_RECONCILED_BY_CLIENT_ID",
                "order_id": observed_order.get("order_id"),
                "client_order_id": client_order_id,
                "ts_ms": (identity or {}).get("ts_ms"),
                "order_status": observed_status,
            }

        try:
            if ack.get("stage") == "ORDER_ACK_RECONCILED_BY_CLIENT_ID" and identity:
                stamp_ms = int(identity.get("ts_ms") or identity.get("order_submit_at_ms") or start.get("ts_ms") or 0)
            else:
                stamp_ms = int(start.get("ts_ms") or ack.get("ts_ms") or 0)
        except (TypeError, ValueError):
            stamp_ms = 0
        age_sec = (int(now.timestamp() * 1000) - stamp_ms) / 1000.0 if stamp_ms > 0 else float("inf")
        if age_sec < 0 or age_sec > min(max_age_seconds, 1800.0):
            continue
        return {
            "event_id": event_id,
            "row": start,
            "result": {"status": "entry_state_unverified", "execution_call_start_ts": (identity or {}).get("ledger_ts") if ack.get("stage") == "ORDER_ACK_RECONCILED_BY_CLIENT_ID" else start.get("ledger_ts"), "order": {"order_id": ack.get("order_id"), "client_order_id": ack.get("client_order_id")}},
            "source": "execution_ledger_client_id_reconciliation" if ack.get("stage") == "ORDER_ACK_RECONCILED_BY_CLIENT_ID" else "execution_ledger",
        }
    return None


def reconcile_all_open_positions() -> bool:
    """Reconcile positions; False means ownership/protection is not trustworthy."""
    started = time.time()
    unresolved = False
    try:
        positions = get_positions(timeout_sec=float(os.environ.get("RECONCILIATION_HTTP_TIMEOUT_SEC", "5")), retryable=False)
    except Exception as exc:
        log.error("[RECON] positions fetch failed: %s", exc)
        return False

    active = _load_active_trades_file()

    # If a previous entry could not be proven terminal, keep trying to cancel and
    # reconcile it on later engine cycles. A one-time best-effort cancellation is
    # not enough: a late fill after registration could otherwise outgrow fixed-size
    # SL/TP orders. While unresolved, the normal protection repair below still runs
    # against a fresh position snapshot, and new entries remain blocked.
    unsettled_entry_ids = [
        str(event_id) for event_id, trade in active.items()
        if isinstance(trade, dict) and not trade.get("closed")
        and _trade_has_unverified_entry_remainder(trade)
    ]
    if unsettled_entry_ids:
        reconciliation_changed = False
        for event_id in unsettled_entry_ids:
            trade = active.get(event_id, {})
            symbol = str(trade.get("symbol", ""))
            entry_order = trade.get("entry_order") if isinstance(trade.get("entry_order"), dict) else {}
            try:
                settlement = _settle_entry_order_before_protection(
                    symbol, entry_order,
                    timeout_sec=max(0.5, float(os.environ.get("ENTRY_ORDER_RECONCILE_TIMEOUT_SEC", "0.75"))),
                )
            except Exception as exc:
                settlement = {"status": "unverified", "error": f"{type(exc).__name__}: {exc}"}
            if not update_active_trade_entry_reconciliation(event_id, settlement):
                unresolved = True
                log.critical(
                    "[RECON_ENTRY_REMAINDER_STATE] event=%s symbol=%s could not persist settlement evidence; entries blocked",
                    event_id, symbol,
                )
            elif settlement.get("status") != "settled":
                unresolved = True
                log.critical(
                    "[RECON_ENTRY_REMAINDER_PENDING] event=%s symbol=%s status=%s error=%s; retaining ownership and retrying cancellation",
                    event_id, symbol, settlement.get("status"), settlement.get("error"),
                )
            else:
                log.warning(
                    "[RECON_ENTRY_REMAINDER_SETTLED] event=%s symbol=%s status=%s; position snapshot will be refreshed",
                    event_id, symbol, settlement.get("order_status"),
                )
            reconciliation_changed = True
        if reconciliation_changed:
            # Cancellation may race a partial fill. Never size protection from the
            # position list captured before the cancellation/settlement attempts.
            try:
                positions = get_positions(
                    timeout_sec=float(os.environ.get("RECONCILIATION_HTTP_TIMEOUT_SEC", "5")),
                    retryable=False,
                )
                active = _load_active_trades_file()
            except Exception as exc:
                log.critical("[RECON_ENTRY_REMAINDER_POSITION_REFRESH] failed after entry settlement checks: %s", exc)
                return False

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

        candidates = [
            (event_id, trade) for event_id, trade in active.items()
            if isinstance(trade, dict) and not trade.get("closed")
            and _display_symbol(trade.get("symbol", "")) == _display_symbol(symbol)
            and str(trade.get("direction", "")).upper() == side
        ]
        if len(candidates) != 1:
            owner_ids = [str(event_id) for event_id, _ in candidates]
            same_symbol_other_side = [
                (str(event_id), str(trade.get("direction", "")).upper())
                for event_id, trade in active.items()
                if isinstance(trade, dict) and not trade.get("closed")
                and _display_symbol(trade.get("symbol", "")) == _display_symbol(symbol)
                and str(trade.get("direction", "")).upper() != side
            ]
            if not candidates and same_symbol_other_side:
                status = "FOUND_SIDE_MISMATCH"
                message = f"exchange position {symbol}:{side} conflicts with local owner direction(s): {same_symbol_other_side}"
            else:
                status = "ORPHAN_EXCHANGE_POSITION" if not candidates else "STATE_CONFLICT"
                message = (
                    f"exchange position {symbol}:{side} has no unique local owner"
                    if not candidates else
                    f"exchange position {symbol}:{side} has multiple local owners: {owner_ids}"
                )
            log.critical("[RECON_%s] %s qty=%s avg=%s owners=%s", status, message, qty, avg, owner_ids)
            telemetry.record_position_reconciliation(
                event_id=owner_ids[0] if len(owner_ids) == 1 else None,
                attempt_id=None, position_id=None, symbol=symbol, direction=side,
                status=status, internal_remaining_qty=None, exchange_position_qty=qty,
                exchange_avg_price=avg, local_owner_event_ids=owner_ids,
            )
            if not candidates and not same_symbol_other_side:
                unresolved_entry = _find_recent_unresolved_engine_entry(symbol, side)
                if unresolved_entry and unresolved_entry.get("event_id"):
                    event_id = str(unresolved_entry["event_id"])
                    log.critical("[RECON_ORPHAN_ENGINE_RECOVERY] attempting emergency close for known unresolved engine entry event=%s symbol=%s side=%s qty=%s", event_id, symbol, side, qty)
                    try:
                        close_result = _emergency_close_and_verify(symbol, side, qty, event_id)
                    except Exception as close_exc:
                        close_result = {"status": "close_unverified", "error": str(close_exc)}
                    if str(close_result.get("status", "")).lower() == "closed_verified":
                        try:
                            cleanup_result = _cancel_engine_protection_before_emergency_close(symbol, side)
                        except Exception as cleanup_exc:
                            cleanup_result = {"status": "error", "error": str(cleanup_exc)}
                        _append_execution_ledger(event_id, None, "ORPHAN_ENGINE_POSITION_RECOVERY", {
                            "status": "closed_verified", "symbol": symbol, "direction": side,
                            "previous_exchange_qty": qty, "previous_exchange_avg_price": avg,
                            "close": close_result, "protection_cleanup": cleanup_result,
                        })
                        telemetry.record_position_reconciliation(
                            event_id=event_id, attempt_id=None, position_id=event_id,
                            symbol=symbol, direction=side, status="ORPHAN_ENGINE_POSITION_CLOSED_VERIFIED",
                            internal_remaining_qty=qty, exchange_position_qty=0.0,
                            exchange_avg_price=avg, position_gone=True,
                        )
                        log.critical("[RECON_ORPHAN_ENGINE_RECOVERY] close verified event=%s symbol=%s side=%s", event_id, symbol, side)
                        continue
                    unresolved = True
                    _append_execution_ledger(event_id, None, "ORPHAN_ENGINE_POSITION_RECOVERY", {
                        "status": "close_unverified", "symbol": symbol, "direction": side,
                        "exchange_qty": qty, "exchange_avg_price": avg, "close": close_result,
                        "action": "left_existing_exchange_protection_untouched",
                    })
                    log.critical("[RECON_ORPHAN_ENGINE_RECOVERY_FAILED] event=%s symbol=%s side=%s qty=%s; preserving exchange protection and blocking new entries", event_id, symbol, side, qty)
                else:
                    unresolved = True
            if candidates or same_symbol_other_side:
                unresolved = True
            if candidates:
                telemetry.record_state_conflict(
                    event_id=None, attempt_id=None, position_id=None, order_id=None,
                    symbol=symbol, direction=side, conflict_type="MULTIPLE_LOCAL_POSITION_OWNERS",
                    message=message, owner_event_ids=owner_ids,
                )
            continue

        event_id, trade = candidates[0]
        setup = trade.get("setup", {}) if isinstance(trade.get("setup"), dict) else {}
        stop_loss_pct = _reconciliation_stop_loss_pct(trade)
        if stop_loss_pct is None:
            unresolved = True
            log.critical(
                "[RECON_RISK_STATE_UNTRUSTED] event=%s symbol=%s side=%s; "
                "planned risk is missing or invalid, leaving exchange protection untouched and blocking entries",
                event_id, symbol, side,
            )
            telemetry.record_state_conflict(
                event_id=event_id, attempt_id=trade.get("attempt_id"), position_id=trade.get("position_id") or event_id,
                order_id=None, symbol=symbol, direction=side,
                conflict_type="INVALID_PLANNED_RISK_STATE",
                message="cannot safely rebuild directional protection without a valid planned_risk_pct or setup.risk_pct",
                owner_event_ids=[event_id],
            )
            continue
        tp_levels = setup.get("tp_levels") if isinstance(setup.get("tp_levels"), list) else None
        if not tp_levels:
            tp_levels = [
                {"leg": "tp1", "pnl_pct": TP1_PCT, "close_fraction": 0.50},
                {"leg": "tp2", "pnl_pct": TP2_PCT, "close_fraction": 0.50},
            ]

        hit_legs = set(trade.get("hit_legs", []))

        # Recover original TP identities from immutable setup provenance before
        # rebuilding protection. A current tp_orders entry may have been replaced
        # by an error record (no order_id) even though the original exchange order
        # later filled. Never infer a TP fill from position-quantity deltas alone.
        original_protection = setup.get("protection_result", {}) if isinstance(setup.get("protection_result"), dict) else {}
        original_tp_orders = original_protection.get("tp_orders", []) if isinstance(original_protection.get("tp_orders"), list) else []
        active_tp_orders = trade.get("tp_orders", []) if isinstance(trade.get("tp_orders"), list) else []
        active_ids = {str(x.get("order_id")) for x in active_tp_orders if isinstance(x, dict) and x.get("order_id")}
        active_tp_client_ids = {
            str(x.get("client_order_id") or x.get("clientOrderId") or "").strip().casefold()
            for x in active_tp_orders if isinstance(x, dict) and x.get("order_id")
            and (x.get("client_order_id") or x.get("clientOrderId"))
        }
        completed_tp_legs = {str(x).casefold() for x in (trade.get("completed_tp_legs") or [])}
        recovered_filled_orders = []
        for original_tp in original_tp_orders:
            if not isinstance(original_tp, dict):
                continue
            original_id = str(original_tp.get("order_id") or "")
            leg = str(original_tp.get("leg") or "")
            original_client_id = str(original_tp.get("client_order_id") or original_tp.get("clientOrderId") or "").strip().casefold()
            if (
                not original_id or original_id in active_ids or leg.casefold() in hit_legs
                or leg.casefold() in completed_tp_legs
                or (original_client_id and original_client_id in active_tp_client_ids)
            ):
                continue
            try:
                evidence = get_order_by_identity(
                    symbol, original_id,
                    client_order_id=str(original_tp.get("client_order_id") or "") or None,
                    order_lookup=get_order,
                )
            except Exception as exc:
                log.warning("[RECON_TP_IDENTITY] order lookup failed symbol=%s event=%s leg=%s id=%s error=%s", symbol, event_id, leg, original_id, exc)
                continue
            if (isinstance(evidence, dict) and evidence.get("status") == "ok"
                    and str(evidence.get("order_status", "")).upper() == "FILLED"
                    and float(evidence.get("executed_qty") or 0.0) > 0):
                recovered = dict(original_tp)
                recovered["status"] = "FILLED"
                if evidence.get("order_id"):
                    recovered["previous_exchange_order_id"] = original_id
                    recovered["order_id"] = str(evidence.get("order_id"))
                recovered["client_order_id"] = str(original_tp.get("client_order_id") or evidence.get("client_order_id") or "")
                recovered["executed_qty"] = float(evidence.get("executed_qty") or 0.0)
                recovered["avg_price"] = float(evidence.get("avg_price") or 0.0)
                recovered["exchange_update_time_ms"] = evidence.get("update_time_ms")
                recovered["recovered_execution_pending"] = True
                recovered_filled_orders.append(recovered)

        # The pending state is durable across reconciliation cycles/restarts.
        # Once the recovered order has been merged into active_trades, its ID is
        # now in active_ids and will not be re-discovered above; derive the gate
        # from persisted state too, otherwise the next pass can recreate TPs before
        # tracker has booked TP1 and moved the stop to break-even.
        persisted_recovery_pending = bool(trade.get("tp_recovery_pending")) or any(
            isinstance(x, dict) and x.get("recovered_execution_pending")
            for x in active_tp_orders
        )
        tp_recovery_pending = bool(recovered_filled_orders) or persisted_recovery_pending
        if recovered_filled_orders:
            if not merge_reconciled_tp_orders(str(event_id), recovered_filled_orders):
                log.error("[RECON_TP_IDENTITY] could not persist recovered TP identities event=%s symbol=%s", event_id, symbol)
                continue
            log.warning("[RECON_TP_IDENTITY] recovered exchange-filled TP order identity event=%s symbol=%s legs=%s; deferring TP recreation until tracker books execution", event_id, symbol, [x.get("leg") for x in recovered_filled_orders])

        remaining_levels = [x for x in tp_levels if str(x.get("leg", "")) not in hit_legs]
        # Keep the original strategy fractions. Renormalizing all remaining legs
        # to 100% can turn a missed TP1 into a second TP allocation on only half
        # the position (the ACHUSDT 449/448 failure mode).
        if tp_recovery_pending:
            # Repair/verify the stop for the actual exchange quantity now, but do
            # not place any new TP until tracker has booked the recovered fill and
            # activated BE where appropriate.
            tp_levels = []
        else:
            tp_levels = [
                {"leg": str(x.get("leg")), "pnl_pct": float(x.get("pnl_pct", 0.0)), "close_fraction": float(x.get("close_fraction", 0.0))}
                for x in remaining_levels
            ]

        try:
            try:
                result = ensure_directional_protection(
                    symbol, side, avg, qty, stop_loss_pct, tp_levels, trade_id=str(event_id)
                )
            except Exception as protection_exc:
                result = {
                    "status": "PROTECTION_FAILED",
                    "error": f"ensure_directional_protection raised: {type(protection_exc).__name__}: {protection_exc}",
                }
                log.exception("[RECON_PROTECTION_EXCEPTION] %s %s event=%s; escalating to emergency close", symbol, side, event_id)
            if tp_recovery_pending and result.get("status") == "PROTECTED" and not result.get("tp_orders"):
                # A confirmed SL alone is not a complete TP recovery. Keep the SL,
                # but make the incomplete state visible to reconciliation and block
                # new entries until tracker books the recovered fill and restores TP.
                result["status"] = "SL_ONLY"
                result["error"] = "tp_recovery_pending: TP recreation deferred until recovered fill is booked"
            if result.get("status") in {"PROTECTED", "SL_ONLY"}:
                updated = update_active_trade_protection(
                    symbol, side, result.get("tp_orders", []), result.get("sl_result", {}),
                    result.get("effective_tp_levels", []), result.get("tp_mode"),
                    result.get("effective_weighted_rr"), event_id=str(event_id),
                )
                if not updated:
                    unresolved = True
                    telemetry.record_state_conflict(
                        event_id=str(event_id), attempt_id=trade.get("attempt_id"),
                        position_id=trade.get("position_id") or str(event_id), order_id=None,
                        symbol=symbol, direction=side, conflict_type="PROTECTION_UPDATE_OWNER_LOST",
                        message="unique owner disappeared or no longer matched before protection state update",
                        owner_event_ids=[str(event_id)],
                    )
                if result.get("status") == "SL_ONLY":
                    # A verified SL is valuable protection and must not be removed
                    # by an unnecessary emergency close, but missing TP protection
                    # means the trade is not fully reconciled. Keep SL, persist any
                    # TP repairs made, and block new entries until a later cycle
                    # proves the complete protection set.
                    unresolved = True
                    telemetry.record_state_conflict(
                        event_id=str(event_id), attempt_id=trade.get("attempt_id"),
                        position_id=trade.get("position_id") or str(event_id), order_id=None,
                        symbol=symbol, direction=side, conflict_type="PROTECTION_TP_INCOMPLETE",
                        message="SL is verified but one or more expected TP legs are not fully verified; keeping the SL and retrying reconciliation",
                        owner_event_ids=[str(event_id)],
                    )
                    _append_execution_ledger(str(event_id), None, "RECON_PROTECTION_PARTIAL", {
                        "status": "SL_ONLY", "symbol": symbol, "direction": side,
                        "error": result.get("error"), "action": "keep_verified_sl_retry_missing_tp_and_block_new_entries",
                    })
                    log.error("[RECON_PROTECTION_PARTIAL] %s %s event=%s: verified SL retained; TP protection incomplete, blocking new entries until repaired", symbol, side, event_id)
            else:
                unresolved = True
                log.critical("[RECON_PROTECTION_UNVERIFIED] %s %s event=%s status=%s error=%s; attempting verified emergency close", symbol, side, event_id, result.get("status"), result.get("error"))
                try:
                    close_result = _emergency_close_and_verify(symbol, side, qty, str(event_id))
                except Exception as close_exc:
                    close_result = {"status": "close_unverified", "error": str(close_exc)}
                if str(close_result.get("status", "")).lower() == "closed_verified":
                    try:
                        record_verified_emergency_close(str(event_id), close_result, reason="EMERGENCY_CLOSE_AFTER_PROTECTION_FAILURE")
                    except Exception as state_exc:
                        log.critical("[RECON_EMERGENCY_CLOSE_STATE] event=%s verified flat but could not persist evidence: %s", event_id, state_exc)
                    try:
                        cleanup_result = _cancel_engine_protection_before_emergency_close(symbol, side)
                    except Exception as cleanup_exc:
                        cleanup_result = {"status": "error", "error": str(cleanup_exc)}
                    _append_execution_ledger(str(event_id), None, "RECON_PROTECTION_FAILURE_EMERGENCY_CLOSE", {"status": "closed_verified", "symbol": symbol, "direction": side, "close": close_result, "protection_cleanup": cleanup_result})
                else:
                    _append_execution_ledger(str(event_id), None, "RECON_PROTECTION_FAILURE_EMERGENCY_CLOSE", {"status": "close_unverified", "symbol": symbol, "direction": side, "close": close_result, "action": "existing_exchange_protection_left_untouched"})
                    log.critical("[RECON_PROTECTION_CLOSE_UNVERIFIED] event=%s symbol=%s side=%s; preserving any exchange orders and blocking entries", event_id, symbol, side)
        except Exception as exc:
            unresolved = True
            log.exception("[RECON] protection repair failed for %s %s event=%s: %s", symbol, side, event_id, exc)

    # Second pass: local active trades missing from the exchange position snapshot.
    exchange_keys = _position_keys(positions)
    normalized_exchange_keys = {
        (str(symbol).upper().replace("-USDT", "").replace("-", ""), str(direction).upper())
        for symbol, direction in exchange_keys
    }
    for event_id, trade in active.items():
        if not isinstance(trade, dict) or trade.get("closed"):
            continue
        symbol = str(trade.get("symbol", "")).upper()
        direction = str(trade.get("direction", "")).upper()
        norm_symbol = symbol.replace("-USDT", "").replace("-", "")
        if not symbol or direction not in {"LONG", "SHORT"}:
            continue
        if (norm_symbol, direction) not in normalized_exchange_keys:
            telemetry.record_position_reconciliation(
                event_id=event_id, attempt_id=trade.get("attempt_id"), position_id=trade.get("position_id") or event_id,
                symbol=symbol, direction=direction, status="ORPHAN_LOCAL_POSITION",
                internal_remaining_qty=float(trade.get("remaining_qty", 0.0) or 0.0),
                exchange_position_qty=0.0, exchange_avg_price=None,
                local_tp_filled_qty=trade.get("tp_filled_qty") or {},
                local_be_activated=bool(trade.get("be_activated")),
                local_sl_order_id=(trade.get("sl_order") or {}).get("order_id") if isinstance(trade.get("sl_order"), dict) else None,
                exchange_error=None, position_gone=True,
            )
            unresolved = True

    return not unresolved


def _load_active_trades_file(*, fresh_runtime_start: bool = False) -> dict[str, dict]:
    path = DATA / "active_trades.json"

    def quarantine(reason: str) -> None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        quarantine_path = path.with_name(f"{path.name}.quarantine.{stamp}")
        suffix = 1
        while quarantine_path.exists():
            quarantine_path = path.with_name(f"{path.name}.quarantine.{stamp}.{suffix}")
            suffix += 1
        try:
            shutil.copy2(path, quarantine_path)
            log.critical("[STATE_CORRUPT] active_trades quarantined copy=%s reason=%s", quarantine_path, reason)
        except Exception as exc:
            log.critical("[STATE_CORRUPT] active_trades quarantine failed: %s", exc)

    def fail(reason: str) -> None:
        quarantine(reason)
        raise RuntimeError(f"active_trades.json is untrusted: {reason}")

    if not path.exists():
        if fresh_runtime_start:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text_durable(path, "{}\n")
            log.warning("[STATE_BOOTSTRAP] fresh runtime start detected; initialized empty active_trades.json")
            return {}
        fail("active_trades.json is missing")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"invalid JSON: {type(exc).__name__}: {exc}")

    if not isinstance(raw, dict):
        fail(f"root={type(raw).__name__}; expected object")

    out: dict[str, dict] = {}
    symbol_direction_owners: dict[tuple[str, str], str] = {}
    order_owners: dict[str, str] = {}
    for raw_event_id, trade in raw.items():
        try:
            validate_active_trade_record(raw_event_id, trade)
        except ValueError as exc:
            fail(str(exc))
        if trade.get("closed"):
            continue
        event_id = str(trade.get("event_id") or raw_event_id)
        sym = str(trade.get("symbol", "")).upper()
        side = str(trade.get("direction", "")).upper()
        if not event_id or not sym or side not in {"LONG", "SHORT"}:
            fail(f"invalid active trade state for {raw_event_id!r}")
        if event_id in out:
            fail(f"duplicate active event_id={event_id!r}")
        if str(raw_event_id) != event_id:
            fail(f"state key/event_id mismatch key={raw_event_id!r} event_id={event_id!r}")
        try:
            initial_qty = float(trade.get("initial_qty", 0) or 0)
            remaining_qty = float(trade.get("remaining_qty", 0) or 0)
        except (TypeError, ValueError):
            fail(f"non-numeric quantities for event_id={event_id!r}")
        if not math.isfinite(initial_qty) or initial_qty <= 0 or not math.isfinite(remaining_qty) or remaining_qty < 0:
            fail(f"invalid quantities for event_id={event_id!r}: initial={initial_qty!r} remaining={remaining_qty!r}")

        identity = (sym.replace("-USDT", "").replace("-", ""), side)
        previous_owner = symbol_direction_owners.get(identity)
        if previous_owner is not None and previous_owner != event_id:
            fail(f"cross-event position ownership collision {identity}: {previous_owner} vs {event_id}")
        symbol_direction_owners[identity] = event_id

        ids: set[str] = set()
        containers = [
            trade.get("entry_order") if isinstance(trade.get("entry_order"), dict) else {},
            trade.get("sl_order") if isinstance(trade.get("sl_order"), dict) else {},
        ]
        containers += [x for x in (trade.get("tp_orders") if isinstance(trade.get("tp_orders"), list) else []) if isinstance(x, dict)]
        for key in ("be_order_id", "exit_order_id"):
            if trade.get(key):
                ids.add(str(trade[key]))
        for container in containers:
            for key in ("order_id", "orderId", "orderID"):
                if container.get(key):
                    ids.add(str(container[key]))
        for oid in ids:
            previous_order_owner = order_owners.get(oid)
            if previous_order_owner is not None and previous_order_owner != event_id:
                fail(f"cross-event exchange order ownership collision order_id={oid}: {previous_order_owner} vs {event_id}")
            order_owners[oid] = event_id

        out[event_id] = trade
    return out


def _live_state_preflight(signal: dict[str, Any]) -> tuple[bool, str]:
    """Return whether local active state is trusted enough to allow a live order.

    State integrity is an execution safety prerequisite.  A corrupt/unreadable or
    internally conflicting active-trade file must block the order before any exchange
    write.  This check is repeated inside execute_new_position so callers cannot bypass
    the main-loop preflight.
    """
    try:
        _load_active_trades_file()
        return True, "ok"
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        event_id = str(signal.get("event_id") or "")
        symbol = str(signal.get("symbol") or "")
        direction = str(signal.get("type") or "").upper()
        try:
            telemetry.record_state_conflict(
                event_id=event_id or None,
                attempt_id=signal.get("attempt_id"),
                position_id=event_id or None,
                order_id=None,
                symbol=symbol,
                direction=direction or "UNKNOWN",
                conflict_type="LIVE_EXECUTION_BLOCKED_STATE_UNTRUSTED",
                message=reason,
                owner_event_ids=[],
            )
        except Exception:
            pass
        _append_execution_ledger(event_id, signal.get("attempt_id"), "EXECUTION_BLOCKED_STATE", {
            "status": "BLOCKED_STATE_UNTRUSTED",
            "error": reason,
        })
        try:
            _append_jsonl(ACTIONS_PATH, {
                "ts": int(time.time() * 1000),
                "action": "LIVE_EXECUTION_BLOCKED_STATE_UNTRUSTED",
                "event_id": event_id,
                "symbol": symbol,
                "direction": direction,
                "error": reason,
            })
        except Exception as exc:
            log.error("[STATE_BLOCK_TELEMETRY_WRITE_FAILED] event_id=%s error=%s", event_id, exc)
        log.critical(
            "[LIVE_BLOCKED_STATE] %s %s event=%s | active-trade state is untrusted: %s",
            symbol, direction, event_id, reason,
        )
        return False, reason


def _build_setup(signal: dict[str, Any]) -> dict[str, Any]:
    risk_pct = float(signal["risk_pct"])
    return {
        "engine_version": ENGINE_VERSION,
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
        "attempt_id": signal.get("attempt_id"),
    }



def _validate_trade_geometry(
    signal: dict[str, Any], *, require_structure_room: bool = True,
) -> tuple[bool, str]:
    """Validate protection geometry; optionally enforce the pre-entry structural-room filter."""
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
    if require_structure_room and obstacle_price is None and REQUIRE_STRUCTURE_OBSTACLE:
        return False, "missing_structural_obstacle"
    if require_structure_room and obstacle_price is not None and risk_abs > 0:
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


def _cancel_engine_protection_before_emergency_close(
    symbol: str,
    direction: str,
    known_order_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Best-effort cleanup of engine-owned SL/TP orders before a MARKET rollback.

    When exact protection order ids are known, restrict cancellation to those ids so
    a failed registration cannot accidentally cancel a different event's protection.
    """
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
        if known_order_ids is not None and oid not in {str(x) for x in known_order_ids}:
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
    """Close a safety-rollback position without overlapping ambiguous MARKET closes.

    A second close POST is allowed only when the preceding request is proven to
    have been rejected (non-zero exchange response) or its order is verified in
    a terminal non-fill state. A transport error, missing order ID, accepted
    order still pending, or failed order lookup is ambiguous and must never
    trigger a blind second MARKET close.
    """
    attempts: list[dict[str, Any]] = []
    configured_attempts = max(1, int(os.environ.get("EMERGENCY_CLOSE_ATTEMPTS", "4")))
    verify_polls = max(2, int(os.environ.get("EMERGENCY_CLOSE_VERIFY_POLLS", "6")))
    last_qty = max(0.0, float(qty or 0.0))
    previous_close: dict[str, Any] | None = None

    def _order_id_from_close(close_result: dict[str, Any] | None) -> str | None:
        if not isinstance(close_result, dict):
            return None
        response = close_result.get("response")
        if isinstance(response, dict):
            data = response.get("data")
            if isinstance(data, dict):
                order = data.get("order") if isinstance(data.get("order"), dict) else data
                if isinstance(order, dict) and order.get("orderId") not in (None, ""):
                    return str(order.get("orderId"))
        if close_result.get("order_id") not in (None, ""):
            return str(close_result.get("order_id"))
        return None

    def _previous_close_definitively_rejected(close_result: dict[str, Any] | None) -> bool:
        if not isinstance(close_result, dict):
            return False
        # close_position_market returns a response only when the exchange sent a
        # definite non-zero error code. Exceptions/timeouts carry no response and
        # remain ambiguous: the order may have reached the exchange.
        response = close_result.get("response")
        if close_result.get("status") == "error" and isinstance(response, dict):
            code = response.get("code")
            # _request uses -1 as a local sentinel for missing credentials,
            # transport errors and timeouts. That is not an exchange rejection: the
            # MARKET POST outcome is unknown, so reconcile only and never resubmit.
            if str(code) == "-1" or code in (None, 0, "0"):
                return False
            return True
        order_id = _order_id_from_close(close_result)
        if not order_id:
            return False
        try:
            order_info = get_order(symbol, order_id)
        except Exception as exc:
            attempts.append({"verification": "prior_close_order", "order_id": order_id, "status": "query_error", "error": str(exc)})
            return False
        attempts.append({"verification": "prior_close_order", "order_id": order_id, "order": order_info})
        if not isinstance(order_info, dict) or order_info.get("status") != "ok":
            return False
        order_status = str(order_info.get("order_status", "")).upper()
        if order_status not in {"REJECTED", "CANCELED", "CANCELLED", "EXPIRED"}:
            return False
        # A canceled/expired order may have partially filled. Do not submit the
        # previous full quantity again unless the exchange confirms zero executed
        # quantity; otherwise a stale position snapshot could cause an oversized
        # second close. The next retry re-reads the current directional quantity.
        executed_qty_known = order_info.get("executed_qty_known")
        if executed_qty_known is False:
            return False
        # Test doubles and older connector wrappers may not expose the explicit
        # known flag; require the field to be present in that case. The live
        # BingX get_order wrapper sets executed_qty_known based on raw evidence.
        if executed_qty_known is None and "executed_qty" not in order_info:
            return False
        try:
            executed_qty = abs(float(order_info.get("executed_qty", 0.0) or 0.0))
        except (TypeError, ValueError):
            return False
        return executed_qty <= 1e-12

    for attempt in range(configured_attempts):
        try:
            current = get_position_directional(symbol, direction)
        except Exception as exc:
            current = {"status": "error", "error": str(exc)}
        if current.get("status") == "not_found":
            return {"status": "closed_verified", "attempts": attempts, "verification": current}
        if current.get("status") != "found":
            attempts.append({"status": "position_check_error", "error": current.get("error", "position not authoritatively found")})
            # Never submit based on an unknown position state.
            break
        raw_current_qty = current.get("positionAmt")
        if raw_current_qty is not None:
            try:
                last_qty = abs(float(raw_current_qty))
            except (TypeError, ValueError):
                pass
        if last_qty <= 0:
            return {"status": "closed_verified", "attempts": attempts, "verification": current}

        if previous_close is not None and not _previous_close_definitively_rejected(previous_close):
            # Previous POST may still be accepted/in-flight or may have succeeded
            # while the position endpoint is lagging. Switch to verification-only.
            break

        try:
            close_result = close_position_market(symbol, direction, last_qty, trade_id=trade_id)
        except Exception as exc:
            close_result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        attempts.append(close_result)
        previous_close = close_result
        # Avoid tight resubmission; on next iteration we re-read position and
        # require explicit rejection evidence before issuing another MARKET POST.
        time.sleep(0.35 * (attempt + 1))

    verification: dict[str, Any] = {"status": "verification_error", "error": "no verification attempted"}
    for poll in range(verify_polls):
        try:
            verification = get_position_directional(symbol, direction)
        except Exception as exc:
            verification = {"status": "verification_error", "error": str(exc)}
        if verification.get("status") == "not_found":
            return {"status": "closed_verified", "attempts": attempts, "verification": verification, "verify_poll": poll + 1}
        if verification.get("status") == "found":
            raw_verified_qty = verification.get("positionAmt")
            if raw_verified_qty is not None:
                try:
                    last_qty = abs(float(raw_verified_qty))
                except (TypeError, ValueError):
                    pass
        time.sleep(0.35)

    return {
        "status": "close_unverified",
        "attempts": attempts,
        "verification": verification,
        "remaining_qty": last_qty if verification.get("status") == "found" else None,
    }


def _emergency_close_preserving_protection(
    symbol: str,
    direction: str,
    qty: float,
    trade_id: str,
    *,
    known_order_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attempt a verified close while retaining existing SL/TP until flatness is proven."""
    try:
        close_result = _emergency_close_and_verify(symbol, direction, qty, trade_id)
    except Exception as exc:
        close_result = {"status": "close_unverified", "attempts": [], "error": f"{type(exc).__name__}: {exc}"}
    if str(close_result.get("status", "")).lower() == "closed_verified":
        try:
            cleanup = _cancel_engine_protection_before_emergency_close(
                symbol, direction, known_order_ids=known_order_ids,
            )
        except Exception as exc:
            cleanup = {"status": "error", "cancelled": [], "errors": [str(exc)]}
    else:
        # Cancelling the last known SL/TP before an unverified close can leave a
        # still-open position naked. Preserve exchange protection and retry close.
        cleanup = {"status": "deferred_until_flat_verified", "cancelled": [], "errors": []}
    return close_result, cleanup


def _rollback_after_registration_failure(
    *,
    event_id: str,
    symbol: str,
    direction: str,
    qty: float,
    reason: str,
    order: dict[str, Any] | None,
    position: dict[str, Any] | None,
    protection: dict[str, Any] | None,
    executed_signal: dict[str, Any] | None,
) -> dict[str, Any]:
    """Safety rollback when a real protected entry cannot acquire a local owner.

    This is intentionally exception-safe: state persistence failures, duplicate-owner
    rejection, and unexpected registration exceptions all converge to the same explicit
    rollback path.  The result is deterministic: either the exchange position is
    verified closed, or the result is explicitly marked unverified.
    """
    try:
        telemetry.record_state_conflict(
            event_id=event_id,
            attempt_id=None,
            position_id=event_id,
            order_id=(order or {}).get("order_id") if isinstance(order, dict) else None,
            symbol=symbol,
            direction=direction,
            conflict_type="ACTIVE_TRADE_REGISTRATION_FAILED_AFTER_PROTECTED_ENTRY",
            message=str(reason),
            owner_event_ids=[event_id],
        )
    except Exception:
        pass

    _append_execution_ledger(
        event_id,
        None,
        "ACTIVE_TRADE_REGISTRATION_FAILED",
        {
            "status": "ROLLBACK_REQUIRED",
            "error": str(reason),
            "entry_order_id": (order or {}).get("order_id") if isinstance(order, dict) else None,
            "fill_price": (position or {}).get("avgPrice") if isinstance(position, dict) else None,
            "fill_qty": (position or {}).get("positionAmt") if isinstance(position, dict) else qty,
        },
    )

    known_ids: set[str] = set()
    if isinstance(protection, dict):
        sl_result = protection.get("sl_result") if isinstance(protection.get("sl_result"), dict) else {}
        for container in [sl_result]:
            for key in ("order_id", "orderId", "orderID"):
                if container.get(key):
                    known_ids.add(str(container[key]))
        for tp in protection.get("tp_orders", []) if isinstance(protection.get("tp_orders"), list) else []:
            if not isinstance(tp, dict):
                continue
            for key in ("order_id", "orderId", "orderID"):
                if tp.get(key):
                    known_ids.add(str(tp[key]))

    close_result, cleanup = _emergency_close_preserving_protection(
        symbol, direction, qty, event_id, known_order_ids=known_ids or None,
    )

    rollback_verified = str(close_result.get("status")) == "closed_verified"
    final_status = "opened_then_emergency_closed" if rollback_verified else "opened_registration_failed_close_unverified"
    _append_execution_ledger(
        event_id,
        None,
        "REGISTRATION_ROLLBACK_RESULT",
        {
            "status": final_status,
            "rollback_verified": rollback_verified,
            "protection_cleanup": cleanup,
            "close": close_result,
        },
    )
    try:
        telemetry.record_position_reconciliation(
            event_id=event_id,
            attempt_id=None,
            position_id=event_id,
            symbol=symbol,
            direction=direction,
            status="ROLLBACK_CLOSED" if rollback_verified else "ROLLBACK_CLOSE_UNVERIFIED",
            internal_remaining_qty=qty,
            exchange_position_qty=0.0 if rollback_verified else close_result.get("remaining_qty"),
            exchange_avg_price=None,
            local_owner_event_id=None,
            rollback_reason=str(reason),
            protection_cleanup_status=cleanup.get("status"),
        )
    except Exception as exc:
        _append_execution_ledger(event_id, None, "REGISTRATION_ROLLBACK_TELEMETRY_FAILED", {"error": f"{type(exc).__name__}: {exc}"})
    try:
        _append_jsonl(
            TRADES_PATH,
            {
                "record_type": "TRADE_ROLLBACK",
                "event_id": event_id,
                "symbol": symbol,
                "direction": direction,
                "entry_order": order or {},
                "fill_position": position or {},
                "protection": protection or {},
                "rollback_reason": str(reason),
                "protection_cleanup": cleanup,
                "close": close_result,
                "status": final_status,
                "ts_ms": int(time.time() * 1000),
            },
        )
    except Exception as exc:
        _append_execution_ledger(event_id, None, "REGISTRATION_ROLLBACK_JOURNAL_FAILED", {"error": f"{type(exc).__name__}: {exc}"})
    return {
        "status": final_status,
        "error": str(reason),
        "order": order,
        "position": position,
        "protection": protection,
        "protection_cleanup": cleanup,
        "close": close_result,
        "executed_signal": executed_signal,
        "rollback_verified": rollback_verified,
    }

def _adverse_signal_drift_pct(signal_entry: float, executable_price: float, direction: str) -> float:
    """Economic adverse entry drift relative to the original signal price.

    LONG entries are economically worse when the executable price is above the
    signal price; SHORT entries are economically worse when the executable price
    is below the signal price. A separate absolute displacement gate protects
    against a stale/broken thesis moving too far in either direction.
    """
    if signal_entry <= 0 or executable_price <= 0:
        return float("inf")
    if str(direction).upper() == "LONG":
        return max(0.0, executable_price - signal_entry) / signal_entry * 100.0
    if str(direction).upper() == "SHORT":
        return max(0.0, signal_entry - executable_price) / signal_entry * 100.0
    return float("inf")


def _signal_price_displacement_pct(signal_entry: float, executable_price: float) -> tuple[float, str]:
    """Absolute signal-to-executable movement, independent of trade direction."""
    if signal_entry <= 0 or executable_price <= 0:
        return float("inf"), "UNKNOWN"
    delta_pct = (executable_price - signal_entry) / signal_entry * 100.0
    if abs(delta_pct) <= 1e-12:
        return 0.0, "UNCHANGED"
    return abs(delta_pct), "UP" if delta_pct > 0 else "DOWN"


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


def _emergency_execution_status(close_result: dict[str, Any] | None) -> str:
    """Return a closed status only when the exchange position is verified flat."""
    if isinstance(close_result, dict) and str(close_result.get("status", "")).lower() == "closed_verified":
        return "opened_then_emergency_closed"
    return "opened_emergency_close_unverified"


def _is_stale_entry_rejection(execution: dict[str, Any]) -> bool:
    """Whether a rejected stale entry should release its pending zone visit."""
    status = str(execution.get("status", "")).lower()
    error = str(execution.get("error", ""))
    return (
        execution.get("stale_trigger_reason") in {"5m_trigger_too_old", "invalid_5m_trigger_time"}
        or (status == "skipped_stale_signal" and "signal_price_displacement_pct" in error)
    )


PROTECTED_EXECUTION_STATUSES = frozenset({"opened_protected", "entry_state_unverified_protected"})


def _is_protected_execution_status(execution: dict[str, Any]) -> bool:
    """True when a live position was registered with verified protection.

    The entry remainder may still be unresolved; that is a protected, owned live
    position requiring reconciliation, not a failed signal eligible for replay.
    """
    return str(execution.get("status", "")).strip().lower() in PROTECTED_EXECUTION_STATUSES


def _is_terminal_execution_failure(execution: dict[str, Any]) -> bool:
    status = str(execution.get("status", "")).lower()
    error = str(execution.get("error", "")).lower()
    if status in {"skipped_stale_signal", "skipped_tp_min_qty", "skipped_min_qty", "skipped_invalid_setup", "opened_then_emergency_closed", "opened_emergency_close_unverified", "opened_registration_failed_close_unverified", "entry_state_unverified", "entry_not_filled"}:
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


def _reconciliation_stop_loss_pct(trade: dict[str, Any]) -> float | None:
    """Return a trustworthy saved stop-risk percentage for protection repair.

    Prefer the canonical active-trade field, then the immutable setup snapshot.
    Never guess a materially different fallback such as 1% when state is missing
    or corrupt: callers must fail closed rather than rebuild protection at a
    potentially incorrect distance.
    """
    setup = trade.get("setup") if isinstance(trade.get("setup"), dict) else {}
    for candidate in (trade.get("planned_risk_pct"), setup.get("risk_pct")):
        if candidate in (None, "") or isinstance(candidate, bool):
            continue
        try:
            value = float(candidate)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    return None


def _elapsed_seconds(start_ts: Any, end_ts: Any) -> float | None:
    """Return elapsed seconds only when both timestamps are valid and ordered.

    Unknown timestamps remain unknown (None); negative durations are data-order
    anomalies and are never silently coerced to zero.
    """
    try:
        if start_ts in (None, "") or end_ts in (None, ""):
            return None
        a = _coerce_utc_timestamp(start_ts)
        b = _coerce_utc_timestamp(end_ts)
        if a is None or b is None:
            return None
        seconds = (b - a).total_seconds()
        if not math.isfinite(float(seconds)) or seconds < 0:
            return None
        return float(seconds)
    except Exception:
        return None


def _elapsed_minutes(start_ts: Any, end_ts: Any) -> float | None:
    sec = _elapsed_seconds(start_ts, end_ts)
    return sec / 60.0 if sec is not None else None


def _five_m_trigger_age_minutes(signal: dict[str, Any], now_ts: Any = None) -> tuple[float | None, str | None]:
    """Return age from the closed 5m trigger boundary, with legacy fallback."""
    if str(signal.get("trigger_timeframe", "1h")).lower() != "5m":
        return None, None
    try:
        trigger_ts = _coerce_utc_timestamp(
            signal.get("trigger_bar_close_time") or signal.get("decision_boundary_ts")
            or signal.get("trigger_bar_time") or signal.get("time")
        )
        if trigger_ts is None:
            return None, "invalid_5m_trigger_time"
        now = _coerce_utc_timestamp(now_ts) if now_ts is not None else pd.Timestamp.now(tz="UTC")
        if now is None:
            return None, "invalid_execution_clock"
        return max(0.0, (now - trigger_ts).total_seconds() / 60.0), None
    except Exception:
        return None, "invalid_5m_trigger_time"



def _settle_entry_order_before_protection(
    symbol: str, order: dict[str, Any], *, timeout_sec: float | None = None,
) -> dict[str, Any]:
    """Ensure an accepted market entry cannot continue growing unnoticed.

    A visible position is not proof that its originating MARKET order is terminal:
    it may be a partial fill while the remainder remains executable. Query the
    exchange order, cancel a still-active remainder, and require a terminal state
    before sizing protection. When the exchange cannot be queried/cancelled, return
    an explicit unverified result so callers can preserve/reconcile the live exposure
    rather than silently claiming the entry is final.
    """
    if not isinstance(order, dict):
        return {"status": "unverified", "error": "entry order result is not an object"}
    initial_status = str(order.get("order_status") or "").strip().upper()
    order_id = str(order.get("order_id") or order.get("orderId") or "").strip()
    client_order_id = str(order.get("client_order_id") or order.get("clientOrderId") or "").strip()
    if initial_status == "FILLED":
        # A terminal label must not override a contradictory quantity payload.
        # When the response includes the requested/original quantity, require its
        # executed quantity to reconcile before trusting this short path.
        try:
            expected_qty = float(order.get("orig_qty") or order.get("origQty") or order.get("qty") or 0.0)
            executed_qty = float(order.get("executed_qty") or order.get("executedQty") or 0.0)
        except (TypeError, ValueError):
            expected_qty, executed_qty = 0.0, 0.0
        qty_tolerance = max(1e-10, abs(expected_qty) * 1e-6)
        if expected_qty > 0 and executed_qty > 0 and abs(executed_qty - expected_qty) <= qty_tolerance:
            return {
                "status": "settled", "order_status": "FILLED",
                "executed_qty": executed_qty,
                "position_refresh_required": False,
                "source": "market_order_response",
            }
        if not client_order_id:
            return {
                "status": "unverified", "order_status": "FILLED",
                "executed_qty": executed_qty,
                "position_refresh_required": True,
                "error": "FILLED entry response lacks matching executed/original quantity evidence and durable clientOrderId",
            }
    if not client_order_id:
        # The live execution path durably writes a clientOrderId before POST. If
        # it is absent here, we cannot prove this order belongs to the current
        # event; do not query/cancel an arbitrary historical order by numeric ID.
        return {"status": "unverified", "error": "entry order has no durable clientOrderId identity"}

    try:
        timeout = max(0.5, float(timeout_sec if timeout_sec is not None else os.environ.get("ENTRY_ORDER_SETTLE_TIMEOUT_SEC", "3.0")))
    except (TypeError, ValueError):
        timeout = 3.0
    deadline = time.monotonic() + timeout
    cancel_attempted = False
    last_error: str | None = None
    last_state: dict[str, Any] | None = None
    active_statuses = {"NEW", "PENDING", "PARTIALLY_FILLED"}
    terminal_statuses = {"FILLED", "CANCELED", "CANCELLED", "EXPIRED", "REJECTED"}
    polls = 0

    while polls < 12 and time.monotonic() <= deadline:
        polls += 1
        try:
            state = get_order(
                symbol,
                order_id=order_id if order_id else None,
                client_order_id=None if order_id else client_order_id,
            )
        except Exception as exc:
            state = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        if isinstance(state, dict) and state.get("status") == "ok":
            last_state = state
            order_id = str(state.get("order_id") or order_id)
            current_status = str(state.get("order_status") or "").strip().upper()
            try:
                executed_qty = abs(float(state.get("executed_qty") or 0.0))
            except (TypeError, ValueError):
                executed_qty = 0.0
            if current_status in terminal_statuses:
                if current_status == "FILLED":
                    raw_expected = state.get("orig_qty")
                    try:
                        exchange_expected_qty = float(raw_expected) if raw_expected not in (None, "") else 0.0
                    except (TypeError, ValueError):
                        exchange_expected_qty = 0.0
                    quantity_known = state.get("executed_qty_known", True) is not False
                    quantity_tolerance = max(1e-10, abs(exchange_expected_qty) * 1e-6)
                    if exchange_expected_qty <= 0 or not quantity_known or executed_qty <= 0 or abs(executed_qty - exchange_expected_qty) > quantity_tolerance:
                        last_error = (
                            "exchange reports FILLED without consistent executedQty/origQty evidence: "
                            f"executedQty={executed_qty} origQty={exchange_expected_qty} "
                            f"executedQtyKnown={quantity_known}"
                        )
                        # Retry one fresh status read in case the exchange snapshot
                        # was internally inconsistent; never size protection as final
                        # from the contradictory response itself.
                        if polls < 2 and time.monotonic() < deadline:
                            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
                            continue
                        return {
                            "status": "unverified", "order_status": current_status,
                            "executed_qty": executed_qty, "position_refresh_required": True,
                            "cancel_attempted": cancel_attempted, "order_state": state,
                            "error": last_error,
                        }
                return {
                    "status": "settled",
                    "order_status": current_status,
                    "executed_qty": executed_qty,
                    "position_refresh_required": bool(
                        initial_status in active_statuses or cancel_attempted or current_status != "FILLED"
                    ),
                    "source": "exchange_order_query",
                    "cancel_attempted": cancel_attempted,
                    "order_state": state,
                }
            if current_status in active_statuses:
                if not cancel_attempted:
                    try:
                        if order_id:
                            cancel_order(symbol, order_id)
                        elif client_order_id:
                            cancel_order(symbol, client_order_id=client_order_id)
                        else:
                            raise RuntimeError("active entry order has neither orderId nor clientOrderId")
                        last_error = "entry remainder cancellation requested; awaiting terminal order state"
                    except Exception as exc:
                        last_error = f"entry cancellation request failed: {type(exc).__name__}: {exc}"
                    cancel_attempted = bool(order_id or client_order_id)
                else:
                    last_error = f"entry order remains active after cancellation: {current_status}"
            else:
                last_error = f"unrecognized entry order status={current_status or 'EMPTY'}"
        else:
            last_error = str((state or {}).get("error") or "entry order query returned no authoritative state")
            if (order_id or client_order_id) and not cancel_attempted:
                # A failed GET does not prove the order is absent. Cancel by the
                # server ID when known; otherwise use the durable clientOrderId,
                # which BingX also accepts for DELETE /trade/order.
                try:
                    if order_id:
                        cancel_order(symbol, order_id)
                    else:
                        cancel_order(symbol, client_order_id=client_order_id)
                except Exception as exc:
                    last_error += f"; cancellation attempt: {type(exc).__name__}: {exc}"
                cancel_attempted = True
        if polls < 12 and time.monotonic() < deadline:
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

    return {
        "status": "unverified",
        "order_status": str((last_state or {}).get("order_status") or initial_status or "UNKNOWN").upper(),
        "executed_qty": (last_state or {}).get("executed_qty", order.get("executed_qty")),
        "position_refresh_required": bool(cancel_attempted or initial_status in active_statuses),
        "cancel_attempted": cancel_attempted,
        "order_state": last_state,
        "error": last_error or "entry order failed to reach an authoritative terminal status",
    }


def _trade_has_unverified_entry_remainder(trade: dict[str, Any]) -> bool:
    """Read the durable marker that an entry MARKET remainder was not settled."""
    if not isinstance(trade, dict):
        return False
    setup = trade.get("setup") if isinstance(trade.get("setup"), dict) else {}
    entry_order = trade.get("entry_order") if isinstance(trade.get("entry_order"), dict) else {}
    reconciliation_candidates = (
        trade.get("entry_order_reconciliation"),
        setup.get("entry_order_reconciliation"),
        entry_order.get("entry_reconciliation"),
    )
    reconciliation = next((candidate for candidate in reconciliation_candidates
                           if isinstance(candidate, dict) and candidate.get("status")), {})
    status = str(reconciliation.get("status", "")).lower()
    flagged = bool(
        trade.get("entry_remainder_unverified")
        or setup.get("entry_remainder_unverified")
        or status == "unverified"
    )
    return flagged and status != "settled"


def execute_new_position(signal: dict[str, Any]) -> dict[str, Any]:
    symbol = str(signal["symbol"])
    direction = str(signal["type"]).upper()
    event_id = str(signal["event_id"])
    entry_price = float(signal["entry"])

    state_ready, state_reason = _live_state_preflight(signal)
    if not state_ready:
        return {
            "status": "blocked_state_untrusted",
            "error": state_reason,
            "order": None,
            "position": None,
        }
    execution_started_ts = pd.Timestamp.now(tz="UTC")
    signal["execution_started_ts"] = execution_started_ts.isoformat()
    _append_execution_ledger(event_id, signal.get("attempt_id"), "EXECUTION_START", {"symbol": symbol, "direction": direction, "signal_price": entry_price, "trigger_bar_time": signal.get("trigger_bar_time") or signal.get("time")})

    # Validate the planned setup before making any network call or opening a position.
    valid, reason = _validate_trade_geometry(signal)
    if not valid:
        log.warning("[EXEC_SKIPPED] %s %s | invalid_setup | %s", symbol, direction, reason)
        _append_execution_ledger(event_id, signal.get("attempt_id"), "SETUP_REJECTED", {"status": "skipped_invalid_setup", "error": reason})
        return {"status": "skipped_invalid_setup", "error": reason, "symbol": symbol, "direction": direction}

    # Local ownership invariant: the engine must never create a second active
    # local owner for the same exchange position identity, even when exchange
    # position polling temporarily reports flat/stale state. A state race between
    # the admission preflight above and this ownership read is also fail-closed.
    try:
        local_owner_conflict = has_active_trade_conflict(symbol, direction, exclude_event_id=event_id)
    except ActiveTradeStateCorrupt as exc:
        reason = f"active-trade state became untrusted before ownership check: {exc}"
        log.critical("[LIVE_BLOCKED_STATE] %s %s event=%s | %s", symbol, direction, event_id, reason)
        _append_execution_ledger(event_id, signal.get("attempt_id"), "EXECUTION_BLOCKED_STATE", {
            "status": "BLOCKED_STATE_UNTRUSTED",
            "error": reason,
        })
        return {"status": "blocked_state_untrusted", "error": reason, "order": None, "position": None}
    if local_owner_conflict:
        conflict = "local_active_trade_conflict"
        log.error("[EXEC_BLOCKED_LOCAL_OWNER] %s %s | event=%s | %s", symbol, direction, event_id, conflict)
        _append_execution_ledger(event_id, signal.get("attempt_id"), "LOCAL_OWNER_CONFLICT", {"status": "blocked_local_owner_conflict", "error": conflict})
        telemetry.record_state_conflict(
            event_id=event_id, attempt_id=signal.get("attempt_id"), position_id=None, order_id=None,
            symbol=symbol, direction=direction, conflict_type="LOCAL_ACTIVE_TRADE_DUPLICATE",
            message=conflict,
        )
        return {"status": "blocked_local_owner_conflict", "symbol": symbol, "direction": direction, "error": conflict}

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
        _append_execution_ledger(event_id, signal.get("attempt_id"), "PROTECTION_PREFLIGHT_BLOCKED", {"status": "blocked_protection_preflight", "error": reason, "protection_preflight": protection_capacity})
        error_code, error_message = telemetry.parse_exchange_error(protection_capacity.get("error") or reason)
        if protection_capacity.get("status") == "error" or error_code not in (None, "", 0, "0"):
            telemetry.record_exchange_error(
                event_id=event_id,
                attempt_id=signal.get("attempt_id"),
                position_id=None,
                order_id=None,
                symbol=symbol,
                endpoint=OPEN_ORDERS_PATH if "openOrders" in error_message or "100410" in str(error_code) else POSITION_PATH,
                method="GET",
                error_code=error_code,
                message=error_message,
                error_class="PROTECTION_PREFLIGHT",
                blocking=True,
                business_impact="MISSED_ENTRY",
            )
        return {
            "status": "blocked_protection_preflight",
            "symbol": symbol,
            "direction": direction,
            "error": reason,
            "protection_preflight": protection_capacity,
        }
    _append_execution_ledger(event_id, signal.get("attempt_id"), "PROTECTION_PREFLIGHT_READY", {"cancelled_order_ids": protection_capacity.get("cancelled_order_ids", []), "required_new_protection_orders": protection_capacity.get("required_new_protection_orders")})

    # Revalidate against the actual BingX top-of-book immediately before MARKET.
    # The previous implementation validated the stale signal price and only
    # discovered changed risk/structure after the fill.
    execution_quote = get_execution_quote(symbol, reference_price=entry_price)
    if execution_quote.get("status") != "ok":
        reason = execution_quote.get("error", "execution quote unavailable")
        log.error("[EXEC_BLOCKED_QUOTE] %s %s | %s", symbol, direction, reason)
        _append_execution_ledger(event_id, signal.get("attempt_id"), "QUOTE_UNAVAILABLE", {"status": "execution_quote_unavailable", "error": reason})
        return {"status": "execution_quote_unavailable", "error": reason, "symbol": symbol, "direction": direction}

    _append_execution_ledger(event_id, signal.get("attempt_id"), "PRE_ENTRY_QUOTE", {
        "bid": execution_quote.get("bid"),
        "ask": execution_quote.get("ask"),
        "spread_pct": execution_quote.get("spread_pct"),
        "quote_source": execution_quote.get("quote_source"),
        "execution_reference_quality": execution_quote.get("execution_reference_quality"),
        "quote_sources_attempted": execution_quote.get("quote_sources_attempted"),
        "quote_fallback_reason": execution_quote.get("quote_fallback_reason"),
        "quote_time": execution_quote.get("time"),
        "quote_exchange_time_ms": execution_quote.get("quote_exchange_time_ms"),
        "quote_observed_at_ms": execution_quote.get("quote_observed_at_ms"),
    })
    telemetry.record_quote_snapshot(
        event_id=event_id,
        attempt_id=signal.get("attempt_id"),
        symbol=symbol,
        direction=direction,
        quote=execution_quote,
        signal_price=entry_price,
        stage="PRE_ENTRY_QUOTE",
    )
    observed_at_ms = execution_quote.get("quote_observed_at_ms")
    if observed_at_ms is not None and EXECUTION_QUOTE_MAX_AGE_SEC > 0:
        try:
            quote_age_sec = max(0.0, (time.time() * 1000.0 - float(observed_at_ms)) / 1000.0)
        except (TypeError, ValueError):
            quote_age_sec = None
        if quote_age_sec is not None and quote_age_sec > EXECUTION_QUOTE_MAX_AGE_SEC:
            reason = f"execution_quote_age={quote_age_sec:.3f}s > {EXECUTION_QUOTE_MAX_AGE_SEC:.3f}s"
            log.warning("[EXEC_REJECT_QUOTE_AGE] %s %s | %s", symbol, direction, reason)
            _append_execution_ledger(event_id, signal.get("attempt_id"), "QUOTE_REJECTED_STALE", {"error": reason, "quote_age_sec": quote_age_sec})
            return {"status": "skipped_stale_signal", "error": reason, "symbol": symbol, "direction": direction, "execution_quote": execution_quote}
    executable_price = float(execution_quote["ask"] if direction == "LONG" else execution_quote["bid"])
    signal_drift_pct = _adverse_signal_drift_pct(entry_price, executable_price, direction)
    signal_price_displacement_pct, signal_price_displacement_direction = _signal_price_displacement_pct(
        entry_price, executable_price
    )
    if signal_price_displacement_pct > MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT:
        reason = (
            f"signal_price_displacement_pct={signal_price_displacement_pct:.4f}% "
            f"> {MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT:.4f}% "
            f"direction={signal_price_displacement_direction}"
        )
        log.warning("[EXEC_REJECT_SIGNAL_DISPLACEMENT] %s %s | %s | signal=%s executable=%s", symbol, direction, reason, entry_price, executable_price)
        _append_execution_ledger(event_id, signal.get("attempt_id"), "SIGNAL_PRICE_DISPLACEMENT_REJECTED", {
            "status": "skipped_stale_signal",
            "error": reason,
            "signal_price": entry_price,
            "execution_reference_price": executable_price,
            "signal_price_displacement_pct": signal_price_displacement_pct,
            "signal_price_displacement_direction": signal_price_displacement_direction,
            "signal_price_displacement_limit_pct": MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT,
            "signal_drift_pct": signal_drift_pct,
        })
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
            "signal_price_displacement_pct": signal_price_displacement_pct,
            "signal_price_displacement_direction": signal_price_displacement_direction,
            "signal_price_displacement_limit_pct": MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT,
        }
    if signal_drift_pct > MAX_ENTRY_SLIPPAGE_PCT:
        reason = f"signal_drift_pct={signal_drift_pct:.4f}% > {MAX_ENTRY_SLIPPAGE_PCT:.4f}%"
        log.warning("[EXEC_REJECT_STALE] %s %s | %s | signal=%s executable=%s", symbol, direction, reason, entry_price, executable_price)
        _append_execution_ledger(event_id, signal.get("attempt_id"), "SIGNAL_DRIFT_REJECTED", {"status": "skipped_stale_signal", "error": reason, "signal_price": entry_price, "execution_reference_price": executable_price, "signal_drift_pct": signal_drift_pct})
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
        _append_execution_ledger(event_id, signal.get("attempt_id"), "PRE_ENTRY_GEOMETRY_REJECTED", {"status": "skipped_invalid_setup", "error": reason})
        return {"status": "skipped_invalid_setup", "error": reason, "symbol": symbol, "direction": direction}

    valid, reason = _validate_trade_geometry(preflight_signal)
    if not valid:
        log.warning("[EXEC_REJECT_GEOMETRY] %s %s | %s", symbol, direction, reason)
        _append_execution_ledger(event_id, signal.get("attempt_id"), "PRE_ENTRY_GEOMETRY_REJECTED", {"status": "skipped_invalid_setup", "error": reason, "execution_reference_price": executable_price, "signal_drift_pct": signal_drift_pct})
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
        _append_execution_ledger(event_id, signal.get("attempt_id"), "PRE_ENTRY_PRECISION_REJECTED", {"status": "skipped_invalid_setup", "error": reason})
        return {"status": "skipped_invalid_setup", "error": reason, "symbol": symbol, "direction": direction}

    # Re-check state immediately before the exchange write. Earlier validation can
    # take seconds; a state file disappearing/corrupting during those network calls
    # must block the order rather than relying on the earlier snapshot.
    state_ready, state_reason = _live_state_preflight(signal)
    if not state_ready:
        return {
            "status": "blocked_state_untrusted",
            "error": state_reason,
            "order": None,
            "position": None,
        }

    # The signal may have aged while preflight, quote validation and context
    # collection were running. Recheck the 5m timestamp before announcing order
    # submission; the same guard runs a second time immediately before MARKET POST.
    trigger_age_min, trigger_age_error = _five_m_trigger_age_minutes(signal)
    if trigger_age_error or (trigger_age_min is not None and trigger_age_min > MAX_5M_TRIGGER_AGE_MINUTES):
        reject_reason = trigger_age_error or "5m_trigger_too_old"
        details = {
            "status": "skipped_stale_signal",
            "reason": reject_reason,
            "trigger_age_minutes": trigger_age_min,
            "max_trigger_age_minutes": MAX_5M_TRIGGER_AGE_MINUTES,
            "trigger_bar_time": signal.get("trigger_bar_time") or signal.get("time"),
            "order_submitted": False,
        }
        _append_execution_ledger(event_id, signal.get("attempt_id"), "EXECUTION_REJECTED_TRIGGER_AGE", details)
        log.warning("[EXEC_REJECT_5M_AGE_PRE_POST] %s %s | reason=%s age_min=%s allowed=%.2f", symbol, direction, reject_reason, f"{trigger_age_min:.3f}" if trigger_age_min is not None else "unknown", MAX_5M_TRIGGER_AGE_MINUTES)
        return {
            "status": "skipped_stale_signal", "error": reject_reason,
            "stale_trigger_reason": reject_reason, "stale_trigger_age_minutes": trigger_age_min,
            "symbol": symbol, "direction": direction, "order_submitted": False,
        }
    signal["execution_age_minutes_at_order_ready"] = trigger_age_min

    setup = _build_setup(signal)
    # ``order_submit_ts`` is reserved for an actual MARKET POST. Do not populate
    # it with the function-call start time because rejected/stale attempts did not
    # submit an order. ``execution_call_start_ts`` is recorded separately below.
    signal.pop("order_submit_ts", None)
    telemetry.record_order_event(
        event_id=event_id,
        attempt_id=signal.get("attempt_id"),
        order_id=None,
        symbol=symbol,
        direction=direction,
        leg="ENTRY",
        status="SUBMITTING",
        order_type="MARKET",
        reduce_only=False,
        requested_entry_price=entry_price,
        execution_reference_price=executable_price,
        quote_source=execution_quote.get("quote_source"),
        quote_exchange_time_ms=execution_quote.get("quote_exchange_time_ms"),
        quote_observed_at_ms=execution_quote.get("quote_observed_at_ms"),
        quote_local_age_sec=execution_quote.get("quote_local_age_sec"),
        quote_exchange_age_sec=execution_quote.get("quote_exchange_age_sec"),
        quote_freshness_source=execution_quote.get("quote_freshness_source"),
        quote_freshness_status=execution_quote.get("quote_freshness_status"),
        quote_unknown_age_policy=execution_quote.get("quote_unknown_age_policy"),
        execution_reference_quality=execution_quote.get("execution_reference_quality"),
    )
    # Final freshness check: no network call, sleep or other work may occur
    # between this check and open_market's MARKET POST. If the order-aged signal
    # crossed its configured boundary during local telemetry writes, reject it.
    trigger_age_min, trigger_age_error = _five_m_trigger_age_minutes(signal)
    if trigger_age_error or (trigger_age_min is not None and trigger_age_min > MAX_5M_TRIGGER_AGE_MINUTES):
        reject_reason = trigger_age_error or "5m_trigger_too_old"
        details = {
            "status": "skipped_stale_signal", "reason": reject_reason,
            "trigger_age_minutes": trigger_age_min,
            "max_trigger_age_minutes": MAX_5M_TRIGGER_AGE_MINUTES,
            "trigger_bar_time": signal.get("trigger_bar_time") or signal.get("time"),
            "order_submitted": False,
        }
        telemetry.record_order_event(
            event_id=event_id, attempt_id=signal.get("attempt_id"), order_id=None,
            symbol=symbol, direction=direction, leg="ENTRY", status="REJECTED",
            order_type="MARKET", reduce_only=False, error=reject_reason,
        )
        _append_execution_ledger(event_id, signal.get("attempt_id"), "ORDER_NOT_SUBMITTED_TRIGGER_STALE", details)
        log.warning("[EXEC_REJECT_5M_AGE_FINAL] %s %s | reason=%s age_min=%s allowed=%.2f", symbol, direction, reject_reason, f"{trigger_age_min:.3f}" if trigger_age_min is not None else "unknown", MAX_5M_TRIGGER_AGE_MINUTES)
        return {
            "status": "skipped_stale_signal", "error": reject_reason,
            "stale_trigger_reason": reject_reason, "stale_trigger_age_minutes": trigger_age_min,
            "symbol": symbol, "direction": direction, "order_submitted": False,
        }
    signal["execution_age_minutes_at_order_submit"] = trigger_age_min
    trigger_bar_time_ms = None
    if str(signal.get("trigger_timeframe", "1h")).lower() == "5m":
        trigger_ts = _coerce_utc_timestamp(signal.get("trigger_bar_close_time") or signal.get("decision_boundary_ts") or signal.get("trigger_bar_time") or signal.get("time"))
        # The downstream parameter name is historical; pass the close boundary, not candle open.
        trigger_bar_time_ms = int(trigger_ts.timestamp() * 1000) if trigger_ts is not None else None
    order = open_market(
        symbol, direction, entry_price, event_id,
        execution_quote=execution_quote,
        attempt_id=signal.get("attempt_id"),
        trigger_bar_time_ms=trigger_bar_time_ms,
        max_trigger_age_minutes=MAX_5M_TRIGGER_AGE_MINUTES if str(signal.get("trigger_timeframe", "1h")).lower() == "5m" else None,
        order_identity_callback=lambda client_order_id: _append_execution_ledger(
            event_id, signal.get("attempt_id"), "ENTRY_ORDER_IDENTITY",
            {"symbol": symbol, "direction": direction, "client_order_id": client_order_id},
            raise_on_error=True,
        ),
    )
    order_status = str(order.get("status", "")).lower() if isinstance(order, dict) else ""
    response = order.get("response") if isinstance(order, dict) and isinstance(order.get("response"), dict) else {}
    telemetry.record_order_event(
        event_id=event_id,
        attempt_id=signal.get("attempt_id"),
        order_id=order.get("order_id") if isinstance(order, dict) else None,
        symbol=symbol,
        direction=direction,
        leg="ENTRY",
        status=("ACKNOWLEDGED" if order_status in {"opened", "entry_pending"} else "REJECTED"),
        order_type="MARKET",
        result_status=order.get("status") if isinstance(order, dict) else None,
        exchange_code=response.get("code"),
        exchange_message=response.get("msg"),
        client_order_id=order.get("client_order_id") if isinstance(order, dict) else None,
        order_submit_at_ms=order.get("order_submit_at_ms") if isinstance(order, dict) else None,
        idempotency=order.get("idempotency") if isinstance(order, dict) else None,
        error=order.get("error") if isinstance(order, dict) else None,
    )
    exact_order_submit_ms = order.get("order_submit_at_ms") if isinstance(order, dict) else None
    try:
        if exact_order_submit_ms is not None:
            signal["order_submit_ts"] = pd.to_datetime(int(exact_order_submit_ms), unit="ms", utc=True).isoformat()
    except (TypeError, ValueError, OverflowError):
        log.warning("[EXEC_LEDGER] invalid order_submit_at_ms for %s: %r", event_id, exact_order_submit_ms)
    if order.get("status") == "skipped_stale_signal" and isinstance(order, dict):
        order_error = str(order.get("error") or "")
        if "5m_trigger_too_old_at_post" in order_error or "invalid_5m_trigger_time_at_post" in order_error:
            stale_reason = "invalid_5m_trigger_time" if "invalid_5m_trigger_time_at_post" in order_error else "5m_trigger_too_old"
            stale_age = order.get("trigger_age_minutes")
            details = {
                "status": "skipped_stale_signal",
                "reason": stale_reason,
                "trigger_age_minutes": stale_age,
                "max_trigger_age_minutes": order.get("max_trigger_age_minutes", MAX_5M_TRIGGER_AGE_MINUTES),
                "trigger_bar_time": signal.get("trigger_bar_time") or signal.get("time"),
                "order_submitted": False,
                "rejected_inside_market_submitter": True,
            }
            _append_execution_ledger(event_id, signal.get("attempt_id"), "ORDER_NOT_SUBMITTED_TRIGGER_STALE", details)
            return {
                "status": "skipped_stale_signal",
                "error": order_error,
                "stale_trigger_reason": stale_reason,
                "stale_trigger_age_minutes": stale_age,
                "order_submitted": False,
                "order": order,
            }

    if order.get("status") == "skipped_min_qty":
        _append_execution_ledger(event_id, signal.get("attempt_id"), "ORDER_REJECTED", {"status": order.get("status"), "error": order.get("error")})
        telemetry.record_exchange_error(
            event_id=event_id,
            attempt_id=signal.get("attempt_id"),
            position_id=None,
            order_id=order.get("order_id"),
            symbol=symbol,
            endpoint=ORDER_PATH,
            method="POST",
            error_code=response.get("code"),
            message=str(order.get("error") or response.get("msg") or "minimum quantity pre-check rejected entry"),
            error_class="ORDER_REJECTED",
            blocking=True,
            business_impact="MISSED_ENTRY",
        )
        return order
    if order.get("status") not in {"opened", "entry_pending"}:
        _append_execution_ledger(event_id, signal.get("attempt_id"), "ORDER_REJECTED", {"status": order.get("status"), "error": order.get("error")})
        error_code, error_message = telemetry.parse_exchange_error(order.get("error") or response, default_code=response.get("code"))
        telemetry.record_exchange_error(
            event_id=event_id,
            attempt_id=signal.get("attempt_id"),
            position_id=None,
            order_id=order.get("order_id"),
            symbol=symbol,
            endpoint=ORDER_PATH,
            method="POST",
            error_code=error_code,
            message=error_message,
            error_class="ORDER_REJECTED",
            blocking=True,
            business_impact="MISSED_ENTRY",
        )
        return {"status": str(order.get("status", "error")).upper(), "error": order.get("error"), "order": order}

    # open_market owns the final venue-quote gate immediately before POST. Use that
    # authoritative quote for all post-fill execution-quality calculations rather
    # than the earlier preflight quote captured before the idempotency/protection path.
    authoritative_execution_quote = order.get("execution_quote")
    if isinstance(authoritative_execution_quote, dict) and authoritative_execution_quote.get("status") == "ok":
        execution_quote = authoritative_execution_quote

    _append_execution_ledger(event_id, signal.get("attempt_id"), "ORDER_ACK", {"order_id": order.get("order_id"), "client_order_id": order.get("client_order_id"), "qty": order.get("qty"), "sizing_price": order.get("sizing_price"), "order_submit_at_ms": order.get("order_submit_at_ms"), "execution_quote": execution_quote})

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
            # A flat position snapshot does not prove the acknowledged MARKET
            # order never filled. Inspect the order itself before allowing retry.
            order_state = None
            try:
                order_id = (order or {}).get("order_id") if isinstance(order, dict) else None
                if order_id:
                    order_state = get_order(symbol, str(order_id))
            except Exception as exc:
                order_state = {"status": "error", "error": str(exc)}
            order_status = str((order_state or {}).get("order_status", "")).upper()
            try:
                executed_qty = abs(float((order_state or {}).get("executed_qty") or 0.0))
            except (TypeError, ValueError):
                executed_qty = 0.0
            if (order_state or {}).get("status") == "ok" and order_id and order_status in {"NEW", "PENDING", "PARTIALLY_FILLED"}:
                # Prevent an acknowledged but not-yet-terminal order from filling
                # after the position snapshot was observed flat.
                try:
                    cancel_result = cancel_order(symbol, str(order_id))
                    if isinstance(cancel_result, dict) and cancel_result.get("code") in (0, "0"):
                        order_state_after_cancel = get_order(symbol, str(order_id))
                        if isinstance(order_state_after_cancel, dict):
                            order_state = order_state_after_cancel
                            order_status = str(order_state.get("order_status", "")).upper()
                            try:
                                executed_qty = abs(float(order_state.get("executed_qty") or 0.0))
                            except (TypeError, ValueError):
                                executed_qty = 0.0
                except Exception as cancel_exc:
                    log.critical("[EXEC_ENTRY_ORDER_CANCEL_UNVERIFIED] %s order=%s error=%s", symbol, order_id, cancel_exc)
            if (order_state or {}).get("status") == "ok" and order_status in {"CANCELED", "CANCELLED", "REJECTED", "EXPIRED"} and executed_qty <= 0:
                log.warning("[EXEC_ENTRY_NOT_FILLED] %s %s | order=%s terminal_status=%s executed_qty=0", symbol, direction, (order or {}).get("order_id"), order_status)
                return {
                    "status": "entry_not_filled", "order": order, "order_state": order_state,
                    "position": reconciled_position, "fill_poll": position,
                    "error": f"market entry order terminal without fills: {order_status}",
                }
            log.critical(
                "[EXEC_ENTRY_STATE_UNVERIFIED] %s %s | position is currently flat but order fill state is not proven absent; blocking retries. order_state=%s",
                symbol, direction, order_state,
            )
            return {
                "status": "entry_state_unverified", "order": order, "order_state": order_state,
                "position": reconciled_position, "fill_poll": position,
                "error": "market order acknowledged; position is flat now but no authoritative no-fill order status was confirmed",
            }
        else:
            error_detail = reconciled_position.get("error") or position.get("error") or position.get("last_poll_error")
            telemetry.record_exchange_error(
                event_id=event_id,
                attempt_id=signal.get("attempt_id"),
                position_id=event_id,
                order_id=order.get("order_id"),
                symbol=symbol,
                endpoint=POSITION_PATH,
                method="GET",
                error_code=reconciled_position.get("code"),
                message=str(error_detail or "entry position reconciliation failed"),
                error_class="POSITION_RECONCILIATION",
                blocking=True,
                business_impact="ENTRY_STATE_UNVERIFIED",
            )
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

    entry_settlement = _settle_entry_order_before_protection(symbol, order)
    order["entry_reconciliation"] = dict(entry_settlement)
    if entry_settlement.get("status") == "settled":
        _append_execution_ledger(
            event_id, signal.get("attempt_id"), "ENTRY_ORDER_SETTLED",
            {"order_id": order.get("order_id"), "client_order_id": order.get("client_order_id"), **entry_settlement},
        )
        if entry_settlement.get("position_refresh_required"):
            try:
                refreshed_position = get_position_directional(symbol, direction)
            except Exception as exc:
                refreshed_position = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            if refreshed_position.get("status") == "found":
                position = refreshed_position
            elif refreshed_position.get("status") == "not_found":
                log.critical(
                    "[EXEC_ENTRY_POSITION_DISAPPEARED] %s %s after entry-order settlement; refusing to register a phantom trade",
                    symbol, direction,
                )
                return {
                    "status": "entry_position_disappeared", "order": order,
                    "order_state": entry_settlement.get("order_state"),
                    "position": refreshed_position,
                    "error": "position disappeared while the market order remainder was being settled",
                }
            else:
                # Preserve the last known live position for best-effort protection,
                # but surface that its quantity could not be refreshed after cancel.
                order["entry_reconciliation"]["position_refresh_error"] = refreshed_position.get("error") or "position query did not return a known state"
                log.critical(
                    "[EXEC_ENTRY_POSITION_REFRESH_UNVERIFIED] %s %s; using last known position size for protective placement: %s",
                    symbol, direction, order["entry_reconciliation"]["position_refresh_error"],
                )
    else:
        _append_execution_ledger(
            event_id, signal.get("attempt_id"), "ENTRY_ORDER_REMAINDER_UNVERIFIED",
            {"order_id": order.get("order_id"), "client_order_id": order.get("client_order_id"), **entry_settlement},
        )
        log.critical(
            "[EXEC_ENTRY_REMAINDER_UNVERIFIED] %s %s; placing protection for currently observed position, while blocking any claim that entry quantity is final: %s",
            symbol, direction, entry_settlement.get("error"),
        )

    fill_observed_ts = pd.Timestamp.now(tz="UTC")
    fill_observed_ts_ms = int(fill_observed_ts.timestamp() * 1000)
    signal["fill_observed_ts"] = fill_observed_ts.isoformat()
    avg_price = float(position["avgPrice"])
    qty = abs(float(position["positionAmt"]))
    _append_execution_ledger(event_id, signal.get("attempt_id"), "POSITION_FILL", {"avg_price": avg_price, "qty": qty, "position_side": direction, "fill_observed_ts": signal.get("fill_observed_ts")})
    telemetry.record_order_event(
        event_id=event_id,
        attempt_id=signal.get("attempt_id"),
        order_id=order.get("order_id") if isinstance(order, dict) else None,
        symbol=symbol,
        direction=direction,
        leg="ENTRY",
        status="FILL_OBSERVED",
        exchange_fill_price=avg_price,
        executed_qty=qty,
        fill_observed_ts_ms=fill_observed_ts_ms,
        position_side=direction,
    )
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

    idempotent_replay = bool(isinstance(order, dict) and order.get("idempotency") == "existing_client_order_resolved_before_post")
    if idempotent_replay:
        # The position may have been filled before this scan and the preflight quote
        # is not contemporaneous with that fill. Do not convert current mark-to-market
        # loss into entry slippage and prematurely liquidate a valid existing position.
        execution_slippage_pct = None
        signed_entry_slippage_pct = None
    if execution_slippage_pct is not None and execution_slippage_pct > MAX_ENTRY_SLIPPAGE_PCT:
        reason = f"execution_slippage_pct={execution_slippage_pct:.4f}% > {MAX_ENTRY_SLIPPAGE_PCT:.4f}%"
        log.critical("[SAFETY_CLOSE] %s %s | %s", symbol, direction, reason)
        close_result, cleanup = _emergency_close_preserving_protection(symbol, direction, qty, event_id)
        return {
            "status": _emergency_execution_status(close_result),
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
        close_result, cleanup = _emergency_close_preserving_protection(symbol, direction, qty, event_id)
        return {"status": _emergency_execution_status(close_result), "error": reason, "order": order, "position": position, "close": close_result, "protection_cleanup": cleanup, "executed_signal": dict(signal)}

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
        close_result, cleanup = _emergency_close_preserving_protection(symbol, direction, qty, event_id)
        return {"status": _emergency_execution_status(close_result), "error": precision_reason, "order": order, "position": position, "close": close_result, "protection_cleanup": cleanup, "executed_signal": actual_signal}
    valid, reason = _validate_trade_geometry(actual_signal, require_structure_room=False)
    if not valid:
        log.critical("[SAFETY_CLOSE] %s %s | invalid post-fill protection geometry | %s", symbol, direction, reason)
        close_result, cleanup = _emergency_close_preserving_protection(symbol, direction, qty, event_id)
        return {"status": _emergency_execution_status(close_result), "error": reason, "order": order, "position": position, "close": close_result, "protection_cleanup": cleanup, "executed_signal": actual_signal}

    setup["entry_reference"] = avg_price
    setup["signal_price"] = entry_price
    setup["pre_entry_bid"] = pre_entry_bid
    setup["pre_entry_ask"] = pre_entry_ask
    setup["execution_reference_price"] = executable_reference_price
    setup["signal_drift_pct"] = signal_drift_pct
    setup["signal_price_displacement_pct"] = signal_price_displacement_pct
    setup["signal_price_displacement_direction"] = signal_price_displacement_direction
    setup["signal_price_displacement_limit_pct"] = MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT
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
    setup["entry_order_reconciliation"] = dict(entry_settlement)
    setup["entry_remainder_unverified"] = entry_settlement.get("status") != "settled"
    setup["fill_position"] = position if isinstance(position, dict) else {}
    setup["execution_snapshot"] = {
        "strategy_version": _effective_strategy_version(),
        "signal_entry": entry_price,
        "requested_entry": entry_price,
        "pre_entry_bid": pre_entry_bid,
        "pre_entry_ask": pre_entry_ask,
        "execution_reference_price": executable_reference_price,
        "signal_drift_pct": signal_drift_pct,
        "signal_price_displacement_pct": signal_price_displacement_pct,
        "signal_price_displacement_direction": signal_price_displacement_direction,
        "signal_price_displacement_limit_pct": MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT,
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
        "execution_reference_quality": execution_quote.get("execution_reference_quality"),
        "quote_sources_attempted": execution_quote.get("quote_sources_attempted"),
        "quote_fallback_reason": execution_quote.get("quote_fallback_reason"),
        "quote_time": execution_quote.get("time"),
        "execution_started_ts": signal.get("execution_started_ts"),
        "order_submit_ts": signal.get("order_submit_ts"),
        "fill_observed_ts": signal.get("fill_observed_ts"),
        "protection_started_ts": signal.get("protection_started_ts"),
        "protection_finished_ts": signal.get("protection_finished_ts"),
        "trigger_to_order_minutes": _elapsed_minutes(signal.get("trigger_bar_close_time") or signal.get("decision_boundary_ts") or signal.get("trigger_bar_time") or signal.get("time"), signal.get("order_submit_ts")),
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
    try:
        protection = ensure_directional_protection(
            symbol, direction, avg_price, qty,
            actual_risk_pct, setup["tp_levels"], trade_id=event_id,
        )
    except Exception as exc:
        # A protection helper exception after a fill is a protection failure, not
        # a reason to abandon the position without rollback/reconciliation.
        protection = {
            "status": "PROTECTION_FAILED",
            "error": f"ensure_directional_protection raised: {type(exc).__name__}: {exc}",
            "sl_result": {"status": "error", "error": str(exc)},
            "tp_orders": [],
        }
        log.exception("[PROTECTION_EXCEPTION] %s %s after fill; routing to emergency close", symbol, direction)
    protection_finished_ts = pd.Timestamp.now(tz="UTC")
    signal["protection_finished_ts"] = protection_finished_ts.isoformat()
    # The snapshot is persisted with the active trade after protection is complete.
    # Update it here so protection timings cannot be lost because the snapshot was
    # initially assembled before the protection orders were submitted.
    execution_snapshot = setup.setdefault("execution_snapshot", {})
    execution_snapshot.update({
        "protection_started_ts": signal.get("protection_started_ts"),
        "protection_finished_ts": signal.get("protection_finished_ts"),
        "fill_to_protection_seconds": _elapsed_seconds(signal.get("fill_observed_ts"), signal.get("protection_started_ts")),
        "protection_seconds": _elapsed_seconds(signal.get("protection_started_ts"), signal.get("protection_finished_ts")),
    })
    setup["execution_snapshot"]["protection_status"] = protection.get("status")
    _append_execution_ledger(event_id, signal.get("attempt_id"), "PROTECTION_RESULT", {"status": protection.get("status"), "error": protection.get("error"), "sl_order": protection.get("sl_result"), "tp_orders": protection.get("tp_orders", [])})
    sl_result = protection.get("sl_result") if isinstance(protection.get("sl_result"), dict) else {}
    if str(sl_result.get("status", "")).lower() in {"error", "rejected", "failed", "unverified", "sl_unverified"} or sl_result.get("error"):
        error_code, error_message = telemetry.parse_exchange_error(sl_result.get("error"), default_code=sl_result.get("code"))
        telemetry.record_exchange_error(
            event_id=event_id,
            attempt_id=signal.get("attempt_id"),
            position_id=event_id,
            order_id=sl_result.get("order_id"),
            symbol=symbol,
            endpoint=ORDER_PATH,
            method="POST",
            error_code=error_code,
            message=error_message,
            error_class="PROTECTION_ORDER",
            blocking=True,
            business_impact="PROTECTION_INCOMPLETE",
            leg="SL",
        )
    telemetry.record_protection_event(
        event_id=event_id,
        attempt_id=signal.get("attempt_id"),
        position_id=event_id,
        order_id=sl_result.get("order_id"),
        symbol=symbol,
        direction=direction,
        leg="SL",
        status=str(sl_result.get("status") or protection.get("status") or "UNKNOWN"),
        price=sl_result.get("price") or sl_result.get("stop_price"),
        qty=sl_result.get("qty"),
        error=sl_result.get("error"),
        verified=(str(protection.get("status")) in {"PROTECTED", "SL_ONLY"}),
        protection_order_verified=sl_result.get("protection_order_verified"),
        execution_verified=sl_result.get("execution_verified"),
    )
    for tp in protection.get("tp_orders", []) if isinstance(protection.get("tp_orders"), list) else []:
        if not isinstance(tp, dict):
            continue
        if str(tp.get("status", "")).lower() in {"error", "rejected", "failed", "unverified"} or tp.get("error"):
            error_code, error_message = telemetry.parse_exchange_error(tp.get("error"), default_code=tp.get("code"))
            telemetry.record_exchange_error(
                event_id=event_id,
                attempt_id=signal.get("attempt_id"),
                position_id=event_id,
                order_id=tp.get("order_id"),
                symbol=symbol,
                endpoint=ORDER_PATH,
                method="POST",
                error_code=error_code,
                message=error_message,
                error_class="PROTECTION_ORDER",
                blocking=True,
                business_impact="PROTECTION_INCOMPLETE",
                leg=str(tp.get("leg", "TP")).upper(),
            )
        telemetry.record_protection_event(
            event_id=event_id,
            attempt_id=signal.get("attempt_id"),
            position_id=event_id,
            order_id=tp.get("order_id"),
            symbol=symbol,
            direction=direction,
            leg=str(tp.get("leg", "TP")).upper(),
            status=str(tp.get("status", "UNKNOWN")),
            price=tp.get("price"),
            qty=tp.get("qty"),
            pnl_pct=tp.get("pnl_pct"),
            execution_verified=tp.get("execution_verified"),
            protection_order_verified=tp.get("protection_order_verified"),
            fill_evidence_status=tp.get("fill_evidence_status"),
            error=tp.get("error"),
        )
    if protection.get("status") != "PROTECTED":
        log.critical("[SAFETY_CLOSE] %s %s | mandatory protection incomplete | %s", symbol, direction, protection)
        # Mandatory rule: never leave a newly-opened position live without BOTH
        # a verified SL and both TP legs. Attempt an immediate market rollback.
        close_result, cleanup = _emergency_close_preserving_protection(symbol, direction, qty, event_id)
        try:
            time.sleep(0.25)
            verify_closed = get_position_directional(symbol, direction)
        except Exception as exc:
            verify_closed = {"status": "verification_error", "error": str(exc)}
        return {
            "status": _emergency_execution_status(close_result),
            "error": protection.get("error") or protection.get("status"),
            "order": order,
            "position": position,
            "protection": protection,
            "close": close_result,
            "protection_cleanup": cleanup,
            "close_verification": verify_closed,
            "executed_signal": actual_signal,
        }

    _append_execution_ledger(event_id, signal.get("attempt_id"), "PROTECTED_POSITION", {"fill_price": avg_price, "sl_price": sl_price, "tp1_price": tp1_price, "tp2_price": tp2_price, "risk_pct": actual_risk_pct, "planned_weighted_rr": setup["planned_weighted_rr"]})

    registration_reason = "active_trade_registration_failed"
    registration_setup = _setup_with_protection_result(setup, protection)
    try:
        registered = register_active_trade(
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
            setup=registration_setup,
            requested_entry_price=entry_price,
            entry_ts_ms=fill_observed_ts_ms,
        )
    except ActiveTradeStateCorrupt as exc:
        registered = False
        registration_reason = f"active_trade_state_untrusted_after_entry: {exc}"
    except Exception as exc:
        registered = False
        registration_reason = f"active_trade_registration_exception: {type(exc).__name__}: {exc}"

    if registered is False:
        log.critical(
            "[SAFETY_CLOSE] %s %s | active-trade registration failed after protected entry; rolling back: %s",
            symbol, direction, registration_reason,
        )
        return _rollback_after_registration_failure(
            event_id=event_id,
            symbol=symbol,
            direction=direction,
            qty=qty,
            reason=registration_reason,
            order=order,
            position=position,
            protection=protection,
            executed_signal=actual_signal,
        )

    final_execution_status = (
        "opened_protected" if entry_settlement.get("status") == "settled"
        else "entry_state_unverified_protected"
    )
    return {
        "status": final_execution_status,
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
            "trigger_to_order_minutes": _elapsed_minutes(signal.get("trigger_bar_close_time") or signal.get("decision_boundary_ts") or signal.get("trigger_bar_time") or signal.get("time"), signal.get("order_submit_ts")),
            "order_to_fill_seconds": _elapsed_seconds(signal.get("order_submit_ts"), signal.get("fill_observed_ts")),
            "fill_to_protection_seconds": _elapsed_seconds(signal.get("fill_observed_ts"), signal.get("protection_started_ts")),
            "protection_seconds": _elapsed_seconds(signal.get("protection_started_ts"), signal.get("protection_finished_ts")),
        },
        "executed_signal": actual_signal,
    }

def _write_runtime_version_metadata() -> None:
    """Persist current runtime/code provenance without rewriting historical records."""
    path = DATA / "runtime_version.json"
    payload = {
        "schema_version": 1,
        "record_type": "RUNTIME_VERSION",
        "engine_version": ENGINE_VERSION,
        "strategy_version": _effective_strategy_version(),
        "zone_trigger_mode": ZONE_TRIGGER_MODE,
        "execution_quote_max_future_skew_sec": os.environ.get("EXECUTION_QUOTE_MAX_FUTURE_SKEW_SEC", "1.0"),
        "code_commit_sha": CODE_COMMIT_SHA,
        "written_ts": int(time.time() * 1000),
    }
    try:
        DATA.mkdir(parents=True, exist_ok=True)
        _atomic_write_text_durable(path, json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        log.warning("[VERSION_METADATA] write failed: %s", type(exc).__name__)


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


def _fetch_missing_research_5m_bars(payload: dict[str, Any], analysis_meta: dict[str, dict], limit: int) -> list[dict]:
    """Fetch deferred 5m bars with instrumentation but without touching research journals.

    This helper is safe to run in a small thread pool: it performs read-only HTTP
    requests and returns plain rows. Persistence and per-symbol research processing
    remain serialized below so ordering and duplicate-handling are unchanged.
    """
    symbol = str(payload.get("symbol") or "")
    meta = analysis_meta.get(symbol, {})
    provider = str(payload.get("provider") or "binance").lower()
    with request_metrics.symbol_context(symbol):
        with request_metrics.phase("research_deferred_bars"):
            if provider == "bingx":
                return fetch_bingx_klines(symbol, "5m", limit=limit, retryable=False) or []
            binance_symbol = str(meta.get("binance_symbol") or symbol.replace("-", ""))
            return fetch_binance_klines(binance_symbol, "5m", limit=limit, retryable=False) or []


def _prefer_longer_research_5m_history(
    existing_bars: list[dict] | None,
    prefetched_bars: list[dict] | None,
) -> list[dict]:
    """Use the more complete 5m history; never discard a successful extension fetch."""
    existing = list(existing_bars or [])
    prefetched = list(prefetched_bars or [])
    return prefetched if len(prefetched) > len(existing) else existing


def _setup_with_protection_result(
    setup: dict[str, Any], protection: dict[str, Any]
) -> dict[str, Any]:
    """Prepare setup metadata for persistence on the active trade.

    ``effective_tp_levels`` is determined by protection sizing and may differ
    from the originally planned ``tp_levels``. The tracker consumes it from the
    setup's top level, so do not leave it buried only inside protection_result.
    """
    enriched = dict(setup) if isinstance(setup, dict) else {}
    enriched["protection_status"] = protection.get("status")
    enriched["protection_result"] = protection
    if isinstance(protection.get("effective_tp_levels"), list):
        enriched["effective_tp_levels"] = protection["effective_tp_levels"]
    return enriched


def _research_bar_payload(
    *,
    research_enabled: bool,
    has_research_boundary: bool,
    persist_bars_for_forward: bool,
    bars_1h: list[dict[str, Any]] | None,
    bars_5m: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select raw bars needed by either a current observation or a maturing one.

    A symbol can have no active zones and therefore no new boundary while still
    having an older observation awaiting its forward outcome. Keep both source
    timeframes in that case so the outcome worker does not lose its price path.
    """
    if not research_enabled or not (has_research_boundary or persist_bars_for_forward):
        return [], []
    return list(bars_1h or []), list(bars_5m or [])


def _research_payload_needs_5m_history(payload: dict[str, Any]) -> bool:
    """Only fetch extended 5m bars when this payload can persist research rows."""
    if bool(payload.get("persist_bars_for_forward")):
        return True
    if payload.get("signals"):
        return True
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    if diagnostics.get("touch_events") or diagnostics.get("rearm_events"):
        return True
    try:
        near_limit = max(0.0, float(os.environ.get("RESEARCH_NEAREST_APPROACH_MAX_PCT", "1.0")))
    except (TypeError, ValueError):
        near_limit = 1.0
    for zone_diag in (diagnostics.get("zones") or {}).values():
        if not isinstance(zone_diag, dict) or not isinstance(zone_diag.get("closest_midpoint_bar"), dict):
            continue
        try:
            distance = float(zone_diag.get("closest_midpoint_distance_pct"))
        except (TypeError, ValueError):
            continue
        if 0.0 < distance <= near_limit:
            return True
    return False


def _prefetch_missing_research_5m_bars(
    payload_batch: list[dict[str, Any]],
    analysis_meta: dict[str, dict],
    *,
    limit: int,
    max_workers: int = 2,
) -> dict[int, list[dict]]:
    """Fetch missing forward bars concurrently, preserving serial persistence.

    Read-only market-data requests use a small bounded pool. The results are keyed
    by object identity so two symbols with similar payloads can never share the
    wrong bars. Persistence and per-symbol application remain serial below.
    """
    pending = [
        payload for payload in payload_batch
        if _research_payload_needs_5m_history(payload)
        and len(payload.get("bars_5m") or []) < max(1, int(limit))
    ]
    if not pending:
        return {}
    workers = max(1, min(4, int(max_workers), len(pending)))
    results: dict[int, list[dict]] = {}
    if workers == 1:
        for payload in pending:
            try:
                results[id(payload)] = _fetch_missing_research_5m_bars(payload, analysis_meta, limit)
            except Exception as exc:
                log.warning("[RESEARCH_PENDING_BARS_POST] %s | unavailable: %s", payload.get("symbol"), exc)
                results[id(payload)] = []
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_missing_research_5m_bars, payload, analysis_meta, limit): payload
            for payload in pending
        }
        for future in as_completed(futures):
            payload = futures[future]
            try:
                results[id(payload)] = future.result()
            except Exception as exc:
                log.warning("[RESEARCH_PENDING_BARS_POST] %s | unavailable: %s", payload.get("symbol"), exc)
                results[id(payload)] = []
    return results


def _account_context_asof(decision_ts: Any) -> dict[str, Any] | None:
    """Return the latest persisted account snapshot no later than a decision boundary."""
    target = research._parse_timestamp(decision_ts)
    path = research.ACCOUNT_CONTEXT_PATH
    if target is None or not path.exists():
        return None
    best_row = None
    best_ts = None
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                stamp = research._parse_timestamp(row.get("captured_at_ms") or row.get("captured_at") or row.get("recorded_at"))
                if stamp is None or stamp > target:
                    continue
                if best_ts is None or stamp > best_ts:
                    best_row, best_ts = row, stamp
    except OSError as exc:
        log.warning("[RESEARCH_ACCOUNT_ASOF] could not read prior account snapshots: %s", exc)
        return None
    if best_row is None:
        return None
    result = dict(best_row)
    result["persisted"] = True
    result["account_context_asof"] = True
    result["account_context_age_ms_at_decision"] = max(0.0, (target - best_ts).total_seconds() * 1000.0)
    return result


def _research_persistence_batch_size(raw_value: str | None = None) -> int:
    """Return a bounded batch size; a malformed optional env var must not abort a run."""
    raw = os.environ.get("RESEARCH_PERSISTENCE_BATCH_SIZE", "32") if raw_value is None else raw_value
    try:
        configured = int(raw)
    except (TypeError, ValueError):
        log.warning("[CONFIG] invalid RESEARCH_PERSISTENCE_BATCH_SIZE=%r; using default 32", raw)
        configured = 32
    return max(4, min(32, configured))


def main() -> None:
    request_metrics.reset()
    started = time.time()
    run_started_perf = time.perf_counter()
    fresh_runtime_start = not DATA.exists()
    DATA.mkdir(parents=True, exist_ok=True)
    _write_runtime_version_metadata()
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
            "open_position_cap_enabled": False, "trades_per_cycle_cap_enabled": False, "max_entry_slippage_pct": MAX_ENTRY_SLIPPAGE_PCT, "max_entry_signal_displacement_pct": MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT,
            "execution_quote_max_age_sec": EXECUTION_QUOTE_MAX_AGE_SEC,
            "execution_require_exchange_timestamp": os.environ.get("EXECUTION_REQUIRE_EXCHANGE_TIMESTAMP", "true"),
            "execution_unknown_quote_age_policy": os.environ.get("EXECUTION_UNKNOWN_QUOTE_AGE_POLICY", "block"),
            "research_rich_context_enabled": RESEARCH_RICH_CONTEXT_ENABLED,
            "entry_trigger_reference": "TRIGGER_BAR_CLOSE",
            "trend_filter_enabled": TREND_FILTER_ENABLED,
            "trend_filter_mode": TREND_FILTER_MODE,
            "trend_filter_version": "trend-v1-2026-10-06",
            "trend_filter_min_1h_bars": TREND_FILTER_MIN_1H_BARS,
            "trend_filter_min_4h_bars": TREND_FILTER_MIN_4H_BARS,
            "trend_filter_persistence_lookback_1h": TREND_FILTER_PERSISTENCE_LOOKBACK_1H,
            "trend_filter_persistence_lookback_4h": TREND_FILTER_PERSISTENCE_LOOKBACK_4H,
            "trend_filter_slope_lookback_4h": TREND_FILTER_SLOPE_LOOKBACK_4H,
            "trend_filter_require_persistence": TREND_FILTER_REQUIRE_PERSISTENCE,
            "entry_context_telemetry_enabled": ENTRY_CONTEXT_TELEMETRY_ENABLED,
            "entry_context_depth_levels": ENTRY_CONTEXT_DEPTH_LEVELS,
            "entry_context_trades_limit": ENTRY_CONTEXT_TRADES_LIMIT,
            "entry_context_http_timeout_sec": ENTRY_CONTEXT_HTTP_TIMEOUT_SEC,
            "research_schema_version": research.RESEARCH_SCHEMA_VERSION,
            "research_defer_until_after_execution": RESEARCH_DEFER_UNTIL_AFTER_EXECUTION,
            "max_market_spread_pct": MAX_MARKET_SPREAD_PCT, "max_5m_trigger_age_minutes": MAX_5M_TRIGGER_AGE_MINUTES,
            "shadow_short_filter_experiment": research.SHADOW_SHORT_FILTER_EXPERIMENT,
            "shadow_short_body_to_range_gt": research.SHADOW_SHORT_BODY_TO_RANGE_GT,
            "shadow_short_lower_wick_lt": research.SHADOW_SHORT_LOWER_WICK_LT,
            "shadow_short_btc_ema50_gt_025": research.SHADOW_SHORT_BTC_EMA50_GT_025,
            "shadow_short_btc_ema50_gt_050": research.SHADOW_SHORT_BTC_EMA50_GT_050,
        },
    )

    # 1) Private account layer is optional for a scan. Never let missing credentials
    #    prevent public market analysis from running.
    private_ready = _private_layer_ready()
    # Final tracking is allowed after an ambiguous execution only when the local
    # state AND mandatory private preflight passed. Credentials/state-file validity
    # alone are insufficient: position-mode, provenance, tracker, and reconciliation
    # must get through their guarded preflight calls before the final checkpoint runs.
    private_checkpoint_state_trusted = False
    if private_ready:
        try:
            _load_active_trades_file(fresh_runtime_start=fresh_runtime_start)
            log.info("[STATE_PREFLIGHT] active-trade state integrity OK")
        except Exception as exc:
            private_ready = False
            log.critical(
                "[LIVE_BLOCKED_STATE] active-trade state preflight failed; execution/reconciliation disabled for this run: %s",
                exc,
            )
            try:
                _append_jsonl(ACTIONS_PATH, {
                    "ts": int(time.time() * 1000),
                    "action": "LIVE_EXECUTION_BLOCKED_STATE_UNTRUSTED",
                    "event_id": None,
                    "symbol": None,
                    "direction": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "scope": "cycle_preflight",
                })
            except Exception:
                pass
    if private_ready:
        try:
            with request_metrics.phase("private_preflight"):
                mode = get_position_mode(timeout_sec=float(os.environ.get("PRIVATE_PREFLIGHT_TIMEOUT_SEC", "5")))
            log.info("[AUTH] BingX private preflight OK | position_mode=%s", mode)
        except Exception as exc:
            log.error("[AUTH] BingX private preflight failed: %s; execution/reconciliation disabled for this run", exc)
            private_ready = False
        if private_ready:
            try:
                backfilled = backfill_active_trade_provenance()
                if backfilled:
                    log.info("[TRACKER_PROVENANCE] backfilled code_commit_sha for %d active trade(s)", backfilled)
            except ActiveTradeStateCorrupt as exc:
                private_ready = False
                private_checkpoint_state_trusted = False
                log.critical("[LIVE_BLOCKED_STATE] tracker provenance found untrusted active state: %s", exc)
            except Exception as exc:
                private_ready = False
                private_checkpoint_state_trusted = False
                log.critical("[LIVE_BLOCKED_STATE] tracker provenance/backfill failed; execution disabled for this run: %s", exc)
            if private_ready:
                try:
                    with request_metrics.phase("pre_scan_tracker"):
                        _update_active_trades_with_snapshot()
                except ActiveTradeStateCorrupt as exc:
                    private_ready = False
                    private_checkpoint_state_trusted = False
                    log.critical("[LIVE_BLOCKED_STATE] tracker update found untrusted active state: %s", exc)
                except Exception as exc:
                    private_ready = False
                    private_checkpoint_state_trusted = False
                    log.exception("[LIVE_BLOCKED_STATE] active trade update failed; execution disabled for this run: %s", exc)
            if private_ready:
                try:
                    with request_metrics.phase("pre_scan_reconcile"):
                        reconciliation_ok = reconcile_all_open_positions()
                    # A completed reconciliation call may report unresolved exchange
                    # state (private_ready becomes false) but is still safe to retry at
                    # the final checkpoint. Exceptions and earlier preflight failures
                    # leave this false, preventing permissive tracker loads on corrupt
                    # or account-mode-unverified local state.
                    private_checkpoint_state_trusted = True
                    if reconciliation_ok is False:
                        private_ready = False
                        log.critical("[LIVE_BLOCKED_RECON] unresolved exchange/local position state; new entries disabled for this run")
                except Exception as exc:
                    private_ready = False
                    private_checkpoint_state_trusted = False
                    log.exception("[LIVE_BLOCKED_RECON] reconciliation failed; execution disabled for this run: %s", exc)
    else:
        log.info("[PRIVATE] BingX private layer unavailable | reconciliation/execution disabled")

    # 2) Dynamic universe. Crypto uses Binance public Spot/Vision candles.
    # TradFi/equity contracts are analyzed from BingX candles because Binance
    # public Spot market data does not expose the corresponding stock universe.
    with request_metrics.phase("universe_setup"):
        bingx_symbols = get_scan_symbols()
        bingx_contract_map = contracts()
    log.info("[SCAN] %s symbols selected from BingX | count=%d", "WATCHLIST" if WATCHLIST_ONLY else "FULL_UNIVERSE", len(bingx_symbols))
    try:
        with request_metrics.phase("universe_setup"):
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
    if research.RESEARCH_ENABLED and not RESEARCH_DEFER_UNTIL_AFTER_EXECUTION:
        try:
            with request_metrics.phase("research_preload_account"):
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
    if research.RESEARCH_ENABLED and not RESEARCH_DEFER_UNTIL_AFTER_EXECUTION:
        try:
            with request_metrics.phase("research_preload_btc_5m"):
                btc_raw_5m = fetch_binance_klines("BTCUSDT", interval="5m", limit=288, retryable=False)
            if btc_raw_5m:
                btc_research_df_5m = pd.DataFrame(btc_raw_5m)
                research.persist_market_bars("BTC-USDT", "5m", btc_raw_5m, provider="binance", source="binance_spot_btc_context", scan_id=scan_id, code_commit_sha=CODE_COMMIT_SHA)
        except Exception as exc:
            log.warning("[RESEARCH_BTC] 5m context unavailable: %s", exc)
        try:
            with request_metrics.phase("research_preload_btc_1h"):
                btc_raw_1h = fetch_binance_klines("BTCUSDT", interval="1h", limit=300, retryable=False)
            if btc_raw_1h:
                btc_research_df_1h = pd.DataFrame(btc_raw_1h)
                research.persist_market_bars("BTC-USDT", "1h", btc_raw_1h, provider="binance", source="binance_spot_btc_context", scan_id=scan_id, code_commit_sha=CODE_COMMIT_SHA)
        except Exception as exc:
            log.warning("[RESEARCH_BTC] 1h context unavailable: %s", exc)

    if TREND_FILTER_ENABLED and (btc_research_df_1h is None or btc_research_df_1h.empty):
        try:
            with request_metrics.phase("trend_preload_btc_1h"):
                btc_trend_rows = fetch_binance_klines("BTCUSDT", interval="1h", limit=max(12, TREND_FILTER_MIN_1H_BARS + TREND_FILTER_HISTORY_BUFFER_BARS), retryable=False)
            if btc_trend_rows:
                btc_research_df_1h = pd.DataFrame(btc_trend_rows)
        except Exception as exc:
            log.warning("[TREND_BTC] 1h context unavailable: %s", exc)

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
            with request_metrics.phase("execution_state_load"):
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
    deferred_research_rows: list[dict[str, Any]] = []
    deferred_research_lock = threading.Lock()

    @request_metrics.instrument_scan_symbol
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
            min_bars = max(SWING_LEN * 2 + 10, SIGNALS_MIN_BARS)
            if len(bars) < min_bars:
                return {
                    "symbol": symbol, "current_price": None, "price_position": "INSUFFICIENT_DATA",
                    "fresh_signal": "—", "active_demand": 0, "active_supply": 0,
                    "zones": {"demand": [], "supply": []}, "last_signal_count": 0,
                    "error": f"insufficient_1h_candles:{len(bars)}<{min_bars}", "signals": [],
                }

            df, supply, demand, signals = generate_zone_signals(pd.DataFrame(bars), symbol=symbol, mode=DIAGNOSTICS_MODE)
            # The provider payload may include the still-forming 1H candle. The
            # strategy drops it before counting MIN_BARS, so classify a 70-raw / 69-closed
            # history as insufficient data rather than misleadingly reporting "no zones".
            if len(df) < SIGNALS_MIN_BARS:
                return {
                    "symbol": symbol, "current_price": None, "price_position": "INSUFFICIENT_DATA",
                    "fresh_signal": "—", "active_demand": 0, "active_supply": 0,
                    "zones": {"demand": [], "supply": []}, "last_signal_count": 0,
                    "error": f"insufficient_closed_1h_candles:{len(df)}<{SIGNALS_MIN_BARS}", "signals": [],
                }
            latest_price = float(df["close"].iloc[-1])
            bingx_price = _bingx_last_price(contract)
            latest_closed_idx = len(df) - 1
            latest_bar_open_time = _coerce_utc_timestamp(df["timestamp"].iloc[-1])
            if latest_bar_open_time is None:
                raise ValueError("latest 1H bar timestamp is invalid")
            if "close_time" in df.columns:
                latest_closed_time = _coerce_utc_timestamp(df["close_time"].iloc[-1])
            else:
                latest_closed_time = latest_bar_open_time + pd.Timedelta(hours=1)
            if latest_closed_time is None:
                latest_closed_time = latest_bar_open_time + pd.Timedelta(hours=1)
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
                    "latest_bar_open_time": latest_bar_open_time.isoformat(),
                    "latest_bar_close_time": latest_closed_time.isoformat(),
                    "latest_closed_time": latest_closed_time.isoformat(),
                    "trigger_status": "DATA_STALE_REJECT",
                    "data_stale_reject": True,
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
                if research.RESEARCH_ENABLED and not RESEARCH_DEFER_UNTIL_AFTER_EXECUTION and str(symbol).upper() in research_pending_forward_symbols:
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
            decision_boundary_ts = pd.Timestamp.now(tz="UTC")
            for sig in recent:
                try:
                    # Shadow features for a signal must be anchored to that signal's
                    # closed 5m trigger boundary, not the later scan wall-clock.
                    boundary_ms = _signal_decision_boundary_ts_ms(sig)
                    shadow_decision_ts = pd.Timestamp(boundary_ms, unit="ms", tz="UTC").isoformat() if boundary_ms is not None else sig.get("decision_boundary_ts")
                    sig["shadow_experiments"] = shadow.build_entry_snapshot(
                        sig, df_5m=pd.DataFrame(trigger_bars_raw), df_1h=df,
                        account_context=research_account_context, decision_ts=shadow_decision_ts
                    )
                except Exception as shadow_exc:
                    log.warning("[SHADOW_TELEMETRY] %s | snapshot build failed: %s", _display_symbol(symbol), shadow_exc)
                market_snapshot = sig.setdefault("market_snapshot", {})
                trigger_ts = _coerce_utc_timestamp(sig.get("trigger_bar_close_time") or sig.get("decision_boundary_ts") or sig.get("trigger_bar_time") or sig.get("time")) or pd.Timestamp.now(tz="UTC")
                market_snapshot.update({
                    "bingx_live_price": bingx_price,
                    "binance_live_price": binance_live_price,
                    "market_spread_pct": spread_pct,
                    "venue_price_deviation_pct": spread_pct,
                    "market_spread_metric": "cross_venue_price_deviation",
                    "scan_time": pd.Timestamp.now(tz="UTC").isoformat(),
                    "trigger_age_minutes_at_scan": max(0.0, (pd.Timestamp.now(tz="UTC") - trigger_ts).total_seconds() / 60.0),
                })

            # Research context is captured only for symbols with a new boundary,
            # but raw bars are also retained for symbols with maturing observations.
            # A pending observation still needs both timeframes even when its zone
            # is no longer active and no new visit boundary is generated.
            research_context = None
            research_bars_5m_raw: list[dict[str, Any]] = []
            research_bars_1h_raw: list[dict[str, Any]] = []
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
            persist_bars_for_forward = str(symbol).upper() in research_pending_forward_symbols
            research_bars_1h_raw, research_bars_5m_raw = _research_bar_payload(
                research_enabled=research.RESEARCH_ENABLED,
                has_research_boundary=has_research_boundary,
                persist_bars_for_forward=persist_bars_for_forward,
                bars_1h=bars,
                bars_5m=trigger_bars_raw,
            )
            collect_rich_context = RESEARCH_RICH_CONTEXT_ENABLED and not RESEARCH_DEFER_UNTIL_AFTER_EXECUTION and (bool(research_signals) or (rich_rejection_context and bool(touch_events or rearm_events)))
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

            # Trend Filter is diagnostic-only by default. It evaluates the same
            # candidate against causal 4H/1H data at the closed 5m trigger boundary.
            if TREND_FILTER_ENABLED and recent:
                for sig in recent:
                    try:
                        boundary_ms = _signal_decision_boundary_ts_ms(sig)
                        if boundary_ms is None:
                            raise ValueError("missing_decision_boundary")
                        h1_for_trend, h1_refetched = _ensure_trend_history(symbol=symbol, timeframe="1h", frame=df, provider=provider, binance_symbol=binance_symbol, decision_ts_ms=boundary_ms, min_bars=TREND_FILTER_MIN_1H_BARS)
                        h4_for_trend, h4_refetched = _ensure_trend_history(symbol=symbol, timeframe="4h", frame=None, provider=provider, binance_symbol=binance_symbol, decision_ts_ms=boundary_ms, min_bars=TREND_FILTER_MIN_4H_BARS)
                        snapshot=evaluate_trend_filter(symbol=symbol, direction=str(sig.get("type","")), event_type=str((sig.get("trigger") or {}).get("type") or "ZONE_5M"), df_1h=h1_for_trend, df_4h=h4_for_trend, btc_1h_df=btc_research_df_1h, decision_ts_ms=boundary_ms, min_bars_1h=TREND_FILTER_MIN_1H_BARS, min_bars_4h=TREND_FILTER_MIN_4H_BARS, persistence_lookback_1h=TREND_FILTER_PERSISTENCE_LOOKBACK_1H, persistence_lookback_4h=TREND_FILTER_PERSISTENCE_LOOKBACK_4H, slope_lookback_4h=TREND_FILTER_SLOPE_LOOKBACK_4H, require_persistence=TREND_FILTER_REQUIRE_PERSISTENCE, mode=TREND_FILTER_MODE)
                        snapshot.update({"decision_boundary_ts":pd.Timestamp(boundary_ms,unit="ms",tz="UTC").isoformat(),"history_1h_refetched":bool(h1_refetched),"history_4h_refetched":bool(h4_refetched),"source_1h":provider,"source_4h":provider,"source_btc":"binance_spot"})
                        sig["trend_filter"]=snapshot
                    except Exception as trend_exc:
                        log.warning("[TREND_FILTER] %s | diagnostic failed: %s", _display_symbol(symbol), trend_exc)
                        sig["trend_filter"]={"version":"trend-v1-2026-10-06","enabled":True,"mode":TREND_FILTER_MODE,"trend_decision":"REJECT","trend_reject_reason":f"TREND_RUNTIME_ERROR:{type(trend_exc).__name__}"}

            # When live execution is enabled, keep the production critical path free of
            # research persistence/I/O. Capture the exact decision-time inputs in memory
            # and flush them after all selected live attempts are finished.
            research_decision_ts = decision_boundary_ts.isoformat()
            if research.RESEARCH_ENABLED and RESEARCH_DEFER_UNTIL_AFTER_EXECUTION:
                with deferred_research_lock:
                    deferred_research_rows.append({
                        "symbol": symbol,
                        "strategy_version": _effective_strategy_version(),
                        "code_commit_sha": CODE_COMMIT_SHA,
                        "provider": provider,
                        "source": source_name,
                        "bars_1h": research_bars_1h_raw,
                        "bars_5m": research_bars_5m_raw,
                        "df_1h": df.copy(),
                        "demand": demand,
                        "supply": supply,
                        "diagnostics": five_min_diag,
                        "symbol_state": symbol_state,
                        "signals": research_signals,
                        "decision_ts": research_decision_ts,
                        "market_context": None,
                        "persist_bars_for_forward": persist_bars_for_forward,
                    })
            elif research.RESEARCH_ENABLED:
                research.record_scan_symbol(
                    scan_id=scan_id, symbol=symbol, strategy_version=_effective_strategy_version(),
                    code_commit_sha=CODE_COMMIT_SHA, provider=provider, source=source_name,
                    bars_1h=research_bars_1h_raw, bars_5m=research_bars_5m_raw, df_1h=df, demand=demand, supply=supply,
                    diagnostics=five_min_diag, symbol_state=symbol_state, signals=research_signals,
                    decision_ts=research_decision_ts, market_context=research_context,
                    persist_bars_for_forward=persist_bars_for_forward,
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
                "latest_bar_open_time": latest_bar_open_time.isoformat(),
                "latest_bar_close_time": latest_closed_time.isoformat(),
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
    scan_started_perf = time.perf_counter()
    for batch_start in range(0, total, SCAN_BATCH_SIZE):
        batch_started_perf = time.perf_counter()
        batch = symbols[batch_start: batch_start + SCAN_BATCH_SIZE]
        batch_results: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=min(SCAN_WORKERS, len(batch))) as executor:
            future_to_symbol = {executor.submit(scan_one, symbol): symbol for symbol in batch}
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                except Exception as exc:  # defensive: scan_one already catches errors
                    fallback_source_name = str((analysis_meta.get(symbol) or {}).get("source") or "unknown")
                    result = {
                        "symbol": symbol, "current_price": None, "binance_price": None, "bingx_price": None, "market_spread_pct": None,
                        "market_source": fallback_source_name, "binance_symbol": analysis_meta.get(symbol, {}).get("binance_symbol"),
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
        log.info("[SCAN_BATCH_DONE] batch=%d scanned=%d/%d symbols=%d seconds=%.3f", batch_start // max(1, SCAN_BATCH_SIZE) + 1, scanned, total, len(batch), time.perf_counter() - batch_started_perf)
        if scanned < total and SCAN_BATCH_PAUSE_SEC > 0:
            time.sleep(SCAN_BATCH_PAUSE_SEC)

    scan_errors = sum(1 for row in scan_rows if row.get("price_position") == "ERROR")
    scan_insufficient_data = sum(1 for row in scan_rows if row.get("price_position") == "INSUFFICIENT_DATA")
    scan_contract_not_found = sum(1 for row in scan_rows if row.get("price_position") == "CONTRACT_NOT_FOUND")
    scan_stale_rejects = sum(
        1 for row in scan_rows
        if row.get("data_stale_reject") is True
        or str(row.get("trigger_status", "")) in {"BLOCKED_STALE_TOUCH", "DATA_STALE_REJECT"}
        or "DATA_STALE_REJECT" in str(row.get("fresh_signal", ""))
    )
    scan_skips = scan_insufficient_data + scan_contract_not_found + scan_stale_rejects
    scan_duration_sec = time.perf_counter() - scan_started_perf
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
                trigger_ts = _coerce_utc_timestamp(signal.get("trigger_bar_close_time") or signal.get("decision_boundary_ts") or signal.get("trigger_bar_time") or signal.get("time"))
                if trigger_ts is None:
                    raise ValueError("invalid_5m_trigger_timestamp")
                age_min = max(0.0, (now_exec - trigger_ts).total_seconds() / 60.0)
            except Exception:
                log.warning("[EXEC_REJECT_TRIGGER_TIME] %s %s | invalid 5m trigger time=%s", _display_symbol(signal.get("symbol")), signal.get("type"), signal.get("trigger_bar_time"))
                _exec_gate("invalid_5m_trigger_time", signal)
                if _clear_stale_pending_zone_visit(zone_visit_state, signal):
                    _save_zone_visit_state(zone_visit_state)
                continue
            signal["execution_age_minutes"] = age_min
            if age_min > MAX_5M_TRIGGER_AGE_MINUTES:
                log.info("[EXEC_REJECT_5M_AGE] %s %s | age_min=%.2f allowed=%.2f", _display_symbol(signal["symbol"]), signal["type"], age_min, MAX_5M_TRIGGER_AGE_MINUTES)
                _exec_gate("5m_trigger_too_old", signal)
                # A trigger that expired after scanning must not leave its zone
                # locked with an un-retryable pending_signal.
                if _clear_stale_pending_zone_visit(zone_visit_state, signal):
                    _save_zone_visit_state(zone_visit_state)
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
        trend_snapshot = signal.get("trend_filter")
        if TREND_FILTER_ENABLED and TREND_FILTER_MODE == "enforce" and _trend_filter_enforce_reject(trend_snapshot, TREND_FILTER_MODE):
            _exec_gate(f"trend_filter:{(trend_snapshot or {}).get('trend_reject_reason', 'unknown')}", signal)
            continue
        executable.append(signal)

    log.info("[EXEC_INPUT] fresh_signals=%d unique_symbols=%d open_positions=%d successful_event_ids=%d terminal_event_ids=%d failed_signal_records=%d", len(fresh_signals), len(latest_by_symbol), len(open_keys), len(successful_ids), len(terminal_event_ids), len(failed_ids))

    # Safety ordering: newest signal bar first; score only breaks ties.
    executable.sort(key=lambda x: (-int(x["idx"]), -float(x.get("score", 0.0))))
    log.info("[EXEC_GATE_SUMMARY] input_signals=%d executable=%d rejected=%d reasons=%s", len(latest_by_symbol), len(executable), sum(exec_gate_stats.values()), json.dumps(exec_gate_stats, ensure_ascii=False, sort_keys=True))

    executed = 0
    # No engine-imposed portfolio-size or per-cycle admission cap. Every signal that
    # survives the normal execution gates below is submitted in deterministic order.
    # Exchange-native/account-native limits, protection readiness, duplicate event
    # claims, and existing symbol ownership checks remain enforced.
    selected_signals = executable
    for rank, signal in enumerate(selected_signals, start=1):
        execution_gate_ts = pd.Timestamp.now(tz="UTC")
        signal["execution_gate_ts"] = execution_gate_ts.isoformat()
        trigger_raw = signal.get("trigger_bar_close_time") or signal.get("decision_boundary_ts") or signal.get("trigger_bar_time") or signal.get("time")
        try:
            trigger_ts = _coerce_utc_timestamp(trigger_raw)
            if trigger_ts is None:
                raise ValueError("invalid_trigger_timestamp")
            signal["trigger_to_execution_gate_minutes"] = max(0.0, (execution_gate_ts - trigger_ts).total_seconds() / 60.0)
        except Exception:
            signal["trigger_to_execution_gate_minutes"] = None
        _record_entry_decision(
            scan_id,
            signal,
            "EXECUTION_SELECTED",
            "selected_for_execution",
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
            blocked = _blocked_private_execution_result()
            log.error("[EXEC_BLOCKED] %s %s: %s", signal["symbol"], signal["type"], blocked["error"])
            try:
                _append_jsonl(TRADES_PATH, {
                    "record_type": "TRADE_BLOCKED",
                    "event_id": signal["event_id"],
                    "symbol": signal["symbol"],
                    "direction": signal["type"],
                    "score": signal["score"],
                    "signal": signal,
                    "result": blocked,
                })
            except Exception as blocked_log_exc:
                log.error("[TRADE_BLOCKED_JOURNAL_FAILED] event_id=%s error=%s", signal.get("event_id"), blocked_log_exc)
            block_reason = _private_block_decision_reason(blocked)
            _record_entry_decision(
                scan_id, signal, "EXECUTION_RESULT", block_reason,
                selection_rank=rank, execution_status=blocked.get("status"), terminal=True,
            )
            _send_signal(signal, blocked)
            continue
        event_id = str(signal["event_id"])
        claimed, claim_reason, attempt_id = _claim_event_for_execution(event_id)
        if not claimed:
            _record_entry_decision(scan_id, signal, "EXECUTION_CLAIM", claim_reason, selection_rank=rank, execution_status="CLAIM_REJECTED", terminal=(claim_reason in {"terminal", "completed", "claim_error"}))
            log.info("[EXEC_SKIP_CLAIM] %s %s | event_id=%s reason=%s", signal["symbol"], signal["type"], event_id, claim_reason)
            continue
        _record_entry_decision(scan_id, signal, "EXECUTION_CLAIM", "claimed", selection_rank=rank, attempt_id=attempt_id, execution_status="CLAIMED", terminal=False)
        signal["attempt_id"] = attempt_id
        # Capture decision-time market context after the event claim but before the exchange order.
        # The attempt id is therefore present in the immutable context record without changing
        # the existing order/protection semantics. Collection remains fail-open.
        if ENTRY_CONTEXT_TELEMETRY_ENABLED and research.RESEARCH_ENABLED:
            try:
                boundary_ms = _signal_decision_boundary_ts_ms(signal)
                ctx = fetch_research_market_context(
                    signal["symbol"], depth_limit=ENTRY_CONTEXT_DEPTH_LEVELS, trades_limit=ENTRY_CONTEXT_TRADES_LIMIT,
                    timeout_sec=ENTRY_CONTEXT_HTTP_TIMEOUT_SEC,
                )
                capture_now_ms = int(time.time() * 1000)
                capture_start_ms = ctx.get("captured_at_ms")
                capture_completed_ms = ctx.get("capture_completed_at_ms") or capture_now_ms
                boundary_iso = pd.Timestamp(boundary_ms, unit="ms", tz="UTC").isoformat() if boundary_ms else None
                trigger_to_capture = (capture_now_ms - boundary_ms) if boundary_ms else None
                ctx.update({
                    "scan_id": scan_id, "event_id": event_id, "attempt_id": attempt_id,
                    "symbol": signal["symbol"],
                    # This record contains BingX execution-venue telemetry. Keep
                    # the venue provider explicit and preserve the analysis/bar
                    # provider separately; conflating them breaks research provenance.
                    "provider": "bingx",
                    "analysis_provider": str(((signal.get("market_snapshot") or {}).get("analysis_provider") or "binance")),
                    "analysis_source": str(((signal.get("market_snapshot") or {}).get("analysis_source") or "")) or None,
                    "context_provider": "bingx",
                    "context_source": "bingx_swap_public",
                    "source": "pre_execution_entry_context",
                    "strategy_version": signal.get("strategy_version", _effective_strategy_version()),
                    "engine_version": ENGINE_VERSION, "code_commit_sha": CODE_COMMIT_SHA,
                    "decision_boundary_ts": boundary_iso,
                    "pre_execution_decision_ts": pd.Timestamp(capture_now_ms, unit="ms", tz="UTC").isoformat(),
                    "context_age_ms_at_order_ready": max(0, capture_now_ms - int(capture_completed_ms)),
                    "context_lag_ms_from_decision_boundary": trigger_to_capture,
                    "capture_phase": "PRE_EXECUTION",
                    "feature_time_semantics": "CAPTURED_PRE_ORDER_NOT_TRIGGER_BOUNDARY",
                })
                ctx["persisted"] = bool(research.record_market_context(ctx))
                signal["pre_execution_market_context"] = ctx
                _record_entry_decision(
                    scan_id, signal, "PRE_EXECUTION_CONTEXT", "captured", selection_rank=rank,
                    attempt_id=attempt_id, execution_status="CONTEXT_CAPTURED", terminal=False,
                )
            except Exception as ctx_exc:
                log.warning("[ENTRY_CONTEXT] %s | collection/persistence failed (fail-open): %s", _display_symbol(signal.get("symbol")), ctx_exc)
                signal["pre_execution_market_context"] = {
                    "schema_version": research.RESEARCH_SCHEMA_VERSION, "status": "error",
                    "capture_phase": "PRE_EXECUTION", "feature_time_semantics": "CAPTURE_FAILED",
                    "error_type": type(ctx_exc).__name__, "captured_at_ms": None, "persisted": False,
                    "attempt_id": attempt_id, "event_id": event_id,
                }
                _record_entry_decision(
                    scan_id, signal, "PRE_EXECUTION_CONTEXT", "collection_failed", selection_rank=rank,
                    attempt_id=attempt_id, execution_status="CONTEXT_COLLECTION_FAILED", terminal=False,
                )
        execution_call_start_ts = pd.Timestamp.now(tz="UTC")
        pre_ctx = signal.get("pre_execution_market_context")
        if isinstance(pre_ctx, dict):
            try:
                completed_ms = pre_ctx.get("capture_completed_at_ms")
                if completed_ms is not None:
                    pre_ctx["context_age_ms_at_execution_call_start"] = max(0, int(execution_call_start_ts.timestamp() * 1000) - int(completed_ms))
                pre_ctx["execution_call_start_ts"] = execution_call_start_ts.isoformat()
                signal["pre_execution_market_context"] = pre_ctx
            except (TypeError, ValueError):
                pass
        execution = execute_new_position(signal)
        execution_finish_ts = pd.Timestamp.now(tz="UTC")
        execution["attempt_id"] = attempt_id
        execution["execution_call_start_ts"] = execution_call_start_ts.isoformat()
        execution["execution_finish_ts"] = execution_finish_ts.isoformat()
        execution["execution_wall_clock_seconds"] = max(0.0, (execution_finish_ts - execution_call_start_ts).total_seconds())
        execution["trigger_to_execution_finish_minutes"] = _elapsed_minutes(signal.get("trigger_bar_close_time") or signal.get("decision_boundary_ts") or signal.get("trigger_bar_time") or signal.get("time"), execution_finish_ts.isoformat())
        execution_status = str(execution.get("status", ""))
        if execution_status in {"opened_emergency_close_unverified", "opened_registration_failed_close_unverified", "entry_state_unverified"}:
            private_ready = False
            log.critical("[LIVE_BLOCKED_EXECUTION_STATE] status=%s symbol=%s direction=%s; no further entries this cycle", execution_status, signal.get("symbol"), signal.get("type"))
        rejected_stale_entry = _is_stale_entry_rejection(execution)
        if rejected_stale_entry:
            if _clear_stale_pending_zone_visit(zone_visit_state, signal):
                _save_zone_visit_state(zone_visit_state)
        _append_execution_ledger(event_id, attempt_id, "EXECUTION_RESULT", {
            "status": execution_status,
            "error": execution.get("error"),
            "execution_call_start_ts": execution_call_start_ts.isoformat(),
            "execution_finish_ts": execution_finish_ts.isoformat(),
            "execution_wall_clock_seconds": execution.get("execution_wall_clock_seconds"),
            "trigger_to_execution_finish_minutes": execution.get("trigger_to_execution_finish_minutes"),
            "order_id": (execution.get("order") or {}).get("order_id") if isinstance(execution.get("order"), dict) else None,
            "fill_price": (execution.get("position") or {}).get("avgPrice") if isinstance(execution.get("position"), dict) else None,
        })
        if execution_status in {"skipped_min_qty", "skipped_tp_min_qty", "skipped_invalid_setup"}:
            log.warning(
                "[EXEC_SKIPPED] %s %s | status=%s | reason=%s | error=%s | qty=%s min_qty=%s required_margin=%.4f configured_margin=%.4f",
                signal["symbol"], signal["type"], execution_status, execution.get("reason", "invalid_setup"), execution.get("error"), execution.get("qty"), execution.get("min_qty"),
                float(execution.get("required_margin_usdt", 0.0) or 0.0), float(execution.get("configured_margin_usdt", MARGIN_USDT) or MARGIN_USDT),
            )
        elif not _is_protected_execution_status(execution):
            log.error("[EXEC_FAILED] %s %s | status=%s | error=%s | order=%s", signal["symbol"], signal["type"], execution_status, execution.get("error"), execution.get("order"))
            if execution_status not in {"DISABLED", "BLOCKED_MISSING_CREDENTIALS", "BLOCKED_PRIVATE_PREFLIGHT"}:
                terminal_failure = _is_terminal_execution_failure(execution)
                _mark_failed_signal(
                    signal["event_id"], execution_status, execution.get("error", ""),
                    force_terminal=terminal_failure,
                )
            else:
                terminal_failure = False
        else:
            terminal_failure = False
            if execution_status == "entry_state_unverified_protected":
                # Protecting/registering the currently observed quantity is the
                # safest immediate action, but an unresolved entry remainder can
                # still change exposure. Do not admit unrelated signals later in
                # this same scan before final reconciliation retries cancellation
                # and repairs protection from a fresh exchange position snapshot.
                private_ready = False
                terminal_event_ids.add(event_id)
                log.critical(
                    "[EXEC_PROTECTED_ENTRY_UNVERIFIED] %s %s event=%s is protected and registered, "
                    "but entry order remains unsettled; retaining the event claim and blocking further entries until reconciliation",
                    signal["symbol"], signal["type"], event_id,
                )
        _record_entry_decision(
            scan_id,
            signal,
            "EXECUTION_RESULT",
            execution_status or "empty_execution_status",
            selection_rank=rank,
            attempt_id=attempt_id,
            execution_status=execution_status,
            terminal=(True if _is_protected_execution_status(execution) else bool(_is_terminal_execution_failure(execution))),
        )
        if _is_protected_execution_status(execution):
            bx_position_symbol = str(((execution.get("position") or {}).get("symbol") or signal.get("symbol"))).upper()
            open_keys.add((bx_position_symbol, str(signal.get("type")).upper()))
            _finalize_event_claim(event_id, attempt_id, terminal=True, status=execution_status)
        elif _is_terminal_execution_failure(execution):
            terminal_event_ids.add(event_id)
            _finalize_event_claim(event_id, attempt_id, terminal=True, status=execution_status)
        else:
            _finalize_event_claim(event_id, attempt_id, terminal=False, status=execution_status)
        try:
            _append_jsonl(TRADES_PATH, {
                "record_type": _trade_event_record_type(execution_status),
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
        except Exception as trade_log_exc:
            # Exchange execution / active-trade registration has already happened.
            # A secondary journal failure must not skip the final tracker checkpoint
            # or runtime-state artifact; fail closed for additional entries this run.
            private_ready = False
            log.critical(
                "[TRADE_EVENT_JOURNAL_FAILED] event_id=%s status=%s error=%s; further entries blocked, final reconciliation retained",
                event_id, execution_status, trade_log_exc, exc_info=True,
            )
        if execution_status == "opened_then_emergency_closed":
            emergency_close = _emergency_trade_close_record(signal, execution)
            if emergency_close is not None:
                try:
                    _append_jsonl(TRADES_PATH, emergency_close)
                except Exception as close_log_exc:
                    private_ready = False
                    log.critical(
                        "[EMERGENCY_CLOSE_JOURNAL_FAILED] event_id=%s error=%s; final reconciliation retained",
                        event_id, close_log_exc, exc_info=True,
                    )
            else:
                log.error(
                    "[EMERGENCY_CLOSE_LEDGER_PENDING] event_id=%s: rollback is not sufficiently evidenced for a canonical close record",
                    event_id,
                )
        _send_signal(signal, execution)
        if str(execution.get("status")) == "opened_protected":
            executed += 1

    # Persist the scan summary and run the final position/protection checkpoint
    # before slow deferred research I/O. This does not skip tracker/reconciliation;
    # it prevents research writes from delaying TP1->BE and close-state observation.
    try:
        save_scan(scan_rows, fresh_signals, duration_sec=scan_duration_sec, scan_id=scan_id)
    except Exception as scan_save_exc:
        log.exception("[SCAN_SAVE_FAILED] scan summary could not be persisted; continuing final tracker/reconciliation: %s", scan_save_exc)
    # Final tracking checkpoint: orders/positions can change during scan/execution.
    # This remains a full pass with existing error handling and protection checks.
    if private_checkpoint_state_trusted:
        try:
            with request_metrics.phase("final_tracker"):
                _update_active_trades_with_snapshot()
        except Exception as exc:
            log.exception("[TRACKER_FINAL] active trade update failed: %s", exc)
        try:
            with request_metrics.phase("final_reconcile"):
                reconcile_all_open_positions()
        except Exception as exc:
            log.exception("[RECON_FINAL] reconciliation failed: %s", exc)

    # Deferred research persistence is deliberately last: decision-boundary
    # observations remain pre-entry data, while the live tracker/protection
    # checkpoint above is not delayed by the slow write phase.
    if research.RESEARCH_ENABLED and RESEARCH_DEFER_UNTIL_AFTER_EXECUTION and deferred_research_rows:
        # Post-execution research phase: never attach post-trade context to earlier
        # decision-time observations, because that would create a lookahead artifact.
        log.info("[RESEARCH_DEFERRED_FLUSH] rows=%d | live execution path complete", len(deferred_research_rows))
        try:
            with request_metrics.phase("research_post_account"):
                post_account = fetch_research_account_snapshot()
            if post_account is not None:
                post_account.update({
                    "scan_id": scan_id,
                    "strategy_version": _effective_strategy_version(),
                    "code_commit_sha": CODE_COMMIT_SHA,
                    "capture_phase": "post_execution",
                    "decision_context_valid": False,
                    "account_context_id": post_account.get("account_context_id") or research.stable_id(
                        "account-context-post-execution-v1", scan_id, post_account.get("captured_at_ms", ""), prefix="AC_"
                    ),
                })
                research.record_account_context(post_account)
        except Exception as exc:
            log.warning("[RESEARCH_ACCOUNT_POST] account context unavailable: %s", exc)
        btc_raw_5m: list[dict[str, Any]] = []
        btc_raw_1h: list[dict[str, Any]] = []
        try:
            with request_metrics.phase("research_post_btc_5m"):
                btc_raw_5m = fetch_binance_klines("BTCUSDT", interval="5m", limit=288, retryable=False)
            if btc_raw_5m:
                research.persist_market_bars("BTC-USDT", "5m", btc_raw_5m, provider="binance", source="binance_spot_btc_context_post_execution", scan_id=scan_id, code_commit_sha=CODE_COMMIT_SHA)
        except Exception as exc:
            log.warning("[RESEARCH_BTC_POST] 5m context unavailable: %s", exc)
        try:
            with request_metrics.phase("research_post_btc_1h"):
                btc_raw_1h = fetch_binance_klines("BTCUSDT", interval="1h", limit=300, retryable=False)
            if btc_raw_1h:
                research.persist_market_bars("BTC-USDT", "1h", btc_raw_1h, provider="binance", source="binance_spot_btc_context_post_execution", scan_id=scan_id, code_commit_sha=CODE_COMMIT_SHA)
        except Exception as exc:
            log.warning("[RESEARCH_BTC_POST] 1h context unavailable: %s", exc)

        btc_df_5m = pd.DataFrame(btc_raw_5m) if btc_raw_5m else None
        btc_df_1h = pd.DataFrame(btc_raw_1h) if btc_raw_1h else None
        deferred_rows = list(deferred_research_rows)
        research_batch_size = _research_persistence_batch_size()
        queued_observation_count = 0
        queued_bar1h_count = 0
        queued_bar5m_count = 0
        persisted_observation_count = 0
        persisted_bar1h_count = 0
        persisted_bar5m_count = 0
        persistence_error_count = 0
        for batch_offset in range(0, len(deferred_rows), research_batch_size):
            payload_batch = deferred_rows[batch_offset:batch_offset + research_batch_size]
            research_batch_started = time.perf_counter()
            batch_observations = 0
            batch_bars1h = 0
            batch_bars5m = 0
            pending_limit = max(
                12,
                int(os.environ.get("RESEARCH_PENDING_5M_FETCH_LIMIT", "24")),
                max(300, int(os.environ.get("RESEARCH_5M_HISTORY_LIMIT", "288"))),
            )
            try:
                prefetch_workers = int(os.environ.get("RESEARCH_DEFERRED_BAR_FETCH_WORKERS", "4"))
            except (TypeError, ValueError):
                prefetch_workers = 4
            prefetched_bars = _prefetch_missing_research_5m_bars(
                payload_batch, analysis_meta, limit=pending_limit, max_workers=prefetch_workers,
            )
            with research.batch_persistence() as persistence_result:
                for payload in payload_batch:
                    bars_5m = _prefer_longer_research_5m_history(payload.get("bars_5m"), prefetched_bars.get(id(payload)))
                    try:
                        counts = research.record_scan_symbol(
                            scan_id=scan_id, symbol=payload["symbol"], strategy_version=payload["strategy_version"],
                            code_commit_sha=payload["code_commit_sha"], provider=payload["provider"], source=payload["source"],
                            bars_1h=payload["bars_1h"], bars_5m=bars_5m, df_1h=payload["df_1h"],
                            demand=payload["demand"], supply=payload["supply"], diagnostics=payload["diagnostics"],
                            symbol_state=payload["symbol_state"], signals=payload["signals"], decision_ts=payload["decision_ts"],
                            market_context=None,
                            account_context=_account_context_asof(payload.get("decision_ts")),
                            btc_df_5m=btc_df_5m, btc_df_1h=btc_df_1h,
                            persist_bars_for_forward=bool(payload.get("persist_bars_for_forward")),
                        )
                        batch_observations += int(counts.get("observations", 0))
                        batch_bars1h += int(counts.get("bars_1h", 0))
                        batch_bars5m += int(counts.get("bars_5m", 0))
                    except Exception as exc:
                        log.warning("[RESEARCH_DEFERRED_FLUSH] %s | persistence failed: %s", payload["symbol"], exc)
            queued_observation_count += batch_observations
            queued_bar1h_count += batch_bars1h
            queued_bar5m_count += batch_bars5m
            actual = persistence_result.get("flush_result", {}) if isinstance(persistence_result, dict) else {}
            persisted_observation_count += int(actual.get("observations_written", 0))
            persisted_bar1h_count += int(actual.get("bars_1h_written", 0))
            persisted_bar5m_count += int(actual.get("bars_5m_written", 0))
            persistence_error_count += int(actual.get("persistence_errors", 0))
            log.info("[RESEARCH_BATCH_DONE] batch=%d rows=%d/%d queued_observations=%d queued_bars1h=%d queued_bars5m=%d seconds=%.3f", batch_offset // research_batch_size + 1, min(batch_offset + len(payload_batch), len(deferred_rows)), len(deferred_rows), batch_observations, batch_bars1h, batch_bars5m, time.perf_counter() - research_batch_started)

    total_duration_sec = time.perf_counter() - run_started_perf
    request_metrics.log_summary(log)
    log.info("[DONE] symbols=%d fresh_signals=%d executed=%d duration=%.1fs", len(scan_rows), len(fresh_signals), executed, total_duration_sec)


if __name__ == "__main__":
    main()
