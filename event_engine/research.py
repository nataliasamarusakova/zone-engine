from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

log = logging.getLogger("event_engine.research")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RESEARCH_ENABLED = os.environ.get("RESEARCH_ENABLED", "true").lower() == "true"
RESEARCH_SCHEMA_VERSION = 1

ZONE_OBSERVATIONS_PATH = DATA_DIR / "zone_observations.jsonl"
ENTRY_DECISIONS_PATH = DATA_DIR / "entry_decisions.jsonl"
MARKET_BARS_1H_PATH = DATA_DIR / "market_bars_1h.jsonl"
MARKET_BARS_5M_PATH = DATA_DIR / "market_bars_5m.jsonl"
RESEARCH_BAR_CURSORS_PATH = DATA_DIR / "research_bar_cursors.json"
RESEARCH_MANIFEST_PATH = DATA_DIR / "research_manifest.json"
RESEARCH_ERRORS_PATH = DATA_DIR / "research_persistence_errors.jsonl"
RESEARCH_OUTCOMES_PATH = DATA_DIR / "research_outcomes.jsonl"
RESEARCH_OUTCOME_STATE_PATH = DATA_DIR / "research_outcome_state.json"
MARKET_CONTEXT_PATH = DATA_DIR / "market_context.jsonl"
ACCOUNT_CONTEXT_PATH = DATA_DIR / "account_context.jsonl"

INITIAL_1H_BARS = max(1, int(os.environ.get("RESEARCH_INITIAL_1H_BARS", "48")))
INITIAL_5M_BARS = max(1, int(os.environ.get("RESEARCH_INITIAL_5M_BARS", "24")))
NEAREST_APPROACH_MAX_PCT = max(0.0, float(os.environ.get("RESEARCH_NEAREST_APPROACH_MAX_PCT", "1.0")))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse ISO or epoch-second/millisecond timestamps deterministically as UTC."""
    if value is None:
        return None
    try:
        if isinstance(value, pd.Timestamp):
            ts = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                return None
            magnitude = abs(float(value))
            unit = "ms" if magnitude >= 1e11 else "s" if magnitude >= 1e8 else None
            ts = pd.to_datetime(value, unit=unit, utc=True) if unit else pd.to_datetime(value, utc=True)
        else:
            text = str(value).strip()
            if not text:
                return None
            numeric = pd.to_numeric(text, errors="coerce")
            if pd.notna(numeric):
                magnitude = abs(float(numeric))
                unit = "ms" if magnitude >= 1e11 else "s" if magnitude >= 1e8 else None
                ts = pd.to_datetime(numeric, unit=unit, utc=True) if unit else pd.to_datetime(text, utc=True)
            else:
                ts = pd.to_datetime(text, utc=True)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts if not pd.isna(ts) else None
    except Exception:
        return None


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sanitize(value: Any) -> Any:
    """Convert pandas/numpy values and non-finite floats to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return sanitize(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def stable_id(*parts: Any, prefix: str = "") -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()[:24]
    return f"{prefix}{digest}" if prefix else digest


def bar_id(symbol: str, timeframe: str, ts: Any, provider: str = "") -> str:
    return stable_id(str(symbol).upper(), str(timeframe).lower(), str(ts), str(provider).lower(), prefix="BAR_")


def observation_id(event_type: str, symbol: str, direction: str, zone_id: str, ts: Any, *, visit_id: str = "") -> str:
    parsed = _parse_timestamp(ts)
    canonical_ts = parsed.isoformat() if parsed is not None else str(ts)
    return stable_id(
        "observation-v1", event_type, str(symbol).upper(), str(direction).upper(), zone_id, visit_id, canonical_ts,
        prefix="OBS_",
    )


def decision_id(scan_id: str, event_id: str, stage: str, reason: str, attempt_id: str | None = None) -> str:
    return stable_id(
        "decision-v1", scan_id, event_id, stage, reason, attempt_id or "", prefix="DEC_"
    )


def _lock_path(path: Path) -> Path:
    return path.with_name(path.name + ".lock")


class _FileLock:
    def __init__(self, path: Path) -> None:
        self.path = _lock_path(path)
        self.handle = None

    def __enter__(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()


def _append_jsonl_locked(path: Path, records: Iterable[dict[str, Any]]) -> int:
    rows = [sanitize(r) for r in records]
    if not rows:
        return 0
    with _FileLock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return len(rows)


def _load_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("[RESEARCH_STATE_CORRUPT] path=%s error_type=%s error=%s", path, type(exc).__name__, exc)
        return default


def _atomic_json_locked(path: Path, payload: Any) -> None:
    with _FileLock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(sanitize(payload), fh, ensure_ascii=False, indent=2, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def _record_error(kind: str, path: Path, exc: Exception, *, symbol: str | None = None) -> None:
    payload = {
        "ts": _now_iso(),
        "kind": kind,
        "path": str(path.relative_to(PROJECT_ROOT)) if path.is_absolute() and path.is_relative_to(PROJECT_ROOT) else str(path),
        "symbol": symbol,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    try:
        _append_jsonl_locked(RESEARCH_ERRORS_PATH, [payload])
    except Exception:
        import logging
        logging.getLogger(__name__).error("research persistence error: %s", payload)


def record_zone_observations(records: Iterable[dict[str, Any]]) -> int:
    if not RESEARCH_ENABLED:
        return 0
    rows = []
    for row in records:
        payload = dict(row)
        payload.setdefault("schema_version", RESEARCH_SCHEMA_VERSION)
        payload.setdefault("record_type", "ZONE_OBSERVATION")
        payload.setdefault("recorded_at", _now_iso())
        if not payload.get("observation_id"):
            payload["observation_id"] = observation_id(
                payload.get("event_type", "UNKNOWN"),
                payload.get("symbol", ""),
                payload.get("direction", ""),
                payload.get("zone_id", ""),
                payload.get("observation_ts", payload.get("ts", "")),
                visit_id=str(payload.get("zone_visit_id", "")),
            )
        rows.append(payload)
    unique_rows = []
    seen_ids: set[str] = set()
    for row in rows:
        oid = str(row.get("observation_id", ""))
        if oid and oid in seen_ids:
            continue
        if oid:
            seen_ids.add(oid)
        unique_rows.append(row)
    rows = unique_rows
    try:
        count = _append_jsonl_locked(ZONE_OBSERVATIONS_PATH, rows)
        _bump_manifest("zone_observations_written", count)
        return count
    except Exception as exc:
        _record_error("zone_observation_write", ZONE_OBSERVATIONS_PATH, exc)
        _bump_manifest("persistence_errors", 1)
        return 0


def record_entry_decision(row: dict[str, Any], *, path: Path | None = None) -> bool:
    if not RESEARCH_ENABLED:
        return False
    payload = dict(row)
    payload.setdefault("schema_version", RESEARCH_SCHEMA_VERSION)
    payload.setdefault("record_type", "ENTRY_DECISION")
    payload.setdefault("recorded_at", _now_iso())
    payload.setdefault(
        "decision_id",
        decision_id(
            str(payload.get("scan_id", "")), str(payload.get("event_id", "")),
            str(payload.get("stage", "")), str(payload.get("reason", "")), payload.get("attempt_id"),
        ),
    )
    target = path or ENTRY_DECISIONS_PATH
    try:
        count = _append_jsonl_locked(target, [payload])
        if target == ENTRY_DECISIONS_PATH:
            _bump_manifest("entry_decisions_written", count)
        return bool(count)
    except Exception as exc:
        _record_error("entry_decision_write", target, exc, symbol=str(payload.get("symbol", "")))
        if target == ENTRY_DECISIONS_PATH:
            _bump_manifest("persistence_errors", 1)
        return False


def _cursor_key(symbol: str, timeframe: str, provider: str) -> str:
    return f"{str(symbol).upper()}|{str(timeframe).lower()}|{str(provider).lower()}"


def _load_cursors() -> dict[str, Any]:
    raw = _load_json(RESEARCH_BAR_CURSORS_PATH, {})
    return raw if isinstance(raw, dict) else {}


def _save_cursors(cursors: dict[str, Any]) -> None:
    _atomic_json_locked(RESEARCH_BAR_CURSORS_PATH, cursors)


def _update_cursor(key: str, value: dict[str, Any]) -> None:
    """Atomically update one cursor key without losing concurrent workers' keys."""
    with _FileLock(RESEARCH_BAR_CURSORS_PATH):
        cursors = _load_json(RESEARCH_BAR_CURSORS_PATH, {})
        if not isinstance(cursors, dict):
            cursors = {}
        cursors[key] = value
        RESEARCH_BAR_CURSORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=RESEARCH_BAR_CURSORS_PATH.name + ".", suffix=".tmp", dir=str(RESEARCH_BAR_CURSORS_PATH.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(sanitize(cursors), fh, ensure_ascii=False, indent=2, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, RESEARCH_BAR_CURSORS_PATH)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def persist_market_bars(
    symbol: str,
    timeframe: str,
    bars: Iterable[dict[str, Any]] | pd.DataFrame,
    *,
    provider: str,
    source: str,
    scan_id: str,
    code_commit_sha: str | None = None,
) -> int:
    if not RESEARCH_ENABLED:
        return 0
    tf = str(timeframe).lower()
    if tf not in {"1h", "5m"}:
        raise ValueError(f"unsupported research timeframe: {timeframe}")
    try:
        iterable = bars.to_dict("records") if isinstance(bars, pd.DataFrame) else list(bars)
        if not iterable:
            return 0
        delta = pd.Timedelta(hours=1) if tf == "1h" else pd.Timedelta(minutes=5)
        now = pd.Timestamp.now(tz="UTC")
        normalized: list[dict[str, Any]] = []
        invalid_count = 0
        for raw in iterable:
            ts_value = _parse_timestamp(raw.get("timestamp"))
            if ts_value is None:
                invalid_count += 1
                continue
            try:
                open_price = float(raw.get("open")); high = float(raw.get("high")); low = float(raw.get("low")); close = float(raw.get("close")); volume = float(raw.get("volume"))
            except (TypeError, ValueError):
                invalid_count += 1
                continue
            if not all(math.isfinite(v) for v in (open_price, high, low, close, volume)):
                invalid_count += 1
                continue
            if min(open_price, high, low, close) <= 0 or volume < 0 or high < low or high < max(open_price, close) or low > min(open_price, close):
                invalid_count += 1
                continue
            raw_close_ts = _parse_timestamp(raw.get("close_time") or raw.get("closeTime"))
            effective_close_ts = raw_close_ts if raw_close_ts is not None else ts_value + delta
            if effective_close_ts > now:
                continue
            record = {
                "bar_id": bar_id(symbol, tf, ts_value.isoformat(), provider),
                "record_type": "MARKET_BAR", "schema_version": RESEARCH_SCHEMA_VERSION,
                "symbol": str(symbol).upper(), "timeframe": tf, "provider": str(provider).lower(),
                "source": source, "timestamp": ts_value.isoformat(), "close_time": effective_close_ts.isoformat(),
                "open": open_price, "high": high, "low": low, "close": close, "volume": volume,
                "scan_id": scan_id, "code_commit_sha": code_commit_sha, "recorded_at": _now_iso(),
            }
            for raw_key in ("quote_volume", "taker_buy_base", "taker_buy_quote", "taker_flow_valid", "bar_delta_usdt", "trade_count"):
                if raw.get(raw_key) is not None:
                    value = raw.get(raw_key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
                        continue
                    record[raw_key] = value
            normalized.append(record)
        if invalid_count:
            _bump_manifest("market_bars_invalid", invalid_count)
        normalized.sort(key=lambda r: str(r["timestamp"]))
        unique_by_timestamp = {}
        for row in normalized:
            unique_by_timestamp[str(row["timestamp"])] = row
        normalized = list(unique_by_timestamp.values())
        key = _cursor_key(symbol, tf, provider)
        cursors = _load_cursors()
        previous = str((cursors.get(key) or {}).get("last_timestamp", ""))
        if previous:
            fresh = [r for r in normalized if str(r["timestamp"]) > previous]
        else:
            bootstrap = INITIAL_1H_BARS if tf == "1h" else INITIAL_5M_BARS
            fresh = normalized[-bootstrap:]
        if not fresh:
            return 0
        path = MARKET_BARS_1H_PATH if tf == "1h" else MARKET_BARS_5M_PATH
        written = _append_jsonl_locked(path, fresh)
        if written:
            _update_cursor(key, {"last_timestamp": fresh[-1]["timestamp"], "last_bar_id": fresh[-1]["bar_id"], "updated_at": _now_iso(), "scan_id": scan_id})
            _bump_manifest(f"market_bars_{tf}_written", written)
        return written
    except Exception as exc:
        target = MARKET_BARS_1H_PATH if tf == "1h" else MARKET_BARS_5M_PATH
        _record_error("market_bar_write", target, exc, symbol=symbol)
        _bump_manifest("persistence_errors", 1)
        return 0


def _bump_manifest(key: str, amount: int) -> None:
    try:
        with _FileLock(RESEARCH_MANIFEST_PATH):
            current = _load_json(RESEARCH_MANIFEST_PATH, {})
            if not isinstance(current, dict):
                current = {}
            current.setdefault("schema_version", RESEARCH_SCHEMA_VERSION)
            current.setdefault("created_at", _now_iso())
            current["updated_at"] = _now_iso()
            current[key] = int(current.get(key, 0) or 0) + int(amount)
            current["research_enabled"] = RESEARCH_ENABLED
            RESEARCH_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=RESEARCH_MANIFEST_PATH.name + ".", suffix=".tmp", dir=str(RESEARCH_MANIFEST_PATH.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(current, fh, ensure_ascii=False, indent=2, allow_nan=False)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, RESEARCH_MANIFEST_PATH)
            finally:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
    except Exception as exc:
        _record_error("manifest_update", RESEARCH_MANIFEST_PATH, exc)


def update_manifest_run(*, scan_id: str, code_commit_sha: str | None, effective_config: dict[str, Any]) -> None:
    if not RESEARCH_ENABLED:
        return
    try:
        with _FileLock(RESEARCH_MANIFEST_PATH):
            current = _load_json(RESEARCH_MANIFEST_PATH, {})
            if not isinstance(current, dict):
                current = {}
            current.setdefault("schema_version", RESEARCH_SCHEMA_VERSION)
            current.setdefault("created_at", _now_iso())
            current.update({
                "updated_at": _now_iso(),
                "last_scan_id": scan_id,
                "last_code_commit_sha": code_commit_sha,
                "effective_config": sanitize(effective_config),
                "files": {
                    "zone_observations": str(ZONE_OBSERVATIONS_PATH.relative_to(PROJECT_ROOT)),
                    "entry_decisions": str(ENTRY_DECISIONS_PATH.relative_to(PROJECT_ROOT)),
                    "market_bars_1h": str(MARKET_BARS_1H_PATH.relative_to(PROJECT_ROOT)),
                    "market_bars_5m": str(MARKET_BARS_5M_PATH.relative_to(PROJECT_ROOT)),
                    "bar_cursors": str(RESEARCH_BAR_CURSORS_PATH.relative_to(PROJECT_ROOT)),
                    "outcomes": str(RESEARCH_OUTCOMES_PATH.relative_to(PROJECT_ROOT)),
                    "outcome_state": str(RESEARCH_OUTCOME_STATE_PATH.relative_to(PROJECT_ROOT)),
                    "manifest": str(RESEARCH_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
                    "errors": str(RESEARCH_ERRORS_PATH.relative_to(PROJECT_ROOT)),
                    "market_context": str(MARKET_CONTEXT_PATH.relative_to(PROJECT_ROOT)),
                    "account_context": str(ACCOUNT_CONTEXT_PATH.relative_to(PROJECT_ROOT)),
                },
            })
            fd, tmp_name = tempfile.mkstemp(prefix=RESEARCH_MANIFEST_PATH.name + ".", suffix=".tmp", dir=str(RESEARCH_MANIFEST_PATH.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(sanitize(current), fh, ensure_ascii=False, indent=2, allow_nan=False)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_name, RESEARCH_MANIFEST_PATH)
            finally:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
    except Exception as exc:
        _record_error("manifest_run_update", RESEARCH_MANIFEST_PATH, exc)




def _closed_market_frame(df: pd.DataFrame | None, decision_ts: Any, timeframe: str) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x["timestamp"] = x["timestamp"].map(_parse_timestamp)
    delta = pd.Timedelta(minutes=5) if str(timeframe).lower() == "5m" else pd.Timedelta(hours=1)
    if "close_time" in x.columns:
        x["close_time"] = x["close_time"].map(_parse_timestamp)
    else:
        x["close_time"] = x["timestamp"] + delta
    for col in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote", "bar_delta_usdt"):
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["timestamp", "close_time", "close"]).sort_values("timestamp")
    asof = _parse_timestamp(decision_ts)
    if asof is not None:
        x = x.loc[x["close_time"] <= asof]
    return x.reset_index(drop=True)


def _simple_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    if len(df) < period or not all(c in df.columns for c in ("high", "low", "close")):
        return None
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    value = tr.tail(period).mean()
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None


def _session_name(ts: pd.Timestamp | None) -> str | None:
    if ts is None:
        return None
    h = int(ts.hour)
    if 13 <= h < 16:
        return "LONDON_NY_OVERLAP"
    if 7 <= h < 13:
        return "LONDON"
    if 16 <= h < 21:
        return "NEW_YORK"
    return "ASIA_OFF_HOURS"


def _derive_time_series_features(*, direction: str, df_5m: pd.DataFrame | None, df_1h: pd.DataFrame | None, decision_ts: Any, btc_df_5m: pd.DataFrame | None, btc_df_1h: pd.DataFrame | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "session_utc": _session_name(_parse_timestamp(decision_ts)),
        "5m_atr14": None, "5m_range_atr": None,
        "5m_return_5m_pct": None, "5m_return_15m_pct": None, "5m_return_30m_pct": None, "5m_return_1h_pct": None,
        "5m_return_3h_pct": None, "5m_return_6h_pct": None, "5m_return_12h_pct": None, "5m_return_24h_pct": None,
        "5m_ema20_distance_pct": None, "5m_ema50_distance_pct": None,
        "5m_ema20_slope_5m_pct": None, "5m_ema50_slope_5m_pct": None,
        "5m_atr_pct": None, "5m_return_volatility_20": None, "market_regime": None,
        "session_vwap_approx": None, "session_vwap_kind": "UTC_DAY_TP_VWAP", "session_vwap_distance_pct": None, "session_cvd_proxy_quote": None,
        "structure_high_20": None, "structure_low_20": None, "distance_to_structure_high_pct": None, "distance_to_structure_low_pct": None,
        "break_above_prior_20": False, "break_below_prior_20": False,
        "liquidity_sweep_bullish_20": False, "liquidity_sweep_bearish_20": False,
        "fvg_bullish_5m": False, "fvg_bearish_5m": False,
        "htf4h_return_4h_pct": None, "htf4h_ema50_distance_pct": None, "htf4h_ema200_distance_pct": None,
        "htf4h_trend": None,
        "btc_return_5m_pct": None, "btc_return_1h_pct": None,
        "btc_ema50_distance_pct": None, "btc_ema200_distance_pct": None,
    }
    x5 = _closed_market_frame(df_5m, decision_ts, "5m")
    if len(x5) >= 2:
        current = _safe_float(x5["close"].iloc[-1])
        if current:
            for n, key in ((1, "5m_return_5m_pct"), (3, "5m_return_15m_pct"), (6, "5m_return_30m_pct"), (12, "5m_return_1h_pct"), (36, "5m_return_3h_pct"), (72, "5m_return_6h_pct"), (144, "5m_return_12h_pct"), (288, "5m_return_24h_pct")):
                if len(x5) > n:
                    prev = _safe_float(x5["close"].iloc[-1-n])
                    if prev and prev > 0:
                        raw = (current / prev - 1.0) * 100.0
                        out[key] = raw if str(direction).upper() == "LONG" else -raw
        atr = _simple_atr(x5, 14)
        out["5m_atr14"] = atr
        if atr and current:
            out["5m_range_atr"] = float((float(x5["high"].iloc[-1]) - float(x5["low"].iloc[-1])) / atr)
            out["5m_atr_pct"] = float(atr / current * 100.0)
        for span, key, slope_key in ((20, "5m_ema20_distance_pct", "5m_ema20_slope_5m_pct"), (50, "5m_ema50_distance_pct", "5m_ema50_slope_5m_pct")):
            ema_series = x5["close"].ewm(span=span, adjust=False, min_periods=span).mean()
            ema = ema_series.iloc[-1]
            if current and pd.notna(ema) and float(ema) > 0:
                out[key] = float((current / float(ema) - 1.0) * 100.0)
                if len(ema_series) >= 2 and pd.notna(ema_series.iloc[-2]) and float(ema_series.iloc[-2]) > 0:
                    out[slope_key] = float((float(ema) / float(ema_series.iloc[-2]) - 1.0) * 100.0)
        if len(x5) >= 21:
            rets = x5["close"].pct_change().tail(20).dropna()
            if len(rets) >= 10:
                out["5m_return_volatility_20"] = float(rets.std(ddof=1))
            ema20 = x5["close"].ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
            ema50 = x5["close"].ewm(span=50, adjust=False, min_periods=50).mean().iloc[-1]
            if pd.notna(ema20) and pd.notna(ema50) and current and current > 0:
                dist20 = abs(current / float(ema20) - 1.0) * 100.0 if float(ema20) > 0 else 0.0
                dist50 = abs(current / float(ema50) - 1.0) * 100.0 if float(ema50) > 0 else 0.0
                if float(ema20) > float(ema50) and current > float(ema20):
                    out["market_regime"] = "TREND_UP"
                elif float(ema20) < float(ema50) and current < float(ema20):
                    out["market_regime"] = "TREND_DOWN"
                elif max(dist20, dist50) < 1.0:
                    out["market_regime"] = "RANGE_NEAR_EMAS"
                else:
                    out["market_regime"] = "TRANSITION"
        dts = _parse_timestamp(decision_ts)
        day_start = dts.normalize() if dts is not None else x5["timestamp"].iloc[-1].normalize()
        day = x5.loc[x5["timestamp"] >= day_start]
        if not day.empty and "volume" in day.columns:
            vol_sum = float(day["volume"].sum())
            if vol_sum > 0:
                typical = (day["high"] + day["low"] + day["close"]) / 3.0
                vwap = float((typical * day["volume"]).sum() / vol_sum)
                out["session_vwap_approx"] = vwap
                if current:
                    out["session_vwap_distance_pct"] = float((current / vwap - 1.0) * 100.0)
        if "bar_delta_usdt" in day.columns:
            delta = pd.to_numeric(day["bar_delta_usdt"], errors="coerce").dropna()
            if not delta.empty:
                out["session_cvd_proxy_quote"] = float(delta.sum())
        if len(x5) >= 21:
            prior20 = x5.iloc[:-1].tail(20)
            ph, pl = float(prior20["high"].max()), float(prior20["low"].min())
            out["structure_high_20"], out["structure_low_20"] = ph, pl
            if current:
                out["distance_to_structure_high_pct"] = float((ph / current - 1.0) * 100.0)
                out["distance_to_structure_low_pct"] = float((current / pl - 1.0) * 100.0)
            latest = x5.iloc[-1]
            out["break_above_prior_20"] = bool(float(latest["close"]) > ph)
            out["break_below_prior_20"] = bool(float(latest["close"]) < pl)
            out["liquidity_sweep_bullish_20"] = bool(float(latest["low"]) < pl and float(latest["close"]) > pl)
            out["liquidity_sweep_bearish_20"] = bool(float(latest["high"]) > ph and float(latest["close"]) < ph)
        if len(x5) >= 3:
            a, c = x5.iloc[-3], x5.iloc[-1]
            out["fvg_bullish_5m"] = bool(float(c["low"]) > float(a["high"]))
            out["fvg_bearish_5m"] = bool(float(c["high"]) < float(a["low"]))
    # Derive a closed 4H context from 1H bars without requiring another API call.
    if df_1h is not None and not df_1h.empty:
        try:
            h1 = _closed_market_frame(df_1h, decision_ts, "1h")
            if not h1.empty:
                h1 = h1.set_index("timestamp")
                agg = h1.resample("4h", label="left", closed="left").agg(
                    open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                    volume=("volume", "sum"), bar_count=("close", "count"),
                ).dropna(subset=["open","high","low","close"])
                # Only use complete 4H candles; a currently-forming 4H bucket built
                # from one to three closed 1H bars is still an incomplete HTF candle.
                agg = agg.loc[agg["bar_count"] == 4].copy()
                if len(agg) >= 3:
                    c4 = float(agg["close"].iloc[-1])
                    out["htf4h_return_4h_pct"] = float((c4 / float(agg["close"].iloc[-2]) - 1.0) * 100.0) if float(agg["close"].iloc[-2]) > 0 else None
                    ema50_4h = agg["close"].ewm(span=50, adjust=False, min_periods=50).mean().iloc[-1]
                    ema200_4h = agg["close"].ewm(span=200, adjust=False, min_periods=200).mean().iloc[-1]
                    out["htf4h_ema50_distance_pct"] = float((c4 / float(ema50_4h) - 1.0) * 100.0) if pd.notna(ema50_4h) and float(ema50_4h) > 0 else None
                    out["htf4h_ema200_distance_pct"] = float((c4 / float(ema200_4h) - 1.0) * 100.0) if pd.notna(ema200_4h) and float(ema200_4h) > 0 else None
                    if pd.notna(ema50_4h):
                        out["htf4h_trend"] = "UP" if c4 > float(ema50_4h) else "DOWN"
        except Exception:
            pass

    bx5 = _closed_market_frame(btc_df_5m, decision_ts, "5m")
    if len(bx5) >= 13:
        bc = _safe_float(bx5["close"].iloc[-1]); p1 = _safe_float(bx5["close"].iloc[-2]); p12 = _safe_float(bx5["close"].iloc[-13])
        if bc and p1 and p1 > 0: out["btc_return_5m_pct"] = (bc / p1 - 1.0) * 100.0
        if bc and p12 and p12 > 0: out["btc_return_1h_pct"] = (bc / p12 - 1.0) * 100.0
    bx1 = _closed_market_frame(btc_df_1h, decision_ts, "1h")
    if len(bx1) >= 50:
        bc = _safe_float(bx1["close"].iloc[-1]); ema = bx1["close"].ewm(span=50, adjust=False, min_periods=50).mean().iloc[-1]
        if bc and pd.notna(ema) and float(ema) > 0: out["btc_ema50_distance_pct"] = (bc / float(ema) - 1.0) * 100.0
    if len(bx1) >= 200:
        bc = _safe_float(bx1["close"].iloc[-1]); ema = bx1["close"].ewm(span=200, adjust=False, min_periods=200).mean().iloc[-1]
        if bc and pd.notna(ema) and float(ema) > 0: out["btc_ema200_distance_pct"] = (bc / float(ema) - 1.0) * 100.0
    return out

def build_research_features(
    *,
    symbol: str,
    direction: str,
    zone: dict[str, Any],
    bar: dict[str, Any],
    df_1h: pd.DataFrame | None = None,
    df_5m: pd.DataFrame | None = None,
    touch_count_before_trigger: int | None = None,
    structure_room_r: float | None = None,
    decision_ts: str | None = None,
    market_context: dict[str, Any] | None = None,
    account_context: dict[str, Any] | None = None,
    btc_df_5m: pd.DataFrame | None = None,
    btc_df_1h: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute shadow-only features known no later than the research decision time."""
    top = _safe_float(zone.get("top"))
    bottom = _safe_float(zone.get("btm"))
    poi = _safe_float(zone.get("poi"))
    close = _safe_float(bar.get("close"))
    high = _safe_float(bar.get("high"))
    low = _safe_float(bar.get("low"))
    open_ = _safe_float(bar.get("open"))
    volume = _safe_float(bar.get("volume"))
    width = top - bottom if top is not None and bottom is not None else None
    rng = high - low if high is not None and low is not None else None
    body = abs(close - open_) if close is not None and open_ is not None else None
    features: dict[str, Any] = {
        "zone_age_bars": zone.get("age_bars"),
        "zone_origin_ts_ms": zone.get("origin_ts_ms"),
        "zone_age_hours": None,
        "departure_impulse_atr": None,
        "departure_body_to_range": None,
        "departure_volume_ratio_20": None,
        "market_context_id": (market_context or {}).get("context_id"),
        "account_context_id": (account_context or {}).get("account_context_id"),
        "feature_availability_ts": decision_ts,
        "zone_width_abs": width if width is not None and width > 0 else None,
        "zone_width_pct": (width / close * 100.0) if width is not None and close and close > 0 else None,
        "distance_to_midpoint_pct": (abs(close - poi) / poi * 100.0) if close is not None and poi and poi > 0 else None,
        "zone_penetration_ratio": None,
        "body_to_range": (body / rng) if body is not None and rng and rng > 0 else None,
        "upper_wick_ratio": ((high - max(open_, close)) / rng) if high is not None and open_ is not None and close is not None and rng and rng > 0 else None,
        "lower_wick_ratio": ((min(open_, close) - low) / rng) if low is not None and open_ is not None and close is not None and rng and rng > 0 else None,
        "close_location": ((close - low) / rng) if close is not None and low is not None and rng and rng > 0 else None,
        "range_pct": (rng / close * 100.0) if rng is not None and close and close > 0 else None,
        "volume": volume,
        "touch_count_before_trigger": touch_count_before_trigger,
        "structure_room_R": structure_room_r,
        "directional_candle_ok_current": (close >= open_) if direction == "LONG" and close is not None and open_ is not None else (close <= open_ if direction == "SHORT" and close is not None and open_ is not None else None),
        "shadow_directional_candle_ok": None,
        "shadow_directional_threshold": 0.30,
        "shadow_penetration_le_20pct": None,
        "shadow_penetration_le_50pct": None,
        "shadow_volume_ge_0_8": None,
        "shadow_volume_ge_1_2": None,
        "shadow_volume_ge_1_5": None,
        "shadow_structure_ge_0_5R": None,
        "shadow_structure_ge_1_0R": None,
        "shadow_structure_ge_1_2R": None,
        "shadow_structure_ge_1_5R": None,
        "shadow_age_le_24h": None,
        "shadow_age_le_48h": None,
        "shadow_age_le_72h": None,
    }
    if width is not None and width > 0 and high is not None and low is not None:
        penetration = ((top - low) / width) if direction == "LONG" and top is not None else ((high - bottom) / width if direction == "SHORT" and bottom is not None else None)
        features["zone_penetration_ratio"] = penetration
        if penetration is not None:
            features["shadow_penetration_le_20pct"] = penetration <= 0.20
            features["shadow_penetration_le_50pct"] = penetration <= 0.50

    vr = None
    if volume is not None and df_5m is not None and not df_5m.empty and "volume" in df_5m.columns and "timestamp" in df_5m.columns:
        try:
            trigger_ts = _parse_timestamp(bar.get("timestamp"))
            x5 = df_5m.copy()
            x5["timestamp"] = x5["timestamp"].map(_parse_timestamp)
            x5["volume"] = pd.to_numeric(x5["volume"], errors="coerce")
            x5 = x5.dropna(subset=["timestamp", "volume"]).sort_values("timestamp")
            prior = x5.loc[x5["timestamp"] < trigger_ts, "volume"].tail(20) if trigger_ts is not None else x5["volume"].tail(20)
            avg = float(prior.mean()) if len(prior) >= 20 else float("nan")
            if math.isfinite(avg) and avg > 0:
                vr = volume / avg
        except Exception:
            vr = None
    features["volume_ratio_5m20"] = vr
    # Explicit approach-to-zone metrics are computed from bars strictly before the trigger bar.
    if df_5m is not None and not df_5m.empty and bar.get("timestamp") is not None:
        try:
            trigger_ts = _parse_timestamp(bar.get("timestamp"))
            x5a = _closed_market_frame(df_5m, decision_ts, "5m")
            if trigger_ts is not None and not x5a.empty:
                pre = x5a.loc[x5a["timestamp"] < trigger_ts].copy()
                if not pre.empty:
                    approach_close = _safe_float(pre["close"].iloc[-1])
                    for n, key in ((1, "approach_return_5m_pct"), (3, "approach_return_15m_pct"), (6, "approach_return_30m_pct"), (12, "approach_return_1h_pct")):
                        if len(pre) > n and approach_close and approach_close > 0:
                            prev = _safe_float(pre["close"].iloc[-1-n])
                            if prev and prev > 0:
                                raw = (approach_close / prev - 1.0) * 100.0
                                features[key] = raw if direction == "LONG" else -raw
                    if features.get("approach_return_15m_pct") is not None:
                        features["approach_speed_15m_pct_per_min"] = abs(float(features["approach_return_15m_pct"])) / 15.0
                    if features.get("approach_return_30m_pct") is not None:
                        features["approach_speed_30m_pct_per_min"] = abs(float(features["approach_return_30m_pct"])) / 30.0
                    approach_atr = _simple_atr(pre, 14)
                    if approach_atr and approach_atr > 0:
                        for count, key in ((3, "approach_range_atr_15m"), (6, "approach_range_atr_30m")):
                            if len(pre) >= count:
                                rr = float((pre["high"].tail(count) - pre["low"].tail(count)).sum()) / approach_atr
                                features[key] = rr
                    # Count consecutive close-to-close moves in the signal direction; this
                    # remains meaningful even when individual candles are small/doji-like.
                    streak = 0
                    desired_positive = direction == "LONG"
                    for i in range(len(pre) - 1, 0, -1):
                        prev_close = _safe_float(pre["close"].iloc[i - 1]); curr_close = _safe_float(pre["close"].iloc[i])
                        if prev_close is None or curr_close is None or curr_close == prev_close:
                            break
                        same = curr_close > prev_close if desired_positive else curr_close < prev_close
                        if not same:
                            break
                        streak += 1
                    features["approach_directional_streak"] = streak
        except Exception:
            pass

    if vr is not None:
        features["shadow_volume_ge_0_8"] = vr >= 0.8
        features["shadow_volume_ge_1_2"] = vr >= 1.2
        features["shadow_volume_ge_1_5"] = vr >= 1.5

    if structure_room_r is not None:
        try:
            room = float(structure_room_r)
            if math.isfinite(room):
                features["shadow_structure_ge_0_5R"] = room >= 0.5
                features["shadow_structure_ge_1_0R"] = room >= 1.0
                features["shadow_structure_ge_1_2R"] = room >= 1.2
                features["shadow_structure_ge_1_5R"] = room >= 1.5
        except (TypeError, ValueError):
            pass

    if close is not None and open_ is not None and rng and rng > 0 and high is not None and low is not None:
        body_ratio = abs(close - open_) / rng
        if direction == "LONG":
            features["shadow_directional_candle_ok"] = close > open_ and ((close - low) / rng) >= 0.70 and body_ratio >= 0.30
        elif direction == "SHORT":
            features["shadow_directional_candle_ok"] = close < open_ and ((high - close) / rng) >= 0.70 and body_ratio >= 0.30

    origin_ms = zone.get("origin_ts_ms")
    try:
        origin = pd.to_datetime(int(origin_ms), unit="ms", utc=True) if origin_ms is not None else None
        trigger = _parse_timestamp(bar.get("timestamp"))
        if origin is not None and trigger is not None:
            features["zone_age_hours"] = max(0.0, (trigger - origin).total_seconds() / 3600.0)
    except Exception:
        pass
    age_h = features.get("zone_age_hours")
    if age_h is not None:
        features["shadow_age_le_24h"] = age_h <= 24
        features["shadow_age_le_48h"] = age_h <= 48
        features["shadow_age_le_72h"] = age_h <= 72

    # Higher-timeframe features are anchored to the latest fully closed 1H candle
    # known at the decision timestamp, preventing use of an in-progress 1H candle.
    if df_1h is not None and not df_1h.empty and "close" in df_1h.columns:
        try:
            asof_ts = _parse_timestamp(decision_ts or bar.get("decision_ts") or bar.get("observation_ts")) or _parse_timestamp(bar.get("timestamp"))
            x1 = df_1h.copy()
            x1["timestamp"] = x1["timestamp"].map(_parse_timestamp)
            x1["close"] = pd.to_numeric(x1["close"], errors="coerce")
            x1 = x1.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
            if asof_ts is not None:
                x1 = x1.loc[x1["timestamp"] + pd.Timedelta(hours=1) <= asof_ts]
            closes = x1["close"]
            if len(closes) >= 2:
                current = float(closes.iloc[-1])
                for label, bars_back in (("1h",1),("3h",3),("6h",6),("12h",12),("24h",24)):
                    if len(closes) > bars_back and current > 0:
                        prior = float(closes.iloc[-1-bars_back])
                        if math.isfinite(prior) and prior > 0:
                            features[f"return_{label}_pct"] = (current / prior - 1.0) * 100.0
                ema50_series = closes.ewm(span=50, adjust=False, min_periods=50).mean()
                ema200_series = closes.ewm(span=200, adjust=False, min_periods=200).mean()
                ema50 = ema50_series.iloc[-1]
                ema200 = ema200_series.iloc[-1]
                features["ema50_distance_pct"] = (current / float(ema50) - 1.0) * 100.0 if pd.notna(ema50) and float(ema50) > 0 else None
                features["ema200_distance_pct"] = (current / float(ema200) - 1.0) * 100.0 if pd.notna(ema200) and float(ema200) > 0 else None
                features["ema200_trend_up"] = bool(ema200 > ema200_series.iloc[-2]) if len(ema200_series) > 1 and pd.notna(ema200) and pd.notna(ema200_series.iloc[-2]) else None
                features["ema50_trend_up"] = bool(ema50 > ema50_series.iloc[-2]) if len(ema50_series) > 1 and pd.notna(ema50) and pd.notna(ema50_series.iloc[-2]) else None
        except Exception:
            pass
    features.update(_derive_time_series_features(
        direction=direction, df_5m=df_5m, df_1h=df_1h, decision_ts=decision_ts,
        btc_df_5m=btc_df_5m, btc_df_1h=btc_df_1h,
    ))
    context = dict(market_context or {})
    if account_context:
        context["account_context"] = account_context
    account = context.get("account_context") or {}
    if isinstance(account, dict):
        for src, dst in (("account_context_id", "account_context_id"), ("equity", "account_equity"), ("available_margin", "account_available_margin"), ("used_margin", "account_used_margin"), ("unrealized_profit", "account_unrealized_profit"), ("realized_profit", "account_realized_profit"), ("freezed_margin", "account_freezed_margin"), ("open_positions_count", "account_open_positions_count"), ("long_positions_count", "account_long_positions_count"), ("short_positions_count", "account_short_positions_count"), ("open_positions_notional_usdt", "account_open_positions_notional_usdt"), ("open_positions_unrealized_profit", "account_open_positions_unrealized_profit"), ("recent_fill_count", "account_recent_fill_count"), ("recent_fill_fee_total", "account_recent_fill_fee_total"), ("recent_fill_realized_pnl_total", "account_recent_fill_realized_pnl_total"), ("recent_force_order_count", "account_recent_force_order_count"), ("recent_liquidation_count", "account_recent_liquidation_count"), ("recent_adl_count", "account_recent_adl_count")):
            if src in account:
                features[dst] = account.get(src)
        try:
            used = float(account.get("used_margin")); equity = float(account.get("equity"))
            if math.isfinite(used) and math.isfinite(equity) and equity > 0:
                features["account_used_margin_ratio"] = used / equity
        except (TypeError, ValueError):
            features["account_used_margin_ratio"] = None
    for src, dst in (("funding_rate","funding_rate"),("mark_price","mark_price"),("index_price","index_price"),("open_interest","open_interest"),("open_interest_ts","open_interest_ts"),("premium_index_ts","premium_index_ts"),("next_funding_time_ms","next_funding_time_ms")):
        if src in context: features[dst] = context.get(src)
    book = context.get("order_book") or {}
    for src, dst in (("best_bid","book_best_bid"),("best_ask","book_best_ask"),("spread_pct","book_spread_pct"),("bid_qty_5","book_bid_qty_5"),("ask_qty_5","book_ask_qty_5"),("bid_qty_10","book_bid_qty_10"),("ask_qty_10","book_ask_qty_10"),("book_imbalance_5","book_imbalance_5"),("book_imbalance_10","book_imbalance_10"),("bid_quote_5","book_bid_quote_5"),("ask_quote_5","book_ask_quote_5"),("bid_quote_10","book_bid_quote_10"),("ask_quote_10","book_ask_quote_10"),("book_quote_imbalance_5","book_quote_imbalance_5"),("book_quote_imbalance_10","book_quote_imbalance_10"),("microprice","book_microprice"),("bid_depth_quote_0_1pct","book_bid_depth_quote_0_1pct"),("ask_depth_quote_0_1pct","book_ask_depth_quote_0_1pct"),("bid_depth_quote_0_5pct","book_bid_depth_quote_0_5pct"),("ask_depth_quote_0_5pct","book_ask_depth_quote_0_5pct"),("bid_depth_quote_1pct","book_bid_depth_quote_1pct"),("ask_depth_quote_1pct","book_ask_depth_quote_1pct")):
        if src in book: features[dst] = book.get(src)
    trades = context.get("recent_trades") or {}
    for src, dst in (("valid_trade_count","recent_trade_count"),("buy_aggressor_quote","recent_buy_aggressor_quote"),("sell_aggressor_quote","recent_sell_aggressor_quote"),("aggressor_delta_quote","recent_aggressor_delta_quote"),("buy_aggressor_ratio","recent_buy_aggressor_ratio"),("trade_min_price","recent_trade_min_price"),("trade_max_price","recent_trade_max_price"),("last_trade_ts","recent_last_trade_ts"),("first_trade_ts","recent_first_trade_ts"),("sample_span_seconds","recent_trade_sample_span_seconds"),("avg_trade_quote","recent_avg_trade_quote")):
        if src in trades: features[dst] = trades.get(src)
    # Zone-departure measurements from the existing origin; these never gate production.
    if df_1h is not None and not df_1h.empty and zone.get("start") is not None:
        try:
            x1 = _closed_market_frame(df_1h, decision_ts, "1h")
            start = int(zone.get("start"))
            if 0 <= start < len(x1):
                move = min(start + 6, len(x1)-1)
                base = _safe_float(x1.iloc[start].get("close")); endc = _safe_float(x1.iloc[move].get("close")); atr = _safe_float(x1.iloc[start].get("atr50")) if "atr50" in x1.columns else None
                if base and endc and atr and atr > 0: features["departure_impulse_atr"] = abs(endc-base)/atr
                o=_safe_float(x1.iloc[start].get("open")); h=_safe_float(x1.iloc[start].get("high")); l=_safe_float(x1.iloc[start].get("low")); c=_safe_float(x1.iloc[start].get("close")); v=_safe_float(x1.iloc[start].get("volume"))
                if o is not None and h is not None and l is not None and c is not None and h>l: features["departure_body_to_range"] = abs(c-o)/(h-l)
                if v is not None and start >= 20:
                    pv=pd.to_numeric(x1.iloc[start-20:start]["volume"],errors="coerce").dropna()
                    if len(pv)==20 and float(pv.mean())>0: features["departure_volume_ratio_20"] = v/float(pv.mean())
        except Exception: pass
    return sanitize(features)


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def build_observation_from_touch_event(
    *,
    event: dict[str, Any],
    symbol: str,
    zone: dict[str, Any] | None,
    direction: str,
    df_1h: pd.DataFrame | None,
    df_5m: pd.DataFrame | None = None,
    zone_visit_id: str | None = None,
    scan_id: str,
    decision_ts: str | None = None,
    strategy_version: str,
    code_commit_sha: str | None,
    provider: str | None = None,
    source: str | None = None,
    market_context: dict[str, Any] | None = None,
    account_context: dict[str, Any] | None = None,
    btc_df_5m: pd.DataFrame | None = None,
    btc_df_1h: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    zone = dict(zone or {})
    bar = dict(event.get("bar") or {})
    ts = event.get("timestamp")
    if not ts:
        return None
    zone_id_value = str(zone.get("zone_id") or event.get("zone_key") or "")
    if not zone_id_value:
        zone_id_value = stable_id(symbol, direction, zone.get("origin_ts_ms"), zone.get("top"), zone.get("btm"), prefix="ZONE_")
    event_type_raw = str(event.get("reason", "UNKNOWN"))
    mapping = {
        "signal_created": "SIGNAL_CREATED",
        "ambiguous_overlap": "TOUCH_BLOCKED",
        "previous_5m_midpoint_touch": "TOUCH_BLOCKED",
        "previous_5m_zone_touch": "TOUCH_BLOCKED",
        "stale_midpoint_touch_ignored": "TOUCH_STALE",
        "stale_zone_touch_ignored": "TOUCH_STALE",
        "directional_candle_required": "TOUCH_REJECTED",
        "zone_not_active": "ACTIVATION_BLOCK",
    }
    event_type = mapping.get(event_type_raw, "TOUCH_REJECTED")
    structure_room = None
    structural_distance = None
    try:
        raw_distance = event.get("structural_distance")
        if raw_distance is not None:
            structural_distance = float(raw_distance)
        else:
            raw_room = event.get("structure_room_R")
            structure_room = float(raw_room) if raw_room is not None else None
        if structural_distance is not None and math.isfinite(structural_distance):
            reference_for_r = event.get("entry_ref") or bar.get("close")
            reference_for_r = float(reference_for_r)
            stop_pct = float(os.environ.get("FIXED_STOP_PCT", "10.0"))
            nominal_risk = reference_for_r * (stop_pct / 100.0)
            if reference_for_r > 0 and nominal_risk > 0 and math.isfinite(nominal_risk):
                structure_room = structural_distance / nominal_risk
        if structural_distance is not None:
            event["structural_distance_price"] = structural_distance
    except Exception:
        structure_room = None
    features = build_research_features(
        symbol=symbol,
        direction=direction,
        zone=zone,
        bar=bar,
        df_1h=df_1h,
        df_5m=df_5m,
        touch_count_before_trigger=event.get("touch_count_before_trigger"),
        decision_ts=decision_ts,
        structure_room_r=structure_room,
        market_context=market_context, account_context=account_context, btc_df_5m=btc_df_5m, btc_df_1h=btc_df_1h,
    )
    reference_price = event.get("entry_ref")
    if reference_price is None:
        reference_price = bar.get("close")
    return {
        "observation_id": observation_id(event_type, symbol, direction, zone_id_value, ts, visit_id=str(zone_visit_id or "")),
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "record_type": "ZONE_OBSERVATION",
        "event_type": event_type,
        "source_event_reason": event_type_raw,
        "scan_id": scan_id,
        "strategy_version": strategy_version,
        "code_commit_sha": code_commit_sha,
        "provider": provider,
        "source": source,
        "market_context_id": (market_context or {}).get("context_id"),
        "account_context_id": (account_context or {}).get("account_context_id"),
        "symbol": str(symbol).upper(),
        "direction": str(direction).upper(),
        "observation_ts": decision_ts or _now_iso(),
        "source_event_ts": ts,
        "reference_price": reference_price,
        "zone_id": zone_id_value,
        "zone_visit_id": zone_visit_id or event.get("zone_visit_id"),
        "zone": sanitize(zone),
        "entry_bar": sanitize(bar),
        "previous_bar": sanitize(event.get("previous_bar") or {}),
        "trigger": {
            "touch_mode": event.get("touch_mode"),
            "midpoint": event.get("midpoint"),
            "event_id": event.get("event_id"),
            "age_min": event.get("age_min"),
        },
        "features": features,
        "research": {
            "source": "shadow_research_v1",
            "production_gate_applied": False,
            "outcome_class": "HYPOTHETICAL_FORWARD_OUTCOME" if event_type != "SIGNAL_CREATED" else "SIGNAL_FORWARD_OUTCOME",
        },
        "recorded_at": _now_iso(),
    }


def _zone_lookup(demand: list[dict[str, Any]], supply: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for zone in demand:
        zid = str(zone.get("zone_id", ""))
        key = f"DEMAND:{zid}" if zid else f"DEMAND:{zone.get('origin_ts_ms')}:{zone.get('top')}:{zone.get('btm')}"
        out[key] = zone
    for zone in supply:
        zid = str(zone.get("zone_id", ""))
        key = f"SUPPLY:{zid}" if zid else f"SUPPLY:{zone.get('origin_ts_ms')}:{zone.get('top')}:{zone.get('btm')}"
        out[key] = zone
    return out


def _structure_room_for_signal(signal: dict[str, Any]) -> float | None:
    try:
        entry = float(signal.get("entry"))
        stop = float(signal.get("sl"))
        obstacle = (signal.get("target") or {}).get("obstacle_price")
        if entry <= 0 or obstacle is None:
            return None
        risk = abs(entry - stop)
        if risk <= 0:
            return None
        direction = str(signal.get("type", "")).upper()
        distance = float(obstacle) - entry if direction == "LONG" else entry - float(obstacle)
        return distance / risk
    except (TypeError, ValueError):
        return None


def _observation_from_signal(
    signal: dict[str, Any],
    *,
    scan_id: str,
    strategy_version: str,
    code_commit_sha: str | None,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame | None = None,
    provider: str | None = None,
    source: str | None = None,
    decision_ts: str | None = None,
    market_context: dict[str, Any] | None = None,
    account_context: dict[str, Any] | None = None,
    btc_df_5m: pd.DataFrame | None = None,
    btc_df_1h: pd.DataFrame | None = None,
) -> dict[str, Any]:
    zone = dict(signal.get("zone") or {})
    direction = str(signal.get("type", "")).upper()
    bar = dict(signal.get("entry_bar") or {})
    features = build_research_features(
        symbol=str(signal.get("symbol", "")),
        direction=direction,
        zone=zone,
        bar=bar,
        df_1h=df_1h,
        df_5m=df_5m,
        touch_count_before_trigger=(signal.get("zone_visit") or {}).get("touch_count_before_trigger"),
        decision_ts=decision_ts,
        structure_room_r=_structure_room_for_signal(signal),
        market_context=market_context, account_context=account_context, btc_df_5m=btc_df_5m, btc_df_1h=btc_df_1h,
    )
    features.update({
        "signal_score": signal.get("score"),
        "tp1_rr": signal.get("tp1_rr"),
        "tp2_rr": signal.get("tp2_rr"),
        "signal_drift_pct": signal.get("signal_drift_pct"),
        "market_spread_pct": (signal.get("market_snapshot") or {}).get("market_spread_pct"),
        "venue_price_deviation_pct": (signal.get("market_snapshot") or {}).get("venue_price_deviation_pct"),
        "execution_age_minutes": signal.get("execution_age_minutes"),
        "execution_age_bars": signal.get("execution_age_bars"),
    })
    ts = signal.get("trigger_bar_time") or signal.get("time")
    event_id = str(signal.get("event_id", ""))
    zid = str(zone.get("zone_id", ""))
    visit_id = str((signal.get("zone_visit") or {}).get("visit_id") or (signal.get("trigger") or {}).get("zone_visit_id") or "")
    return {
        "observation_id": observation_id("SIGNAL_CREATED", str(signal.get("symbol", "")), direction, zid, ts, visit_id=visit_id),
        "schema_version": RESEARCH_SCHEMA_VERSION,
        "record_type": "ZONE_OBSERVATION",
        "event_type": "SIGNAL_CREATED",
        "source_event_reason": "signal_created",
        "scan_id": scan_id,
        "strategy_version": strategy_version,
        "code_commit_sha": code_commit_sha,
        "provider": provider,
        "source": source,
        "market_context_id": (market_context or {}).get("context_id"),
        "account_context_id": (account_context or {}).get("account_context_id"),
        "symbol": str(signal.get("symbol", "")).upper(),
        "direction": direction,
        "observation_ts": decision_ts or _now_iso(),
        "source_event_ts": ts,
        "reference_price": signal.get("entry"),
        "event_id": event_id,
        "zone_id": zid,
        "zone_visit_id": visit_id,
        "zone": sanitize(zone),
        "entry_bar": sanitize(bar),
        "previous_bar": sanitize(signal.get("previous_bar") or {}),
        "trigger": sanitize(signal.get("trigger") or {}),
        "features": sanitize(features),
        "production_status_at_generation": "SIGNAL_CREATED",
        "research": {
            "source": "shadow_research_v1",
            "production_gate_applied": False,
            "outcome_class": "SIGNAL_FORWARD_OUTCOME",
        },
        "recorded_at": _now_iso(),
    }


def record_market_context(row: dict[str, Any]) -> bool:
    if not RESEARCH_ENABLED or not row:
        return False
    payload = sanitize(dict(row))
    payload.setdefault("schema_version", RESEARCH_SCHEMA_VERSION)
    payload.setdefault("record_type", "MARKET_CONTEXT")
    payload.setdefault("context_id", stable_id("market-context-v1", payload.get("scan_id", ""), payload.get("symbol", ""), payload.get("provider", ""), payload.get("captured_at_ms", ""), prefix="MC_"))
    try:
        written = _append_jsonl_locked(MARKET_CONTEXT_PATH, [payload])
        if written: _bump_manifest("market_context_written", written)
        return bool(written)
    except Exception as exc:
        _record_error("market_context_write", MARKET_CONTEXT_PATH, exc, symbol=str(payload.get("symbol", "")))
        _bump_manifest("persistence_errors", 1)
        return False


def record_account_context(row: dict[str, Any]) -> bool:
    if not RESEARCH_ENABLED:
        return False
    payload = sanitize(dict(row))
    payload.setdefault("schema_version", RESEARCH_SCHEMA_VERSION)
    payload.setdefault("record_type", "ACCOUNT_CONTEXT")
    payload.setdefault("recorded_at", _now_iso())
    payload.setdefault("account_context_id", stable_id("account-context-v1", payload.get("scan_id", ""), payload.get("captured_at_ms", ""), prefix="AC_"))
    try:
        written = _append_jsonl_locked(ACCOUNT_CONTEXT_PATH, [payload])
        if written:
            _bump_manifest("account_context_written", written)
        return bool(written)
    except Exception as exc:
        _record_error("account_context_write", ACCOUNT_CONTEXT_PATH, exc)
        _bump_manifest("persistence_errors", 1)
        return False


def record_scan_symbol(
    *,
    scan_id: str,
    symbol: str,
    strategy_version: str,
    code_commit_sha: str | None,
    provider: str,
    source: str,
    bars_1h: Iterable[dict[str, Any]],
    bars_5m: Iterable[dict[str, Any]],
    df_1h: pd.DataFrame,
    demand: list[dict[str, Any]],
    supply: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    symbol_state: dict[str, Any],
    signals: list[dict[str, Any]],
    decision_ts: str | None = None,
    market_context: dict[str, Any] | None = None,
    account_context: dict[str, Any] | None = None,
    btc_df_5m: pd.DataFrame | None = None,
    btc_df_1h: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Persist one symbol's shadow observations and market bars.

    This function never decides whether a production signal is valid; it only
    records what the existing decision path observed and what shadow filters
    would have said about the same bar.
    """
    if not RESEARCH_ENABLED:
        return {"observations": 0, "bars_1h": 0, "bars_5m": 0}

    decision_ts = decision_ts or _now_iso()
    bars_1h = list(bars_1h)
    bars_5m = list(bars_5m)
    counts = {"observations": 0, "bars_1h": 0, "bars_5m": 0, "market_context": 0}
    market_context_persisted = False
    if account_context and "persisted" not in account_context:
        account_payload = dict(account_context)
        account_payload.update({"scan_id": scan_id, "strategy_version": strategy_version, "code_commit_sha": code_commit_sha})
        account_payload.setdefault("account_context_id", stable_id("account-context-v1", scan_id, account_payload.get("captured_at_ms", ""), prefix="AC_"))
        if record_account_context(account_payload):
            account_payload["persisted"] = True
            account_context = account_payload
        else:
            account_context = dict(account_payload, persisted=False)
    if market_context:
        context_payload = dict(market_context)
        context_payload.update({"scan_id": scan_id, "strategy_version": strategy_version, "code_commit_sha": code_commit_sha, "symbol": symbol.upper(), "provider": provider, "source": source})
        context_payload.setdefault("context_id", stable_id("market-context-v1", scan_id, symbol, provider, context_payload.get("captured_at_ms", ""), prefix="MC_"))
        market_context_persisted = record_market_context(context_payload)
        counts["market_context"] = 1 if market_context_persisted else 0
        context_payload["persisted"] = market_context_persisted
        context_payload["decision_ts"] = decision_ts
        context_payload["context_age_ms_at_decision"] = None
        captured_at = _parse_timestamp(context_payload.get("captured_at"))
        decision_parsed = _parse_timestamp(decision_ts)
        if captured_at is not None and decision_parsed is not None:
            context_payload["context_age_ms_at_decision"] = max(0.0, (decision_parsed - captured_at).total_seconds() * 1000.0)
        market_context = context_payload
    closed_5m: list[dict[str, Any]] = []
    if bars_5m:
        try:
            x5 = pd.DataFrame(bars_5m).copy()
            if not x5.empty:
                x5["timestamp"] = x5["timestamp"].map(_parse_timestamp)
                for col in ("open", "high", "low", "close", "volume"):
                    x5[col] = pd.to_numeric(x5[col], errors="coerce")
                x5 = x5.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).sort_values("timestamp").drop_duplicates("timestamp")
                x5 = x5.loc[(x5["open"] > 0) & (x5["high"] > 0) & (x5["low"] > 0) & (x5["close"] > 0) & (x5["volume"] >= 0)]
                x5 = x5.loc[(x5["high"] >= x5[["open", "close"]].max(axis=1)) & (x5["low"] <= x5[["open", "close"]].min(axis=1)) & (x5["high"] >= x5["low"])].copy()
                now = pd.Timestamp.now(tz="UTC")
                x5 = x5.loc[x5["timestamp"] + pd.Timedelta(minutes=5) <= now]
                closed_5m = x5.to_dict("records")
        except Exception:
            closed_5m = []

    closed_5m_df = pd.DataFrame(closed_5m) if closed_5m else pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    zones = _zone_lookup(demand, supply)
    observations: list[dict[str, Any]] = []

    # Production signals receive the richest record, including target/score/visit data.
    for signal in signals:
        observations.append(_observation_from_signal(
            signal, scan_id=scan_id, strategy_version=strategy_version,
            code_commit_sha=code_commit_sha, df_1h=df_1h, df_5m=closed_5m_df, decision_ts=decision_ts,
            provider=provider, source=source, market_context=market_context, account_context=account_context, btc_df_5m=btc_df_5m, btc_df_1h=btc_df_1h,
        ))

    # Every actual touch boundary that did NOT create a signal is retained as research data.
    for event in (diagnostics.get("touch_events") or []):
        reason = str(event.get("reason", ""))
        if reason == "signal_created":
            continue
        zone_key = str(event.get("zone_key", ""))
        zone = zones.get(zone_key)
        direction = str(event.get("direction", "")).upper()
        built = build_observation_from_touch_event(
            event=event, symbol=symbol, zone=zone, direction=direction, df_1h=df_1h, df_5m=closed_5m_df,
            zone_visit_id=str(event.get("visit_id") or ""), scan_id=scan_id, decision_ts=decision_ts,
            strategy_version=strategy_version, code_commit_sha=code_commit_sha, provider=provider, source=source, market_context=market_context, account_context=account_context, btc_df_5m=btc_df_5m, btc_df_1h=btc_df_1h,
        )
        if built:
            observations.append(built)

    # Rearm events are explicit research events; they do not constitute production entries.
    for event in (diagnostics.get("rearm_events") or []):
        zone_key = str(event.get("zone_key", ""))
        zone = zones.get(zone_key) or {}
        direction = str(event.get("direction", "")).upper()
        bar = {
            "timestamp": event.get("timestamp"), "open": event.get("open", event.get("close")),
            "high": event.get("high", event.get("close")), "low": event.get("low", event.get("close")),
            "close": event.get("close"), "volume": event.get("volume"),
        }
        observations.append({
            "observation_id": observation_id("REARM", symbol, direction, str(zone.get("zone_id") or zone_key), event.get("timestamp"), visit_id=str(event.get("visit_id") or "")),
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "record_type": "ZONE_OBSERVATION",
            "event_type": "REARM",
            "source_event_reason": "zone_rearmed",
            "scan_id": scan_id, "strategy_version": strategy_version, "code_commit_sha": code_commit_sha,
            "provider": provider, "source": source, "market_context_id": (market_context or {}).get("context_id"), "account_context_id": (account_context or {}).get("account_context_id"),
            "symbol": symbol.upper(), "direction": direction, "observation_ts": decision_ts, "source_event_ts": event.get("timestamp"),
            "reference_price": event.get("close"), "zone_id": zone.get("zone_id") or zone_key,
            "zone_visit_id": event.get("visit_id"), "zone": sanitize(zone), "entry_bar": sanitize(bar),
            "features": build_research_features(symbol=symbol, direction=direction, zone=zone, bar=bar, df_1h=df_1h, df_5m=closed_5m_df, decision_ts=decision_ts, market_context=market_context, btc_df_5m=btc_df_5m, btc_df_1h=btc_df_1h),
            "research": {"source": "shadow_research_v1", "production_gate_applied": False, "outcome_class": "HYPOTHETICAL_FORWARD_OUTCOME"},
            "recorded_at": _now_iso(),
        })

    # One nearest-approach observation per active zone per scan, based only on the newly processed 5m bars.
    for zone_key, zdiag in (diagnostics.get("zones") or {}).items():
        bar = zdiag.get("closest_midpoint_bar")
        if not isinstance(bar, dict):
            continue
        distance = bar.get("distance_pct")
        if distance is None or float(distance) <= 0 or float(distance) > NEAREST_APPROACH_MAX_PCT:
            continue
        zone = zones.get(str(zone_key)) or {}
        direction = str(zdiag.get("direction", "")).upper()
        ts = bar.get("timestamp")
        zid = str(zone.get("zone_id") or zone_key)
        observations.append({
            "observation_id": observation_id("NEAREST_APPROACH", symbol, direction, zid, ts),
            "schema_version": RESEARCH_SCHEMA_VERSION,
            "record_type": "ZONE_OBSERVATION",
            "event_type": "NEAREST_APPROACH",
            "source_event_reason": "closest_midpoint_without_touch",
            "scan_id": scan_id, "strategy_version": strategy_version, "code_commit_sha": code_commit_sha,
            "provider": provider, "source": source, "market_context_id": (market_context or {}).get("context_id"), "account_context_id": (account_context or {}).get("account_context_id"),
            "symbol": symbol.upper(), "direction": direction, "observation_ts": decision_ts, "source_event_ts": ts,
            "reference_price": bar.get("close"), "zone_id": zid,
            "zone_visit_id": (symbol_state.get("zones", {}).get(zone_key) or {}).get("visit_id"),
            "zone": sanitize(zone), "entry_bar": sanitize(bar),
            "features": {**build_research_features(symbol=symbol, direction=direction, zone=zone, bar=bar, df_1h=df_1h, df_5m=closed_5m_df, decision_ts=decision_ts, market_context=market_context, btc_df_5m=btc_df_5m, btc_df_1h=btc_df_1h), "distance_to_midpoint_pct": distance},
            "research": {"source": "shadow_research_v1", "production_gate_applied": False, "outcome_class": "HYPOTHETICAL_FORWARD_OUTCOME"},
            "recorded_at": _now_iso(),
        })

    # Persist raw market bars only when this symbol produced at least one research observation.
    # This keeps the low-level API safe even when called directly, and prevents irrelevant
    # 5m/1h universe history from growing the research journals.
    if observations:
        counts["bars_1h"] = persist_market_bars(
            symbol, "1h", bars_1h, provider=provider, source=source, scan_id=scan_id, code_commit_sha=code_commit_sha
        )
        if closed_5m:
            counts["bars_5m"] = persist_market_bars(
                symbol, "5m", closed_5m, provider=provider, source=source, scan_id=scan_id, code_commit_sha=code_commit_sha
            )

    account_context_persisted = bool(account_context and account_context.get("account_context_id")) and bool(account_context.get("persisted", True))
    account_context_age_ms_at_decision = None
    if account_context:
        captured_at = _parse_timestamp(account_context.get("captured_at"))
        decision_parsed = _parse_timestamp(decision_ts)
        if captured_at is not None and decision_parsed is not None:
            account_context_age_ms_at_decision = max(0.0, (decision_parsed - captured_at).total_seconds() * 1000.0)
    for observation in observations:
        observation["market_context_persisted"] = market_context_persisted
        observation["market_context_status"] = ("PERSISTED" if market_context_persisted else ("PERSISTENCE_FAILED" if market_context else "NOT_COLLECTED"))
        observation["account_context_persisted"] = account_context_persisted
        observation["account_context_status"] = ("PERSISTED" if account_context_persisted else ("PERSISTENCE_FAILED" if account_context else "NOT_COLLECTED"))
        if market_context:
            observation["market_context_age_ms_at_decision"] = market_context.get("context_age_ms_at_decision")
        observation["account_context_age_ms_at_decision"] = account_context_age_ms_at_decision

    try:
        expected = len({str(r.get("observation_id", "")) for r in observations if r.get("observation_id")})
        counts["observations"] = record_zone_observations(observations)
        if counts["observations"] != expected:
            exc = RuntimeError(f"research observation persistence mismatch: expected={expected} written={counts['observations']}")
            _record_error("scan_symbol_observation_batch", ZONE_OBSERVATIONS_PATH, exc, symbol=symbol)
            _bump_manifest("persistence_errors", 1)
    except Exception as exc:
        _record_error("scan_symbol_observation_batch", ZONE_OBSERVATIONS_PATH, exc, symbol=symbol)
    return counts
