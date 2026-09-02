from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("event_engine.binance")

BASE_URL = os.environ.get("BINANCE_MARKET_BASE_URL", "https://fapi.binance.com").rstrip("/")
EXCHANGE_INFO_PATH = "/fapi/v1/exchangeInfo"
KLINES_PATH = "/fapi/v1/klines"
TICKER_24H_PATH = "/fapi/v1/ticker/24hr"

_CACHE: dict[str, Any] = {"ts": 0.0, "symbols": {}, "raw": []}
_CACHE_TTL = max(60, int(os.environ.get("BINANCE_EXCHANGE_INFO_TTL_SEC", "1800")))
_LOCAL = threading.local()


def _session() -> requests.Session:
    session = getattr(_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=2,
            pool_maxsize=2,
            max_retries=Retry(
                total=2,
                connect=2,
                read=2,
                status=2,
                backoff_factor=0.25,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
                respect_retry_after_header=True,
                raise_on_status=False,
            ),
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _LOCAL.session = session
    return session


def _get(path: str, params: dict[str, Any] | None = None, timeout: float = 8.0) -> Any:
    response = _session().get(BASE_URL + path, params=params or {}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("code") not in (None, 0):
        raise RuntimeError(f"Binance API error: code={payload.get('code')} msg={payload.get('msg')}")
    return payload


def refresh_exchange_info() -> dict[str, dict[str, Any]]:
    payload = _get(EXCHANGE_INFO_PATH, timeout=float(os.environ.get("BINANCE_HTTP_TIMEOUT_SEC", "10")))
    raw = payload.get("symbols", []) if isinstance(payload, dict) else []
    symbols: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        if str(item.get("status", "")).upper() not in {"TRADING", "BREAK"}:
            continue
        if str(item.get("quoteAsset", "")).upper() != "USDT":
            continue
        if str(item.get("contractType", "")).upper() != "PERPETUAL":
            continue
        symbols[symbol] = item
    _CACHE.update(ts=time.time(), symbols=symbols, raw=raw)
    log.info("[BINANCE] Active USDT perpetual symbols=%d", len(symbols))
    return symbols


def symbols() -> dict[str, dict[str, Any]]:
    if _CACHE["symbols"] and time.time() - float(_CACHE["ts"]) < _CACHE_TTL:
        return _CACHE["symbols"]
    return refresh_exchange_info()


def normalize_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if value.endswith("-USDT"):
        return value[:-5] + "USDT"
    if value.endswith("USDT"):
        return value
    return value.replace("-", "")


def classify_asset(info: dict[str, Any]) -> str:
    underlying = str(info.get("underlyingType", "")).upper()
    subtypes = info.get("underlyingSubType") or []
    joined = " ".join(str(x).upper() for x in subtypes) if isinstance(subtypes, list) else str(subtypes).upper()
    combined = f"{underlying} {joined}"
    if "EQUITY" in combined or "STOCK" in combined or "TRADFI" in combined:
        return "EQUITY"
    if not underlying or underlying in {"COIN", "CRYPTO", "CRYPTOCURRENCY", "TOKEN"}:
        return "CRYPTO"
    return "UNKNOWN"


def get_symbol_info(symbol: str) -> dict[str, Any] | None:
    return symbols().get(normalize_symbol(symbol))


def analysis_symbols_for_bingx(bingx_symbols: list[str]) -> list[dict[str, Any]]:
    available = symbols()
    overrides: dict[str, str] = {}
    try:
        overrides = {str(k).upper(): str(v).upper() for k, v in json.loads(os.environ.get("BINANCE_SYMBOL_MAP", "{}")).items()}
    except Exception:
        overrides = {}

    result: list[dict[str, Any]] = []
    for bx in bingx_symbols:
        bx_upper = str(bx).upper()
        b_symbol = overrides.get(bx_upper, normalize_symbol(bx_upper))
        info = available.get(b_symbol)
        result.append(
            {
                "bingx_symbol": bx_upper,
                "binance_symbol": b_symbol,
                "binance_available": bool(info),
                "asset_class": classify_asset(info) if info else None,
                "binance_info": info or {},
            }
        )
    return result


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 120, *, retryable: bool = True) -> list[dict[str, Any]]:
    bsym = normalize_symbol(symbol)
    payload = _get(
        KLINES_PATH,
        {"symbol": bsym, "interval": interval, "limit": int(limit)},
        timeout=float(os.environ.get("BINANCE_HTTP_TIMEOUT_SEC", "10")),
    )
    if not isinstance(payload, list):
        raise ValueError("Binance klines payload is not a list")
    rows: list[dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 6:
            continue
        rows.append(
            {
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time": int(row[6]) if len(row) > 6 else None,
                "quote_volume": float(row[7]) if len(row) > 7 else None,
                "trade_count": int(row[8]) if len(row) > 8 and str(row[8]).isdigit() else None,
                "source": "binance_futures",
                "binance_symbol": bsym,
            }
        )
    return rows


def fetch_24h_ticker(symbol: str) -> dict[str, Any] | None:
    bsym = normalize_symbol(symbol)
    payload = _get(TICKER_24H_PATH, {"symbol": bsym}, timeout=8.0)
    return payload if isinstance(payload, dict) else None
