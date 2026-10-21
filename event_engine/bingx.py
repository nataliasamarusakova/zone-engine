# bingx.py

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
import threading
import uuid
from contextlib import contextmanager
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from decimal import (
    Decimal,
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
)
from typing import Any, Callable
from urllib.parse import quote, urlencode

import requests
from requests.adapters import HTTPAdapter

from event_engine import telemetry
from event_engine.request_metrics import timed_request
from urllib3.util.retry import Retry

log = logging.getLogger("event_engine.bingx")

BASE_URL = os.environ.get("BINGX_BASE_URL", "https://open-api-vst.bingx.com").rstrip("/")
MARGIN_USDT = float(os.environ.get("BINGX_MARGIN_USDT", "1"))
LEVERAGE = int(os.environ.get("BINGX_LEVERAGE", "10"))
MAX_LEVERAGE = int(os.environ.get("BINGX_MAX_LEVERAGE", "10"))
MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT = max(0.0, float(os.environ.get("MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT", "1.00")))

SYMBOL_MAP = {}
try:
    SYMBOL_MAP = json.loads(os.environ.get("BINGX_SYMBOL_MAP", "{}"))
except Exception:
    SYMBOL_MAP = {}

CONTRACTS_PATH = "/openApi/swap/v2/quote/contracts"
KLINE_PATH = "/openApi/swap/v3/quote/klines"
ORDER_PATH = "/openApi/swap/v2/trade/order"
POSITION_PATH = os.environ.get("BINGX_POSITIONS_PATH", "/openApi/swap/v2/user/positions")
LEVERAGE_PATH = "/openApi/swap/v2/trade/leverage"
POSITION_MODE_PATH = "/openApi/swap/v1/positionSide/dual"
OPEN_ORDERS_PATH = "/openApi/swap/v2/trade/openOrders"
TWAP_OPEN_ORDERS_PATH = "/openApi/swap/v1/twap/openOrders"
BOOK_TICKER_PATH = "/openApi/swap/v2/quote/bookTicker"
TICKER_PATH = "/openApi/swap/v2/quote/ticker"
DEPTH_PATH = "/openApi/swap/v2/quote/depth"
TRADES_PATH = "/openApi/swap/v2/quote/trades"
PREMIUM_INDEX_PATH = "/openApi/swap/v2/quote/premiumIndex"
OPEN_INTEREST_PATH = "/openApi/swap/v2/quote/openInterest"
FUNDING_RATE_PATH = "/openApi/swap/v2/quote/fundingRate"
BALANCE_PATH = "/openApi/swap/v3/user/balance"
COMMISSION_RATE_PATH = "/openApi/swap/v2/user/commissionRate"
INCOME_PATH = "/openApi/swap/v2/user/income"
FORCE_ORDERS_PATH = "/openApi/swap/v2/trade/forceOrders"
ALL_FILL_ORDERS_PATH = "/openApi/swap/v2/trade/allFillOrders"
ALL_ORDERS_MAX_LIMIT = 1000
ALL_ORDERS_MAX_WINDOW_MS = 7 * 24 * 60 * 60 * 1000 - 1
_ALL_ORDERS_MIN_REQUEST_INTERVAL_SEC = 0.21  # official endpoint limit: 5 requests/s
_ALL_ORDERS_REQUEST_LOCK = threading.Lock()
_ALL_ORDERS_LAST_REQUEST_AT = 0.0

# BingX has returned allFillOrders payloads both as a list and as an object
# containing a fill row/row list. Keep normalization conservative: an explicit
# empty list means no fills; an unknown object must never be silently treated as
# an empty execution history.
_FILL_ROWS_WRAPPER_KEYS = (
    "rows", "list", "items", "data", "result", "records", "fills",
    "fillOrders", "fill_orders", "fillOrderList", "fill_order_list",
    "orders", "orderList", "order_list", "tradeList", "trade_list",
    "listData", "dataList",
)
_FILL_ROW_ID_KEYS = ("tradeId", "tradeID", "trade_id", "orderId", "orderID", "order_id")
_FILL_ROW_DETAIL_KEYS = (
    "symbol", "side", "price", "tradePrice", "trade_price", "qty", "quantity",
    "volume", "executedQty", "realizedPnl", "realisedPnl", "realized_pnl",
    "fee", "commission", "time", "tradeTime", "trade_time",
)

def _normalize_fill_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize the common swap/contract fill field aliases into one schema."""
    item: dict[str, Any] = {}
    for sources, dst in (
        (("tradeId", "tradeID", "trade_id"), "trade_id"),
        (("orderId", "orderID", "order_id"), "order_id"),
        (("symbol",), "symbol"),
        (("side",), "side"),
        (("positionSide", "position_side"), "position_side"),
    ):
        for src in sources:
            value = row.get(src)
            if value not in (None, ""):
                item[dst] = str(value)
                break

    for sources, dst in (
        (("price", "tradePrice", "trade_price"), "price"),
        (("qty", "quantity", "volume", "executedQty", "executed_qty"), "qty"),
        (("realizedPnl", "realisedPnl", "realized_pnl", "realised_pnl", "profit"), "realized_pnl"),
        (("fee", "commission", "commissionFee", "commission_fee"), "fee"),
        (("time", "tradeTime", "trade_time", "timeMs", "timestamp"), "time_ms"),
    ):
        for src in sources:
            value = row.get(src)
            if value in (None, ""):
                continue
            try:
                number = float(value)
                if not math.isfinite(number):
                    item[dst] = None
                else:
                    item[dst] = int(number) if dst == "time_ms" else number
            except (TypeError, ValueError):
                item[dst] = None
            break
    return item

def _looks_like_fill_row(payload: dict[str, Any]) -> bool:
    """Recognize a single fill record without confusing arbitrary wrappers for rows."""
    has_identifier = any(payload.get(key) not in (None, "") for key in _FILL_ROW_ID_KEYS)
    detail_count = sum(payload.get(key) not in (None, "") for key in _FILL_ROW_DETAIL_KEYS)
    has_price_and_quantity = (
        any(payload.get(key) not in (None, "") for key in ("price", "tradePrice", "trade_price"))
        and any(payload.get(key) not in (None, "") for key in ("qty", "quantity", "volume", "executedQty"))
    )
    return has_identifier and (detail_count >= 2 or has_price_and_quantity)

def _normalize_all_fill_orders_data(raw_data: Any, *, _depth: int = 0) -> list[dict[str, Any]] | None:
    """Return fill rows, [] only for an explicit empty list, or None if unknown.

    Only recurse through documented/common response wrapper names. An arbitrary
    nested list is not enough evidence that it contains fills.
    """
    if isinstance(raw_data, list):
        return list(raw_data) if all(isinstance(row, dict) for row in raw_data) else None
    if not isinstance(raw_data, dict):
        return None

    for key in _FILL_ROWS_WRAPPER_KEYS:
        candidate = raw_data.get(key)
        if isinstance(candidate, list):
            return list(candidate) if all(isinstance(row, dict) for row in candidate) else None

    if _depth < 2:
        for key in _FILL_ROWS_WRAPPER_KEYS:
            candidate = raw_data.get(key)
            if isinstance(candidate, dict):
                normalized = _normalize_all_fill_orders_data(candidate, _depth=_depth + 1)
                if normalized is not None:
                    return normalized

    if _looks_like_fill_row(raw_data):
        return [raw_data]
    return None

def _fill_payload_shape(raw_data: Any) -> str:
    if isinstance(raw_data, dict):
        keys = sorted(str(key) for key in raw_data.keys())
        shown = keys[:24]
        suffix = ",..." if len(keys) > len(shown) else ""
        return f"dict keys={shown}{suffix}"
    return type(raw_data).__name__

CACHE = {
    "ts": 0.0,
    "data": {},
    "by_display_name": {},
}
TTL = 3600
SERVER_TIME_OFFSET_MS = 0
_POSITION_MODE_CACHE: dict[str, Any] = {"ts": 0.0, "dual": None}
_BOOK_TICKER_LOCK = threading.Lock()
_LAST_BOOK_TICKER_TS = 0.0
_RESEARCH_RATE_LOCK_GUARD = threading.Lock()
_RESEARCH_RATE_LOCKS: dict[str, threading.Lock] = {}
_RESEARCH_RATE_LAST_TS: dict[str, float] = {}

# Requests Session is not used concurrently across scan worker threads.
# Public scan requests get one Session per worker thread, each with a small bounded
# connection pool. This avoids urllib3 "Connection pool is full" churn while still
# keeping the public scan concurrent. Private/retryable requests use a dedicated
# process-wide Session because those calls are serialized by the orchestration layer.
SESSION = requests.Session()
_SESSION_ADAPTER = HTTPAdapter(
    pool_connections=8,
    pool_maxsize=8,
    max_retries=Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    ),
)
SESSION.mount("https://", _SESSION_ADAPTER)
SESSION.mount("http://", _SESSION_ADAPTER)

_FAST_LOCAL = threading.local()

def _get_fast_session() -> requests.Session:
    session = getattr(_FAST_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _FAST_LOCAL.session = session
    return session


def _get_retry_session() -> requests.Session:
    """Return a thread-local session with the same GET retry policy as SESSION."""
    session = getattr(_FAST_LOCAL, "retry_session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=8,
            pool_maxsize=8,
            max_retries=Retry(
                total=3,
                connect=3,
                read=3,
                status=3,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
                respect_retry_after_header=True,
                raise_on_status=False,
            ),
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        _FAST_LOCAL.retry_session = session
    return session


@contextmanager
def thread_local_retry_session():
    """Use an isolated retry-enabled HTTP session for read-only worker calls."""
    previous = getattr(_FAST_LOCAL, "use_retry_session", False)
    _FAST_LOCAL.use_retry_session = True
    try:
        yield
    finally:
        if previous:
            _FAST_LOCAL.use_retry_session = previous
        else:
            try:
                delattr(_FAST_LOCAL, "use_retry_session")
            except AttributeError:
                pass


def get_credentials() -> tuple[str, str]:
    """Read private BingX credentials at call time, not only at module import."""
    return os.environ.get("BINGX_API_KEY", "").strip(), os.environ.get("BINGX_SECRET_KEY", "").strip()


def credentials_available() -> bool:
    key, secret = get_credentials()
    return bool(key and secret)

def _canonical_params(params: dict[str, Any]) -> str:
    """BingX canonical signing string: ASCII-sort keys; values are not URL encoded."""
    return "&".join(f"{key}={params[key]}" for key in sorted(params))


def _validate_signed_values(params: dict[str, Any]) -> None:
    """Reject query-string metacharacters that could alter signed semantics."""
    forbidden = re.compile(r"[&=?#\r\n]")
    for key, value in params.items():
        text = str(value)
        if forbidden.search(text):
            raise ValueError(f"parameter {key!r} contains forbidden query character")


def _signed_query(params: dict[str, Any]) -> str:
    """Build BingX's actual query string from the unencoded canonical params."""
    canonical = _canonical_params(params)
    needs_encoding = "[" in canonical or "{" in canonical
    parts = []
    for key in sorted(params):
        value = str(params[key])
        if needs_encoding:
            value = quote(value, safe="-_.~")
        parts.append(f"{key}={value}")
    return "&".join(parts)


def _sign(params: dict[str, Any]) -> str:
    _, secret_key = get_credentials()
    canonical = _canonical_params(params)
    return hmac.new(secret_key.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def _apply_request_timestamp(params: dict[str, Any]) -> str:
    params.pop("signature", None)
    params["timestamp"] = int(time.time() * 1000) + SERVER_TIME_OFFSET_MS
    signature = _sign(params)
    params["signature"] = signature
    return signature


def _update_server_time_offset(response: Any) -> bool:
    """Synchronize the signing timestamp from the HTTP Date header.

    BingX can reject signed requests when local clock drift exceeds the allowed
    window. The request wrapper retries once after refreshing this offset.
    """
    global SERVER_TIME_OFFSET_MS
    try:
        date_header = response.headers.get("Date") if response is not None else None
        if not date_header:
            return False
        dt = parsedate_to_datetime(date_header)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        server_ms = int(dt.timestamp() * 1000)
        local_ms = int(time.time() * 1000)
        SERVER_TIME_OFFSET_MS = server_ms - local_ms
        log.warning("[BINGX] Server time offset synchronized: %d ms", SERVER_TIME_OFFSET_MS)
        return True
    except Exception as exc:
        log.warning("[BINGX] Failed to synchronize server time: %s", exc)
        return False

def _base_urls() -> list[str]:
    primary = BASE_URL
    if primary.endswith(".com"):
        fallback = primary[:-4] + ".pro"
        return [primary, fallback] if fallback != primary else [primary]
    return [primary]


def _is_network_error(exc: Exception) -> bool:
    return isinstance(exc, (requests.Timeout, requests.ConnectionError))


def _request(
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    signed: bool = True,
    *,
    timeout_sec: float | None = None,
    retryable: bool = True,
):
    base_params = dict(params or {})
    try:
        request_timeout = float(timeout_sec) if timeout_sec is not None else float(os.environ.get("BINGX_HTTP_TIMEOUT_SEC", "10"))
    except (TypeError, ValueError):
        request_timeout = 10.0
    request_timeout = max(1.0, min(request_timeout, 60.0))
    if not retryable:
        session = _get_fast_session()
    elif getattr(_FAST_LOCAL, "use_retry_session", False):
        # Concurrent tracker reads must not share requests.Session with the main
        # orchestration thread. Preserve the normal retry and fallback behavior.
        session = _get_retry_session()
    else:
        session = SESSION

    try:
        source_key_required = True
        api_key, secret_key = get_credentials()
        if signed and (not api_key or not secret_key):
            return {"code": -1, "msg": "missing BingX credentials"}

        last_error: Exception | None = None
        for base_url in _base_urls():
            for attempt in range(2 if signed else 1):
                request_params = dict(base_params)
                headers = {"X-SOURCE-KEY": "BX-AI-SKILL"} if source_key_required else {}
                if signed:
                    _apply_request_timestamp(request_params)
                    headers["X-BX-APIKEY"] = api_key

                signature = str(request_params.get("signature", "")) if signed else ""
                wire_params = dict(request_params)

                try:
                    if method.upper() == "POST":
                        # BingX expects the exact canonical parameter string in the
                        # application/x-www-form-urlencoded POST body. Sending a dict
                        # lets requests rebuild/encode the body independently, which
                        # can produce a different byte representation from the string
                        # that was signed. Send the signed canonical string verbatim.
                        if signed:
                            # Validate the exact same signed-value constraints for POST
                            # bodies as for GET query strings before canonicalization/signing.
                            signed_values = {k: v for k, v in request_params.items() if k != "signature"}
                            _validate_signed_values(signed_values)
                            canonical = _canonical_params(signed_values)
                            body = f"{canonical}&signature={signature}"
                            headers["Content-Type"] = "application/x-www-form-urlencoded"
                        else:
                            body = urlencode(wire_params, doseq=False)
                            headers["Content-Type"] = "application/x-www-form-urlencoded"
                        response = timed_request(
                            "bingx", path, method, session.request,
                            method=method, url=base_url + path, data=body,
                            headers=headers, timeout=request_timeout,
                        )
                    else:
                        if signed:
                            _validate_signed_values({k: v for k, v in request_params.items() if k != "signature"})
                            query = _signed_query({k: v for k, v in request_params.items() if k != "signature"})
                            query = f"{query}&signature={signature}"
                            url = f"{base_url + path}?{query}"
                            request_kwargs = {"method": method, "url": url, "headers": headers, "timeout": request_timeout}
                        else:
                            request_kwargs = {"method": method, "url": base_url + path, "params": wire_params, "headers": headers, "timeout": request_timeout}
                        response = timed_request(
                            "bingx", path, method, session.request, **request_kwargs
                        )
                    payload = response.json()
                except Exception as exc:
                    last_error = exc
                    # NEVER resend a non-idempotent POST after a transport error.
                    # The exchange may have accepted the order while the response
                    # was lost; sending the same MARKET/STOP/TP POST to another
                    # endpoint can create a duplicate order. GET/DELETE remain
                    # eligible for the endpoint fallback.
                    if retryable and _is_network_error(exc) and method.upper() != "POST" and base_url != _base_urls()[-1]:
                        log.warning("[BINGX] Network failure on %s; trying fallback domain: %s", base_url, _safe_exception_message(exc))
                        break
                    return {"code": -1, "msg": _safe_exception_message(exc)}

                try:
                    code = int(payload.get("code"))
                except (TypeError, ValueError, AttributeError):
                    code = None

                # Keep one timestamp retry for signed requests after syncing from
                # the server's Date header.
                if signed and code == 109400 and attempt == 0:
                    if _update_server_time_offset(response):
                        continue

                return payload

        return {"code": -1, "msg": str(last_error) if last_error else "request failed"}
    except Exception as exc:
        log.exception("[BINGX] Request wrapper failure: %s %s", method, path)
        return {"code": -1, "msg": _safe_exception_message(exc)}

def refresh_contracts() -> dict[str, Any]:
    resp = _request("GET", CONTRACTS_PATH, signed=False)
    if resp.get("code") != 0:
        raise RuntimeError(f"[BINGX] Contracts error: {resp.get('msg')}")

    data = {}
    by_name = {}

    for c in resp.get("data", []) or []:
        sym = str(c.get("symbol", "")).strip().upper()
        name = str(c.get("displayName", "")).strip().upper()
        if sym:
            data[sym] = c
        if name:
            by_name[name] = c

    CACHE.update(ts=time.time(), data=data, by_display_name=by_name)
    log.debug("[BINGX] Active contracts=%d", len(data))
    return data


def contracts() -> dict[str, dict]:
    if CACHE["data"] and time.time() - CACHE["ts"] < TTL:
        return CACHE["data"]
    try:
        return refresh_contracts()
    except Exception as exc:
        log.error("[BINGX] contracts fetch failed: %s", exc)
        return CACHE["data"]


def get_contract(symbol: str) -> dict | None:
    s = (symbol or "").strip().upper()
    if not s:
        return None

    mapped = SYMBOL_MAP.get(s)
    if mapped:
        c = contracts().get(str(mapped).strip().upper())
        if c:
            return c

    direct = s if s.endswith("-USDT") else f"{s.replace('-', '')}-USDT"
    c = contracts().get(direct)
    if c:
        return c

    base = s.replace("-USDT", "").replace("-", "")
    for c in CACHE["data"].values():
        cs = str(c.get("symbol", "")).upper()
        if cs == f"{base}-USDT" or cs == base:
            return c

    norm_base = base.replace("-", "").replace("/", "").replace(" ", "")
    for c in CACHE["data"].values():
        name = str(c.get("displayName", "")).upper().replace("-", "").replace("/", "").replace(" ", "")
        if name == f"{norm_base}USDT" or name == norm_base:
            return c

    return CACHE["by_display_name"].get(f"{base}-USDT")


def to_bx_symbol(symbol: str) -> str | None:
    c = get_contract(symbol)
    if not c:
        return None
    return str(c.get("symbol", "")).upper()


def contract_exists(symbol: str) -> bool:
    c = get_contract(symbol)
    return bool(c and c.get("status") == 1 and str(c.get("apiStateOpen", "")).lower() == "true")


def fetch_klines(
    symbol: str,
    interval: str,
    limit: int = 250,
    *,
    timeout_sec: float | None = None,
    retryable: bool = True,
) -> list[dict]:
    bx = to_bx_symbol(symbol)
    if not bx:
        raise ValueError(f"[BINGX] No contract found for {symbol}")

    resp = _request(
        "GET",
        KLINE_PATH,
        {"symbol": bx, "interval": interval, "limit": limit},
        signed=False,
        timeout_sec=timeout_sec,
        retryable=retryable,
    )
    code = resp.get("code")
    if code not in (0, "0"):
        raise RuntimeError(f"[BINGX] Klines error {bx}/{interval}: code={code} msg={resp.get('msg')}")

    rows = resp.get("data") or []
    out: list[dict] = []
    now_ms = int(time.time() * 1000) + SERVER_TIME_OFFSET_MS

    duration_ms = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
        "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000,
    }.get(interval)

    for row in rows:
        if isinstance(row, (list, tuple)):
            if len(row) < 6:
                continue
            try:
                open_time = int(row[0])
                open_price = float(row[1])
                high = float(row[2])
                low = float(row[3])
                close = float(row[4])
                volume = float(row[5])
                close_time = int(row[6]) if (len(row) >= 7 and row[6] is not None) else (open_time + duration_ms if duration_ms else open_time)
                quote_volume = float(row[7]) if (len(row) >= 8 and row[7] is not None) else None
                taker_buy_base = float(row[9]) if (len(row) >= 10 and row[9] is not None) else None
                taker_buy_quote = float(row[10]) if (len(row) >= 11 and row[10] is not None) else None
            except (TypeError, ValueError, IndexError):
                continue
        elif isinstance(row, dict):
            def pick(*names):
                for name in names:
                    if name in row and row[name] is not None:
                        return row[name]
                return None
            try:
                open_time = int(pick("openTime", "open_time", "time"))
                open_price = float(pick("open"))
                high = float(pick("high"))
                low = float(pick("low"))
                close = float(pick("close"))
                volume = float(pick("volume"))
                raw_close_time = pick("closeTime", "close_time")
                close_time = int(raw_close_time) if raw_close_time is not None else (open_time + duration_ms if duration_ms else open_time)
                quote_volume_raw = pick("quoteAssetVolume", "quoteVolume", "quote_volume")
                taker_base_raw = pick("takerBuyBaseVolume", "taker_buy_base", "takerBuyBase", "buyVolume")
                taker_quote_raw = pick("takerBuyQuoteVolume", "taker_buy_quote", "takerBuyQuote", "buyQuoteVolume")
                quote_volume = float(quote_volume_raw) if quote_volume_raw is not None else None
                taker_buy_base = float(taker_base_raw) if taker_base_raw is not None else None
                taker_buy_quote = float(taker_quote_raw) if taker_quote_raw is not None else None
            except (TypeError, ValueError, KeyError):
                continue
        else:
            continue

        if close_time > now_ms:
            continue
        if open_price <= 0 or high <= 0 or low <= 0 or close <= 0 or volume < 0:
            continue
        if high < low or high < open_price or high < close or low > open_price or low > close:
            continue

        taker_flow_valid = (
            quote_volume is not None and taker_buy_base is not None and taker_buy_quote is not None
            and quote_volume >= 0 and taker_buy_base >= 0 and taker_buy_quote >= 0
            and taker_buy_base <= volume * 1.001 + 1e-8
            and taker_buy_quote <= quote_volume * 1.001 + 1e-8
        )
        bar_delta_usdt = 2.0 * taker_buy_quote - quote_volume if taker_flow_valid else None

        out.append(
            {
                "timestamp": open_time,
                "open_time": open_time,
                "close_time": close_time,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_volume": quote_volume,
                "taker_buy_base": taker_buy_base,
                "taker_buy_quote": taker_buy_quote,
                "taker_flow_valid": taker_flow_valid,
                "bar_delta_usdt": bar_delta_usdt,
            }
        )

    out.sort(key=lambda x: x["close_time"])
    deduped = []
    seen_close_times = set()

    for bar in out:
        ct = bar["close_time"]
        if ct in seen_close_times:
            continue
        seen_close_times.add(ct)
        deduped.append(bar)

    return deduped


def get_position_mode(*, force_refresh: bool = False, timeout_sec: float | None = None) -> str:
    """Return BingX position mode: HEDGE or ONE_WAY. Never guess on API failure."""
    override = os.environ.get("BINGX_POSITION_MODE", "").strip().upper()
    if override in {"HEDGE", "ONE_WAY"} and os.environ.get("BINGX_POSITION_MODE_OVERRIDE", "false").strip().lower() == "true":
        return override

    now = time.time()
    cached = _POSITION_MODE_CACHE.get("dual")
    if not force_refresh and cached in (True, False) and now - float(_POSITION_MODE_CACHE.get("ts", 0.0)) < 300:
        return "HEDGE" if cached else "ONE_WAY"

    resp = _request("GET", POSITION_MODE_PATH, {}, signed=True, timeout_sec=timeout_sec, retryable=False)
    if resp.get("code") != 0:
        raise RuntimeError(f"position mode query failed: code={resp.get('code')} msg={resp.get('msg')}")
    data = resp.get("data") or {}
    dual = data.get("dualSidePosition")
    if isinstance(dual, str):
        dual = dual.strip().lower() == "true"
    if not isinstance(dual, bool):
        raise RuntimeError(f"position mode response missing dualSidePosition: {data}")
    _POSITION_MODE_CACHE.update({"ts": now, "dual": dual})
    mode = "HEDGE" if dual else "ONE_WAY"
    log.debug("[BINGX] Position mode=%s", mode)
    return mode


def position_side_param(direction: str, *, force_refresh: bool = False) -> str:
    direction = str(direction).upper()
    if direction not in {"LONG", "SHORT"}:
        raise ValueError(f"invalid direction={direction}")
    return direction if get_position_mode(force_refresh=force_refresh) == "HEDGE" else "BOTH"


def _set_leverage(bx_symbol: str, leverage: int, direction: str = "LONG") -> bool:
    try:
        side = position_side_param(direction)
    except Exception as exc:
        log.error("[BINGX] Cannot determine position mode before leverage: %s", exc)
        return False
    resp = _request("POST", LEVERAGE_PATH, {"symbol": bx_symbol, "side": side, "leverage": str(leverage)})
    if resp.get("code") == 0:
        return True
    log.error("[BINGX] Leverage failed: side=%s code=%s msg=%s", side, resp.get("code"), resp.get("msg"))
    return False


def _response_code_is_success(resp: Any) -> bool:
    if not isinstance(resp, dict):
        return False
    try:
        return int(resp.get("code", -1)) == 0
    except (TypeError, ValueError):
        return False


def _rows_from_payload(value: Any, *, context: str) -> list[dict]:
    """Normalize an explicitly recognized row container without fail-open defaults."""
    if isinstance(value, list):
        if any(not isinstance(row, dict) for row in value):
            raise RuntimeError(f"[BINGX] {context} contains a non-object row")
        return value
    if isinstance(value, dict) and value:
        return [value]
    raise RuntimeError(f"[BINGX] {context} has an unrecognized response shape")


def _normalize_orders_list(resp: Any) -> list[dict]:
    """Normalize order rows; an unknown envelope is an error, never an empty book.

    Callers use an empty list as authoritative evidence that no orders exist. Keep
    that meaning only for known API shapes such as ``data: []`` or
    ``data: {"orders": []}``.
    """
    if not isinstance(resp, dict) or "data" not in resp:
        raise RuntimeError("[BINGX] orders response is not an object with a data field")
    data = resp.get("data")
    if isinstance(data, list):
        return _rows_from_payload(data, context="orders data")
    if isinstance(data, dict):
        for key in ("orders", "order"):
            if key in data:
                return _rows_from_payload(data[key], context=f"orders data.{key}")
        # A few order lookup responses return the order object directly.
        if "symbol" in data and any(key in data for key in ("orderId", "clientOrderId", "type")):
            return [data]
    raise RuntimeError("[BINGX] orders response returned an unrecognized response shape; refusing to assume no orders exist")


def _normalize_positions_list(resp: Any) -> list[dict]:
    """Normalize known position-response envelopes and validate every returned row.

    Unknown envelopes, rows without a symbol/amount, and non-finite amounts are
    not evidence of a flat account. Treat them as unavailable state so entry,
    emergency-close verification, and reconciliation fail closed.
    """
    if not isinstance(resp, dict) or "data" not in resp:
        raise RuntimeError("[BINGX] positions response is not an object with a data field")
    data = resp.get("data")
    rows: list[dict]
    if isinstance(data, list):
        rows = _rows_from_payload(data, context="positions data")
    elif isinstance(data, dict):
        rows = []
        recognized = False
        for key in ("positions", "position"):
            if key in data:
                rows = _rows_from_payload(data[key], context=f"positions data.{key}")
                recognized = True
                break
        if not recognized and "symbol" in data and "positionAmt" in data:
            rows = [data]
            recognized = True
        if not recognized:
            raise RuntimeError("[BINGX] get_positions returned an unrecognized response shape; refusing to assume the account is flat")
    else:
        raise RuntimeError("[BINGX] get_positions returned an unrecognized response shape; refusing to assume the account is flat")

    for index, row in enumerate(rows):
        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            raise RuntimeError(f"[BINGX] position row {index} is missing symbol; account flatness cannot be verified")
        if "positionAmt" not in row or row.get("positionAmt") in (None, ""):
            raise RuntimeError(f"[BINGX] position row {index} for {symbol} is missing positionAmt")
        try:
            amount = float(row.get("positionAmt"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"[BINGX] position row {index} for {symbol} has invalid positionAmt={row.get('positionAmt')!r}") from exc
        if not math.isfinite(amount):
            raise RuntimeError(f"[BINGX] position row {index} for {symbol} has non-finite positionAmt")
        if abs(amount) > 0:
            raw_price = row.get("avgPrice") or row.get("entryPrice")
            try:
                avg_price = float(raw_price)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"[BINGX] open position row {index} for {symbol} has invalid entry price={raw_price!r}") from exc
            if not math.isfinite(avg_price) or avg_price <= 0:
                raise RuntimeError(f"[BINGX] open position row {index} for {symbol} has no valid positive entry price")
    return rows


def get_positions(*, timeout_sec: float | None = None, retryable: bool = True) -> list[dict]:
    resp = _request(
        "GET",
        POSITION_PATH,
        {},
        signed=True,
        timeout_sec=timeout_sec,
        retryable=retryable,
    )
    if not _response_code_is_success(resp):
        code = resp.get("code") if isinstance(resp, dict) else "invalid_response"
        message = resp.get("msg") if isinstance(resp, dict) else repr(resp)
        raise RuntimeError(f"[BINGX] get_positions failed: code={code} msg={message}")
    return _normalize_positions_list(resp)


def get_all_open_orders(*, timeout_sec: float | None = None, retryable: bool = True) -> list[dict]:
    """Fetch all open swap orders for clean-start safety checks.

    The symbol parameter is intentionally omitted so the exchange can return the
    account-wide open-order set. Any API error fails closed at the caller.
    """
    resp = _request(
        "GET",
        OPEN_ORDERS_PATH,
        {},
        signed=True,
        timeout_sec=timeout_sec,
        retryable=retryable,
    )
    if resp.get("code") != 0:
        raise RuntimeError(f"[BINGX] get_all_open_orders failed: code={resp.get('code')} msg={resp.get('msg')}")
    data = resp.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("orders", "order"):
            value = data.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return [value]
        # A single order object is accepted, but an unrecognized object is not
        # interpreted as an empty account.
        if "orderId" in data or "clientOrderId" in data:
            return [data]
    raise RuntimeError("[BINGX] get_all_open_orders returned an unrecognized response shape; refusing clean start")


def get_all_open_twap_orders(*, timeout_sec: float | None = None, retryable: bool = True) -> list[dict]:
    """Fetch account-wide open TWAP strategy orders for clean-start safety checks."""
    resp = _request(
        "GET",
        TWAP_OPEN_ORDERS_PATH,
        {},
        signed=True,
        timeout_sec=timeout_sec,
        retryable=retryable,
    )
    if resp.get("code") != 0:
        raise RuntimeError(f"[BINGX] get_all_open_twap_orders failed: code={resp.get('code')} msg={resp.get('msg')}")
    data = resp.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("orders", "list"):
            value = data.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return [value]
        if "mainOrderId" in data or "orderId" in data:
            return [data]
    raise RuntimeError("[BINGX] get_all_open_twap_orders returned an unrecognized response shape; refusing clean start")


def get_order(
    symbol: str,
    order_id: str | int | None = None,
    *,
    client_order_id: str | None = None,
) -> dict:
    """Query an order by system orderId or exchange-supported clientOrderId."""
    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}
    if order_id in (None, "") and client_order_id in (None, ""):
        return {"status": "error", "error": "order_id_or_client_order_id_required"}

    params: dict[str, Any] = {"symbol": bx}
    if client_order_id not in (None, ""):
        params["clientOrderId"] = str(client_order_id)
    else:
        params["orderId"] = str(order_id)

    resp = _request("GET", ORDER_PATH, params, signed=True)
    if resp.get("code") != 0:
        return {"status": "error", "error": resp.get("msg"), "code": resp.get("code")}

    data = resp.get("data") or {}
    order = data.get("order") or data
    if not isinstance(order, dict):
        return {"status": "error", "error": "order_payload_missing"}

    def _num(key: str, fallback: str | None = None) -> float:
        raw = order.get(key)
        if raw in (None, "") and fallback:
            raw = order.get(fallback)
        try:
            value = float(raw) if raw not in (None, "") else 0.0
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    executed_qty_raw = order.get("executedQty")
    if executed_qty_raw in (None, ""):
        executed_qty_raw = order.get("cumQty")
    executed_qty_known = executed_qty_raw not in (None, "")

    return {
        "status": "ok",
        "order_id": str(order.get("orderID") or order.get("orderId") or order_id or ""),
        "order_status": str(order.get("status", "")).upper(),
        "symbol": str(order.get("symbol", bx)).upper(),
        "side": str(order.get("side", "")).upper(),
        "position_side": str(order.get("positionSide", "")).upper(),
        "order_type": str(order.get("type", "")).upper(),
        "avg_price": _num("avgPrice"),
        "trigger_price": _num("stopPrice"),
        "executed_qty": _num("executedQty", "cumQty"),
        "executed_qty_known": executed_qty_known,
        "orig_qty": _num("origQty", "quantity"),
        "client_order_id": str(order.get("clientOrderId", client_order_id or "")),
        "time_ms": int(_num("time")) if _num("time") > 0 else None,
        "update_time_ms": int(_num("updateTime")) if _num("updateTime") > 0 else None,
    }


def get_fill_orders(
    symbol: str,
    start_time_ms: int,
    end_time_ms: int,
    *,
    order_id: str | int | None = None,
    limit: int = 1000,
) -> list[dict]:
    """Return exchange-confirmed perpetual-swap fills in a time window.

    Unlike an order-status response, fill history is actual execution evidence and
    contains per-fill quantity/price/realized-PnL/fee fields.  The tracker uses this
    endpoint only as a reconciliation path, never as an entry gate.
    """
    # Unit tests and offline tooling must not accidentally perform private I/O.
    # Check credentials before contract resolution because to_bx_symbol() may itself
    # refresh the contracts cache over the network.
    if not credentials_available():
        return []
    bx = to_bx_symbol(symbol)
    if not bx:
        return []
    start_ms = max(0, int(start_time_ms))
    end_ms = max(start_ms, int(end_time_ms))
    params: dict[str, Any] = {
        "tradingUnit": "CONT",
        "startTs": start_ms,
        "endTs": end_ms,
        "currency": "USDT",
    }
    if order_id not in (None, ""):
        params["orderId"] = str(order_id)
    resp = _request(
        "GET",
        ALL_FILL_ORDERS_PATH,
        params,
        signed=True,
        timeout_sec=min(5.0, float(os.environ.get("FILL_HISTORY_TIMEOUT_SEC", "5"))),
        retryable=False,
    )
    if not isinstance(resp, dict):
        raise RuntimeError("allFillOrders returned invalid response")
    try:
        response_code = int(resp.get("code", -1))
    except (TypeError, ValueError):
        response_code = -1
    if response_code != 0:
        raise RuntimeError(f"allFillOrders failed: code={response_code} msg={resp.get('msg')}")
    raw = resp.get("data")
    rows = _normalize_all_fill_orders_data(raw)
    if rows is None:
        raise RuntimeError(
            "allFillOrders returned unrecognized data shape: "
            f"{_fill_payload_shape(raw)}"
        )

    return [_normalize_fill_row(row) for row in rows]


def cancel_order(symbol: str, order_id: str | int) -> dict:
    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}

    return _request("DELETE", ORDER_PATH, {"symbol": bx, "orderId": str(order_id)}, signed=True)



def _request_all_orders(params: dict[str, Any]) -> dict:
    """Serialize historical-order queries under BingX's per-endpoint rate limit."""
    global _ALL_ORDERS_LAST_REQUEST_AT
    with _ALL_ORDERS_REQUEST_LOCK:
        now = time.monotonic()
        wait = _ALL_ORDERS_MIN_REQUEST_INTERVAL_SEC - (now - _ALL_ORDERS_LAST_REQUEST_AT)
        if wait > 0:
            time.sleep(wait)
        _ALL_ORDERS_LAST_REQUEST_AT = time.monotonic()
        return _request("GET", "/openApi/swap/v2/trade/allOrders", params, signed=True)


def get_all_orders(
    symbol: str,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    limit: int = 1000,
) -> list[dict]:
    """Fetch complete historical-order coverage for the requested time range.

    BingX ``allOrders`` supports at most 1000 rows per request and limits a
    ``startTime``/``endTime`` query to seven days. A full page is treated as
    potentially truncated: the time range is recursively split until each leaf
    query is below the limit. If a saturated range cannot be split, raise instead
    of silently presenting incomplete history as complete.
    """
    bx = to_bx_symbol(symbol)
    if not bx:
        return []
    try:
        page_limit = min(max(int(limit), 1), ALL_ORDERS_MAX_LIMIT)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid allOrders limit: {limit!r}") from exc

    start = int(start_time_ms) if start_time_ms is not None else None
    end = int(end_time_ms) if end_time_ms is not None else None
    if start is not None and end is not None and end < start:
        raise ValueError(f"allOrders end_time_ms {end} is before start_time_ms {start}")

    max_window_ms = ALL_ORDERS_MAX_WINDOW_MS

    def fetch_window(window_start: int | None, window_end: int | None) -> list[dict]:
        params: dict[str, Any] = {"symbol": bx, "limit": page_limit}
        if window_start is not None:
            params["startTime"] = int(window_start)
        if window_end is not None:
            params["endTime"] = int(window_end)
        resp = _request_all_orders(params)
        if not isinstance(resp, dict):
            raise RuntimeError("BingX allOrders returned a non-object response")
        try:
            response_code = int(resp.get("code", -1))
        except (TypeError, ValueError):
            response_code = -1
        if response_code != 0:
            raise RuntimeError(
                f"BingX allOrders failed: code={response_code} msg={resp.get('msg')} "
                f"range={window_start}..{window_end}"
            )
        rows = _normalize_orders_list(resp)
        if len(rows) < page_limit:
            return rows
        if window_start is None or window_end is None or window_end <= window_start:
            raise RuntimeError(
                "BingX allOrders result reached the requested limit and the range "
                f"cannot be split safely (limit={page_limit}, range={window_start}..{window_end})"
            )
        midpoint = (window_start + window_end) // 2
        if midpoint < window_start or midpoint >= window_end:
            raise RuntimeError(
                "BingX allOrders result may be truncated at millisecond resolution "
                f"(limit={page_limit}, range={window_start}..{window_end})"
            )
        left = fetch_window(window_start, midpoint)
        right = fetch_window(midpoint + 1, window_end)
        return left + right

    all_rows: list[dict] = []
    if start is not None and end is not None:
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(end, chunk_start + max_window_ms)
            all_rows.extend(fetch_window(chunk_start, chunk_end))
            chunk_start = chunk_end + 1
    else:
        all_rows.extend(fetch_window(start, end))

    # Time-range recursion should not overlap, but de-duplicate defensively by
    # exchange order identity so boundary semantics or API quirks cannot duplicate
    # one historical order in reconciliation.
    def order_timestamp(item: dict, fields: tuple[str, ...]) -> int:
        for field in fields:
            try:
                value = float(item.get(field) or 0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                return int(value)
        return 0

    unique: dict[str, dict] = {}
    for row in all_rows:
        order_id = row.get("orderId", row.get("orderID", row.get("order_id")))
        client_id = row.get("clientOrderId", row.get("client_order_id"))
        if order_id not in (None, ""):
            identity = f"order:{order_id}"
        elif client_id not in (None, ""):
            identity = f"client:{str(client_id).upper()}"
        else:
            identity = "row:" + json.dumps(row, sort_keys=True, default=str, separators=(",", ":"))
        existing = unique.get(identity)
        if existing is None or order_timestamp(row, ("updateTime", "time", "createTime")) >= order_timestamp(existing, ("updateTime", "time", "createTime")):
            unique[identity] = row

    return sorted(
        unique.values(),
        key=lambda item: order_timestamp(item, ("time", "createTime", "updateTime")),
    )


def close_position_market(symbol: str, direction: str, qty: float, *, reduce_only: bool = True, trade_id: str | None = None, attempt_id: str | None = None) -> dict:
    """Close an existing directional position with a MARKET order. Used only as
    a safety rollback when mandatory protection cannot be established."""
    direction = str(direction).upper()
    if direction not in {"LONG", "SHORT"}:
        return {"status": "error", "error": f"invalid direction={direction}"}
    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}
    contract = get_contract(symbol) or {}
    try:
        prec = int(contract.get("quantityPrecision") or 0)
    except (TypeError, ValueError):
        prec = 0
    close_qty = _round_qty(abs(float(qty)), prec)
    if close_qty <= 0:
        return {"status": "error", "error": "invalid close quantity", "qty": close_qty}

    position_side = position_side_param(direction)
    params = {
        "symbol": bx,
        "side": "SELL" if direction == "LONG" else "BUY",
        "positionSide": position_side,
        "type": "MARKET",
        "quantity": _format_qty(close_qty, prec),
    }
    # BingX hedge mode rejects reduceOnly. It is only valid/needed in ONE-WAY
    # (positionSide=BOTH).
    if position_side == "BOTH" and reduce_only:
        params["reduceOnly"] = "true"
    if trade_id:
        nonce = str(attempt_id or uuid.uuid4().hex).upper()[:12]
        params["clientOrderId"] = f"EVT_{_trade_digest(trade_id)}_RB_{nonce}"[:40]
    resp = _request("POST", ORDER_PATH, params)
    if not isinstance(resp, dict) or resp.get("code") != 0:
        return {"status": "error", "error": f"close failed: code={resp.get('code') if isinstance(resp, dict) else None} msg={resp.get('msg') if isinstance(resp, dict) else resp}", "response": resp}
    return {"status": "closed", "symbol": bx, "direction": direction, "qty": close_qty, "response": resp}

def has_open_position(symbol: str, direction: str) -> bool:
    bx = to_bx_symbol(symbol)
    if not bx:
        raise RuntimeError("cannot verify position: contract symbol is unavailable")
    want = str(direction).upper()
    if want not in {"LONG", "SHORT"}:
        raise ValueError(f"invalid direction={direction!r}")
    target_symbol = str(bx).strip().upper().replace("-", "").replace("_", "")
    positions = get_positions()

    for p in positions:
        row_symbol = str(p.get("symbol", "")).strip().upper().replace("-", "").replace("_", "")
        if not row_symbol:
            raise RuntimeError("position row is missing symbol; cannot verify requested position")
        if row_symbol != target_symbol:
            continue
        position_side = str(p.get("positionSide", "")).upper()
        if position_side not in {"", "LONG", "SHORT", "BOTH"}:
            raise RuntimeError(f"position row for {row_symbol} has unrecognized positionSide={position_side!r}")
        raw_amt = p.get("positionAmt")
        if raw_amt in (None, ""):
            raise RuntimeError(f"position row for {row_symbol} is missing positionAmt")
        try:
            amt = float(raw_amt)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"position row for {row_symbol} has invalid positionAmt={raw_amt!r}") from exc
        if not math.isfinite(amt):
            raise RuntimeError(f"position row for {row_symbol} has non-finite positionAmt")
        if amt == 0:
            continue
        # In hedge mode the explicit LONG/SHORT side is authoritative; signed
        # amount is used to infer direction only for one-way BOTH/legacy rows.
        if position_side in {"LONG", "SHORT"}:
            if position_side == want:
                return True
            continue
        if position_side == "BOTH" or not position_side:
            if (want == "LONG" and amt > 0) or (want == "SHORT" and amt < 0):
                return True
    return False


def _trade_digest(trade_id: str) -> str:
    return hashlib.sha256(str(trade_id).upper().encode()).hexdigest().upper()[:16]


def _new_open_client_order_id(bx_symbol: str, trade_id: str) -> str:
    """Return one deterministic clientOrderId for one logical entry event.

    BingX requires clientOrderId to be unique per order and supports querying an
    order by this ID. A deterministic ID lets a retry/concurrent worker refer to
    the same logical order instead of manufacturing a second identity.
    """
    raw = f"{bx_symbol.upper()}:{trade_id}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()[:24]
    return f"EVT_OPEN_{digest}"[:40]


def _find_recent_order_by_client_id(symbol: str, client_order_id: str, lookback_ms: int = 120_000) -> dict | None:
    """Resolve an ambiguous order POST without issuing a duplicate order."""
    try:
        orders = get_all_orders(
            symbol,
            start_time_ms=max(0, int(time.time() * 1000) - int(lookback_ms)),
            end_time_ms=int(time.time() * 1000) + 5_000,
            limit=1000,
        )
    except Exception:
        return None
    target = str(client_order_id).upper()
    for order in orders:
        if str(order.get("clientOrderId", "")).upper() == target:
            return order
    return None


def _signal_price_displacement_pct(signal_price: float, executable_price: float) -> tuple[float, str]:
    """Absolute signal-to-executable movement, independent of direction."""
    if signal_price <= 0 or executable_price <= 0:
        return float("inf"), "UNKNOWN"
    delta_pct = (executable_price - signal_price) / signal_price * 100.0
    if abs(delta_pct) <= 1e-12:
        return 0.0, "UNCHANGED"
    return abs(delta_pct), "UP" if delta_pct > 0 else "DOWN"


def open_market(
    symbol: str,
    direction: str,
    price: float,
    trade_id: str,
    *,
    execution_quote: dict[str, Any] | None = None,
    attempt_id: str | None = None,
    trigger_bar_time_ms: int | None = None,
    max_trigger_age_minutes: float | None = None,
    order_identity_callback: Callable[[str], None] | None = None,
) -> dict:
    direction = str(direction).upper()
    if direction not in {"LONG", "SHORT"}:
        return {"status": "error", "error": f"invalid direction={direction}"}

    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}

    c = get_contract(symbol) or {}
    if not contract_exists(symbol):
        return {"status": "error", "error": "contract_unavailable", "symbol": bx}

    try:
        if has_open_position(symbol, direction):
            return {"status": "existing_position", "symbol": bx, "direction": direction}
    except Exception as exc:
        return {"status": "error", "error": f"position_check_failed: {exc}", "symbol": bx}

    try:
        prec = int(c.get("quantityPrecision") or 0)
        min_qty = float(c.get("tradeMinQuantity") or c.get("minQty") or 0)
        mult = float(c.get("multiplier") or 1)
        max_lev = int(c.get("maxShortLeverage" if direction == "SHORT" else "maxLongLeverage") or c.get("maxLeverage") or MAX_LEVERAGE)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": f"invalid contract parameters: {exc}", "symbol": bx}

    # Do not use an early quote for sizing or admission. The only quote that can
    # authorize the MARKET order is the final venue quote captured immediately
    # before the final drift/freshness checks below. This avoids stale preflight
    # quotes becoming an accidental execution reference.
    leverage = min(LEVERAGE, MAX_LEVERAGE, max_lev)

    side = "BUY" if direction == "LONG" else "SELL"
    try:
        position_side = position_side_param(direction)
    except Exception as exc:
        return {"status": "error", "error": f"position_mode_query_failed: {exc}", "symbol": bx}
    client_order_id = _new_open_client_order_id(bx, trade_id)
    # Durable identity breadcrumb before any MARKET POST. This is deliberately
    # before the final executable-quote gate, so journal I/O never adds latency
    # between that gate and order submission; persistence failure blocks the entry.
    if order_identity_callback is not None:
        try:
            order_identity_callback(client_order_id)
        except Exception as exc:
            # In the engine execution path this breadcrumb is needed to reconcile
            # an accepted order if the process dies before ORDER_ACK is written.
            # Fail closed rather than opening a position with no durable owner ID.
            log.critical("[BINGX] entry order identity persistence failed; blocking MARKET POST: %s", exc)
            return {
                "status": "error",
                "error": f"entry_order_identity_persistence_failed: {type(exc).__name__}: {exc}",
                "symbol": bx,
                "client_order_id": client_order_id,
                "order_submitted": False,
            }

    # Exchange-side idempotency guard: if another worker already submitted this
    # logical event, resolve its existing order instead of placing a second one.
    try:
        existing_order = get_order(symbol, client_order_id=client_order_id)
    except Exception:
        existing_order = {"status": "error"}
    if existing_order.get("status") == "ok":
        same_order = (
            existing_order.get("symbol", bx) == bx
            and existing_order.get("side", "") == side
            and existing_order.get("order_type", "") == "MARKET"
            and existing_order.get("position_side", position_side) == position_side
        )
        if same_order:
            order_status = str(existing_order.get("order_status", "")).upper()
            if order_status in {"NEW", "PARTIALLY_FILLED", "FILLED"}:
                return {
                    "status": "opened",
                    "symbol": bx,
                    "qty": None,
                    "leverage": leverage,
                    "sizing_price": None,
                    "signal_price": float(price),
                    "order_reference_price": None,
                    "order_id": existing_order.get("order_id"),
                    "client_order_id": existing_order.get("client_order_id") or client_order_id,
                    "idempotency": "existing_client_order_resolved_before_post",
                    "response": {"code": 0, "data": existing_order},
                    "execution_quote": execution_quote,
                }
            if order_status in {"CANCELED", "CANCELLED", "EXPIRED", "REJECTED"}:
                return {
                    "status": "error",
                    "symbol": bx,
                    "error": f"clientOrderId already used by terminal exchange order status={order_status}; refusing reuse",
                    "order_id": existing_order.get("order_id"),
                    "client_order_id": existing_order.get("client_order_id") or client_order_id,
                    "idempotency": "terminal_client_order_prevents_reuse",
                    "response": {"code": 0, "data": existing_order},
                    "execution_quote": execution_quote,
                }


    # Final venue quote is deliberately obtained after position-mode/idempotency
    # REST calls and before the leverage mutation. This is the quote used for final
    # sizing and signal-drift validation.
    final_quote = get_execution_quote(symbol, reference_price=price)
    if final_quote.get("status") != "ok":
        return {"status": "error", "error": final_quote.get("error", "final execution quote unavailable"), "symbol": bx}
    try:
        max_quote_age = max(0.0, float(os.environ.get("EXECUTION_QUOTE_MAX_AGE_SEC", "2.0")))
    except (TypeError, ValueError):
        max_quote_age = 2.0
    fresh, local_age, exchange_age, freshness_source = _quote_freshness(final_quote, max_quote_age)
    if exchange_age is None and (_require_exchange_quote_timestamp() or _unknown_exchange_quote_age_policy() == "block"):
        return {"status": "skipped_stale_signal", "error": "execution_quote_missing_exchange_timestamp", "symbol": bx, "execution_quote": final_quote}
    final_quote["quote_local_age_sec"] = local_age
    final_quote["quote_exchange_age_sec"] = exchange_age
    final_quote["quote_freshness_source"] = freshness_source
    if not fresh:
        return {"status": "skipped_stale_signal", "error": f"execution_quote_stale local_age={local_age} exchange_age={exchange_age} limit={max_quote_age}", "symbol": bx, "execution_quote": final_quote}
    bid = float(final_quote["bid"])
    ask = float(final_quote["ask"])
    sizing_price = ask if direction == "LONG" else bid
    if sizing_price <= 0 or float(price) <= 0:
        return {"status": "error", "error": "invalid final execution price", "symbol": bx, "execution_quote": final_quote}
    try:
        max_entry_slippage_pct = max(0.0, float(os.environ.get("MAX_ENTRY_SLIPPAGE_PCT", "1.00")))
    except (TypeError, ValueError):
        max_entry_slippage_pct = 1.0
    signal_drift_pct = (
        max(0.0, (sizing_price - float(price)) / float(price) * 100.0)
        if direction == "LONG"
        else max(0.0, (float(price) - sizing_price) / float(price) * 100.0)
        if direction == "SHORT"
        else float("inf")
    )
    signal_price_displacement_pct, signal_price_displacement_direction = _signal_price_displacement_pct(float(price), sizing_price)
    if signal_price_displacement_pct > MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT:
        return {
            "status": "skipped_stale_signal",
            "error": (
                f"signal_price_displacement_pct={signal_price_displacement_pct:.4f}% "
                f"> {MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT:.4f}% direction={signal_price_displacement_direction}; "
                f"signal_drift_pct={signal_drift_pct:.4f}%"
            ),
            "symbol": bx,
            "execution_quote": final_quote,
            "signal_drift_pct": signal_drift_pct,
            "signal_price_displacement_pct": signal_price_displacement_pct,
            "signal_price_displacement_direction": signal_price_displacement_direction,
            "signal_price_displacement_limit_pct": MAX_ENTRY_SIGNAL_DISPLACEMENT_PCT,
            "signal_price": float(price),
            "execution_reference_price": sizing_price,
        }
    if signal_drift_pct > max_entry_slippage_pct:
        return {"status": "skipped_stale_signal", "error": f"signal_drift_pct={signal_drift_pct:.4f}% > {max_entry_slippage_pct:.4f}% at final order gate", "symbol": bx, "execution_quote": final_quote, "signal_drift_pct": signal_drift_pct, "signal_price": float(price), "execution_reference_price": sizing_price}

    # Recompute quantity from the final quote, not from a quote captured before leverage/idempotency calls.
    raw_qty = (MARGIN_USDT * leverage) / max(sizing_price * mult, 1e-12)
    q = Decimal(str(raw_qty)).quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
    qty = float(q)
    if qty <= 0:
        # A positive theoretical size can round down to zero at contract precision.
        # When exchange minimum quantity is known, this is a safe admission skip,
        # not an execution error: do not attempt to set leverage or submit MARKET.
        if min_qty > 0:
            required_margin = (min_qty * sizing_price * mult) / max(leverage, 1)
            return {
                "status": "skipped_min_qty",
                "error": f"qty={qty} rounds to zero at quantityPrecision={prec}; min_qty={min_qty}",
                "reason": "exchange_min_quantity",
                "symbol": bx,
                "qty": qty,
                "raw_qty": raw_qty,
                "min_qty": min_qty,
                "required_margin_usdt": required_margin,
                "configured_margin_usdt": MARGIN_USDT,
                "leverage": leverage,
                "sizing_price": sizing_price,
                "execution_quote": final_quote,
            }
        return {"status": "error", "error": "calculated quantity is <= 0", "symbol": bx, "qty": qty, "raw_qty": raw_qty, "min_qty": min_qty, "leverage": leverage, "sizing_price": sizing_price, "execution_quote": final_quote}
    if min_qty > 0 and qty < min_qty:
        required_margin = (min_qty * sizing_price * mult) / max(leverage, 1)
        return {"status": "skipped_min_qty", "error": f"qty={qty} < min_qty={min_qty} at configured leverage={leverage}", "reason": "exchange_min_quantity", "symbol": bx, "qty": qty, "min_qty": min_qty, "required_margin_usdt": required_margin, "configured_margin_usdt": MARGIN_USDT, "leverage": leverage, "sizing_price": sizing_price, "execution_quote": final_quote}
    if min_qty > 0 and qty < (2.0 * min_qty):
        required_margin = (2.0 * min_qty * sizing_price * mult) / max(leverage, 1)
        return {"status": "skipped_tp_min_qty", "error": f"qty={qty} cannot support 2 TP legs with min_qty={min_qty}", "reason": "tp_two_leg_min_quantity", "symbol": bx, "qty": qty, "min_qty": min_qty, "required_margin_usdt": required_margin, "configured_margin_usdt": MARGIN_USDT, "leverage": leverage, "sizing_price": sizing_price, "execution_quote": final_quote}

    # Only change leverage after the final executable quote has passed all admission
    # checks. The final micro-freshness check below covers the small REST delay from
    # this call to the MARKET POST.
    if not _set_leverage(bx, leverage, direction):
        return {"status": "error", "error": f"failed to set leverage={leverage} for {direction}", "symbol": bx, "leverage": leverage, "execution_quote": final_quote}

    execution_quote = final_quote


    params = {
        "symbol": bx,
        "side": side,
        "positionSide": position_side,
        "type": "MARKET",
        "quantity": f"{qty:.{prec}f}",
        "clientOrderId": client_order_id,
    }

    # Last trigger-age micro-check immediately before the network POST. Unlike
    # an earlier run_once check, this is inside the function that actually submits
    # MARKET, after idempotency/position/contract/quote work has completed.
    if max_trigger_age_minutes is not None:
        try:
            trigger_ms = int(trigger_bar_time_ms or 0)
            max_trigger_age = max(0.0, float(max_trigger_age_minutes))
        except (TypeError, ValueError, OverflowError):
            trigger_ms = 0
            max_trigger_age = 0.0
        if trigger_ms <= 0:
            return {
                "status": "skipped_stale_signal",
                "error": "invalid_5m_trigger_time_at_post",
                "symbol": bx,
                "order_submitted": False,
            }
        trigger_age_minutes = max(0.0, (time.time() * 1000.0 - trigger_ms) / 60_000.0)
        if trigger_age_minutes > max_trigger_age:
            return {
                "status": "skipped_stale_signal",
                "error": f"5m_trigger_too_old_at_post age_min={trigger_age_minutes:.3f} limit={max_trigger_age:.2f}",
                "symbol": bx,
                "trigger_age_minutes": trigger_age_minutes,
                "max_trigger_age_minutes": max_trigger_age,
                "order_submitted": False,
            }

    # Last micro-check immediately before the network POST. This prevents a slow Python/network
    # path from turning a freshly-read quote into a stale order reference.
    fresh_now, local_age_now, exchange_age_now, freshness_source_now = _quote_freshness(execution_quote, max_quote_age)
    if exchange_age_now is None and (_require_exchange_quote_timestamp() or _unknown_exchange_quote_age_policy() == "block"):
        return {"status": "skipped_stale_signal", "error": "execution_quote_missing_exchange_timestamp_at_post", "symbol": bx, "execution_quote": execution_quote}
    execution_quote["quote_local_age_sec_at_post"] = local_age_now
    execution_quote["quote_exchange_age_sec_at_post"] = exchange_age_now
    execution_quote["quote_freshness_source_at_post"] = freshness_source_now
    if not fresh_now:
        return {"status": "skipped_stale_signal", "error": f"execution_quote_stale_at_post local_age={local_age_now} exchange_age={exchange_age_now} limit={max_quote_age}", "symbol": bx, "execution_quote": execution_quote}

    # Preserve the exact venue quote that passed the final pre-POST freshness/drift
    # gate. Do not write telemetry here: filesystem I/O between the final gate and
    # MARKET POST would itself add avoidable execution latency.
    execution_quote["execution_reference_price"] = sizing_price
    execution_quote["signal_drift_pct"] = signal_drift_pct

    order_submit_at_ms = int(time.time() * 1000)
    execution_quote["order_submit_at_ms"] = order_submit_at_ms
    response = _request("POST", ORDER_PATH, params)

    # Persist the authoritative pre-POST quote immediately after the POST attempt,
    # with the actual submit timestamp. Telemetry remains best-effort and cannot
    # influence order admission, submission, or reconciliation.
    try:
        telemetry.record_quote_snapshot(
            event_id=trade_id,
            attempt_id=attempt_id,
            symbol=symbol,
            direction=direction,
            quote=execution_quote,
            signal_price=float(price),
            stage="FINAL_PRE_POST",
        )
    except Exception:
        pass
    log.info("[BINGX] OPEN order response: symbol=%s direction=%s positionSide=%s code=%s msg=%s", bx, direction, position_side, response.get("code") if isinstance(response, dict) else None, response.get("msg") if isinstance(response, dict) else None)

    if isinstance(response, dict) and response.get("code") != 0:
        # Audit P1-4 (order idempotency): a transport-level failure (-1) leaves
        # the outcome unknown -- the order may have been created even though we
        # did not receive an ack. Never blindly retry a POST; verify the result
        # via the position instead. The pre-flight has_open_position check above
        # guarantees any position present now was opened by THIS order.
        transport_error = (
            response.get("code") == -1
            and "missing bingx credentials" not in str(response.get("msg", "")).lower()
        )
        if transport_error:
            log.warning("[BINGX] Order POST transport error for %s (%s); reconciling clientOrderId before using position state...", bx, response.get("msg"))
            resolved_order = get_order(symbol, client_order_id=client_order_id)
            if resolved_order.get("status") == "ok":
                same_order = (
                    resolved_order.get("symbol", bx) == bx
                    and resolved_order.get("side", "") == side
                    and resolved_order.get("order_type", "") == "MARKET"
                    and resolved_order.get("position_side", position_side) == position_side
                )
                if same_order:
                    order_status = str(resolved_order.get("order_status", "")).upper()
                    if order_status in {"NEW", "PARTIALLY_FILLED", "FILLED"}:
                        log.warning("[BINGX] Matching active/filled clientOrderId found after transport error; treating MARKET order as resolved.")
                        return {
                            "status": "opened",
                            "symbol": bx,
                            "qty": qty,
                            "leverage": leverage,
                            "sizing_price": sizing_price,
                            "signal_price": float(price),
                            "order_reference_price": sizing_price,
                            "order_id": resolved_order.get("order_id"),
                            "client_order_id": resolved_order.get("client_order_id") or client_order_id,
                            "idempotency": "client_order_id_verified_after_transport_error",
                            "response": response,
                            "execution_quote": execution_quote,
                            "order_submit_at_ms": order_submit_at_ms,
                            "historical_order": resolved_order,
                        }
                    if order_status in {"CANCELED", "CANCELLED", "EXPIRED", "REJECTED"}:
                        return {
                            "status": "error",
                            "symbol": bx,
                            "error": f"clientOrderId already used by terminal exchange order status={order_status}; refusing reuse",
                            "order_id": resolved_order.get("order_id"),
                            "client_order_id": resolved_order.get("client_order_id") or client_order_id,
                            "idempotency": "terminal_client_order_prevents_reuse",
                            "response": response,
                            "execution_quote": execution_quote,
                            "historical_order": resolved_order,
                        }
            historical_order = _find_recent_order_by_client_id(symbol, client_order_id)
            if historical_order is not None:
                log.warning("[BINGX] Matching historical clientOrderId found after transport error; treating MARKET order as resolved.")
                return {
                    "status": "opened",
                    "symbol": bx,
                    "qty": qty,
                    "leverage": leverage,
                    "sizing_price": sizing_price,
                    "signal_price": float(price),
                    "order_reference_price": sizing_price,
                    "order_id": historical_order.get("orderId"),
                    "client_order_id": historical_order.get("clientOrderId") or client_order_id,
                    "idempotency": "client_order_id_verified_after_transport_error",
                    "response": response,
                    "execution_quote": execution_quote,
                    "order_submit_at_ms": order_submit_at_ms,
                    "historical_order": historical_order,
                }
            try:
                if has_open_position(symbol, direction):
                    log.warning("[BINGX] Position found after transport error with no matching historical order; treating as unresolved-but-opened without re-POST.")
                    return {
                        "status": "opened",
                        "symbol": bx,
                        "qty": qty,
                        "leverage": leverage,
                        "sizing_price": sizing_price,
                        "signal_price": float(price),
                        "order_reference_price": sizing_price,
                        "order_id": None,
                        "client_order_id": client_order_id,
                        "idempotency": "position_verified_after_transport_error_no_order_match",
                        "response": response,
                        "execution_quote": execution_quote,
                        "order_submit_at_ms": order_submit_at_ms,
                    }
            except Exception as exc:
                log.error("[BINGX] Post-error position verification failed: %s", exc)

        return {"status": "error", "error": str(response.get("msg", "")), "symbol": bx, "clientOrderId": client_order_id, "response": response}

    data = response.get("data") or {}
    order = data.get("order") or {}
    order_id = order.get("orderId") or data.get("orderId")

    return {
        "status": "opened",
        "symbol": bx,
        "qty": qty,
        "leverage": leverage,
        "sizing_price": sizing_price,
        "signal_price": float(price),
        "order_reference_price": sizing_price,
        "order_id": order_id,
        "client_order_id": order.get("clientOrderId") or client_order_id,
        "response": response,
        "execution_quote": execution_quote,
        "order_submit_at_ms": order_submit_at_ms,
    }


def get_position_directional(symbol: str, direction: str) -> dict:
    bx_symbol = to_bx_symbol(symbol)
    direction = str(direction).upper()
    if not bx_symbol:
        return {"status": "error", "error": "contract_not_found", "symbol": bx_symbol}
    if direction not in {"LONG", "SHORT"}:
        return {"status": "error", "error": f"invalid direction={direction!r}", "symbol": bx_symbol}

    try:
        resp = _request("GET", POSITION_PATH, {"symbol": bx_symbol})
    except Exception as exc:
        return {"status": "error", "error": f"get_position request failed: {type(exc).__name__}: {exc}", "symbol": bx_symbol}
    if not _response_code_is_success(resp):
        code = resp.get("code") if isinstance(resp, dict) else "invalid_response"
        message = resp.get("msg") if isinstance(resp, dict) else repr(resp)
        return {"status": "error", "error": f"get_position failed: code={code} msg={message}", "symbol": bx_symbol}

    try:
        rows = _normalize_positions_list(resp)
    except Exception as exc:
        return {"status": "error", "error": f"get_position response unavailable: {exc}", "symbol": bx_symbol, "positionSide": direction}

    target_symbol = str(bx_symbol).strip().upper().replace("-", "").replace("_", "")
    for p in rows:
        row_symbol = str(p.get("symbol", "")).strip().upper().replace("-", "").replace("_", "")
        if row_symbol != target_symbol:
            continue
        position_side = str(p.get("positionSide", "")).upper()
        if position_side not in {"LONG", "SHORT", "BOTH"}:
            return {
                "status": "error",
                "error": f"position row for {bx_symbol} has unrecognized positionSide={position_side!r}",
                "symbol": bx_symbol,
                "positionSide": direction,
            }
        raw_amt = p.get("positionAmt")
        try:
            raw_amount = float(raw_amt)
            avg_price = float(p.get("avgPrice") or p.get("entryPrice") or 0)
        except (TypeError, ValueError) as exc:
            return {
                "status": "error",
                "error": f"malformed position values for {bx_symbol}: {exc}",
                "symbol": bx_symbol,
                "positionSide": direction,
            }
        if not math.isfinite(raw_amount) or not math.isfinite(avg_price):
            return {
                "status": "error",
                "error": f"non-finite position values for {bx_symbol}",
                "symbol": bx_symbol,
                "positionSide": direction,
            }
        if position_side == "BOTH":
            if direction == "LONG" and raw_amount < 0:
                continue
            if direction == "SHORT" and raw_amount > 0:
                continue
        elif position_side != direction:
            continue

        qty = abs(raw_amount)
        if qty <= 0:
            continue
        if avg_price <= 0:
            return {
                "status": "error",
                "error": f"open position for {bx_symbol} has no valid positive avgPrice/entryPrice",
                "symbol": bx_symbol,
                "positionSide": direction,
            }
        try:
            entry_price = float(p.get("entryPrice") or avg_price)
        except (TypeError, ValueError):
            entry_price = 0.0
        if not math.isfinite(entry_price) or entry_price <= 0:
            entry_price = avg_price
        return {
            "status": "found",
            "symbol": p.get("symbol", bx_symbol),
            "positionSide": direction,
            "avgPrice": avg_price,
            "positionAmt": qty,
            "entryPrice": entry_price,
        }

    return {"status": "not_found", "symbol": bx_symbol, "positionSide": direction}


def wait_for_position_fill_directional(symbol: str, direction: str, timeout_sec: int = 30, poll_interval: float = 0.5) -> dict:
    started = time.time()
    last_error: dict | None = None
    while time.time() - started < timeout_sec:
        pos = get_position_directional(symbol, direction)
        if pos.get("status") == "found":
            return pos
        if pos.get("status") == "error":
            last_error = pos
        time.sleep(poll_interval)

    # One final authoritative read. A transient API error during the normal poll
    # interval must not be mistaken for a definitive non-fill.
    final = get_position_directional(symbol, direction)
    if final.get("status") == "found":
        return final
    if final.get("status") == "error":
        return {**final, "last_poll_error": last_error}
    return {"status": "timeout", "symbol": to_bx_symbol(symbol), "positionSide": str(direction).upper(), "last_poll_error": last_error}


def get_open_protection_directional(
    symbol: str,
    direction: str,
    *,
    timeout_sec: float | None = None,
    retryable: bool = True,
) -> dict:
    bx_symbol = to_bx_symbol(symbol)
    direction = str(direction).upper()
    if not bx_symbol:
        return {"status": "error", "error": "contract_not_found", "tp_orders": [], "sl_orders": []}
    try:
        expected_position_side = position_side_param(direction)
    except Exception as exc:
        return {"status": "error", "error": f"position mode unavailable: {exc}", "tp_orders": [], "sl_orders": []}

    resp = _request(
        "GET",
        OPEN_ORDERS_PATH,
        {"symbol": bx_symbol},
        timeout_sec=timeout_sec,
        retryable=retryable,
    )
    if not _response_code_is_success(resp):
        code = resp.get("code") if isinstance(resp, dict) else "invalid_response"
        message = resp.get("msg") if isinstance(resp, dict) else repr(resp)
        return {"status": "error", "error": f"openOrders failed: code={code} msg={message}", "tp_orders": [], "sl_orders": []}

    try:
        orders = _normalize_orders_list(resp)
    except Exception as exc:
        return {"status": "error", "error": f"openOrders response unavailable: {exc}", "tp_orders": [], "sl_orders": []}

    tp_orders = []
    sl_orders = []
    target_symbol = str(bx_symbol).strip().upper().replace("-", "").replace("_", "")

    for index, order in enumerate(orders):
        raw_symbol = str(order.get("symbol", "")).strip()
        if not raw_symbol:
            return {"status": "error", "error": f"openOrders row {index} is missing symbol; protection state is unknown", "tp_orders": [], "sl_orders": []}
        row_symbol = raw_symbol.upper().replace("-", "").replace("_", "")
        if row_symbol != target_symbol:
            continue
        # Include both plausible directional labels so reconciliation can see and
        # clean stale engine-owned orders. Unknown labels cannot prove that the
        # response contains no protection for this position.
        position_side = str(order.get("positionSide", "")).upper()
        if position_side not in {"LONG", "SHORT", "BOTH"}:
            return {"status": "error", "error": f"openOrders row {index} for {bx_symbol} has unknown positionSide={position_side!r}", "tp_orders": [], "sl_orders": []}
        order_type = str(order.get("type", "")).upper()
        if not order_type:
            return {"status": "error", "error": f"openOrders row {index} for {bx_symbol} is missing order type; protection state is unknown", "tp_orders": [], "sl_orders": []}
        if position_side not in {direction, "BOTH"}:
            continue
        if order_type in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
            if not str(order.get("orderId", "")).strip():
                return {"status": "error", "error": f"protective TP row {index} for {bx_symbol} is missing orderId", "tp_orders": [], "sl_orders": []}
            tp_orders.append(order)
        elif order_type in {"STOP", "STOP_MARKET"}:
            if not str(order.get("orderId", "")).strip():
                return {"status": "error", "error": f"protective SL row {index} for {bx_symbol} is missing orderId", "tp_orders": [], "sl_orders": []}
            sl_orders.append(order)

    return {"status": "ok", "symbol": bx_symbol, "positionSide": direction, "tp_orders": tp_orders, "sl_orders": sl_orders}


def prepare_protection_capacity(symbol: str, direction: str) -> dict:
    """Ensure a flat symbol is not carrying stale engine protection.

    A new trade needs one SL plus two TP legs. Opening the market position first
    and only then discovering stale TP/SL orders can trigger the exchange order
    limit and force an emergency exit. When no position exists, engine-owned
    protective orders are stale by definition and can be safely cancelled.

    Manual/non-engine protection is never cancelled. Its presence blocks a new
    engine entry for this symbol/direction because the exchange-side protection
    capacity cannot be proven safely.
    """
    direction = str(direction).upper()
    if direction not in {"LONG", "SHORT"}:
        return {"status": "error", "error": f"invalid direction={direction}"}

    position = get_position_directional(symbol, direction)
    if position.get("status") == "found":
        return {
            "status": "blocked_existing_position",
            "symbol": symbol,
            "direction": direction,
            "position": position,
        }
    if position.get("status") not in {"not_found", "found"}:
        return {
            "status": "error",
            "symbol": symbol,
            "direction": direction,
            "error": position.get("error", "position state unavailable"),
        }

    existing = get_open_protection_directional(symbol, direction)
    if existing.get("status") != "ok":
        return {"status": "error", "symbol": symbol, "direction": direction, "error": existing.get("error", "openOrders unavailable")}

    orders = list(existing.get("sl_orders", [])) + list(existing.get("tp_orders", []))
    stale_engine_ids = []
    external_ids = []
    for order in orders:
        oid = str(order.get("orderId", ""))
        cid = str(order.get("clientOrderId", "")).upper()
        if not oid:
            continue
        if cid.startswith("EVT_"):
            stale_engine_ids.append(oid)
        else:
            external_ids.append(oid)

    if external_ids:
        return {
            "status": "blocked_external_protection",
            "symbol": symbol,
            "direction": direction,
            "external_order_ids": external_ids,
            "existing_engine_order_ids": stale_engine_ids,
            "existing_sl_count": len(existing.get("sl_orders", [])),
            "existing_tp_count": len(existing.get("tp_orders", [])),
            "error": "non-engine protection orders are present; exchange protection capacity cannot be proven safely",
        }

    cancelled = []
    for order_id in stale_engine_ids:
        try:
            resp = cancel_order(symbol, order_id)
        except Exception as exc:
            return {
                "status": "error",
                "symbol": symbol,
                "direction": direction,
                "error": f"stale engine protection cleanup failed for {order_id}: {exc}",
                "cancelled_order_ids": cancelled,
            }
        if not isinstance(resp, dict) or resp.get("code") not in (0, "0"):
            return {
                "status": "error",
                "symbol": symbol,
                "direction": direction,
                "error": f"stale engine protection cleanup failed for {order_id}: {resp}",
                "cancelled_order_ids": cancelled,
            }
        cancelled.append(order_id)

    if stale_engine_ids:
        verified = get_open_protection_directional(symbol, direction)
        if verified.get("status") != "ok":
            return {
                "status": "error",
                "symbol": symbol,
                "direction": direction,
                "error": "stale engine protection cleanup verification failed",
                "cancelled_order_ids": cancelled,
            }
        remaining = [
            str(o.get("orderId", ""))
            for o in list(verified.get("sl_orders", [])) + list(verified.get("tp_orders", []))
            if str(o.get("orderId", "")) in set(stale_engine_ids)
        ]
        if remaining:
            return {
                "status": "error",
                "symbol": symbol,
                "direction": direction,
                "error": f"stale engine protection still visible after cleanup: {remaining}",
                "cancelled_order_ids": cancelled,
            }

    return {
        "status": "ready",
        "symbol": symbol,
        "direction": direction,
        "cancelled_order_ids": cancelled,
        "existing_sl_count": 0,
        "existing_tp_count": 0,
        "required_new_protection_orders": 3,
    }


def _round_qty(qty: float, precision: int) -> float:
    if precision < 0:
        return float(qty)
    return float(Decimal(str(qty)).quantize(Decimal(1).scaleb(-precision), rounding=ROUND_DOWN))


def _format_qty(qty: float, precision: int) -> str:
    return f"{qty:.{precision}f}"


def _format_price(price: float, precision: int) -> str:
    return f"{price:.{precision}f}"


def build_tp_client_order_id(leg: str, trade_id: str | None = None) -> str:
    leg_u = str(leg).upper()
    if trade_id:
        return f"EVT_{_trade_digest(trade_id)}_{leg_u}"
    return f"EVT_{leg_u}"


def build_sl_client_order_id(trade_id: str | None = None) -> str:
    if trade_id:
        return f"EVT_{_trade_digest(trade_id)}_SL"
    return "EVT_SL"


def _repair_client_order_id(base_client_order_id: str) -> str:
    """Create a bounded, unique ID for a replacement protection order.

    Protection repairs must not reuse an ID while a stale order with that ID may
    still be open. Keep the normal ID unchanged for first placement/retries; only
    repairs use this alternate form. The digest in a shortened stem prevents two
    long base IDs from collapsing to the same 30-character prefix.
    """
    base = str(base_client_order_id or "EVT_REPAIR").upper()
    if len(base) <= 30:
        stem = base
    else:
        stem = f"{base[:21]}_{hashlib.sha256(base.encode('utf-8')).hexdigest().upper()[:8]}"
    suffix = f"_R{uuid.uuid4().hex.upper()[:8]}"
    return f"{stem[:30]}{suffix}"[:40]


def _client_order_id_matches_base(client_order_id: str, base_client_order_id: str) -> bool:
    """Match a canonical clientOrderId and repair IDs derived from it."""
    client_id = str(client_order_id or "").upper()
    base = str(base_client_order_id or "").upper()
    if not client_id or not base:
        return False
    if client_id == base:
        return True
    if not re.search(r"_R[0-9A-F]{8}$", client_id):
        return False
    stem = client_id[:-10]
    if len(base) <= 30:
        return stem == base
    expected_stem = f"{base[:21]}_{hashlib.sha256(base.encode('utf-8')).hexdigest().upper()[:8]}"
    return stem == expected_stem


def _cancel_engine_protection_orders_verified(
    symbol: str, direction: str, order_ids: list[str], *, attempts: int = 4,
) -> tuple[bool, str | None]:
    """Cancel known engine-owned protective orders and verify exchange absence.

    The caller must have identified these IDs as belonging to the repaired TP
    leg/SL. A cancel acknowledgement alone is insufficient because cancellation
    may be asynchronous; openOrders is polled before success is reported.
    """
    unique_ids = list(dict.fromkeys(str(value) for value in order_ids if str(value)))
    for order_id in unique_ids:
        try:
            before = get_open_protection_directional(symbol, direction)
        except Exception as exc:
            return False, f"pre-cancel openOrders failed for {order_id}: {exc}"
        if not isinstance(before, dict) or before.get("status") != "ok":
            return False, f"pre-cancel openOrders unavailable for {order_id}"
        all_orders = list(before.get("sl_orders", [])) + list(before.get("tp_orders", []))
        matched = next((item for item in all_orders if str(item.get("orderId", "")) == order_id), None)
        if matched is None:
            continue
        if not str(matched.get("clientOrderId", "")).upper().startswith("EVT_"):
            return False, f"refusing to cancel non-engine order {order_id}"
        try:
            cancel_order(symbol, order_id)
        except Exception as exc:
            # Still verify the exchange state: a timeout can occur after the
            # exchange accepted the cancellation.
            log.warning("[BINGX] cancel request raised for stale protection %s: %s", order_id, exc)
        absent = False
        last_error = "cancel not yet visible as absent"
        for attempt in range(max(1, attempts)):
            if attempt:
                time.sleep(0.15 * attempt)
            try:
                after = get_open_protection_directional(symbol, direction)
            except Exception as exc:
                last_error = f"post-cancel openOrders failed: {exc}"
                continue
            if not isinstance(after, dict) or after.get("status") != "ok":
                last_error = "post-cancel openOrders unavailable"
                continue
            after_orders = list(after.get("sl_orders", [])) + list(after.get("tp_orders", []))
            if not any(str(item.get("orderId", "")) == order_id for item in after_orders):
                absent = True
                break
            last_error = f"order {order_id} remains visible after cancellation"
        if not absent:
            return False, last_error
    return True, None


def _allocate_tp_quantities(position_qty: float, precision: int, min_qty: float, fractions: list[float]) -> list[float]:
    if position_qty <= 0:
        raise ValueError("position_qty must be > 0")
    if not fractions or any(f <= 0 for f in fractions):
        raise ValueError("fractions must be positive")

    step = Decimal(1).scaleb(-precision) if precision >= 0 else Decimal("1")
    pos = Decimal(str(position_qty))
    min_q = Decimal(str(max(min_qty, 0.0)))
    min_leg = max(step, min_q)
    k = len(fractions)

    if pos < min_leg * k:
        raise ValueError(f"position_qty={position_qty} cannot support {k} TP legs with min_leg={min_leg}")

    total_fraction = sum(Decimal(str(f)) for f in fractions)
    normalized = [Decimal(str(f)) / total_fraction for f in fractions]
    raw_targets = [pos * f for f in normalized]

    quantities = []
    for raw in raw_targets:
        q_step = (raw / step).to_integral_value(rounding=ROUND_FLOOR) * step
        quantities.append(max(q_step, min_leg))

    while sum(quantities) > pos:
        best_reduce = max(range(k), key=lambda i: (quantities[i] - min_leg, quantities[i] - raw_targets[i]))
        if quantities[best_reduce] <= min_leg:
            raise ValueError("Cannot reduce leg below min_leg")
        quantities[best_reduce] -= step

    while sum(quantities) < pos:
        best_add = max(range(k), key=lambda i: raw_targets[i] - quantities[i])
        quantities[best_add] += step

    remainder = pos - sum(quantities)
    if remainder != 0:
        quantities[-1] += remainder
        quantities[-1] = quantities[-1].quantize(step)

    return [float(q) for q in quantities]


def _normalize_tp_levels(tp_levels: list) -> list[dict]:
    if isinstance(tp_levels, list) and len(tp_levels) == 0:
        return []
    normalized = []
    for tp in (tp_levels or []):
        leg = str(tp.get("leg", f"tp{len(normalized) + 1}"))
        try:
            pnl_pct = float(tp.get("pnl_pct", 0))
            fraction = float(tp.get("close_fraction", 0))
        except (TypeError, ValueError):
            continue

        if not math.isfinite(pnl_pct) or not math.isfinite(fraction) or pnl_pct <= 0 or fraction <= 0:
            continue
        normalized.append({"leg": leg, "pnl_pct": pnl_pct, "close_fraction": fraction})

    if not normalized:
        return []

    total = sum(x["close_fraction"] for x in normalized)
    if not math.isfinite(total) or total <= 0:
        return []
    for x in normalized:
        x["close_fraction"] /= total
    return normalized


def _tp_leg_from_order(order: dict, expected_leg: str, expected_price: float, price_precision: int, trade_id: str | None = None, direction: str | None = None) -> bool:
    # Existing exchange orders are reusable only if both the close side and
    # account-mode positionSide match the position they are meant to protect.
    if direction not in {"LONG", "SHORT"}:
        return False
    expected_side = "SELL" if direction == "LONG" else "BUY"
    if str(order.get("side", "")).upper() != expected_side:
        return False
    try:
        expected_position_side = position_side_param(direction)
    except Exception:
        return False
    if str(order.get("positionSide", "")).upper() != expected_position_side:
        return False
    order_type = str(order.get("type", "")).upper()
    if order_type not in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
        return False

    try:
        actual_price = float(order.get("stopPrice", 0) or order.get("price", 0) or 0)
    except (TypeError, ValueError):
        return False

    if actual_price <= 0:
        return False

    expected_leg = str(expected_leg).upper()
    client_id = str(order.get("clientOrderId", "")).upper()

    expected_formatted = _format_price(expected_price, price_precision)
    actual_formatted = _format_price(actual_price, price_precision)

    if trade_id:
        base_client_order_id = f"EVT_{_trade_digest(trade_id)}_{expected_leg}"
        # Repair IDs retain the canonical leg identity with a unique _Rxxxxxxxx
        # suffix. Recognize those IDs as the same logical TP leg; otherwise every
        # later reconciliation would treat a successfully repaired TP as stale
        # and replace it again on every run.
        if _client_order_id_matches_base(client_id, base_client_order_id):
            return actual_formatted == expected_formatted

    if client_id.startswith("EVT_") and f"_{expected_leg}_" in f"_{client_id}_":
        return actual_formatted == expected_formatted
    return False


def _valid_quote(bid: float, ask: float) -> bool:
    return (
        math.isfinite(bid)
        and math.isfinite(ask)
        and bid > 0
        and ask > 0
        and ask >= bid
    )


def _normalize_market_timestamp_ms(value: Any) -> int | None:
    """Normalize a BingX market timestamp to epoch milliseconds."""
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(raw) or raw <= 0:
        return None
    if raw >= 1e11:
        return int(raw)
    if raw >= 1e8:
        return int(raw * 1000.0)
    return None


def _require_exchange_quote_timestamp() -> bool:
    raw = str(os.environ.get("EXECUTION_REQUIRE_EXCHANGE_TIMESTAMP", "true")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _unknown_exchange_quote_age_policy() -> str:
    """Explicit policy when the venue response has no usable exchange timestamp."""
    raw = str(os.environ.get("EXECUTION_UNKNOWN_QUOTE_AGE_POLICY", "block")).strip().lower()
    aliases = {
        "allow": "allow_with_local_age",
        "allow_with_local": "allow_with_local_age",
        "allow_with_local_age": "allow_with_local_age",
        "block": "block",
        "reject": "block",
    }
    return aliases.get(raw, "block")


def _quote_freshness(quote: dict[str, Any], max_age_sec: float) -> tuple[bool, float | None, float | None, str]:
    limit = max(0.0, float(max_age_sec))
    observed_ms = _normalize_market_timestamp_ms(quote.get("quote_observed_at_ms"))
    exchange_ms = _normalize_market_timestamp_ms(quote.get("quote_exchange_time_ms"))
    local_now_ms = int(time.time() * 1000)
    exchange_now_ms = local_now_ms + int(SERVER_TIME_OFFSET_MS)
    local_age = max(0.0, (local_now_ms - observed_ms) / 1000.0) if observed_ms is not None else None
    exchange_age = None
    if exchange_ms is not None:
        raw_age = (exchange_now_ms - exchange_ms) / 1000.0
        try:
            future_tolerance = max(0.0, float(os.environ.get("EXECUTION_QUOTE_MAX_FUTURE_SKEW_SEC", "1.0")))
        except (TypeError, ValueError):
            future_tolerance = 1.0
        if raw_age >= 0:
            exchange_age = raw_age
        elif raw_age >= -future_tolerance:
            # BingX's HTTP Date synchronization is only second-granular; a tiny
            # negative age is normal clock/rounding skew, not a future quote.
            exchange_age = 0.0
        else:
            return False, local_age, raw_age, "exchange_timestamp_future"
    if limit > 0 and exchange_age is not None and exchange_age > limit:
        return False, local_age, exchange_age, "exchange_timestamp"
    if limit > 0 and local_age is not None and local_age > limit:
        return False, local_age, exchange_age, "local_observed_elapsed"
    return True, local_age, exchange_age, "exchange_timestamp" if exchange_age is not None else "local_observed_elapsed"


def _parse_top_of_book_payload(resp: Any, symbol: str, source: str) -> tuple[float, float, Any] | None:
    """Parse BingX futures book/ticker payloads across current and legacy envelopes.

    Current futures Book Ticker responses may nest the row under
    ``data.book_ticker``; older/alternate responses may expose ``data`` directly
    or as a one-element/list payload. Keep all accepted forms here so execution
    quote fallback is driven by actual schema incompatibility, not by a parser gap.
    """
    if not isinstance(resp, dict) or resp.get("code") != 0:
        return None
    data = resp.get("data")
    row: dict[str, Any] | None = None

    def _select_row(payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            nested = payload.get("book_ticker") or payload.get("bookTicker")
            if isinstance(nested, (dict, list)):
                selected = _select_row(nested)
                if selected is not None:
                    return selected
            return payload
        if isinstance(payload, list):
            match = next((x for x in payload if isinstance(x, dict) and str(x.get("symbol", "")).upper() == symbol), None)
            if match is not None:
                return match
            if len(payload) == 1 and isinstance(payload[0], dict):
                return payload[0]
        return None

    row = _select_row(data)
    if not isinstance(row, dict):
        return None
    try:
        bid = float(row.get("bidPrice", 0) or 0)
        ask = float(row.get("askPrice", 0) or 0)
    except (TypeError, ValueError):
        return None
    if not _valid_quote(bid, ask):
        return None
    return bid, ask, row.get("time")


def _parse_depth_top(resp: Any) -> tuple[float, float, Any] | None:
    if not isinstance(resp, dict) or resp.get("code") != 0:
        return None
    data = resp.get("data")
    if not isinstance(data, dict):
        return None
    bids = data.get("bids") or []
    asks = data.get("asks") or []

    def _price(level: Any) -> float | None:
        try:
            if isinstance(level, (list, tuple)) and level:
                return float(level[0])
            if isinstance(level, dict):
                return float(level.get("price") or level.get("p") or 0)
        except (TypeError, ValueError):
            return None
        return None

    bid = next((p for p in (_price(x) for x in bids) if p and math.isfinite(p) and p > 0), None)
    ask = next((p for p in (_price(x) for x in asks) if p and math.isfinite(p) and p > 0), None)
    if bid is None or ask is None or not _valid_quote(bid, ask):
        return None
    return bid, ask, data.get("T") or data.get("timestamp") or data.get("time")


def _safe_exception_message(exc: Exception) -> str:
    """Redact query strings/signatures from network exception text before telemetry."""
    text = str(exc)
    if "?" in text:
        text = text.split("?", 1)[0] + "?[REDACTED]"
    return text


def _quote_error(source: str, resp: Any, symbol: str) -> str:
    code = resp.get("code") if isinstance(resp, dict) else None
    msg = resp.get("msg") if isinstance(resp, dict) else None
    return f"{source} unavailable: code={code} msg={msg or 'invalid response'}"


def _quote_provenance_fields(quote_time: Any, *, observed_at_ms: int | None = None) -> dict[str, Any]:
    exchange_time_ms = _normalize_market_timestamp_ms(quote_time)
    observed_ms = int(time.time() * 1000) if observed_at_ms is None else int(observed_at_ms)
    fresh, local_age, exchange_age, source = _quote_freshness(
        {"quote_observed_at_ms": observed_ms, "quote_exchange_time_ms": exchange_time_ms},
        max(0.0, float(os.environ.get("EXECUTION_QUOTE_MAX_AGE_SEC", "2.0"))),
    )
    policy = _unknown_exchange_quote_age_policy()
    if exchange_time_ms is None:
        freshness_status = "EXCHANGE_AGE_UNKNOWN_LOCAL_ALLOWED" if policy == "allow_with_local_age" else "EXCHANGE_AGE_UNKNOWN_BLOCKED"
        quality = "LOCAL_OBSERVED_ONLY"
    elif fresh and exchange_time_ms is not None and exchange_age is not None and source != "exchange_timestamp_future":
        freshness_status = "EXCHANGE_TIME_VERIFIED"
        quality = "EXCHANGE_TIME_VERIFIED"
    elif source == "exchange_timestamp_future":
        freshness_status = "EXCHANGE_TIMESTAMP_FUTURE"
        quality = "EXCHANGE_TIME_UNVERIFIED"
    else:
        freshness_status = "EXCHANGE_TIME_STALE"
        quality = "EXCHANGE_TIME_UNVERIFIED"
    return {
        "quote_exchange_time_ms": exchange_time_ms,
        "quote_observed_at_ms": observed_ms,
        "quote_local_age_sec": local_age,
        "quote_exchange_age_sec": exchange_age,
        "quote_freshness_source": source,
        "quote_freshness_status": freshness_status,
        "quote_unknown_age_policy": policy,
        "execution_reference_quality": quality,
    }


def _quote_acceptance(quote_time: Any, *, observed_at_ms: int | None = None) -> tuple[bool, dict[str, Any], str | None]:
    provenance = _quote_provenance_fields(quote_time, observed_at_ms=observed_at_ms)
    exchange_time_ms = provenance.get("quote_exchange_time_ms")
    if exchange_time_ms is None and (_require_exchange_quote_timestamp() or _unknown_exchange_quote_age_policy() == "block"):
        return False, provenance, "execution_quote_missing_exchange_timestamp"
    source = provenance.get("quote_freshness_source")
    if source == "exchange_timestamp_future":
        return False, provenance, "execution_quote_exchange_timestamp_in_future"
    try:
        max_age = max(0.0, float(os.environ.get("EXECUTION_QUOTE_MAX_AGE_SEC", "2.0")))
    except (TypeError, ValueError):
        max_age = 2.0
    fresh, _, _, _ = _quote_freshness(
        {"quote_observed_at_ms": provenance.get("quote_observed_at_ms"), "quote_exchange_time_ms": exchange_time_ms},
        max_age,
    )
    if not fresh:
        return False, provenance, (
            f"execution_quote_stale local_age={provenance.get('quote_local_age_sec')} "
            f"exchange_age={provenance.get('quote_exchange_age_sec')} limit={max_age}"
        )
    return True, provenance, None


def get_execution_quote(symbol: str, *, min_interval_sec: float | None = None, reference_price: float | None = None) -> dict[str, Any]:
    """Return a valid BingX executable top-of-book quote immediately before entry.

    Fallback order is BingX-only: bookTicker -> ticker -> depth. No Binance or
    candle-close fallback is allowed because entry safety must be based on the
    execution venue's live market.
    """
    global _LAST_BOOK_TICKER_TS
    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found", "symbol": symbol}
    try:
        interval = float(
            os.environ.get("BINGX_BOOK_TICKER_MIN_INTERVAL_SEC", "1.05")
            if min_interval_sec is None else min_interval_sec
        )
    except (TypeError, ValueError):
        interval = 1.05
    interval = max(0.0, interval)

    def _call(path: str) -> Any:
        return _request(
            "GET", path, {"symbol": bx}, signed=True,
            timeout_sec=float(os.environ.get("BINGX_BOOK_TICKER_TIMEOUT_SEC", "3")),
            retryable=False,
        )

    with _BOOK_TICKER_LOCK:
        wait = interval - (time.monotonic() - _LAST_BOOK_TICKER_TS)
        if wait > 0:
            time.sleep(wait)
        attempts: list[tuple[str, Any]] = []
        resp = _call(BOOK_TICKER_PATH)
        _LAST_BOOK_TICKER_TS = time.monotonic()
        attempts.append(("bookTicker", resp))

        parsed = _parse_top_of_book_payload(resp, bx, "bookTicker")
        if parsed is not None:
            bid, ask, quote_time = parsed
            accepted, provenance, rejection_reason = _quote_acceptance(quote_time)
            exchange_time_ms = provenance.get("quote_exchange_time_ms")
            if not accepted:
                log.warning("[EXEC_QUOTE_REJECT] %s | bookTicker | %s", bx, rejection_reason)
            else:
                return {
                    "status": "ok", "symbol": bx, "bid": bid, "ask": ask,
                    "spread_pct": ((ask - bid) / bid * 100.0) if bid > 0 else None,
                    "time": quote_time, "last_price": None, "quote_source": "bookTicker",
                    "quote_freshness_status": provenance.get("quote_freshness_status"),
                    "quote_unknown_age_policy": provenance.get("quote_unknown_age_policy"),
                    "quote_sources_attempted": ["bookTicker"],
                    "quote_fallback_reason": None,
                    **provenance,
                }

        ticker_resp = _call(TICKER_PATH)
        attempts.append(("ticker", ticker_resp))
        parsed = _parse_top_of_book_payload(ticker_resp, bx, "ticker")
        if parsed is not None:
            bid, ask, quote_time = parsed
            accepted, provenance, rejection_reason = _quote_acceptance(quote_time)
            exchange_time_ms = provenance.get("quote_exchange_time_ms")
            if not accepted:
                log.warning("[EXEC_QUOTE_REJECT] %s | ticker | %s", bx, rejection_reason)
            else:
                log.warning("[EXEC_QUOTE_FALLBACK] %s | bookTicker invalid/unavailable -> ticker", bx)
                return {
                    "status": "ok", "symbol": bx, "bid": bid, "ask": ask,
                    "spread_pct": ((ask - bid) / bid * 100.0) if bid > 0 else None,
                    "time": quote_time, "last_price": None, "quote_source": "ticker",
                    "quote_freshness_status": provenance.get("quote_freshness_status"),
                    "quote_unknown_age_policy": provenance.get("quote_unknown_age_policy"),
                    "quote_sources_attempted": [source for source, _ in attempts],
                    "quote_fallback_reason": _quote_error("bookTicker", attempts[0][1], bx) if attempts else "bookTicker_invalid",
                    **provenance,
                }

        depth_resp = _call(DEPTH_PATH)
        attempts.append(("depth", depth_resp))
        parsed = _parse_depth_top(depth_resp)
        if parsed is not None:
            bid, ask, quote_time = parsed
            accepted, provenance, rejection_reason = _quote_acceptance(quote_time)
            exchange_time_ms = provenance.get("quote_exchange_time_ms")
            if not accepted:
                log.warning("[EXEC_QUOTE_REJECT] %s | depth | %s", bx, rejection_reason)
            else:
                log.warning("[EXEC_QUOTE_FALLBACK] %s | bookTicker/ticker invalid -> depth", bx)
                return {
                    "status": "ok", "symbol": bx, "bid": bid, "ask": ask,
                    "spread_pct": ((ask - bid) / bid * 100.0) if bid > 0 else None,
                    "time": quote_time, "last_price": None, "quote_source": "depth",
                    "quote_freshness_status": provenance.get("quote_freshness_status"),
                    "quote_unknown_age_policy": provenance.get("quote_unknown_age_policy"),
                    "quote_sources_attempted": [source for source, _ in attempts],
                    "quote_fallback_reason": "; ".join(_quote_error(source, response, bx) for source, response in attempts[:-1]),
                    **provenance,
                }

    details = "; ".join(_quote_error(source, response, bx) for source, response in attempts)
    return {
        "status": "error",
        "error": f"BingX executable quote unavailable after bookTicker->ticker->depth | {details}",
        "symbol": bx,
        "quote_sources_attempted": [source for source, _ in attempts],
    }



def _research_endpoint_lock(endpoint_key: str) -> threading.Lock:
    with _RESEARCH_RATE_LOCK_GUARD:
        lock = _RESEARCH_RATE_LOCKS.get(endpoint_key)
        if lock is None:
            lock = threading.Lock()
            _RESEARCH_RATE_LOCKS[endpoint_key] = lock
        return lock


def _research_public_get(endpoint_key: str, path: str, params: dict[str, Any], *, timeout_sec: float = 3.0) -> dict[str, Any]:
    """Rate-limit each public research endpoint independently to the documented BingX rate."""
    try:
        interval = max(1.01, float(os.environ.get("RESEARCH_BINGX_ENDPOINT_INTERVAL_SEC", "1.05")))
    except (TypeError, ValueError):
        interval = 1.05
    lock = _research_endpoint_lock(endpoint_key)
    with lock:
        wait = interval - (time.monotonic() - _RESEARCH_RATE_LAST_TS.get(endpoint_key, 0.0))
        if wait > 0:
            time.sleep(wait)
        try:
            resp = _request("GET", path, params, signed=False, timeout_sec=timeout_sec, retryable=False)
        finally:
            _RESEARCH_RATE_LAST_TS[endpoint_key] = time.monotonic()
    return resp if isinstance(resp, dict) else {"code": -1, "msg": "invalid response type"}


def _research_symbol_row(data: Any, symbol: str) -> dict[str, Any] | None:
    bx = str(symbol).upper()
    if isinstance(data, dict):
        if str(data.get("symbol", "")).upper() in {"", bx}:
            return data
        return None
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict) and str(row.get("symbol", "")).upper() == bx:
                return row
        if len(data) == 1 and isinstance(data[0], dict):
            return data[0]
    return None


def _research_depth_metrics(data: Any, *, mid_price: float | None = None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"status": "error", "error": "depth_data_invalid"}
    bids = data.get("bids") or []
    asks = data.get("asks") or []

    def parse_level(level: Any) -> tuple[float, float] | None:
        try:
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                price, qty = float(level[0]), float(level[1])
            elif isinstance(level, dict):
                price = float(level.get("price") or level.get("p"))
                qty = float(level.get("quantity") or level.get("qty") or level.get("q"))
            else:
                return None
            if not (math.isfinite(price) and math.isfinite(qty)) or price <= 0 or qty < 0:
                return None
            return price, qty
        except (TypeError, ValueError):
            return None

    # Normalize provider payload order before selecting top-N.
    bp = sorted((x for x in (parse_level(v) for v in bids) if x), key=lambda x: x[0], reverse=True)
    ap = sorted((x for x in (parse_level(v) for v in asks) if x), key=lambda x: x[0])
    best_bid = bp[0][0] if bp else None
    best_ask = ap[0][0] if ap else None
    book_mid = ((best_bid + best_ask) / 2.0) if best_bid and best_ask else None
    depth_reference_type = "MARK_PRICE" if mid_price is not None and float(mid_price) > 0 else "BOOK_MID"
    mid = float(mid_price) if mid_price is not None and float(mid_price) > 0 else book_mid

    def depth_quote(levels: list[tuple[float, float]], pct: float) -> float:
        if mid is None or mid <= 0:
            return 0.0
        lo = mid * (1.0 - pct / 100.0)
        hi = mid * (1.0 + pct / 100.0)
        return sum(price * qty for price, qty in levels if lo <= price <= hi)

    bid_qty_5 = sum(q for _, q in bp[:5])
    ask_qty_5 = sum(q for _, q in ap[:5])
    bid_qty_10 = sum(q for _, q in bp[:10])
    ask_qty_10 = sum(q for _, q in ap[:10])
    total5 = bid_qty_5 + ask_qty_5
    total10 = bid_qty_10 + ask_qty_10
    bid_quote_5 = sum(price * qty for price, qty in bp[:5])
    ask_quote_5 = sum(price * qty for price, qty in ap[:5])
    bid_quote_10 = sum(price * qty for price, qty in bp[:10])
    ask_quote_10 = sum(price * qty for price, qty in ap[:10])
    quote_total5 = bid_quote_5 + ask_quote_5
    quote_total10 = bid_quote_10 + ask_quote_10
    microprice = None
    if best_bid is not None and best_ask is not None and bp and ap:
        if (bp[0][1] + ap[0][1]) > 0:
            microprice = ((best_ask * bp[0][1]) + (best_bid * ap[0][1])) / (bp[0][1] + ap[0][1])
    spread_pct = ((best_ask - best_bid) / ((best_ask + best_bid) / 2.0) * 100.0) if best_bid and best_ask and best_bid > 0 else None
    microprice_distance_pct = (microprice / book_mid - 1.0) * 100.0 if microprice is not None and book_mid and book_mid > 0 else None
    raw_ts = data.get("T") or data.get("timestamp") or data.get("time")
    try:
        book_timestamp_ms = int(float(raw_ts)) if raw_ts is not None else None
    except (TypeError, ValueError):
        book_timestamp_ms = None
    return {
        "status": "ok",
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_pct": spread_pct,
        "mid_price": mid,
        "book_mid_price": book_mid,
        "depth_reference_price": mid,
        "depth_reference_type": depth_reference_type,
        "bid_qty_5": bid_qty_5,
        "ask_qty_5": ask_qty_5,
        "bid_qty_10": bid_qty_10,
        "ask_qty_10": ask_qty_10,
        "book_imbalance_5": ((bid_qty_5 - ask_qty_5) / total5) if total5 > 0 else None,
        "book_imbalance_10": ((bid_qty_10 - ask_qty_10) / total10) if total10 > 0 else None,
        "bid_quote_5": bid_quote_5,
        "ask_quote_5": ask_quote_5,
        "bid_quote_10": bid_quote_10,
        "ask_quote_10": ask_quote_10,
        "book_quote_imbalance_5": ((bid_quote_5 - ask_quote_5) / quote_total5) if quote_total5 > 0 else None,
        "book_quote_imbalance_10": ((bid_quote_10 - ask_quote_10) / quote_total10) if quote_total10 > 0 else None,
        "microprice": microprice,
        "microprice_distance_pct": microprice_distance_pct,
        "bid_depth_quote_0_1pct": depth_quote(bp, 0.1),
        "ask_depth_quote_0_1pct": depth_quote(ap, 0.1),
        "bid_depth_quote_0_5pct": depth_quote(bp, 0.5),
        "ask_depth_quote_0_5pct": depth_quote(ap, 0.5),
        "bid_depth_quote_1pct": depth_quote(bp, 1.0),
        "ask_depth_quote_1pct": depth_quote(ap, 1.0),
        "bids_top10": [[p, q] for p, q in bp[:10]],
        "asks_top10": [[p, q] for p, q in ap[:10]],
        "timestamp": raw_ts,
        "timestamp_ms": book_timestamp_ms,
    }


def _research_sanitize(value: Any) -> Any:
    """Convert optional research payload values to strict JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(k): _research_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_research_sanitize(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return _research_sanitize(value.item())
        except Exception:
            pass
    return value


def _research_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _research_trade_metrics(data: Any) -> dict[str, Any]:
    if not isinstance(data, list):
        return {"status": "error", "error": "trades_data_invalid"}
    buy_quote = sell_quote = 0.0
    buy_count = sell_count = 0
    buyer_maker_present_count = buyer_maker_missing_count = 0
    prices: list[float] = []
    latest_ts = None
    earliest_ts = None
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            price = float(row.get("price"))
            raw_quote = row.get("quoteQty") or row.get("quote_qty")
            quote_qty = float(raw_quote) if raw_quote is not None else (price * float(row.get("qty")))
            ts = int(row.get("time")) if row.get("time") is not None else None
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(price) and price > 0 and math.isfinite(quote_qty) and quote_qty >= 0):
            continue
        prices.append(price)
        buyer_maker_raw = row.get("buyerMaker") if "buyerMaker" in row else None
        if buyer_maker_raw is None:
            buyer_maker_missing_count += 1
            # Missing maker/taker direction is unknown. Never classify it as BUY.
            continue
        buyer_maker_present_count += 1
        maker_buyer = _research_bool(buyer_maker_raw)
        # buyerMaker=true means the buyer was the maker; the aggressor was the seller.
        if maker_buyer:
            sell_quote += quote_qty
            sell_count += 1
        else:
            buy_quote += quote_qty
            buy_count += 1
        if ts is not None:
            latest_ts = max(latest_ts or ts, ts)
            earliest_ts = min(earliest_ts or ts, ts)
    total = buy_quote + sell_quote
    return {
        "status": "ok",
        "sample_count": len(data),
        "valid_trade_count": len(prices),
        "buy_aggressor_quote": buy_quote,
        "sell_aggressor_quote": sell_quote,
        "buy_aggressor_count": buy_count,
        "sell_aggressor_count": sell_count,
        "buyer_maker_field_present_count": buyer_maker_present_count,
        "buyer_maker_field_missing_count": buyer_maker_missing_count,
        "buyer_maker_field_coverage": (buyer_maker_present_count / len(prices)) if prices else None,
        "aggressor_validity": "valid" if buyer_maker_present_count > 0 and buyer_maker_missing_count == 0 else ("partial" if buyer_maker_present_count > 0 else "unknown"),
        "aggressor_delta_quote": (buy_quote - sell_quote) if buyer_maker_present_count > 0 else None,
        "buy_aggressor_ratio": (buy_quote / total) if total > 0 and buyer_maker_present_count > 0 else None,
        "trade_min_price": min(prices) if prices else None,
        "trade_max_price": max(prices) if prices else None,
        "last_trade_ts": latest_ts,
        "first_trade_ts": earliest_ts,
        "sample_span_seconds": ((latest_ts - earliest_ts) / 1000.0) if latest_ts is not None and earliest_ts is not None and latest_ts >= earliest_ts else None,
        "avg_trade_quote": (total / len(prices)) if prices else None,
    }


def fetch_research_market_context(
    symbol: str,
    *,
    depth_limit: int = 20,
    trades_limit: int = 100,
    timeout_sec: float = 3.0,
) -> dict[str, Any]:
    """Collect a decision-time BingX market snapshot for research only.

    This is deliberately best-effort: failure of one optional public context endpoint
    must never block the production signal/execution path. The returned record contains
    enough metadata to distinguish missing context from a real zero measurement.
    """
    bx = to_bx_symbol(symbol)
    captured_ms = int(time.time() * 1000)
    collection_started = time.monotonic()
    result: dict[str, Any] = {
        "schema": "bingx_research_market_context_v2",
        "context_schema_version": 2,
        "symbol": str(symbol).upper(),
        "bingx_symbol": bx,
        "provider": "bingx",
        "context_provider": "bingx",
        "context_source": "bingx_swap_public",
        "captured_at_ms": captured_ms,
        "captured_at": datetime.fromtimestamp(captured_ms / 1000.0, tz=timezone.utc).isoformat(),
        "capture_started_at_ms": captured_ms,
        "capture_phase": "PRE_EXECUTION",
        "feature_time_semantics": "PRE_EXECUTION_CONTEXT_NOT_TRIGGER_BOUNDARY",
        "status": "ok",
        "errors": [],
        "endpoint_status": {},
        "endpoint_latency_ms": {},
    }
    if not bx:
        result["status"] = "error"
        result["errors"].append("contract_not_found")
        return result

    def call(key: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        try:
            resp = _research_public_get(key, path, params, timeout_sec=max(0.5, float(timeout_sec)))
            result["endpoint_latency_ms"][key] = round((time.monotonic() - started) * 1000.0, 3)
            if int(resp.get("code")) != 0:
                result["endpoint_status"][key] = "error"
                result["errors"].append(f"{key}:code={resp.get('code')}:msg={resp.get('msg')}")
            else:
                result["endpoint_status"][key] = "ok"
            return resp
        except Exception as exc:
            result["endpoint_latency_ms"][key] = round((time.monotonic() - started) * 1000.0, 3)
            result["endpoint_status"][key] = "error"
            result["errors"].append(f"{key}:{type(exc).__name__}:{exc}")
            return {"code": -1, "msg": _safe_exception_message(exc)}

    premium = call("premiumIndex", PREMIUM_INDEX_PATH, {"symbol": bx})
    prow = _research_symbol_row(premium.get("data"), bx) if isinstance(premium, dict) else None
    if prow:
        for source_key, out_key in (("markPrice", "mark_price"), ("indexPrice", "index_price"), ("lastFundingRate", "funding_rate"), ("nextFundingTime", "next_funding_time_ms"), ("time", "premium_index_ts")):
            if prow.get(source_key) is not None:
                try:
                    result[out_key] = float(prow[source_key]) if source_key not in {"nextFundingTime", "time"} else int(prow[source_key])
                except (TypeError, ValueError):
                    result[out_key] = None
        result["funding_rate_unit"] = "DECIMAL_RATE"
        result["funding_rate_pct"] = (float(result["funding_rate"]) * 100.0) if result.get("funding_rate") is not None else None

    if result.get("mark_price") and result.get("index_price"):
        idx = float(result["index_price"])
        if idx > 0:
            result["mark_index_basis_pct"] = (float(result["mark_price"]) / idx - 1.0) * 100.0

    oi = call("openInterest", OPEN_INTEREST_PATH, {"symbol": bx})
    orow = _research_symbol_row(oi.get("data"), bx) if isinstance(oi, dict) else None
    if orow:
        raw_oi = orow.get("openInterest")
        try:
            result["open_interest"] = float(raw_oi)
        except (TypeError, ValueError):
            result["open_interest"] = None
        result["open_interest_raw"] = raw_oi
        result["open_interest_unit"] = str(orow.get("unit") or "PROVIDER_NATIVE_UNSPECIFIED")
        result["open_interest_ts"] = orow.get("time")

    book = call("depth", DEPTH_PATH, {"symbol": bx, "limit": int(depth_limit)})
    if isinstance(book, dict) and isinstance(book.get("data"), dict):
        result["order_book"] = _research_depth_metrics(book["data"], mid_price=result.get("mark_price"))
        result["order_book_timestamp"] = result["order_book"].get("timestamp")

    trades = call("trades", TRADES_PATH, {"symbol": bx, "limit": int(trades_limit)})
    result["recent_trades"] = _research_trade_metrics(trades.get("data")) if isinstance(trades, dict) else {"status": "error", "error": "trades_response_invalid"}

    if result["errors"]:
        result["status"] = "partial" if len(result["errors"]) < 4 else "error"
    completed_ms = int(time.time() * 1000)
    result["capture_completed_at_ms"] = completed_ms
    result["capture_completed_at"] = datetime.fromtimestamp(completed_ms / 1000.0, tz=timezone.utc).isoformat()
    result["context_capture_span_ms"] = max(0, completed_ms - captured_ms)
    result["collection_latency_ms"] = round((time.monotonic() - collection_started) * 1000.0, 3)
    return _research_sanitize(result)


def fetch_research_account_snapshot() -> dict[str, Any]:
    """Collect a read-only BingX futures account snapshot for research/risk attribution.

    This is called at most once per scan, never from the per-symbol worker pool.
    No order execution depends on this snapshot.
    """
    captured_ms = int(time.time() * 1000)
    result: dict[str, Any] = {
        "schema": "bingx_research_account_context_v1",
        "captured_at_ms": captured_ms,
        "captured_at": datetime.fromtimestamp(captured_ms / 1000.0, tz=timezone.utc).isoformat(),
        "status": "ok",
        "errors": [],
        "account_realized_profit_source": "BALANCE_TOTAL",
        "recent_realized_pnl_total": None,
        "recent_realized_pnl_source": "UNAVAILABLE",
        "recent_fill_source": "UNAVAILABLE",
    }
    api_key, secret_key = get_credentials()
    if not api_key or not secret_key:
        result["status"] = "unavailable"
        result["errors"].append("missing_credentials")
        return result
    try:
        balance = _request("GET", BALANCE_PATH, {}, signed=True, timeout_sec=min(5.0, float(os.environ.get("RESEARCH_ACCOUNT_TIMEOUT_SEC", "5"))), retryable=False)
        if not isinstance(balance, dict) or int(balance.get("code", -1)) != 0:
            result["status"] = "partial"
            result["errors"].append(f"balance:code={balance.get('code') if isinstance(balance, dict) else 'invalid'}:msg={balance.get('msg') if isinstance(balance, dict) else 'invalid_response'}")
        else:
            rows = balance.get("data")
            row = rows[0] if isinstance(rows, list) and rows else rows if isinstance(rows, dict) else None
            if isinstance(row, dict):
                for src, dst in (("asset", "asset"), ("balance", "balance"), ("equity", "equity"), ("unrealizedProfit", "unrealized_profit"), ("realisedProfit", "realized_profit"), ("realizedProfit", "realized_profit"), ("availableMargin", "available_margin"), ("usedMargin", "used_margin"), ("freezedMargin", "freezed_margin")):
                    value = row.get(src)
                    if value is None:
                        continue
                    if dst == "asset":
                        result[dst] = str(value)
                    else:
                        try:
                            num = float(value)
                            result[dst] = num if math.isfinite(num) else None
                        except (TypeError, ValueError):
                            result[dst] = None
            else:
                result["status"] = "partial"
                result["errors"].append("balance:payload_missing")
    except Exception as exc:
        result["status"] = "partial"
        result["errors"].append(f"balance:{type(exc).__name__}:{exc}")

    activity_lookback_min = max(5, min(60, int(os.environ.get("RESEARCH_ACCOUNT_ACTIVITY_LOOKBACK_MINUTES", "15"))))
    activity_start_ms = max(0, captured_ms - activity_lookback_min * 60 * 1000)
    activity_end_ms = captured_ms
    try:
        fills = _request(
            "GET", ALL_FILL_ORDERS_PATH,
            {"tradingUnit": "CONT", "startTs": activity_start_ms, "endTs": activity_end_ms, "currency": "USDT"},
            signed=True, timeout_sec=min(5.0, float(os.environ.get("RESEARCH_ACCOUNT_TIMEOUT_SEC", "5"))), retryable=False,
        )
        if isinstance(fills, dict) and int(fills.get("code", -1)) == 0:
            raw_data = fills.get("data")
            raw_rows = _normalize_all_fill_orders_data(raw_data)

            if raw_rows is None:
                # Unknown response shape is not equivalent to an empty trading
                # window. Keep financial totals unknown so INCOME can be used as
                # a fallback instead of emitting false zeros. Include only payload
                # keys (never values) to make a future schema mismatch diagnosable.
                result["recent_fill_status"] = "UNRECOGNIZED_RESPONSE_SHAPE"
                result["recent_fill_error"] = f"unrecognized allFillOrders data shape: {_fill_payload_shape(raw_data)}"
                result["recent_fill_count"] = None
                result["recent_fill_financial_status"] = "UNAVAILABLE"
                result["recent_fill_fee_status"] = "UNAVAILABLE"
                result["recent_fill_realized_pnl_status"] = "UNAVAILABLE"
                result["recent_fill_fee_total"] = None
                result["recent_fill_realized_pnl_total"] = None
                result["recent_realized_pnl_total"] = None
                result["recent_realized_pnl_source"] = "UNAVAILABLE"
                result["recent_fill_source"] = "UNAVAILABLE"
                result.setdefault("errors", []).append(result["recent_fill_error"])
                raw_rows = []
                skip_fill_financial_classification = True
            else:
                skip_fill_financial_classification = False
            activity_rows = []
            fee_values: list[float] = []
            realized_values: list[float] = []
            for row in raw_rows:
                if not isinstance(row, dict):
                    continue
                item = _normalize_fill_row(row)
                if item.get("fee") is not None:
                    fee_values.append(float(item["fee"]))
                if item.get("realized_pnl") is not None:
                    realized_values.append(float(item["realized_pnl"]))
                activity_rows.append(item)
            fill_count = len(activity_rows)
            fees_complete = len(fee_values) == fill_count
            realized_complete = len(realized_values) == fill_count
            result["recent_fill_window_start_ms"] = activity_start_ms
            result["recent_fill_window_end_ms"] = activity_end_ms
            result["recent_fills"] = activity_rows
            if skip_fill_financial_classification:
                # The empty list above is only a parser sentinel; it is not proof
                # of an empty exchange activity window.
                pass
            else:
                result["recent_fill_count"] = fill_count
                result["recent_fill_status"] = "EMPTY" if fill_count == 0 else "OK"
            if skip_fill_financial_classification:
                pass
            elif fill_count == 0:
                result["recent_fill_financial_status"] = "NO_FILLS"
                result["recent_fill_fee_status"] = "NO_FILLS"
                result["recent_fill_realized_pnl_status"] = "NO_FILLS"
                result["recent_fill_fee_total"] = 0.0
                result["recent_fill_realized_pnl_total"] = 0.0
                result["recent_realized_pnl_total"] = 0.0
                result["recent_realized_pnl_source"] = "ALL_FILL_ORDERS_EMPTY"
                result["recent_fill_source"] = "ALL_FILL_ORDERS_EMPTY"
            else:
                result["recent_fill_fee_status"] = "CONFIRMED" if fees_complete else "PARTIAL_MISSING_FIELD"
                result["recent_fill_realized_pnl_status"] = "CONFIRMED" if realized_complete else "PARTIAL_MISSING_FIELD"
                result["recent_fill_fee_total"] = sum(fee_values) if fees_complete else None
                result["recent_fill_realized_pnl_total"] = sum(realized_values) if realized_complete else None
                if fees_complete and realized_complete:
                    result["recent_fill_financial_status"] = "CONFIRMED"
                    result["recent_realized_pnl_total"] = sum(realized_values)
                    result["recent_realized_pnl_source"] = "ALL_FILL_ORDERS"
                else:
                    result["recent_fill_financial_status"] = "PARTIAL_MISSING_FIELDS"
                    # A complete realized-PnL field remains authoritative even when
                    # fee fields are incomplete. Do not replace it with a different
                    # source merely because the combined financial row is partial.
                    result["recent_realized_pnl_total"] = sum(realized_values) if realized_complete else None
                    result["recent_realized_pnl_source"] = "ALL_FILL_ORDERS_PARTIAL_MISSING"
                result["recent_fill_source"] = "ALL_FILL_ORDERS"
        else:
            response_code = fills.get("code") if isinstance(fills, dict) else "invalid"
            response_message = str(fills.get("msg") or "").replace("\r", " ").replace("\n", " ").strip()[:240] if isinstance(fills, dict) else ""
            fill_error = f"code={response_code}"
            if response_message:
                fill_error += f" msg={response_message}"
            result["recent_fill_status"] = "error"
            result["recent_fill_error"] = fill_error
            result.setdefault("errors", []).append(f"fills:{fill_error}")
    except Exception as exc:
        result["recent_fill_status"] = "error"
        result["recent_fill_error"] = f"{type(exc).__name__}:{exc}"
        result.setdefault("errors", []).append(f"fills:{type(exc).__name__}:{exc}")

    try:
        income = _request(
            "GET", INCOME_PATH,
            {"startTime": activity_start_ms, "endTime": activity_end_ms, "limit": 1000},
            signed=True, timeout_sec=min(5.0, float(os.environ.get("RESEARCH_ACCOUNT_TIMEOUT_SEC", "5"))), retryable=False,
        )
        if isinstance(income, dict) and int(income.get("code", -1)) == 0:
            raw_rows = income.get("data") if isinstance(income.get("data"), list) else []
            income_rows = []
            totals: dict[str, float] = {}
            for row in raw_rows:
                if not isinstance(row, dict):
                    continue
                item: dict[str, Any] = {}
                for src, dst in (("symbol", "symbol"), ("incomeType", "income_type"), ("asset", "asset"), ("info", "info"), ("tranId", "tran_id"), ("tradeId", "trade_id")):
                    if row.get(src) not in (None, ""):
                        item[dst] = str(row.get(src))
                for src, dst in (("income", "income"), ("time", "time_ms")):
                    if row.get(src) in (None, ""):
                        continue
                    try:
                        val = float(row.get(src))
                        if dst == "time_ms":
                            val = int(val)
                        item[dst] = val if math.isfinite(float(val)) else None
                    except (TypeError, ValueError):
                        item[dst] = None
                income_type = str(item.get("income_type") or "UNKNOWN")
                value = item.get("income")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    totals[income_type] = totals.get(income_type, 0.0) + float(value)
                income_rows.append(item)
            result["recent_income"] = income_rows
            result["recent_income_count"] = len(income_rows)
            result["recent_income_totals"] = totals
            if result.get("recent_realized_pnl_total") is None and result.get("recent_realized_pnl_source") in {"UNAVAILABLE", "ALL_FILL_ORDERS_EMPTY", "ALL_FILL_ORDERS_PARTIAL_MISSING"}:
                realized_from_income = totals.get("REALIZED_PNL")
                if isinstance(realized_from_income, (int, float)) and math.isfinite(float(realized_from_income)):
                    result["recent_realized_pnl_total"] = float(realized_from_income)
                    result["recent_realized_pnl_source"] = "INCOME_REALIZED_PNL_FALLBACK"
            if result.get("recent_realized_pnl_source") == "ALL_FILL_ORDERS":
                result["recent_fill_source"] = "ALL_FILL_ORDERS"
            elif result.get("recent_realized_pnl_source") == "INCOME_REALIZED_PNL_FALLBACK":
                result["recent_fill_source"] = "INCOME_FALLBACK"
        else:
            result.setdefault("errors", []).append(f"income:code={income.get('code') if isinstance(income, dict) else 'invalid'}")
    except Exception as exc:
        result.setdefault("errors", []).append(f"income:{type(exc).__name__}:{exc}")

    try:
        forced = _request(
            "GET", FORCE_ORDERS_PATH,
            {"currency": "USDT", "startTime": activity_start_ms, "endTime": activity_end_ms, "limit": 100},
            signed=True, timeout_sec=min(5.0, float(os.environ.get("RESEARCH_ACCOUNT_TIMEOUT_SEC", "5"))), retryable=False,
        )
        if isinstance(forced, dict) and int(forced.get("code", -1)) == 0:
            raw_rows = forced.get("data") if isinstance(forced.get("data"), list) else []
            force_rows = []
            liq_n = adl_n = 0
            for row in raw_rows:
                if not isinstance(row, dict):
                    continue
                item: dict[str, Any] = {}
                for src, dst in (("symbol", "symbol"), ("side", "side"), ("positionSide", "position_side"), ("autoCloseType", "auto_close_type"), ("orderId", "order_id"), ("time", "time_ms")):
                    if row.get(src) not in (None, ""):
                        item[dst] = str(row.get(src)) if dst != "time_ms" else int(float(row.get(src)))
                for src, dst in (("price", "price"), ("origQty", "qty"), ("avgPrice", "avg_price")):
                    if row.get(src) in (None, ""):
                        continue
                    try:
                        val = float(row.get(src))
                        item[dst] = val if math.isfinite(val) else None
                    except (TypeError, ValueError):
                        item[dst] = None
                kind = str(item.get("auto_close_type") or "").upper()
                liq_n += kind == "LIQUIDATION"
                adl_n += kind == "ADL"
                force_rows.append(item)
            result["recent_force_orders"] = force_rows
            result["recent_force_order_count"] = len(force_rows)
            result["recent_liquidation_count"] = liq_n
            result["recent_adl_count"] = adl_n
        else:
            result.setdefault("errors", []).append(f"force_orders:code={forced.get('code') if isinstance(forced, dict) else 'invalid'}")
    except Exception as exc:
        result.setdefault("errors", []).append(f"force_orders:{type(exc).__name__}:{exc}")

    try:
        positions = _request("GET", POSITION_PATH, {}, signed=True, timeout_sec=min(5.0, float(os.environ.get("RESEARCH_ACCOUNT_TIMEOUT_SEC", "5"))), retryable=False)
        if isinstance(positions, dict) and int(positions.get("code", -1)) == 0:
            rows = _normalize_positions_list(positions)
            position_rows = []
            total_notional = 0.0
            total_unrealized = 0.0
            long_n = short_n = 0
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item: dict[str, Any] = {}
                for src, dst in (("symbol", "symbol"), ("positionSide", "position_side"), ("side", "side")):
                    if row.get(src) not in (None, ""):
                        item[dst] = str(row.get(src))
                for src, dst in (("positionAmt", "position_amt"), ("positionAmtAbs", "position_amt_abs"), ("avgPrice", "entry_price"), ("entryPrice", "entry_price"), ("markPrice", "mark_price"), ("liquidationPrice", "liquidation_price"), ("leverage", "leverage"), ("unrealizedProfit", "unrealized_profit"), ("initialMargin", "initial_margin")):
                    if row.get(src) in (None, ""):
                        continue
                    try:
                        val = float(row.get(src))
                        item[dst] = val if math.isfinite(val) else None
                    except (TypeError, ValueError):
                        item[dst] = None
                if item.get("position_amt") is not None and item.get("mark_price") is not None:
                    item["notional_usdt"] = abs(float(item["position_amt"]) * float(item["mark_price"]))
                else:
                    item["notional_usdt"] = None
                if item.get("position_amt") is not None and abs(float(item["position_amt"])) <= 0:
                    continue
                if item.get("notional_usdt") is not None:
                    total_notional += float(item["notional_usdt"])
                if item.get("unrealized_profit") is not None:
                    total_unrealized += float(item["unrealized_profit"])
                direction = str(item.get("side") or item.get("position_side") or "").upper()
                if direction in {"LONG", "BUY"}:
                    long_n += 1
                elif direction in {"SHORT", "SELL"}:
                    short_n += 1
                position_rows.append(item)
            result["positions"] = position_rows
            result["open_positions_count"] = len(position_rows)
            result["long_positions_count"] = long_n
            result["short_positions_count"] = short_n
            result["open_positions_notional_usdt"] = total_notional
            result["open_positions_unrealized_profit"] = total_unrealized
        else:
            result.setdefault("errors", []).append(f"positions:code={positions.get('code') if isinstance(positions, dict) else 'invalid'}")
    except Exception as exc:
        result.setdefault("errors", []).append(f"positions:{type(exc).__name__}:{exc}")

    try:
        commission = _request("GET", COMMISSION_RATE_PATH, {}, signed=True, timeout_sec=min(5.0, float(os.environ.get("RESEARCH_ACCOUNT_TIMEOUT_SEC", "5"))), retryable=False)
        if isinstance(commission, dict) and int(commission.get("code", -1)) == 0:
            row = commission.get("data") if isinstance(commission.get("data"), dict) else commission.get("data", {})
            if isinstance(row, dict):
                payload = row.get("commission") if isinstance(row.get("commission"), dict) else row
                for src, dst in (("takerCommissionRate", "taker_commission_rate"), ("makerCommissionRate", "maker_commission_rate")):
                    try:
                        val = float(payload.get(src))
                        result[dst] = val if math.isfinite(val) else None
                    except (TypeError, ValueError):
                        result[dst] = None
        else:
            result.setdefault("errors", []).append(f"commission:code={commission.get('code') if isinstance(commission, dict) else 'invalid'}")
    except Exception as exc:
        result["errors"].append(f"commission:{type(exc).__name__}:{exc}")

    if result["errors"] and result["status"] == "ok":
        result["status"] = "partial"
    return _research_sanitize(result)


def _current_close_price(symbol: str) -> float | None:
    try:
        rows = fetch_klines(symbol, "1m", limit=2)
    except Exception as exc:
        log.warning("[BINGX] Failed to read current 1m price for %s: %s", symbol, exc)
        return None

    if not rows:
        return None
    try:
        price = float(rows[-1].get("close", 0) or 0)
    except (TypeError, ValueError):
        return None

    return price if (math.isfinite(price) and price > 0) else None



def _qty_matches_position(order_qty: float, position_qty: float) -> bool:
    if order_qty <= 0 or position_qty <= 0:
        return False
    return abs(order_qty - position_qty) <= max(position_qty * 1e-6, 1e-12)


def _effective_weighted_rr(levels: list[dict], stop_loss_pct: float) -> float | None:
    if stop_loss_pct <= 0 or not levels:
        return None
    total = 0.0
    weighted = 0.0
    for level in levels:
        try:
            pnl = float(level.get("pnl_pct", 0))
            weight = float(level.get("qty", 0) or level.get("close_fraction", 0))
        except (TypeError, ValueError):
            continue
        if pnl > 0 and weight > 0:
            total += weight
            weighted += weight * (pnl / stop_loss_pct)
    return weighted / total if total > 0 else None



def _order_qty(order: dict) -> float:
    try:
        return abs(float(order.get("origQty", 0) or order.get("quantity", 0) or 0))
    except (TypeError, ValueError):
        return 0.0


def _find_open_order_by_client_id(orders: list[dict], client_order_id: str) -> dict | None:
    target = str(client_order_id or "").upper()
    if not target:
        return None
    for order in orders:
        if str(order.get("clientOrderId", "")).upper() == target:
            return order
    return None


def _verify_open_order(
    symbol: str,
    direction: str,
    *,
    client_order_id: str,
    order_kind: str,
    expected_price: float,
    expected_qty: float,
    price_precision: int,
    max_attempts: int = 3,
) -> dict:
    """Reconcile an order after POST, including ambiguous network failures.

    We never retransmit the POST. Instead we poll openOrders and accept the
    already-created exchange order when its clientOrderId/side/price/qty match.
    """
    expected_kind = str(order_kind).upper()
    for attempt in range(max(1, max_attempts)):
        try:
            book = get_open_protection_directional(symbol, direction, retryable=False)
        except Exception as exc:
            book = {"status": "error", "error": str(exc)}
        if book.get("status") == "ok":
            orders = list(book.get("sl_orders", [])) + list(book.get("tp_orders", []))
            for order in orders:
                if str(order.get("clientOrderId", "")).upper() != str(client_order_id).upper():
                    continue
                qty = _order_qty(order)
                actual_price = float(order.get("stopPrice", 0) or order.get("price", 0) or 0)
                type_ok = str(order.get("type", "")).upper() == expected_kind
                expected_side = "SELL" if str(direction).upper() == "LONG" else "BUY"
                side_ok = str(order.get("side", "")).upper() == expected_side
                try:
                    expected_position_side = position_side_param(direction)
                except Exception:
                    expected_position_side = None
                position_side_ok = bool(expected_position_side) and str(order.get("positionSide", "")).upper() == expected_position_side
                price_ok = _format_price(actual_price, price_precision) == _format_price(expected_price, price_precision)
                qty_ok = _qty_matches_position(qty, expected_qty)
                if type_ok and side_ok and position_side_ok and price_ok and qty_ok:
                    return {"status": "verified", "order": order, "attempt": attempt + 1}
        if attempt + 1 < max(1, max_attempts):
            time.sleep(0.25 * (attempt + 1))
    return {"status": "not_found", "client_order_id": client_order_id}


def _verify_market_reduce_order(
    symbol: str,
    direction: str,
    order_id: str | None,
    expected_qty: float,
    pre_qty: float,
    attempts: int = 5,
) -> dict:
    """Verify a crossed-TP MARKET reduction from exchange state.

    A successful POST acknowledgement is not sufficient: the market order must
    show execution and/or the live position must decrease. This helper never
    retries the MARKET POST itself.
    """
    direction = str(direction).upper()
    expected_qty = max(0.0, float(expected_qty))
    pre_qty = max(0.0, float(pre_qty))
    last = {"status": "unverified", "executed_qty": 0.0, "remaining_qty": pre_qty}
    for attempt in range(max(1, attempts)):
        executed_qty = 0.0
        order_info = None
        if order_id:
            try:
                order_info = get_order(symbol, order_id)
            except Exception:
                order_info = None
            if isinstance(order_info, dict) and order_info.get("status") == "ok":
                try:
                    executed_qty = max(0.0, float(order_info.get("executed_qty", 0.0) or 0.0))
                except (TypeError, ValueError):
                    executed_qty = 0.0

        try:
            pos = get_position_directional(symbol, direction)
        except Exception as exc:
            pos = {"status": "error", "error": str(exc)}

        if pos.get("status") == "found":
            try:
                remaining_qty = max(0.0, float(pos.get("positionAmt", 0.0) or 0.0))
            except (TypeError, ValueError):
                remaining_qty = pre_qty
        elif pos.get("status") == "not_found":
            remaining_qty = 0.0
        else:
            remaining_qty = pre_qty

        reduced_qty = max(0.0, pre_qty - remaining_qty)
        last = {
            "status": "unverified",
            "executed_qty": executed_qty,
            "remaining_qty": remaining_qty,
            "reduced_qty": reduced_qty,
            "order": order_info,
        }

        # Require real evidence of execution. Either the order reports fills,
        # or the exchange position demonstrably shrank. For a position that
        # disappeared entirely, the residual check itself is authoritative.
        if executed_qty > 0 and (reduced_qty > 0 or remaining_qty <= 1e-12):
            last["status"] = "verified"
            return last
        if reduced_qty > 0 and (executed_qty > 0 or order_info is None):
            last["status"] = "verified"
            return last

        if attempt + 1 < max(1, attempts):
            time.sleep(0.2 * (attempt + 1))

    if expected_qty <= 0 and last.get("remaining_qty", pre_qty) <= 1e-12:
        last["status"] = "verified"
    return last


def ensure_directional_protection(
    symbol: str, direction: str, avg_price: float, qty: float,
    stop_loss_pct: float, tp_levels: list, trade_id: str | None = None,
) -> dict:
    direction = str(direction).upper()
    if direction not in {"LONG", "SHORT"}:
        return {"status": "error", "error": f"invalid direction={direction}"}

    try:
        avg_price = float(avg_price)
        qty = abs(float(qty))
        stop_loss_pct = float(stop_loss_pct)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc)}

    if not math.isfinite(avg_price) or not math.isfinite(qty) or not math.isfinite(stop_loss_pct) or avg_price <= 0 or qty <= 0 or not (0 < stop_loss_pct <= 25):
        return {"status": "error", "error": "invalid protection parameters"}

    bx_symbol = to_bx_symbol(symbol)
    contract = get_contract(symbol)
    if not bx_symbol or not contract:
        return {"status": "error", "error": f"contract not found: {bx_symbol}"}

    try:
        precision = int(contract.get("quantityPrecision") or 0)
        price_precision = int(contract.get("pricePrecision") or 4)
        min_qty = float(contract.get("tradeMinQuantity") or contract.get("minQty") or 0)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": f"invalid contract parameters: {exc}"}

    position_qty = _round_qty(qty, precision)
    if position_qty <= 0 or (min_qty > 0 and position_qty < min_qty):
        return {"status": "error", "error": f"qty={position_qty} < minQty={min_qty}"}

    existing = get_open_protection_directional(symbol, direction)
    if existing.get("status") != "ok":
        return {"status": "PROTECTION_FAILED", "error": existing.get("error", "openOrders unavailable")}

    existing_tp = list(existing.get("tp_orders", []))
    existing_sl = list(existing.get("sl_orders", []))
    tp_levels_norm = _normalize_tp_levels(tp_levels)
    if tp_levels and not tp_levels_norm:
        return {
            "status": "PROTECTION_FAILED",
            "symbol": symbol,
            "bx_symbol": bx_symbol,
            "direction": direction,
            "avg_price": avg_price,
            "qty": position_qty,
            "error": "invalid_nonempty_tp_configuration",
            "failure_class": "INVALID_TP_CONFIGURATION",
        }

    valid_existing_sl = None
    engine_owned_invalid_sl_ids: list[str] = []
    # A read-only reconciliation can trust the successful open-orders snapshot
    # already fetched above. A second read is required only after we mutate SL
    # state (creation or cleanup); otherwise it doubles private API latency for
    # every already-protected position on every scan cycle.
    sl_requery_required = False
    for sl in existing_sl:
        order_type = str(sl.get("type", "")).upper()
        if order_type not in {"STOP", "STOP_MARKET"}:
            continue
        try:
            sl_price = float(sl.get("stopPrice", 0) or sl.get("price", 0) or 0)
            sl_qty = float(sl.get("origQty", 0) or sl.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            continue

        if sl_price <= 0 or sl_qty <= 0:
            continue

        expected_side = "SELL" if direction == "LONG" else "BUY"
        side_matches = str(sl.get("side", "")).upper() == expected_side
        try:
            expected_position_side = position_side_param(direction)
        except Exception:
            expected_position_side = None
        position_side_matches = bool(expected_position_side) and str(sl.get("positionSide", "")).upper() == expected_position_side
        protective_side = (sl_price <= avg_price) if direction == "LONG" else (sl_price >= avg_price)
        qty_matches = _qty_matches_position(sl_qty, position_qty)
        client_id = str(sl.get("clientOrderId", "")).upper()
        engine_owned = client_id.startswith("EVT_")
        if protective_side and side_matches and position_side_matches and qty_matches and engine_owned and valid_existing_sl is None:
            valid_existing_sl = sl
            continue

        # A stale/wrong-side/order-size-mismatched SL created by this engine
        # must never remain alongside a newly-created protective SL. Manual
        # orders are intentionally left untouched; the engine only cleans its
        # own EVT_* protection orders.
        order_id = str(sl.get("orderId", ""))
        if order_id and engine_owned:
            engine_owned_invalid_sl_ids.append(order_id)

    if engine_owned_invalid_sl_ids:
        # Keep the old engine SL as a safety net until a valid replacement has
        # been created and independently observed on the exchange.
        sl_requery_required = True

    if valid_existing_sl is not None:
        sl = valid_existing_sl
        sl_result = {
            "status": "already_exists",
            "order_id": str(sl.get("orderId", "")),
            "client_order_id": str(sl.get("clientOrderId", "")),
            "stop_price": float(sl.get("stopPrice", 0) or sl.get("price", 0) or 0),
            "qty": float(sl.get("origQty", 0) or sl.get("quantity", 0) or position_qty),
            "side": str(sl.get("side", "")).upper(),
            "positionSide": str(sl.get("positionSide", "")).upper(),
            "protection_order_verified": True,
            "execution_verified": None,
        }
    else:
        sl_requery_required = True
        sl_price = avg_price * (1.0 - stop_loss_pct / 100.0) if direction == "LONG" else avg_price * (1.0 + stop_loss_pct / 100.0)
        base_client_order_id = build_sl_client_order_id(trade_id)
        # Keep the old invalid SL alive until its replacement is visible. If an
        # engine-owned invalid SL already exists, its deterministic client ID may
        # still be occupied at BingX; use a unique repair ID for the replacement.
        client_order_id = (
            _repair_client_order_id(base_client_order_id)
            if engine_owned_invalid_sl_ids else base_client_order_id
        )
        current_price = _current_close_price(symbol)
        if current_price is not None:
            sl_side_valid = (sl_price < current_price) if direction == "LONG" else (sl_price > current_price)
            if not sl_side_valid:
                return {
                    "status": "PROTECTION_FAILED",
                    "symbol": symbol,
                    "direction": direction,
                    "avg_price": avg_price,
                    "qty": position_qty,
                    "error": f"stop_price_crossed_before_post: sl={sl_price} current={current_price}",
                    "current_price": current_price,
                }
        params = {
            "symbol": bx_symbol,
            "side": "SELL" if direction == "LONG" else "BUY",
            "positionSide": position_side_param(direction),
            "type": "STOP_MARKET",
            "stopPrice": _format_price(sl_price, price_precision),
            "quantity": _format_qty(position_qty, precision),
            "clientOrderId": client_order_id,
        }

        resp = _request("POST", ORDER_PATH, params)
        order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
        if resp.get("code") != 0:
            verify = _verify_open_order(
                symbol, direction, client_order_id=client_order_id, order_kind="STOP_MARKET",
                expected_price=sl_price, expected_qty=position_qty, price_precision=price_precision,
            )
            if verify.get("status") != "verified":
                log.error("[BINGX] SL failed/unverified: code=%s msg=%s", resp.get("code"), resp.get("msg"))
                return {
                    "status": "PROTECTION_FAILED",
                    "error": f"SL failed: {resp.get('msg')}",
                    "sl_result": {"status": "error", "error": resp.get("msg")},
                    "tp_orders": [],
                }
            order = verify["order"]
            sl_result = {
                "status": "reconciled_after_post_error",
                "order_id": str(order.get("orderId", "")),
                "client_order_id": order.get("clientOrderId") or client_order_id,
                "stop_price": float(order.get("stopPrice", 0) or order.get("price", 0) or sl_price),
                "qty": _order_qty(order),
                "side": str(order.get("side", "")).upper(),
                "positionSide": str(order.get("positionSide", "")).upper(),
                "protection_order_verified": True,
                "execution_verified": None,
            }
        else:
            sl_result = {
                "status": "created",
                "order_id": str(order.get("orderId", "")),
                "client_order_id": order.get("clientOrderId") or client_order_id,
                "stop_price": sl_price,
                "qty": position_qty,
                "side": "SELL" if direction == "LONG" else "BUY",
                "positionSide": position_side_param(direction),
                "protection_order_verified": True,
                "execution_verified": None,
            }

    if sl_requery_required:
        verified = get_open_protection_directional(symbol, direction)
        verified_sl = list(verified.get("sl_orders", [])) if verified.get("status") == "ok" else []
        verified_sl_valid = any(_validate_sl_order_for_position(o, direction, avg_price, position_qty) for o in verified_sl)
    else:
        # No SL write/cancel occurred after the original successful exchange
        # snapshot. Revalidating the exact observed order locally is equivalent
        # for this pass and avoids a redundant openOrders HTTP request.
        verified_sl_valid = bool(
            isinstance(valid_existing_sl, dict)
            and _validate_sl_order_for_position(valid_existing_sl, direction, avg_price, position_qty)
        )

    if not verified_sl_valid:
        return {
            "status": "SL_UNVERIFIED",
            "symbol": symbol,
            "bx_symbol": bx_symbol,
            "direction": direction,
            "avg_price": avg_price,
            "qty": position_qty,
            "sl_result": sl_result,
            "tp_orders": [],
            "error": "SL created but not visible on exchange",
        }

    # Only now that a valid SL is exchange-visible may stale engine-owned SLs
    # be cancelled. If cleanup fails, return an explicit failure rather than
    # silently leaving ambiguous duplicate protection behind.
    if engine_owned_invalid_sl_ids:
        cleanup_ok, cleanup_error = _cancel_engine_protection_orders_verified(
            symbol, direction, engine_owned_invalid_sl_ids,
        )
        if not cleanup_ok:
            return {
                "status": "SL_UNVERIFIED",
                "symbol": symbol,
                "direction": direction,
                "avg_price": avg_price,
                "qty": position_qty,
                "sl_result": sl_result,
                "tp_orders": [],
                "error": f"engine SL cleanup could not be verified: {cleanup_error}",
            }

    if not tp_levels_norm:
        return {
            "status": "PROTECTED",
            "symbol": symbol,
            "bx_symbol": bx_symbol,
            "direction": direction,
            "avg_price": avg_price,
            "qty": position_qty,
            "sl_result": sl_result,
            "tp_orders": [],
            "tp_mode": "none",
            "effective_tp_levels": [],
            "effective_weighted_rr": 0.0,
        }

    tp_mode = "multi_tp"
    if min_qty > 0 and position_qty < min_qty * len(tp_levels_norm):
        return {
            "status": "PROTECTION_FAILED",
            "symbol": symbol,
            "bx_symbol": bx_symbol,
            "direction": direction,
            "avg_price": avg_price,
            "qty": position_qty,
            "sl_result": sl_result,
            "tp_orders": [],
            "error": f"position_qty={position_qty} cannot support required {len(tp_levels_norm)} TP legs with minQty={min_qty}",
        }

    try:
        desired_qtys = _allocate_tp_quantities(
            position_qty=position_qty,
            precision=precision,
            min_qty=min_qty,
            fractions=[x["close_fraction"] for x in tp_levels_norm],
        )
    except ValueError as exc:
        return {
            "status": "PROTECTION_FAILED",
            "symbol": symbol,
            "bx_symbol": bx_symbol,
            "direction": direction,
            "avg_price": avg_price,
            "qty": position_qty,
            "sl_result": sl_result,
            "tp_orders": [],
            "error": str(exc),
        }

    tp_results = []
    tp_cleanup_errors: list[dict[str, str]] = []
    current_price = None
    current_price_checked = False

    for level, tp_qty in zip(tp_levels_norm, desired_qtys):
        leg = str(level["leg"])
        pnl_pct = float(level["pnl_pct"])
        tp_price = avg_price * (1.0 + pnl_pct / 100.0) if direction == "LONG" else avg_price * (1.0 - pnl_pct / 100.0)

        base_client_order_id = build_tp_client_order_id(leg, trade_id)
        same_leg_orders = [
            order for order in existing_tp
            if _client_order_id_matches_base(order.get("clientOrderId", ""), base_client_order_id)
        ]
        matching_price_orders = [
            order for order in existing_tp
            if _tp_leg_from_order(order, leg, tp_price, price_precision, trade_id, direction)
        ]
        existing_leg = matching_price_orders[0] if matching_price_orders else None
        stale_tp_order_ids: list[str] = []

        if existing_leg:
            existing_order_id = str(existing_leg.get("orderId", ""))
            existing_qty = float(existing_leg.get("origQty", 0) or existing_leg.get("quantity", 0) or 0)
            # An existing TP is reusable only when both price and quantity match.
            # Any other orders owned by this same leg are stale duplicates and may
            # only be removed after a valid replacement/keeper is exchange-visible.
            stale_tp_order_ids = [
                str(order.get("orderId", "")) for order in same_leg_orders
                if str(order.get("orderId", "")) and str(order.get("orderId", "")) != existing_order_id
            ]
            qty_matches = abs(existing_qty - tp_qty) <= max(tp_qty * 1e-6, 1e-12)
            if qty_matches:
                if stale_tp_order_ids:
                    cleanup_ok, cleanup_error = _cancel_engine_protection_orders_verified(
                        symbol, direction, stale_tp_order_ids,
                    )
                    if not cleanup_ok:
                        tp_cleanup_errors.append({"leg": leg, "error": str(cleanup_error or "stale TP cleanup unverified")})
                tp_results.append(
                    {
                        "leg": leg,
                        "status": "already_exists",
                        "order_id": existing_order_id,
                        "client_order_id": str(existing_leg.get("clientOrderId", "")),
                        "price": float(existing_leg.get("stopPrice", 0) or existing_leg.get("price", 0) or 0),
                        "qty": existing_qty,
                        "pnl_pct": pnl_pct,
                        "expected_tp_price": tp_price,
                        "protection_order_verified": True,
                        "execution_verified": None,
                    }
                )
                continue
            # Same-price, wrong-size orders are replaced create-new-first. Do not
            # cancel the old target before its replacement can be verified.
            stale_tp_order_ids = [
                str(order.get("orderId", "")) for order in same_leg_orders
                if str(order.get("orderId", ""))
            ]
            if not stale_tp_order_ids and existing_order_id:
                # A legacy TP matched by price but does not carry this trade's
                # canonical ID. Do not cancel an order whose ownership cannot be
                # established automatically; fail closed instead of risking a
                # second TP for the same leg.
                tp_results.append({
                    "leg": leg, "status": "error",
                    "error": "existing TP quantity mismatch but clientOrderId ownership is not verifiable",
                    "qty": tp_qty, "pnl_pct": pnl_pct,
                })
                continue
        else:
            # Include same-trade TP IDs even when their trigger price is wrong.
            # The old implementation ignored these, then POSTed with the exact
            # same deterministic clientOrderId and could remain stuck in SL_ONLY.
            stale_tp_order_ids = [
                str(order.get("orderId", "")) for order in same_leg_orders
                if str(order.get("orderId", ""))
            ]

        if not current_price_checked:
            current_price = _current_close_price(symbol)
            current_price_checked = True

        if current_price is None:
            tp_results.append({"leg": leg, "status": "deferred", "reason": "current_price_unavailable", "price": tp_price, "qty": tp_qty, "pnl_pct": pnl_pct})
            continue

        trigger_invalid = (direction == "LONG" and tp_price <= current_price) or (direction == "SHORT" and tp_price >= current_price)

        if trigger_invalid:
            # Initial protection is established before local ownership is committed.
            # Never reduce the just-opened position with a market TP in this window;
            # return a protection failure so the caller uses its verified rollback path.
            log.critical(
                "[BINGX] TP target already crossed during initial protection for %s %s: price=%s current=%s",
                symbol, leg, _format_price(tp_price, price_precision), _format_price(current_price, price_precision),
            )
            return {
                "status": "PROTECTION_FAILED",
                "symbol": symbol,
                "bx_symbol": bx_symbol,
                "direction": direction,
                "avg_price": avg_price,
                "qty": position_qty,
                "sl_result": sl_result,
                "tp_orders": tp_results,
                "error": f"tp_target_crossed_before_initial_protection: leg={leg} tp={tp_price} current={current_price}",
                "failed_leg": leg,
                "failure_class": "TP_TARGET_CROSSED_BEFORE_PROTECTION",
            }

        client_order_id = (
            _repair_client_order_id(base_client_order_id)
            if stale_tp_order_ids else base_client_order_id
        )
        params = {
            "symbol": bx_symbol,
            "side": "SELL" if direction == "LONG" else "BUY",
            "positionSide": position_side_param(direction),
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": _format_price(tp_price, price_precision),
            "quantity": _format_qty(tp_qty, precision),
            "clientOrderId": client_order_id,
        }

        resp = _request("POST", ORDER_PATH, params)
        order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
        if resp.get("code") != 0:
            verify = _verify_open_order(
                symbol, direction, client_order_id=client_order_id, order_kind="TAKE_PROFIT_MARKET",
                expected_price=tp_price, expected_qty=tp_qty, price_precision=price_precision,
            )
            if verify.get("status") != "verified":
                log.error("[BINGX] TP order failed/unverified: %s code=%s msg=%s", leg, resp.get("code"), resp.get("msg"))
                tp_results.append({"leg": leg, "status": "error", "error": f"code={resp.get('code')} msg={resp.get('msg')}", "qty": tp_qty, "pnl_pct": pnl_pct})
                continue
            order = verify["order"]
            if stale_tp_order_ids:
                cleanup_ok, cleanup_error = _cancel_engine_protection_orders_verified(
                    symbol, direction, stale_tp_order_ids,
                )
                if not cleanup_ok:
                    tp_cleanup_errors.append({"leg": leg, "error": str(cleanup_error or "stale TP cleanup unverified")})
            tp_results.append({
                "leg": leg, "status": "reconciled_after_post_error",
                "order_id": str(order.get("orderId", "")),
                "client_order_id": order.get("clientOrderId") or client_order_id,
                "price": float(order.get("stopPrice", 0) or order.get("price", 0) or tp_price),
                "qty": _order_qty(order), "pnl_pct": pnl_pct,
                "protection_order_verified": True,
                "execution_verified": None,
            })
            continue

        # A code=0 acknowledgement is not enough for protection. Re-read the
        # exchange open-order book and verify exact client ID, type, price and qty.
        verify = _verify_open_order(
            symbol, direction, client_order_id=client_order_id, order_kind="TAKE_PROFIT_MARKET",
            expected_price=tp_price, expected_qty=tp_qty, price_precision=price_precision,
        )
        if verify.get("status") != "verified":
            tp_results.append({"leg": leg, "status": "error", "error": "TP acknowledged but not verifiable on exchange", "qty": tp_qty, "pnl_pct": pnl_pct})
            continue
        order = verify["order"]
        if stale_tp_order_ids:
            cleanup_ok, cleanup_error = _cancel_engine_protection_orders_verified(
                symbol, direction, stale_tp_order_ids,
            )
            if not cleanup_ok:
                tp_cleanup_errors.append({"leg": leg, "error": str(cleanup_error or "stale TP cleanup unverified")})
        tp_results.append({
            "leg": leg, "status": "created",
            "order_id": str(order.get("orderId", "")),
            "client_order_id": order.get("clientOrderId") or client_order_id,
            "price": float(order.get("stopPrice", 0) or order.get("price", 0) or tp_price),
            "qty": _order_qty(order), "pnl_pct": pnl_pct,
            "expected_tp_price": tp_price, "protection_order_verified": True, "execution_verified": None,
        })

    successful_tps = [t for t in tp_results if t.get("status") in {"created", "already_exists", "reconciled_after_post_error"}]
    tp_geometry_valid = True
    tp_qty_valid = True
    tp_identity_valid = not bool(tp_cleanup_errors)
    seen_tp_order_ids: set[str] = set()
    for level, expected_qty in zip(tp_levels_norm, desired_qtys):
        leg = str(level["leg"])
        pnl_pct = float(level["pnl_pct"])
        expected_price = avg_price * (1.0 + pnl_pct / 100.0) if direction == "LONG" else avg_price * (1.0 - pnl_pct / 100.0)
        matches = [t for t in successful_tps if str(t.get("leg", "")) == leg]
        if len(matches) != 1:
            tp_identity_valid = False
            continue
        item = matches[0]
        oid = str(item.get("order_id") or "")
        if not oid or oid in seen_tp_order_ids:
            tp_identity_valid = False
        seen_tp_order_ids.add(oid)
        actual_price = float(item.get("price") or 0.0)
        actual_qty = float(item.get("qty") or 0.0)
        expected_price_norm = float(_format_price(expected_price, price_precision))
        actual_price_norm = float(_format_price(actual_price, price_precision))
        # Exchange pricePrecision is authoritative: 103.12669 submitted to a
        # 4-decimal market is 103.1267. Compare the normalized exchange price,
        # not the unrounded mathematical target, otherwise valid protection is
        # falsely downgraded to SL_ONLY and can trigger an emergency close.
        qty_tol = max(1e-12, abs(expected_qty) * 1e-8)
        if actual_price <= 0 or actual_price_norm != expected_price_norm:
            tp_geometry_valid = False
        if actual_qty <= 0 or abs(actual_qty - float(expected_qty)) > qty_tol:
            tp_qty_valid = False
        if item.get("protection_order_verified") is not True or item.get("execution_verified") is True:
            tp_identity_valid = False

    tp_qty_sum = sum(float(t.get("qty") or 0.0) for t in successful_tps)
    tp_qty_valid = tp_qty_valid and abs(tp_qty_sum - float(position_qty)) <= max(1e-12, abs(float(position_qty)) * 1e-8)

    # Validate the actual rounded exchange trigger ladder. Rounding must not make
    # TP1/TP2 collapse onto each other or cross the entry, otherwise the protection
    # is economically invalid even when each individual order matches its target.
    rounded_tp_prices = []
    for level in tp_levels_norm:
        pnl_pct = float(level["pnl_pct"])
        expected = avg_price * (1.0 + pnl_pct / 100.0) if direction == "LONG" else avg_price * (1.0 - pnl_pct / 100.0)
        rounded_tp_prices.append(float(_format_price(expected, price_precision)))
    if any(p <= 0 for p in rounded_tp_prices):
        tp_geometry_valid = False
    elif direction == "LONG":
        if any(p <= avg_price for p in rounded_tp_prices) or any(b <= a for a, b in zip(rounded_tp_prices, rounded_tp_prices[1:])):
            tp_geometry_valid = False
    elif direction == "SHORT":
        if any(p >= avg_price for p in rounded_tp_prices) or any(b >= a for a, b in zip(rounded_tp_prices, rounded_tp_prices[1:])):
            tp_geometry_valid = False

    if not verified_sl_valid:
        final_status = "PROTECTION_FAILED"
    elif len(successful_tps) == len(tp_levels_norm) and tp_geometry_valid and tp_qty_valid and tp_identity_valid:
        final_status = "PROTECTED"
    else:
        final_status = "SL_ONLY"

    effective_levels = [
        {
            "leg": str(level["leg"]),
            "pnl_pct": float(level["pnl_pct"]),
            "close_fraction": float(qty / position_qty) if position_qty > 0 else 0.0,
            "qty": float(qty),
        }
        for level, qty in zip(tp_levels_norm, desired_qtys)
    ]
    effective_weighted_rr = _effective_weighted_rr(effective_levels, stop_loss_pct)

    return {
        "status": final_status,
        "symbol": symbol,
        "bx_symbol": bx_symbol,
        "direction": direction,
        "avg_price": avg_price,
        "qty": position_qty,
        "tp_mode": tp_mode,
        "effective_tp_levels": effective_levels,
        "effective_weighted_rr": effective_weighted_rr,
        "protection_validation": {
            "sl_verified": bool(verified_sl_valid),
            "tp_geometry_valid": bool(tp_geometry_valid),
            "tp_qty_valid": bool(tp_qty_valid),
            "tp_identity_valid": bool(tp_identity_valid),
            "tp_qty_sum": float(tp_qty_sum),
            "position_qty": float(position_qty),
        },
        "tp_orders": tp_results,
        "tp_cleanup_errors": tp_cleanup_errors,
        "sl_result": sl_result,
    }


def _validate_sl_order_for_position(order: dict, direction: str, avg_price: float, position_qty: float | None = None) -> bool:
    order_type = str(order.get("type", "")).upper()
    if order_type not in {"STOP", "STOP_MARKET"}:
        return False

    try:
        sl_price = float(order.get("stopPrice", 0) or order.get("price", 0) or 0)
        qty = float(order.get("origQty", 0) or order.get("quantity", 0) or 0)
    except (TypeError, ValueError):
        return False

    if sl_price <= 0 or qty <= 0 or avg_price <= 0:
        return False

    direction = str(direction).upper()
    if direction not in {"LONG", "SHORT"}:
        return False
    expected_side = "SELL" if direction == "LONG" else "BUY"
    if str(order.get("side", "")).upper() != expected_side:
        return False
    try:
        expected_position_side = position_side_param(direction)
    except Exception:
        return False
    if str(order.get("positionSide", "")).upper() != expected_position_side:
        return False
    if position_qty is not None and not _qty_matches_position(qty, float(position_qty)):
        return False

    if direction == "LONG":
        return sl_price <= avg_price
    if direction == "SHORT":
        return sl_price >= avg_price

    return False
