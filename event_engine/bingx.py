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
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from decimal import (
    Decimal,
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
)
from typing import Any
from urllib.parse import quote, urlencode

import requests
from requests.adapters import HTTPAdapter

from event_engine import telemetry
from urllib3.util.retry import Retry

log = logging.getLogger("event_engine.bingx")

BASE_URL = os.environ.get("BINGX_BASE_URL", "https://open-api-vst.bingx.com").rstrip("/")
MARGIN_USDT = float(os.environ.get("BINGX_MARGIN_USDT", "1"))
LEVERAGE = int(os.environ.get("BINGX_LEVERAGE", "10"))
MAX_LEVERAGE = int(os.environ.get("BINGX_MAX_LEVERAGE", "10"))

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
    session = SESSION if retryable else _get_fast_session()

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
                            canonical = _canonical_params({k: v for k, v in request_params.items() if k != "signature"})
                            body = f"{canonical}&signature={signature}"
                            headers["Content-Type"] = "application/x-www-form-urlencoded"
                        else:
                            body = urlencode(wire_params, doseq=False)
                            headers["Content-Type"] = "application/x-www-form-urlencoded"
                        response = session.request(
                            method=method,
                            url=base_url + path,
                            data=body,
                            headers=headers,
                            timeout=request_timeout,
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
                        response = session.request(**request_kwargs)
                    payload = response.json()
                except Exception as exc:
                    last_error = exc
                    # NEVER resend a non-idempotent POST after a transport error.
                    # The exchange may have accepted the order while the response
                    # was lost; sending the same MARKET/STOP/TP POST to another
                    # endpoint can create a duplicate order. GET/DELETE remain
                    # eligible for the endpoint fallback.
                    if retryable and _is_network_error(exc) and method.upper() != "POST" and base_url != _base_urls()[-1]:
                        log.warning("[BINGX] Network failure on %s; trying fallback domain: %s", base_url, exc)
                        break
                    return {"code": -1, "msg": str(exc)}

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
        return {"code": -1, "msg": str(exc)}

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


def _normalize_orders_list(resp: dict) -> list[dict]:
    data = resp.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("orders", "positions", "order", "position"):
            val = data.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return [val]
    return []


def get_positions(*, timeout_sec: float | None = None, retryable: bool = True) -> list[dict]:
    resp = _request(
        "GET",
        POSITION_PATH,
        {},
        signed=True,
        timeout_sec=timeout_sec,
        retryable=retryable,
    )
    if resp.get("code") != 0:
        raise RuntimeError(f"[BINGX] get_positions failed: code={resp.get('code')} msg={resp.get('msg')}")
    return _normalize_orders_list(resp)


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

    return {
        "status": "ok",
        "order_id": str(order.get("orderId", order_id or "")),
        "order_status": str(order.get("status", "")).upper(),
        "symbol": str(order.get("symbol", bx)).upper(),
        "side": str(order.get("side", "")).upper(),
        "position_side": str(order.get("positionSide", "")).upper(),
        "order_type": str(order.get("type", "")).upper(),
        "avg_price": _num("avgPrice"),
        "trigger_price": _num("stopPrice"),
        "executed_qty": _num("executedQty", "cumQty"),
        "orig_qty": _num("origQty", "quantity"),
        "client_order_id": str(order.get("clientOrderId", client_order_id or "")),
        "time_ms": int(_num("time")) if _num("time") > 0 else None,
        "update_time_ms": int(_num("updateTime")) if _num("updateTime") > 0 else None,
    }


def cancel_order(symbol: str, order_id: str | int) -> dict:
    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}

    return _request("DELETE", ORDER_PATH, {"symbol": bx, "orderId": str(order_id)}, signed=True)



def get_all_orders(
    symbol: str,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """Fetch recent historical orders used for deterministic exit reconciliation."""
    bx = to_bx_symbol(symbol)
    if not bx:
        return []
    params: dict[str, Any] = {"symbol": bx, "limit": min(max(int(limit), 1), 100)}
    if start_time_ms is not None:
        params["startTime"] = int(start_time_ms)
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    resp = _request("GET", "/openApi/swap/v2/trade/allOrders", params, signed=True)
    if not isinstance(resp, dict) or resp.get("code") != 0:
        return []
    return _normalize_orders_list(resp)


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
        return False

    want = "LONG" if direction.upper() == "LONG" else "SHORT"
    positions = get_positions()

    for p in positions:
        if str(p.get("symbol", "")).upper() != bx:
            continue
        side = str(p.get("positionSide", p.get("positionAmt", ""))).upper()
        try:
            amt = float(p.get("positionAmt", 0) or 0)
        except Exception:
            amt = 0.0

        if amt != 0 and (want in side or (want == "LONG" and amt > 0) or (want == "SHORT" and amt < 0)):
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
            limit=100,
        )
    except Exception:
        return None
    target = str(client_order_id).upper()
    for order in orders:
        if str(order.get("clientOrderId", "")).upper() == target:
            return order
    return None


def open_market(symbol: str, direction: str, price: float, trade_id: str, *, execution_quote: dict[str, Any] | None = None, attempt_id: str | None = None) -> dict:
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
    signal_drift_pct = max(0.0, (sizing_price - float(price)) / float(price) * 100.0) if direction == "LONG" else max(0.0, (float(price) - sizing_price) / float(price) * 100.0)
    if signal_drift_pct > max_entry_slippage_pct:
        return {"status": "skipped_stale_signal", "error": f"signal_drift_pct={signal_drift_pct:.4f}% > {max_entry_slippage_pct:.4f}% at final order gate", "symbol": bx, "execution_quote": final_quote, "signal_drift_pct": signal_drift_pct, "signal_price": float(price), "execution_reference_price": sizing_price}

    # Recompute quantity from the final quote, not from a quote captured before leverage/idempotency calls.
    qty = (MARGIN_USDT * leverage) / max(sizing_price * mult, 1e-12)
    q = Decimal(str(qty)).quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
    qty = float(q)
    if qty <= 0:
        return {"status": "error", "error": "calculated quantity is <= 0", "symbol": bx, "qty": qty, "min_qty": min_qty, "leverage": leverage, "sizing_price": sizing_price, "execution_quote": final_quote}
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

    # Last micro-check immediately before the network POST. This prevents a slow Python/network
    # path from turning a freshly-read quote into a stale order reference.
    fresh_now, local_age_now, exchange_age_now, freshness_source_now = _quote_freshness(execution_quote, max_quote_age)
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

    resp = _request("GET", POSITION_PATH, {"symbol": bx_symbol})
    if resp.get("code") != 0:
        return {"status": "error", "error": f"get_position failed: {resp.get('msg')}", "symbol": bx_symbol}

    for p in _normalize_orders_list(resp):
        position_side = str(p.get("positionSide", "")).upper()
        if position_side not in (direction, "BOTH"):
            continue

        try:
            qty = abs(float(p.get("positionAmt", 0) or 0))
            avg_price = float(p.get("avgPrice", 0) or p.get("entryPrice", 0) or 0)
        except (TypeError, ValueError):
            continue

        if qty <= 0 or avg_price <= 0:
            continue

        if position_side == "BOTH":
            try:
                raw_amt = float(p.get("positionAmt", 0) or 0)
            except (TypeError, ValueError):
                continue
            if direction == "LONG" and raw_amt < 0:
                continue
            if direction == "SHORT" and raw_amt > 0:
                continue

        return {
            "status": "found",
            "symbol": p.get("symbol", bx_symbol),
            "positionSide": direction,
            "avgPrice": avg_price,
            "positionAmt": qty,
            "entryPrice": float(p.get("entryPrice", 0) or avg_price),
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

    resp = _request(
        "GET",
        OPEN_ORDERS_PATH,
        {"symbol": bx_symbol},
        timeout_sec=timeout_sec,
        retryable=retryable,
    )
    if resp.get("code") != 0:
        return {"status": "error", "error": f"openOrders failed: {resp.get('msg')}", "tp_orders": [], "sl_orders": []}

    tp_orders = []
    sl_orders = []

    for order in _normalize_orders_list(resp):
        position_side = str(order.get("positionSide", "")).upper()
        if position_side not in (direction, "BOTH"):
            continue

        order_type = str(order.get("type", "")).upper()
        if order_type in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
            tp_orders.append(order)
        elif order_type in {"STOP", "STOP_MARKET"}:
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
        normalized = [{"leg": "tp1", "pnl_pct": 2.0, "close_fraction": 1.0}]

    total = sum(x["close_fraction"] for x in normalized)
    for x in normalized:
        x["close_fraction"] /= total
    return normalized


def _tp_leg_from_order(order: dict, expected_leg: str, expected_price: float, price_precision: int, trade_id: str | None = None) -> bool:
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
        if client_id == f"EVT_{_trade_digest(trade_id)}_{expected_leg}":
            return actual_formatted == expected_formatted

    if f"_{expected_leg}_" in f"_{client_id}_":
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
        if raw_age >= 0:
            exchange_age = raw_age
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


def _quote_error(source: str, resp: Any, symbol: str) -> str:
    code = resp.get("code") if isinstance(resp, dict) else None
    msg = resp.get("msg") if isinstance(resp, dict) else None
    return f"{source} unavailable: code={code} msg={msg or 'invalid response'}"


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
            return {
                "status": "ok", "symbol": bx, "bid": bid, "ask": ask,
                "spread_pct": ((ask - bid) / bid * 100.0) if bid > 0 else None,
                "time": quote_time, "quote_exchange_time_ms": _normalize_market_timestamp_ms(quote_time), "quote_observed_at_ms": int(time.time() * 1000), "last_price": None, "quote_source": "bookTicker",
                "quote_sources_attempted": ["bookTicker"],
                "quote_fallback_reason": None,
            }

        ticker_resp = _call(TICKER_PATH)
        attempts.append(("ticker", ticker_resp))
        parsed = _parse_top_of_book_payload(ticker_resp, bx, "ticker")
        if parsed is not None:
            bid, ask, quote_time = parsed
            log.warning("[EXEC_QUOTE_FALLBACK] %s | bookTicker invalid/unavailable -> ticker", bx)
            return {
                "status": "ok", "symbol": bx, "bid": bid, "ask": ask,
                "spread_pct": ((ask - bid) / bid * 100.0) if bid > 0 else None,
                "time": quote_time, "quote_exchange_time_ms": _normalize_market_timestamp_ms(quote_time), "quote_observed_at_ms": int(time.time() * 1000), "last_price": None, "quote_source": "ticker",
                "quote_sources_attempted": [source for source, _ in attempts],
                "quote_fallback_reason": _quote_error("bookTicker", attempts[0][1], bx) if attempts else "bookTicker_invalid",
            }

        depth_resp = _call(DEPTH_PATH)
        attempts.append(("depth", depth_resp))
        parsed = _parse_depth_top(depth_resp)
        if parsed is not None:
            bid, ask, quote_time = parsed
            log.warning("[EXEC_QUOTE_FALLBACK] %s | bookTicker/ticker invalid -> depth", bx)
            return {
                "status": "ok", "symbol": bx, "bid": bid, "ask": ask,
                "spread_pct": ((ask - bid) / bid * 100.0) if bid > 0 else None,
                "time": quote_time, "quote_exchange_time_ms": _normalize_market_timestamp_ms(quote_time), "quote_observed_at_ms": int(time.time() * 1000), "last_price": None, "quote_source": "depth",
                "quote_sources_attempted": [source for source, _ in attempts],
                "quote_fallback_reason": "; ".join(_quote_error(source, response, bx) for source, response in attempts[:-1]),
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

    bp = [x for x in (parse_level(v) for v in bids) if x]
    ap = [x for x in (parse_level(v) for v in asks) if x]
    best_bid = bp[0][0] if bp else None
    best_ask = ap[0][0] if ap else None
    mid = mid_price or ((best_bid + best_ask) / 2.0 if best_bid and best_ask else None)

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
    return {
        "status": "ok",
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread_pct": ((best_ask - best_bid) / ((best_ask + best_bid) / 2.0) * 100.0) if best_bid and best_ask and best_bid > 0 else None,
        "mid_price": mid,
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
        "bid_depth_quote_0_1pct": depth_quote(bp, 0.1),
        "ask_depth_quote_0_1pct": depth_quote(ap, 0.1),
        "bid_depth_quote_0_5pct": depth_quote(bp, 0.5),
        "ask_depth_quote_0_5pct": depth_quote(ap, 0.5),
        "bid_depth_quote_1pct": depth_quote(bp, 1.0),
        "ask_depth_quote_1pct": depth_quote(ap, 1.0),
        "bids_top10": [[p, q] for p, q in bp[:10]],
        "asks_top10": [[p, q] for p, q in ap[:10]],
        "timestamp": data.get("T") or data.get("timestamp") or data.get("time"),
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
        "schema": "bingx_research_market_context_v1",
        "symbol": str(symbol).upper(),
        "bingx_symbol": bx,
        "captured_at_ms": captured_ms,
        "captured_at": datetime.fromtimestamp(captured_ms / 1000.0, tz=timezone.utc).isoformat(),
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
            resp = _research_public_get(key, path, params)
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
            return {"code": -1, "msg": str(exc)}

    premium = call("premiumIndex", PREMIUM_INDEX_PATH, {"symbol": bx})
    prow = _research_symbol_row(premium.get("data"), bx) if isinstance(premium, dict) else None
    if prow:
        for source_key, out_key in (("markPrice", "mark_price"), ("indexPrice", "index_price"), ("lastFundingRate", "funding_rate"), ("nextFundingTime", "next_funding_time_ms"), ("time", "premium_index_ts")):
            if prow.get(source_key) is not None:
                try:
                    result[out_key] = float(prow[source_key]) if source_key not in {"nextFundingTime", "time"} else int(prow[source_key])
                except (TypeError, ValueError):
                    result[out_key] = None

    if result.get("mark_price") and result.get("index_price"):
        idx = float(result["index_price"])
        if idx > 0:
            result["mark_index_basis_pct"] = (float(result["mark_price"]) / idx - 1.0) * 100.0

    oi = call("openInterest", OPEN_INTEREST_PATH, {"symbol": bx})
    orow = _research_symbol_row(oi.get("data"), bx) if isinstance(oi, dict) else None
    if orow:
        try:
            result["open_interest"] = float(orow.get("openInterest"))
        except (TypeError, ValueError):
            result["open_interest"] = None
        result["open_interest_ts"] = orow.get("time")

    book = call("depth", DEPTH_PATH, {"symbol": bx, "limit": int(depth_limit)})
    if isinstance(book, dict) and isinstance(book.get("data"), dict):
        result["order_book"] = _research_depth_metrics(book["data"], mid_price=result.get("mark_price"))
        result["order_book_timestamp"] = result["order_book"].get("timestamp")

    trades = call("trades", TRADES_PATH, {"symbol": bx, "limit": int(trades_limit)})
    result["recent_trades"] = _research_trade_metrics(trades.get("data")) if isinstance(trades, dict) else {"status": "error", "error": "trades_response_invalid"}

    if result["errors"]:
        result["status"] = "partial" if len(result["errors"]) < 4 else "error"
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
            raw_rows = fills.get("data") if isinstance(fills.get("data"), list) else []
            activity_rows = []
            fee_total = realized_total = 0.0
            for row in raw_rows:
                if not isinstance(row, dict):
                    continue
                item: dict[str, Any] = {}
                for src, dst in (("tradeId", "trade_id"), ("orderId", "order_id"), ("symbol", "symbol"), ("side", "side"), ("positionSide", "position_side")):
                    if row.get(src) not in (None, ""):
                        item[dst] = str(row.get(src))
                for src, dst in (("price", "price"), ("qty", "qty"), ("realizedPnl", "realized_pnl"), ("fee", "fee"), ("time", "time_ms")):
                    if row.get(src) in (None, ""):
                        continue
                    try:
                        val = float(row.get(src))
                        if dst == "time_ms":
                            val = int(val)
                        item[dst] = val if math.isfinite(float(val)) else None
                    except (TypeError, ValueError):
                        item[dst] = None
                if item.get("fee") is not None:
                    fee_total += float(item["fee"])
                if item.get("realized_pnl") is not None:
                    realized_total += float(item["realized_pnl"])
                activity_rows.append(item)
            result["recent_fill_window_start_ms"] = activity_start_ms
            result["recent_fill_window_end_ms"] = activity_end_ms
            result["recent_fills"] = activity_rows
            result["recent_fill_count"] = len(activity_rows)
            result["recent_fill_fee_total"] = fee_total
            result["recent_fill_realized_pnl_total"] = realized_total
        else:
            result.setdefault("errors", []).append(f"fills:code={fills.get('code') if isinstance(fills, dict) else 'invalid'}")
    except Exception as exc:
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
            rows = _normalize_orders_list(positions)
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
                price_ok = _format_price(actual_price, price_precision) == _format_price(expected_price, price_precision)
                qty_ok = _qty_matches_position(qty, expected_qty)
                if type_ok and price_ok and qty_ok:
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

    valid_existing_sl = None
    engine_owned_invalid_sl_ids: list[str] = []
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

        protective_side = (sl_price <= avg_price) if direction == "LONG" else (sl_price >= avg_price)
        qty_matches = _qty_matches_position(sl_qty, position_qty)
        if protective_side and qty_matches and valid_existing_sl is None:
            valid_existing_sl = sl
            continue

        # A stale/wrong-side/order-size-mismatched SL created by this engine
        # must never remain alongside a newly-created protective SL. Manual
        # orders are intentionally left untouched; the engine only cleans its
        # own EVT_* protection orders.
        client_id = str(sl.get("clientOrderId", ""))
        order_id = str(sl.get("orderId", ""))
        if order_id and client_id.upper().startswith("EVT_"):
            engine_owned_invalid_sl_ids.append(order_id)

    if engine_owned_invalid_sl_ids:
        for old_id in engine_owned_invalid_sl_ids:
            try:
                cancel_resp = cancel_order(symbol, old_id)
            except Exception as exc:
                return {"status": "SL_UNVERIFIED", "symbol": symbol, "direction": direction, "avg_price": avg_price, "qty": position_qty, "error": f"engine SL cleanup failed for {old_id}: {exc}"}
            if not isinstance(cancel_resp, dict) or cancel_resp.get("code") not in (0, "0"):
                return {"status": "SL_UNVERIFIED", "symbol": symbol, "direction": direction, "avg_price": avg_price, "qty": position_qty, "error": f"engine SL cleanup failed for {old_id}: {cancel_resp}"}

        post_cleanup = get_open_protection_directional(symbol, direction)
        if post_cleanup.get("status") != "ok":
            return {"status": "SL_UNVERIFIED", "symbol": symbol, "direction": direction, "avg_price": avg_price, "qty": position_qty, "error": "engine SL cleanup could not be verified"}
        remaining_ids = {str(o.get("orderId", "")) for o in post_cleanup.get("sl_orders", [])}
        if any(old_id in remaining_ids for old_id in engine_owned_invalid_sl_ids):
            return {"status": "SL_UNVERIFIED", "symbol": symbol, "direction": direction, "avg_price": avg_price, "qty": position_qty, "error": "engine SL cleanup not visible on exchange"}

    if valid_existing_sl is not None:
        sl = valid_existing_sl
        sl_result = {
            "status": "already_exists",
            "order_id": str(sl.get("orderId", "")),
            "client_order_id": str(sl.get("clientOrderId", "")),
            "stop_price": float(sl.get("stopPrice", 0) or sl.get("price", 0) or 0),
            "qty": float(sl.get("origQty", 0) or sl.get("quantity", 0) or position_qty),
        }
    else:
        sl_price = avg_price * (1.0 - stop_loss_pct / 100.0) if direction == "LONG" else avg_price * (1.0 + stop_loss_pct / 100.0)
        client_order_id = build_sl_client_order_id(trade_id)
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
            }
        else:
            sl_result = {
                "status": "created",
                "order_id": str(order.get("orderId", "")),
                "client_order_id": order.get("clientOrderId") or client_order_id,
                "stop_price": sl_price,
                "qty": position_qty,
            }

    verified = get_open_protection_directional(symbol, direction)
    verified_sl = list(verified.get("sl_orders", [])) if verified.get("status") == "ok" else []
    verified_sl_valid = any(_validate_sl_order_for_position(o, direction, avg_price, position_qty) for o in verified_sl)

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
    current_price = None
    current_price_checked = False

    for level, tp_qty in zip(tp_levels_norm, desired_qtys):
        leg = str(level["leg"])
        pnl_pct = float(level["pnl_pct"])
        tp_price = avg_price * (1.0 + pnl_pct / 100.0) if direction == "LONG" else avg_price * (1.0 - pnl_pct / 100.0)

        existing_leg = None
        for order in existing_tp:
            if _tp_leg_from_order(order, leg, tp_price, price_precision, trade_id):
                existing_leg = order
                break

        if existing_leg:
            existing_qty = float(existing_leg.get("origQty", 0) or existing_leg.get("quantity", 0) or 0)
            # An existing TP is reusable only when both price and quantity match
            # the current desired leg. Reusing a smaller/older order can leave
            # part of the position unprotected; reusing a larger one can over-close.
            qty_matches = abs(existing_qty - tp_qty) <= max(tp_qty * 1e-6, 1e-12)
            if qty_matches:
                tp_results.append(
                    {
                        "leg": leg,
                        "status": "already_exists",
                        "order_id": str(existing_leg.get("orderId", "")),
                        "client_order_id": str(existing_leg.get("clientOrderId", "")),
                        "price": float(existing_leg.get("stopPrice", 0) or existing_leg.get("price", 0) or 0),
                        "qty": existing_qty,
                        "pnl_pct": pnl_pct,
                    }
                )
                continue

            old_order_id = str(existing_leg.get("orderId", ""))
            if old_order_id:
                cancel_resp = cancel_order(symbol, old_order_id)
                if not isinstance(cancel_resp, dict) or cancel_resp.get("code") not in (0, "0"):
                    tp_results.append({
                        "leg": leg,
                        "status": "error",
                        "error": f"stale TP cancel failed: code={cancel_resp.get('code') if isinstance(cancel_resp, dict) else None} msg={cancel_resp.get('msg') if isinstance(cancel_resp, dict) else cancel_resp}",
                        "qty": tp_qty,
                        "pnl_pct": pnl_pct,
                    })
                    continue

        if not current_price_checked:
            current_price = _current_close_price(symbol)
            current_price_checked = True

        if current_price is None:
            tp_results.append({"leg": leg, "status": "deferred", "reason": "current_price_unavailable", "price": tp_price, "qty": tp_qty, "pnl_pct": pnl_pct})
            continue

        trigger_invalid = (direction == "LONG" and tp_price <= current_price) or (direction == "SHORT" and tp_price >= current_price)

        if trigger_invalid:
            log.warning("[BINGX] TP market execution for %s %s: price=%s current=%s (trigger crossed)", symbol, leg, _format_price(tp_price, price_precision), _format_price(current_price, price_precision))
            client_order_id = build_tp_client_order_id(leg, trade_id)
            market_params = {
                "symbol": bx_symbol,
                "side": "SELL" if direction == "LONG" else "BUY",
                "positionSide": position_side_param(direction),
                "type": "MARKET",
                "quantity": _format_qty(tp_qty, precision),
                "clientOrderId": client_order_id,
            }

            pre_position_qty = position_qty
            resp = _request("POST", ORDER_PATH, market_params)
            order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
            order_id = str(order.get("orderId", ""))
            if resp.get("code") != 0:
                # Never blind-retry an ambiguous MARKET reduction. Reconcile
                # the order/position and only accept the leg when execution is
                # actually observable.
                verification = _verify_market_reduce_order(
                    symbol, direction, order_id, tp_qty, pre_position_qty
                )
            else:
                verification = _verify_market_reduce_order(
                    symbol, direction, order_id, tp_qty, pre_position_qty
                )

            if verification.get("status") != "verified":
                tp_results.append({
                    "leg": leg,
                    "status": "error",
                    "error": (
                        f"TP market close acknowledged/attempted but execution could not be verified: "
                        f"response_code={resp.get('code')} verification={verification}"
                    ),
                    "qty": tp_qty,
                    "pnl_pct": pnl_pct,
                })
            else:
                tp_results.append({
                    "leg": leg,
                    "status": "created",
                    "order_id": order_id,
                    "client_order_id": order.get("clientOrderId") or client_order_id,
                    "price": current_price,
                    "qty": float(verification.get("reduced_qty") or verification.get("executed_qty") or tp_qty),
                    "pnl_pct": pnl_pct,
                    "execution_verified": True,
                    "executed_qty": float(verification.get("executed_qty", 0.0) or 0.0),
                    "remaining_qty": float(verification.get("remaining_qty", 0.0) or 0.0),
                })
            continue

        client_order_id = build_tp_client_order_id(leg, trade_id)
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
            tp_results.append({
                "leg": leg, "status": "reconciled_after_post_error",
                "order_id": str(order.get("orderId", "")),
                "client_order_id": order.get("clientOrderId") or client_order_id,
                "price": float(order.get("stopPrice", 0) or order.get("price", 0) or tp_price),
                "qty": _order_qty(order), "pnl_pct": pnl_pct,
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
        tp_results.append({
            "leg": leg, "status": "created",
            "order_id": str(order.get("orderId", "")),
            "client_order_id": order.get("clientOrderId") or client_order_id,
            "price": float(order.get("stopPrice", 0) or order.get("price", 0) or tp_price),
            "qty": _order_qty(order), "pnl_pct": pnl_pct,
        })

    successful_tps = [t for t in tp_results if t.get("status") in {"created", "already_exists", "reconciled_after_post_error"}]
    if not verified_sl_valid:
        final_status = "PROTECTION_FAILED"
    elif len(successful_tps) == len(tp_levels_norm):
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
        "tp_orders": tp_results,
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
    if position_qty is not None and not _qty_matches_position(qty, float(position_qty)):
        return False

    if direction == "LONG":
        return sl_price <= avg_price
    if direction == "SHORT":
        return sl_price >= avg_price

    return False
