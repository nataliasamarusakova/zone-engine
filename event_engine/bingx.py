# bingx.py

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import time
import threading
from email.utils import parsedate_to_datetime
from decimal import (
    Decimal,
    ROUND_CEILING,
    ROUND_DOWN,
    ROUND_FLOOR,
)
from typing import Any
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
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

CACHE = {
    "ts": 0.0,
    "data": {},
    "by_display_name": {},
}
TTL = 3600
SERVER_TIME_OFFSET_MS = 0
_POSITION_MODE_CACHE: dict[str, Any] = {"ts": 0.0, "dual": None}

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
                        # BingX signs the sorted, unencoded canonical form and accepts
                        # the signed form fields as application/x-www-form-urlencoded.
                        wire_params.pop("signature", None)
                        if signed:
                            wire_params["signature"] = signature
                        response = session.request(
                            method=method,
                            url=base_url + path,
                            data=wire_params,
                            headers=headers,
                            timeout=request_timeout,
                        )
                    else:
                        response = session.request(
                            method=method,
                            url=base_url + path,
                            params=wire_params,
                            headers=headers,
                            timeout=request_timeout,
                        )
                    payload = response.json()
                except Exception as exc:
                    last_error = exc
                    if retryable and _is_network_error(exc) and base_url != _base_urls()[-1]:
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
    log.info("[BINGX] Active contracts=%d", len(data))
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
    log.info("[BINGX] Position mode=%s", mode)
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


def get_order(symbol: str, order_id: str | int) -> dict:
    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}

    resp = _request("GET", ORDER_PATH, {"symbol": bx, "orderId": str(order_id)}, signed=True)
    if resp.get("code") != 0:
        return {"status": "error", "error": resp.get("msg"), "code": resp.get("code")}

    data = resp.get("data") or {}
    order = data.get("order") or data

    avg_price_raw = order.get("avgPrice")
    try:
        avg_price = float(avg_price_raw) if avg_price_raw not in (None, "") else 0.0
    except (TypeError, ValueError):
        avg_price = 0.0
    try:
        trigger_price = float(order.get("stopPrice", 0) or 0)
    except (TypeError, ValueError):
        trigger_price = 0.0

    return {
        "status": "ok",
        "order_id": str(order.get("orderId", order_id)),
        "order_status": str(order.get("status", "")).upper(),
        "avg_price": avg_price,
        "trigger_price": trigger_price,
        "executed_qty": float(order.get("executedQty", 0) or order.get("cumQty", 0) or 0),
        "orig_qty": float(order.get("origQty", 0) or order.get("quantity", 0) or 0),
        "client_order_id": str(order.get("clientOrderId", "")),
    }


def cancel_order(symbol: str, order_id: str | int) -> dict:
    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}

    return _request("DELETE", ORDER_PATH, {"symbol": bx, "orderId": str(order_id)}, signed=True)


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
    digest = hashlib.sha256(f"{bx_symbol}:{trade_id}".encode()).hexdigest().upper()[:24]
    return f"EVT_OPEN_{digest}"


def open_market(symbol: str, direction: str, price: float, trade_id: str) -> dict:
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

    sizing_price = _current_close_price(symbol) or float(price)
    if sizing_price <= 0:
        return {"status": "error", "error": "invalid sizing price", "symbol": bx}

    leverage = min(LEVERAGE, MAX_LEVERAGE, max_lev)
    qty = (MARGIN_USDT * leverage) / max(sizing_price * mult, 1e-12)
    q = Decimal(str(qty)).quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
    qty = float(q)

    # Never silently increase leverage. A $1-class position may legitimately
    # be too small to support all TP legs at the exchange minQty.
    if qty <= 0 or qty < min_qty:
        return {
            "status": "error",
            "error": f"qty={qty} < min_qty={min_qty} at configured leverage={leverage}",
            "symbol": bx, "qty": qty, "min_qty": min_qty,
            "leverage": leverage, "sizing_price": sizing_price,
        }

    if not _set_leverage(bx, leverage, direction):
        return {"status": "error", "error": f"failed to set leverage={leverage} for {direction}", "symbol": bx, "leverage": leverage}

    side = "BUY" if direction == "LONG" else "SELL"
    try:
        position_side = position_side_param(direction)
    except Exception as exc:
        return {"status": "error", "error": f"position_mode_query_failed: {exc}", "symbol": bx}
    client_order_id = _new_open_client_order_id(bx, trade_id)

    params = {
        "symbol": bx,
        "side": side,
        "positionSide": position_side,
        "type": "MARKET",
        "quantity": f"{qty:.{prec}f}",
        "clientOrderId": client_order_id,
    }

    response = _request("POST", ORDER_PATH, params)
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
            log.warning("[BINGX] Order POST transport error for %s (%s); verifying via position...", bx, response.get("msg"))
            try:
                if has_open_position(symbol, direction):
                    log.warning("[BINGX] Position found after transport error -> treating order as filled (idempotent).")
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
                        "idempotency": "position_verified_after_transport_error",
                        "response": response,
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
    while time.time() - started < timeout_sec:
        pos = get_position_directional(symbol, direction)
        if pos.get("status") in {"found", "error"}:
            return pos
        time.sleep(poll_interval)

    return {"status": "timeout", "symbol": to_bx_symbol(symbol), "positionSide": str(direction).upper()}


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

        protective_side = (sl_price <= avg_price * 1.002) if direction == "LONG" else (sl_price >= avg_price * 0.998)
        if protective_side and _qty_matches_position(sl_qty, position_qty):
            valid_existing_sl = sl
            break

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
        if resp.get("code") != 0:
            log.error("[BINGX] SL failed: code=%s msg=%s", resp.get("code"), resp.get("msg"))
            return {
                "status": "PROTECTION_FAILED",
                "error": f"SL failed: {resp.get('msg')}",
                "sl_result": {"status": "error", "error": resp.get("msg")},
                "tp_orders": [],
            }

        order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
        sl_result = {
            "status": "created",
            "order_id": str(order.get("orderId", "")),
            "client_order_id": order.get("clientOrderId") or client_order_id,
            "stop_price": sl_price,
            "qty": position_qty,
        }

    verified = get_open_protection_directional(symbol, direction)
    verified_sl = list(verified.get("sl_orders", [])) if verified.get("status") == "ok" else []
    verified_sl_valid = any(_validate_sl_order_for_position(o, direction, avg_price) for o in verified_sl)

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

            resp = _request("POST", ORDER_PATH, market_params)
            if resp.get("code") != 0:
                log.error("[BINGX] TP market close failed: %s msg=%s", leg, resp.get("msg"))
                tp_results.append({"leg": leg, "status": "error", "error": f"code={resp.get('code')} msg={resp.get('msg')}", "qty": tp_qty, "pnl_pct": pnl_pct})
            else:
                order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
                tp_results.append({
                    "leg": leg,
                    "status": "created",
                    "order_id": str(order.get("orderId", "")),
                    "client_order_id": order.get("clientOrderId") or client_order_id,
                    "price": current_price,
                    "qty": tp_qty,
                    "pnl_pct": pnl_pct,
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
        if resp.get("code") != 0:
            log.error("[BINGX] TP order failed: %s code=%s msg=%s", leg, resp.get("code"), resp.get("msg"))
            tp_results.append({"leg": leg, "status": "error", "error": f"code={resp.get('code')} msg={resp.get('msg')}", "qty": tp_qty, "pnl_pct": pnl_pct})
            continue

        order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
        tp_results.append({
            "leg": leg,
            "status": "created",
            "order_id": str(order.get("orderId", "")),
            "client_order_id": order.get("clientOrderId") or client_order_id,
            "price": tp_price,
            "qty": tp_qty,
            "pnl_pct": pnl_pct,
        })

    # Remove only legacy TP3 orders created by this engine. Manual/unrelated
    # take-profit orders are left untouched.
    for old_order in existing_tp:
        cid = str(old_order.get("clientOrderId", "")).upper()
        old_id = str(old_order.get("orderId", ""))
        if old_id and ("_TP3" in cid or cid.endswith("TP3")):
            try:
                cancel_order(symbol, old_id)
            except Exception as exc:
                log.warning("[BINGX] Could not remove legacy TP3 %s: %s", old_id, exc)

    successful_tps = [t for t in tp_results if t.get("status") in {"created", "already_exists"}]
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


def _validate_sl_order_for_position(order: dict, direction: str, avg_price: float) -> bool:
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
    if direction == "LONG":
        return sl_price <= avg_price * 1.003
    if direction == "SHORT":
        return sl_price >= avg_price * 0.997

    return False
