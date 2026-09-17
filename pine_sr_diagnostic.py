#!/usr/bin/env python3
"""Read-only diagnostic for the visible Ajay R5.41 Pine S/R values.

This script intentionally does NOT import the trading engine. It is a standalone
translation of the supplied Pine S/R block so that its output can be compared
with the values visible on a TradingView 1H chart.
"""
from __future__ import annotations

import argparse
import sys
import time
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

import requests

from pine_r541_sr import Candle, compute_current_sr

BASE_URL = "https://data-api.binance.vision"
KLINES_URL = f"{BASE_URL}/api/v3/klines"
EXCHANGE_INFO_URL = f"{BASE_URL}/api/v3/exchangeInfo"


def normalize_symbol(symbol: str) -> str:
    s = symbol.strip().upper().replace("/", "-")
    if s.endswith("-USDT"):
        s = s[:-5] + "USDT"
    return s.replace("-", "")


def _get(url: str, params: dict[str, Any], timeout: float = 15.0) -> Any:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_1h(symbol: str, limit: int = 1000) -> list[Candle]:
    payload = _get(
        KLINES_URL,
        {"symbol": normalize_symbol(symbol), "interval": "1h", "limit": max(300, min(int(limit), 1000))},
    )
    if not isinstance(payload, list):
        raise ValueError("Binance klines response is not a list")

    now_ms = int(time.time() * 1000)
    candles: list[Candle] = []
    for row in payload:
        if not isinstance(row, list) or len(row) < 6:
            continue
        open_time = int(row[0])
        close_time = int(row[6]) if len(row) > 6 else open_time + 3_599_999
        # The Pine chart can display a developing bar, but its visible SR state
        # changes from confirmed pivot events only. For a stable cross-check,
        # use the same closed-bar boundary for every run.
        if close_time > now_ms:
            continue
        candles.append(
            Candle(
                ts=open_time,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            )
        )
    if len(candles) < 300:
        raise ValueError(f"Only {len(candles)} closed 1H candles available")
    return candles


def fetch_tick_size(symbol: str) -> float | None:
    try:
        payload = _get(EXCHANGE_INFO_URL, {"symbol": normalize_symbol(symbol)}, timeout=10.0)
        rows = payload.get("symbols", []) if isinstance(payload, dict) else []
        if not rows:
            return None
        filters = rows[0].get("filters", [])
        for item in filters:
            if item.get("filterType") == "PRICE_FILTER":
                value = item.get("tickSize")
                if value is not None and float(value) > 0:
                    return float(value)
    except Exception:
        return None
    return None


def format_price(value: float | None, tick_size: float | None) -> str:
    if value is None:
        return "None"
    if tick_size is not None and tick_size > 0:
        try:
            quantum = Decimal(str(tick_size))
            rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
            places = max(0, -quantum.as_tuple().exponent)
            return f"{rounded:.{places}f}"
        except (InvalidOperation, ValueError):
            pass
    text = f"{float(value):.8f}".rstrip("0").rstrip(".")
    return text or "0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ajay R5.41 visible Pine SR value diagnostic")
    parser.add_argument("--symbol", required=True, help="e.g. BTC-USDT")
    parser.add_argument("--limit", type=int, default=1000, help="closed 1H bars, 300-1000")
    args = parser.parse_args()

    candles = fetch_1h(args.symbol, args.limit)
    state = compute_current_sr(candles)
    tick_size = fetch_tick_size(args.symbol)

    print(f"AJAY R5.41 | {normalize_symbol(args.symbol)}")
    print(f"HIGH LEVEL: {format_price(state['highestph'], tick_size)}")
    print("SR LEVELS:")
    for level in state["levels"]:
        print(format_price(float(level), tick_size))
    print(f"LOW LEVEL: {format_price(state['lowestpl'], tick_size)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
