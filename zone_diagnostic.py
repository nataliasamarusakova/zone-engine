from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from event_engine.binance import (
    analysis_symbols_for_bingx,
    fetch_24h_ticker,
    fetch_klines as fetch_binance_klines,
)
from event_engine.bingx import (
    fetch_klines as fetch_bingx_klines,
    get_contract,
)
from event_engine.signals import SWING_LEN, generate_zone_signals


DEFAULT_1H_LIMIT = int(os.environ.get("KLINE_LIMIT_1H", "1000"))
DEFAULT_5M_LIMIT = int(os.environ.get("KLINE_LIMIT_5M", "288"))


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def pct_distance(price: float, level: float) -> float | None:
    if price <= 0:
        return None
    return (price - level) / price * 100.0


def touch_rows(df_5m: pd.DataFrame, midpoint: float) -> list[dict[str, Any]]:
    if df_5m.empty:
        return []
    hits: list[dict[str, Any]] = []
    for _, row in df_5m.iterrows():
        low = float(row["low"])
        high = float(row["high"])
        if low <= midpoint <= high:
            hits.append(
                {
                    "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                    "open": float(row["open"]),
                    "high": high,
                    "low": low,
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
    return hits


def zone_payload(
    zone: dict[str, Any],
    kind: str,
    latest_1h_idx: int,
    current_price: float,
    df_5m: pd.DataFrame,
) -> dict[str, Any]:
    top = float(zone["top"])
    bottom = float(zone["btm"])
    midpoint = (top + bottom) / 2.0
    start = int(zone.get("start", -1))
    age = max(0, latest_1h_idx - start) if start >= 0 else None
    touches = touch_rows(df_5m, midpoint)
    last_touch = touches[-1] if touches else None
    last_touch_age_min = None
    if last_touch:
        age = utc_now() - pd.Timestamp(last_touch["timestamp"])
        last_touch_age_min = round(max(0.0, age.total_seconds() / 60.0), 2)
    if current_price > top:
        price_position = "ABOVE"
    elif current_price < bottom:
        price_position = "BELOW"
    else:
        price_position = "INSIDE"

    return {
        "kind": kind,
        "start_idx": start,
        "age_1h_bars": age if isinstance(age, int) else (max(0, latest_1h_idx - start) if start >= 0 else None),
        "top": top,
        "midpoint": midpoint,
        "bottom": bottom,
        "width_abs": top - bottom,
        "width_pct_of_price": (top - bottom) / current_price * 100.0 if current_price > 0 else None,
        "price_position": price_position,
        "distance_midpoint_pct_from_price": pct_distance(current_price, midpoint),
        "distance_top_pct_from_price": pct_distance(current_price, top),
        "distance_bottom_pct_from_price": pct_distance(current_price, bottom),
        "touch_count_5m_in_loaded_window": len(touches),
        "last_5m_midpoint_touch": last_touch,
        "last_touch_age_minutes": last_touch_age_min,
        "all_5m_midpoint_touches": touches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Print active 1H Demand/Supply zones and 5m midpoint touches for one symbol.")
    parser.add_argument("--symbol", default=os.environ.get("DIAG_SYMBOL", "BTC-USDT"), help="BingX-style symbol, e.g. BTC-USDT")
    parser.add_argument("--bars-1h", type=int, default=DEFAULT_1H_LIMIT)
    parser.add_argument("--bars-5m", type=int, default=DEFAULT_5M_LIMIT)
    args = parser.parse_args()

    symbol = str(args.symbol).strip().upper()
    if not symbol:
        raise SystemExit("symbol is empty")

    print("=" * 100)
    print(f"ZONE DIAGNOSTIC | {symbol}")
    print(f"UTC now: {utc_now().isoformat()}")
    print(f"1H bars requested: {args.bars_1h} | 5M bars requested: {args.bars_5m}")
    print("This workflow is READ-ONLY: it does not execute orders and does not modify trade/visit state.")
    print("=" * 100)

    contract = get_contract(symbol)
    if not contract:
        raise SystemExit(f"BingX contract not found: {symbol}")
    status = contract.get("status")
    api_open = str(contract.get("apiStateOpen", "")).lower()
    print(f"BingX contract: {contract.get('symbol')} | status={status} | apiStateOpen={api_open}")

    mapping = analysis_symbols_for_bingx([symbol])[0]
    binance_symbol = str(mapping.get("binance_symbol") or "")
    provider = "binance" if mapping.get("binance_available") else "bingx"
    asset_class = mapping.get("asset_class") or "UNKNOWN"
    print(f"Analysis mapping: binance_symbol={binance_symbol or '-'} | provider={provider} | asset_class={asset_class}")

    if provider == "binance":
        bars_1h = fetch_binance_klines(binance_symbol, "1h", limit=args.bars_1h, retryable=False)
        bars_5m = fetch_binance_klines(binance_symbol, "5m", limit=args.bars_5m, retryable=False)
        ticker = fetch_24h_ticker(binance_symbol)
    else:
        bars_1h = fetch_bingx_klines(symbol, "1h", limit=args.bars_1h, retryable=False)
        bars_5m = fetch_bingx_klines(symbol, "5m", limit=args.bars_5m, retryable=False)
        ticker = None

    min_bars = SWING_LEN * 2 + 10
    if len(bars_1h) < min_bars:
        raise SystemExit(f"Insufficient 1H candles: {len(bars_1h)} < {min_bars}")

    df_5m = pd.DataFrame(bars_5m)
    if not df_5m.empty:
        df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"], utc=True)
        df_5m = df_5m.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        # Only closed 5m candles are relevant for the diagnostic, matching production trigger semantics.
        now = utc_now()
        df_5m = df_5m[pd.to_datetime(df_5m["timestamp"], utc=True) <= now - pd.Timedelta(minutes=5)].reset_index(drop=True)

    df_1h, supply, demand, signals = generate_zone_signals(pd.DataFrame(bars_1h), symbol=symbol, mode="historical")
    current_price = float(df_1h["close"].iloc[-1])
    latest_1h_time = pd.Timestamp(df_1h["timestamp"].iloc[-1]).isoformat()
    latest_5m_time = pd.Timestamp(df_5m["timestamp"].iloc[-1]).isoformat() if not df_5m.empty else None

    print("\n--- MARKET ---")
    print(f"current closed 1H close: {current_price:.12g}")
    print(f"latest closed 1H bar:    {latest_1h_time}")
    print(f"latest closed 5M bar:    {latest_5m_time or '-'}")
    if ticker:
        print(f"Binance 24h last price:   {ticker.get('lastPrice', '-')}")
        print(f"Binance 24h volume:       {ticker.get('quoteVolume', '-')}")

    payload = {
        "symbol": symbol,
        "generated_at": utc_now().isoformat(),
        "analysis_provider": provider,
        "binance_symbol": binance_symbol,
        "asset_class": asset_class,
        "latest_closed_1h_time": latest_1h_time,
        "latest_closed_5m_time": latest_5m_time,
        "current_closed_1h_price": current_price,
        "zone_counts": {"supply": len(supply), "demand": len(demand)},
        "zones": {
            "supply": [zone_payload(z, "SUPPLY", len(df_1h) - 1, current_price, df_5m) for z in supply],
            "demand": [zone_payload(z, "DEMAND", len(df_1h) - 1, current_price, df_5m) for z in demand],
        },
        "signals_from_1h_engine": signals,
    }

    for title, zones in (("SUPPLY", payload["zones"]["supply"]), ("DEMAND", payload["zones"]["demand"])):
        print(f"\n--- {title} ZONES ({len(zones)}) ---")
        if not zones:
            print("NONE")
            continue
        for n, z in enumerate(zones, 1):
            print(
                f"[{title} #{n}] start_idx={z['start_idx']} age_1h={z['age_1h_bars']} "
                f"bottom={z['bottom']:.12g} midpoint={z['midpoint']:.12g} top={z['top']:.12g} "
                f"width={z['width_abs']:.12g} ({z['width_pct_of_price']:.4f}% price) "
                f"price_position={z['price_position']} "
                f"dist_mid={z['distance_midpoint_pct_from_price']:.4f}% "
                f"5m_touch_count={z['touch_count_5m_in_loaded_window']} "
                f"last_touch_age_min={z['last_touch_age_minutes']}")
            if z["last_5m_midpoint_touch"]:
                t = z["last_5m_midpoint_touch"]
                print(
                    f"    last_touch: {t['timestamp']} | O={t['open']:.12g} H={t['high']:.12g} "
                    f"L={t['low']:.12g} C={t['close']:.12g} V={t['volume']:.12g}"
                )

    print("\n--- 5M RECENT MIDPOINT TOUCHES ---")
    recent = []
    for kind, zones in payload["zones"].items():
        for idx, z in enumerate(zones, 1):
            if z["all_5m_midpoint_touches"]:
                recent.append((kind.upper(), idx, z["midpoint"], z["all_5m_midpoint_touches"][-5:]))
    if not recent:
        print("No midpoint touches found in the loaded closed 5m window.")
    else:
        for kind, idx, midpoint, hits in recent:
            print(f"{kind} #{idx} midpoint={midpoint:.12g}")
            for h in hits:
                print(f"    {h['timestamp']} | H={h['high']:.12g} L={h['low']:.12g} C={h['close']:.12g}")

    print("\n--- RAW JSON (for machine-readable log artifacts) ---")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
