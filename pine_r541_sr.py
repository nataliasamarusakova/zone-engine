#!/usr/bin/env python3
"""Exact-ish, standalone reproduction of the Ajay R5.41 S/R block supplied by the user.

This module intentionally does not import or modify the zone-engine trading code.
It reproduces only the Pine S/R calculations needed for the visible price values:
- rb = 10 pivot confirmation
- prd = 284 lookback
- ChannelW = 10
- strengthSR = 2
- sr_levels array (up to 20 levels)
- highestph / lowestpl

Important Pine semantics reproduced here:
- ph/pl are emitted on the confirmation bar, not the pivot bar.
- The current displayed SR arrays are the state from the most recent bar where
  ph or pl fires; between pivot confirmations they persist.
- When ph and pl somehow coexist on the same confirmation bar, SR candidate
  selection uses ph precedence (matching "ph[x] ? ... : low[x+rb]") while
  highestph/lowestpl consider both, and sr_levels storage follows Pine's two
  successive if statements (pl overwrites ph at the same countpp).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _is_pivot_high(candles: list[Candle], pivot_idx: int, rb: int) -> bool:
    if pivot_idx - rb < 0 or pivot_idx + rb >= len(candles):
        return False
    v = candles[pivot_idx].high
    for j in range(pivot_idx - rb, pivot_idx + rb + 1):
        if candles[j].high > v:
            return False
    return True


def _is_pivot_low(candles: list[Candle], pivot_idx: int, rb: int) -> bool:
    if pivot_idx - rb < 0 or pivot_idx + rb >= len(candles):
        return False
    v = candles[pivot_idx].low
    for j in range(pivot_idx - rb, pivot_idx + rb + 1):
        if candles[j].low < v:
            return False
    return True


def build_confirmed_pivots(candles: list[Candle], rb: int = 10) -> tuple[list[float | None], list[float | None]]:
    """Return ph/pl series indexed by the Pine confirmation bar."""
    n = len(candles)
    ph: list[float | None] = [None] * n
    pl: list[float | None] = [None] * n
    for confirm_idx in range(n):
        pivot_idx = confirm_idx - rb
        if pivot_idx < rb or pivot_idx + rb >= n:
            continue
        if _is_pivot_high(candles, pivot_idx, rb):
            ph[confirm_idx] = candles[pivot_idx].high
        if _is_pivot_low(candles, pivot_idx, rb):
            pl[confirm_idx] = candles[pivot_idx].low
    return ph, pl


def _rolling_high(candles: list[Candle], end_idx: int, length: int) -> float:
    start = max(0, end_idx - length + 1)
    return max(c.high for c in candles[start:end_idx + 1])


def _rolling_low(candles: list[Candle], end_idx: int, length: int) -> float:
    start = max(0, end_idx - length + 1)
    return min(c.low for c in candles[start:end_idx + 1])


def _event_pivots(
    candles: list[Candle],
    ph: list[float | None],
    pl: list[float | None],
    event_idx: int,
    prd: int,
) -> list[tuple[int, float, str, float | None, float | None]]:
    """Newest-to-oldest pivot events visible to Pine's x=0..prd loops."""
    out: list[tuple[int, float, str, float | None, float | None]] = []
    for x in range(prd + 1):
        idx = event_idx - x
        if idx < 0:
            break
        if ph[idx] is None and pl[idx] is None:
            continue
        # Pine candidate price: ph ? high[x+rb] : low[x+rb].
        if ph[idx] is not None:
            candidate_price = float(ph[idx])
            candidate_kind = "H"
        else:
            candidate_price = float(pl[idx])
            candidate_kind = "L"
        out.append((idx, candidate_price, candidate_kind, ph[idx], pl[idx]))
        if len(out) >= 41:
            break
    return out


def calculate_event_sr(
    candles: list[Candle],
    ph: list[float | None],
    pl: list[float | None],
    event_idx: int,
    rb: int = 10,
    prd: int = 284,
    channel_w: float = 10.0,
    strength_sr: int = 2,
) -> dict[str, Any]:
    """Recreate the Pine code's single `if ph or pl` calculation block."""
    window_high = _rolling_high(candles, event_idx, prd)
    window_low = _rolling_low(candles, event_idx, prd)
    cwidth = (window_high - window_low) * channel_w / 100.0

    highestph = window_low
    lowestpl = window_high

    pivots = _event_pivots(candles, ph, pl, event_idx, prd)

    # Pine's highestph/lowestpl pass uses both ph and pl when both exist.
    for _, _, _, phv, plv in pivots:
        if phv is not None:
            highestph = max(highestph, float(phv))
            lowestpl = min(lowestpl, float(phv))
        if plv is not None:
            highestph = max(highestph, float(plv))
            lowestpl = min(lowestpl, float(plv))

    aas = [True] * 41
    sr_levels: list[float | None] = [None] * 21
    accepted: list[dict[str, Any]] = []

    countpp = 0
    for _, pivot_price, pivot_kind, phv, plv in pivots:
        countpp += 1
        if countpp > 40:
            break
        if not aas[countpp]:
            continue

        upl = pivot_price + cwidth
        dnl = pivot_price - cwidth
        tmp = [True] * 41
        cnt = 0
        tpoint = 0

        for _, price2, _, _, _ in pivots:
            cnt += 1
            if cnt > 40:
                break
            # Pine: if array.get(aas, cnt)
            if not aas[cnt]:
                continue
            if dnl <= price2 <= upl:
                tpoint += 1
                tmp[cnt] = False

        if tpoint >= strength_sr:
            for g in range(41):
                if not tmp[g]:
                    aas[g] = False

            if countpp < 21:
                # Pine has two independent if statements; a simultaneous pl
                # overwrites the sr_levels[countpp] written by ph.
                if phv is not None:
                    sr_levels[countpp] = float(phv)
                if plv is not None:
                    sr_levels[countpp] = float(plv)
                accepted.append(
                    {
                        "countpp": countpp,
                        "price": sr_levels[countpp],
                        "kind": "L" if plv is not None else ("H" if phv is not None else pivot_kind),
                        "cluster_width": cwidth,
                        "points": tpoint,
                    }
                )

    levels = [v for v in sr_levels[1:] if v is not None]
    return {
        "levels": levels,
        "sr_levels": sr_levels,
        "highestph": highestph,
        "lowestpl": lowestpl,
        "cwidth": cwidth,
        "event_idx": event_idx,
        "event_timestamp": candles[event_idx].ts,
        "accepted": accepted,
    }


def compute_current_sr(candles: list[Candle], rb: int = 10, prd: int = 284, channel_w: float = 10.0, strength_sr: int = 2) -> dict[str, Any]:
    if len(candles) < prd + rb + 2:
        raise ValueError(f"Need at least {prd + rb + 2} candles, got {len(candles)}")
    ph, pl = build_confirmed_pivots(candles, rb=rb)
    current: dict[str, Any] | None = None
    for i in range(len(candles)):
        if ph[i] is not None or pl[i] is not None:
            current = calculate_event_sr(candles, ph, pl, i, rb=rb, prd=prd, channel_w=channel_w, strength_sr=strength_sr)
    if current is None:
        return {
            "levels": [],
            "sr_levels": [None] * 21,
            "highestph": None,
            "lowestpl": None,
            "cwidth": None,
            "event_idx": None,
            "event_timestamp": None,
            "accepted": [],
        }
    current = dict(current)
    current["latest_closed_timestamp"] = candles[-1].ts
    current["latest_closed_close"] = candles[-1].close
    current["pivot_high_confirmed_at_latest"] = ph[-1]
    current["pivot_low_confirmed_at_latest"] = pl[-1]
    return current
