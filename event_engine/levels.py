from __future__ import annotations

import math
from typing import Any


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _zone_level(zone: dict[str, Any], side: str, source: str, zone_id: str) -> dict[str, Any] | None:
    top = _safe_float(zone.get("top"))
    bottom = _safe_float(zone.get("btm"))
    poi = _safe_float(zone.get("poi"))
    if top is None or bottom is None or poi is None:
        return None
    if side == "BLUE":
        price = bottom
        high = top
        low = bottom
    else:
        price = top
        high = top
        low = bottom
    return {
        "level_id": zone_id,
        "color": side,
        "kind": "DEMAND" if side == "BLUE" else "SUPPLY",
        "source": source,
        "price": round(float(price), 12),
        "low": round(float(low), 12),
        "high": round(float(high), 12),
        "poi": round(float(poi), 12),
        "start_idx": int(zone.get("start", -1)),
        "age_bars": None,
        "status": "ACTIVE",
    }


def _dedupe_levels(levels: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    if not levels:
        return []
    tolerance = max(float(tolerance), 0.0)
    levels = sorted(levels, key=lambda x: float(x["price"]))
    clusters: list[dict[str, Any]] = []
    for level in levels:
        price = float(level["price"])
        if not clusters or abs(price - float(clusters[-1]["price"])) > tolerance:
            clusters.append(level)
            continue
        current = clusters[-1]
        sources = list(current.get("sources", [current.get("source")]))
        src = level.get("source")
        if src and src not in sources:
            sources.append(src)
        kinds = list(current.get("kinds", [current.get("kind")]))
        kind = level.get("kind")
        if kind and kind not in kinds:
            kinds.append(kind)
        strength = max(int(current.get("strength", 1)), int(level.get("strength", 1))) + 1
        # Keep the level closest to the cluster median; deterministic tie -> newer source entry.
        current["strength"] = strength
        current["sources"] = sources
        current["kinds"] = kinds
        current["cluster_prices"] = sorted(
            list(current.get("cluster_prices", [current["price"]])) + [price]
        )
        current["price"] = round(float(sum(current["cluster_prices"]) / len(current["cluster_prices"])), 12)
        current["low"] = min(float(current.get("low", price)), float(level.get("low", price)))
        current["high"] = max(float(current.get("high", price)), float(level.get("high", price)))
    for level in clusters:
        level.setdefault("sources", [level.get("source")])
        level.setdefault("kinds", [level.get("kind")])
        level.setdefault("strength", 1)
        level.setdefault("cluster_prices", [level["price"]])
    return clusters


def build_level_snapshot(
    *,
    entry_price: float,
    active_demand: list[dict[str, Any]],
    active_supply: list[dict[str, Any]],
    sr_levels: list[dict[str, Any]] | list[float] | None = None,
    pivot_lows: list[dict[str, Any]] | None = None,
    pivot_highs: list[dict[str, Any]] | None = None,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Build a unified blue/red structural level pool from existing builders.

    This function does not construct Demand/Supply/SR itself. It only normalizes
    already-built structures so downstream logic can reason about them uniformly.
    """
    blue: list[dict[str, Any]] = []
    red: list[dict[str, Any]] = []

    for idx, zone in enumerate(active_demand):
        item = _zone_level(zone, "BLUE", "demand_zone", f"DEM_{int(zone.get('start', idx))}")
        if item:
            item["age_bars"] = None
            blue.append(item)

    for idx, zone in enumerate(active_supply):
        item = _zone_level(zone, "RED", "supply_zone", f"SUP_{int(zone.get('start', idx))}")
        if item:
            red.append(item)

    for idx, raw in enumerate(sr_levels or []):
        if isinstance(raw, dict):
            price = _safe_float(raw.get("price"))
            level_kind = str(raw.get("kind") or "SR").upper()
            level_id = str(raw.get("level_id") or f"SR_{idx}")
        else:
            price = _safe_float(raw)
            level_kind = "SR"
            level_id = f"SR_{idx}"
        if price is None or price <= 0:
            continue
        side = "BLUE" if price < entry_price else "RED"
        level = {
            "level_id": level_id,
            "color": side,
            "kind": "SUPPORT" if side == "BLUE" else "RESISTANCE",
            "source": "pine_sr",
            "price": round(price, 12),
            "low": round(price, 12),
            "high": round(price, 12),
            "poi": round(price, 12),
            "start_idx": None,
            "age_bars": None,
            "status": "ACTIVE",
            "pine_kind": level_kind,
        }
        (blue if side == "BLUE" else red).append(level)

    for idx, raw in enumerate(pivot_lows or []):
        price = _safe_float(raw.get("price") if isinstance(raw, dict) else raw)
        if price is None or price <= 0 or price >= entry_price:
            continue
        pidx = raw.get("pivot_idx") if isinstance(raw, dict) else None
        blue.append({
            "level_id": f"PL_{pidx if pidx is not None else idx}",
            "color": "BLUE",
            "kind": "PIVOT_LOW",
            "source": "pivot_low",
            "price": round(price, 12),
            "low": round(price, 12),
            "high": round(price, 12),
            "poi": round(price, 12),
            "start_idx": pidx,
            "age_bars": None,
            "status": "ACTIVE",
        })

    for idx, raw in enumerate(pivot_highs or []):
        price = _safe_float(raw.get("price") if isinstance(raw, dict) else raw)
        if price is None or price <= entry_price:
            continue
        pidx = raw.get("pivot_idx") if isinstance(raw, dict) else None
        red.append({
            "level_id": f"PH_{pidx if pidx is not None else idx}",
            "color": "RED",
            "kind": "PIVOT_HIGH",
            "source": "pivot_high",
            "price": round(price, 12),
            "low": round(price, 12),
            "high": round(price, 12),
            "poi": round(price, 12),
            "start_idx": pidx,
            "age_bars": None,
            "status": "ACTIVE",
        })

    blue = _dedupe_levels(blue, tolerance)
    red = _dedupe_levels(red, tolerance)
    blue.sort(key=lambda x: float(x["price"]), reverse=True)
    red.sort(key=lambda x: float(x["price"]))
    return {
        "entry_price": round(float(entry_price), 12),
        "blue": blue,
        "red": red,
    }


def select_protective_level(
    direction: str,
    entry_price: float,
    level_snapshot: dict[str, Any],
    *,
    exclude_level_ids: set[str] | None = None,
    min_distance: float = 0.0,
) -> dict[str, Any] | None:
    """Select the closest same-color structural level beyond entry."""
    direction = str(direction).upper()
    exclude_level_ids = exclude_level_ids or set()
    pool = level_snapshot.get("blue", []) if direction == "LONG" else level_snapshot.get("red", [])
    entry = float(entry_price)
    distance = max(float(min_distance), 0.0)
    if direction == "LONG":
        candidates = [
            x for x in pool
            if str(x.get("level_id")) not in exclude_level_ids
            and _safe_float(x.get("price"), 0.0) < entry - distance
            and x.get("status") == "ACTIVE"
        ]
        return max(candidates, key=lambda x: float(x["price"]), default=None)
    candidates = [
        x for x in pool
        if str(x.get("level_id")) not in exclude_level_ids
        and _safe_float(x.get("price"), 0.0) > entry + distance
        and x.get("status") == "ACTIVE"
    ]
    return min(candidates, key=lambda x: float(x["price"]), default=None)


def select_opposing_levels(
    direction: str,
    entry_price: float,
    level_snapshot: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    direction = str(direction).upper()
    entry = float(entry_price)
    pool = level_snapshot.get("red", []) if direction == "LONG" else level_snapshot.get("blue", [])
    if direction == "LONG":
        candidates = [x for x in pool if _safe_float(x.get("price"), 0.0) > entry and x.get("status") == "ACTIVE"]
        candidates.sort(key=lambda x: float(x["price"]))
    else:
        candidates = [x for x in pool if 0 < _safe_float(x.get("price"), 0.0) < entry and x.get("status") == "ACTIVE"]
        candidates.sort(key=lambda x: float(x["price"]), reverse=True)
    return candidates[: max(0, int(limit))]


def stop_from_level(direction: str, level: dict[str, Any], atr: float, buffer_atr: float = 0.10) -> tuple[float, float]:
    """Return structural stop and buffer from a selected level."""
    direction = str(direction).upper()
    # For a clustered level, the stop belongs beyond the outer edge of the
    # cluster, not at its arithmetic mean. This preserves the "behind the
    # level" invariant even when Support/Demand/Pivot confirmations overlap.
    if direction == "LONG":
        level_price = float(level.get("low", level["price"]))
    else:
        level_price = float(level.get("high", level["price"]))
    buffer = max(float(atr) * float(buffer_atr), 0.0)
    if direction == "LONG":
        return level_price - buffer, buffer
    return level_price + buffer, buffer
