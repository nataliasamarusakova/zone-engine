from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable


DEFAULT_CLUSTER_ATR = 0.20
DEFAULT_CLUSTER_PCT = 0.0015


def _num(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _make_level_id(source: str, price: float, anchor: Any = "") -> str:
    raw = f"{source}|{anchor}|{price:.12f}".encode("utf-8")
    return f"LVL_{hashlib.sha1(raw).hexdigest()[:12].upper()}"


def _base_level(
    *,
    color: str,
    kind: str,
    source: str,
    price: float,
    lower: float | None = None,
    upper: float | None = None,
    age_bars: int | None = None,
    created_idx: int | None = None,
    strength: int = 1,
    status: str = "ACTIVE",
    anchor: Any = "",
) -> dict[str, Any]:
    lo = price if lower is None else min(lower, price, upper if upper is not None else price)
    hi = price if upper is None else max(upper, price, lower if lower is not None else price)
    return {
        "level_id": _make_level_id(source, price, anchor),
        "color": color,
        "kind": kind,
        "source": source,
        "price": float(price),
        "lower": float(lo),
        "upper": float(hi),
        "age_bars": int(age_bars) if age_bars is not None else None,
        "created_idx": int(created_idx) if created_idx is not None else None,
        "strength": max(1, int(strength)),
        "status": status,
    }


def _within_cluster(a: dict[str, Any], b: dict[str, Any], tolerance: float) -> bool:
    return abs(float(a["price"]) - float(b["price"])) <= tolerance or not (
        float(a["upper"]) < float(b["lower"]) - tolerance
        or float(b["upper"]) < float(a["lower"]) - tolerance
    )


def _cluster_levels(levels: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for level in sorted(levels, key=lambda x: float(x["price"])):
        assigned = None
        for cluster in clusters:
            if _within_cluster(level, cluster, tolerance):
                assigned = cluster
                break
        if assigned is None:
            assigned = {
                "members": [level],
                "lower": float(level["lower"]),
                "upper": float(level["upper"]),
                "price": float(level["price"]),
                "strength": int(level.get("strength", 1)),
            }
            clusters.append(assigned)
        else:
            assigned["members"].append(level)
            assigned["lower"] = min(float(assigned["lower"]), float(level["lower"]))
            assigned["upper"] = max(float(assigned["upper"]), float(level["upper"]))
            assigned["price"] = (float(assigned["lower"]) + float(assigned["upper"])) / 2.0
            assigned["strength"] = sum(int(x.get("strength", 1)) for x in assigned["members"])
    out: list[dict[str, Any]] = []
    for cluster in clusters:
        members = cluster["members"]
        by_source: list[str] = []
        for m in members:
            src = str(m.get("kind") or m.get("source") or "UNKNOWN")
            if src not in by_source:
                by_source.append(src)
        base = max(members, key=lambda x: (int(x.get("strength", 1)), x.get("created_idx") or -1))
        color = str(base["color"])
        kind = "CLUSTER"
        cluster_id = _make_level_id(f"{color}_CLUSTER", float(cluster["price"]), "|".join(sorted(str(m["level_id"]) for m in members)))
        out.append({
            "level_id": cluster_id,
            "color": color,
            "kind": kind,
            "source": "cluster",
            "price": round(float(cluster["price"]), 12),
            "lower": round(float(cluster["lower"]), 12),
            "upper": round(float(cluster["upper"]), 12),
            "age_bars": min((m["age_bars"] for m in members if m.get("age_bars") is not None), default=None),
            "created_idx": max((m["created_idx"] for m in members if m.get("created_idx") is not None), default=None),
            "strength": int(cluster["strength"]),
            "status": "ACTIVE",
            "member_level_ids": [m["level_id"] for m in members],
            "member_kinds": by_source,
        })
    return out


def build_level_pool(
    *,
    demand: Iterable[dict[str, Any]],
    supply: Iterable[dict[str, Any]],
    support_resistance: Iterable[dict[str, Any]] = (),
    pivot_lows: Iterable[dict[str, Any]] = (),
    pivot_highs: Iterable[dict[str, Any]] = (),
    current_idx: int | None = None,
    reference_price: float | None = None,
    atr: float | None = None,
    cluster_atr: float = DEFAULT_CLUSTER_ATR,
    cluster_pct: float = DEFAULT_CLUSTER_PCT,
) -> dict[str, list[dict[str, Any]]]:
    raw: list[dict[str, Any]] = []

    for z in demand:
        lower = _num(z.get("btm"))
        upper = _num(z.get("top"))
        if lower is None or upper is None:
            continue
        start = z.get("start")
        age = max(0, int(current_idx) - int(start)) if current_idx is not None and start is not None else None
        raw.append(_base_level(color="BLUE", kind="DEMAND", source="demand_zone", price=(lower + upper) / 2.0, lower=lower, upper=upper, age_bars=age, created_idx=int(start) if start is not None else None, strength=2, anchor=start))

    for z in supply:
        lower = _num(z.get("btm"))
        upper = _num(z.get("top"))
        if lower is None or upper is None:
            continue
        start = z.get("start")
        age = max(0, int(current_idx) - int(start)) if current_idx is not None and start is not None else None
        raw.append(_base_level(color="RED", kind="SUPPLY", source="supply_zone", price=(lower + upper) / 2.0, lower=lower, upper=upper, age_bars=age, created_idx=int(start) if start is not None else None, strength=2, anchor=start))

    for level in support_resistance:
        price = _num(level.get("price"))
        if price is None or price <= 0:
            continue
        semantic = str(level.get("semantic") or level.get("kind") or "").upper()
        if semantic not in {"SUPPORT", "RESISTANCE"}:
            continue
        color = "BLUE" if semantic == "SUPPORT" else "RED"
        sr_status = "ACTIVE"
        if reference_price is not None:
            if semantic == "SUPPORT" and reference_price < price:
                sr_status = "BROKEN"
            if semantic == "RESISTANCE" and reference_price > price:
                sr_status = "BROKEN"
        raw.append(_base_level(color=color, kind=semantic, source="pine_sr", price=price, age_bars=level.get("age_bars"), created_idx=level.get("created_idx"), strength=max(1, int(level.get("strength", 1))), status=sr_status, anchor=level.get("pivot_idx", price)))

    for level in pivot_lows:
        price = _num(level.get("price"))
        if price is None or price <= 0:
            continue
        pivot_status = "ACTIVE" if reference_price is None or reference_price >= price else "BROKEN"
        raw.append(_base_level(color="BLUE", kind="PIVOT_LOW", source="pivot_low", price=price, age_bars=level.get("age_bars"), created_idx=level.get("created_idx"), strength=1, status=pivot_status, anchor=level.get("created_idx", price)))

    for level in pivot_highs:
        price = _num(level.get("price"))
        if price is None or price <= 0:
            continue
        pivot_status = "ACTIVE" if reference_price is None or reference_price <= price else "BROKEN"
        raw.append(_base_level(color="RED", kind="PIVOT_HIGH", source="pivot_high", price=price, age_bars=level.get("age_bars"), created_idx=level.get("created_idx"), strength=1, status=pivot_status, anchor=level.get("created_idx", price)))

    active_raw = [x for x in raw if str(x.get("status", "ACTIVE")).upper() == "ACTIVE"]
    invalid_raw = [x for x in raw if str(x.get("status", "ACTIVE")).upper() != "ACTIVE"]
    tolerance = max(float(atr or 0.0) * float(cluster_atr), max(abs(float(x["price"])) for x in active_raw) * float(cluster_pct), 1e-12) if active_raw else 1e-12
    blue = _cluster_levels([x for x in active_raw if x["color"] == "BLUE"], tolerance)
    red = _cluster_levels([x for x in active_raw if x["color"] == "RED"], tolerance)
    invalid = sorted(invalid_raw, key=lambda x: (x["color"], float(x["price"])))
    blue.sort(key=lambda x: float(x["price"]))
    red.sort(key=lambda x: float(x["price"]))
    return {"blue": blue, "red": red, "all": blue + red, "invalid": invalid, "cluster_tolerance": tolerance}


def active_levels_below(levels: Iterable[dict[str, Any]], entry: float) -> list[dict[str, Any]]:
    """Return active levels entirely below the entry, excluding levels that overlap it."""
    return [
        x for x in levels
        if str(x.get("status", "ACTIVE")) == "ACTIVE" and float(x["upper"]) < entry
    ]


def active_levels_above(levels: Iterable[dict[str, Any]], entry: float) -> list[dict[str, Any]]:
    """Return active levels entirely above the entry, excluding levels that overlap it."""
    return [
        x for x in levels
        if str(x.get("status", "ACTIVE")) == "ACTIVE" and float(x["lower"]) > entry
    ]


def select_protective_level(direction: str, entry: float, blue: Iterable[dict[str, Any]], red: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    direction = str(direction).upper()
    candidates = active_levels_below(blue, entry) if direction == "LONG" else active_levels_above(red, entry)
    if not candidates:
        return None
    if direction == "LONG":
        return max(candidates, key=lambda x: float(x["upper"]))
    return min(candidates, key=lambda x: float(x["lower"]))


def select_opposing_levels(direction: str, entry: float, blue: Iterable[dict[str, Any]], red: Iterable[dict[str, Any]], limit: int = 2) -> list[dict[str, Any]]:
    direction = str(direction).upper()
    candidates = active_levels_above(red, entry) if direction == "LONG" else active_levels_below(blue, entry)
    if direction == "LONG":
        candidates.sort(key=lambda x: float(x["lower"]))
    else:
        candidates.sort(key=lambda x: float(x["upper"]), reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for level in candidates:
        lid = str(level.get("level_id"))
        if lid in seen:
            continue
        seen.add(lid)
        out.append(level)
        if len(out) >= max(1, int(limit)):
            break
    return out


def protective_price(level: dict[str, Any], direction: str, buffer: float) -> float:
    if str(direction).upper() == "LONG":
        return float(level["lower"]) - float(buffer)
    return float(level["upper"]) + float(buffer)


def target_price_before_level(level: dict[str, Any], direction: str, buffer: float) -> float:
    if str(direction).upper() == "LONG":
        return float(level["lower"]) - float(buffer)
    return float(level["upper"]) + float(buffer)


def select_entry_level_for_zone(direction: str, zone: dict[str, Any], levels: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    direction = str(direction).upper()
    wanted_kind = "DEMAND" if direction == "LONG" else "SUPPLY"
    low = _num(zone.get("btm"))
    high = _num(zone.get("top"))
    if low is None or high is None:
        return None
    pool = levels.get("blue" if direction == "LONG" else "red", [])
    matches = [x for x in pool if wanted_kind in {str(k).upper() for k in x.get("member_kinds", [])} and not (float(x["upper"]) < low or float(x["lower"]) > high)]
    if not matches:
        return None
    return min(matches, key=lambda x: abs(float(x["price"]) - ((low + high) / 2.0)))
