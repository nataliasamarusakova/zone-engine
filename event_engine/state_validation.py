"""Shared structural validation for persisted active-trade records."""
from __future__ import annotations

import math
from typing import Any


_LIST_FIELDS = (
    "tp_orders",
    "hit_legs",
    "completed_tp_legs",
    "tp_fill_events",
    "tp_levels",
    "effective_tp_levels",
)
_DICT_FIELDS = (
    "entry_order",
    "sl_order",
    "tp_filled_qty",
    "tp_fill_classification",
    "setup",
    "signal_snapshot",
    "entry_bar",
    "previous_bar",
    "fill_position",
    "execution_snapshot",
    "mfe_milestones_r",
)
_DICT_LIST_FIELDS = ("tp_orders", "tp_fill_events", "tp_levels", "effective_tp_levels")
_NON_NULL_FIELDS = _LIST_FIELDS + ("tp_filled_qty",)


def _validate_setup_tp_levels(event_id: Any, setup: dict) -> None:
    """Reject malformed persisted TP plans before reconciliation consumes them."""
    if "tp_levels" not in setup or setup["tp_levels"] is None:
        return
    levels = setup["tp_levels"]
    if not isinstance(levels, list):
        raise ValueError(f"event_id={event_id!r} setup.tp_levels must be an array")

    seen_legs: set[str] = set()
    for index, level in enumerate(levels):
        if not isinstance(level, dict):
            raise ValueError(f"event_id={event_id!r} setup.tp_levels[{index}] must be an object")
        raw_leg = level.get("leg")
        if not isinstance(raw_leg, str):
            raise ValueError(f"event_id={event_id!r} setup.tp_levels[{index}].leg must be a string")
        leg = raw_leg.strip()
        if not leg:
            raise ValueError(f"event_id={event_id!r} setup.tp_levels[{index}].leg must be non-empty")
        key = leg.casefold()
        if key in seen_legs:
            raise ValueError(f"event_id={event_id!r} setup.tp_levels contains duplicate leg {leg!r}")
        seen_legs.add(key)

        for field in ("pnl_pct", "close_fraction"):
            raw = level.get(field)
            if isinstance(raw, bool):
                raise ValueError(f"event_id={event_id!r} setup.tp_levels[{index}].{field} must be numeric")
            try:
                number = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"event_id={event_id!r} setup.tp_levels[{index}].{field} must be numeric") from exc
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"event_id={event_id!r} setup.tp_levels[{index}].{field} must be finite and positive")


def validate_active_trade_record(event_id: Any, trade: Any) -> None:
    """Raise ValueError for nested shapes that make tracker operations unsafe.

    Missing optional legacy fields are tolerated because the tracker supplies defaults;
    explicitly present fields must have the type their consumers require. A malformed
    record is rejected before it can be used to reconcile a live exchange position.
    """
    if not isinstance(trade, dict):
        raise ValueError(f"event_id={event_id!r} must be an object")

    for field in _NON_NULL_FIELDS:
        if field in trade and trade[field] is None:
            raise ValueError(f"event_id={event_id!r} field={field!r} must not be null")

    for field in _DICT_FIELDS:
        if field in trade and trade[field] is not None and not isinstance(trade[field], dict):
            raise ValueError(
                f"event_id={event_id!r} field={field!r} must be an object, "
                f"got {type(trade[field]).__name__}"
            )
    if isinstance(trade.get("setup"), dict):
        _validate_setup_tp_levels(event_id, trade["setup"])

    for field in _LIST_FIELDS:
        if field in trade and trade[field] is not None and not isinstance(trade[field], list):
            raise ValueError(
                f"event_id={event_id!r} field={field!r} must be an array, "
                f"got {type(trade[field]).__name__}"
            )

    for field in _DICT_LIST_FIELDS:
        value = trade.get(field)
        if isinstance(value, list):
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    raise ValueError(
                        f"event_id={event_id!r} field={field!r}[{index}] must be an object, "
                        f"got {type(item).__name__}"
                    )

    hit_legs = trade.get("hit_legs")
    if isinstance(hit_legs, list) and any(not isinstance(item, str) for item in hit_legs):
        raise ValueError(f"event_id={event_id!r} field='hit_legs' must contain only strings")

    completed_legs = trade.get("completed_tp_legs")
    if isinstance(completed_legs, list) and any(not isinstance(item, str) for item in completed_legs):
        raise ValueError(f"event_id={event_id!r} field='completed_tp_legs' must contain only strings")

    filled_qty = trade.get("tp_filled_qty")
    if isinstance(filled_qty, dict):
        for leg, qty in filled_qty.items():
            if not isinstance(leg, str):
                raise ValueError(f"event_id={event_id!r} tp_filled_qty keys must be strings")
            if isinstance(qty, bool):
                raise ValueError(f"event_id={event_id!r} tp_filled_qty[{leg!r}] must be numeric")
            try:
                numeric_qty = float(qty)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"event_id={event_id!r} tp_filled_qty[{leg!r}] must be numeric") from exc
            if numeric_qty < 0 or numeric_qty != numeric_qty or numeric_qty in (float("inf"), float("-inf")):
                raise ValueError(f"event_id={event_id!r} tp_filled_qty[{leg!r}] must be finite and non-negative")
