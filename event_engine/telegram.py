from __future__ import annotations

import html
import os
from decimal import Decimal, InvalidOperation
from typing import Any

import requests


def _chat_ids() -> list[str]:
    raw = os.environ.get("TG_CHAT_IDS") or os.environ.get("TG_CHAT_ID") or ""
    # Secret values are frequently copied/merged during workflow maintenance.
    # Preserve target order but never deliver the same notification twice to the
    # same Telegram chat because its ID was accidentally listed more than once.
    result: list[str] = []
    seen: set[str] = set()
    for value in raw.replace(";", ",").split(","):
        chat_id = value.strip()
        if chat_id and chat_id not in seen:
            seen.add(chat_id)
            result.append(chat_id)
    return result


def send(text: str) -> bool:
    token = os.environ.get("TG_BOT_TOKEN", "").strip()
    ids = _chat_ids()
    if not token or not ids:
        print("[TELEGRAM] missing TG_BOT_TOKEN or TG_CHAT_IDS")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    ok_all = True
    for chat_id in ids:
        try:
            response = requests.post(
                url,
                data={"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": "true"},
                timeout=15,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok"):
                ok_all = False
                print(f"[TELEGRAM] rejected: {payload.get('description', 'unknown')}")
        except Exception as exc:
            ok_all = False
            print(f"[TELEGRAM] send failed chat_id={chat_id}: {exc}")
    return ok_all


def _esc(value: Any) -> str:
    return html.escape("—" if value is None or value == "" else str(value), quote=False)


def _decimal_price_text(value: Any) -> tuple[Decimal | None, str | None]:
    """Return a finite decimal and its exact plain-string representation.

    Exchange prices must not be rounded to an arbitrary number of significant
    digits for display: that can move a visible entry/SL/TP by several ticks.
    Decimal(str(value)) preserves the shortest decimal representation supplied
    by the exchange/parser without exposing binary-float artifacts.
    """
    if value is None or value == "" or isinstance(value, bool):
        return None, None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None, None
    if not number.is_finite():
        return None, None
    if number.is_zero():
        return number, "0"
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    # Guard unexpected pathological values from producing oversized chat text.
    if len(text) > 80:
        return None, None
    return number, text


def _fmt_price(value: Any) -> str:
    """Render a price without arbitrary rounding or scientific notation."""
    _, text = _decimal_price_text(value)
    return _esc(text if text is not None else value)


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if not (number == number) or number in (float("inf"), float("-inf")):
        return _esc(value)
    return _esc(f"{number:.{digits}f}".rstrip("0").rstrip("."))


def _fmt_price_grouped(value: Any) -> str:
    number, text = _decimal_price_text(value)
    if number is None or text is None:
        return _esc(value)
    if abs(number) >= 1000:
        integer, separator, fraction = text.partition(".")
        try:
            integer = f"{int(integer):,}"
        except ValueError:
            return _esc(value)
        text = integer + (separator + fraction if separator else "")
    return _esc(text)


def _fmt_confluence(value: Any) -> str:
    if not isinstance(value, dict) or value.get("status") not in {"confluent", "not_confluent"}:
        return "—"
    kind = str(value.get("kind") or "").upper()
    one_hour = value.get("one_hour") if isinstance(value.get("one_hour"), dict) else {}
    three_hour = value.get("three_hour") if isinstance(value.get("three_hour"), dict) else {}
    poi_1h = one_hour.get("poi")
    poi_3h = three_hour.get("poi")
    if poi_1h is None or poi_3h is None:
        return "—"
    try:
        distance = float(value.get("poi_distance_pct"))
    except (TypeError, ValueError):
        return "—"
    if not (distance == distance) or distance in (float("inf"), float("-inf")):
        return "—"
    distance_label = "overlap" if value.get("overlap") else f"{distance:.2f}%"
    if value.get("status") == "not_confluent":
        distance_label += ", far"
    return (
        f"{_esc(kind)} {_fmt_price_grouped(poi_1h)} ↔ "
        f"{_esc(kind)} {_fmt_price_grouped(poi_3h)} ({_esc(distance_label)})"
    )


def _fmt_funding(funding: dict[str, Any] | None) -> str:
    if not isinstance(funding, dict) or funding.get("status") != "ok":
        return "—"
    try:
        percent = float(funding.get("funding_rate_pct"))
    except (TypeError, ValueError):
        return "—"
    if not (percent == percent) or percent in (float("inf"), float("-inf")):
        return "—"
    return f"{percent:+.4f}%"


def format_signal(
    event: dict[str, Any],
    setup: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
    score: float | None = None,
    funding: dict[str, Any] | None = None,
    **_: Any,
) -> str:
    """Render the compact production entry notice; no research/debug fields leak into chat."""
    direction = str(event.get("type") or event.get("direction") or "").upper()
    symbol = str(event.get("symbol", "")).replace("-", "").upper()
    icon = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    execution = execution or {}
    zone = event.get("zone", {}) if isinstance(event.get("zone"), dict) else {}
    confirmation = event.get("confirmation", {}) if isinstance(event.get("confirmation"), dict) else {}
    target = event.get("target", {}) if isinstance(event.get("target"), dict) else {}
    tp1_pct_text = _fmt_num(target.get("tp1_pct"), 1) if target.get("tp1_pct") is not None else "—"
    tp2_pct_text = _fmt_num(target.get("tp2_pct"), 1) if target.get("tp2_pct") is not None else "—"
    tp1_fraction = target.get("tp1_close_fraction", 0.5)
    tp2_fraction = target.get("tp2_close_fraction", 0.5)
    try:
        tp1_fraction_text = _fmt_num(float(tp1_fraction) * 100.0, 1)
    except (TypeError, ValueError):
        tp1_fraction_text = "50"
    try:
        tp2_fraction_text = _fmt_num(float(tp2_fraction) * 100.0, 1)
    except (TypeError, ValueError):
        tp2_fraction_text = "50"
    score_value = score if score is not None else event.get("score")
    try:
        score_text = f"{float(score_value):.1f}" if score_value not in (None, "") else "—"
    except (TypeError, ValueError):
        score_text = "—"
    zone_kind = zone.get("kind") or ("DEMAND" if direction == "LONG" else "SUPPLY")
    volume_ratio = confirmation.get("volume_ratio")
    mtf = confirmation.get("mtf_zone_confluence")

    lines = [
        f"<b>{icon} · {_esc(symbol)}</b>",
        "",
        f"Score: <b>{_esc(score_text)}</b>",
        f"Zone: <b>{_esc(zone_kind)}</b>",
        f"Entry reference: <code>{_fmt_price(event.get('entry'))}</code>",
        f"SL: <code>{_fmt_price(event.get('sl'))}</code>",
        f"TP1: <code>{_fmt_price(event.get('tp1'))}</code> ({tp1_pct_text}% / {_fmt_num(event.get('tp1_rr'), 2)}R / {tp1_fraction_text}%)",
        f"TP2: <code>{_fmt_price(event.get('tp2'))}</code> ({tp2_pct_text}% / {_fmt_num(event.get('tp2_rr'), 2)}R / {tp2_fraction_text}%)",
        f"Risk: <code>{_fmt_num(event.get('risk_pct'), 1)}%</code>",
        "",
        "<b>CONFIRMATION</b>",
        f"Volume / SMA20: <code>{_fmt_num(volume_ratio, 3) if volume_ratio not in (None, '') else '—'}</code>",
        f"1H / 3H zones: <code>{_fmt_confluence(mtf)}</code>",
        f"Funding (last): <code>{_esc(_fmt_funding(funding))}</code>",
    ]
    if execution:
        order = execution.get("order") if isinstance(execution.get("order"), dict) else {}
        protection = execution.get("protection") if isinstance(execution.get("protection"), dict) else {}
        lines += [
            "",
            "<b>EXECUTION</b>",
            f"Status: <code>{_esc(execution.get('status'))}</code>",
            f"Order: <code>{_esc(order.get('order_id') or order.get('client_order_id'))}</code>",
            f"Protection: <code>{_esc(protection.get('status'))}</code>",
            f"Error: <code>{_esc(execution.get('error') or order.get('error'))}</code>",
        ]
    return "\n".join(lines)
