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
    for target_index, chat_id in enumerate(ids, start=1):
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
            # requests exceptions may embed the full request URL, which contains
            # the Telegram bot token. Never log exception text or chat IDs.
            print(
                f"[TELEGRAM] send failed target_index={target_index} "
                f"error={type(exc).__name__}"
            )
    return ok_all


def _esc(value: Any) -> str:
    return html.escape("—" if value is None or value == "" else str(value), quote=False)


def _code(value: Any) -> str:
    """Render a Telegram inline-code entity whose contents are safe to copy."""
    return f"<code>{_esc(value)}</code>"


def _code_escaped(value: str) -> str:
    """Wrap text already HTML-escaped by a formatter in a code entity."""
    return f"<code>{value}</code>"


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


def _fmt_distance_pct(value: Any) -> str:
    try:
        distance = float(value)
    except (TypeError, ValueError):
        return "—"
    if not (distance == distance) or distance in (float("inf"), float("-inf")) or distance < 0:
        return "—"
    # Keep ordinary distances compact while avoiding a misleading 0.00% for
    # small but non-zero separations between two valid zones.
    if distance >= 0.01:
        digits = 2
    elif distance >= 0.0001:
        digits = 4
    else:
        digits = 8
    return f"{distance:.{digits}f}".rstrip("0").rstrip(".") + "%"


def _fmt_mtf_zone_lines(value: Any) -> list[str]:
    """Render the 1h/3h zone snapshot as separate prices plus a clear relation."""
    one_hour = value.get("one_hour") if isinstance(value, dict) and isinstance(value.get("one_hour"), dict) else {}
    three_hour = value.get("three_hour") if isinstance(value, dict) and isinstance(value.get("three_hour"), dict) else {}
    kind = str(value.get("kind") or "").upper() if isinstance(value, dict) else ""
    if kind not in {"DEMAND", "SUPPLY"}:
        kind = "—"

    def zone_text(zone: dict[str, Any]) -> str:
        number, text = _decimal_price_text(zone.get("poi"))
        if number is None or number <= 0 or text is None or kind == "—":
            return "—"
        return f"{kind} {_fmt_price_grouped(text)}"

    one_hour_text = zone_text(one_hour)
    three_hour_text = zone_text(three_hour)
    lines = [
        f"Zone 1H: {one_hour_text}",
        f"Zone 3H: {three_hour_text}",
        "Calculation: —",
    ]

    if not isinstance(value, dict) or value.get("status") not in {"confluent", "not_confluent"}:
        return lines
    if one_hour_text == "—" or three_hour_text == "—":
        return lines

    distance_text = _fmt_distance_pct(value.get("poi_distance_pct"))
    if distance_text == "—":
        return lines
    if value.get("overlap") is True:
        relation = "OVERLAP"
    elif value.get("status") == "confluent":
        relation = "NEAR"
    else:
        relation = "FAR"
    lines[2] = f"Calculation: {relation} · {distance_text}"
    return lines


def _fmt_funding(funding: dict[str, Any] | None) -> str:
    if not isinstance(funding, dict):
        return "UNAVAILABLE"
    if funding.get("status") != "ok":
        status = str(funding.get("status") or "unavailable").strip().upper()
        return status if status in {"STALE", "UNVERIFIED", "UNAVAILABLE"} else "UNAVAILABLE"
    try:
        percent = float(funding.get("funding_rate_pct"))
    except (TypeError, ValueError):
        return "UNAVAILABLE"
    if not (percent == percent) or percent in (float("inf"), float("-inf")):
        return "UNAVAILABLE"
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
    icon = "🟢" if direction == "LONG" else "🔴"
    icon_url = (
        "https://web.telegram.org/k/assets/img/emoji/1f7e2.png"
        if direction == "LONG"
        else "https://web.telegram.org/k/assets/img/emoji/1f534.png"
    )
    execution = execution or {}
    zone = event.get("zone", {}) if isinstance(event.get("zone"), dict) else {}
    confirmation = event.get("confirmation", {}) if isinstance(event.get("confirmation"), dict) else {}
    target = event.get("target", {}) if isinstance(event.get("target"), dict) else {}
    tp1_pct_text = _fmt_num(target.get("tp1_pct"), 1) if target.get("tp1_pct") is not None else "—"
    tp2_pct_text = _fmt_num(target.get("tp2_pct"), 1) if target.get("tp2_pct") is not None else "—"
    tp3_pct_text = _fmt_num(target.get("tp3_pct"), 1) if target.get("tp3_pct") is not None else "—"
    tp1_fraction = target.get("tp1_close_fraction", 0.5)
    tp2_fraction = target.get("tp2_close_fraction", 0.25)
    tp3_fraction = target.get("tp3_close_fraction", 0.25)
    try:
        tp1_fraction_text = _fmt_num(float(tp1_fraction) * 100.0, 1)
    except (TypeError, ValueError):
        tp1_fraction_text = "50"
    try:
        tp2_fraction_text = _fmt_num(float(tp2_fraction) * 100.0, 1)
    except (TypeError, ValueError):
        tp2_fraction_text = "25"
    try:
        tp3_fraction_text = _fmt_num(float(tp3_fraction) * 100.0, 1)
    except (TypeError, ValueError):
        tp3_fraction_text = "25"
    score_value = score if score is not None else event.get("score")
    try:
        score_text = f"{float(score_value):.1f}" if score_value not in (None, "") else "—"
    except (TypeError, ValueError):
        score_text = "—"
    zone_kind = zone.get("kind") or ("DEMAND" if direction == "LONG" else "SUPPLY")
    volume_ratio = confirmation.get("volume_ratio")
    mtf = confirmation.get("mtf_zone_confluence")

    mtf_lines = []
    for mtf_line in _fmt_mtf_zone_lines(mtf):
        label, separator, value = mtf_line.partition(": ")
        mtf_lines.append(
            f"{label}: {_code_escaped(value)}" if separator else _esc(mtf_line)
        )

    tp1_value = (
        f"{_fmt_price(event.get('tp1'))} ({tp1_pct_text}% / "
        f"{_fmt_num(event.get('tp1_rr'), 2)}R / {tp1_fraction_text}%)"
    )
    tp2_value = (
        f"{_fmt_price(event.get('tp2'))} ({tp2_pct_text}% / "
        f"{_fmt_num(event.get('tp2_rr'), 2)}R / {tp2_fraction_text}%)"
    )
    tp3_value = (
        f"{_fmt_price(event.get('tp3'))} ({tp3_pct_text}% / "
        f"{_fmt_num(event.get('tp3_rr'), 2)}R / {tp3_fraction_text}%)"
    ) if event.get("tp3") is not None else None
    volume_value = _fmt_num(volume_ratio, 3) if volume_ratio not in (None, "") else "—"

    lines = [
        f'<a href="{icon_url}">{icon}</a> {"LONG" if direction == "LONG" else "SHORT"} · {_code(symbol)}',
        "",
        f"Score: {_code(score_text)}",
        f"Zone: {_code(zone_kind)}",
        f"Entry reference: {_code_escaped(_fmt_price(event.get('entry')))}",
        f"SL: {_code_escaped(_fmt_price(event.get('sl')))}",
        f"TP1: {_code_escaped(tp1_value)}",
        f"TP2: {_code_escaped(tp2_value)}",
        *([f"TP3: {_code_escaped(tp3_value)}"] if tp3_value is not None else []),
        f"Risk: {_code_escaped(_fmt_num(event.get('risk_pct'), 1) + '%')}",
        "",
        "CONFIRMATION",
        f"Volume / SMA20: {_code_escaped(volume_value)}",
        *mtf_lines,
        f"Funding: {_code(_fmt_funding(funding))}",
    ]
    if execution:
        order = execution.get("order") if isinstance(execution.get("order"), dict) else {}
        protection = execution.get("protection") if isinstance(execution.get("protection"), dict) else {}
        status_value = str(execution.get("status") or "unknown").strip()
        error_value = execution.get("error") or order.get("error") or protection.get("error")
        if error_value not in (None, ""):
            # Keep the notification to one status line. Never render a separate
            # Error field; escape the detail through _code below.
            error_text = " ".join(str(error_value).split())[:240]
            status_value = f"{status_value} · {error_text}"
        lines += [
            "",
            "EXECUTION",
            f"Status: {_code(status_value)}",
            f"Order: {_code(order.get('order_id') or order.get('client_order_id'))}",
            f"Protection: {_code(protection.get('status'))}",
        ]
    return "\n".join(lines)
