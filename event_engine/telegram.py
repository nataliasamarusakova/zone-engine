from __future__ import annotations

import html
import os
from typing import Any

import requests


def _chat_ids() -> list[str]:
    raw = os.environ.get("TG_CHAT_IDS") or os.environ.get("TG_CHAT_ID") or ""
    return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]


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


def _fmt_price(value: Any) -> str:
    """Human-readable price with no binary-float tail or scientific notation."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if not (number == number) or number in (float("inf"), float("-inf")):
        return _esc(value)
    text = format(number, ".6g")
    if "e" in text.lower():
        exponent = int(text.lower().split("e", 1)[1])
        decimals = max(0, min(14, 5 - exponent))
        text = f"{number:.{decimals}f}".rstrip("0").rstrip(".")
    return _esc(text)


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if not (number == number) or number in (float("inf"), float("-inf")):
        return _esc(value)
    return _esc(f"{number:.{digits}f}".rstrip("0").rstrip("."))


def format_signal(event: dict[str, Any], setup: dict[str, Any] | None = None, execution: dict[str, Any] | None = None, score: float | None = None, **_: Any) -> str:
    direction = str(event.get("type") or event.get("direction") or "").upper()
    symbol = str(event.get("symbol", "")).replace("-", "").upper()
    icon = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    setup = setup or {}
    execution = execution or {}
    zone = event.get("zone", {}) if isinstance(event.get("zone"), dict) else {}
    confirmation = event.get("confirmation", {}) if isinstance(event.get("confirmation"), dict) else {}
    counts = event.get("zone_counts", {}) if isinstance(event.get("zone_counts"), dict) else {}
    if not counts:
        setup_counts = setup.get("zone_counts", {}) if isinstance(setup.get("zone_counts"), dict) else {}
        counts = {
            "demand": setup_counts.get("demand", setup.get("active_demand", 0)),
            "supply": setup_counts.get("supply", setup.get("active_supply", 0)),
        }
    demand_count = counts.get("demand", 0)
    supply_count = counts.get("supply", 0)

    def _ratio(value: Any) -> str:
        return _fmt_num(value, 2) if value not in (None, "") else "—"

    target = event.get("target", {}) if isinstance(event.get("target"), dict) else {}
    tp1_pct_text = _fmt_num(target.get("tp1_pct"), 1) if target.get("tp1_pct") is not None else "—"
    tp2_pct_text = _fmt_num(target.get("tp2_pct"), 1) if target.get("tp2_pct") is not None else "—"

    lines = [
        f"<b>{icon} · {_esc(symbol)}</b>",
        "",
        f"Score: <b>{_esc(f'{score:.0f}/100' if score is not None else event.get('score'))}</b>",
        f"Zone: <b>{_esc(zone.get('kind'))}</b>",
        "Signal: <code>Demand/Supply Zone First · zone touch</code>",
        f"Entry reference: <code>{_fmt_price(event.get('entry'))}</code>",
        f"SL: <code>{_fmt_price(event.get('sl'))}</code>",
        f"TP1: <code>{_fmt_price(event.get('tp1'))}</code> ({tp1_pct_text}% / {_ratio(event.get('tp1_rr'))}R / 50%)",
        f"TP2: <code>{_fmt_price(event.get('tp2'))}</code> ({tp2_pct_text}% / {_ratio(event.get('tp2_rr'))}R / 50%)",
        f"Risk: <code>{_fmt_num(event.get('risk_pct'), 1)}%</code>",
        "",
        "<b>ZONE</b>",
        f"Demand zones: <code>{_esc(demand_count)}</code>",
        f"Supply zones: <code>{_esc(supply_count)}</code>",
        f"Bottom: <code>{_fmt_price(zone.get('btm'))}</code>",
        f"Top: <code>{_fmt_price(zone.get('top'))}</code>",
        f"POI: <code>{_fmt_price(zone.get('poi'))}</code>",
        f"Age: <code>{_esc(zone.get('age_bars'))}</code> bars",
        f"Impulse: <code>{_fmt_num(zone.get('impulse_atr'), 3) if zone.get('impulse_atr') not in (None, '') else '—'}</code> ATR",
        "",
        "<b>CONFIRMATION</b>",
        f"Volume / SMA20: <code>{_fmt_num(confirmation.get('volume_ratio'), 3) if confirmation.get('volume_ratio') not in (None, '') else '—'}</code>",
        f"Candle body / ATR: <code>{_fmt_num(confirmation.get('candle_body_atr'), 3) if confirmation.get('candle_body_atr') not in (None, '') else '—'}</code>",
        f"Range / ATR: <code>{_fmt_num(confirmation.get('range_atr'), 3) if confirmation.get('range_atr') not in (None, '') else '—'}</code>",
    ]
    if execution:
        lines += [
            "",
            "<b>EXECUTION</b>",
            f"Status: <code>{_esc(execution.get('status'))}</code>",
            f"Order: <code>{_esc((execution.get('order') or {}).get('order_id') or (execution.get('order') or {}).get('client_order_id'))}</code>",
            f"Protection: <code>{_esc((execution.get('protection') or {}).get('status'))}</code>",
            f"Error: <code>{_esc(execution.get('error') or (execution.get('order') or {}).get('error'))}</code>",
        ]
    return "\n".join(lines)
