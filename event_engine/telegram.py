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


def format_signal(event: dict[str, Any], setup: dict[str, Any] | None = None, execution: dict[str, Any] | None = None, score: float | None = None, **_: Any) -> str:
    direction = str(event.get("type") or event.get("direction") or "").upper()
    symbol = str(event.get("symbol", ""))
    icon = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"
    setup = setup or {}
    execution = execution or {}
    zone = event.get("zone", {}) if isinstance(event.get("zone"), dict) else {}
    confirmation = event.get("confirmation", {}) if isinstance(event.get("confirmation"), dict) else {}

    lines = [
        f"<b>{icon} · {_esc(symbol)}</b>",
        "",
        f"Score: <b>{_esc(f'{score:.0f}/100' if score is not None else event.get('score'))}</b>",
        f"Zone: <b>{_esc(zone.get('kind'))}</b>",
        f"Signal: <code>HMA 5/13 CROSS + ZONE TOUCH</code>",
        f"Entry reference: <code>{_esc(event.get('entry'))}</code>",
        f"SL: <code>{_esc(event.get('sl'))}</code>",
        f"TP1: <code>{_esc(event.get('tp1'))}</code> (1.0R / 50%)",
        f"TP2: <code>{_esc(event.get('tp2'))}</code> (2.0R / 50%)",
        f"Risk: <code>{_esc(event.get('risk_pct'))}%</code>",
        "",
        "<b>ZONE</b>",
        f"Bottom: <code>{_esc(zone.get('btm'))}</code>",
        f"Top: <code>{_esc(zone.get('top'))}</code>",
        f"POI: <code>{_esc(zone.get('poi'))}</code>",
        f"Age: <code>{_esc(zone.get('age_bars'))}</code> bars",
        f"Impulse: <code>{_esc(zone.get('impulse_atr'))}</code> ATR",
        "",
        "<b>CONFIRMATION</b>",
        f"Volume / SMA20: <code>{_esc(confirmation.get('volume_ratio'))}x</code>",
        f"Candle body / ATR: <code>{_esc(confirmation.get('candle_body_atr'))}x</code>",
        f"Range / ATR: <code>{_esc(confirmation.get('range_atr'))}x</code>",
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
