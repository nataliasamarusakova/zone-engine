from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path("data")
SCAN_JSONL = DATA_DIR / "scan_history.jsonl"
SIGNALS_JSONL = DATA_DIR / "signal_history.jsonl"
LATEST_SCAN_JSON = DATA_DIR / "latest_scan.json"
LATEST_SCAN_TXT = DATA_DIR / "latest_scan.txt"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _atomic_json(path: Path, payload: Any) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _line(row: dict[str, Any]) -> str:
    return (
        f"Монета={row.get('symbol')} | Цена={row.get('current_price')} | "
        f"Положение={row.get('price_position')} | Сигнал={row.get('fresh_signal', '—')} | "
        f"DEMAND={row.get('active_demand', 0)} | SUPPLY={row.get('active_supply', 0)} | "
        f"Source={row.get('market_source', 'binance_futures')} | "
        f"Binance={row.get('binance_price')} | BingX={row.get('bingx_price')} | "
        f"Spread={row.get('market_spread_pct')}% | Asset={row.get('asset_class', 'UNKNOWN')}"
    )


def save_scan(scan_rows: list[dict[str, Any]], signals: list[dict[str, Any]], *, duration_sec: float, scan_id: str) -> str:
    now = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for row in scan_rows:
        record = {"scan_id": scan_id, "ts": now, **row}
        rows.append(record)
        _append_jsonl(SCAN_JSONL, record)
    for signal in signals:
        _append_jsonl(SIGNALS_JSONL, {"scan_id": scan_id, "ts": now, **signal})

    lines = [
        f"SCAN {scan_id}",
        f"UTC={now}",
        f"Symbols={len(scan_rows)} | Signals={len(signals)} | Duration={float(duration_sec):.3f}s",
        "",
    ]
    lines.extend(_line(r) for r in scan_rows)
    text = "\n".join(lines) + "\n"

    snapshot = {
        "scan_id": scan_id,
        "ts": now,
        "duration_sec": round(float(duration_sec), 3),
        "symbols": len(scan_rows),
        "signals": len(signals),
        "rows": rows,
        "signals_detail": signals,
        "text": text,
    }
    _atomic_json(LATEST_SCAN_JSON, snapshot)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_SCAN_TXT.write_text(text, encoding="utf-8")
    return text
