from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCAN_JSONL = DATA_DIR / "scan_history.jsonl"
MAX_SCAN_HISTORY_BYTES = max(1_048_576, int(os.environ.get("MAX_SCAN_HISTORY_BYTES", str(30 * 1024 * 1024))))
SIGNALS_JSONL = DATA_DIR / "signal_history.jsonl"
LATEST_SCAN_JSON = DATA_DIR / "latest_scan.json"
LATEST_SCAN_TXT = DATA_DIR / "latest_scan.txt"


def _json_safe(value: Any) -> Any:
    """Recursively convert non-finite floats to JSON null values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if hasattr(value, "item"):
            item = value.item()
            if isinstance(item, float) and not math.isfinite(item):
                return None
            if item is not value:
                return _json_safe(item)
    except Exception:
        pass
    return value


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    safe_row = _json_safe(row)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(safe_row, ensure_ascii=False, default=str, allow_nan=False) + "\n")


def _rotate_scan_history() -> None:
    """Keep a bounded local scan journal without ever deleting the newest rows."""
    try:
        if not SCAN_JSONL.exists() or SCAN_JSONL.stat().st_size <= MAX_SCAN_HISTORY_BYTES:
            return
        data = SCAN_JSONL.read_bytes()
        # Retain only complete JSONL records from the newest tail. The next write
        # appends after this bounded snapshot.
        tail = data[-MAX_SCAN_HISTORY_BYTES:]
        first_newline = tail.find(b"\n")
        if first_newline >= 0:
            tail = tail[first_newline + 1:]
        tmp = SCAN_JSONL.with_suffix(SCAN_JSONL.suffix + ".tmp")
        tmp.write_bytes(tail)
        os.replace(tmp, SCAN_JSONL)
    except Exception as exc:
        # Analytics retention must never stop trading.
        return


def _atomic_json(path: Path, payload: Any) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    safe_payload = _json_safe(payload)
    tmp.write_text(json.dumps(safe_payload, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _line(row: dict[str, Any]) -> str:
    symbol = str(row.get("symbol") or "").upper().replace("-", "")
    return (
        f"Монета={symbol} | Цена={row.get('current_price')} | "
        f"Положение={row.get('price_position')} | Сигнал={row.get('fresh_signal', '—')} | "
        f"DEMAND={row.get('active_demand', 0)} | SUPPLY={row.get('active_supply', 0)} | "
        f"Source={row.get('market_source', 'binance_spot')} | "
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
        _rotate_scan_history()
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
