from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tabulate import tabulate

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


def save_scan(scan_rows: list[dict[str, Any]], signals: list[dict[str, Any]], *, duration_sec: float, scan_id: str) -> str:
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for row in scan_rows:
        record = {"scan_id": scan_id, "ts": now, **row}
        rows.append(record)
        _append_jsonl(SCAN_JSONL, record)
    for signal in signals:
        _append_jsonl(SIGNALS_JSONL, {"scan_id": scan_id, "ts": now, **signal})

    table_rows = [
        [
            r["symbol"],
            r["current_price"],
            r["price_position"],
            r["fresh_signal"],
            r["active_demand"],
            r["active_supply"],
        ]
        for r in scan_rows
    ]
    headers = ["Монета", "Текущая цена", "Положение цены", "Свежий сигнал", "Активных зон DEMAND", "Активных зон SUPPLY"]
    table = tabulate(table_rows, headers=headers, tablefmt="fancy_grid", showindex=False)

    snapshot = {
        "scan_id": scan_id,
        "ts": now,
        "duration_sec": round(float(duration_sec), 3),
        "symbols": len(scan_rows),
        "signals": len(signals),
        "table": table,
        "rows": scan_rows,
        "signals_detail": signals,
    }
    _atomic_json(LATEST_SCAN_JSON, snapshot)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_SCAN_TXT.write_text(table + "\n", encoding="utf-8")
    return table
