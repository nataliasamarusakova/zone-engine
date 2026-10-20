from __future__ import annotations

"""Guarded first-run bootstrap for an empty BingX VST account.

This module is only for a repository with no previously verified runtime-state
artifact. It refuses to initialize state unless the account can be queried on
BingX VST and both positions and open orders are proven empty.
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from event_engine import bingx
from event_engine.runtime_state_bundle import REQUIRED_FILES
from event_engine.version import ENGINE_VERSION


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _position_is_open(position: dict[str, Any]) -> bool:
    raw = position.get("positionAmt")
    if raw is None:
        raise RuntimeError(f"Cannot prove position is flat: positionAmt missing in response {position!r}")
    try:
        amount = float(raw or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse positionAmt in response {position!r}") from exc
    if not math.isfinite(amount):
        raise RuntimeError(f"Cannot prove position is flat: non-finite positionAmt in response {position!r}")
    return abs(amount) > 0.0


def verify_vst_account_is_flat() -> None:
    env = os.environ.get("BINGX_ENV", "").strip().lower()
    base_url = os.environ.get("BINGX_BASE_URL", "").strip()
    parsed_url = urlparse(base_url)
    if env != "vst" or parsed_url.scheme.lower() != "https" or (parsed_url.hostname or "").lower() != "open-api-vst.bingx.com":
        raise RuntimeError("Clean-start bootstrap is permitted only when BINGX_ENV=vst and BINGX_BASE_URL is exactly the HTTPS BingX VST host")
    if not os.environ.get("BINGX_API_KEY", "").strip() or not os.environ.get("BINGX_SECRET_KEY", "").strip():
        raise RuntimeError("Clean-start bootstrap requires BingX VST API credentials")

    # A network/API failure must propagate; an empty fallback is never accepted.
    positions = bingx.get_positions(timeout_sec=5, retryable=False)
    open_positions = [item for item in positions if _position_is_open(item)]
    if open_positions:
        raise RuntimeError(f"Clean-start refused: BingX VST reports {len(open_positions)} non-zero position(s)")

    orders = bingx.get_all_open_orders(timeout_sec=5, retryable=False)
    if orders:
        raise RuntimeError(f"Clean-start refused: BingX VST reports {len(orders)} open order(s)")
    twap_orders = bingx.get_all_open_twap_orders(timeout_sec=5, retryable=False)
    if twap_orders:
        raise RuntimeError(f"Clean-start refused: BingX VST reports {len(twap_orders)} open TWAP order(s)")
    print("Clean-start account guard passed: BingX VST positions=0 open_orders=0 open_twap_orders=0")


def initialize_empty_runtime(data_dir: Path) -> None:
    """Create the minimum complete, schema-safe runtime inventory for run one."""
    data_dir = Path(data_dir)
    # Check the path before resolving it; resolve() would hide a symlink and could
    # otherwise make a clean start delete a directory outside the checkout.
    if data_dir.is_symlink():
        raise RuntimeError(f"Refusing to replace symlinked runtime data directory: {data_dir}")
    data_dir = data_dir.resolve()
    # This is intentionally destructive and is called only after the VST account
    # has been proven flat. Do not mix a clean bootstrap with tracked legacy state.
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    json_defaults: dict[str, Any] = {
        "active_trades.json": {},
        "event_execution_claims.json": {},
        "failed_signals.json": {},
        "latest_scan.json": {},
        "zone_visit_state.json": {"version": 1, "trigger_mode": os.environ.get("ZONE_TRIGGER_MODE", "zone"), "symbols": {}},
        "research_bar_cursors.json": {},
        "research_manifest.json": {"schema_version": 3, "research_enabled": True},
        "research_outcome_state.json": {"schema_version": 1, "processed_observation_ids": [], "pending_observations_v1": {}},
        "data_retention_manifest.json": {"schema_version": 1, "mode": "clean_start", "files": {}},
        "runtime_version.json": {"engine_version": ENGINE_VERSION, "runtime_initialized": True},
    }
    jsonl_names = {
        Path(rel).name for rel in REQUIRED_FILES
        if rel.startswith("data/") and Path(rel).suffix == ".jsonl"
    }
    for name in jsonl_names:
        path = data_dir / name
        if not path.exists():
            path.touch()
    for name in ("latest_scan.txt",):
        path = data_dir / name
        if not path.exists():
            path.write_text("Clean-start: no prior scan.\n", encoding="utf-8")
    for name, payload in json_defaults.items():
        path = data_dir / name
        if not path.exists():
            _write_json(path, payload)

    missing = []
    for relative in REQUIRED_FILES:
        if not relative.startswith("data/"):
            continue
        path = data_dir / Path(relative).relative_to("data")
        if not path.is_file() or path.is_symlink():
            missing.append(relative)
    if missing:
        raise RuntimeError("Clean-start runtime inventory is incomplete: " + ", ".join(missing))
    print(f"Initialized clean runtime state: files={len(list(data_dir.rglob('*')))}; no legacy data or recovery seed imported")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    verify_vst_account_is_flat()
    initialize_empty_runtime(args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
