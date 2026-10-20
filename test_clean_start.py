from __future__ import annotations

import json
from pathlib import Path

import pytest

from event_engine import clean_start
from event_engine.runtime_state_bundle import REQUIRED_FILES


def test_clean_start_requires_explicit_vst_configuration(monkeypatch):
    monkeypatch.setenv("BINGX_ENV", "live")
    monkeypatch.setenv("BINGX_BASE_URL", "https://open-api.bingx.com")
    monkeypatch.setenv("BINGX_API_KEY", "test-key")
    monkeypatch.setenv("BINGX_SECRET_KEY", "test-secret")
    with pytest.raises(RuntimeError, match="only when BINGX_ENV=vst"):
        clean_start.verify_vst_account_is_flat()


def test_clean_start_rejects_lookalike_vst_host(monkeypatch):
    monkeypatch.setenv("BINGX_ENV", "vst")
    monkeypatch.setenv("BINGX_BASE_URL", "https://open-api-vst.bingx.com.attacker.invalid")
    monkeypatch.setenv("BINGX_API_KEY", "test-key")
    monkeypatch.setenv("BINGX_SECRET_KEY", "test-secret")
    with pytest.raises(RuntimeError, match="exactly the HTTPS BingX VST host"):
        clean_start.verify_vst_account_is_flat()


def test_clean_start_fails_closed_when_positions_exist(monkeypatch):
    monkeypatch.setenv("BINGX_ENV", "vst")
    monkeypatch.setenv("BINGX_BASE_URL", "https://open-api-vst.bingx.com")
    monkeypatch.setenv("BINGX_API_KEY", "test-key")
    monkeypatch.setenv("BINGX_SECRET_KEY", "test-secret")
    monkeypatch.setattr(clean_start.bingx, "get_positions", lambda **kwargs: [{"symbol": "BTC-USDT", "positionAmt": "0.01"}])
    monkeypatch.setattr(clean_start.bingx, "get_all_open_orders", lambda **kwargs: [])
    monkeypatch.setattr(clean_start.bingx, "get_all_open_twap_orders", lambda **kwargs: [])
    with pytest.raises(RuntimeError, match="reports 1 non-zero position"):
        clean_start.verify_vst_account_is_flat()


def test_clean_start_fails_closed_when_position_query_errors(monkeypatch):
    monkeypatch.setenv("BINGX_ENV", "vst")
    monkeypatch.setenv("BINGX_BASE_URL", "https://open-api-vst.bingx.com")
    monkeypatch.setenv("BINGX_API_KEY", "test-key")
    monkeypatch.setenv("BINGX_SECRET_KEY", "test-secret")
    monkeypatch.setattr(clean_start.bingx, "get_positions", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("API down")))
    with pytest.raises(RuntimeError, match="API down"):
        clean_start.verify_vst_account_is_flat()


def test_clean_start_fails_closed_when_open_orders_exist(monkeypatch):
    monkeypatch.setenv("BINGX_ENV", "vst")
    monkeypatch.setenv("BINGX_BASE_URL", "https://open-api-vst.bingx.com")
    monkeypatch.setenv("BINGX_API_KEY", "test-key")
    monkeypatch.setenv("BINGX_SECRET_KEY", "test-secret")
    monkeypatch.setattr(clean_start.bingx, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(clean_start.bingx, "get_all_open_orders", lambda **kwargs: [{"symbol": "BTC-USDT", "orderId": "1"}])
    monkeypatch.setattr(clean_start.bingx, "get_all_open_twap_orders", lambda **kwargs: [])
    with pytest.raises(RuntimeError, match="reports 1 open order"):
        clean_start.verify_vst_account_is_flat()


def test_clean_start_fails_closed_when_open_twap_orders_exist(monkeypatch):
    monkeypatch.setenv("BINGX_ENV", "vst")
    monkeypatch.setenv("BINGX_BASE_URL", "https://open-api-vst.bingx.com")
    monkeypatch.setenv("BINGX_API_KEY", "test-key")
    monkeypatch.setenv("BINGX_SECRET_KEY", "test-secret")
    monkeypatch.setattr(clean_start.bingx, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(clean_start.bingx, "get_all_open_orders", lambda **kwargs: [])
    monkeypatch.setattr(clean_start.bingx, "get_all_open_twap_orders", lambda **kwargs: [{"mainOrderId": "TWAP1"}])
    with pytest.raises(RuntimeError, match="reports 1 open TWAP order"):
        clean_start.verify_vst_account_is_flat()


def test_clean_start_initializes_required_files_without_recovery_seeds(tmp_path):
    data = tmp_path / "data"
    clean_start.initialize_empty_runtime(data)
    assert (data / ".gitkeep").is_file()
    for relative in REQUIRED_FILES:
        assert (data / Path(relative).relative_to("data")).is_file(), relative
    assert json.loads((data / "active_trades.json").read_text(encoding="utf-8")) == {}
    assert json.loads((data / "zone_visit_state.json").read_text(encoding="utf-8"))["symbols"] == {}
    assert not (tmp_path / "bootstrap").exists()
    assert not (data / "research_history_seed").exists()


def test_clean_start_refuses_unparseable_or_nonfinite_position_quantity():
    with pytest.raises(RuntimeError, match="Cannot parse positionAmt"):
        clean_start._position_is_open({"symbol": "BTC-USDT", "positionAmt": "unknown"})
    with pytest.raises(RuntimeError, match="non-finite positionAmt"):
        clean_start._position_is_open({"symbol": "BTC-USDT", "positionAmt": "NaN"})


def test_clean_start_replaces_existing_state_only_when_initializer_is_called(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text('{"OLD": {}}', encoding="utf-8")
    (data / "old-recovery-seed.jsonl").write_text("old\n", encoding="utf-8")
    clean_start.initialize_empty_runtime(data)
    assert json.loads((data / "active_trades.json").read_text(encoding="utf-8")) == {}
    assert not (data / "old-recovery-seed.jsonl").exists()


def test_bingx_clean_start_queries_fail_closed_on_unrecognized_account_response(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "_request", lambda *args, **kwargs: {"code": 0, "data": {"unexpected": []}})
    with pytest.raises(RuntimeError, match="unrecognized response shape"):
        bingx.get_positions(timeout_sec=1, retryable=False)
    with pytest.raises(RuntimeError, match="unrecognized response shape"):
        bingx.get_all_open_orders(timeout_sec=1, retryable=False)
    with pytest.raises(RuntimeError, match="unrecognized response shape"):
        bingx.get_all_open_twap_orders(timeout_sec=1, retryable=False)


def test_bingx_clean_start_uses_account_wide_open_orders_endpoint(monkeypatch):
    from event_engine import bingx

    calls = []
    monkeypatch.setattr(bingx, "_request", lambda method, path, params=None, **kwargs: calls.append((method, path, params)) or {"code": 0, "data": []})
    assert bingx.get_positions(timeout_sec=1, retryable=False) == []
    assert bingx.get_all_open_orders(timeout_sec=1, retryable=False) == []
    assert bingx.get_all_open_twap_orders(timeout_sec=1, retryable=False) == []
    assert calls[0][1] == bingx.POSITION_PATH
    assert calls[1] == ("GET", bingx.OPEN_ORDERS_PATH, {})
    assert calls[2] == ("GET", bingx.TWAP_OPEN_ORDERS_PATH, {})


def test_clean_start_refuses_to_delete_through_data_directory_symlink(tmp_path):
    target = tmp_path / "real-data"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    link = tmp_path / "data"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinked runtime data directory"):
        clean_start.initialize_empty_runtime(link)
    assert marker.read_text(encoding="utf-8") == "keep"
