from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_repository_research_data(monkeypatch, tmp_path):
    """Never allow pytest research tests to mutate the repository's production data/."""
    from event_engine import research

    runtime_data = tmp_path / "research-data"
    # Do not create this directory up-front: individual tests may intentionally
    # construct their own tmp/data directory for retention-policy assertions.
    research_paths = {
        "DATA_DIR": runtime_data,
        "ZONE_OBSERVATIONS_PATH": runtime_data / "zone_observations.jsonl",
        "ENTRY_DECISIONS_PATH": runtime_data / "entry_decisions.jsonl",
        "MARKET_BARS_1H_PATH": runtime_data / "market_bars_1h.jsonl",
        "MARKET_BARS_5M_PATH": runtime_data / "market_bars_5m.jsonl",
        "RESEARCH_BAR_CURSORS_PATH": runtime_data / "research_bar_cursors.json",
        "RESEARCH_MANIFEST_PATH": runtime_data / "research_manifest.json",
        "RESEARCH_ERRORS_PATH": runtime_data / "research_persistence_errors.jsonl",
        "RESEARCH_OUTCOMES_PATH": runtime_data / "research_outcomes.jsonl",
        "RESEARCH_OUTCOME_STATE_PATH": runtime_data / "research_outcome_state.json",
        "COUNTERFACTUAL_EXPERIMENTS_PATH": runtime_data / "counterfactual_experiments.jsonl",
        "MARKET_CONTEXT_PATH": runtime_data / "market_context.jsonl",
        "ACCOUNT_CONTEXT_PATH": runtime_data / "account_context.jsonl",
    }
    for name, value in research_paths.items():
        monkeypatch.setattr(research, name, value)
