# v5.48.5 — Forensic correctness hardening

- Isolate pytest runtime writes from production data/.
- Preserve unknown quote age explicitly with a configurable VST-safe policy.
- Return None for unknown/invalid elapsed durations instead of 0.0.
- Correct weighted RR to include realized and remaining legs.
- Add reconciliation statuses for side mismatch and orphan local/exchange positions.
- Persist runtime version metadata without rewriting historical records.

# v5.48.4 — Shadow methodology correctness

- Fixed shadow Asia-session hypothesis to the exact `00:00 <= UTC < 06:00` interval.
- Added explicit production management baseline (current production SL/TP1/TP2 and 50% TP1 fraction).
- Delayed-BE and ATR-trailing counterfactuals now run as management-only simulations against production SL/TP, isolating management effects from structural-SL/1.5R scenarios.
- Management counterfactual outcomes now return weighted trade-level PnL including the 50% TP1 realization and residual 50% exit.
- Delayed-BE activation is based on completed bars after TP1 rather than activating one bar too early.
- Dynamic ATR trailing uses the persisted pre-entry + post-entry market-bar history available to the research layer.
- Added regression tests for exact session boundaries, production management isolation and weighted counterfactual PnL.
- Production trading logic remains unchanged; every shadow experiment is observational-only (`applied=false`).

Validation: 278 tests passed; compileall passed.

# v5.48.3 — Shadow / Counterfactual Telemetry

- Added an observational-only counterfactual telemetry layer. Every production signal gets a versioned, immutable shadow snapshot; no shadow value is read by production execution gates.
- Added correctly dimensioned 5m/5m and 1h/1h volume ratios; the trigger bar is excluded from its own volume baseline.
- Added shadow calculations for structural SL, 1.5R/3R targets, delta confirmation, 1H EMA200 alignment, 5m ATR minimum-volatility grids, session, zone invalidation, fee/cost scenarios, delayed BE and ATR trailing.
- Added `data/counterfactual_experiments.jsonl` for snapshot/outcome lineage. Matured counterfactual outcomes are also embedded in `research_outcomes.jsonl`.
- Reused the existing persisted 5m/1h market-bar journals for future path simulation; no duplicate per-trade candle journal is required.
- Backfilled the currently stored SIGNAL_CREATED observations and both active VST trades with shadow telemetry while leaving all real entry/fill/SL/TP/BE values unchanged.
- No production strategy rule, sizing, stop, target, BE or execution gate was changed.

Validation: 273 tests passed; compileall passed.

# v5.48.1 — State persistence fix

- Persistent VST engine/reconciliation state is now committed between GitHub Actions runs.
- Unbounded/ephemeral scan history, diagnostics and retention archives remain ignored.
- Added a clean-checkout persistence integration test.
- `ENGINE_VERSION` is `5.48.1`; strategy version is unchanged.

# v5.48 — State ownership, protection correctness and quote provenance hardening

- Runtime `data/` is intentionally absent from the release archive; it is created on first runtime use. Persistent engine/reconciliation state is committed between VST GitHub Actions runs, while unbounded/ephemeral artifacts remain ignored.
- Local active-trade ownership is now exclusive per `(symbol, direction)` and protection updates are keyed by `event_id`.
- Reconciliation refuses to mutate protection when a live exchange position has zero or multiple local owners.
- TP fills are economically validated against entry; an adverse-side fill is classified as an anomaly and can never trigger BE.
- Exchange order IDs are checked for cross-trade ownership conflicts before active state is persisted.
- Execution quote telemetry records both exchange timestamp age (when BingX provides it) and local observation age. Because the current BingX bookTicker contract documentation documents bid/ask but not a quote timestamp, exchange-timestamp enforcement is opt-in via `EXECUTION_REQUIRE_EXCHANGE_TIMESTAMP=true`; default execution remains local-observation based rather than silently blocking all entries.
- Runtime telemetry records `engine_version=5.48.1` for this patch release.
- Existing strategy version remains unchanged because no new entry/exit edge is promoted in this release.

## Validation
- Full regression suite must pass before deployment.
- Release package is built from source only; no runtime journals, locks, caches or prior audit artifacts are included.

# Historical v5 — Safe runtime-data/Git migration

- This historical migration release temporarily kept runtime `data/` out of Git while retention/recovery safety was hardened.
- The current v5.48.1 policy supersedes that temporary state: persistent engine/reconciliation state is committed between VST GitHub Actions runs, while unbounded/ephemeral artifacts remain ignored.
- Existing live data is preserved on disk and by the retention external archive.
- Retention remains event-aware for market bars, zone observations, research outcomes and position reconciliation.
- Unpushed rejected commits containing >100 MiB blobs must be rebuilt from `origin/main` before push.
- `TRACKER_TRADE_CLOSED` logging fix is included.
- Regression suite: 238 passed.

### v9 retention/research safety
- Pending research observations retain the minimal replay payload needed to calculate forward outcomes after raw journal compaction.
- `zone_observations.jsonl` size guard can archive old records across all event types while preserving a verified full snapshot.
- Pruned archive deltas are chunked to avoid creating oversized Git objects.
- Full test suite: 250 passed.



# v5.48.2 — telemetry/accounting correctness

- Persist `code_commit_sha` from the setup into `active_trades.json`, so the close record can retain exact engine provenance.
- When `allFillOrders` is unavailable, `fetch_research_account_snapshot()` now uses BingX `REALIZED_PNL` income records as an explicit recent-realized-PnL fallback instead of leaving recent realized PnL unclassified.
- Added explicit `recent_realized_pnl_source` / `recent_fill_source` fields and preserved the original fill-endpoint error for auditability.
- Position reconciliation now emits `FOUND_QTY_MISMATCH` when local and exchange quantities differ materially instead of labeling the state simply `FOUND`.
- No trading strategy, entry, sizing, SL, TP, BE, ownership or order-submission rules were changed.

# v5.48.2 — existing-state provenance backfill

- On startup, missing `code_commit_sha` for an already-open local trade is deterministically restored from its matching immutable `TRADE_OPEN` journal record by `event_id`.
- This migration is local-only and does not modify exchange state or submit/cancel orders.

## v5.48.3 — Shadow / Counterfactual Telemetry

- Added observational-only counterfactual experiment snapshots; no production gate reads these fields.
- Added correctly dimensioned 5m/5m and 1h/1h volume ratios.
- Added shadow calculations for structural SL, 1.5R/3R targets, delta confirmation, 1H EMA200 alignment, 5m ATR thresholds, session, zone invalidation state, fees/cost scenarios, delayed BE and ATR trailing.
- Added persisted `counterfactual_experiments.jsonl` for immutable experiment snapshots and matured outcomes.
- Existing 5m/1h research bars remain the source for post-entry counterfactual path simulation; no duplicate trade-path journal is required.
- Production strategy, entry gates, stop, TP, BE and trailing behavior are unchanged.
