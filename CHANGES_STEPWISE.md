# Stepwise changes applied

## Step 1 — Execution telemetry (observational only)
- Added `event_engine/telemetry.py`.
- Added append-only JSONL telemetry for quotes, order lifecycle, protection lifecycle, position reconciliation and exchange errors.
- Linked records by `event_id`, `attempt_id`, `position_id`, `order_id` where available.
- Added exchange/local timestamps to quote telemetry.
- No changes to entry/SL/TP/BE behavior.

## Step 2 — SHORT shadow candidates (observational only)
- Added frozen experiment id `short_entry_filters_v1`.
- Added candidate thresholds:
  - `body_to_range > 0.70`
  - `lower_wick_ratio < 0.05`
  - BTC 5m close above EMA50 by `0.25%` / `0.50%`.
- Added combined shadow labels in research features.
- No production blocking gate was added.

## Step 3 — Exchange/protection error telemetry
- Recorded protection-preflight exchange errors (including BingX `100410`), order rejects and position reconciliation errors.
- Recorded TP/SL order-query and protection-order errors with lifecycle identifiers.
- Added telemetry schema documentation.
- Test runtime state is isolated from repository production data.

## Step 4 — BingX bookTicker parser compatibility
- Updated futures top-of-book parser to accept current `data.book_ticker` envelope as well as legacy flat/list payloads.
- Added regression coverage for the nested envelope.
- This addresses the observed `bookTicker` -> `ticker` fallback pattern without changing the fallback safety behavior.

## Step 5 — Authoritative final quote telemetry
- `open_market()` now accepts optional `attempt_id` and passes it into final quote telemetry.
- The exact venue quote that passed the final pre-POST freshness/drift gate is preserved in memory and journaled immediately **after** the MARKET POST attempt, avoiding telemetry I/O between the final gate and submission.
- Final quote telemetry includes `execution_reference_price`, `signal_drift_pct` and the actual `order_submit_at_ms`.
- Existing `bookTicker`/fallback execution behavior is unchanged.

## Step 6 — Telemetry write health
- Telemetry write failures are no longer completely silent.
- Added a process-local `health_snapshot()` with failure count, last failure timestamp and exception type.
- Added rate-limited operator warning logs; telemetry remains best-effort and cannot block or alter trading.

## Validation
- Full test suite: **232 passed**.
- `py_compile` passed for production and test Python modules.
- Production entry, fixed 10% stop, TP1/TP2 and BE rules were not changed by these steps.

## Step 7 — Safe data retention migration (no trade replay)
- Added `event_engine/data_retention.py` as a one-time, idempotent data compaction tool.
- Default retention:
  - 5m market bars: 36h, with per-symbol extension for still-unprocessed research observations.
  - 1h market bars: 168h.
  - `NEAREST_APPROACH` observations: 36h; all signal/rejection/rearm/activation observations are retained.
- Before compaction, the tool verifies pending research observations and preserves their required 5m history instead of deleting it.
- Removed rows are gzip-archived outside the repository; active/trade/research state is never rewritten.
- Uses the same research file locks as the writer to avoid races while the engine is operating.
- Refuses destructive 5m compaction when the journal is stale and would otherwise be emptied.
- Added regression tests for state preservation and stale-data refusal.
- Added `.gitignore` rules for ephemeral logs, scan snapshots, Python caches and lock/temp files.
- No entry, SL, TP, BE or position execution behavior changed.

### Migration command
```bash
python -m event_engine.data_retention --apply
```
Run once before the commit/push. The command does not connect to BingX and does not open/close/cancel any trade.

## 2026-09-23 — retention/recovery hardening

- Fixed `TRACKER_TRADE_CLOSED` logging placeholder/argument mismatch observed in production logs.
- Added byte-faithful pre-compaction gzip snapshots for every JSONL file touched by retention.
- Added SHA-256/size/row-count verification before replacing a working file.
- Added event-aware retention for `research_outcomes.jsonl`.
- Added event-aware retention for `position_reconciliation.jsonl`: active positions and anomalies/terminal records are retained; repetitive healthy snapshots for old closed positions are archived.
- Changed retention compaction to stream JSONL instead of loading entire files into RAM.
- Protected production state files are checksum-verified and never rewritten.
- Added regression tests proving the full pre-compaction dataset remains recoverable from snapshots.
- Full test suite: 236 passed.

## v6 — pre-commit data-size guard

- Added a pre-commit size guard for persistent JSONL data.
- Guard threshold: 90,000,000 bytes; target: 80,000,000 bytes.
- The guard runs before `git add`/commit and therefore prevents oversized blobs from entering a new commit.
- When triggered, the guard first creates a lossless chunked gzip snapshot under `data/retention_archive/<UTC_RUN>/` and then applies progressively stronger, audit-safe retention policies.
- Pending research observations, active-trade references, and trade/decision-linked audit records are preserved.
- Position reconciliation compaction preserves active positions, anomalies, terminal records, and the latest healthy FOUND snapshot for old closed positions.
- After staging, `--check-staged` fails closed if any `data/` file is still at or above 90 MB; the commit step is skipped in that case.
- `data/` remains tracked in Git; only scan-history and diagnostic log files remain ignored.

## Step 7 — Research forward performance hardening
- Optimized `research_forward.py` to avoid per-observation DataFrame copies, boolean rescans and `iterrows()` loops.
- Added a reusable numeric `_BarIndex` for each symbol/provider 5m series.
- Streams `zone_observations.jsonl` and only materializes mature, unprocessed observations.
- Limits market-bar DataFrames to candidate symbols.
- Uses `research_outcome_state.json` as the idempotency source after a one-time legacy outcome bootstrap.
- Preserves all pre-existing keys in `research_outcome_state.json` during bootstrap/update.
- Added progress/timing logs for observation scan, bar load and outcome calculation.
- Added a 3-minute GitHub Actions timeout for the research-outcome step; timeout/failure does not block safe state commits, and a warning is emitted.
- No entry, execution, SL, TP or BE logic changed.

## v8 — incremental research-forward observation cursor

- Added an append-only `zone_observations.jsonl` cursor stored in `research_outcome_state.json`.
- Subsequent `research_forward.py --write` runs read only the newly appended observation tail plus a bounded pending set instead of rescanning the full observation journal.
- Mature-but-not-yet-computable observations are retained by byte offset and revisited later.
- Cursor validity is protected by file size, inode, prefix hash, and the hash of the bytes immediately preceding the saved offset; retention/rewrite therefore triggers a safe full rebuild.
- Existing `processed_observation_ids` and all unknown state keys are preserved.
- No trade/execution/SL/TP/BE logic changed.
- Full test suite: 247 passed.

## Step: remove internal GitHub Actions schedule

- Removed `schedule:` / `cron` from `.github/workflows/event-engine.yml`.
- Kept only `repository_dispatch` (`run_event_engine`) as the workflow trigger.
- Updated `test_workflow_stages_all_data_but_excludes_scan_history` to require the external-dispatch model and explicitly reject the internal cron.
- No trading, execution, retention, or research-forward logic changed in this step.

## Step 9 — safe pending-observation archival and chunked size guard

- `research_forward.py` now stores replayable pending-observation metadata in `research_outcome_state.json` (`pending_observations_v1`, schema v2).
- After journal compaction/truncation, full rebuild preserves replayable pending state instead of discarding it.
- `data_retention.py` may archive old `zone_observations.jsonl` rows across all event types; the raw source remains recoverable from a verified pre-compaction snapshot.
- Large pruned deltas are chunked using the same 50 MB raw-byte safety boundary as source snapshots.
- Retention remains fail-safe if the remaining working window itself cannot fit under the 90 MB guard.
