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


