# v5 — Safe runtime-data/Git migration

- Runtime `data/` is no longer tracked by Git.
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
