# Runtime-state and retention-archive runbook

This workflow persists the engine's mutable state outside Git. The source repository
contains code and configuration; `data/` is runtime state and is ignored except for
`data/.gitkeep`. Treat the runtime-state artifact and the cumulative retention
archive as two separate, checksum-verified recovery chains.

## Runtime-state artifact (`zone-engine-runtime-state`)

The workflow publishes a self-verifying runtime-state artifact. The standard restore
candidate is from a completed, successful event-engine run.

1. The `event-engine.yml` workflow is serialized with the `zone-engine` concurrency
group and `cancel-in-progress: false`. A runner first queries the Actions artifact
API for unexpired `zone-engine-runtime-state` artifacts produced by this workflow
on `main`, excluding its own run. It requires the producing run to be `completed`.
2. The normal restore path selects an artifact from a completed, successful event-engine run. A late workflow failure is an explicit exception: an artifact
from that run is eligible only when the engine run, runtime-state bundle creation,
artifact durability verification, and download/checksum verification steps all
completed successfully. Runtime-state artifacts from failed runs are not selected
when those state-critical checks did not all succeed. This exception preserves a
newer, already-verified state snapshot if a later, unrelated workflow step fails.
3. The runner downloads the selected artifact and calls
`event_engine/runtime_state_bundle.py restore`. The manifest identifies the
`zone-engine-runtime-state` format and schema; every bundled file is checksum
verified before the data directory is replaced. A missing, malformed, unsupported,
or unverifiable artifact is not replaced with guessed empty state: the workflow
fails closed.
4. The uploaded artifact is separately checked for visibility in GitHub, downloaded
again, and verified file-by-file. Only after those checks may the workflow prune
older snapshots; it retains the two newest verified runtime-state artifacts.
5. A guarded clean-start is a bootstrap path, not a restore. It is used only when
there is no usable prior state artifact (or an older supported migration schema
requests bootstrap). The bootstrap requires `BINGX_ENV=vst`, the exact HTTPS BingX
VST host, credentials, and authoritative confirmation that positions, regular
open orders, and TWAP orders are all empty. API errors or unknown position amounts
abort the bootstrap. The engine is run with `EXECUTION_ENABLED=false` for that
bootstrap run: **trading remains disabled** until a later run restores the newly
created and verified runtime-state artifact.

The artifact contains the required trade, execution, reconciliation, research,
telemetry, retention-manifest, and runtime-version files listed in
`event_engine/runtime_state_bundle.py`. It intentionally excludes temporary files,
local diagnostics, and the retention archive payload, which has its own artifact
chain. The manifest records each included path, byte size, and SHA-256 digest.

## Cumulative retention archive (`zone-engine-retention-archive`)

The retention archive preserves pre-compaction journal data and is independent of
the runtime-state bundle. The `retention_archive_state.json` marker stores the
archive fingerprint and `last_uploaded_by_run_id`, identifying the workflow run
that uploaded the exact archive content represented by that fingerprint. Restore
uses that run ID to locate the corresponding unexpired artifact rather than
blindly selecting the latest successful workflow run; this avoids restoring an
older archive when a newer upload and marker update were completed before a later
unrelated workflow failure.

Candidates must belong to the event-engine workflow on `main`, be completed, and
have a successful run conclusion or the specifically verified archive durability
steps. For the marker's run, the archive-upload and marker-update steps both completed successfully; this is the required evidence when the overall workflow later
failed. The downloaded archive is fingerprint-verified against the marker before
it replaces the local archive. If the marker points to an unavailable artifact,
the fingerprint does not match, or the GitHub API cannot establish a safe chain,
retention compaction/upload is paused; the workflow does not silently start a new
archive chain. Position management can continue when only this retention recovery
is unavailable.

After a successful engine run and the quality/runtime-data gates, the pre-commit
size guard runs only when the cumulative archive was safely restored and verified.
The configured guard is 90,000,000 bytes and the target is 80,000,000 bytes. Before
compacting eligible journal rows, the guard writes recoverable pre-compaction
snapshots into the archive. The archive fingerprint is then checked, an upload is
performed when content changed or the 45-day refresh is due, and the marker is
updated only after the artifact upload succeeds. The marker update re-fingerprints
the archive and refuses to commit a different byte set.

If the archive upload or marker update cannot be completed, the workflow restores
the pre-compaction journal snapshots and the commit gate refuses to commit a
compacted state without a durable matching archive. Artifact retention is 90 days;
the refresh interval is shorter so the archive is refreshed before expiration.

## Operational checks

- Inspect the `Restore latest durable runtime-state artifact` step and its selected
  run ID before allowing execution. A restore/API/checksum failure is fail-closed.
- Inspect the `Restore fingerprint-matched cumulative retention archive` and
  `Verify restored retention archive fingerprint` steps before relying on compaction.
- Check `last_uploaded_archive_sha256`, `last_uploaded_at`, and
  `last_uploaded_by_run_id` together. Do not manually advance the marker without
  uploading the matching archive bytes first.
- Keep `data/` out of source archives and Git commits. Never replace a missing
  runtime-state artifact with an empty `active_trades.json` or empty journals.
