# Zone Engine runtime-state and retention operations

## Clean first start

A source-only checkout contains no trading state or research recovery seed. If no usable schema-2 runtime-state artifact exists, the workflow runs a guarded clean-start via `event_engine.clean_start` before tests or the engine. Older schema-1 state is never restored as current trading state. Bootstrap is permitted only when `BINGX_ENV=vst`, the configured base URL is the BingX VST host, private API credentials are present, and exchange responses prove there are zero non-zero positions, zero regular open orders, and zero open TWAP strategy orders. API errors, malformed responses, any position, or any open order stop bootstrap.

The first run is forced to `EXECUTION_ENABLED=false`; trading remains disabled for this run. It initializes empty runtime journals, runs the engine without opening trades, creates the runtime-state artifact, then downloads and checksum-verifies that artifact. Trading can be enabled only on a later run that restores a self-verifying runtime-state artifact from a completed, successful event-engine run (`event-engine.yml`) on `main`. The bundle manifest, per-file SHA-256 hashes, and critical JSON state are validated before restore. Artifacts from failed runs are not selected as restore sources; if no eligible successful artifact remains, the workflow fails closed rather than triggering an empty bootstrap. Operational caveat: an engine process that crashes after an exchange-side change but before state-artifact publication may leave the last successful snapshot behind the exchange. Startup reconciliation should block new entries when it detects an ownership mismatch, but any such crash requires exchange-side position/order verification before live trading resumes.

## Runtime-state artifact (`zone-engine-runtime-state`)

`event_engine.runtime_state_bundle` verifies a manifest and SHA-256 for every file, requires the critical execution/reconciliation/research files, rejects unexpected paths and symlinks, and restores `data/` transactionally. Runtime state is not silently replaced with an empty state when an artifact exists but is expired, ineligible, unavailable, or corrupt: the workflow fails closed.

The legacy `bootstrap/research_history_seed` path and the one-time `check-legacy` migration path are not supported. A clean start begins new research history. The forward research calculation reads current bars and retention-pruned deltas only; it does not import recovery seeds.

## Retention and compaction

The cumulative retention archive remains a separate artifact chain because the pruned deltas are needed to reconstruct research history. The workflow restores and verifies that archive before compaction, uploads changed archive contents before committing compacted journals, and refuses the state commit if archive recovery or upload verification is unsafe. The size guard must not delete audit-critical data merely to satisfy a size threshold; if safe compaction cannot reach the configured target, the workflow stops without committing the compacted journals.

## Operational validation

After changes, run the complete pytest suite and Python syntax checks. Before the first VST run, confirm the configured secrets point only to VST credentials and the account has no positions, regular open orders, or open TWAP orders. A successful local test suite does not replace the first-run artifact upload/download verification, and no live-trading readiness claim should be made until that workflow succeeds.
