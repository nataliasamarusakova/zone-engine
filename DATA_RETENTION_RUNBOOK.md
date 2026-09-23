# Safe data retention / Git push runbook

## 0. Pre-commit 90 MB size guard

Every GitHub Actions cycle runs the size guard **before `git add`/commit**. If any managed audit journal reaches 90,000,000 bytes, the guard snapshots the exact original bytes into `data/retention_archive/<UTC_RUN>/...`, then progressively archives older safe-to-prune rows until the working file is below 90 MB, targeting 80 MB. Pending observations, active trades, and trade/decision-linked records are protected.

The workflow then stages `data/` and runs `python -m event_engine.data_retention --check-staged`. If **any staged file under `data/` is still >=90,000,000 bytes, the commit step is skipped and the workflow fails**. This prevents GitHub's 100 MB hard limit from being hit. GitHub enforces the 100 MB single-object limit in regular Git repositories. citeturn899502search1turn899502search5

This is intentionally a **pre-commit** guard. A workflow cannot repair a manual commit that has already been rejected by GitHub for exceeding the file-size limit; the cleanup must happen before the workflow's own commit/push.

This migration is designed for the **current live Zone Engine state**. It does **not** replay, open, close, cancel, or recreate trades.

## 1. Audit-chain safety guarantee

Before any JSONL file is compacted, the migration takes a **byte-faithful gzip snapshot of the complete original file** under the external archive root:

```text
zone-engine-data-archive/<UTC_RUN>/snapshots_before/<file>.before.jsonl.gz
```

The manifest records the original SHA-256, byte count, and row count. Therefore the exact pre-migration file can be restored even if a later analyst needs the full historical chain.

Pruned rows are also written to a smaller `*.pruned.jsonl.gz` delta archive for convenience.

## 2. Never-rewrite production state

These files are not rewritten by retention and their SHA-256 hashes are checked before/after:

- `data/trades.jsonl`
- `data/active_trades.json`
- `data/entry_decisions.jsonl`
- `data/execution_ledger.jsonl`
- `data/account_context.jsonl`
- `data/failed_signals.json`
- `data/research_outcome_state.json`
- `data/research_bar_cursors.json`
- `data/research_manifest.json`
- `data/zone_visit_state.json`
- `data/event_execution_claims.json`
- `data/actions.jsonl`
- `data/signal_history.jsonl`

These are the operational anchors needed to reconstruct the production lifecycle.

## 3. Files compacted safely

### Market bars

- `market_bars_5m.jsonl`: keep 36h, plus older per-symbol history required by unprocessed research observations.
- `market_bars_1h.jsonl`: keep 168h.

### Zone observations

Only old `NEAREST_APPROACH` observations are eligible for archival. Signal/rejection/rearm/activation observations remain in the working journal.

### Research outcomes

`research_outcomes.jsonl` is **not deleted**. Mature outcomes older than the configured 90-day working retention are moved to the external archive. Outcomes linked to pending observations or currently active trade references remain in the working file.

The processed observation state remains untouched, so archived outcomes are not regenerated or replayed.

### Position reconciliation

`position_reconciliation.jsonl` is compacted only for **old, closed positions**:

- active positions: all reconciliation rows remain;
- recently closed positions: all rows remain for the configured 14-day retention;
- old closed positions: anomaly/terminal records remain in the working journal, plus the last healthy `FOUND` snapshot;
- older repetitive healthy `FOUND` snapshots are moved to the archive.

Unknown/unlinked positions are retained conservatively.

## 4. Safe execution

First inspect without changing data:

```bash
python -m event_engine.data_retention
```

Then apply:

```bash
python -m event_engine.data_retention --apply
```

The migration uses the same file locks as the normal research writer. It snapshots a file while holding that file's lock and only replaces the working file after the snapshot has been verified against the original SHA-256 and byte count.

If the latest 5m data is stale beyond the configured safety window, the migration refuses destructive compaction.

## 5. Recovery / restoring the exact pre-migration file

The `snapshots_before` file is the authoritative recovery copy.

Example:

```bash
gzip -cd /path/to/zone-engine-data-archive/<UTC_RUN>/snapshots_before/research_outcomes.jsonl.before.jsonl.gz \
  > data/research_outcomes.jsonl.restore
```

Then validate its SHA-256 against `data_retention_manifest.json` before replacing anything.

Do **not** restore directly over live state without first stopping the process and validating the manifest.

## 6. Verify before commit

```bash
python -m pytest -q
python -m py_compile event_engine/data_retention.py run_once.py research_forward.py event_engine/*.py
```

Check the retention manifest:

```bash
cat data/data_retention_manifest.json
```

For files intended to remain tracked by Git, inspect their staged size:

```bash
git add .gitignore event_engine/data_retention.py event_engine/tracker.py test_all.py CHANGES_STEPWISE.md DATA_RETENTION_RUNBOOK.md

git cat-file -s :data/market_bars_5m.jsonl 2>/dev/null || true
git cat-file -s :data/zone_observations.jsonl 2>/dev/null || true
git cat-file -s :data/research_outcomes.jsonl 2>/dev/null || true
git cat-file -s :data/position_reconciliation.jsonl 2>/dev/null || true
```

Any individual file at or above `100000000` bytes will still be rejected by GitHub. The retention manifest reports files near or above that threshold.

If an audit journal is still too large after the default retention window, it is safe to shorten the working retention because the migration has already created an exact external snapshot. Example for a 7-day working window:

```bash
DATA_RETENTION_OUTCOME_HOURS=168 DATA_RETENTION_RECON_HOURS=168 \
  python -m event_engine.data_retention --apply
```

This does not delete the older records: they remain in the timestamped external archive.

## 7. Git policy for runtime data

The entire `data/` directory is runtime/research state and is intentionally **not tracked by Git**. Retention still compacts it locally and creates an external byte-faithful archive. This prevents GitHub's 100 MiB object limit from ever becoming a deployment blocker while preserving the full audit dataset outside Git. GitHub recommends storing programmatically generated files outside Git; files over 100 MiB are blocked in regular Git repositories.

The working `data/` directory must remain on the live machine. Do not run `git clean -fd` against it.

After retention has been applied, verify that no runtime data is staged:

```bash
git status --short
git diff --cached --name-only
git ls-files data/
```

`git ls-files data/` should return only `data/.gitkeep` (or nothing if the directory is not needed in the repository).

### Rebuild the unpushed local history safely

Because rejected local commits may already contain >100 MiB blobs, merely adding a later cleanup commit is insufficient. Rebuild the local, not-yet-pushed history from `origin/main` while keeping the current working tree and live data intact:

```bash
git fetch origin
git branch backup-before-data-git-cleanup
git reset --soft origin/main
git restore --staged data
git add -A
git status --short
git ls-files data/

# Run the test suite before creating the replacement commit.
python -m pytest -q

git commit -m "zone engine snapshot $(date -u +%Y-%m-%dT%H:%M:%SZ)"
git push origin main
```

This does **not** delete or reset live `data/`; `--soft` only rewrites the local commit graph. The backup branch preserves the old local commits until the push is confirmed.

If `git ls-files data/` still shows real JSONL/JSON files, stop and do not push.

## 7. Rejected commit handling

The rejected push happened on local commit `627ebe9`.

If the oversized data blobs were introduced by that unpushed commit and were not present in `HEAD^`, **amend the existing commit rather than creating another commit on top**:

```bash
git ls-tree -l HEAD^ -- data/market_bars_5m.jsonl data/zone_observations.jsonl

git commit --amend --no-edit
git push origin main
```

Do not use `git reset --hard`.

Do not delete `active_trades.json`, research cursors, research outcome state, trades, or execution journals as a way to solve the Git size problem.

## 8. Production logging fix included in this release

`TRACKER_TRADE_CLOSED` had a placeholder/argument mismatch that produced `TypeError: must be real number, not str` during otherwise successful trade-close logging. The format string now has the correct number and ordering of placeholders. This affects logging only; it does not change trade state or execution logic.
