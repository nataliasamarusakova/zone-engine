# Safe data retention / Git push runbook

## 0. Pre-commit 90 MB size guard

Every GitHub Actions cycle runs the size guard **before `git add`/commit**. If any managed audit journal reaches 90,000,000 bytes, the guard snapshots the exact original bytes into `data/retention_archive/<UTC_RUN>/...`, then progressively archives rows until the working file is below 90 MB, targeting 80 MB. Active trade/execution state and trade/decision-linked audit records remain protected. For 5m market bars, if ordinary age-based passes cannot reach the target (for example, because timestamps are unparseable or too many recent rows remain), a final byte-budget policy keeps only the newest valid recent bars that fit the budget and archives the rest. This is lossless archival, not deletion: `research_forward.py` reads the immutable `*.pruned*.jsonl.gz` deltas on subsequent runs. The archive must be durably uploaded and verified before a compacted journal can be committed.

The workflow then stages `data/` and runs `python -m event_engine.data_retention --check-staged`. If **any staged file under `data/` is still >=90,000,000 bytes, the commit step is skipped and the workflow fails**. This prevents GitHub's 100 MB hard limit from being hit. GitHub enforces the 100 MB single-object limit in regular Git repositories. citeturn899502search1turn899502search5

This is intentionally a **pre-commit** guard. A workflow cannot repair a manual commit that has already been rejected by GitHub for exceeding the file-size limit; the cleanup must happen before the workflow's own commit/push.

This migration is designed for the **current live Zone Engine state**. It does **not** replay, open, close, cancel, or recreate trades.

## 1. Audit-chain safety guarantee

Before any JSONL file is compacted, the migration takes a **byte-faithful gzip snapshot of the complete original file** under the selected archive root.

In GitHub Actions the workflow explicitly selects `data/retention_archive` (ignored by Git):

```text
data/retention_archive/<UTC_RUN>/snapshots_before/<file>.before.jsonl.gz
```

When the command is run without `--archive-dir`, the implementation defaults to the external sibling directory `../zone-engine-data-archive`, keeping large recovery snapshots outside the repository working tree.

In GitHub Actions, `data/retention_archive/` is ignored by Git and is uploaded as a cumulative `zone-engine-retention-archive` artifact after the size guard when needed. The next run queries successful `event-engine.yml` runs on `main` and restores an eligible archive before further compaction. The artifact is configured for 90-day retention. To avoid uploading the full cumulative payload every cycle, `event_engine.retention_archive` fingerprints relative paths and file bytes; upload happens when the archive changes, upload metadata is missing, or the last upload is at least 45 days old. An unchanged recent archive is still restored and verified, but not uploaded again. If the prior chain is missing or APIs/downloads fail, compaction is paused. This mechanism must still be verified on the first real Actions run after deployment; local tests cannot prove GitHub's artifact round trip.

The manifest records the original SHA-256, byte count, and row count. Therefore the exact pre-migration file can be restored while its archive remains available.

If the size guard succeeds but the cumulative artifact upload fails, the workflow runs `python -m event_engine.data_retention --restore-pre-compaction-snapshots`. It validates **all** snapshots first (chunk paths must remain under `data/retention_archive/`, each compressed size must match, gzip must decode, and decompressed SHA-256/byte/row counts must equal the manifest). Only after every snapshot validates are the compacted JSONL targets atomically replaced by their originals. If validation or restoration fails, the commit gate fails closed. If rollback succeeds, the commit step uses `--check-staged --limit-bytes 100000000`; each staged file must be strictly smaller than 100,000,000 bytes, otherwise no commit is made. This rollback path is covered by local regression tests but still needs a real GitHub Actions failure-injection run.

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

- `market_bars_5m.jsonl`: normally keep the recent 36h working window; the size guard may escalate to 24h, 12h, 6h, 3h, and 1h, then use a byte-budgeted recent subset if needed. Removed rows stay in immutable archive deltas and remain available to `research_forward.py`; pending-observation state itself is never rewritten.
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
gzip -cd <ARCHIVE_ROOT>/<UTC_RUN>/snapshots_before/research_outcomes.jsonl.before.jsonl.gz \
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
git add .gitignore event_engine/data_retention.py event_engine/tracker.py test_all.py DATA_RETENTION_RUNBOOK.md

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

Large, mutable runtime state is persisted in the self-verifying **`zone-engine-runtime-state` GitHub Actions artifact**, not by creating a new Git snapshot commit every five minutes. The artifact includes active-trade/execution state, research cursors and outcomes, the working JSONL journals, and the immutable research-history seed. Its manifest contains a SHA-256 and byte size for every bundled file. The workflow verifies the complete inventory before restoring it and refuses to run the engine if the newest state artifact is unavailable or invalid after migration.

The first run after deploying this workflow may still perform the existing slow checkout, because the current `main` tree still contains the legacy data blobs. That run uses the complete tracked state, performs the normal quality and retention gates, uploads and verifies the runtime-state artifact, and only then removes the legacy data files from the Git index. Subsequent checkouts no longer need to download those large blobs. The workflow retains only the two newest runtime-state artifacts after a successful upload.

`data/retention_archive/` remains managed by its separate cumulative retention-artifact chain. It is intentionally excluded from the runtime-state artifact and must still be restored, verified, and uploaded before compacted journals are considered safe. Ephemeral scan history, diagnostic logs, Python caches, locks, and temporary files are not included in the runtime-state artifact.

After migration, `git ls-files data/` should be empty for runtime journals, and `git ls-files bootstrap/research_history_seed/` should also be empty. Do not manually stage runtime `data/` or the research-history seed; the workflow owns their artifact lifecycle. If the runtime-state artifact API/download or checksum validation fails, the workflow stops before trading rather than starting with empty or stale state.

Useful checks:

```bash
git status --short
git ls-files data/
git ls-files bootstrap/research_history_seed/
git check-ignore -v data/active_trades.json data/market_bars_5m.jsonl bootstrap/research_history_seed/2026-10-16/manifest.json
```

### Rebuild the unpushed local history safely

Because rejected local commits may already contain >100 MiB blobs, merely adding a later cleanup commit is insufficient for those oversized histories. Rebuild the local, not-yet-pushed history from `origin/main` while keeping the current working tree and live data intact:

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

## 8. Rejected legacy commit handling

For older local histories that already contain oversized runtime-data blobs, a later cleanup commit does not shrink those earlier Git objects. The workflow migration only removes the data blobs from the current `main` tree after the current runtime state has been successfully uploaded and verified as an artifact; it does not erase the historical record or the authoritative state.

Do not use `git reset --hard`, and do not manually delete active trades, research cursors, research outcome state, trades, or execution journals to solve the Git size problem. If the artifact migration has not completed, retain the legacy working data and let the workflow perform the migration only after the state artifact is durable.

## 9. Production logging fix included in this release

`TRACKER_TRADE_CLOSED` had a placeholder/argument mismatch that produced `TypeError: must be real number, not str` during otherwise successful trade-close logging. The format string now has the correct number and ordering of placeholders. This affects logging only; it does not change trade state or execution logic.

### Counterfactual experiment journal

`data/counterfactual_experiments.jsonl` is persistent audit telemetry. It contains observational experiment snapshots and matured counterfactual outcomes and is compacted with a 90-day working window while references linked to active/trade state are protected.

## Workflow performance profile (run 39127233428)

Measured runtime was about 6m32s. The production engine itself took about 254.8s: tracker update of the 20 active trades took 41.1s before scanning and 39.1s after; reconciliation took 21.7s and 20.2s respectively; scanning 330 symbols took 69.6s; deferred research persistence took 56.7s. The follow-up `research_forward.py --write` took about 26s, including 17.2s to evaluate 6,694 matured candidates (none had a complete enough price path to become ready outcomes). GitHub checkout/fetch took about 24s; commit/push took about 14s.

The old retention-artifact discovery consumed about 36s even though there was no artifact. The workflow now queries the artifacts endpoint directly and tries only the newest three candidates. Deferred 5m bar requests are fetched with concurrency capped at two while persistence remains serial. Verified existing SL orders reuse the already-successful `openOrders` snapshot; after any SL create/cleanup, the exchange re-query remains mandatory. These are conservative optimizations; expected savings are based on code-path counts, not yet measured in a new live workflow run.

The checked-out Git repository still contains tracked runtime state under `data/`, so checkout/push cost will remain until runtime persistence is migrated to a separate durable store. Do not delete remote `data/` while it contains active positions, execution claims, order IDs, journals, or cursors. The delivered source ZIP omits `data/`, but that does not mean the live repo's runtime directory can safely be deleted.

## One-time research outcome seed restoration

`bootstrap/research_history_seed/` contains immutable, hash-verified source journals from the pre-compaction snapshot. This directory is source recovery material and may ship with the source bundle even when the runtime `data/` directory is excluded. Before `run_once.py`, the workflow runs `restore_research_history_seed.py` to merge the recovery seed for `research_outcomes.jsonl` into the live outcome journal. The step verifies the decompressed byte count, SHA-256, row count, JSONL validity, and unique `outcome_id` values before changing the target. Conflicting IDs fail without rewriting the live journal.

The step writes `data/research_outcomes_restore_state.json` only after a verified merge or after confirming that every seed outcome is already present. This marker is committed with normal runtime state and survives size-guard compaction. It prevents the immutable old seed from being added back to `research_outcomes.jsonl` on every clean runner after a later compaction. The step has `continue-on-error: true` intentionally so a damaged/missing research seed cannot stop tracker/reconciliation work for existing VST positions; a warning is emitted and the data restore must be investigated.

## Защита от потери цепочки retention-архива

Workflow восстанавливает `zone-engine-retention-archive` только из артефактов, связанных с успешными запусками этого workflow на `main`: из artifact API берутся `workflow_run.id`, после чего metadata этих конкретных запусков проверяется точечным REST-запросом по workflow ID, ветке, событию и статусу. Так поиск не зависит от того, сколько общих запусков прошло с момента последнего upload. Скачиваются не более трёх самых новых подходящих архивов, без обхода истории всех запусков. Модуль `event_engine.retention_archive` вычисляет SHA-256 по путям и содержимому всех архивных файлов, поэтому повторная загрузка не нужна, если архив не изменился и предыдущая загрузка моложе 45 дней. Это снижает сетевые затраты и дублирование накопительного artifact storage; обновление выполняется при изменении архива и не позднее чем за 45 дней до 90-дневного истечения срока хранения.

Перед size-guard после восстановления каталог сверяется по fingerprint с `data/retention_archive_state.json`; несовпадение запрещает compaction. После успешного upload, если в `data/retention_archive/` действительно есть файлы, workflow атомарно создаёт `data/retention_archive_state.json`. Этот маркер защищён от переписывания механизмом retention и фиксирует, что цепочка архивирования ранее существовала. Он не содержит торговых данных и не заменяет сами архивы.

Если GitHub API недоступен, подходящий архив не найден/истёк или последние три кандидата не скачиваются, workflow выставляет `recovery_safe=false`: size-guard compaction и upload архива не запускаются. Обслуживание текущих позиций и обычные изменения состояния могут продолжаться, только если итоговые staged-файлы проходят лимит размера. Если архивные файлы уже присутствуют в checkout, отдельная проверка перед commit требует либо успешной загрузки изменившегося/просроченного архива и записи нового маркера, либо совпадения текущего отпечатка с последней успешной загрузкой не старше 45 дней; иначе commit блокируется. Это намеренно выбирает сохранность истории вместо рискованного уменьшения размера журнала.

Первый запуск, когда нет ни артефактов, ни маркера, разрешает начальное создание архива. После того как маркер создан, отсутствие действующей архивной копии больше не трактуется как чистый старт. При ручном удалении артефактов нужно восстановить архив из независимой резервной копии либо решить вопрос сохранности истории перед повторной compaction; не удаляйте маркер просто для обхода защиты.
