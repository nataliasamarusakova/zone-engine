v5.48.5 shadow methodology: session block is 00:00<=UTC<06:00; delayed-BE/trailing experiments are management-only against current production SL/TP and return weighted trade-level PnL.

# Shadow / Counterfactual Telemetry v5.48.5

## Purpose

The shadow layer records and later simulates proposed entry/stop/target/management ideas without changing production trading behavior.

Every shadow experiment is marked `applied=false`. No production execution gate reads these values.

## Entry snapshot

`data/counterfactual_experiments.jsonl` receives one immutable `COUNTERFACTUAL_SNAPSHOT` per `event_id`.

The snapshot records:

- structural stop: zone far edge + 0.5× 5m ATR, including 1–6% risk bounds;
- hypothetical TP1=1.5R and TP2=3R from that structural risk;
- correctly dimensioned 5m/5m and 1h/1h volume ratios and threshold grids;
- trigger-bar delta confirmation;
- 1H EMA200 alignment and distance;
- 5m ATR(14) percent and minimum-volatility grid;
- UTC session;
- zone body-close invalidation state before entry;
- available open-position symbols for later correlation analysis;
- actual taker/maker fee rates when account context is available;
- delayed-BE and ATR-trailing experiment specifications.

## Forward outcome

`research_forward.py` uses the persisted 5m market-bar journal to calculate counterfactual outcomes after the forward path matures.

Matured results are written as `COUNTERFACTUAL_OUTCOME` records in the same journal and copied into the normal `research_outcomes.jsonl` row under `counterfactual_experiments`.

The production strategy is unchanged.

## Important semantics

- 5m volume ratio = trigger 5m volume / mean of the previous 20 completed 5m bars. The trigger bar is not included in its own baseline.
- 1h volume ratio = latest fully closed 1h volume known at decision time / mean of the previous 20 completed 1h bars.
- 1H EMA200 uses only fully closed 1h candles available at the signal timestamp.
- 5m ATR(14) uses closed bars through the trigger bar.
- Post-entry counterfactual path starts after the trigger candle close.
- Same-bar SL/TP ambiguity is explicitly reported instead of pretending the intrabar order is known.
- Correlation is deferred when peer-symbol market bars are unavailable at decision time; the open-position symbols are preserved so the audit can compute it from the persisted market-bar journal.
- Cost scenarios are observational only. The slippage scenario field is explicitly round-trip and is added to the two-sided taker fee estimate.

## Audit workflow

1. Join `COUNTERFACTUAL_SNAPSHOT` and `COUNTERFACTUAL_OUTCOME` by `event_id` / `observation_id`.
2. Compare each individual experiment to the actual outcome.
3. For filters, calculate losses removed, winners removed, PnL retained/lost, trade frequency and stability.
4. For structural SL/TP/trailing/BE experiments, use the simulated exit and path fields rather than recomputing from post-entry data manually.
5. Do not promote a shadow rule directly to production. Validate it chronologically and across symbols/directions first.
