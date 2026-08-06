# Cross-window robustness check: FTX (2022-11) replicates the LUNA (2022-05) leakage result

**Date:** Day 12/13
**Question:** Does the Phase 2 leakage-demonstration result generalize beyond
LUNA (2022-05), or was it specific to that particular event's dynamics?

## Setup

Reran the full real-vs-dummy comparison on BTCUSDT-aggTrades-2022-11 (the
FTX collapse, concentrated in November 2022), refitting the K=4 IMM, the
label horizon, and the null-shift construction fresh on this window's own
data -- not reusing any May-derived parameter.

**One real bug caught and fixed before this could run correctly:**
`build_phase2_dataset.py` had hardcoded `ELEVATED_STATES = [2, 3]` from
May's specific HMM fit. HMM state indices are arbitrary per EM run and
are not guaranteed to align the same way across different months' data --
this needed to be derived per-window by variance rank (the same logic
`k4_recheck.py` already used), not hardcoded. Confirmed working correctly
on the new window: states [1, 2] selected, with a clear variance
separation (1.925e-5 and 1.59e-6, both well above 2.7e-7 and 6e-8) -- a
genuine gap, not an arbitrary median split forced onto four similar
values.

## Result: the core finding replicates

| | LUNA (2022-05) | FTX (2022-11) |
|---|---|---|
| naive null diff (mean +/- std) | +0.349 +/- 0.118 | +0.292 +/- 0.149 |
| naive shifts positive | 20/20 (p~1e-6) | 20/20 (p~1e-6) |
| purged null diff (mean +/- std) | +0.043 +/- 0.055 | +0.008 +/- 0.060 |
| purged shifts positive | 14/20 (p~0.04) | **10/20 (coin flip)** |
| real label diff, naive (sigma above null) | ~3.7 sigma | ~4.0 sigma |
| real label diff, purged (sigma above null) | ~6.1 sigma | ~7.4 sigma |

**Naive k-fold's leakage signature is essentially identical across two
independent, structurally different crisis windows** -- same overwhelming
20/20 pattern, same order-of-magnitude effect size. This is strong
evidence the leakage demonstration is a real property of the
methodology, not an artifact specific to LUNA's particular dynamics.

**The real label's genuine signal replicates too, and slightly more
strongly** -- both sigma-above-baseline figures are higher on FTX than on
May, consistent with a real, generalizable finding rather than a fluke of
one month's data.

## Bonus: resolves `ISSUES.md` #3

May's purged-walk-forward residual (+0.043, 14/20 shifts positive,
p~0.04) was flagged as "suggestive but not conclusive" -- real leak, or
sampling noise at only 20 shifts? **FTX's purged residual is +0.008 with
exactly 10 of 20 shifts positive -- no directional signal whatsoever.**
Two independent windows, and only one shows anything resembling a
residual, at a magnitude now looking much more like sampling noise than a
genuine small leak surviving purging. Treated as effectively resolved:
purged/embargoed walk-forward removes the leakage to within noise, on
both windows tested.

## Reproducibility

```
python scripts/build_phase2_dataset.py data/raw/BTCUSDT-aggTrades-2022-11.zip
python scripts/phase2_real_vs_dummy_comparison.py data/processed/phase2_dataset_BTCUSDT-aggTrades-2022-11.parquet
```

## Next step

Cross-window check complete for one additional window (FTX). COVID
(2020-02/04) and May 2021 remain downloaded and unused if a third
confirmation is ever wanted, but two independent, structurally different
crisis windows replicating cleanly is already a strong result -- not
treated as blocking further progress.
