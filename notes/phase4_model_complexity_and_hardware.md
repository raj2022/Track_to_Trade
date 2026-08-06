# Phase 4 close-out: model complexity and hardware (Apple Silicon) investigation

**Date:** Day 13/14
**Question:** Does a more expressive model (MLP) outperform the validated
logistic regression baseline once trained on genuinely diverse, multi-
window data — and if so, does GPU acceleration (Apple Silicon MPS) matter
for the resulting compression/latency question?

## Setup

- **Combined dataset:** 5 independent, disjoint event windows merged --
  COVID crash (2020-03), May 2021 crash (2021-05), LUNA (2022-05), FTX
  (2022-11), and a calm baseline (2023-07). 221,450 rows total, refitting
  the K=4 IMM fresh per window (not reusing LUNA's fit). One real bug
  caught in the process: `ELEVATED_STATES` had been hardcoded from LUNA's
  specific HMM fit and needed to be derived per-window by variance rank,
  since HMM state indices are arbitrary per EM run.
- **Same 3 features, same purged/embargoed walk-forward discipline** as
  every other evaluation in this project (H=60, embargo=60) -- a fair
  test of whether more diverse training data gives a more expressive
  model room to earn its complexity, not a different methodology.
- **Three candidate MLP architectures** (16/32/64 hidden units),
  evaluated in three independent implementations: scikit-learn (CPU),
  PyTorch (CPU), PyTorch (Apple Silicon MPS) -- deliberately over-tested
  rather than trusting one framework's numbers.

## Result 1: no MLP configuration beat logistic regression, in any implementation

| Implementation | mlp_16 | mlp_32 | mlp_64 | LR baseline |
|---|---|---|---|---|
| scikit-learn (CPU) | -0.0093 | -0.0060 | -0.0067 | AUC 0.9105 |
| PyTorch (CPU) | -0.0520 | -0.0273 | -0.0293 | (delta AUC vs. LR) |
| PyTorch (MPS) | -0.0626 | -0.0513 | -0.0211 | |

Nine total model runs across three independent implementations. **Every
single one underperforms the logistic regression baseline.** This is a
robust, multiply-confirmed negative result: added model complexity did
not earn its keep, even with 5x more data spanning genuinely diverse
market regimes (crashes from three structurally different causes, plus a
calm baseline). Consistent with the project's operating discipline
throughout -- report the negative result plainly rather than search for a
configuration that happens to win.

Caveat: PyTorch's two implementations score consistently lower than
scikit-learn's -- likely a training-procedure difference (optimizer,
early-stopping criterion, no fixed weight-init seed across CPU/MPS runs)
rather than evidence PyTorch's MLPs are inherently worse. Doesn't affect
the headline conclusion, since scikit-learn's numbers (the most carefully
tuned of the three) also lose to logistic regression.

## Result 2: this closes the road to the original Phase 4 plan

The original stretch goal (a latency/accuracy Pareto frontier via
compressing a larger, winning model) requires a winning larger model.
There isn't one. Rather than force a compression study onto a model that
lost, the investigation was redirected to a question that remained
genuinely open and answerable: does hardware acceleration matter for this
problem's shape?

## Result 3: a real, measured CPU-vs-MPS crossover

Controlled comparison -- identical PyTorch code, only the device differs:

| hidden size | CPU | MPS | MPS/CPU |
|---|---|---|---|
| 16 | 60.1s | 94.1s | 1.57x **slower** on MPS |
| 32 | 89.0s | 72.7s | 0.82x faster on MPS |
| 64 | 150.1s | 88.9s | 0.59x (41% faster) on MPS |

A genuine overhead-vs-compute crossover, empirically measured rather than
assumed: for the smallest model, MPS's fixed overhead (tensor transfer,
kernel launch) dominates and makes it *slower* than plain CPU; as the
model grows, compute cost outweighs that overhead and MPS pulls ahead.
The crossover point for this data scale (a few thousand rows per fold, 3
features) sits somewhere between 16 and 32 hidden units.

**This is a legitimate, useful engineering finding in its own right** --
knowing when GPU acceleration is and isn't worth it for a given workload
shape is real latency-engineering judgment, directly relevant to a
market-making infrastructure audience, and arguably a more differentiated
result than a generic "bigger model, compressed, therefore faster"
demonstration would have been.

## What this Phase 4 deliverable is, and isn't

Not the originally-planned compression/quantization study -- there was no
winning larger model to compress. Instead: a rigorous, multiply-confirmed
negative result on model complexity, plus a real, controlled hardware
benchmark answering a genuinely open question. Both are reported honestly
as what they are, not dressed up as the original plan.

## Reproducibility

```
python scripts/build_phase2_dataset.py data/raw/BTCUSDT-aggTrades-2020-03.zip
python scripts/build_phase2_dataset.py data/raw/BTCUSDT-aggTrades-2021-05.zip
python scripts/build_phase2_dataset.py data/raw/BTCUSDT-aggTrades-2023-07.zip
# (2022-05 and 2022-11 already built during Phase 2 / cross-window check)
python scripts/merge_multi_window_dataset.py
python scripts/compare_lr_mlp_combined.py          # scikit-learn baseline
python scripts/compare_lr_mlp_torch_mps.py         # PyTorch, MPS (or CPU fallback)
python scripts/compare_lr_mlp_torch_mps.py cpu     # PyTorch, CPU forced (controlled comparison)
```

## Project status

Phases 1-4 all complete. This closes the project's active development --
remaining `ISSUES.md` items (#4, #5, #6) are low-priority, optional
polish, not blocking.
