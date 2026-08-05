# IMM filter result and Phase 1 close-out

**Date:** Day 5/6
**Question:** Does replacing the baseline's single, continuously-adapting Q
with an IMM (discrete regime hypotheses, each with fixed, derived Q/R)
produce a persistent anomaly signal, where the baseline was shown to
re-absorb the anomaly within about a day?

## Setup

- 5 regime hypotheses, K selected by BIC over a Gaussian HMM fit to 2022-05
  returns (K=2..5 tested; K=5 won by a real but comparatively modest margin
  over K=4 — 608 BIC points, vs. 1507 for the K=3→K=4 step. Worth flagging
  as a less secure margin than the earlier additions).
- Each hypothesis: fixed Q_k from the HMM's per-state variance, fixed R_k
  from Roll's estimator computed on tick pairs restricted to that state's
  bins (`scripts/per_state_R.py`), decomposed via Q_k = HMM_variance_k - 2R_k.
- Transition matrix from the same HMM fit (EM/Baum-Whlch), used as the
  IMM's mixing probabilities.
- Algorithm: standard Blom & Bar-Shalom (1988) IMM recursion
  (`src/imm.py`), validated against synthetic two-regime data before use
  (recovered injected Q/R closely; mode probability correctly jumped and
  *persisted* through a sustained synthetic regime, unlike a single-Q
  filter).

## Cross-validation of the R derivation (unplanned, worth noting)

State 4's independently-derived R (3.620e-11, from tick pairs in that
state's bins) landed almost exactly on the original calm-window R
(3.661e-11, derived weeks earlier from unrelated June 2023 data via a
completely different windowing). Real corroboration of the original R
derivation via an unrelated route, not designed for this purpose.

## Result: persistence, but decaying — reported honestly, not oversold

Mean P(state in {elevated, extreme}):

| Period | Mean probability |
|---|---|
| Pre-event (05-01 to 05-09) | 0.159 |
| During, early (05-09 to 05-13) | 0.830 |
| During, late (05-13 to 05-19) | 0.424 |

The signal decays across the 10-day window rather than holding flat — this
is real and should not be hidden behind the flattering 10-day aggregate
(0.587). But 0.424 is still **~2.7x the pre-event baseline**, a materially
different shape from the single-hypothesis baseline filter
(`baseline_filter_evaluation.md`), which showed no meaningful clustering
around the event at all — flags were roughly uniform across pre/during/post
periods (1.945%/1.476%/0.789%, all far above target, in the *wrong* relative
order). The IMM's decay-but-persist pattern is a qualitative improvement:
evidence has to accumulate to shift probability mass between discrete
hypotheses, rather than one continuously-adapting parameter silently
relabeling the new level as normal.

## Open item: state 0/3 flickering

States 0 and 3 (both low-variance "calm" hypotheses) show persistent
high-frequency alternation throughout the entire month, including well
outside the event window — visible in the mode-probability plot as constant
sharp spikes, not settled regime occupancy. Traced to the fitted transition
matrix itself: state 0's self-transition is only 0.5812, with a 0.3916
probability of jumping directly to state 3 — these are not sticky,
independent regimes as fit.

**Not resolved here.** Candidate explanation: this may be exactly the
marginal 5th-state benefit BIC weakly preferred — states 0 and 3 might be
one genuine "thin/quiet calm" regime that K=5 split unnecessarily. Flagged
as a concrete thing to test before this filter feeds Phase 2 (e.g., refit
with K=4 and check whether the elevated/extreme persistence result above
is materially unchanged — if so, the extra state was not adding real
value and can be dropped for a simpler, less chattery model).

## Reproducibility

- `python scripts/check_R_regime_invariance.py data/raw/BTCUSDT-aggTrades-2022-05.zip`
- `python scripts/fit_regime_hmm.py data/raw/BTCUSDT-aggTrades-2022-05.zip`
- `python scripts/per_state_R.py data/raw/BTCUSDT-aggTrades-2022-05.zip`
- `python scripts/run_imm_filter.py data/raw/BTCUSDT-aggTrades-2022-05.zip`

## Phase 1 status: functionally complete

- Baseline filter: built, derived (R via Roll, rolling Q), evaluated,
  found to fail in a specific, quantified, understood way.
- IMM: built specifically to address that failure, every parameter
  (K, per-regime Q/R, transition matrix) derived rather than asserted,
  validated on synthetic data before real use, shown to produce a
  materially different (though decaying, honestly reported) persistence
  pattern than the baseline.
- Naive baseline model (trailing-median rule, per the Phase 1 revision)
  and the chi-square/bipower-variation jump-diffusion split are still
  outstanding — carry into the start of Phase 2 rather than blocking on
  them here, since the core filtering result is now in hand.
- Open item: state 0/3 flickering / possible K=4-vs-K=5 re-check, to
  resolve before this filter's output feeds the Phase 2 classifier.

## Next step

Phase 2: the artifact stress test. Construct the regime/event classifier
on top of this IMM output, build the leaky vs. purged label sets, and
implement the walk-forward evaluation scheme.
