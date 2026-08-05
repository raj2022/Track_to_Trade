# Phase 2 leakage stress test — full arc and final result

**Date:** Day 8/9
**Question:** Does naive k-fold cross-validation show inflated apparent
predictive skill (relative to purged/embargoed walk-forward) on a label
constructed to carry no genuine relationship to the features — the direct
financial analogue of the CERN conditional-generalization failure this
project is built around?

## Setup

- **H (label horizon):** derived from the K=4 IMM's own transition matrix
  via absorbing-Markov-chain expected sojourn time in the elevated set,
  weighted by the quasi-stationary distribution — 54 minutes, rounded to
  60 (`scripts/derive_horizon.py`).
- **Real label:** mean elevated-probability over the forward window
  (t+1, t+H] exceeds 0.5.
- **Features:** current elevated-prob, its 1-step lag, and a 1h rolling
  return volatility — all strictly causal.
- **Splitters:** naive contiguous k-fold (no purge/embargo) vs. expanding-
  window walk-forward with purge width H (drops training points whose
  forward label window would overlap the test period) and embargo width H
  (matches the 1h `rolling_vol_1h` feature's own lookback — drops points
  immediately after each test fold from ever being used in later training).

## The null label required two complete redesigns before it was trustworthy

**Attempt 1 (rejected): block permutation.** Independently reordering
H-length blocks of the label. Real-data result: AUC 0.449 (naive) / 0.452
(purged), both ~15-17 std below chance. Diagnosis: the label is
imbalanced (~28% positive) and highly autocorrelated; randomly relocating
rare positive blocks across a mostly-low-feature timeline causes them to
land disproportionately on low-feature periods purely by base-rate
arithmetic — a systematic artifact of the permutation scheme, not a
finding about leakage.

**Attempt 2 (rejected): single circular shift.** Rotating the entire label
series by one large random offset, to preserve the full marginal
distribution and avoid the block-fragmentation issue above. Real-data
result: AUC 0.285 (naive) / 0.593 (purged) on the first shift drawn — both
large deviations, in *opposite* directions. Diagnosis: this dataset has a
known ~8h periodicity (`scripts/diurnal_check.py`, from the original Q
derivation work). A circular shift preserves periodic structure by
construction; a shift near a multiple of the true period re-aligns nearly
in phase (spurious positive correlation), a near-half-period offset
anti-aligns (spurious negative correlation). A single shift draw is not
robust to this.

**Final method: many circular shifts (20), explicitly excluding offsets
within 30 steps of a multiple of the known 480-step (8h) or 1440-step
(24h) periods**, reporting the distribution across shifts rather than one
number (`scripts/build_phase2_dataset.py`).

## A third, previously invisible confound: the null label's own autocorrelation

Even with a clean null construction, the raw AUC results were still
confusing: purged walk-forward showed a **consistent** positive bias
across all 20 shifts (0.527-0.727, mean 0.594), while naive k-fold was
noisy and centered near chance (0.285-0.646, mean 0.479) -- the *opposite*
pattern from what the leakage hypothesis predicted, and too consistent
across 20 independent shifts to dismiss as noise.

**Diagnosis, via a decisive control:** a completely feature-blind
classifier (predicting only the training fold's class prior -- literally
using no features) was run through the same pipeline. It showed AUC 0.130
(naive) / 0.551 (purged) -- both far from 0.5, matching the direction and
rough magnitude of the "real" model's bias in each scheme. This proves the
bias is **not feature-driven at all**: it comes from the interaction
between the (circularly-shifted, still-autocorrelated-by-design) label's
own serial structure and each CV scheme's mechanics --

- naive k-fold's structural bias is systematically *negative*: leaving a
  large, above-average-positive-share block out of training causes the
  remaining training prior to under-estimate that block's true rate -- a
  mathematical property of leave-block-out estimation on block-structured
  data, not shift-dependent.
- walk-forward's structural bias is systematically *positive*: an
  expanding window's training prior genuinely tracks whether the recent
  past was elevated, which is real (if trivial) predictive information for
  an autocorrelated series, independent of any feature.

This is a legitimate, important finding on its own -- for an autocorrelated
label, "zero relationship to the features" (what circular shift achieves)
is not the same as "zero exploitable structure" once serial correlation
and CV mechanics interact.

## Correctly isolating the feature-driven effect

Fix: compute **real model AUC minus dummy (feature-blind) AUC**, per
shift, per scheme -- this subtracts out each scheme's label-autocorrelation
structural bias and isolates what the features specifically contribute
(`scripts/phase2_real_vs_dummy_comparison.py`).

| | naive_kfold | purged_walk_forward |
|---|---|---|
| mean real AUC | 0.479 | 0.594 |
| mean dummy AUC | 0.130 | 0.551 |
| **mean diff (real - dummy)** | **+0.349** | **+0.043** |
| std of diff | 0.118 | 0.055 |
| shifts with positive diff | **20 / 20** | 14 / 20 |

## Result

**naive_kfold: overwhelming, unambiguous leakage.** Every one of 20
independent null shifts shows a positive real-minus-dummy gap
(probability of this occurring by chance under a fair-coin null: 0.5^20 ~
1 in a million). The features consistently add ~0.35 AUC to a label with
zero true relationship to them -- this is what feature-driven leakage under
naive cross-validation looks like, cleanly isolated from the two
confounds above.

**purged_walk_forward: mostly, not perfectly, eliminated.** The mean
diff (+0.043) is smaller than its own std (0.055), and only 14/20 shifts
are positive (sign-test p~0.04 -- suggestive, not conclusive). Purging and
embargo reduce the feature-driven leakage effect by roughly 8x (0.349 ->
0.043) relative to naive k-fold, but the honest conclusion is "the large
majority of leakage is removed," not "leakage is proven to be exactly
zero." Worth flagging as an open item rather than rounding up to a
cleaner-sounding claim.

## Why this matters for the project's core thesis

Had the raw AUC-vs-0.5 comparison from the first stress test run been
trusted (naive ~0.48, "near chance, no leakage detected"), the conclusion
would have been **backwards** -- naive k-fold's structural anti-correlation
bias was masking a real, large, feature-driven leakage effect underneath
it. Only by building a proper dummy-baseline control -- itself only
necessary because two earlier null constructions failed for diagnosable
reasons -- was the true effect uncovered. This is a direct, hard-won
demonstration of the project's central thesis: apparent predictive power
can be an artifact of validation methodology rather than genuine signal,
and verifying that requires more than one plausible-looking null.

## Reproducibility

- `python scripts/derive_horizon.py`
- `python scripts/build_phase2_dataset.py data/raw/BTCUSDT-aggTrades-2022-05.zip`
- `python scripts/phase2_leakage_stress_test.py` (raw real-label/null-label AUCs)
- `python scripts/phase2_real_vs_dummy_comparison.py` (the decisive, confound-corrected result)

## Next step

With the leakage mechanism now cleanly demonstrated, extend the real-label
purged-walk-forward result with proper decision-theoretic calibration
(Phase 3) -- the naive-vs-purged gap on the *real* label (0.918 vs. 0.914,
still small) can now be interpreted with confidence, since the validation
methodology itself has been stress-tested rather than assumed sound.
