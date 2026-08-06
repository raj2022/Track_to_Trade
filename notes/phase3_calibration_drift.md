# Phase 3 calibration — three failed static corrections and the real finding

**Date:** Day 10/11
**Question:** Are the purged-walk-forward classifier's predicted probabilities
on `real_label` well-calibrated (needed before deriving/applying a
Bayes-risk decision threshold)?

## Setup

- Purged-walk-forward OOF predictions for `real_label` (43,140 predictions,
  30 folds), saved via `scripts/save_oof_predictions.py`.
- Log score and Brier score (both proper scoring rules; CRPS reduces
  exactly to Brier score for a binary outcome), benchmarked against a
  climatology baseline (always predict the base rate) as a skill-score
  reference rather than reporting raw numbers without context.
- Reliability diagrams via quantile-based bins, with the binomial standard
  error reported per bin explicitly, so a bin's departure from perfect
  calibration can be judged in units of its own uncertainty (|diff|/SE)
  rather than by eye off a plot.

## Result 1: genuine skill confirmed, but the full-dataset reliability check found real miscalibration

Log score skill = +0.465, Brier skill = +0.521 over climatology -- strong,
positive, consistent with Phase 2's validated ~6 sigma signal finding. But
the reliability check (`scripts/calibration_check.py`) found systematic
**underconfidence** across bins spanning roughly predicted probability
0.05-0.5 (5 of 10 bins, up to 8 sigma off, all in the same direction),
while the two highest-confidence bins were well-calibrated. The plot alone
looked close to the diagonal everywhere; only the printed per-bin SE
column revealed the departure was real and systematic, not noise.

This mattered concretely: the Bayes threshold about to be derived from the
Roll spread ratio (~0.097) sits inside the miscalibrated range, so
applying it to raw probabilities risked systematically under-triggering.

## Attempt 1 (rejected): isotonic regression, chronological time split

Calibrate on the first half of the OOF period (05-02 to 05-16, crisis-
heavy), evaluate on the second half (05-16 to 05-31, calm aftermath).
Result: **worse**, not better -- log score 0.305 -> 0.366, Brier 0.092 ->
0.109. The raw miscalibration direction on this held-out half was also the
*opposite* of the full-dataset check: systematic **overconfidence** (up to
-15 sigma), not underconfidence.

## Attempt 2 (rejected): regime-conditional split (elevated_prob median)

Hypothesis at the time: calibration is regime-dependent (crisis vs. calm),
so condition the calibration split on the model's own `elevated_prob`
signal instead of time. Fit separate isotonic maps for "elevated" and
"calm" halves (each split further into an internal chronological
calibrate/evaluate half). Result: **worse in both regimes**, more
severely than the plain time split -- elevated regime's log score went
0.548 -> 0.872, with calibrated diff/SE reaching -28 sigma. The regime
hypothesis is refuted by this test.

## The actual finding: calibration drifts over time, not over regime

Every attempt so far -- the plain time split, and both halves of the
regime-conditional split -- shares one structural property: calibrate on
an earlier chronological chunk, evaluate on a later one. **Every single
one failed in the same direction (overconfidence on the later, held-out
portion), regardless of how the data was sliced.** Regime-conditioning
didn't fix the problem, it just reproduced it inside two smaller buckets.

The common factor is time, not regime: whatever calibration relationship
holds early in the walk-forward period does not hold later in it, and no
static split -- by date or by regime -- captures a mapping that transfers
forward. This is the same lesson the baseline filter's fixed Q, the
single-shift null, and the time-split miscalibration attempt have each
independently produced: anything fit once on a chunk of this data and
assumed to generalize forward has failed every time it's been tested here.

Secondary note, not the main finding: the elevated_prob median split point
(0.0632) was very low, since the feature is heavily right-skewed (most of
the month is calm) -- this particular split likely separated "extremely
calm" from "everything else" rather than two genuinely distinct regimes,
a real confound in that specific test. It doesn't change the conclusion,
since the drift pattern appeared identically in both halves regardless.

## Next step

Calibrate walk-forward, the same way the classifier itself is already
evaluated -- refit the isotonic mapping on an expanding or rolling window
as time progresses, rather than fitting one static map anywhere and
applying it forward. Not yet built.

## Reproducibility

- `python scripts/save_oof_predictions.py`
- `python scripts/calibration_check.py` (full-dataset reliability check, Result 1)
- `python scripts/recalibrate.py` (Attempt 1, rejected)
- `python scripts/recalibrate_by_regime.py` (Attempt 2, rejected)
