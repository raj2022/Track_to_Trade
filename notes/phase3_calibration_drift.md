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

## Attempt 3 (rejected): walk-forward isotonic, expanding window

Refit isotonic daily on all prior data (expanding), mirroring the
classifier's own purged walk-forward cadence. Tested first on synthetic
data with continuous drift: did not clearly help (log score got slightly
worse, several bins' |diff/SE| increased). Rationale for the failure:
an expanding window still averages together an increasingly long, and
therefore increasingly stale, history of past miscalibration levels if
the true drift is continuous rather than a one-time shift -- the same
underlying problem as the static splits, just spread across more, smaller
steps instead of one big one.

## Attempt 4 (rejected): walk-forward isotonic, rolling window

Same daily refit cadence, but using only the most recent N days (5, then
compared against 3 and 10) rather than the full expanding history --
intended to track continuous drift more responsively. On real data:
**worse than raw on every metric** — log score 0.339 → 0.454, diff/SE
peak growing from 7.8σ (raw) to 18.4σ (calibrated). Also surfaced a
separate real issue: ~10% of predictions (4,317) were skipped because
their 5-day window contained only one class — a genuine fragility of
short rolling windows against a rare (~28%), autocorrelated label.

## Attempt 5 (rejected, but instructive): walk-forward Platt scaling, rolling window

Hypothesis: isotonic regression's flexibility (low bias, high variance) is
the real problem, not the window shape — it overfits small calibration
windows and fails to generalize even one window forward. Platt scaling
(2-parameter logistic fit on the log-odds of the raw prediction) is far
more constrained and should be more stable on small samples. Tested on the
same synthetic continuous-drift data first: **succeeded there** — log
score improved (0.580→0.551), Brier improved, diff/SE dropped from
double-digit σ to mostly under 4σ. On real data: **did not clearly
succeed** — log score and Brier moved only marginally (0.3391→0.3382,
0.1041→0.1033), while the per-bin diff/SE table got *worse* in several
bins (peak 7.8σ raw → 12.9σ calibrated). A small aggregate-metric
improvement sitting on top of larger local miscalibration is not a fix —
it's the same problem redistributed, not resolved.

## Final assessment: recalibration abandoned, not indefinitely pursued

Five attempts, spanning both axes that plausibly mattered — split scheme
(static time, static regime, expanding walk-forward, rolling walk-forward)
and calibrator flexibility (isotonic, Platt) — failed to produce a robust
fix. Continuing to search for a sixth variant that happens to work on this
one month of data would itself become a version of the exact failure mode
this project is built to catch: fitting increasingly specific corrections
to one dataset until something looks clean, rather than accepting a
genuine negative result.

**Most likely underlying reason:** the true miscalibration probably
depends on more than the single scalar raw probability that every
recalibration method here could see — plausibly on which latent regime is
actually active, not just the classifier's point estimate of it. A 1-D
recalibration function, however constructed, structurally cannot capture
that. Fixing this properly would require calibration conditional on
richer state information than was available to any of these attempts, not
a better choice of window or calibrator.

**Decision: proceed to the Bayes-risk threshold using the RAW (uncalibrated)
probabilities**, with the known miscalibration in the ~0.05–0.5 predicted-
probability range (Result 1, above) reported as an explicit, documented
limitation on the resulting decision — since the derived threshold
(≈0.097) sits inside that range. This is the honest choice: a stated
limitation the reader can weigh, rather than false confidence built on a
recalibration that did not demonstrably hold up under real-data testing.

## Reproducibility

- `python scripts/save_oof_predictions.py`
- `python scripts/calibration_check.py` (full-dataset reliability check, Result 1)
- `python scripts/recalibrate.py` (Attempt 1, rejected)
- `python scripts/recalibrate_by_regime.py` (Attempt 2, rejected)
- `python scripts/calibrate_walk_forward.py` (Attempts 3–5, all rejected —
  toggle `CALIBRATION_METHOD` and `ROLLING_WINDOW_DAYS` to reproduce each)

## Next step

Derive and apply the Bayes-risk decision threshold to the RAW probabilities
(`data/processed/real_label_oof_predictions.parquet`), using the Roll
spread ratio (R_crisis/R_calm ≈ 9.3, giving p* ≈ 0.097), with the known
0.05–0.5 miscalibration range stated explicitly as a limitation on the
resulting decision rather than silently ignored.


