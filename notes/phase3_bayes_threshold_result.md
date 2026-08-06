# Phase 3 close-out: Bayes-risk decision threshold

**Date:** Day 11/12
**Question:** Given the purged walk-forward classifier's predictions on
`real_label`, what decision threshold minimizes expected cost under an
explicit, derived cost asymmetry — and does it outperform the naive
default of 0.5?

## Derivation

Cost framing, tied directly to Phase 1's derived quantities rather than
asserted: a false negative (missing a real elevated-regime call) means
continuing to quote the tight calm spread into an actually-elevated
period, exposed to the wider crisis-level adverse selection. A false
positive (false alarm) means needlessly widening to the crisis spread
during what's actually calm, giving up the tighter spread's edge. The
natural cost ratio is the Roll-implied spread ratio itself:

- C_FN / C_FP = spread_crisis / spread_calm = 1.127bp / 0.121bp ≈ 9.314
- Bayes-optimal threshold: p* = C_FP / (C_FP + C_FN) ≈ **0.097**

Applied to the **raw (uncalibrated)** purged walk-forward probabilities —
five recalibration attempts (static time split, static regime split,
walk-forward isotonic in both expanding and rolling form, walk-forward
Platt scaling) all failed to produce a robust fix; see
`notes/phase3_calibration_drift.md`. p* falls inside the known
miscalibrated range (~0.05-0.5, up to 8 sigma off in the original
reliability check), and this is stated explicitly rather than ignored —
the script itself prints a warning when this happens.

## Result

| | p* = 0.097 | naive p = 0.5 |
|---|---|---|
| flagged rate | 47.4% | 24.5% |
| false positives | 9,209 | 1,984 |
| false negatives | 1,011 | 3,680 |
| mean cost/step | 0.4317 | 0.8405 |

**Mean cost reduction: +48.6%.** The derived threshold flags nearly twice
as often as naive 0.5, trading a large increase in false positives for a
large decrease in false negatives — exactly the direction a ~9.3x cost
asymmetry should push a decision rule.

**Sanity check against the miscalibration caveat:** the empirical cost
curve's actual minimum (swept across all thresholds 0.01-0.99) sits close
to the theoretically derived p* — visually in the 0.10-0.13 range, not
displaced to some unrelated part of the curve. Despite the documented
underconfidence in exactly this predicted-probability region, the
derivation held up reasonably well against the empirical optimum. This
doesn't erase the miscalibration finding, but it does mean the derived
threshold wasn't badly undermined by it in this instance.

## What this result is, and isn't

This is a decision-theoretic result under an explicit, derived cost model
— not a backtest, not a profitability claim, consistent with the
project's stated non-goals throughout. The 48.6% figure describes cost
reduction under this specific cost framing (spread-ratio-derived
asymmetry), not realized P&L from any trading activity. No claim is made
that this cost model captures every real consideration a market maker
would weigh (execution, inventory, competition), only that it's a
concrete, derived, defensible starting point rather than an arbitrary
threshold.

## Reproducibility

- `python scripts/bayes_threshold.py`

## Phase 3 status

Reliability/proper-scoring evaluation: done (`calibration_check.py`).
Recalibration: five attempts, all rejected, documented honestly
(`phase3_calibration_drift.md`). Bayes-risk threshold: derived, applied,
validated against the empirical cost-minimizing threshold, with the known
calibration limitation stated throughout rather than hidden. **Phase 3's
core deliverable is complete.**
