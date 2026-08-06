# Open items

Tracked here rather than left implicit in notes/ so nothing gets quietly
forgotten. Revisit before considering the project "done," not necessarily
before moving on day to day.

## #1 — `run_imm_filter.py` still hardcodes the retired K=5 parameters
**Status:** open
**Priority:** medium (cosmetic/correctness risk, not blocking)
Update to the K=4 parameters (`k4_recheck.py`'s output), or retire the
script in favor of `k4_recheck.py` being the canonical filter runner.
Risk: anyone (including future-you) rerunning it unmodified silently
reproduces the superseded model.

## #2 — Cross-window robustness check
**Status:** open
**Priority:** high
All Phase 1 and Phase 2 work has been done exclusively on 2022-05 (LUNA).
Three other event windows are already downloaded and checksum-verified
(COVID 2020-02/04, May 2021 crash, FTX 2022-10/12) but unused. Rerun the
Phase 2 leakage stress test (or at minimum the real-vs-dummy comparison)
on at least one other window to confirm the leakage-demonstration result
isn't an artifact specific to LUNA's particular dynamics.

## #3 — Purged walk-forward's residual leakage: real or noise?
**Status:** open
**Priority:** medium
Current result: mean diff +0.043, std 0.055, 14/20 shifts positive
(p≈0.04) — suggestive but not conclusive at 20 shifts. Increase to 50-100
null shifts to tighten this estimate enough to say with confidence whether
purging leaves a genuine residual leak or whether the current result is
just sampling noise at this shift count.

## #4 — Jump-test calibration overshoot
**Status:** open
**Priority:** low
9 of 31 days flagged as jumps against a derived 1/month target. Plausible
cause: the jump test's implicit noise-homogeneity assumption doesn't
account for the regime-dependent R already established (86.7x higher in
crisis). Doesn't affect the LUNA-specific null finding (no jump during the
crisis days), but worth a proper fix if the jump test is reused elsewhere.

## #5 — ~8h periodicity: funding-settlement hypothesis unverified
**Status:** open
**Priority:** low
Found in the original Q derivation (`diurnal_check.py`) and re-confirmed
as relevant during the Phase 2 null-construction failure. Working
hypothesis (Binance perpetual funding settlement at 00:00/08:00/16:00 UTC
spilling into spot via arbitrage) was never directly checked -- would need
trading-intensity binning by UTC hour. Cheap to verify, not required for
anything currently built.

## ~~#7 — Walk-forward calibration not yet built~~ RESOLVED
**Status:** ~~open~~ **closed — investigated, resolved by decision**
**Priority:** ~~high~~ n/a
~~Three static calibration corrections... all failed the same way...~~
Full investigation completed: five recalibration attempts total (static
time split, static regime split, walk-forward isotonic in both expanding
and rolling form, walk-forward Platt scaling) — all five failed to
produce a robust fix. Diagnosis: miscalibration likely depends on more
than the raw scalar probability any of these methods could see (plausibly
the latent regime itself), which no 1-D recalibration can capture.
**Decision:** proceed with raw (uncalibrated) probabilities for the
Bayes-risk threshold, with the known miscalibrated range (~0.05-0.5)
stated explicitly as a limitation rather than hidden. The derived
threshold (p*≈0.097) was cross-checked against the empirical cost-
minimizing threshold and held up reasonably well despite the caveat.
Full detail: `notes/phase3_calibration_drift.md`,
`notes/phase3_bayes_threshold_result.md`.

## #6 — LOBSTER cross-check never used
**Status:** open, optional
**Priority:** low
Originally scoped as a cross-check for the R derivation against a real
limit order book. R was ultimately derived successfully via Roll's
estimator on Binance tick data alone, so this is a nice-to-have validation
rather than a gap, but noted here so it isn't presented as "used" if asked.

