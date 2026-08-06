# Open items

Tracked here rather than left implicit in notes/ so nothing gets quietly
forgotten. Revisit before considering the project "done," not necessarily
before moving on day to day.

## ~~#1 — `run_imm_filter.py` still hardcodes the retired K=5 parameters~~ RESOLVED
**Status:** ~~open~~ **closed — fixed**
**Priority:** ~~medium~~ n/a
~~Update to the K=4 parameters... Risk: anyone rerunning it unmodified
silently reproduces the superseded model.~~
Updated to the canonical K=4 Q/R/transition matrix from `k4_recheck.py`'s
actual output. Verified the parameters run cleanly through `src/imm.py`
(correct shape, well-formed mode probabilities, no NaNs) before handing
off. Print statements now reference the expected K=4 persistence values
(0.165/0.842/0.447) so a rerun can be checked against them directly.

## ~~#2 — Cross-window robustness check~~ RESOLVED
**Status:** ~~open~~ **closed — confirmed**
**Priority:** ~~high~~ n/a
~~All Phase 1 and Phase 2 work has been done exclusively on 2022-05...~~
Reran on FTX (2022-11): naive k-fold's leakage signature replicated
almost identically (20/20 shifts positive, same order-of-magnitude
effect size as May). The real label's signal also replicated (and was
slightly *stronger*: ≈4.0σ/7.4σ above artifact baseline vs. May's
≈3.7σ/6.1σ). Caught and fixed a real bug along the way — `ELEVATED_STATES`
was hardcoded from May's specific HMM fit and needed to be derived
per-window by variance rank instead, since state indices aren't stable
across independent EM fits. Full detail:
`notes/cross_window_robustness_check.md`. COVID and May 2021 remain
unused if a third confirmation is ever wanted, but not required.

## ~~#3 — Purged walk-forward's residual leakage: real or noise?~~ RESOLVED
**Status:** ~~open~~ **closed — noise**
**Priority:** ~~medium~~ n/a
~~Current result: mean diff +0.043... suggestive but not conclusive...~~
The FTX cross-window check (above) answered this without needing more
shifts on May: FTX's purged residual is +0.008 with exactly 10 of 20
shifts positive — a coin flip, no directional signal. Two independent
windows, and only one (May) showed anything resembling a residual, at a
magnitude that now looks like sampling noise at n=20 rather than a
genuine small leak. Purged/embargoed walk-forward is treated as removing
the leakage to within noise on both windows tested.

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

