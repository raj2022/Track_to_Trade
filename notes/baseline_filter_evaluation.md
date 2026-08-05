# Baseline (single-hypothesis) filter evaluation — motivation for the IMM

**Date:** Day 4/5
**Question:** Does the baseline local-level Kalman filter, using the
already-derived R and rolling Q as calibrated inputs, produce a
well-calibrated innovation gate? If not, why not — and what does the
failure mode imply about the filter's structure?

## Setup

- Model: 1D local-level (random walk) state, x_t = x_{t-1} + w_t,
  z_t = x_t + v_t. No drift term (its process noise has not been derived —
  see `src/kalman.py` docstring).
- R = 3.661341181889e-11 (Roll's estimator, all-tick, verified calm window;
  see `microstructure_noise_R_derivation.md`).
- Q_t = rolling 1-day realized-variance rate × dt (dt=60s), from
  `regime_varying_Q_derivation.md`.
- Gate: chi-square(1) threshold derived from an explicit tolerance of 1
  false alarm/day at 1440 steps/day → alpha = 6.94e-4, threshold = 11.504.
- Data: 2022-05 (LUNA window), 44,639 steps after dropping the 1-day
  warm-up period.
- Script: `scripts/run_baseline_filter.py`

## Result 1: general miscalibration, not event-localized

Overall: 525/44,639 steps flagged (1.176%), ~17x the 0.069% target.

Split by period:

| Period | Flagged | Steps | Rate |
|---|---|---|---|
| Pre (05-01 to 05-09) | 224 | 11,519 | 1.945% |
| During (05-09 to 05-13) | 85 | 5,760 | 1.476% |
| Post (05-13 to 05-31) | 216 | 27,360 | 0.789% |

All three periods sit well above the 0.069% target. **This rules out a
LUNA-specific explanation** — the miscalibration is a general property of
representing BTCUSDT's tick-level return behavior with a single,
smoothly-varying Gaussian process, not something specific to this one
crash. Consistent with known fat-tailed, volatility-clustered behavior in
crypto returns at short timescales (including the ~8h periodicity found in
the diurnal check).

## Result 2: the ordering (pre > during > post) needed its own explanation

The naive read of the split table is backwards from what an
"event-localized surprise" story would predict — pre-event flags exceed
during-event flags. Two competing hypotheses were checked before writing
anything up, rather than accepting either on its face:

**Hypothesis 1 (warm-up contamination): ruled out.** If the 1-day rolling
window for Q was still contaminated by an unrepresentative first day
(2022-05-01) early in the month, that could inflate early flag rates
artificially. Checked via `scripts/pick_calm_window.py` on 2022-05: 05-01's
daily RV (0.000618) sits squarely inside the range of 05-02–05-08
(0.000423–0.001563), not off on its own. No evidence of warm-up
contamination.

**Hypothesis 2 (pre-event was never actually calm): confirmed.** The
period-split comparison implicitly treated "pre-event" as a calm baseline
— the same unverified assumption the April 2022 calm-window search was
built specifically to avoid. Checked directly: every day from 05-01 to
05-08 meets or exceeds the *top* of the verified June 2023 calm baseline
range (0.000168–0.000613), several by 2–3x. Early May 2022 was already
running hot — plausibly broader macro/rate-hike jitters ahead of the
LUNA-specific trigger on 05-09 — not a calm reference period. The
elevated pre-event flag rate is a real, correctly-detected signal, not
model noise.

## Result 3: why "during" is still lower than the raw event severity would suggest

Even with Result 2 resolved, a puzzle remained: 05-11/05-12 have daily RV
≈ 0.0116 (~19x the June calm baseline, and the two most extreme days in
the entire month by a wide margin — confirmed via the same outlier trim,
which excluded exactly 05-09 through 05-12 as MAD outliers), yet the
"during" flag rate (1.476%) is lower than "pre" (1.945%).

**Mechanism:** Q is a trailing 1-day rolling window, so during a sustained
multi-day spike, Q itself ramps up within about a day — by 05-11/05-12 the
trailing window already contains 05-09/05-10's elevated RV, so the filter's
own noise expectation has partially caught up to the new volatility level,
muting further surprise. The gate is not failing to notice the crash; it is
*adapting to* the crash and then treating the new, elevated level as the
baseline, precisely during the period a persistent anomaly signal would be
most valuable.

## Conclusion: motivation for the IMM

A single, continuously-updated Q cannot hold a persistent "this is
anomalous" signal through a sustained regime shift, because it re-absorbs
the anomaly into its own baseline within roughly one window-length. This is
a structural limitation of a one-hypothesis filter, not a tuning problem —
shrinking or lengthening the rolling window trades off how fast it adapts,
but does not remove the fundamental property that a single continuously
updated parameter chases the data rather than maintaining competing
hypotheses about which regime is active.

This is the concrete, derivation-backed case for moving to an Interacting
Multiple Model filter: discrete competing hypotheses (e.g., a "calm
diffusion" state anchored near the verified-calm Q, and a wider
"elevated/crisis" state) do not individually chase the data the way one
continuously-adapting Q does — the model instead has to accumulate evidence
to *switch* which hypothesis is active, which is the behavior a persistent
anomaly signal actually requires.

## Reproducibility

- `python scripts/run_baseline_filter.py data/raw/BTCUSDT-aggTrades-2022-05.zip`
- `python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2022-05.zip` (for the Result 2/3 daily RV checks)

## Next step

Implement the IMM extension: at minimum two regime hypotheses (calm /
elevated), each with its own Q, combined via mixing probabilities fit by
EM/Baum-Welch — per the original Phase 1 plan, now with an empirical result
motivating why the single-hypothesis version isn't sufficient.
