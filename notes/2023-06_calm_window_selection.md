# Calm-window selection for the R (microstructure noise) derivation

**Date:** Day 2
**Question:** Which period should be used as a "calm" baseline to estimate
microstructure noise, separately from a period containing a real regime
transition (2022-05, LUNA)?

## Why this needs its own derivation

Microstructure noise (bid-ask bounce, tick discreteness) is a property of
exchange mechanics and should be roughly regime-invariant. True price
variance is regime-dependent — that's the entire premise of the IMM filter
this project is building toward. Estimating microstructure noise from a
period that contains both a calm stretch and a real volatility event
conflates the two, and the noise estimate ends up contaminated by genuine
signal. So the calm window has to be *verified* calm, not assumed calm from
context (see: rejected first attempt, below).

## Method

1. Compute daily realized variance (1-minute sampling, `rv_1m`) across a
   candidate month.
2. Trim outlier days via a robust modified z-score (median + MAD, Iglewicz &
   Hoya convention, threshold = 3.5) — robust because it's centered on the
   median, so the outliers being screened for don't skew the statistic used
   to detect them.
3. Build a bootstrap null: repeatedly draw random 7-day groupings from the
   *trimmed* daily RVs, compute the largest single day's share of each
   group's total RV. This is "what dominant-day-share looks like among
   genuinely homogeneous days."
4. Set the concentration cutoff at the 90th percentile of that null.
5. Reject any real 7-day window whose largest-day RV share exceeds the
   cutoff; among the rest, take the lowest-total-RV window as the calm
   baseline.
6. Sanity check: does the trim exclude the days that also look like spikes
   by eye in the raw daily table? If not, don't trust the result — see below.

Script: `scripts/pick_calm_window.py`

## Attempt 1 (rejected): 2022-04

```
python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2022-04.zip
```

Result: outlier trim excluded **0 of 30 days**. Not because April is
uniformly calm — the daily RVs range smoothly from 0.000217 to 0.001331, a
continuum rather than a cluster with a few days sticking out, so no day is
far enough in MAD units to trip the threshold. Interpretation: April 2022 is
a gradual pre-crash ramp (it's the month directly before LUNA/UST collapsed
in May), not two separable populations of "calm" and "spike" days. Lowering
the MAD threshold until it excluded something was considered and rejected —
that would be tuning the detector to manufacture the split I wanted rather
than reporting what the data actually shows.

Consequence: fell back to the pre-designated 2023-06 calm baseline instead
of forcing a window out of April.

## Attempt 2 (accepted): 2023-06

```
python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2023-06.zip
```

Result: outlier trim excluded **3 of 30 days** (2023-06-21, 2023-06-23,
2023-06-30) — and these are visibly the three spikes in the raw daily
table (RVs of 0.00128, 0.000957, 0.00152, against a cluster mostly in
0.0002–0.0006 for the rest of the month). The trim agrees with what's
visible by eye — this is the bar for trusting the method, not just the
check passing on its own.

Bootstrap null (5,000 draws, 7-day windows, seed=0): 90th-percentile
largest-day-share cutoff = **29.1%**.

**Selected window: 2023-06-13 → 2023-06-19**
- Total RV: 0.002628 (lowest among passing candidates)
- Largest single-day share: 23.3% (below cutoff, no internal spike)
- All seven days within the same order of magnitude (0.000168–0.000613)
- Does not overlap any of the three trimmed outlier days

## Reproducibility

- Script: `scripts/pick_calm_window.py`
- Command: `python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2023-06.zip`
- Defaults used: `--days 7 --null-percentile 90 --bootstrap-draws 5000 --seed 0`
- MAD trim threshold: 3.5 (cited convention, not fitted)

## Next step

Use this window (2023-06-13 → 2023-06-19) as the input to the multi-offset
signature plot for deriving **R** (microstructure noise), separately from
the regime-varying **Q** derivation on 2022-05.
