# Microstructure noise (R) derivation

**Date:** Day 2/3
**Question:** What is the measurement noise variance R for the Kalman
filter's price observation model, derived from data rather than assumed?

## Why the interval-based (signature plot) approach was abandoned here

The initial plan was to read R off the plateau of a volatility signature
plot (RV vs. sampling interval) on the verified calm window (2023-06-13 to
2023-06-19; see `calm_window_selection.md`). Running the multi-offset
signature plot (`scripts/signature_plot.py`) on that window showed:

- RV *rising* from 1s to a peak near 15-30s, then decaying to a plateau at
  600-3600s — the inverse of the textbook shape, where RV is highest at the
  shortest interval (bid-ask bounce) and decays to the true-variance
  plateau.
- A trade-arrival diagnostic explained why: 44.7% of 1-second bins were
  stale (repeated price, no new trade), decaying to ~0% by 600s. BTCUSDT in
  this calm week doesn't trade fast enough for 1s bins to reflect fresh
  information — short-interval RV was biased *low* by stale prices, not
  inflated by noise.
- Conclusion: the 600-3600s plateau is a good estimate of the window's
  *true* variance level (useful later as a Q benchmark), but the whole
  fixed-interval approach is the wrong tool for estimating R here, since
  the assumed noise mechanism (bounce inflating short-interval RV) doesn't
  hold for this data.

## Method: Roll's (1984) implied-spread estimator

Uses tick-to-tick log returns directly, with no interval choice:

$$\sigma_\varepsilon^2 = -\text{Cov}(r_t, r_{t-1})$$

Bid-ask bounce induces negative first-order autocorrelation in consecutive
trade returns (a trade at the ask followed by one at the bid looks like a
price drop, and vice versa, with no true price movement). Sidesteps the
staleness problem entirely since every return comes from an actual trade,
not a time bin.

Script: `scripts/roll_estimator.py`

Design decisions, made deliberately rather than defaulted:
- **Zero returns kept.** A zero return here means two consecutive real
  trades printed at the same price — an observed outcome, not a
  manufactured artifact. (Distinct from the earlier stale-price problem:
  there, "no change" was manufactured by resampling a bin with no trade in
  it at all. Same surface symptom, different and non-transferable
  mechanism — worth keeping these separate rather than treating "zero
  return" as one uniform category across the project.)
- **Same-timestamp consecutive pairs initially kept**, then stress-tested
  (below) rather than assumed safe.

## Result

```
python scripts/roll_estimator.py data/raw/BTCUSDT-aggTrades-2023-06.zip \
  --start 2023-06-13 --end 2023-06-19
```

- 4,363,118 ticks in window.
- Lag-1 return autocovariance: **-3.661341181889e-11** (negative — Roll's
  model holds; a positive value here would have meant the model doesn't
  apply and a different approach would be needed).
- All-tick R: **3.661341181889e-11**

### Sanity check (unit conversion)

Roll's implied effective spread: $2\sqrt{R}$.

$\sqrt{3.661\times10^{-11}} = 6.05\times10^{-6}$, so $2\sqrt{R} = 1.210\times10^{-5}$
in log-return units ≈ **0.121 bp** (corrected — an earlier back-of-envelope
pass mis-scaled this to 1.2bp; caught by converting to dollar terms and
checking the order of magnitude rather than trusting the first pass).

At ~$25-26k BTC, this is roughly **$0.03 per BTC** — above the $0.01 tick
size, and a plausible effective spread for BTCUSDT spot, the most liquid
pair in crypto, during a verified calm week. This should be read as a
Roll-implied *effective* spread from aggregate-trade prices, not equated
directly with the exchange's displayed quoted spread.

### Same-timestamp robustness check

Concern: Binance emits one aggTrade row per price level when a single
aggressive order sweeps multiple levels, all sharing one millisecond
timestamp. Consecutive rows within such a sweep move monotonically (walking
deeper into the book) rather than bouncing — a different serial-correlation
mechanism than the bounce effect Roll's estimator is meant to isolate, and
one that could bias the covariance toward positive (understating R).

- Fraction of consecutive pairs sharing a timestamp: **33.6%** — not a
  negligible edge case.
- R excluding same-timestamp pairs (2,448,507 observations remaining):
  **4.098231808572e-11**
- Difference from all-tick R: **+11.93%**

Interpretation: the hypothesis was directionally correct — contaminated
pairs pull the covariance toward zero (less negative), consistent with
partially-offsetting monotonic sweep moves. The effect is real and
quantifiable (~12%), not negligible, but also not so large that it
invalidates the all-tick estimate outright. These are two different
estimands, not a wrong and a right answer:
- All-tick R answers "average noise variance including within-sweep
  dynamics."
- Excluded-pair R answers "noise variance from independent, book-refreshing
  trades only."

**Decision: use the all-tick R (3.661e-11)** as the filter's measurement
noise, on the grounds that the Kalman filter will ingest aggTrades rows
as-is, sweep bursts included — the all-tick estimate is the one internally
consistent with what the filter actually observes. If the pipeline is later
changed to deduplicate/collapse sweeps before filtering, switch to the
excluded-pair estimate to match.

## Reproducibility

- Script: `scripts/roll_estimator.py`
- Command: `python scripts/roll_estimator.py data/raw/BTCUSDT-aggTrades-2023-06.zip --start 2023-06-13 --end 2023-06-19`
- Data: 2023-06 aggTrades, filtered to the verified calm window
  2023-06-13 to 2023-06-19.

## Next step

Derive the regime-varying **Q** on 2022-05 (LUNA window), using this R
(3.661e-11) as the calibrated measurement-noise floor.
