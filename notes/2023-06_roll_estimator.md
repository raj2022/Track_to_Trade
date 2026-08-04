# Tick-level Roll estimator for the R (measurement-noise) derivation

**Question:** Does Roll's implied-spread estimator produce a valid
microstructure-noise variance for the verified-calm BTCUSDT window, and is
the estimate sensitive to Binance aggTrades that share a millisecond
timestamp?

## Data and estimator

- Archive: `data/raw/BTCUSDT-aggTrades-2023-06.zip`
- Window: 2023-06-13 through 2023-06-19 UTC (inclusive)
- Observations: 4,363,118 aggregate trades
- Estimator: sample lag-1 covariance of consecutive tick-level log returns,
  with `ddof=1`:

  \[
  R = -\operatorname{Cov}(r_t, r_{t-1}).
  \]

The primary estimate uses every aggregate trade in file order. The loader
uses a stable sort, so this order is preserved when multiple aggregate trades
share a timestamp.

## Primary result

```
Lag-1 return autocovariance: -3.661341181889e-11
Roll noise variance (R):     3.661341181889e-11
```

The covariance is negative, so the model produces a positive implied noise
variance. This sign check is necessary for the Roll interpretation; a
positive covariance would have been reported as model failure rather than
silently negated.

The associated Roll effective spread is

\[
2\sqrt{R} = 1.210180347203\times10^{-5}
\]

in log-return units, or **0.121018 bp**. At BTC prices around $25,000--26,000,
that is approximately **$0.030--$0.031** per BTC. This is an effective spread
inferred from aggregate-trade prices, not a direct measurement of displayed
quoted spread.

## Tick-time decisions

### Zero returns: retained

1,757,531 of 4,363,117 returns (40.28%) are zero. They are retained because
they arise from consecutive observed aggregate trades at the same price.
They are not the stale prices encountered in calendar-time resampling, where
an empty bin is forward-filled and a no-change observation is manufactured
from absent trading. Filtering the observed zero returns would condition the
estimator on price changes and alter its tick-time estimand.

### Same-millisecond returns: retained in the primary estimate, then tested

1,465,607 of 4,363,117 returns (33.59%) occur between aggregate trades with
the same millisecond timestamp. They are retained in the primary tick-time
estimate because they are distinct observed events. But an aggressive order
may sweep multiple price levels within one millisecond, creating monotonic
same-millisecond moves that are not bid-ask bounce. This can bias the
lag-one covariance upward and therefore bias the all-tick Roll estimate of
R downward.

For the robustness check, a covariance observation `(r_t, r_{t-1})` was used
only when *both* returns cross distinct timestamps. This excludes covariance
observations containing a same-millisecond return without filtering the return
array first and accidentally pairing non-adjacent returns. The check retains
2,448,507 of 4,363,116 covariance observations (56.12%).

| Specification | Lag-1 covariance | R | Implied spread |
| --- | ---: | ---: | ---: |
| All aggregate-trade ticks (primary) | -3.661341181889e-11 | 3.661341181889e-11 | 0.121018 bp |
| Exclude covariance observations containing a same-millisecond return | -4.098231808572e-11 | 4.098231808572e-11 | 0.128035 bp |

The robustness estimate is **11.93%** larger. The negative sign persists and
the effective spread changes by only 0.0070 bp, but the difference is not
zero: aggTrades' within-millisecond grouping is a documented source of
estimation uncertainty. The primary calibration remains the literal all-tick
estimator; the timestamp-excluded result should accompany it as a sensitivity
bound rather than being substituted silently.

## Reproducibility

```
python scripts/roll_estimator.py data/raw/BTCUSDT-aggTrades-2023-06.zip \
    --start 2023-06-13 --end 2023-06-19
```

The command prints the all-tick primary estimate and the same-timestamp-
excluded robustness estimate together.
