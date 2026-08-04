# Innovation Gating
### Identifiability and Artifact-Robustness in Latent-State Financial Filtering

A research project applying particle-tracking state-space methods to financial regime detection — and, more importantly, stress-testing whether apparent regime-predictive power is genuine latent structure or an artifact of label construction.

## Motivation

Charged-particle tracking and latent-state estimation in financial time series share identical mathematical structure: a Kalman filter recovers a hidden state (position/momentum, or price/drift/volatility) from noisy sequential measurements, using a process model for the state's dynamics and a measurement model for observation noise. Multi-hypothesis tracking (competing track candidates) maps onto multi-regime filtering (Interacting Multiple Model / regime-switching filters). Thresholds in both domains — hit-to-track association gates, regime-change detection — should be derived from the innovation covariance and an explicit error-rate target, not chosen by hand.

This project also carries over a specific failure mode: a conditional-generalization failure discovered in HEP analysis, where a classifier exploited artifacts of label construction rather than genuine conditional signal, fixed via matched hypothesis sampling and group-aware validation. The financial analogue is well known but usually under-formalized: classifiers "detecting" regimes by exploiting look-ahead/overlap structure in how labels were built, rather than real market dynamics — the same failure mode that motivates purged, embargoed cross-validation (López de Prado, *Advances in Financial Machine Learning*, 2018).

**Core question:** when does a latent-state filter's apparent gain in regime-identification power reflect real dynamical structure, versus an artifact of how the regime labels were constructed? And can every design threshold in the pipeline be derived from a loss function or statistical constraint rather than stated arbitrarily?

This is explicitly **not** an attempt to demonstrate a profitable trading strategy. The deliverable is a methodological/identifiability result.

## Project structure

| Phase | Weeks | Content |
|---|---|---|
| 1 | 1–2 | State-space filter (Kalman → IMM) with every parameter derived: process noise from realized quadratic variation, measurement noise from microstructure-noise estimation, transition probabilities from EM/Baum-Welch, detection gate from the innovation covariance via a chi-square test, jump/diffusion split via bipower variation. |
| 2 | 2–4 | Artifact stress test: construct regime-label sets with and without look-ahead/overlap contamination, quantify apparent predictive power against a permutation-derived null, apply purged + embargoed cross-validation. |
| 3 | 4–6 | Calibration: reliability diagrams and proper scoring rules (log score, CRPS) under purged out-of-sample evaluation; decision threshold derived from an explicit asymmetric cost matrix (Bayes risk), not fixed at 0.5. |
| 4 | 6–8 | Stretch: latency/accuracy Pareto frontier under model compression, or cross-asset extension via a GNN. |

## Data

- **Primary:** Binance public spot market data ([data.binance.vision](https://data.binance.vision), [binance/binance-public-data](https://github.com/binance/binance-public-data)), `aggTrades`, BTCUSDT. Freely downloadable, MIT-licensed access, fully redistributable.
- **Cross-check:** [LOBSTER](https://lobsterdata.com) free academic sample files (NASDAQ, full order-book depth) — used to calibrate the microstructure-noise model against a real limit order book rather than trades alone.

### Pulled windows (BTCUSDT, spot, aggTrades)

Five contiguous 3-month windows chosen to capture actual regime *transitions*, not just periods inside a labeled regime, plus one calm baseline:

- 2020-02 → 2020-04 (COVID crash)
- 2021-04 → 2021-06 (May 2021 crash)
- 2022-04 → 2022-06 (LUNA/UST collapse)
- 2022-10 → 2022-12 (FTX collapse)
- 2023-06 → 2023-08 (calm baseline)

**Note:** Binance SPOT timestamps are in milliseconds before 2025-01-01 and microseconds from 2025-01-01 onward. Any window straddling that boundary needs unit-aware parsing.

## Repository layout

```
data/            raw pulled zips + checksums (not committed — see .gitignore)
scripts/         download, verification, and preprocessing scripts
notebooks/       phase-by-phase analysis
src/             filter implementation, label construction, validation utilities
notes/           derivations — every threshold in this project traces to a written derivation here
```

## Status

- [x] Project scoped
- [ ] Data pulled and checksum-verified
- [ ] Process noise (Q) derived from realized quadratic variation
- [ ] Measurement noise (R) derived from microstructure-noise estimator
- [ ] Baseline Kalman filter implemented
- [ ] IMM extension implemented
- [ ] Detection gate derived from innovation covariance
- [ ] Artifact stress test built
- [ ] Calibration and decision-threshold derivation

## Non-goals

- This is not a backtested trading strategy and makes no profitability claims.
- Public daily/short-horizon data over a few weeks of work is not sufficient evidence of exploitable alpha, and no such claim is made.
