# Innovation Gating
### Identifiability and Artifact-Robustness in Latent-State Financial Filtering

A research project applying particle-tracking state-space methods to financial regime detection — and, more importantly, stress-testing whether apparent regime-predictive power is genuine latent structure or an artifact of label construction.

See `PROJECT_PROPOSAL.md` for the full narrative write-up (motivation, phase-by-phase detail, derivation log, interview talking points). This README is the live status tracker — quick orientation, not the full story.

## Motivation

Charged-particle tracking and latent-state estimation in financial time series share identical mathematical structure: a Kalman filter recovers a hidden state (position/momentum, or price/drift/volatility) from noisy sequential measurements, using a process model for the state's dynamics and a measurement model for observation noise. Multi-hypothesis tracking (competing track candidates) maps onto multi-regime filtering (Interacting Multiple Model / regime-switching filters). Thresholds in both domains — hit-to-track association gates, regime-change detection — should be derived from the innovation covariance and an explicit error-rate target, not chosen by hand.

This project also carries over a specific failure mode: a conditional-generalization failure discovered in HEP analysis, where a classifier exploited artifacts of label construction rather than genuine conditional signal, fixed via matched hypothesis sampling and group-aware validation. The financial analogue is well known but usually under-formalized: classifiers "detecting" regimes by exploiting look-ahead/overlap structure in how labels were built, rather than real market dynamics — the same failure mode that motivates purged, embargoed cross-validation (López de Prado, *Advances in Financial Machine Learning*, 2018).

**Core question:** when does a latent-state filter's apparent gain in regime-identification power reflect real dynamical structure, versus an artifact of how the regime labels were constructed? And can every design threshold in the pipeline be derived from a loss function or statistical constraint rather than stated arbitrarily?

This is explicitly **not** an attempt to demonstrate a profitable trading strategy. The deliverable is a methodological/identifiability result.

## Project structure

| Phase | Weeks | Content | Status |
|---|---|---|---|
| 1 | 1–2 | State-space filter (Kalman → IMM), every parameter derived; baseline filter evaluation; naive-baseline comparison; bipower jump/diffusion split. | **Fully complete** |
| 2 | 2–4 | Artifact stress test: leaky vs. purged label sets, permutation-derived null, walk-forward validation. | Not started |
| 3 | 4–6 | Calibration (reliability diagrams, proper scoring rules) and a Bayes-risk decision threshold; cost-aware reality check using the derived spread estimates. | Not started |
| 4 | 6–8 | Stretch: latency/accuracy Pareto frontier under model compression, or cross-asset extension via a GNN. | Not started |

## Data

- **Primary:** Binance public spot market data ([data.binance.vision](https://data.binance.vision), [binance/binance-public-data](https://github.com/binance/binance-public-data)), `aggTrades`, BTCUSDT. Freely downloadable, fully redistributable.
- **Cross-check:** [LOBSTER](https://lobsterdata.com) free academic sample files (NASDAQ, full order-book depth) — reserved as an optional future check; not used so far, since R was ultimately derived directly from Binance tick data via Roll's estimator.

### Pulled windows (BTCUSDT, spot, aggTrades)

Five contiguous 3-month windows chosen to capture actual regime *transitions*, not just periods inside a labeled regime, plus one calm baseline. All 15 monthly files downloaded and checksum-verified.

- 2020-02 → 2020-04 (COVID crash) — downloaded, not yet used
- 2021-04 → 2021-06 (May 2021 crash) — downloaded, not yet used
- 2022-04 → 2022-06 (LUNA/UST collapse) — **primary working window**, all Phase 1 work done here
- 2022-10 → 2022-12 (FTX collapse) — downloaded, not yet used
- 2023-06 → 2023-08 (calm baseline) — **verified calm window derived here** (2023-06-13 → 2023-06-19), used for the R derivation

**Note:** Binance SPOT timestamps are in milliseconds before 2025-01-01 and microseconds from 2025-01-01 onward. Any window straddling that boundary needs unit-aware parsing (doesn't affect any window pulled so far).

## Model status: K=4 is canonical

The IMM was first fit at K=5 (BIC-selected) but showed a real flickering artifact between two low-variance states (self-transition 0.58). A K=4 recheck resolved this cleanly — persistence result preserved (marginally improved), calmest state's self-transition properly sticky (0.9745), and a clean structural mapping from K=5's other four states. **K=4 is the model that feeds Phase 2.**

`scripts/run_imm_filter.py` still contains the retired K=5 parameters as hardcoded constants — not yet updated. Use `scripts/k4_recheck.py`'s output as the canonical filter run until that script is updated or replaced.

## Repository layout

```
data/            raw pulled zips + checksums (not committed — see .gitignore)
scripts/         download, derivation, and filter-runner scripts
notebooks/       phase-by-phase analysis
src/             filter implementations (kalman.py, imm.py) -- both unit-validated
                 against synthetic data with known ground truth before real use
notes/           derivations — every threshold in this project traces to a written
                 derivation here, including rejected attempts
plots/           generated figures (convention adopted partway through Phase 1;
                 earlier plots saved to the working directory instead)
```

## Status

- [x] Project scoped
- [x] Data pulled and checksum-verified (15 files, 5 windows)
- [x] Calm-window selection derived and verified (2023-06-13 → 2023-06-19)
- [x] Measurement noise (R) derived — Roll's estimator, regime-conditionality tested and confirmed necessary (86.7x higher during crisis)
- [x] Process noise (Q) derived — rolling realized-variance rate, dt=60s, 1D window
- [x] Baseline (single-hypothesis) Kalman filter implemented and evaluated — found miscalibrated in a specific, diagnosed way
- [x] Detection gate derived from innovation covariance via chi-square test, explicit false-alarm-rate target
- [x] IMM extension implemented, K selected via BIC (K=5), then revised to K=4 after a flickering artifact was diagnosed and resolved
- [x] Naive baseline model built and honestly compared (IMM's complexity justified by signal shape/contrast, not raw magnitude)
- [x] Bipower-variation jump/diffusion decomposition — found LUNA registered as sustained diffusion, not a statistical jump; a genuine, useful finding
- [x] **Phase 1 fully complete**
- [ ] Artifact stress test (Phase 2)
- [ ] Calibration and decision-threshold derivation (Phase 3)

See `notes/k4_naive_bipower_phase1_final_closeout.md` for the final Phase 1 write-up.

## Non-goals

- This is not a backtested trading strategy and makes no profitability claims.
- Public daily/short-horizon data over a few weeks of work is not sufficient evidence of exploitable alpha, and no such claim is made.
- The jump-test's ~9-flagged-days-vs-1-target overshoot is noted as an open calibration question, not silently accepted as clean.
