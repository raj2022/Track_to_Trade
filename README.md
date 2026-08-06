# Innovation Gating
### Identifiability and Artifact-Robustness in Latent-State Financial Filtering

A research project applying particle-tracking state-space methods to financial regime detection — and, more importantly, stress-testing whether apparent regime-predictive power is genuine latent structure or an artifact of label construction.

See `PROJECT_PROPOSAL.md` for the full narrative write-up (motivation, phase-by-phase detail, derivation log, interview talking points). This README is the live status tracker — quick orientation, not the full story.

## Motivation

Charged-particle tracking and latent-state estimation in financial time series share identical mathematical structure: a Kalman filter recovers a hidden state (position/momentum, or price/drift/volatility) from noisy sequential measurements, using a process model for the state's dynamics and a measurement model for observation noise. Multi-hypothesis tracking (competing track candidates) maps onto multi-regime filtering (Interacting Multiple Model / regime-switching filters). Thresholds in both domains — hit-to-track association gates, regime-change detection — should be derived from the innovation covariance and an explicit error-rate target, not chosen by hand.

This project also carries over a specific failure mode: a conditional-generalization failure discovered in HEP analysis, where a classifier exploited artifacts of label construction rather than genuine conditional signal, fixed via matched hypothesis sampling and group-aware validation. The financial analogue is well known but usually under-formalized: classifiers "detecting" regimes by exploiting look-ahead/overlap structure in how labels were built, rather than real market dynamics — the same failure mode that motivates purged, embargoed cross-validation (López de Prado, *Advances in Financial Machine Learning*, 2018). **This is no longer just an analogy — Phase 2 built and ran the direct financial test of this failure mode (see below).**

**Core question:** when does a latent-state filter's apparent gain in regime-identification power reflect real dynamical structure, versus an artifact of how the regime labels were constructed? And can every design threshold in the pipeline be derived from a loss function or statistical constraint rather than stated arbitrarily?

This is explicitly **not** an attempt to demonstrate a profitable trading strategy. The deliverable is a methodological/identifiability result.

## Project structure

| Phase | Weeks | Content | Status |
|---|---|---|---|
| 1 | 1–2 | State-space filter (Kalman → IMM), every parameter derived; baseline filter evaluation; naive-baseline comparison; bipower jump/diffusion split. | **Fully complete** |
| 2 | 2–4 | Artifact stress test: leaky vs. purged label sets, permutation-derived null, walk-forward validation. | **Core deliverable complete: leakage mechanism demonstrated (p≈10⁻⁶) and real-label signal validated (≈6σ above artifact baseline); cross-window robustness check outstanding** |
| 3 | 4–6 | Calibration (reliability diagrams, proper scoring rules) and a Bayes-risk decision threshold; cost-aware reality check using the derived spread estimates. | **Complete: five recalibration attempts all failed (honestly documented); Bayes-risk threshold derived and applied to raw probabilities, +48.6% cost reduction vs. naive** |
| 4 | 6–8 | Stretch: latency/accuracy Pareto frontier under model compression, or cross-asset extension via a GNN. | Not started |

## Headline Phase 2 result

A classifier trained on a label engineered to have **zero true relationship** to its features shows overwhelming apparent skill under naive k-fold cross-validation — every one of 20 independent trials came back positive (p≈10⁻⁶ under a fair-coin null) — and that apparent skill drops by roughly 8x under purged, embargoed walk-forward validation. Getting to this result required diagnosing and fixing two separate broken null-label constructions and a previously-invisible confound (the null label's own autocorrelation being separately exploitable by each CV scheme's mechanics, independent of any feature).

**Closing the loop:** the same correction was then applied to the *real* label. Its feature-driven signal (purged diff +0.383) sits ≈6σ above the null-shift-calibrated artifact-only baseline — genuine, not just an autocorrelation artifact — while naive CV's inflated version of that same result decomposes almost exactly into real signal (≈0.38) plus naive's own characteristic leakage inflation (≈0.35, matching the null-label baseline). **Phase 2's core deliverable is complete.** Full arc: `notes/phase2_leakage_stress_test_full_arc.md`.

## Headline Phase 3 result

Reliability checking found the classifier's raw probabilities systematically underconfident in the ~0.05–0.5 range (up to 8σ off). **Five separate recalibration attempts** — static time split, static regime split, walk-forward isotonic regression (both expanding and rolling window), walk-forward Platt scaling — **all failed** to produce a robust fix, a genuine and instructive negative result documented in full rather than papered over. Decision: proceed with raw probabilities, stating the limitation explicitly. The Bayes-risk decision threshold (p*≈0.097, derived from the Roll spread ratio established in Phase 1) was then applied to raw probabilities and reduced mean expected cost by **48.6%** versus the naive p=0.5 cutoff — and its empirical cost-minimizing threshold landed close to the theoretically derived one despite the documented miscalibration, a real if imperfect validation. Full arc: `notes/phase3_calibration_drift.md`, `notes/phase3_bayes_threshold_result.md`.

## Data

- **Primary:** Binance public spot market data ([data.binance.vision](https://data.binance.vision), [binance/binance-public-data](https://github.com/binance/binance-public-data)), `aggTrades`, BTCUSDT. Freely downloadable, fully redistributable.
- **Cross-check:** [LOBSTER](https://lobsterdata.com) free academic sample files (NASDAQ, full order-book depth) — reserved as an optional future check; not used so far, since R was ultimately derived directly from Binance tick data via Roll's estimator.

### Pulled windows (BTCUSDT, spot, aggTrades)

Five contiguous 3-month windows chosen to capture actual regime *transitions*, not just periods inside a labeled regime, plus one calm baseline. All 15 monthly files downloaded and checksum-verified.

- 2020-02 → 2020-04 (COVID crash) — downloaded, not yet used
- 2021-04 → 2021-06 (May 2021 crash) — downloaded, not yet used
- 2022-04 → 2022-06 (LUNA/UST collapse) — **primary working window**, all Phase 1 and Phase 2 work done here
- 2022-10 → 2022-12 (FTX collapse) — downloaded, not yet used
- 2023-06 → 2023-08 (calm baseline) — **verified calm window derived here** (2023-06-13 → 2023-06-19), used for the R derivation

**Note:** Binance SPOT timestamps are in milliseconds before 2025-01-01 and microseconds from 2025-01-01 onward. Any window straddling that boundary needs unit-aware parsing (doesn't affect any window pulled so far).

## Model status: K=4 is canonical

The IMM was first fit at K=5 (BIC-selected) but showed a real flickering artifact between two low-variance states (self-transition 0.58). A K=4 recheck resolved this cleanly — persistence result preserved (marginally improved), calmest state's self-transition properly sticky (0.9745), and a clean structural mapping from K=5's other four states. **K=4 is the model that feeds Phase 2.**

`scripts/run_imm_filter.py` still contains the retired K=5 parameters as hardcoded constants — not yet updated. Use `scripts/k4_recheck.py`'s output as the canonical filter run until that script is updated or replaced.

## Repository layout

```
data/            raw pulled zips + checksums (not committed — see .gitignore)
                 data/processed/ holds the Phase 2 dataset (features + labels)
scripts/         download, derivation, filter-runner, and Phase 2 stress-test scripts
notebooks/       phase-by-phase analysis (not yet populated -- planned as an
                 end-of-project interactive walkthrough, not used during active work)
src/             filter implementations (kalman.py, imm.py) and CV splitters
                 (cv_splits.py) -- all unit-validated against synthetic data
                 with known ground truth before real use
notes/           derivations — every threshold in this project traces to a
                 written derivation here, including rejected attempts
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
- [x] Naive baseline model built and honestly compared
- [x] Bipower-variation jump/diffusion decomposition — found LUNA registered as sustained diffusion, not a statistical jump
- [x] **Phase 1 fully complete**
- [x] Label horizon (H) derived from the IMM's own transition matrix via absorbing-Markov-chain sojourn time
- [x] Real and null labels built (null label required two redesigns before it was trustworthy)
- [x] Leakage mechanism empirically demonstrated: naive k-fold shows overwhelming feature-driven leakage (p≈10⁻⁶); purged walk-forward removes ~8x of it
- [x] Real-label signal validated against the null-shift-calibrated baseline (≈6σ above artifact-only noise, purged scheme) — Phase 2's core deliverable is complete
- [ ] Cross-window robustness check (confirm the leakage result isn't LUNA-specific) — tracked in `ISSUES.md`
- [x] Reliability check on real_label — found systematic underconfidence (up to 8σ) in the 0.05–0.5 range
- [x] Five recalibration attempts, all rejected, honestly documented (`notes/phase3_calibration_drift.md`)
- [x] Bayes-risk threshold derived from the Roll spread ratio (p*≈0.097), applied to raw probabilities — 48.6% cost reduction vs. naive p=0.5, validated against the empirical cost curve
- [x] **Phase 3 complete**

See `notes/phase3_bayes_threshold_result.md` for the final Phase 3 write-up. See `ISSUES.md` for tracked open items not currently blocking progress.

## Non-goals

- This is not a backtested trading strategy and makes no profitability claims.
- Public daily/short-horizon data over a few weeks of work is not sufficient evidence of exploitable alpha, and no such claim is made.
- The jump-test's ~9-flagged-days-vs-1-target overshoot is noted as an open calibration question, not silently accepted as clean.
- Purged walk-forward's leakage reduction is reported as "mostly eliminated" (~8x reduction, residual not conclusively distinguishable from zero given 20 shifts), not "leakage-free."
- The Phase 3 Bayes-risk threshold result (+48.6% cost reduction) is a decision-theoretic result under an explicit, derived cost model applied to raw (documented-as-imperfect) probabilities — not a backtest, not realized P&L, not a profitability claim.
