# Innovation Gating
### Identifiability and Artifact-Robustness in Latent-State Financial Filtering
**Project proposal**

---

## 1. One-line thesis

State-space filtering methods from charged-particle tracking transfer directly to financial regime detection — but the interesting research question is not "can a Kalman filter find regimes," it's **when does apparent regime-predictive power reflect genuine latent structure versus an artifact of how the labels and validation scheme were built**, and can every design threshold in the pipeline be derived from a loss function or statistical constraint instead of stated by hand.

This is a methodological/identifiability project, not a trading-strategy project. No profitability claim is made or implied anywhere in this proposal.

---

## 2. Motivation: why this bridge is real, not decorative

Charged-particle tracking and latent-state estimation in financial time series share identical mathematical structure.

| Track reconstruction | State-space finance |
|---|---|
| Latent state: position, momentum, curvature | Latent state: fair price, local drift, volatility regime |
| Process model: helical motion + multiple scattering | Process model: regime-dependent drift/volatility dynamics |
| Measurement: noisy detector hits | Measurement: trade prints with microstructure noise |
| Multiple track hypotheses (combinatorial ambiguity) | Multiple regime hypotheses (IMM / regime-switching filters) |
| Chi-square gate for hit-to-track association, derived from the innovation covariance | Chi-square gate for outlier/regime-change detection, derived from the innovation covariance |
| Multiple scattering (diffusive noise) vs. hard interaction (discrete energy loss) | Continuous diffusion vs. price jumps (bipower variation) |

This is a legitimate, citable technique, not an invented analogy — Interacting Multiple Model Kalman filter banks are an established approach to regime-switching volatility forecasting in the finance literature. The IMM is implemented, tested, and was itself revised via a genuine model-selection reversal (K=5 → K=4) once a behavioral problem was found in what BIC alone selected (Section 6).

The project also carries over a specific failure mode discovered independently, in a different domain: a conditional-generalization failure in HEP analysis, where a classifier exploited artifacts of label construction rather than genuine conditional signal, fixed via matched hypothesis sampling and group-aware validation. **This is no longer just a motivating analogy — Phase 2 built the direct financial test of the same failure mode and demonstrated it empirically** (Section 6): a classifier shows overwhelming, statistically unambiguous apparent skill (every one of 20 independent trials positive, p≈10⁻⁶ under a fair-coin null) on a label engineered to carry zero true relationship to its features, when evaluated under naive cross-validation — and that apparent skill collapses by roughly 8x under purged, embargoed walk-forward validation. Arriving at the same underlying fix (purged/embargoed CV) independently, from physics, remains the project's genuine differentiator — this project re-derives the principle from a structurally identical problem encountered first in particle tracking, and then built and ran the actual empirical test most people who cite the principle never construct themselves.

---

## 3. Core research question

> When does a latent-state filter's apparent gain in regime-identification power reflect real dynamical structure, versus an artifact of how the regime labels or validation scheme were constructed? And can every design threshold in the pipeline — gating windows, transition probabilities, detection thresholds, decision boundaries — be derived from an explicit loss function or statistical constraint, rather than stated arbitrarily?

Secondary, added after review against current industry practice (Section 8): does a signal that survives the artifact-robustness stress test also survive realistic transaction costs, or does it only look real in a costless world?

---

## 4. Derivation discipline

This is the project's organizing constraint, not a stylistic preference. Every parameter must trace to one of:
- a maximum-likelihood fit to data (e.g. regime transition probabilities via EM/Baum-Welch),
- a statistical test with an explicit error-rate target (e.g. the chi-square detection gate from the innovation covariance),
- a decision-theoretic optimum under an explicit cost function (e.g. the regime-change threshold as a Bayes-risk minimizer), or
- a robust, cited convention with a stated justification for why it's appropriate here (e.g. the MAD-based outlier trim, threshold = 3.5, Iglewicz & Hoya).

No threshold is accepted because "it looked right on a plot." Every `notes/` entry documents the derivation, including rejected attempts — see Section 9.

**Three additions worth stating explicitly after Phase 1 and the Phase 2 stress test.** First, the discipline applies to reporting results, not just choosing parameters — decaying or partial results (the IMM's persistence decay; the purged CV's "mostly, not perfectly, eliminated" leakage residual) are reported as such, not rounded up to cleaner-sounding claims. Second, it applies to model-selection criteria themselves — BIC's K=5 selection was overridden once a behavioral flaw was found. **Third, and new: even a "null" construction — the thing meant to be the trusted reference point for judging everything else — is itself a claim that needs verifying, not a given.** Two null-label constructions failed for diagnosable reasons before a third, correctly-controlled comparison (real model AUC minus a feature-blind dummy baseline) produced a trustworthy result. Nothing in this pipeline is exempt from the same scrutiny, including the tools built specifically to provide scrutiny.

---

## 5. Data

- **Primary:** Binance public spot market data (`data.binance.vision`, mirrored at `github.com/binance/binance-public-data`), BTCUSDT, `aggTrades`. Free, no API key, redistributable, tick-resolution, extending back to 2017.
- **Cross-check:** LOBSTER free academic sample files (NASDAQ, full order-book depth) — reserved for validating the measurement-noise approach against a real limit order book, not just trade prints. Not yet used — R was ultimately derived via Roll's estimator directly on Binance tick data instead (Section 9), so this remains an optional future cross-check rather than a blocking dependency.

### Pulled windows (BTCUSDT spot, aggTrades)

Five contiguous 3-month windows, chosen to capture actual regime *transitions* rather than sit entirely inside one labeled regime:

- 2020-02 → 2020-04 (COVID crash)
- 2021-04 → 2021-06 (May 2021 crash)
- 2022-04 → 2022-06 (LUNA/UST collapse)
- 2022-10 → 2022-12 (FTX collapse)
- 2023-06 → 2023-08 (calm baseline)

All 15 monthly files downloaded and checksum-verified. Row counts confirmed the expected within-window pattern (event month > its calm neighbors) in every window; cross-window row-count comparisons were explicitly rejected as a volatility proxy, since they're confounded with secular growth in exchange activity over 2020–2023.

**Working window so far:** all Phase 1 and Phase 2 work has been done on 2022-05 (LUNA) and 2023-06 (calm baseline). The other three event windows (COVID, May 2021, FTX) are downloaded but not yet used — a natural robustness check to confirm the leakage-stress-test result isn't LUNA-specific.

---

## 6. Project phases

### Phase 1 (weeks 1–2): Filter, with every parameter derived — **fully complete**

- **Baseline (single-hypothesis) local-level Kalman filter: built, derived, evaluated, and found to fail in a specific, quantified way.** Miscalibrated in all three (pre/during/post) periods around the LUNA event, ruling out an event-specific cause; the mechanism — a trailing 1-day Q re-absorbing a sustained anomaly into its own baseline within about a day — became the concrete motivation for the IMM. Full detail: `notes/baseline_filter_evaluation.md`.
- **IMM filter: built, tested, and revised via a genuine model-selection reversal.** K=5 initially BIC-selected, but a flickering, non-sticky state pair (self-transition 0.58) was found and diagnosed; a K=4 refit was tested directly against the actual downstream persistence result (not just BIC) and adopted — persistence preserved, calmest state properly sticky (0.9745), clean structural mapping to K=5's other states. Per-regime Q/R derived via Roll's estimator restricted to each state's own tick pairs; regime-conditionality of R tested and confirmed necessary (86.7x higher in crisis).
- **Naive baseline and bipower jump/diffusion split: built, both yielding genuine findings.** The IMM's complexity was shown to be justified by signal *shape* (a real, stable low-noise baseline state) rather than raw magnitude. The bipower decomposition showed LUNA registered as sustained diffusion, not a discrete jump — direct empirical validation of the IMM's framing for this specific event.

Full detail: `notes/k4_naive_bipower_phase1_final_closeout.md`.

### Phase 2 (weeks 2–4): Artifact stress test — the centerpiece, complete and cross-window validated

- **Label horizon H derived, not asserted.** Treated the IMM's elevated states as transient states of an absorbing Markov chain; computed expected sojourn time via the fundamental matrix, weighted by the quasi-stationary distribution over the elevated set — 54 minutes, rounded to 60 for practical use (`scripts/derive_horizon.py`). Deliberately calibrated to a *typical* elevated episode, not the exceptional multi-day LUNA event, since the classifier trains across the whole dataset.
- **Real and null labels built** (`scripts/build_phase2_dataset.py`): real label = forward-window mean elevated probability exceeding 0.5 (a natural indifference point); null label = a genuinely fake target with the label's real marginal distribution and autocorrelation structure preserved, but its temporal alignment with the features destroyed.
- **The null construction itself failed twice, for two distinct, diagnosable reasons, before it could be trusted.** Block permutation introduced a base-rate mismatch artifact (rare positive blocks relocated onto a mostly-low-feature timeline, producing a spurious, large negative bias). A single circular shift then failed differently: this exact dataset has a known ~8h periodicity (discovered back in the Phase 1 Q derivation), and a circular shift can accidentally realign with real periodic structure, producing large, direction-inconsistent spurious effects. The final method — many circular shifts, explicitly excluding offsets near the known periods — resolved this.
- **A third, previously invisible confound was found and diagnosed via a decisive control.** Even with a clean null, purged walk-forward showed a consistent positive AUC bias across all 20 shifts while naive k-fold was noisy and near chance — backwards from the leakage hypothesis. A feature-blind dummy classifier (predicting only the training fold's class prior) reproduced this same pattern almost exactly, proving the bias was driven entirely by the label's own autocorrelation interacting with each CV scheme's mechanics (a structural, mathematically explicable anti-correlation bias in leave-block-out k-fold; a structural pro-correlation bias in expanding-window walk-forward), not by the features at all.
- **Corrected, decisive result** (`scripts/phase2_real_vs_dummy_comparison.py`): comparing real model AUC minus the dummy baseline, per shift, isolates the genuinely feature-driven effect. **Naive k-fold: +0.349 mean, 20/20 shifts positive (p≈10⁻⁶ under a fair-coin null) — overwhelming, unambiguous leakage.** **Purged walk-forward: +0.043 mean, std 0.055, 14/20 shifts positive (p≈0.04) — the large majority of the leakage effect is removed (~8x reduction), but the honest conclusion is "mostly eliminated," not "proven exactly zero."**
- **Closing the loop: the real label's own signal was calibrated against this same null-shift distribution, not left at face value.** Raw real-label AUC (0.918 naive, 0.914 purged) was always suspect given the label's near-tautological construction from the dominant feature. The real-vs-dummy correction gives real-label diffs of +0.784 (naive) / +0.383 (purged) — and calibrated against the null-shift diff distributions (naive: 0.349±0.118; purged: 0.043±0.055), the real label sits **≈3.7σ and ≈6.1σ above the artifact-only baseline respectively** — genuine, feature-driven signal, not just autocorrelation. A clean decomposition falls out: the real label's naive-vs-purged gap (0.400) nearly matches naive's own characteristic leakage inflation on the null labels (0.349), meaning the naive result ≈ genuine signal (0.38, matching purged) + naive's typical inflation (0.35) ≈ the observed 0.784. **This is Phase 2's complete, calibrated core result.** Full arc: `notes/phase2_leakage_stress_test_full_arc.md`.
- **Cross-window robustness: confirmed.** Reran the full real-vs-dummy comparison on an independent event window (FTX, 2022-11), refitting the K=4 IMM and all Phase 2 parameters fresh on that window's own data. Naive k-fold's leakage signature replicated almost identically (20/20 shifts positive, same effect size). The real label's signal replicated and was slightly *stronger* (≈4.0σ/7.4σ above artifact baseline vs. LUNA's ≈3.7σ/6.1σ). This also resolved whether LUNA's small purged-walk-forward residual (+0.043, 14/20 shifts positive) was a genuine leak: FTX showed +0.008 with exactly 10/20 shifts positive — a coin flip, consistent with sampling noise. One real bug caught in the process: `ELEVATED_STATES` had been hardcoded from May's specific HMM fit, which doesn't generalize since state indices are arbitrary per EM run — fixed to derive elevated states per-window by variance rank. Full detail: `notes/cross_window_robustness_check.md`.
- **Why this mattered methodologically:** the raw, uncorrected AUC comparison from the first run of this test would have supported the *opposite*, wrong conclusion (naive k-fold's raw AUC of ~0.48 looked like "no leakage," when in fact a large positive leakage effect was being masked by an unrelated negative structural bias in the same scheme). Only building a proper feature-blind control — itself only necessary because two earlier null constructions had already failed — uncovered the true, correctly-isolated effect. The same discipline then applied to the real label prevented a second potential error: trusting the raw 0.918 AUC at face value instead of calibrating it against the same null-shift-derived baseline.

### Phase 3 (weeks 4–6): Calibration and decision theory — **complete**

- **Reliability check on the purged-walk-forward `real_label` predictions found real, systematic underconfidence** in roughly the 0.05–0.5 predicted-probability range (5 of 10 quantile bins, up to 8σ off per the binomial standard error, consistent in direction) — not visible on the plot alone, only in the per-bin SE table.
- **Five recalibration attempts, all rejected, each tested and diagnosed rather than assumed:**
  1. Isotonic regression, static chronological time split — made things *worse* (log score 0.305→0.366), in the *opposite* miscalibration direction on the held-out half.
  2. Isotonic regression, regime-conditional split (elevated_prob median) — worse in both regimes, more severely; refuted the regime-dependence hypothesis.
  3–4. Isotonic regression, walk-forward daily refit (expanding, then rolling 5-day window) — neither improved on real data; the rolling version also revealed ~10% of predictions fell in windows containing only one class, a genuine fragility of short windows against a rare, autocorrelated label.
  5. Platt scaling (2-parameter logistic fit), walk-forward rolling — tested the hypothesis that isotonic's flexibility, not the window shape, was the problem; succeeded on synthetic drift data but only marginally moved aggregate scores on real data while making several individual bins *worse*.
  Full arc, including the reasoning discarded at each step: `notes/phase3_calibration_drift.md`.
- **Decision: proceed with raw (uncalibrated) probabilities**, stating the known miscalibrated range explicitly as a limitation rather than continuing to search for a sixth variant — five attempts spanning both plausible axes (split scheme, calibrator flexibility) constitutes a genuine negative result, and further iteration risked becoming exactly the kind of overfitting-to-one-dataset this project is built to catch.
- **Bayes-risk threshold derived from Phase 1's own numbers**, not asserted: cost ratio C_FN/C_FP = Roll-implied spread ratio (crisis/calm) ≈ 9.31, giving p* ≈ 0.097. Applied to raw probabilities, with an explicit runtime warning since p* falls inside the miscalibrated range.
- **Result:** mean expected cost reduced 48.6% versus the naive p=0.5 cutoff (0.4317 vs. 0.8405 per step), trading a large increase in false positives (1,984→9,209) for a large decrease in false negatives (3,680→1,011) — the correct direction under the derived ~9.3x cost asymmetry. **The empirical cost-minimizing threshold, swept independently across all possible thresholds, landed close to the theoretically derived p* despite the documented miscalibration** — a real, if imperfect, validation that the derivation held up under real data. Full result: `notes/phase3_bayes_threshold_result.md`.
- **Explicitly not a backtest or profitability claim** — a decision-theoretic result under a stated cost model, consistent with the project's non-goals throughout.

### Phase 4 (weeks 6–8, stretch — pick one)

- **Latency/accuracy Pareto frontier**, applying model-compression and CUDA background to quantify the accuracy cost of low-latency inference.
- **Cross-asset extension via a GNN**, propagating regime information across a small correlated basket.

---

## 7. Explicit non-goals

- No claim of discovered alpha or a profitable trading strategy.
- No full portfolio construction. Deliberately out of scope — stated explicitly rather than attempted shallowly.
- No claim that Roll-implied spread equals the exchange's displayed quoted spread.
- No claim that purged walk-forward eliminates leakage completely — the residual +0.043 effect (std 0.055, 14/20 shifts positive) is reported as a real, unresolved, mostly-but-not-entirely-eliminated finding.
- No claim that the jump-test's day-count calibration is fully resolved (9-vs-1 overshoot, Phase 1).
- No claim that raw probabilities are well-calibrated — five recalibration attempts all failed, and the known miscalibrated range is stated as an explicit limitation on the Phase 3 Bayes threshold result rather than hidden.
- No claim that the Phase 3 cost-reduction figure (+48.6%) represents realized profitability — it is a decision-theoretic result under an explicit, derived cost model, not a backtest.

---

## 8. Checklist audit (against current industry guidance)

| Area | Status |
|---|---|
| Non-stationarity, walk-forward validation, structural breaks, multiple-testing bias | Core content of Phase 2, now with an empirically demonstrated leakage result, not just a described methodology |
| Microstructure (spreads, noise, order flow) | Substantial — Roll estimator, signature-plot failure diagnosis, staleness/sweep robustness checks, per-regime R decomposition |
| Realistic backtesting (costs, leakage, overlapping labels) | Leakage: empirically demonstrated and quantified in Phase 2. Costs: Phase 3's Bayes-risk threshold, derived from Phase 1's spread numbers, +48.6% cost reduction vs. naive. No full backtest attempted — flagged as a deliberate boundary. |
| Implementation | Python/NumPy/pandas, PyTorch, JAX, scikit-learn |
| Portfolio construction and risk | Explicitly out of scope (Section 7), with reasoning stated |
| Finance fundamentals | Covered at the depth needed for spot crypto microstructure |
| Baseline before complex model | Built and honestly compared (Phase 1); the feature-blind dummy classifier in Phase 2 serves the same role again, at a deeper level |
| Document failed hypotheses | Already the operating discipline — see Section 9, now including two failed null constructions |

---

## 9. Derivation log (summary — full detail in `notes/`)

- **Calm-window selection, R derivation, Q derivation, baseline filter evaluation, HMM fit, K=4 recheck, naive baseline, bipower split:** see Phase 1 entries, summarized in prior sections and fully detailed in `notes/`.
- **Label horizon H:** derived via absorbing-Markov-chain expected sojourn time in the IMM's elevated state set, weighted by the quasi-stationary distribution — not a naive average of individual states' self-transition-implied dwell times, which would have ignored the fact that one state is only reachable/exitable through the other.
- **Null label, attempt 1 (block permutation):** rejected — real-data AUCs ~15-17 std below chance in both CV schemes, traced to a base-rate mismatch from relocating rare positive blocks onto a mostly-low-feature timeline.
- **Null label, attempt 2 (single circular shift):** rejected — real-data AUCs in opposite directions (0.285 vs. 0.593) on a single draw, traced to accidental realignment with the dataset's known ~8h periodicity (a direct callback to the Phase 1 diurnal check).
- **Null label, final method (many shifts, period-excluded):** adopted, reporting a distribution across 20 shifts rather than trusting any single draw.
- **The label-autocorrelation CV-structural confound:** discovered because the corrected null's results still didn't match the leakage hypothesis (purged showing consistent bias, naive showing none) — diagnosed via a feature-blind dummy-classifier control rather than accepted at face value, revealing a genuine, mathematically explicable bias in each CV scheme's mechanics, orthogonal to any feature relationship.
- **Final, confound-corrected leakage result:** real-model-AUC-minus-dummy-AUC per shift. Naive k-fold: 20/20 shifts positive (p≈10⁻⁶). Purged walk-forward: 14/20 shifts positive (p≈0.04), mean smaller than its own std — reported as "mostly eliminated," not "proven zero."
- **Real-label calibration (closing the loop):** the same real-vs-dummy correction was applied to the real label rather than trusting its raw ~0.92 AUC. Calibrated against the null-shift diff distributions as an empirical reference, the real label's diff sits ≈3.7σ (naive) and ≈6.1σ (purged) above the artifact-only baseline — confirmed as genuine signal, with a clean decomposition showing naive's inflated result ≈ real signal + naive's own characteristic leakage bias (both independently measured on the null shifts).
- **Reliability check:** found systematic underconfidence in the purged-walk-forward `real_label` predictions across ~0.05–0.5 predicted probability, up to 8σ off per bin — real and consistent, invisible on the plot alone, only visible via the per-bin standard error.
- **Recalibration, five attempts, all rejected:** static time split (made things worse, opposite-direction bias); static regime split (worse in both regimes, refuting the regime hypothesis); walk-forward isotonic, expanding and rolling window (neither improved real-data results); walk-forward Platt scaling (worked on synthetic drift data, did not clearly work on real data). Decision to proceed with raw probabilities made explicitly, after exhausting both plausible fix axes (split scheme, calibrator flexibility), rather than continuing to search indefinitely.
- **Bayes-risk threshold:** derived from the Roll spread ratio (crisis/calm ≈ 9.31) rather than asserting a cost matrix, giving p*≈0.097. Validated post hoc against an independently swept empirical cost curve — the theoretical and empirical optima landed close together despite the known miscalibration in that region.

---

## 10. Deliverables

- Public GitHub repository (`Track_to_Trade`), MIT- or similarly-licensed, fully reproducible from the download scripts.
- `notes/`: a derivation log for every threshold in the project, including rejected attempts — now including two rejected null-label constructions and five rejected recalibration attempts, all diagnosed rather than silently discarded.
- `ISSUES.md`: tracked open items not currently blocking progress, reviewed before considering the project done.
- `plots/`: all generated figures, referenced from `notes/`.
- `src/kalman.py`, `src/imm.py`, `src/cv_splits.py`: implementations, unit/synthetic-validated before use on real data in every case.
- A technical write-up in the style of an internal quant research note.

## 11. Timeline

| Weeks | Milestone | Status |
|---|---|---|
| 1–2 | Phase 1: filter + all derived parameters + baseline model + jump/diffusion split | **Fully complete** |
| 2–4 | Phase 2: artifact stress test, walk-forward validation | **Complete: leakage mechanism demonstrated, real-label signal validated, and both replicated on an independent event window (FTX)** |
| 4–6 | Phase 3: calibration, decision threshold, cost-aware check | **Complete: five recalibration attempts honestly documented and rejected; Bayes-risk threshold derived and validated, +48.6% cost reduction vs. naive** |
| 6–8 | Phase 4 (stretch): latency/compression Pareto frontier or GNN cross-asset extension | Not started |

## 12. What a 45-minute interview conversation looks like

Anticipated probes and where the project answers them:
- *"How do you know your regime detector isn't just overfitting the backtest?"* → the Phase 2 leakage stress test doesn't just describe the fix, it demonstrates the failure and the fix empirically, with a p≈10⁻⁶ result.
- *"Why should I trust this threshold?"* → every threshold traces to a derivation in `notes/`, walkable on a whiteboard — including the Bayes-risk threshold, derived from Phase 1's own spread numbers rather than asserted.
- *"Did you find alpha?"* → no, and the reasoning for why that claim would be dishonest at this scope is itself part of the answer.
- *"Walk me through a time your first approach was wrong."* → several candidates now, including two consecutive null-label construction failures in Phase 2, and five consecutive recalibration failures in Phase 3.
- *"Walk me through a model-selection decision you second-guessed."* → the K=5 → K=4 reversal, or the two null-construction rejections in Phase 2.
- *"Tell me about a result that surprised you and how you resolved it."* → the label-autocorrelation CV-structural confound — purged walk-forward showing a *consistent* bias where naive k-fold showed none, the opposite of the hypothesis, resolved via a feature-blind dummy-classifier control rather than force-fit into the expected story.
- *"How do you know your added complexity is worth it?"* → the naive-baseline comparison (Phase 1) and, at a deeper level, the dummy-classifier control in Phase 2.
- *"What's a result you have to report with a caveat?"* → the purged walk-forward's residual leakage (+0.043, std 0.055), or the Phase 3 Bayes threshold applied to known-imperfect raw probabilities — both reported honestly rather than rounded up.
- *"When did you decide to stop trying to fix something?"* → the calibration saga — five attempts across two plausible fix axes, then a deliberate decision to proceed with a documented limitation rather than keep searching for a variant that happens to work on one month of data.
- *"How do you know your central result generalizes, not just fits one crisis?"* → the FTX cross-window replication — same leakage signature, same order of magnitude, on a structurally different event, refitting every parameter fresh rather than reusing LUNA's.
