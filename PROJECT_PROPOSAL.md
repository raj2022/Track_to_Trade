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

This is a legitimate, citable technique, not an invented analogy — Interacting Multiple Model Kalman filter banks are an established approach to regime-switching volatility forecasting in the finance literature. **Now implemented, tested, and revised (Section 6) — the IMM is built, validated on synthetic data, run on the actual 2022-05 LUNA window, and its regime count was itself revisited via a genuine model-selection reversal (K=5 → K=4) once a behavioral problem was found in what BIC alone selected.**

The project also carries over a specific failure mode discovered independently, in a different domain: a conditional-generalization failure in HEP analysis, where a classifier exploited artifacts of label construction rather than genuine conditional signal, fixed via matched hypothesis sampling and group-aware validation. The direct financial analogue — classifiers "detecting" regimes by exploiting look-ahead/overlap structure in label construction rather than real market dynamics — is the motivation for purged, embargoed cross-validation (López de Prado, *Advances in Financial Machine Learning*, 2018). Arriving at the same fix independently, from physics, is the project's genuine differentiator: most candidates who know purged CV learned it from the finance literature; this project re-derives the same principle from a structurally identical problem encountered first in particle tracking.

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

**Two additions worth stating explicitly after Phase 1.** First, the discipline applies to reporting results, not just choosing parameters — the IMM's persistence result decays across its own event window, and that decay is reported alongside the favorable aggregate rather than hidden behind it. Second, it applies to *model-selection criteria themselves*: BIC formally preferred K=5 for the regime count, but that selection was not treated as final once a real behavioral flaw (a flickering, non-sticky state pair) was found in the model it selected — a K=4 alternative was tested directly against the actual downstream result rather than deferring to the metric.

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

**Working window so far:** all Phase 1 derivation and filtering work has been done on 2022-05 (LUNA) and 2023-06 (calm baseline). The other three event windows (COVID, May 2021, FTX) are downloaded but not yet used — a natural robustness check for Phase 2, to confirm the IMM result isn't LUNA-specific.

---

## 6. Project phases

### Phase 1 (weeks 1–2): Filter, with every parameter derived — **fully complete**

- **Baseline (single-hypothesis) local-level Kalman filter: built, derived, evaluated, and found to fail in a specific, quantified way.**
  - Q: rolling realized-variance-rate series, dt=60s (validated via a trade-arrival staleness check specific to 2022-05, not assumed from the calm-window value), 1-day trailing window (selected as the shortest window that is stable pre-event while still tracking the LUNA transition; shorter windows were noise-dominated even pre-event, a 3-day window over-smoothed and delayed the visible peak by nearly a week).
  - R: 3.661×10⁻¹¹, Roll's (1984) estimator, all-tick, on the verified calm window (2023-06-13 → 2023-06-19). See Section 9 for the full derivation chain (rejected signature-plot attempts, arithmetic correction, same-timestamp robustness check).
  - Chi-square detection gate: derived from the innovation covariance at an explicit target of 1 false alarm/day (α = 6.94×10⁻⁴, threshold = 11.504) — not a stated significance level.
  - **Result: the baseline is miscalibrated, and not in a way specific to the LUNA event.** Flagged 1.176% of steps against a 0.069% target (~17x over), with pre/during/post-event flag rates of 1.945%/1.476%/0.789% — elevated in *all three* periods, ruling out an event-localized explanation. Two hypotheses were tested to explain the counterintuitive pre > during ordering: warm-up contamination (ruled out) and "pre-event was assumed calm without verification" (confirmed — early May 2022 already ran 2–3x above the verified June baseline). A third finding explained why "during" was lower than the crash's severity would suggest: the trailing 1-day Q re-absorbs a sustained spike into its own baseline within about a day, muting further surprise exactly when persistence would be most valuable. This became the concrete, derivation-backed motivation for the IMM, not just "more sophisticated is better." Full detail: `notes/baseline_filter_evaluation.md`.

- **IMM (Interacting Multiple Model) filter: built, tested, and revised via a genuine model-selection reversal.**
  - Number of regimes (K): selected via BIC over a Gaussian HMM fit (EM/Baum-Welch) to 2022-05 returns, K=2..5 tested. K=5 initially selected, but a real EM numerical-instability bug was caught and fixed first (standardizing returns before fitting, confirmed via synthetic recovery), after which K=5's transition matrix revealed a genuine flickering artifact: two low-variance states with only a 0.58 mutual self-transition probability (implied dwell time ~2.4 minutes), rather than the sticky behavior expected of real regimes.
  - **K=4 recheck (`scripts/k4_recheck.py`), performed rather than accepting K=5's BIC win uncritically:** refit at K=4, compared directly against the K=5 reference on the actual downstream result. Persistence preserved and marginally improved (0.165/0.842/0.447 vs. K=5's 0.159/0.830/0.424), and the calmest K=4 state's self-transition came out properly sticky (0.9745, ~40-minute dwell time). K=4's four states mapped cleanly onto four of K=5's five by variance, with the single calmest K=4 state absorbing the flickering pair — structural, not just summary-statistic, confirmation that the 5th state was an unnecessary split. **K=4 adopted as the model feeding Phase 2**, despite K=5's formally better BIC (91589.59 vs. 92197.93) — BIC alone was not treated as the final word once a real behavioral problem was found in what it selected.
  - Per-regime Q and R: derived, not asserted, via Roll's estimator restricted to tick pairs whose bins share each state's label (`scripts/per_state_R.py`), with Q_k = HMM_variance_k − 2R_k. One state's raw R came out slightly negative (statistically indistinguishable from zero) and was clipped to 0.
  - **Unplanned cross-validation:** the independently-derived per-state R for the calmest K=5 state (3.620×10⁻¹¹) landed almost exactly on the original calm-window R (3.661×10⁻¹¹) derived weeks earlier from unrelated data via a completely different method.
  - Whether R needs to be regime-conditional at all was tested, not assumed: Roll's estimator on the 05-09–05-12 crisis sub-window gave R 86.7x the calm value — large enough that shared-R-across-hypotheses was rejected.

- **Naive baseline built and honestly compared (`scripts/naive_baseline.py`).** A near-parameter-free rule (1h realized variance above the expanding median of all prior values) was evaluated against the same persistence metric. Raw magnitudes alone would be misleading (a median-crossing rule is mechanically centered near 50% regardless of signal); the honest comparison is the during/pre **ratio** — IMM: 5.2x (early), 2.7x (late); naive: 1.8x (early), 1.1x (late, barely above its own baseline). The naive rule also has no stable pre-event resting state in the plot, oscillating between 0 and 1 throughout. **Conclusion: the IMM's complexity is justified, but by the shape of the signal (a genuine low-noise baseline state to deviate from) rather than by raw magnitude** — a materially different and more defensible claim than "the IMM scores higher."

- **Bipower-variation jump/diffusion decomposition built (`scripts/bipower_jump_diffusion.py`), yielding a genuine substantive finding, not just a completed checklist item.** None of the four LUNA days were flagged as statistically significant jumps (z-stats -0.88 to 1.19, jump shares 0.0–4.8%, against a derived threshold of z=1.849); RV and BV track almost exactly through the event. The month's largest jump signals occurred on unrelated, comparatively ordinary days. **This empirically confirms the LUNA collapse manifested as a sustained regime switch to elevated diffusion, not a discrete jump** — direct evidence that the IMM's framing (competing diffusion-rate hypotheses) was the conceptually correct tool for this event, distinct from what a jump-detection approach would have found. Caveat, not resolved: 9 of 31 days were flagged against a 1/month calibration target, plausibly because the jump test's implicit noise-homogeneity assumption doesn't fully account for the regime-dependent R already established — noted as an open calibration question, though it doesn't undermine the LUNA-specific null finding.

### Phase 2 (weeks 2–4): Artifact stress test — the centerpiece

- Construct a regime/event classifier on top of the filter output (using the K=4 IMM as the feature source).
- Build two label sets: one with overlapping/look-ahead-contaminated construction (structurally identical to the CERN failure mode — labels carrying no intrinsic conditional signal beyond their construction artifacts), and one cleaned via matched sampling.
- Formalize with a permutation-derived null: shuffle labels, preserve artifact structure, measure the "predictive power" a model obtains from structure alone. Report model performance relative to this derived null, not as a stated accuracy figure.
- Apply purged, embargoed cross-validation to remove the leakage.
- **Revision:** evaluation protocol changed to explicit **walk-forward** validation (expanding or rolling window, always evaluated forward in time) rather than k-fold-with-purging alone — a better fit for a project whose thesis is regime non-stationarity, since it never allows any fold structure to leak a future regime into training, even indirectly.

### Phase 3 (weeks 4–6): Calibration and decision theory

- Convert the filter's regime posterior into a genuinely calibrated probability: reliability diagrams, proper scoring rules (log score, CRPS), evaluated walk-forward.
- Derive the decision threshold from an explicit asymmetric cost matrix (Bayes risk) rather than defaulting to 0.5 — directly targeting the project's stated calibration weakness.
- **Cost-aware reality check.** Take the Roll-implied effective spread already derived in Phase 1 (0.121bp calm / 1.127bp crisis) plus Binance's published taker fee, and ask whether a signal at the derived Bayes-optimal threshold would clear round-trip costs. This is explicitly *not* a backtest or a profitability claim — it's a single, honest question: is the signal even in the right order of magnitude to matter once realistic frictions are included, using cost inputs the project already derived rather than assumed.

### Phase 4 (weeks 6–8, stretch — pick one)

- **Latency/accuracy Pareto frontier**, applying model-compression and CUDA background to quantify the accuracy cost of low-latency inference — directly relevant to market-making firms specifically, and a differentiator few other candidates will bring.
- **Cross-asset extension via a GNN**, propagating regime information across a small correlated basket, testing whether cross-asset signals survive the same artifact stress test as the single-asset case.

---

## 7. Explicit non-goals

- No claim of discovered alpha or a profitable trading strategy. Public data over a 6–8 week project is not evidence of exploitable edge, and claiming otherwise would signal a misunderstanding of market efficiency and multiple-testing risk, not a result.
- No full portfolio construction (covariance estimation, factor risk, exposure neutralization). Deliberately out of scope — the project's contribution sits upstream of portfolio construction, in signal identifiability. Stated explicitly rather than attempted shallowly.
- No claim that Roll-implied spread equals the exchange's displayed quoted spread — it's an implied *effective* spread from aggregate trade prices, and the write-up says so.
- No claim that the jump-test's day-count calibration is fully resolved — the 9-vs-1 overshoot is reported as an open question, not smoothed over.

---

## 8. Checklist audit (against current industry guidance)

A practicing quant researcher's advice was checked against the existing plan rather than treated as a new plan:

| Area | Status |
|---|---|
| Non-stationarity, walk-forward validation, structural breaks, multiple-testing bias | Core content of Phase 2; evaluation scheme revised to explicit walk-forward |
| Microstructure (spreads, noise, order flow) | Substantial — Roll estimator, signature-plot failure diagnosis, staleness/sweep robustness checks, per-regime R decomposition |
| Realistic backtesting (costs, leakage, overlapping labels) | Leakage: core to Phase 2. Costs: added in Phase 3 via the cost-aware reality check, now with both calm and crisis spread estimates. No full backtest attempted — flagged as a deliberate boundary, not a gap. |
| Implementation | Python/NumPy/pandas, PyTorch, JAX — matches existing background |
| Portfolio construction and risk | Explicitly out of scope (Section 7), with reasoning stated |
| Finance fundamentals | Covered at the depth needed for spot crypto microstructure; not pursued further, consistent with targeting research roles rather than derivatives/pricing roles |
| Baseline before complex model | Built and honestly compared — the IMM's complexity is justified by signal shape, not raw magnitude (Section 6) |
| Document failed hypotheses | Already the operating discipline — see Section 9 |

---

## 9. Derivation log (summary — full detail in `notes/`)

- **Calm-window selection:** first attempt (2022-04) rejected — outlier trimming excluded 0 of 30 days because April is a gradual pre-crash ramp, not two separable calm/spike populations; lowering the trim threshold to force a split was considered and rejected as p-hacking the outlier filter. Fell back to 2023-06, which trimmed cleanly and yielded the verified calm window 2023-06-13 → 2023-06-19.
- **Signature plot (R, attempts 1–2):** rejected — pooling a calm period and the LUNA crash conflated regime-invariant noise with regime-dependent variance, and a stale-price artifact (44.7% of 1s bins stale) invalidated short-interval reads even on the verified calm window.
- **R (final method):** Roll's estimator on tick-level returns. Arithmetic sanity-check error caught and corrected independently. Same-timestamp order-book-sweep contamination quantified (+11.9% shift) and resolved via an explicit "what does the filter ingest" argument.
- **Q (rolling, 2022-05):** dt re-validated for this specific month; window length chosen by bias/variance logic; a hypothesized ~24h diurnal explanation for residual noise was tested and *ruled out* by ACF, replaced by an unexplained ~8h periodicity (tentatively linked to funding settlement, unconfirmed).
- **Baseline filter evaluation:** miscalibration found in all three periods, ruling out an event-specific cause; two competing explanations for a counterintuitive result ordering tested directly (one ruled out, one confirmed); trailing-Q self-adaptation identified as the mechanism muting persistence during the crash itself.
- **HMM regime fit:** K initially selected by BIC (K=5). A real EM numerical-instability bug caught via the library's own warnings, diagnosed and fixed by standardizing before fitting.
- **K=4 recheck:** BIC's K=5 selection was not treated as final once a behavioral flaw (flickering, non-sticky state pair) was found. K=4 tested directly against the actual downstream persistence result, found comparable-to-better, with a properly sticky calmest state and a clean structural mapping to K=5's other states. K=4 adopted despite the formally worse BIC.
- **Per-regime R:** derived separately for each state via tick-pair restriction. One state's R clipped from a small negative value to 0. Unplanned cross-validation against the original calm-window R fell out of this work.
- **IMM result:** validated on synthetic data before real use. On real data, a genuine decay (0.830 → 0.424 under K=5) was found and reported alongside the more favorable aggregate, not in place of it.
- **Naive baseline:** raw magnitude comparison would have been misleading (median-crossing rules are mechanically centered near 50%); the during/pre ratio was used instead, and the naive rule's lack of a stable pre-event state was identified as the real reason the IMM's complexity is justified.
- **Bipower jump/diffusion split:** found LUNA registered as sustained diffusion, not a statistical jump — a genuine, unanticipated finding, not just a completed derivation. A real 9-vs-1 calibration overshoot was flagged as unresolved rather than glossed over.

---

## 10. Deliverables

- Public GitHub repository (`Track_to_Trade`), MIT- or similarly-licensed, fully reproducible from the download scripts.
- `notes/`: a derivation log for every threshold in the project, in the format demonstrated above — question, method, rejected attempts, result, reproduction command.
- `plots/`: all generated figures, referenced from `notes/`.
- `src/kalman.py`, `src/imm.py`: the filter implementations, unit-validated against synthetic data with known ground truth before being pointed at real data, in both cases.
- A technical write-up in the style of an internal quant research note: the core question, the artifact-robustness result, and an honest identifiability boundary — not a paper, not a pitch deck.

## 11. Timeline

| Weeks | Milestone | Status |
|---|---|---|
| 1–2 | Phase 1: filter + all derived parameters + baseline model + jump/diffusion split | **Fully complete** |
| 2–4 | Phase 2: artifact stress test, walk-forward validation | Not started |
| 4–6 | Phase 3: calibration, decision threshold, cost-aware check | Not started |
| 6–8 | Phase 4 (stretch): latency/compression Pareto frontier or GNN cross-asset extension | Not started |

## 12. What a 45-minute interview conversation looks like

Anticipated probes and where the project answers them:
- *"How do you know your regime detector isn't just overfitting the backtest?"* → Phase 2 is built specifically to answer this.
- *"Why should I trust this threshold?"* → every threshold traces to a derivation in `notes/`, walkable on a whiteboard.
- *"Did you find alpha?"* → no, and the reasoning for why that claim would be dishonest at this scope is itself part of the answer.
- *"Why does a particle physicist think this transfers?"* → the chi-square gate, concrete and mechanistic rather than a loose analogy.
- *"Walk me through a time your first approach was wrong."* → the April calm-window rejection, the inverted May signature plot, or the baseline filter's miscalibration.
- *"Walk me through a model-selection decision you second-guessed."* → the K=5 → K=4 reversal — BIC formally preferred K=5, but a behavioral check (a non-sticky, flickering state pair) overrode it, and the reversal was validated on the actual downstream result rather than asserted.
- *"How do you know your added complexity is worth it?"* → the naive-baseline comparison, and specifically the honest observation that raw magnitude comparison would have been the wrong metric — the real justification was the shape of the signal, not the size.
- *"Tell me about a result that surprised you."* → the bipower decomposition showing LUNA was diffusion, not a jump — an unanticipated, genuinely informative finding that also validated the project's core methodological choice.
- *"How did you catch a bug in your own model-fitting code?"* → the EM non-convergence warnings, diagnosed to input-scale precision loss and fixed via standardization, confirmed on synthetic data before trusting the real fit again.
