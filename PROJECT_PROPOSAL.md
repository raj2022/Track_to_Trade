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

This is a legitimate, citable technique, not an invented analogy — Interacting Multiple Model Kalman filter banks are an established approach to regime-switching volatility forecasting in the finance literature.

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

---

## 5. Data

- **Primary:** Binance public spot market data (`data.binance.vision`, mirrored at `github.com/binance/binance-public-data`), BTCUSDT, `aggTrades`. Free, no API key, redistributable, tick-resolution, extending back to 2017.
- **Cross-check:** LOBSTER free academic sample files (NASDAQ, full order-book depth) — reserved for validating the measurement-noise approach against a real limit order book, not just trade prints.

### Pulled windows (BTCUSDT spot, aggTrades)

Five contiguous 3-month windows, chosen to capture actual regime *transitions* rather than sit entirely inside one labeled regime:

- 2020-02 → 2020-04 (COVID crash)
- 2021-04 → 2021-06 (May 2021 crash)
- 2022-04 → 2022-06 (LUNA/UST collapse)
- 2022-10 → 2022-12 (FTX collapse)
- 2023-06 → 2023-08 (calm baseline)

All 15 monthly files downloaded and checksum-verified. Row counts confirmed the expected within-window pattern (event month > its calm neighbors) in every window; cross-window row-count comparisons were explicitly rejected as a volatility proxy, since they're confounded with secular growth in exchange activity over 2020–2023.

---

## 6. Project phases

### Phase 1 (weeks 1–2): Filter, with every parameter derived

- Linear/extended Kalman filter for latent fair-price and volatility state, extended to an Interacting Multiple Model filter over competing regime hypotheses.
- **Process noise (Q):** derived from realized quadratic variation, computed per-regime rather than as a single pooled value (see Section 7 revision) — a single Q derived across a period spanning both calm and crisis data would conflate regime-invariant and regime-dependent quantities, the same class of error identified and corrected in the R derivation below.
- **Measurement noise (R):** derived via Roll's (1984) implied-spread estimator on tick-level returns, rather than a fixed-interval signature plot (the latter was attempted first and rejected — see Section 9). **Status: derived.** R = 3.661×10⁻¹¹ (log-return² units) on the verified calm window 2023-06-13 → 2023-06-19, corresponding to a ~0.121bp implied effective spread (~$0.03 on ~$25–26k BTC) — plausible for the most liquid pair in crypto during a calm period. Stress-tested against same-timestamp order-book-sweep contamination (33.6% of pairs affected; excluding them shifts R by +11.9%, to 4.098×10⁻¹¹); the all-tick estimate was retained as the internally consistent choice given the filter ingests aggTrades rows as-is.
- **Regime transition probabilities:** EM/Baum-Welch maximum likelihood fit, not hand-picked.
- **Detection/gating threshold:** derived from the innovation covariance via a chi-square test at an explicit target false-alarm rate — the track-hit association gate, transplanted directly.
- **Jump/diffusion decomposition:** bipower variation, to separate continuous volatility from discrete jumps rather than using one blunt threshold for both.
- **New: a naive baseline**, evaluated under the same protocol as the full model (e.g. "regime change ⟺ realized vol crosses its own trailing median"), so later results can answer "did the added complexity earn its keep" rather than reporting model performance in a vacuum.

### Phase 2 (weeks 2–4): Artifact stress test — the centerpiece

- Construct a regime/event classifier on top of the filter output.
- Build two label sets: one with overlapping/look-ahead-contaminated construction (structurally identical to the CERN failure mode — labels carrying no intrinsic conditional signal beyond their construction artifacts), and one cleaned via matched sampling.
- Formalize with a permutation-derived null: shuffle labels, preserve artifact structure, measure the "predictive power" a model obtains from structure alone. Report model performance relative to this derived null, not as a stated accuracy figure.
- Apply purged, embargoed cross-validation to remove the leakage.
- **Revision:** evaluation protocol changed to explicit **walk-forward** validation (expanding or rolling window, always evaluated forward in time) rather than k-fold-with-purging alone — a better fit for a project whose thesis is regime non-stationarity, since it never allows any fold structure to leak a future regime into training, even indirectly.

### Phase 3 (weeks 4–6): Calibration and decision theory

- Convert the filter's regime posterior into a genuinely calibrated probability: reliability diagrams, proper scoring rules (log score, CRPS), evaluated walk-forward.
- Derive the decision threshold from an explicit asymmetric cost matrix (Bayes risk) rather than defaulting to 0.5 — directly targeting the project's stated calibration weakness.
- **New: cost-aware reality check.** Take the Roll-implied effective spread already derived in Phase 1 (0.121bp) plus Binance's published taker fee, and ask whether a signal at the derived Bayes-optimal threshold would clear round-trip costs. This is explicitly *not* a backtest or a profitability claim — it's a single, honest question: is the signal even in the right order of magnitude to matter once realistic frictions are included, using cost inputs the project already derived rather than assumed.

### Phase 4 (weeks 6–8, stretch — pick one)

- **Latency/accuracy Pareto frontier**, applying model-compression and CUDA background to quantify the accuracy cost of low-latency inference — directly relevant to market-making firms specifically, and a differentiator few other candidates will bring.
- **Cross-asset extension via a GNN**, propagating regime information across a small correlated basket, testing whether cross-asset signals survive the same artifact stress test as the single-asset case.

---

## 7. Explicit non-goals

- No claim of discovered alpha or a profitable trading strategy. Public data over a 6–8 week project is not evidence of exploitable edge, and claiming otherwise would signal a misunderstanding of market efficiency and multiple-testing risk, not a result.
- No full portfolio construction (covariance estimation, factor risk, exposure neutralization). Deliberately out of scope — the project's contribution sits upstream of portfolio construction, in signal identifiability. Stated explicitly rather than attempted shallowly.
- No claim that Roll-implied spread equals the exchange's displayed quoted spread — it's an implied *effective* spread from aggregate trade prices, and the write-up says so.

---

## 8. Checklist audit (against current industry guidance)

A practicing quant researcher's advice was checked against the existing plan rather than treated as a new plan:

| Area | Status |
|---|---|
| Non-stationarity, walk-forward validation, structural breaks, multiple-testing bias | Core content of Phase 2; evaluation scheme revised to explicit walk-forward |
| Microstructure (spreads, noise, order flow) | Substantial — Roll estimator, signature-plot failure diagnosis, staleness/sweep robustness checks |
| Realistic backtesting (costs, leakage, overlapping labels) | Leakage: core to Phase 2. Costs: added in Phase 3 via the cost-aware reality check. No full backtest attempted — flagged as a deliberate boundary, not a gap. |
| Implementation | Python/NumPy/pandas, PyTorch, JAX — matches existing background |
| Portfolio construction and risk | Explicitly out of scope (Section 7), with reasoning stated |
| Finance fundamentals | Covered at the depth needed for spot crypto microstructure; not pursued further, consistent with targeting research roles rather than derivatives/pricing roles |
| Baseline before complex model | Added to Phase 1 |
| Document failed hypotheses | Already the operating discipline — see Section 9 |

---

## 9. Derivation log (summary — full detail in `notes/`)

- **Calm-window selection:** first attempt (2022-04) rejected — outlier trimming excluded 0 of 30 days because April is a gradual pre-crash ramp, not two separable calm/spike populations; lowering the trim threshold to force a split was considered and rejected as p-hacking the outlier filter. Fell back to 2023-06, which trimmed cleanly (3 of 30 days excluded, agreeing with visual inspection) and yielded the verified calm window 2023-06-13 → 2023-06-19.
- **Signature plot (R, attempt 1):** single-offset, whole-month version on 2022-05 produced a jagged, non-monotonic plot — diagnosed as pooling a calm period and the LUNA crash together, conflating regime-invariant noise with regime-dependent true variance.
- **Signature plot (R, attempt 2):** multi-offset-averaged version on the verified calm window showed RV *rising* then decaying — inverted from the textbook shape. Diagnosed via a trade-arrival-gap check: 44.7% of 1-second bins were stale (no new trade), confirming a stale-price artifact rather than a clean noise signal at short intervals.
- **R (final method):** Roll's estimator on tick-level returns, sidestepping interval choice entirely. Arithmetic sanity-check error caught and corrected independently (0.121bp, not 1.2bp). Same-timestamp order-book-sweep contamination identified, quantified (33.6% of pairs, +11.9% shift in R), and resolved via an explicit "what does the filter ingest" argument rather than a preference between the two resulting estimates.

---

## 10. Deliverables

- Public GitHub repository (`Track_to_Trade`), MIT- or similarly-licensed, fully reproducible from the download scripts.
- `notes/`: a derivation log for every threshold in the project, in the format demonstrated above — question, method, rejected attempts, result, reproduction command.
- A technical write-up in the style of an internal quant research note: the core question, the artifact-robustness result, and an honest identifiability boundary — not a paper, not a pitch deck.

## 11. Timeline

| Weeks | Milestone |
|---|---|
| 1–2 | Phase 1: filter + all derived parameters + baseline model |
| 2–4 | Phase 2: artifact stress test, walk-forward validation |
| 4–6 | Phase 3: calibration, decision threshold, cost-aware check |
| 6–8 | Phase 4 (stretch): latency/compression Pareto frontier or GNN cross-asset extension |

## 12. What a 45-minute interview conversation looks like

Anticipated probes and where the project answers them:
- *"How do you know your regime detector isn't just overfitting the backtest?"* → Phase 2 is built specifically to answer this.
- *"Why should I trust this threshold?"* → every threshold traces to a derivation in `notes/`, walkable on a whiteboard.
- *"Did you find alpha?"* → no, and the reasoning for why that claim would be dishonest at this scope is itself part of the answer.
- *"Why does a particle physicist think this transfers?"* → the chi-square gate, concrete and mechanistic rather than a loose analogy.
- *"Walk me through a time your first approach was wrong."* → the April calm-window rejection, or the inverted May signature plot — both real, both resolved through diagnosis rather than parameter tuning.
