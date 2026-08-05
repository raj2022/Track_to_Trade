# K=4 recheck, naive baseline, and bipower jump/diffusion — Phase 1 close-out

**Date:** Day 6/7
**Questions closed here:**
1. Does K=4 resolve the state 0/3 flickering without losing the IMM's persistence result?
2. Does the IMM's complexity earn its keep against the simplest defensible baseline?
3. Does the LUNA event register as discrete jumps, sustained diffusion, or both?

---

## 1. K=4 recheck: adopted

Refit at K=4 (`scripts/k4_recheck.py`), compared directly against the K=5 reference.

| | pre | during-early (05-09→13) | during-late (05-13→19) |
|---|---|---|---|
| K=5 (reference) | 0.159 | 0.830 | 0.424 |
| K=4 (this run) | 0.165 | 0.842 | 0.447 |

Persistence result is preserved, marginally *better* under K=4. The calmest K=4 state's self-transition is **0.9745** (properly sticky — implied mean dwell time ~40 minutes), against K=5's problematic pair at 0.5812/0.8201 (implied dwell time ~2.4 minutes for the worse of the two).

**Structural confirmation, not just summary-stat agreement:** K=4's four states map cleanly onto four of K=5's five (K4 state 0 ↔ K5 state 4, K4 state 3 ↔ K5 state 1, K4 state 2 ↔ K5 state 2 by variance), with K4's single calmest state absorbing K5's flickering 0/3 pair. This is real evidence the 5th state was an unnecessary split, not a coincidence in the aggregate numbers.

The K=4 plot still looks visually dense despite the fix — worth noting explicitly so this isn't mistaken for residual chattering on a future re-read: with self-transitions of 0.96–0.97 across all four states (dwell times ~25–40 min), a month of same-order-of-magnitude blocks will look busy at full-month zoom regardless of how coherent the underlying dynamics are. Trust the transition matrix over the plot's visual density here.

**Decision: K=4 is adopted as the model that feeds Phase 2.** BIC formally preferred K=5 (91589.59 vs. 92197.93), but the persistence result, the dwell-time sanity check, and the structural state-mapping all point the same direction — simpler, not chattery, and no meaningful loss on the one downstream result available to judge it by.

---

## 2. Naive baseline: the IMM's complexity is justified, but not for the reason the raw numbers suggest

Naive rule (1h RV > expanding median of all prior 1h RV) vs. IMM (`scripts/naive_baseline.py`):

| | pre | during-early | during-late |
|---|---|---|---|
| IMM | 0.159 | 0.830 | 0.424 |
| Naive | 0.548 | 1.000 | 0.618 |

Reading raw magnitudes alone would be misleading — a naive median-crossing rule is mechanically centered near 50% regardless of any real signal, so its pre-event 0.548 is not itself meaningful. **The correct comparison is the during/pre ratio:**

- IMM: 5.2x (early), 2.7x (late)
- Naive: 1.8x (early), 1.1x (late — barely distinguishable from its own baseline)

The naive rule's contrast is materially weaker on both counts. More importantly, the naive rule's plot shows no stable resting state at all pre-event — it oscillates between 0 and 1 well before 05-09, because "above the expanding median" is close to a coin flip on any moderately choppy day, structurally incapable of representing "calm" as a state. The IMM has a genuine low-probability, comparatively stable pre-event baseline to deviate from.

**Conclusion: the added complexity earns its keep, but the honest justification is the *shape* of the signal (a real distinguishable baseline state vs. none), not simply "bigger numbers."** This distinction is worth preserving exactly as-is in any write-up — reporting only "IMM: 0.830 vs. naive: 1.000, IMM still wins" would be a worse and less defensible claim than what the ratio comparison actually supports.

---

## 3. Bipower/jump-diffusion decomposition: LUNA was diffusion, not a jump — a genuine finding, not noise

`scripts/bipower_jump_diffusion.py`, daily RV/BV/jump decomposition with a chi-square-style derived threshold (target 1 false positive/month → z-threshold 1.849).

**None of the four LUNA days (05-09 to 05-12) were flagged as statistically significant jumps** — z-stats of -0.88, 0.07, 1.19, -0.66; jump shares of 0.0%, 0.2%, 4.8%, 0.0%. RV and BV track each other almost exactly through this period in the plot (the two lines overlap). Meanwhile the month's *largest* jump shares and z-statistics (05-28: 20.3%/z=6.70; 05-22: 15.3%/z=4.22; 05-19: 12.7%/z=4.54) occurred on comparatively ordinary, unrelated days.

**This is a real and useful result, not a discrepancy to resolve away:** it empirically confirms that the LUNA collapse manifested as sustained elevated *diffusion* — exactly the kind of structure the IMM was built to capture (a discrete regime *switch* to a higher-variance state, not a one-tick discontinuity) — rather than as price jumps in the bipower-variation sense. The two phenomena are conceptually distinct, and this is the direct evidence that they were empirically distinct here too. Genuinely strengthens the case for the IMM's framing over a jump-detection approach for this specific event, though the scattered unrelated jump days elsewhere show real jumps are present in the data generally, just not what drove this event.

**Caveat, not swept under the rug:** 9 of 31 days were flagged against a ~1/month calibration target — a real overshoot. Given the R-derivation work already established that microstructure noise varies by regime by 86x, it's plausible the jump test (calibrated under an implicit assumption closer to homogeneous noise) is somewhat miscalibrated at dt=60s once regime-dependent noise is accounted for. Flagged as an open question, not resolved here — the qualitative LUNA-specific finding (no jump, sustained diffusion) is robust regardless, since it's about the *absence* of a signal on the days that matter most, not a borderline flagged count.

---

## Reproducibility

- `python scripts/k4_recheck.py data/raw/BTCUSDT-aggTrades-2022-05.zip`
- `python scripts/naive_baseline.py data/raw/BTCUSDT-aggTrades-2022-05.zip`
- `python scripts/bipower_jump_diffusion.py data/raw/BTCUSDT-aggTrades-2022-05.zip`

## Phase 1 status: fully complete

All items from the original and revised Phase 1 scope are now done: baseline filter (built, evaluated, found to fail in a diagnosed way), IMM (built, validated, K selected and then revisited and revised via the K=4 recheck), per-regime Q/R (derived, cross-validated), naive baseline (built, honestly compared), bipower jump/diffusion split (built, yielded a genuine substantive finding). No open items remain blocking Phase 2.

## Next step

Phase 2: the artifact stress test, using the K=4 IMM as the feature source. Construct the leaky vs. purged label sets, the permutation-derived null, and the walk-forward evaluation harness.
