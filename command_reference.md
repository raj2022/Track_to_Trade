# Command reference log

Every command run so far, in order, with what it was for. Keep this updated
as you go — it's meant to make the whole pipeline reproducible from a clean
checkout without reconstructing steps from memory.

## Environment setup

```bash
./setup_venv.sh
python -c "import numpy, pandas, jax, torch, hmmlearn, arch, sklearn, pyarrow; print('all imports OK')"
```

## Data acquisition

```bash
python scripts/download_data.py

for f in data/raw/BTCUSDT-aggTrades-*.zip; do
  n=$(unzip -p "$f" | wc -l)
  echo "$f: $n rows"
done
```

## Calm-window selection (for the R derivation)

```bash
python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2022-04.zip  # rejected
python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2023-06.zip  # accepted, 06-13 to 06-19
python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2022-05.zip  # reused for the "pre-event" check
```

## Microstructure noise (R) derivation

```bash
python scripts/signature_plot.py data/raw/BTCUSDT-aggTrades-2022-05.zip  # rejected
python scripts/signature_plot.py data/raw/BTCUSDT-aggTrades-2023-06.zip \
  --start 2023-06-13 --end 2023-06-19 --label calm_R  # rejected as R source
python scripts/roll_estimator.py data/raw/BTCUSDT-aggTrades-2023-06.zip \
  --start 2023-06-13 --end 2023-06-19  # R = 3.661341181889e-11 (all-tick, selected)
python scripts/check_R_regime_invariance.py data/raw/BTCUSDT-aggTrades-2022-05.zip  # 86.7x higher in crisis
```

## Regime-varying process noise (Q) derivation

```bash
python scripts/rolling_q.py data/raw/BTCUSDT-aggTrades-2022-05.zip  # dt=60s, window=1D
python scripts/diurnal_check.py data/raw/BTCUSDT-aggTrades-2022-05.zip  # 24h ruled out; ~8h found
```

## Baseline (single-hypothesis) Kalman filter

```bash
python scripts/run_baseline_filter.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> miscalibrated in all periods; motivates the IMM
```

## HMM regime fit and per-state derivations (K=5, superseded)

```bash
python scripts/fit_regime_hmm.py data/raw/BTCUSDT-aggTrades-2022-05.zip  # K=5 selected (BIC)
python scripts/per_state_R.py data/raw/BTCUSDT-aggTrades-2022-05.zip     # per-state R_k, Q_k
python scripts/run_imm_filter.py data/raw/BTCUSDT-aggTrades-2022-05.zip  # K=5 IMM (superseded)
```

## K=4 recheck, naive baseline, bipower jump/diffusion (Phase 1 final close-out)

```bash
python scripts/k4_recheck.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> K=4 ADOPTED (persistence preserved, calmest state properly sticky)
python scripts/naive_baseline.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> IMM justified by signal shape (5.2x/2.7x ratio), not raw magnitude
python scripts/bipower_jump_diffusion.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> LUNA = sustained diffusion, not a jump; 9/31 days flagged vs 1/month target (open item)
```

**Phase 1 complete.**

## Phase 2: leakage stress test

```bash
# H derived from the K=4 transition matrix (pure linear algebra, no data file needed)
python scripts/derive_horizon.py
# -> H = 54 min (exact), rounded to 60

# Refits the K=4 IMM, builds features + real label + 20 circular-shift null
# labels (period-excluded). Superseded an earlier block-permutation version
# and an earlier single-shift version -- see notes/phase2_leakage_stress_test_full_arc.md
python scripts/build_phase2_dataset.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> data/processed/phase2_dataset.parquet (real_label + null_label_0..19)

# Raw stress test: real/null label x naive/purged CV, with a permutation-null
# significance check per combination. Useful for the real_label result;
# the raw null_label numbers alone were MISLEADING (see below) -- don't
# trust them without the follow-up comparison.
python scripts/phase2_leakage_stress_test.py
# -> real_label: naive AUC=0.918, purged AUC=0.914 (near-tautological, small gap)
# -> null_label: naive mean=0.479 (noisy), purged mean=0.594 (CONSISTENT, all 20/20
#    shifts >0.5) -- backwards from the leakage hypothesis, needed diagnosis

# Decisive follow-up: real model AUC minus a feature-blind (class-prior-only)
# dummy baseline, per shift, per scheme -- isolates the genuinely feature-
# driven effect from the label's own autocorrelation interacting with each
# CV scheme's mechanics.
python scripts/phase2_real_vs_dummy_comparison.py
# -> naive_kfold: mean diff=+0.349, 20/20 shifts positive (p~1e-6) -- clear leakage
# -> purged_walk_forward: mean diff=+0.043, std=0.055, 14/20 positive (p~0.04)
#    -- mostly eliminated, not proven exactly zero
```

**Reusable modules (not run directly):**
- `src/cv_splits.py` — `naive_kfold_splits`, `purged_embargoed_walk_forward_splits`

## Phase 3: calibration and Bayes-risk threshold (complete)

```bash
# Saves the purged walk-forward OOF predictions on real_label for reuse
python scripts/save_oof_predictions.py
# -> data/processed/real_label_oof_predictions.parquet

# Log score, Brier score (vs. climatology skill), reliability diagram with
# per-bin binomial SE. Found systematic underconfidence in bins spanning
# predicted probability ~0.05-0.5 (up to 8 std off).
python scripts/calibration_check.py

# Attempt 1 (REJECTED): isotonic regression, chronological time split.
python scripts/recalibrate.py

# Attempt 2 (REJECTED): regime-conditional split (elevated_prob median).
python scripts/recalibrate_by_regime.py

# Attempts 3-5 (ALL REJECTED): walk-forward isotonic (expanding), walk-
# forward isotonic (rolling, 5d), walk-forward Platt scaling (rolling, 5d).
# Toggle CALIBRATION_METHOD and ROLLING_WINDOW_DAYS to reproduce each.
python scripts/calibrate_walk_forward.py

# DECISION: five attempts failed. Proceed with RAW probabilities; the
# known miscalibrated range (~0.05-0.5) is stated as an explicit limitation
# rather than hidden. See notes/phase3_calibration_drift.md.

# Bayes-risk threshold, derived from the Roll spread ratio (crisis/calm =
# 1.127/0.121 ~ 9.31), applied to raw probabilities.
python scripts/bayes_threshold.py
# -> p* = 0.097; mean cost reduction vs. naive p=0.5: +48.6%
# -> empirical cost-curve minimum (~0.10-0.13) sits close to p* despite
#    the documented miscalibration -- a real, if imperfect, validation
```

**Phase 3 complete.** Full arc: `notes/phase3_calibration_drift.md`
(five failed recalibration attempts, honestly documented) and
`notes/phase3_bayes_threshold_result.md` (final decision-theoretic result).

## Status

Phase 1 notes:
- `notes/microstructure_noise_R_derivation.md`
- `notes/regime_varying_Q_derivation.md`
- `notes/calm_window_selection.md`
- `notes/baseline_filter_evaluation.md`
- `notes/imm_filter_result_phase1_closeout.md`
- `notes/k4_naive_bipower_phase1_final_closeout.md`

Phase 2 notes:
- `notes/phase2_leakage_stress_test_full_arc.md`
- `notes/cross_window_robustness_check.md`

Phase 3 notes:
- `notes/phase3_calibration_drift.md`
- `notes/phase3_bayes_threshold_result.md`

Phase 4 notes:
- `notes/phase4_model_complexity_and_hardware.md`

**Phase 1: fully complete. Phase 2: complete, cross-window validated. Phase 3: complete. Phase 4: complete. `ISSUES.md` #1-3 resolved.**

## Cross-window robustness check (post-Phase-3 cleanup)

```bash
# Build features/labels fresh on an independent event window (FTX, 2022-11).
# Fixed a real bug in the process: ELEVATED_STATES was hardcoded from May's
# specific HMM fit and needed to be derived per-window by variance rank
# instead (state indices are arbitrary per EM run).
python scripts/build_phase2_dataset.py data/raw/BTCUSDT-aggTrades-2022-11.zip
# -> data/processed/phase2_dataset_BTCUSDT-aggTrades-2022-11.parquet

# Rerun the real-vs-dummy comparison on the new dataset (now accepts an
# optional CLI path argument; defaults to the original LUNA dataset).
python scripts/phase2_real_vs_dummy_comparison.py data/processed/phase2_dataset_BTCUSDT-aggTrades-2022-11.parquet
# -> naive leakage signature replicated: 20/20 shifts positive (same as LUNA)
# -> real label signal replicated, slightly stronger: ~4.0/7.4 sigma vs LUNA's ~3.7/6.1
# -> RESOLVED ISSUES.md #3: purged residual on FTX is +0.008, 10/20 positive
#    (coin flip) -- LUNA's +0.043 was sampling noise, not a genuine leak
```

`scripts/run_imm_filter.py` updated to the canonical K=4 parameters
(`ISSUES.md` #1, resolved) -- verified to run cleanly with the new Q/R/
transition matrix before handing off.

## Phase 4: model complexity + hardware investigation (complete)

```bash
# Build 3 more windows for a genuinely diverse combined dataset (LUNA and
# FTX already built above). Same per-window elevated-state derivation.
python scripts/build_phase2_dataset.py data/raw/BTCUSDT-aggTrades-2020-03.zip
python scripts/build_phase2_dataset.py data/raw/BTCUSDT-aggTrades-2021-05.zip
python scripts/build_phase2_dataset.py data/raw/BTCUSDT-aggTrades-2023-07.zip

# Merge all 5 windows into one combined dataset (auto-discovers all
# phase2_dataset_BTCUSDT-aggTrades-*.parquet files in data/processed/).
python scripts/merge_multi_window_dataset.py
# -> data/processed/phase2_dataset_combined.parquet, 221,450 rows, 5 windows

# Logistic regression vs. MLP (16/32/64 units), scikit-learn implementation.
python scripts/compare_lr_mlp_combined.py
# -> AUC: LR=0.9105, mlp_16=-0.0093, mlp_32=-0.0060, mlp_64=-0.0067 (all lose)

# Same comparison, PyTorch (auto-detects Apple Silicon MPS, or force with
# a CLI arg: "cpu" or "mps", for a controlled hardware comparison).
python scripts/compare_lr_mlp_torch_mps.py          # auto-detect (mps on this machine)
python scripts/compare_lr_mlp_torch_mps.py cpu      # force CPU, same code
# -> MLP still loses to LR in both device runs (9 total model runs, 3 implementations)
# -> controlled CPU-vs-MPS timing: 16-unit MLP 1.57x SLOWER on MPS, 64-unit
#    MLP 41% FASTER on MPS -- real overhead-vs-compute crossover, measured
```

**Decision: no MLP beat logistic regression anywhere, so the original**
**compression-study plan doesn't apply.** Redirected to a controlled
CPU-vs-MPS hardware benchmark instead. Full detail:
`notes/phase4_model_complexity_and_hardware.md`.

## Next commands (not yet run)

- `ISSUES.md` #4, #5, #6 remain open (low priority, optional polish) —
  project's active development is otherwise complete
