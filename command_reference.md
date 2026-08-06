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

**Phase 1: fully complete. Phase 2: leakage mechanism demonstrated; real-label full evaluation and cross-window robustness check outstanding.**

## Next commands (not yet run)

- Update `run_imm_filter.py` to the canonical K=4 parameters, or retire it
- Re-run `build_phase2_dataset.py` / the stress test on another event window
  (COVID, May 2021, or FTX) to confirm the leakage result isn't LUNA-specific
- More null shifts (50-100 instead of 20) to tighten the purged walk-forward
  residual estimate (+0.043, std 0.055) enough to know if it's real
- Phase 3: calibration, Bayes-risk decision threshold, cost-aware check
