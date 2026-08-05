# Command reference log

Every command run so far, in order, with what it was for. Keep this updated
as you go — it's meant to make the whole pipeline reproducible from a clean
checkout without reconstructing steps from memory.

## Environment setup

```bash
./setup_venv.sh
# equivalent to:
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python -c "import numpy, pandas, jax, torch, hmmlearn, arch; print('all imports OK')"
```

Note: if the venv is ever moved, verify it still resolves correctly before
assuming it's broken — check `<venv>/pyvenv.cfg` and `<venv>/bin/python*`
symlinks against the current project path. Rebuild only if they actually
point somewhere stale.

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
# Attempt 1 (rejected): April 2022
python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2022-04.zip

# Attempt 2 (accepted): June 2023 -- window 2023-06-13 to 2023-06-19
python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2023-06.zip

# Reused on 2022-05 to check whether "pre-event" could be assumed calm (it couldn't)
python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2022-05.zip
```

## Microstructure noise (R) derivation

```bash
# Attempt 1 (rejected): single-offset whole-month signature plot on 2022-05
python scripts/signature_plot.py data/raw/BTCUSDT-aggTrades-2022-05.zip

# Attempt 2 (rejected as an R source): multi-offset-averaged plot on the calm window
python scripts/signature_plot.py data/raw/BTCUSDT-aggTrades-2023-06.zip \
  --start 2023-06-13 --end 2023-06-19 --label calm_R

# Final method: Roll's estimator
python scripts/roll_estimator.py data/raw/BTCUSDT-aggTrades-2023-06.zip \
  --start 2023-06-13 --end 2023-06-19
# -> R (all-tick, selected) = 3.661341181889e-11

# Tested regime-invariance -- R is NOT shared across regimes (86.7x higher in crisis)
python scripts/check_R_regime_invariance.py data/raw/BTCUSDT-aggTrades-2022-05.zip
```

## Regime-varying process noise (Q) derivation

```bash
python scripts/rolling_q.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> dt=60s, window=1D selected

python scripts/diurnal_check.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> 24h hypothesis ruled out; ~8h periodicity found instead (unconfirmed cause)
```

## Baseline (single-hypothesis) Kalman filter

```bash
python scripts/run_baseline_filter.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> 1.176% flagged vs. 0.069% target; elevated in ALL periods (pre/during/post),
#    not event-specific -- motivates the IMM (see baseline_filter_evaluation.md)
```

## HMM regime fit and per-state derivations

```bash
# K selected by BIC (K=2..5). First run showed EM non-convergence warnings --
# fixed by standardizing returns before fitting.
python scripts/fit_regime_hmm.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> K=5 selected; later revised (see k4_recheck.py below)

python scripts/per_state_R.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> per-state R_k, Q_k = HMM_variance_k - 2*R_k for all 5 states
# -> state 4's R closely matched the original calm-window R -- unplanned cross-check
```

## IMM filter (K=5, superseded -- see K=4 recheck below)

```bash
python scripts/run_imm_filter.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> persistence: 0.159 (pre) / 0.830 (during-early) / 0.424 (during-late)
# -> open item found: states 0/3 flickering (self-transition 0.58)
```

**Note:** `run_imm_filter.py` still hardcodes the K=5 parameters, now
superseded by the K=4 recheck below. Not yet updated to K=4 -- use
`k4_recheck.py`'s output as canonical until it is.

## K=4 recheck, naive baseline, bipower jump/diffusion (Phase 1 final close-out)

```bash
# Resolves the state 0/3 flickering: refits at K=4, compares directly against
# the K=5 reference on the actual persistence result, not just BIC.
python scripts/k4_recheck.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> persistence: 0.165 / 0.842 / 0.447 (comparable-to-better than K=5)
# -> calmest state self-transition: 0.9745 (properly sticky, vs. K=5's 0.58)
# -> K=4 ADOPTED as the model feeding Phase 2

# Near-parameter-free baseline (1h RV > expanding median), compared to the IMM
# via the during/pre RATIO, not raw magnitude (raw magnitude would mislead here).
python scripts/naive_baseline.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> IMM ratio: 5.2x (early) / 2.7x (late); naive ratio: 1.8x / 1.1x
# -> IMM's complexity justified by signal SHAPE, not magnitude

# Bipower variation jump/diffusion decomposition, daily, with a chi-square-style
# derived z-threshold (target 1 false positive/month).
python scripts/bipower_jump_diffusion.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> NONE of the 4 LUNA days flagged as jumps -- LUNA was sustained diffusion,
#    not a discrete jump. Validates the IMM's framing for this event.
# -> caveat: 9/31 days flagged vs. 1/month target -- open calibration question
```

## Status

- R derived and documented: `notes/microstructure_noise_R_derivation.md`
- Q derived and documented: `notes/regime_varying_Q_derivation.md`
- Calm-window selection documented: `notes/calm_window_selection.md`
- Baseline filter evaluation documented: `notes/baseline_filter_evaluation.md`
- IMM (K=5) result documented: `notes/imm_filter_result_phase1_closeout.md`
- K=4 recheck, naive baseline, bipower split documented: `notes/k4_naive_bipower_phase1_final_closeout.md`
- **Phase 1: fully complete.**

## Next commands (not yet run)

- Update `run_imm_filter.py` to the canonical K=4 parameters, or retire it in
  favor of `k4_recheck.py`'s output
- Phase 2: label-set construction (leaky vs. purged), permutation null,
  walk-forward evaluation harness — not yet written
