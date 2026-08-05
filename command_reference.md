# Command reference log

Every command run so far, in order, with what it was for. Keep this updated
as you go — it's meant to make the whole pipeline reproducible from a clean
checkout without reconstructing steps from memory.

## Environment setup

```bash
# Create and populate the virtual environment
./setup_venv.sh
# equivalent to:
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Verify the environment
python -c "import numpy, pandas, jax, torch, hmmlearn, arch; print('all imports OK')"
```

Note: if the venv is ever moved, verify it still resolves correctly before
assuming it's broken — check `<venv>/pyvenv.cfg` and `<venv>/bin/python*`
symlinks against the current project path. Rebuild only if they actually
point somewhere stale.

## Data acquisition

```bash
# Download and checksum-verify the 15 monthly BTCUSDT aggTrades files
# (5 contiguous 3-month windows around COVID, May 2021, LUNA, FTX, plus a calm baseline)
python scripts/download_data.py

# Sanity-check row counts per file (crash months should exceed calm neighbors
# WITHIN each 3-month window; do not compare magnitudes ACROSS windows —
# confounded by secular growth in exchange activity 2020-2023)
for f in data/raw/BTCUSDT-aggTrades-*.zip; do
  n=$(unzip -p "$f" | wc -l)
  echo "$f: $n rows"
done
```

## Calm-window selection (for the R derivation)

```bash
# Attempt 1 (rejected): April 2022 — outlier trim excluded 0/30 days,
# gradual pre-crash ramp, not a separable calm/spike population
python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2022-04.zip

# Attempt 2 (accepted): June 2023 — trim excluded 3/30 days, matching
# visual outliers. Selected window: 2023-06-13 to 2023-06-19
python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2023-06.zip

# Reused later on 2022-05 to check whether "pre-event" could be assumed
# calm (it couldn't -- see baseline filter evaluation, below)
python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2022-05.zip
```

## Microstructure noise (R) derivation

```bash
# Attempt 1 (rejected): single-offset, whole-month signature plot on
# 2022-05 — jagged/non-monotonic, pooled a calm period with the LUNA crash
python scripts/signature_plot.py data/raw/BTCUSDT-aggTrades-2022-05.zip

# Attempt 2 (rejected as an R source, but useful): multi-offset-averaged
# signature plot on the verified calm window — revealed a stale-price
# artifact at short intervals (44.7% of 1s bins stale), ruling out the
# fixed-interval approach for R here
python scripts/signature_plot.py data/raw/BTCUSDT-aggTrades-2023-06.zip \
  --start 2023-06-13 --end 2023-06-19 --label calm_R

# Final method: Roll's (1984) implied-spread estimator on tick-level
# returns, with a same-timestamp (order-book-sweep) robustness check built in
python scripts/roll_estimator.py data/raw/BTCUSDT-aggTrades-2023-06.zip \
  --start 2023-06-13 --end 2023-06-19
# -> R (all-tick, selected) = 3.661341181889e-11
# -> R (same-timestamp excluded, robustness check) = 4.098231808572e-11 (+11.93%)

# Tested whether R is regime-invariant (flagged as an open item in the
# original R derivation) -- it is NOT: crisis-window R is 86.7x the calm value
python scripts/check_R_regime_invariance.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> R (crisis, 05-09 to 05-12) = 3.175645120926e-09
# -> ratio (crisis/calm) = 86.73x -> R must be regime-conditional, not shared
```

## Regime-varying process noise (Q) derivation

```bash
# Re-validates sampling dt for 2022-05 (don't assume the calm-window's dt
# transfers), then plots rolling RV-rate candidates (1h/4h/12h/1D/3D)
# against the known LUNA depeg date (2022-05-09)
python scripts/rolling_q.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> dt = 60s entered at prompt
# -> window = 1D selected

# Checks whether the noisy short-window series shows a genuine ~24h
# (diurnal) periodicity, as a candidate explanation for why 1D smooths
# the series more than shorter windows
python scripts/diurnal_check.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> 24h hypothesis ruled out; found a ~8h periodicity instead
#    (working hypothesis: Binance perpetual funding settlement, unverified)
```

## Baseline (single-hypothesis) Kalman filter

```bash
# Local-level Kalman filter using the derived R and rolling Q as fixed,
# calibrated inputs. Derives the chi-square gate from an explicit target
# false-alarm rate (1/day) rather than a stated significance level.
python scripts/run_baseline_filter.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> 1.176% of steps flagged vs. a 0.069% target (~17x over)
# -> pre/during/post flag rates: 1.945% / 1.476% / 0.789% (elevated in ALL
#    three periods -- not event-specific; see baseline_filter_evaluation.md)
```

## HMM regime fit and per-state derivations (IMM inputs)

```bash
# Fits a Gaussian HMM (EM/Baum-Welch) to 2022-05 returns, K=2..5, selects
# K by BIC rather than asserting it. First run showed "not converging"
# warnings across every K -- fixed by standardizing returns before fitting.
python scripts/fit_regime_hmm.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> K=5 selected (BIC), modest margin over K=4 (608 pts vs. 1507 for K3->K4)
# -> per-state variances and transition matrix -- see notes/ for full table

# Roll's estimator computed SEPARATELY for each of the 5 HMM states (tick
# pairs restricted to that state's bins), rather than mapping 5 variance
# levels onto only 2 (calm/crisis) R estimates. Refits the same K=5 HMM
# internally (same seed/restarts) to reproduce the state labeling.
python scripts/per_state_R.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> per-state R_k and Q_k = HMM_variance_k - 2*R_k, all 5 states
# -> state 4's R (3.620e-11) closely matched the original calm-window R
#    (3.661e-11) -- unplanned cross-validation
# -> state 0's R clipped from a small negative value to 0
```

## IMM (Interacting Multiple Model) filter

```bash
# Runs the IMM using the derived per-regime Q/R and the fitted transition
# matrix as fixed inputs (not re-fit here). Saves to plots/, per the
# plots-folder convention adopted partway through the project -- earlier
# scripts save plots to the working directory instead; not retroactively
# changed.
python scripts/run_imm_filter.py data/raw/BTCUSDT-aggTrades-2022-05.zip
# -> mean P(elevated/extreme): 0.159 pre-event, 0.830 during 05-09 to 05-13,
#    0.424 during 05-13 to 05-19 (real decay, reported honestly -- not just
#    the flattering 10-day aggregate of 0.587)
# -> open item: states 0/3 show persistent flickering throughout the month,
#    traced to the fitted transition matrix itself -- see notes/ for detail
```

## Status

- R derived and documented: `notes/microstructure_noise_R_derivation.md`
- Q derived and documented: `notes/regime_varying_Q_derivation.md`
- Calm-window selection documented: `notes/calm_window_selection.md`
- Baseline filter evaluation documented: `notes/baseline_filter_evaluation.md`
- IMM result and Phase 1 close-out documented: `notes/imm_filter_result_phase1_closeout.md`

## Next commands (not yet run)

- Naive baseline model (trailing-median rule, evaluated walk-forward) — not yet written
- Bipower-variation jump/diffusion decomposition — not yet written
- K=4 HMM refit-and-compare, to resolve the state 0/3 flickering open item — not yet run
- Phase 2: label-set construction (leaky vs. purged), permutation null, walk-forward evaluation harness — not yet written
