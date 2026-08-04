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

## Status

- R derived and documented: `notes/microstructure_noise_R_derivation.md`
- Q derived and documented: `notes/regime_varying_Q_derivation.md`
- Calm-window selection documented: `notes/calm_window_selection.md`

## Next commands (not yet run)

- Baseline model script (naive rule, evaluated walk-forward) — not yet written
- Kalman/IMM filter implementation — not yet written
