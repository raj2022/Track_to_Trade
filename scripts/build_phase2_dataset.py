"""
Build the Phase 2 dataset: features (strictly causal), the real forward-
looking label, and a block-permuted null label with no genuine feature-
label relationship but the same overlap/autocorrelation structure as the
real one.

Uses the K=4 IMM (per_state_R.py / k4_recheck.py parameters) as the
feature source. H (label horizon) derived in scripts/derive_horizon.py.

Usage:
    python scripts/build_phase2_dataset.py data/raw/BTCUSDT-aggTrades-2022-05.zip
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_regime_hmm import build_returns, fit_best_hmm, DT_SEC
from per_state_R import load_trades, assign_tick_states, roll_R_for_state

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.imm import run_imm

K = 4
N_RESTARTS = 5
SEED = 0
ELEVATED_STATES = [2, 3]  # per k4_recheck.py's above-median-variance classification
H = 60  # derived in scripts/derive_horizon.py (60 steps at dt=60s = 1 hour)
LABEL_THRESHOLD = 0.5  # natural indifference point, not fitted
N_NULL_SHIFTS = 20  # number of independent circular-shift nulls to build

DATA_DIR = Path("data/processed")


def build_features_and_imm(zip_path: Path):
    """Refit the K=4 IMM (same procedure as k4_recheck.py) and return the
    elevated-probability series plus a small causal feature set."""
    df = load_trades(zip_path)
    returns, scale = build_returns(df["price"], DT_SEC)

    model, ll = fit_best_hmm(returns, K, N_RESTARTS, SEED)
    variances = model.covars_.flatten() * (scale ** 2)

    states = model.predict(returns)
    resampled = df["price"].resample(f"{DT_SEC}s").last().ffill()
    log_price = np.log(resampled)
    state_index = log_price.diff().dropna().index
    state_series = pd.Series(states, index=state_index)

    state_per_tick = assign_tick_states(df.index, state_series, DT_SEC)
    prices = df["price"].to_numpy()

    Q = np.empty(K)
    R = np.empty(K)
    for k in range(K):
        r_k, n_pairs = roll_R_for_state(prices, state_per_tick, k)
        r_k_used = 0.0 if (np.isnan(r_k) or r_k < 0) else r_k
        Q[k] = max(variances[k] - 2 * r_k_used, 1e-15)
        R[k] = r_k_used

    z = log_price.to_numpy()
    index = log_price.index
    result = run_imm(z, Q, R, model.transmat_)

    elevated_prob = pd.Series(
        result.mode_probs[:, ELEVATED_STATES].sum(axis=1), index=index, name="elevated_prob"
    )

    log_ret = log_price.diff()
    rolling_vol = log_ret.rolling("1h").std()

    features = pd.DataFrame({
        "elevated_prob": elevated_prob,
        "elevated_prob_lag1": elevated_prob.shift(1),
        "rolling_vol_1h": rolling_vol,
    })

    return features, elevated_prob


def build_real_label(elevated_prob: pd.Series, h: int, threshold: float) -> pd.Series:
    """y_t = 1 if mean elevated-prob over (t+1, t+h] exceeds threshold.
    Strictly forward-looking BY DESIGN -- this is legitimate for a label
    (labels should look forward); the leakage risk is in validation, not
    in constructing the label this way."""
    forward_mean = elevated_prob.shift(-1).rolling(h).mean().shift(-(h - 1))
    return (forward_mean > threshold).astype(float).where(forward_mean.notna())


def build_null_labels(real_label: pd.Series, n_shifts: int, min_shift: int, seed: int) -> list:
    """
    Many circular-shift nulls, not one. A single circular shift can
    accidentally realign with real periodic structure already known to
    exist in this data (the ~8h periodicity found in the diurnal check,
    scripts/diurnal_check.py) -- shifting by a near-multiple of that period
    lands the series back nearly in phase (spurious positive correlation);
    a near-half-period offset lands it anti-phase (spurious negative
    correlation). Real-data testing showed exactly this: one run produced
    AUC=0.28, another AUC=0.59, both far from 0.5, in opposite directions.

    Fix: draw many shifts, explicitly excluding any offset within a buffer
    of a multiple of the known periods (480 steps = 8h, 1440 steps = 24h),
    and treat the SET of resulting labels as the null -- report the
    distribution across shifts, not a single point estimate.
    """
    valid = real_label.dropna()
    values = valid.to_numpy()
    n = len(values)

    known_periods_steps = [480, 1440]  # 8h, 24h at dt=60s
    exclusion_buffer = 30  # steps

    def too_close_to_a_period(shift: int) -> bool:
        for period in known_periods_steps:
            remainder = shift % period
            if remainder < exclusion_buffer or (period - remainder) < exclusion_buffer:
                return True
        return False

    rng = np.random.default_rng(seed)
    labels = []
    attempts = 0
    while len(labels) < n_shifts and attempts < n_shifts * 20:
        attempts += 1
        shift = int(rng.integers(min_shift, n - min_shift))
        if too_close_to_a_period(shift):
            continue
        shifted = np.roll(values, shift)
        labels.append(pd.Series(shifted, index=valid.index, name=f"null_label_{len(labels)}"))

    if len(labels) < n_shifts:
        print(f"WARNING: only found {len(labels)}/{n_shifts} valid shifts avoiding "
              f"known periods after {attempts} attempts -- period exclusion may be "
              f"too aggressive relative to min_shift range.")

    return labels


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_phase2_dataset.py <path_to_zip>")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = Path(sys.argv[1])

    print("Refitting K=4 IMM for the feature source...")
    features, elevated_prob = build_features_and_imm(zip_path)

    print(f"Building real label (H={H} steps, threshold={LABEL_THRESHOLD})...")
    real_label = build_real_label(elevated_prob, H, LABEL_THRESHOLD)

    print(f"Building {N_NULL_SHIFTS} circular-shift null labels "
          f"(excluding offsets near the known 8h/24h periods)...")
    null_labels = build_null_labels(real_label, n_shifts=N_NULL_SHIFTS, min_shift=1440, seed=SEED)

    dataset = features.copy()
    dataset["real_label"] = real_label
    for null_series in null_labels:
        dataset = dataset.join(null_series, how="inner")
    dataset = dataset.dropna()

    print(f"\n{len(dataset):,} complete rows after dropping warm-up/lookahead NaNs")
    print(f"Real label positive rate: {dataset['real_label'].mean():.3f}")
    null_rates = [dataset[f"null_label_{i}"].mean() for i in range(len(null_labels))]
    print(f"Null label positive rates across {len(null_labels)} shifts: "
          f"mean={np.mean(null_rates):.3f} (should match real rate -- "
          f"circular shifts preserve the marginal distribution exactly)")

    out_path = DATA_DIR / "phase2_dataset.parquet"
    dataset.to_parquet(out_path)
    print(f"\nSaved to {out_path} ({len(null_labels)} null_label_N columns)")
    print("\nSanity check before trusting this for anything downstream:")
    print("- All null label positive rates should be IDENTICAL to the real rate")
    print("  (circular shift preserves the marginal distribution exactly, unlike")
    print("  the earlier rejected block-permutation approach).")
    print("- Downstream, evaluate null_label_0..N SEPARATELY and look at the")
    print("  DISTRIBUTION of resulting AUCs across shifts, not a single number --")
    print("  a single shift can still land badly by chance even with period exclusion.")


if __name__ == "__main__":
    main()