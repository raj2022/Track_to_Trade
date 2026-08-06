"""
Regime-conditional recalibration -- follows up on a negative result:
scripts/recalibrate.py's time-split isotonic calibration made things WORSE
on its held-out half, in the OPPOSITE miscalibration direction from the
full-dataset check. Diagnosis: the calibration set was crisis-heavy, the
evaluation set was calm-aftermath-heavy, and calibration itself appears to
be regime-dependent (underconfident near the crisis, overconfident in the
calm aftermath) -- a single static time-split correction moves the wrong
way for whichever regime it wasn't fit on.

Fix tested here: condition the calibration split on the model's OWN
elevated_prob feature (above/below its median) rather than on time,
fitting two separate isotonic mappings -- one for "currently elevated"
predictions, one for "currently calm" -- and evaluate each via a proper
train/test split WITHIN each regime (not across regimes), avoiding the
temporal-regime-mismatch problem directly.

Usage:
    python scripts/recalibrate_by_regime.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

DATA_PATH = Path("data/processed/real_label_oof_predictions.parquet")
FEATURES_PATH = Path("data/processed/phase2_dataset.parquet")
PLOTS_DIR = Path("plots")
N_BINS = 8  # fewer bins than before -- each regime subset is roughly half
            # the data, further split for calibration/eval
EPS = 1e-12


def log_score(y_true, y_pred):
    p = np.clip(y_pred, EPS, 1 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))


def reliability_table(y_true, y_pred, n_bins=N_BINS):
    bin_edges = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9
    bin_idx = np.digitize(y_pred, bin_edges[1:-1])
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        mean_pred = y_pred[mask].mean()
        obs_freq = y_true[mask].mean()
        se = np.sqrt(obs_freq * (1 - obs_freq) / n_b)
        rows.append((mean_pred, obs_freq, n_b, se))
    return rows


def calibrate_and_evaluate_within_regime(sub_df, label):
    """Chronological half-split WITHIN one regime subset -- calibrate on
    the first half of that regime's occurrences, evaluate on the second."""
    sub_df = sub_df.sort_index()
    n = len(sub_df)
    split = n // 2
    calib = sub_df.iloc[:split]
    eval_ = sub_df.iloc[split:]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(calib["y_pred_proba"].to_numpy(), calib["y_true"].to_numpy())

    y_true = eval_["y_true"].to_numpy()
    y_raw = eval_["y_pred_proba"].to_numpy()
    y_cal = iso.predict(y_raw)

    print(f"\n--- {label} regime (n={n:,}, calib={len(calib):,}, eval={len(eval_):,}) ---")
    print(f"  log score:   raw={log_score(y_true, y_raw):.4f}  calibrated={log_score(y_true, y_cal):.4f}")
    print(f"  Brier score: raw={brier_score(y_true, y_raw):.4f}  calibrated={brier_score(y_true, y_cal):.4f}")

    raw_table = reliability_table(y_true, y_raw)
    cal_table = reliability_table(y_true, y_cal)
    print(f"  raw diff/SE per bin:        {[f'{(o-p)/s:+.1f}' for p,o,n,s in raw_table]}")
    print(f"  calibrated diff/SE per bin: {[f'{(o-p)/s:+.1f}' for p,o,n,s in cal_table]}")

    return iso, y_true, y_raw, y_cal, raw_table, cal_table


def main():
    if not DATA_PATH.exists() or not FEATURES_PATH.exists():
        sys.exit("Required data files not found -- run save_oof_predictions.py and "
                  "build_phase2_dataset.py first")

    PLOTS_DIR.mkdir(exist_ok=True)
    oof = pd.read_parquet(DATA_PATH)
    features = pd.read_parquet(FEATURES_PATH)[["elevated_prob"]]
    df = oof.join(features, how="inner")

    median_elevated = df["elevated_prob"].median()
    print(f"Splitting by elevated_prob median ({median_elevated:.4f}), not by time.")

    calm_regime = df[df["elevated_prob"] <= median_elevated]
    elevated_regime = df[df["elevated_prob"] > median_elevated]

    results = {}
    for label, sub in [("calm", calm_regime), ("elevated", elevated_regime)]:
        results[label] = calibrate_and_evaluate_within_regime(sub, label)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for col, label in enumerate(["calm", "elevated"]):
        _, y_true, y_raw, y_cal, raw_table, cal_table = results[label]
        for row, (table, title) in enumerate([(raw_table, "raw"), (cal_table, "calibrated")]):
            ax = axes[row, col]
            preds = [r[0] for r in table]
            obs = [r[1] for r in table]
            ses = [r[3] for r in table]
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.errorbar(preds, obs, yerr=ses, fmt="o-", capsize=3)
            ax.set_title(f"{label} regime, {title}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("mean predicted probability")
            ax.set_ylabel("observed frequency")

    fig.tight_layout()
    out_path = PLOTS_DIR / "recalibration_by_regime.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    print("\nDecide, don't assume:")
    print("- Did BOTH regime-conditional calibrations improve their own held-out half")
    print("  (unlike the time-split version, which made things worse)? If yes, this")
    print("  confirms calibration is regime-dependent and this is the right fix.")
    print("- If either regime's calibration still gets worse, the problem may run deeper")
    print("  than a simple two-regime split -- consider conditioning on the actual IMM")
    print("  decoded state (4 regimes) instead of a single elevated_prob median split.")


if __name__ == "__main__":
    main()
