"""
Fits an isotonic regression recalibration on the FIRST HALF (chronological)
of the purged walk-forward OOF predictions, to correct the systematic
underconfidence bias found in scripts/calibration_check.py (bins 3-7,
several SE off, all in the same direction). Evaluates the corrected
reliability on the SECOND HALF only, to avoid calibrating and evaluating
on the same data -- keeping the same walk-forward discipline (never
evaluate on data used to fit anything) used throughout this project.

Usage:
    python scripts/recalibrate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

DATA_PATH = Path("data/processed/real_label_oof_predictions.parquet")
PLOTS_DIR = Path("plots")
N_BINS = 10
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


def main():
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} not found -- run scripts/save_oof_predictions.py first")

    PLOTS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA_PATH).sort_index()
    n = len(df)
    split = n // 2

    calib_df = df.iloc[:split]
    eval_df = df.iloc[split:]
    print(f"Calibration set: {len(calib_df):,} rows ({calib_df.index.min()} to {calib_df.index.max()})")
    print(f"Evaluation set:  {len(eval_df):,} rows ({eval_df.index.min()} to {eval_df.index.max()})")

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(calib_df["y_pred_proba"].to_numpy(), calib_df["y_true"].to_numpy())

    y_true_eval = eval_df["y_true"].to_numpy()
    y_pred_raw = eval_df["y_pred_proba"].to_numpy()
    y_pred_calibrated = iso.predict(y_pred_raw)

    print(f"\n{'':>20}  {'raw':>10}  {'recalibrated':>14}")
    print(f"{'log score':>20}  {log_score(y_true_eval, y_pred_raw):>10.4f}  "
          f"{log_score(y_true_eval, y_pred_calibrated):>14.4f}")
    print(f"{'Brier score':>20}  {brier_score(y_true_eval, y_pred_raw):>10.4f}  "
          f"{brier_score(y_true_eval, y_pred_calibrated):>14.4f}")

    raw_table = reliability_table(y_true_eval, y_pred_raw)
    calib_table = reliability_table(y_true_eval, y_pred_calibrated)

    print(f"\nRaw reliability (evaluation half only):")
    print(f"  {'pred':>8}  {'obs':>8}  {'n':>7}  {'SE':>8}  {'diff/SE':>8}")
    for pred, obs, n_b, se in raw_table:
        diff_se = (obs - pred) / se if se > 0 else float("nan")
        print(f"  {pred:>8.4f}  {obs:>8.4f}  {n_b:>7}  {se:>8.4f}  {diff_se:>+8.2f}")

    print(f"\nRecalibrated reliability (evaluation half only):")
    print(f"  {'pred':>8}  {'obs':>8}  {'n':>7}  {'SE':>8}  {'diff/SE':>8}")
    for pred, obs, n_b, se in calib_table:
        diff_se = (obs - pred) / se if se > 0 else float("nan")
        print(f"  {pred:>8.4f}  {obs:>8.4f}  {n_b:>7}  {se:>8.4f}  {diff_se:>+8.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, table, title in [(axes[0], raw_table, "Raw (uncalibrated)"),
                               (axes[1], calib_table, "Recalibrated (isotonic)")]:
        preds = [r[0] for r in table]
        obs = [r[1] for r in table]
        ses = [r[3] for r in table]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.errorbar(preds, obs, yerr=ses, fmt="o-", capsize=3)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    out_path = PLOTS_DIR / "recalibration_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    # Save the recalibrated evaluation-half predictions for the Bayes
    # threshold step -- this is the trustworthy probability to threshold,
    # not the raw one.
    out_data = pd.DataFrame({
        "y_true": y_true_eval,
        "y_pred_raw": y_pred_raw,
        "y_pred_calibrated": y_pred_calibrated,
    }, index=eval_df.index)
    out_data_path = Path("data/processed/real_label_calibrated_predictions.parquet")
    out_data.to_parquet(out_data_path)
    print(f"Saved calibrated predictions to {out_data_path}")

    print("\nCheck before trusting the calibrated version:")
    print("- Do the |diff/SE| values in the recalibrated table drop well below 2 across")
    print("  the bins that were previously off (especially the former 8-sigma bin)?")
    print("- Do log score / Brier score improve (lower) after recalibration? Isotonic")
    print("  regression is fit to minimize exactly this kind of miscalibration, so a")
    print("  clear improvement is expected -- but confirm it, don't assume it.")


if __name__ == "__main__":
    main()
