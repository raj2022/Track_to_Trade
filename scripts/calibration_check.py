"""
Reliability diagram and proper scoring rules (log score, Brier score) for
the purged-walk-forward OOF predictions on real_label.

Both scores are benchmarked against a "climatology" baseline (always
predict the overall positive rate) -- reporting a skill score (how much
better than trivial) rather than a raw number without context, same
derivation discipline as everywhere else in this project.

Usage:
    python scripts/calibration_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/real_label_oof_predictions.parquet")
PLOTS_DIR = Path("plots")
N_BINS = 10  # quantile-based bins -- standard convention for reliability
             # diagrams; per-bin sample size and binomial standard error
             # are reported explicitly so bin trustworthiness can be judged
             # directly, rather than trying to over-derive an "optimal" bin
             # count that would still just be a different convention.
EPS = 1e-12  # clipping for log score, avoids log(0)


def log_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = np.clip(y_pred, EPS, 1 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def main():
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} not found -- run scripts/save_oof_predictions.py first")

    PLOTS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred_proba"].to_numpy()
    n = len(df)
    base_rate = y_true.mean()

    print(f"Loaded {n:,} OOF predictions, base rate = {base_rate:.4f}")

    # --- Proper scoring rules, benchmarked against climatology ---
    model_log = log_score(y_true, y_pred)
    climatology_log = log_score(y_true, np.full(n, base_rate))
    log_skill = 1 - model_log / climatology_log

    model_brier = brier_score(y_true, y_pred)
    climatology_brier = brier_score(y_true, np.full(n, base_rate))
    brier_skill = 1 - model_brier / climatology_brier

    print(f"\nLog score:   model={model_log:.4f}  climatology={climatology_log:.4f}  "
          f"skill={log_skill:+.4f} (fraction better than always predicting the base rate)")
    print(f"Brier score: model={model_brier:.4f}  climatology={climatology_brier:.4f}  "
          f"skill={brier_skill:+.4f}")

    # --- Reliability diagram, quantile-based bins ---
    bin_edges = np.quantile(y_pred, np.linspace(0, 1, N_BINS + 1))
    bin_edges[-1] += 1e-9  # ensure the max value falls in the last bin
    bin_idx = np.digitize(y_pred, bin_edges[1:-1])

    print(f"\n{'bin':>4}  {'n':>7}  {'mean predicted':>15}  {'observed freq':>14}  {'binomial SE':>12}")
    bin_mean_pred, bin_obs_freq, bin_n, bin_se = [], [], [], []
    for b in range(N_BINS):
        mask = bin_idx == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        mean_pred = y_pred[mask].mean()
        obs_freq = y_true[mask].mean()
        se = np.sqrt(obs_freq * (1 - obs_freq) / n_b)
        bin_mean_pred.append(mean_pred)
        bin_obs_freq.append(obs_freq)
        bin_n.append(n_b)
        bin_se.append(se)
        print(f"  {b:>4}  {n_b:>7}  {mean_pred:>15.4f}  {obs_freq:>14.4f}  {se:>12.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    ax.errorbar(bin_mean_pred, bin_obs_freq, yerr=bin_se, fmt="o-",
                capsize=3, label="observed (purged walk-forward OOF)")
    ax.set_xlabel("mean predicted probability (bin)")
    ax.set_ylabel("observed frequency (bin)")
    ax.set_title("Reliability diagram — real_label, purged walk-forward")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    out_path = PLOTS_DIR / "reliability_diagram.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    print("\nWhat to check, not just eyeball:")
    print("- Do points sit close to the diagonal, WITHIN their error bars? A point")
    print("  off the diagonal by more than ~2x its own SE is a real calibration gap,")
    print("  not noise -- worth flagging which bins specifically, not just 'roughly ok'.")
    print("- Positive skill scores (both) mean the model beats trivially predicting")
    print("  the base rate every time -- if either is negative or near zero, that")
    print("  contradicts the Phase 2 finding of genuine signal and needs revisiting.")


if __name__ == "__main__":
    main()
