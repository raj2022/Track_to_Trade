"""
Derives the Bayes-risk decision threshold from the Roll-implied spread
ratio (Phase 1), and applies it to the RAW (uncalibrated) purged
walk-forward probabilities -- five recalibration attempts all failed to
produce a robust fix (see notes/phase3_calibration_drift.md), so this
proceeds on raw probabilities with that limitation stated explicitly,
rather than on a recalibration that didn't hold up.

Cost framing: a false negative (missing a real elevated-regime call) means
continuing to quote the tight calm spread into an actually-elevated
period, exposed to the wider crisis-level adverse selection. A false
positive (false alarm) means needlessly widening to the crisis spread
during what's actually calm, giving up the tighter spread's edge. The
natural cost ratio is the spread ratio itself.

Usage:
    python scripts/bayes_threshold.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/processed/real_label_oof_predictions.parquet")
PLOTS_DIR = Path("plots")

# Derived in notes/microstructure_noise_R_derivation.md and
# check_R_regime_invariance -- Roll-implied effective spreads.
SPREAD_CALM_BP = 0.121
SPREAD_CRISIS_BP = 1.127

# Known limitation, stated explicitly rather than ignored: raw probabilities
# are systematically underconfident in roughly the 0.05-0.5 range (up to 8
# sigma off in the original reliability check). The derived threshold below
# will very likely fall inside that range.
MISCALIBRATED_RANGE = (0.05, 0.5)


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray, threshold: float,
                   cost_fp: float, cost_fn: float) -> dict:
    predicted_positive = y_pred > threshold
    fp = np.sum(predicted_positive & (y_true == 0))
    fn = np.sum(~predicted_positive & (y_true == 1))
    tp = np.sum(predicted_positive & (y_true == 1))
    tn = np.sum(~predicted_positive & (y_true == 0))
    n = len(y_true)
    total_cost = fp * cost_fp + fn * cost_fn
    return {"fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
            "total_cost": total_cost, "mean_cost": total_cost / n,
            "flagged_rate": float(predicted_positive.mean())}


def main():
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} not found -- run scripts/save_oof_predictions.py first")

    PLOTS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred_proba"].to_numpy()

    # --- Derive the cost ratio and Bayes-optimal threshold ---
    cost_ratio = SPREAD_CRISIS_BP / SPREAD_CALM_BP  # C_FN / C_FP
    cost_fp = 1.0
    cost_fn = cost_ratio
    p_star = cost_fp / (cost_fp + cost_fn)

    print(f"Cost ratio (C_FN/C_FP) = spread_crisis/spread_calm = "
          f"{SPREAD_CRISIS_BP}/{SPREAD_CALM_BP} = {cost_ratio:.3f}")
    print(f"Bayes-optimal threshold p* = C_FP/(C_FP+C_FN) = {p_star:.4f}")

    if MISCALIBRATED_RANGE[0] <= p_star <= MISCALIBRATED_RANGE[1]:
        print(f"\nWARNING: p* falls inside the known miscalibrated range "
              f"{MISCALIBRATED_RANGE} (raw probabilities are systematically "
              f"underconfident there, up to 8 std off in the original "
              f"reliability check -- see notes/phase3_calibration_drift.md). "
              f"Treat the results below as directionally informative, not "
              f"precisely trustworthy at this exact threshold value.")

    # --- Compare against the naive 0.5 threshold ---
    result_bayes = expected_cost(y_true, y_pred, p_star, cost_fp, cost_fn)
    result_naive = expected_cost(y_true, y_pred, 0.5, cost_fp, cost_fn)

    print(f"\n{'':>20}  {'p*=' + f'{p_star:.3f}':>12}  {'naive p=0.5':>12}")
    print(f"{'flagged rate':>20}  {result_bayes['flagged_rate']:>12.3f}  {result_naive['flagged_rate']:>12.3f}")
    print(f"{'false positives':>20}  {result_bayes['fp']:>12}  {result_naive['fp']:>12}")
    print(f"{'false negatives':>20}  {result_bayes['fn']:>12}  {result_naive['fn']:>12}")
    print(f"{'mean cost/step':>20}  {result_bayes['mean_cost']:>12.4f}  {result_naive['mean_cost']:>12.4f}")

    if result_naive["mean_cost"] > 0:
        cost_reduction = 1 - result_bayes["mean_cost"] / result_naive["mean_cost"]
        print(f"\nMean cost reduction using the derived threshold vs. naive 0.5: "
              f"{cost_reduction:+.1%}")
    else:
        print("\nNaive threshold's cost is exactly 0 on this data -- cost reduction "
              "ratio is undefined (nothing to improve on). Compare absolute mean costs instead.")

    # --- Sweep across thresholds for a visual check ---
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = [expected_cost(y_true, y_pred, t, cost_fp, cost_fn)["mean_cost"] for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, costs, lw=1.5)
    ax.axvline(p_star, color="red", linestyle="--", label=f"derived p*={p_star:.3f}")
    ax.axvline(0.5, color="gray", linestyle="--", label="naive p=0.5")
    ax.axvspan(*MISCALIBRATED_RANGE, alpha=0.15, color="orange",
               label="known miscalibrated range")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("mean expected cost per step")
    ax.set_title("Expected cost vs. decision threshold (raw probabilities)")
    ax.legend()
    fig.tight_layout()
    out_path = PLOTS_DIR / "bayes_threshold_cost_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    print("\nCheck before trusting this result:")
    print("- Is the cost curve's minimum actually near p*, or does the empirical")
    print("  minimum sit somewhere else? p* is theoretically optimal only if the")
    print("  probabilities are well-calibrated -- given they are NOT (documented),")
    print("  the empirical minimum on this data may differ from the theoretical p*,")
    print("  and that gap is itself informative about the calibration issue's impact.")
    print("- Report the cost reduction honestly alongside the miscalibration caveat --")
    print("  a positive result here is suggestive, not a fully validated conclusion.")


if __name__ == "__main__":
    main()
