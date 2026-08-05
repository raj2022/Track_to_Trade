"""
Decisive follow-up to the leakage stress test's puzzling result: purged
walk-forward showed a CONSISTENT positive AUC bias across all 20 null
shifts, while naive k-fold was noisy but roughly unbiased in aggregate --
backwards from the leakage hypothesis this test was built to check.

A synthetic control (see conversation/notes) showed that a completely
FEATURE-BLIND classifier -- one that predicts only the training fold's
class prior, using no features whatsoever -- ALSO produces large, non-
chance AUC deviations under both schemes, purely from the interaction
between an autocorrelated label's own serial structure and the CV
splitting mechanics. That means AUC-above-0.5 alone does not indicate
feature-driven skill for this kind of label.

This script settles the real question directly: for each null shift, does
the actual logistic regression (using real features) achieve MORE than
the trivial prior-only baseline? Reports real AUC, dummy AUC, and the
DIFFERENCE per shift/scheme -- the difference, not either AUC alone, is
the number that actually answers "are the features contributing anything."

Usage:
    python scripts/phase2_real_vs_dummy_comparison.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cv_splits import naive_kfold_splits, purged_embargoed_walk_forward_splits

DATA_PATH = Path("data/processed/phase2_dataset.parquet")
FEATURE_COLS = ["elevated_prob", "elevated_prob_lag1", "rolling_vol_1h"]
H = 60
EMBARGO = 60


def run_real_cv(df, label_col, split_fn, **kwargs):
    X = df[FEATURE_COLS].to_numpy()
    y = df[label_col].to_numpy()
    oof = np.full(len(df), np.nan)
    n_folds = 0
    for train_idx, test_idx in split_fn(df.index, **kwargs):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[train_idx], y_train)
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        n_folds += 1
    valid = ~np.isnan(oof)
    return y[valid], oof[valid], n_folds


def run_dummy_cv(df, label_col, split_fn, **kwargs):
    """Predicts only the training fold's class prior -- structurally blind
    to all features. The trivial 'has the recent past been elevated'
    baseline."""
    y = df[label_col].to_numpy()
    oof = np.full(len(df), np.nan)
    n_folds = 0
    for train_idx, test_idx in split_fn(df.index, **kwargs):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        oof[test_idx] = y_train.mean()
        n_folds += 1
    valid = ~np.isnan(oof)
    return y[valid], oof[valid], n_folds


def main():
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} not found -- run scripts/build_phase2_dataset.py first")

    df = pd.read_parquet(DATA_PATH)
    null_cols = sorted(c for c in df.columns if c.startswith("null_label_"))
    print(f"Comparing real vs. dummy classifier across {len(null_cols)} null shifts...\n")

    rows = []
    for col in null_cols:
        y_true, pred_real, _ = run_real_cv(df, col, naive_kfold_splits, n_splits=30)
        auc_real_naive = roc_auc_score(y_true, pred_real)
        y_true, pred_dummy, _ = run_dummy_cv(df, col, naive_kfold_splits, n_splits=30)
        auc_dummy_naive = roc_auc_score(y_true, pred_dummy)

        y_true, pred_real, _ = run_real_cv(df, col, purged_embargoed_walk_forward_splits, h=H, embargo=EMBARGO)
        auc_real_purged = roc_auc_score(y_true, pred_real)
        y_true, pred_dummy, _ = run_dummy_cv(df, col, purged_embargoed_walk_forward_splits, h=H, embargo=EMBARGO)
        auc_dummy_purged = roc_auc_score(y_true, pred_dummy)

        rows.append({
            "shift": col,
            "naive_real": auc_real_naive, "naive_dummy": auc_dummy_naive,
            "naive_diff": auc_real_naive - auc_dummy_naive,
            "purged_real": auc_real_purged, "purged_dummy": auc_dummy_purged,
            "purged_diff": auc_real_purged - auc_dummy_purged,
        })
        print(f"{col:>14}  naive: real={auc_real_naive:.3f} dummy={auc_dummy_naive:.3f} "
              f"diff={rows[-1]['naive_diff']:+.3f}   "
              f"purged: real={auc_real_purged:.3f} dummy={auc_dummy_purged:.3f} "
              f"diff={rows[-1]['purged_diff']:+.3f}")

    summary = pd.DataFrame(rows)
    print(f"\n{'':>20}  {'naive_kfold':>15}  {'purged_walk_forward':>20}")
    print(f"{'mean real AUC':>20}  {summary['naive_real'].mean():>15.4f}  {summary['purged_real'].mean():>20.4f}")
    print(f"{'mean dummy AUC':>20}  {summary['naive_dummy'].mean():>15.4f}  {summary['purged_dummy'].mean():>20.4f}")
    print(f"{'mean diff (real-dummy)':>20}  {summary['naive_diff'].mean():>+15.4f}  {summary['purged_diff'].mean():>+20.4f}")
    print(f"{'std of diff':>20}  {summary['naive_diff'].std():>15.4f}  {summary['purged_diff'].std():>20.4f}")

    print("\nTHE ANSWER TO THE ACTUAL QUESTION:")
    print("- 'mean diff' near zero, in EITHER scheme, means the logistic regression's")
    print("  features are contributing NOTHING beyond what a trivial 'recent history'")
    print("  baseline already achieves on this null label -- the earlier apparent")
    print("  purged_walk_forward bias was a property of the LABEL'S OWN AUTOCORRELATION")
    print("  interacting with expanding-window evaluation, not feature-driven leakage.")
    print("- A clearly positive mean diff under naive_kfold but not purged_walk_forward")
    print("  would be the ORIGINAL predicted leakage signature, now correctly isolated")
    print("  from the label-autocorrelation confound.")
    print("- Compare 'std of diff' to 'mean diff' -- if std swamps the mean, no firm")
    print("  conclusion should be drawn from this sample of shifts either way.")


if __name__ == "__main__":
    main()