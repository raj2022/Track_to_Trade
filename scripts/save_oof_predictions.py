"""
Extracts and saves the purged/embargoed walk-forward out-of-fold predicted
probabilities and true labels for real_label -- the one result Phase 2
validated as carrying genuine signal (~6 sigma above the null-shift-
calibrated artifact baseline). Everything in Phase 3 (calibration,
reliability diagrams, the Bayes-risk threshold) builds on these saved
predictions rather than re-running the CV loop each time.

Usage:
    python scripts/save_oof_predictions.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cv_splits import purged_embargoed_walk_forward_splits

DATA_PATH = Path("data/processed/phase2_dataset.parquet")
OUT_PATH = Path("data/processed/real_label_oof_predictions.parquet")
FEATURE_COLS = ["elevated_prob", "elevated_prob_lag1", "rolling_vol_1h"]
H = 60
EMBARGO = 60


def main():
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} not found -- run scripts/build_phase2_dataset.py first")

    df = pd.read_parquet(DATA_PATH)
    X = df[FEATURE_COLS].to_numpy()
    y = df["real_label"].to_numpy()

    oof_pred = np.full(len(df), np.nan)
    n_folds = 0

    for train_idx, test_idx in purged_embargoed_walk_forward_splits(df.index, h=H, embargo=EMBARGO):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[train_idx], y_train)
        oof_pred[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        n_folds += 1

    valid = ~np.isnan(oof_pred)
    result = pd.DataFrame({
        "y_true": y[valid],
        "y_pred_proba": oof_pred[valid],
    }, index=df.index[valid])

    print(f"{n_folds} folds, {len(result):,} OOF predictions saved")
    print(f"Positive rate (true): {result['y_true'].mean():.3f}")
    print(f"Predicted probability range: [{result['y_pred_proba'].min():.4f}, "
          f"{result['y_pred_proba'].max():.4f}]")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
