"""
Honest comparison: logistic regression vs. small MLPs (candidate hidden
sizes 16/32/64, smallest-adequate chosen rather than assumed) on the
COMBINED, regime-diverse dataset -- the real test of whether a single
month's homogeneous data was the reason a simple model previously looked
sufficient, or whether it genuinely is sufficient regardless of diversity.

Same purged/embargoed walk-forward discipline as every other evaluation
in this project. Same 3 features as the original logistic regression
baseline, for a fair apples-to-apples comparison -- not a bigger feature
set AND a bigger model at once, which would conflate two questions.

Usage:
    python scripts/compare_lr_mlp_combined.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cv_splits import purged_embargoed_walk_forward_splits

DATA_PATH = Path("data/processed/phase2_dataset_combined.parquet")
FEATURE_COLS = ["elevated_prob", "elevated_prob_lag1", "rolling_vol_1h"]
H = 60
EMBARGO = 60
MLP_HIDDEN_SIZES = [16, 32, 64]
EPS = 1e-12


def log_score(y_true, y_pred):
    p = np.clip(y_pred, EPS, 1 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def build_model(name: str):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name.startswith("mlp_"):
        hidden = int(name.split("_")[1])
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=500,
                           early_stopping=True, n_iter_no_change=10,
                           random_state=0),
        )
    raise ValueError(name)


def run_cv(df: pd.DataFrame, model_name: str):
    X = df[FEATURE_COLS].to_numpy()
    y = df["real_label"].to_numpy()
    oof = np.full(len(df), np.nan)
    n_folds = 0

    for train_idx, test_idx in purged_embargoed_walk_forward_splits(df.index, h=H, embargo=EMBARGO):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        model = build_model(model_name)
        model.fit(X[train_idx], y_train)
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        n_folds += 1

    valid = ~np.isnan(oof)
    return y[valid], oof[valid], n_folds


def main():
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} not found -- run scripts/merge_multi_window_dataset.py first")

    df = pd.read_parquet(DATA_PATH).sort_index()
    print(f"Loaded {len(df):,} rows spanning {df.index.min()} to {df.index.max()}")

    model_names = ["logistic_regression"] + [f"mlp_{h}" for h in MLP_HIDDEN_SIZES]
    results = {}

    for name in model_names:
        start = time.time()
        y_true, y_pred, n_folds = run_cv(df, name)
        elapsed = time.time() - start
        auc = roc_auc_score(y_true, y_pred)
        ls = log_score(y_true, y_pred)
        results[name] = {"auc": auc, "log_score": ls, "n_folds": n_folds, "time_s": elapsed}
        print(f"  {name:>22}: AUC={auc:.4f}  log_score={ls:.4f}  "
              f"({n_folds} folds, {elapsed:.1f}s)")

    print(f"\n{'model':>22}  {'AUC':>8}  {'log score':>10}  {'delta AUC vs LR':>16}")
    lr_auc = results["logistic_regression"]["auc"]
    for name, r in results.items():
        delta = r["auc"] - lr_auc
        print(f"  {name:>22}  {r['auc']:>8.4f}  {r['log_score']:>10.4f}  {delta:>+16.4f}")

    best_mlp = min((n for n in model_names if n.startswith("mlp_")),
                    key=lambda n: -results[n]["auc"])
    print(f"\nBest MLP by AUC: {best_mlp} (AUC={results[best_mlp]['auc']:.4f} "
          f"vs. logistic regression's {lr_auc:.4f})")

    print("\nDecide, don't assume:")
    print("- Is ANY MLP's AUC meaningfully above logistic regression's, or are")
    print("  differences within noise given this is a single walk-forward run")
    print("  (no permutation-null check here -- treat small deltas skeptically)?")
    print("- If an MLP does win, pick the SMALLEST hidden size that wins, not the")
    print("  largest -- that's the actual 'smallest thing that works' derivation,")
    print("  not just reporting whichever number is biggest.")
    print("- If NO MLP beats logistic regression, that's a legitimate, reportable")
    print("  result: added model complexity didn't earn its keep even with genuine")
    print("  regime diversity in the training data -- the compression question")
    print("  this was meant to set up becomes moot, and that's fine to say plainly.")


if __name__ == "__main__":
    main()
