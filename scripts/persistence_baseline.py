"""
Persistence and HAR-style baselines for the real label.

Question this answers (the one a quant screener will ask first):
how much of the ~0.91 out-of-sample AUC is just volatility persistence?
The label is a forward average of elevated_prob, and elevated_prob is
also the main feature, so a model that only says "the current regime
continues" may already score very high.

Compares, under the SAME purged, embargoed walk-forward splits used
everywhere else in this project (src/cv_splits.py):

  1. persistence (no fitting)  -- score = current elevated_prob
  2. persistence, binary       -- score = 1[elevated_prob > 0.5]
  3. HAR-style LR              -- logistic regression on 1h vol and its
                                  4h and 24h trailing means (Corsi-style
                                  short/medium/long volatility components)
  4. project LR (3 features)   -- elevated_prob, its lag, rolling_vol_1h
  5. dummy (class prior)       -- feature-blind reference, as in
                                  phase2_real_vs_dummy_comparison.py

and reports each model's AUC and its margin over persistence. If the
project LR's margin over persistence is small, that is the honest result
and belongs on the CV as such.

Place this file in scripts/ and run from the repo root:
    python scripts/persistence_baseline.py [path_to_dataset.parquet]
    # defaults to data/processed/phase2_dataset.parquet (LUNA window)
    # also run it on the FTX dataset and the Phase 4 merged dataset

HAR features look back 24h, longer than the 60-step embargo, so the HAR
run uses a 1440-step embargo to stay conservative. Persistence needs no
training, so it is scored on exactly the same test rows as the LR.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cv_splits import purged_embargoed_walk_forward_splits

DEFAULT_DATA_PATH = Path("data/processed/phase2_dataset.parquet")
PROJECT_FEATURES = ["elevated_prob", "elevated_prob_lag1", "rolling_vol_1h"]
HAR_FEATURES = ["har_1h", "har_4h", "har_24h"]
LABEL = "real_label"
H = 60
EMBARGO = 60
EMBARGO_HAR = 1440


def add_har_features(df):
    """Causal HAR-style volatility components from the 1h rolling vol."""
    out = df.copy()
    vol = out["rolling_vol_1h"]
    out["har_1h"] = vol
    out["har_4h"] = vol.rolling(240, min_periods=240).mean()
    out["har_24h"] = vol.rolling(1440, min_periods=1440).mean()
    return out


def oof_scores(df, score_fn, embargo):
    """Out-of-fold scores under purged walk-forward. score_fn(train, test)
    returns test scores, or None to skip a fold."""
    oof = np.full(len(df), np.nan)
    for train_idx, test_idx in purged_embargoed_walk_forward_splits(df.index, h=H, embargo=embargo):
        s = score_fn(df.iloc[train_idx], df.iloc[test_idx])
        if s is not None:
            oof[test_idx] = s
    return oof


def lr_scorer(cols):
    def fn(train, test):
        y = train[LABEL].to_numpy()
        if len(np.unique(y)) < 2:
            return None
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(train[cols].to_numpy(), y)
        return model.predict_proba(test[cols].to_numpy())[:, 1]
    return fn


def persistence_scorer(train, test):
    if len(np.unique(train[LABEL])) < 2:
        return None
    return test["elevated_prob"].to_numpy()


def persistence_binary_scorer(train, test):
    if len(np.unique(train[LABEL])) < 2:
        return None
    return (test["elevated_prob"].to_numpy() > 0.5).astype(float)


def dummy_scorer(train, test):
    y = train[LABEL].to_numpy()
    if len(np.unique(y)) < 2:
        return None
    return np.full(len(test), y.mean())


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    if not path.exists():
        sys.exit(f"{path} not found -- run scripts/build_phase2_dataset.py first")

    df = pd.read_parquet(path)
    df = add_har_features(df)
    needed = PROJECT_FEATURES + HAR_FEATURES + [LABEL]
    df = df.dropna(subset=needed)
    print(f"Dataset: {path}  rows after dropping warm-up/NaN: {len(df):,}")
    print(f"Label positive rate: {df[LABEL].mean():.3f}\n")

    runs = {
        "persistence (current elevated_prob)": (persistence_scorer, EMBARGO),
        "persistence, binary (> 0.5)": (persistence_binary_scorer, EMBARGO),
        "HAR-style LR (1h/4h/24h vol)": (lr_scorer(HAR_FEATURES), EMBARGO_HAR),
        "project LR (3 features)": (lr_scorer(PROJECT_FEATURES), EMBARGO),
        "dummy (class prior)": (dummy_scorer, EMBARGO),
    }

    oofs = {name: oof_scores(df, fn, emb) for name, (fn, emb) in runs.items()}

    # Score every model on the rows that ALL models covered, so the AUCs
    # are directly comparable (the HAR run's longer embargo drops a few).
    common = np.ones(len(df), dtype=bool)
    for s in oofs.values():
        common &= ~np.isnan(s)
    y = df[LABEL].to_numpy()[common]
    print(f"Rows scored by every model: {common.sum():,}\n")

    aucs = {name: roc_auc_score(y, s[common]) for name, s in oofs.items()}
    base = aucs["persistence (current elevated_prob)"]
    print(f"{'model':<40}{'AUC':>8}{'vs persistence':>17}")
    for name, auc in aucs.items():
        print(f"{name:<40}{auc:>8.4f}{auc - base:>+17.4f}")

    print("\nRead this as: the CV line should quote the project LR's margin over")
    print("persistence (and over HAR), not the raw AUC. A small or negative margin")
    print("is a real, reportable result, consistent with the rest of the project.")


if __name__ == "__main__":
    main()
