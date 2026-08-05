"""
The centerpiece leakage stress test: run a simple classifier through all
four combinations of {real label, null label} x {naive k-fold, purged
embargoed walk-forward}, and check whether the predicted pattern holds --
naive CV inflates apparent performance on BOTH labels but especially the
null (zero true signal, so any apparent AUC there is unambiguous evidence
of leakage), while purged walk-forward should collapse the null's AUC to
chance and give an honest read on the real label.

Usage:
    python scripts/phase2_leakage_stress_test.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cv_splits import naive_kfold_splits, purged_embargoed_walk_forward_splits

DATA_PATH = Path("data/processed/phase2_dataset.parquet")
PLOTS_DIR = Path("plots")
FEATURE_COLS = ["elevated_prob", "elevated_prob_lag1", "rolling_vol_1h"]
H = 60
EMBARGO = 60
N_PERMUTATIONS = 1000
SEED = 0


def run_cv(df: pd.DataFrame, label_col: str, split_fn, **split_kwargs):
    """Fits a scaled logistic regression per fold, collects out-of-fold
    predicted probabilities and true labels across ALL folds (not per-fold
    AUC, which can be undefined on small/single-class daily test folds)."""
    X = df[FEATURE_COLS].to_numpy()
    y = df[label_col].to_numpy()

    oof_pred = np.full(len(df), np.nan)
    n_folds = 0

    for train_idx, test_idx in split_fn(df.index, **split_kwargs):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue  # can't fit a classifier on a single-class training fold
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[train_idx], y_train)
        oof_pred[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        n_folds += 1

    valid = ~np.isnan(oof_pred)
    return y[valid], oof_pred[valid], n_folds


def permutation_null_auc(y_true: np.ndarray, y_pred: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    """Cheap permutation null: shuffle the true-label/prediction pairing
    (no refitting) and recompute AUC each time. Tests 'what AUC would this
    exact set of predictions achieve against a randomly relabeled truth,'
    i.e. the AUC achievable by chance alone given this base rate."""
    rng = np.random.default_rng(seed)
    null_aucs = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(y_true)
        null_aucs[i] = roc_auc_score(shuffled, y_pred)
    return null_aucs


def evaluate(df: pd.DataFrame, label_col: str, scheme_name: str, split_fn, **split_kwargs) -> dict:
    y_true, y_pred, n_folds = run_cv(df, label_col, split_fn, **split_kwargs)
    observed_auc = roc_auc_score(y_true, y_pred)
    null_aucs = permutation_null_auc(y_true, y_pred, N_PERMUTATIONS, SEED)
    percentile = float((null_aucs < observed_auc).mean() * 100)
    z_score = (observed_auc - null_aucs.mean()) / null_aucs.std()

    print(f"\n[{label_col} x {scheme_name}] {n_folds} folds, {len(y_true):,} OOF predictions")
    print(f"  Observed AUC: {observed_auc:.4f}")
    print(f"  Permutation null: mean={null_aucs.mean():.4f}, "
          f"std={null_aucs.std():.4f}, 95th pct={np.percentile(null_aucs, 95):.4f}")
    print(f"  Observed AUC is at the {percentile:.1f}th percentile of the null distribution "
          f"({z_score:+.2f} std above the null mean -- USE THIS for comparing magnitude across "
          f"combinations, since percentile saturates at 100% for any real effect once N is large)")

    return {"label": label_col, "scheme": scheme_name, "observed_auc": observed_auc,
            "null_aucs": null_aucs, "percentile": percentile, "z_score": z_score, "n_folds": n_folds}


def main():
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} not found -- run scripts/build_phase2_dataset.py first")

    PLOTS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df):,} rows")

    # Match naive k-fold's split count to the walk-forward fold count, so
    # neither scheme is compared unfairly against a different test-set
    # granularity.
    n_wf_folds = sum(1 for _ in purged_embargoed_walk_forward_splits(df.index, h=H, embargo=EMBARGO))
    print(f"Walk-forward produces {n_wf_folds} folds -- matching naive k-fold's split count to this.")

    results = []
    results.append(evaluate(df, "real_label", "naive_kfold", naive_kfold_splits, n_splits=n_wf_folds))
    results.append(evaluate(df, "real_label", "purged_walk_forward", purged_embargoed_walk_forward_splits, h=H, embargo=EMBARGO))

    null_cols = [c for c in df.columns if c.startswith("null_label_")]
    print(f"\nEvaluating {len(null_cols)} independent null-label shifts under each CV scheme "
          f"(reporting the DISTRIBUTION across shifts, not a single number -- a single shift "
          f"can still land badly by chance).")

    null_naive_aucs, null_naive_z = [], []
    null_purged_aucs, null_purged_z = [], []
    for col in null_cols:
        r_naive = evaluate(df, col, "naive_kfold", naive_kfold_splits, n_splits=n_wf_folds)
        r_purged = evaluate(df, col, "purged_walk_forward", purged_embargoed_walk_forward_splits, h=H, embargo=EMBARGO)
        null_naive_aucs.append(r_naive["observed_auc"])
        null_naive_z.append(r_naive["z_score"])
        null_purged_aucs.append(r_purged["observed_auc"])
        null_purged_z.append(r_purged["z_score"])

    null_naive_aucs, null_naive_z = np.array(null_naive_aucs), np.array(null_naive_z)
    null_purged_aucs, null_purged_z = np.array(null_purged_aucs), np.array(null_purged_z)

    print(f"\n{'':>12}  {'scheme':>20}  {'AUC (mean+/-std)':>20}  {'z-score (mean+/-std)':>22}")
    print(f"  {'real_label':>12}  {'naive_kfold':>20}  "
          f"{results[0]['observed_auc']:>20.4f}  {results[0]['z_score']:>22.2f}")
    print(f"  {'real_label':>12}  {'purged_walk_forward':>20}  "
          f"{results[1]['observed_auc']:>20.4f}  {results[1]['z_score']:>22.2f}")
    print(f"  {'null_label':>12}  {'naive_kfold':>20}  "
          f"{null_naive_aucs.mean():>10.4f} +/- {null_naive_aucs.std():<6.4f}  "
          f"{null_naive_z.mean():>10.2f} +/- {null_naive_z.std():<8.2f}")
    print(f"  {'null_label':>12}  {'purged_walk_forward':>20}  "
          f"{null_purged_aucs.mean():>10.4f} +/- {null_purged_aucs.std():<6.4f}  "
          f"{null_purged_z.mean():>10.2f} +/- {null_purged_z.std():<8.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(null_naive_aucs, bins=15, alpha=0.7, color="tab:red", label="naive_kfold")
    axes[0].hist(null_purged_aucs, bins=15, alpha=0.7, color="tab:blue", label="purged_walk_forward")
    axes[0].axvline(0.5, color="black", linestyle="--", label="chance")
    axes[0].set_xlabel("AUC across null shifts")
    axes[0].set_title("Null-label AUC distribution by CV scheme")
    axes[0].legend(fontsize=8)

    axes[1].boxplot([null_naive_z, null_purged_z], tick_labels=["naive_kfold", "purged_walk_forward"])
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set_ylabel("z-score vs. permutation null (per shift)")
    axes[1].set_title("Null-label z-score spread by CV scheme")

    fig.tight_layout()
    out_path = PLOTS_DIR / "phase2_leakage_stress_test.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    print("\nWhat to check, given the null construction had to be fixed twice already --")
    print("read this carefully rather than trusting a single number:")
    print("- Is null_label's mean AUC now close to 0.5 under BOTH schemes, with the naive")
    print("  scheme's mean sitting HIGHER than purged's (even modestly)? That's the actual,")
    print("  now-credible leakage signature.")
    print("- Is the SPREAD (std) across shifts reasonable, or still large enough that no firm")
    print("  conclusion should be drawn? A wide spread means more shifts are needed, not that")
    print("  the result should be reported as settled.")
    print("- real_label's naive vs. purged AUC gap remains small (as before) -- report this")
    print("  honestly alongside the null result, don't let a clean null story overshadow that")
    print("  the 'real' signal here is close to tautological given the feature construction.")


if __name__ == "__main__":
    main()