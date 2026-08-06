"""
Same comparison as compare_lr_mlp_combined.py, but the MLPs are now
PyTorch models trained with Apple Silicon (MPS) acceleration when
available, falling back to CPU otherwise. sklearn's MLPClassifier has no
GPU path at all -- this switch is what makes real hardware-aware timing
possible, and sets up Phase 4's latency/compression study honestly (this
machine has Metal, not CUDA, so benchmarking against MPS is the actual
relevant comparison, not a borrowed CUDA number).

Usage:
    python scripts/compare_lr_mlp_torch_mps.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
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
MAX_EPOCHS = 100
PATIENCE = 10  # early stopping: stop if val loss doesn't improve for this many epochs
EPS = 1e-12

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def log_score(y_true, y_pred):
    p = np.clip(y_pred, EPS, 1 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, hidden: int) -> MLP:
    """Trains one MLP with early stopping on a chronological validation
    tail carved from the training fold (never touches the actual test
    fold) -- same walk-forward discipline, one level down."""
    n = len(X_train)
    val_split = max(int(n * 0.85), n - 2000)  # last ~15% (capped) as validation
    X_tr, X_val = X_train[:val_split], X_train[val_split:]
    y_tr, y_val = y_train[:val_split], y_train[val_split:]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
        # fall back to using all training data for both fit and early-stop
        # check if the tail split happens to be single-class
        X_tr, X_val, y_tr, y_val = X_train, X_train, y_train, y_train

    scaler_mean = X_tr.mean(axis=0)
    scaler_std = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - scaler_mean) / scaler_std
    X_val = (X_val - scaler_mean) / scaler_std

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    model = MLP(X_train.shape[1], hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr_t)
        loss = loss_fn(logits, y_tr_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model._scaler_mean = scaler_mean
    model._scaler_std = scaler_std
    return model


def predict_mlp(model: MLP, X: np.ndarray) -> np.ndarray:
    X_scaled = (X - model._scaler_mean) / model._scaler_std
    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def run_cv_lr(df: pd.DataFrame):
    X = df[FEATURE_COLS].to_numpy()
    y = df["real_label"].to_numpy()
    oof = np.full(len(df), np.nan)
    n_folds = 0
    for train_idx, test_idx in purged_embargoed_walk_forward_splits(df.index, h=H, embargo=EMBARGO):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X[train_idx], y_train)
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        n_folds += 1
    valid = ~np.isnan(oof)
    return y[valid], oof[valid], n_folds


def run_cv_mlp(df: pd.DataFrame, hidden: int):
    X = df[FEATURE_COLS].to_numpy().astype(np.float32)
    y = df["real_label"].to_numpy().astype(np.float32)
    oof = np.full(len(df), np.nan)
    n_folds = 0
    for train_idx, test_idx in purged_embargoed_walk_forward_splits(df.index, h=H, embargo=EMBARGO):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        model = train_mlp(X[train_idx], y_train, hidden)
        oof[test_idx] = predict_mlp(model, X[test_idx])
        n_folds += 1
    valid = ~np.isnan(oof)
    return y[valid].astype(int), oof[valid], n_folds


def main():
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} not found -- run scripts/merge_multi_window_dataset.py first")

    print(f"Device: {DEVICE} "
          f"({'Apple Silicon MPS' if DEVICE.type == 'mps' else 'CPU fallback'})")

    df = pd.read_parquet(DATA_PATH).sort_index()
    print(f"Loaded {len(df):,} rows spanning {df.index.min()} to {df.index.max()}")

    results = {}

    start = time.time()
    y_true, y_pred, n_folds = run_cv_lr(df)
    elapsed = time.time() - start
    results["logistic_regression"] = {
        "auc": roc_auc_score(y_true, y_pred), "log_score": log_score(y_true, y_pred),
        "n_folds": n_folds, "time_s": elapsed,
    }
    print(f"  {'logistic_regression':>22}: AUC={results['logistic_regression']['auc']:.4f}  "
          f"({n_folds} folds, {elapsed:.1f}s)")

    for hidden in MLP_HIDDEN_SIZES:
        name = f"mlp_{hidden}_{DEVICE.type}"
        start = time.time()
        y_true, y_pred, n_folds = run_cv_mlp(df, hidden)
        elapsed = time.time() - start
        results[name] = {
            "auc": roc_auc_score(y_true, y_pred), "log_score": log_score(y_true, y_pred),
            "n_folds": n_folds, "time_s": elapsed,
        }
        print(f"  {name:>22}: AUC={results[name]['auc']:.4f}  "
              f"log_score={results[name]['log_score']:.4f}  ({n_folds} folds, {elapsed:.1f}s)")

    print(f"\n{'model':>22}  {'AUC':>8}  {'log score':>10}  {'delta AUC vs LR':>16}  {'wall time':>10}")
    lr_auc = results["logistic_regression"]["auc"]
    for name, r in results.items():
        delta = r["auc"] - lr_auc
        print(f"  {name:>22}  {r['auc']:>8.4f}  {r['log_score']:>10.4f}  "
              f"{delta:>+16.4f}  {r['time_s']:>9.1f}s")

    print(f"\nTraining device used: {DEVICE} -- total MLP wall time across all "
          f"{len(MLP_HIDDEN_SIZES)} sizes: "
          f"{sum(r['time_s'] for n, r in results.items() if 'mlp' in n):.1f}s")
    print("\nDecide, don't assume:")
    print("- Is ANY MLP's AUC meaningfully above logistic regression's?")
    print("- If an MLP wins, is it the SMALLEST hidden size that does, not the largest?")
    print("- Note the wall-clock time per size -- this is the real, measured basis for")
    print("  Phase 4's latency discussion, not an assumed number.")


if __name__ == "__main__":
    main()
