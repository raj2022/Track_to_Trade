"""
Fit a Gaussian HMM (via EM/Baum-Welch) to the 2022-05 return series, select
the number of regimes K by BIC rather than asserting K=2, and extract the
per-state variances (-> IMM regime Q's) and transition matrix (-> IMM
mixing probabilities).

Usage:
    python scripts/fit_regime_hmm.py data/raw/BTCUSDT-aggTrades-2022-05.zip
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

COLS = ["aggTradeId", "price", "quantity", "firstTradeId", "lastTradeId",
        "timestamp", "isBuyerMaker", "isBestMatch"]

DT_SEC = 60  # validated sampling interval, from the Q derivation
K_CANDIDATES = [2, 3, 4, 5]
N_RESTARTS = 5  # EM can converge to local optima -- restart and keep the best
RANDOM_SEED = 0


def load_trades(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        inner_name = z.namelist()[0]
        with z.open(inner_name) as f:
            df = pd.read_csv(f, header=None, names=COLS,
                              usecols=["price", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def build_returns(price: pd.Series, dt_sec: int) -> tuple[np.ndarray, float]:
    """Returns (standardized returns, scale). Returns at this dt are tiny in
    magnitude (~1e-4), which caused real numerical instability in diagonal-
    covariance EM (hmmlearn's own 'Model is not converging' warnings fired
    on every K). Standardizing before fitting and rescaling afterward is the
    standard fix -- addresses the precision problem directly rather than
    just increasing n_iter and hoping the instability goes away on its own."""
    resampled = price.resample(f"{dt_sec}s").last().ffill()
    log_ret = np.log(resampled).diff().dropna()
    raw = log_ret.to_numpy()
    scale = raw.std()
    standardized = (raw / scale).reshape(-1, 1)
    return standardized, scale


def fit_best_hmm(returns: np.ndarray, k: int, n_restarts: int, seed: int):
    best_model = None
    best_ll = -np.inf
    for i in range(n_restarts):
        model = GaussianHMM(
            n_components=k, covariance_type="diag",
            n_iter=500, tol=1e-6, random_state=seed + i,
        )
        model.fit(returns)
        ll = model.score(returns)
        if ll > best_ll:
            best_ll = ll
            best_model = model
    return best_model, best_ll


def bic(log_likelihood: float, n_params: int, n_obs: int) -> float:
    return -2 * log_likelihood + n_params * np.log(n_obs)


def count_params(k: int) -> int:
    # k means + k variances (diag covariance) + k*(k-1) free transition
    # probs (rows sum to 1) + (k-1) free initial-state probs
    return k + k + k * (k - 1) + (k - 1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python fit_regime_hmm.py <path_to_zip>")
        sys.exit(1)

    df = load_trades(Path(sys.argv[1]))
    returns, scale = build_returns(df["price"], DT_SEC)
    n_obs = len(returns)
    print(f"Fitting on {n_obs:,} standardized returns (dt={DT_SEC}s, scale={scale:.6e})")

    print(f"\n{'K':>3}  {'log-likelihood':>16}  {'n_params':>9}  {'BIC':>14}")
    results = {}
    for k in K_CANDIDATES:
        model, ll = fit_best_hmm(returns, k, N_RESTARTS, RANDOM_SEED)
        n_params = count_params(k)
        bic_val = bic(ll, n_params, n_obs)
        results[k] = (model, ll, bic_val)
        print(f"  {k:>3}  {ll:>16.2f}  {n_params:>9}  {bic_val:>14.2f}")

    best_k = min(results, key=lambda k: results[k][2])
    print(f"\nBIC-selected K = {best_k}")

    model, ll, bic_val = results[best_k]
    variances = model.covars_.flatten() * (scale ** 2)   # rescale to real units
    means = model.means_.flatten() * scale                # rescale to real units
    order = np.argsort(variances)  # sort states calm -> volatile for readability

    print(f"\nRegime variances (per-step, dt={DT_SEC}s, rescaled to real return units), "
          f"sorted calm -> volatile:")
    for rank, state in enumerate(order):
        print(f"  state {state} (rank {rank}): variance = {variances[state]:.6e}, "
              f"mean = {means[state]:.6e}")

    print(f"\nTransition matrix (rows = from-state, cols = to-state), original state order:")
    print(np.array2string(model.transmat_, precision=4, suppress_small=True))

    # Decode the most likely state sequence and plot against the known event.
    states = model.predict(returns)
    resampled = df["price"].resample(f"{DT_SEC}s").last().ffill()
    log_price = np.log(resampled).iloc[1:]  # align with returns (one fewer obs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(log_price.index, log_price.values, lw=0.7)
    ax1.set_ylabel("log(price)")
    ax1.set_title(f"BTCUSDT 2022-05 with HMM-decoded states (K={best_k})")

    ax2.scatter(log_price.index, states, s=2)
    ax2.axvline(pd.Timestamp("2022-05-09", tz="UTC"), color="black",
                linestyle="--", alpha=0.6, label="LUNA depeg")
    ax2.set_ylabel("decoded state")
    ax2.set_xlabel("Date")
    ax2.legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = f"hmm_states_2022-05_K{best_k}.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    print("\nCheck before trusting this fit:")
    print("- Do the highest-variance state(s) concentrate visibly around")
    print("  2022-05-09 through -13, or are they scattered? Scattered would")
    print("  suggest the HMM is fitting noise, not a real regime structure.")
    print("- Compare the BIC gap between the selected K and its neighbors --")
    print("  a razor-thin margin means the model-selection result is fragile")
    print("  and worth treating with more caution than a wide margin would.")


if __name__ == "__main__":
    main()