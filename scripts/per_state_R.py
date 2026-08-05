"""
Compute Roll's estimator SEPARATELY for each of the K=5 HMM-decoded states,
using only tick-to-tick return pairs where both endpoints (and, for lag-1
covariance, the preceding return too) fall in bins assigned to the same
state. This avoids gluing five variance levels onto only two R estimates
(calm/crisis) by fiat.

Then computes Q_k = HMM_variance_k - 2*R_k for each state, since a Kalman
observation model (z_t = x_t + v_t) implies Var(return) ~= Q + 2R, not Q
alone -- the HMM's fitted per-state variance conflates process and
measurement noise together.

Usage:
    python scripts/per_state_R.py data/raw/BTCUSDT-aggTrades-2022-05.zip
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_regime_hmm import build_returns, fit_best_hmm, DT_SEC

COLS = ["aggTradeId", "price", "quantity", "firstTradeId", "lastTradeId",
        "timestamp", "isBuyerMaker", "isBestMatch"]

K = 5           # from the BIC-selected fit
N_RESTARTS = 5  # must match fit_regime_hmm.py's settings to reproduce the same fit
SEED = 0

# HMM per-state variances from the actual K=5 run (real return units, from
# fit_regime_hmm.py's printed output) -- used for the Q = var - 2R decomposition.
HMM_VARIANCES = {
    0: 2.326664e-08,
    3: 2.061623e-07,
    4: 4.792035e-07,
    1: 1.731108e-06,
    2: 1.002155e-05,
}


def load_trades(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        inner_name = z.namelist()[0]
        with z.open(inner_name) as f:
            df = pd.read_csv(f, header=None, names=COLS,
                              usecols=["price", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def assign_tick_states(tick_index: pd.DatetimeIndex, state_series: pd.Series, dt_sec: int) -> np.ndarray:
    """
    Map each tick to the decoded state of the dt-second bin it falls in,
    via integer bin arithmetic (fast, avoids a per-tick dict lookup over
    tens of millions of rows). Ticks before the first valid bin get -1.
    """
    start_time = state_series.index[0]
    elapsed_sec = (tick_index - start_time).total_seconds()
    bin_num = np.floor(elapsed_sec / dt_sec).astype(int)

    states_array = state_series.to_numpy()
    n_bins = len(states_array)

    state_per_tick = np.full(len(tick_index), -1, dtype=int)
    valid = (bin_num >= 0) & (bin_num < n_bins)
    state_per_tick[valid] = states_array[bin_num[valid]]
    return state_per_tick


def roll_R_for_state(prices: np.ndarray, state_per_tick: np.ndarray, k: int) -> tuple[float, int]:
    """
    Roll's estimator restricted to "pure" state-k return pairs: both ticks
    of each return, AND both of two consecutive returns, must fall in
    state-k bins. Returns (R_k, n_pairs_used).
    """
    log_p = np.log(prices)
    returns = np.diff(log_p)  # returns[i] = log_p[i+1] - log_p[i]

    pure = (state_per_tick[:-1] == k) & (state_per_tick[1:] == k)  # aligned with `returns`
    valid_lag1 = pure[1:] & pure[:-1]  # both returns[i] and returns[i+1] pure

    r_t = returns[1:][valid_lag1]
    r_tm1 = returns[:-1][valid_lag1]
    n_pairs = len(r_t)

    if n_pairs < 100:
        return float("nan"), n_pairs

    cov = float(np.cov(r_t, r_tm1, ddof=1)[0, 1])
    return -cov, n_pairs


def main():
    if len(sys.argv) != 2:
        print("Usage: python per_state_R.py <path_to_zip>")
        sys.exit(1)

    df = load_trades(Path(sys.argv[1]))
    print(f"Loaded {len(df):,} trades")

    returns, scale = build_returns(df["price"], DT_SEC)
    print(f"Refitting K={K} HMM (must match fit_regime_hmm.py's run to reproduce "
          f"the same state labeling)...")
    model, ll = fit_best_hmm(returns, K, N_RESTARTS, SEED)
    states = model.predict(returns)

    resampled = df["price"].resample(f"{DT_SEC}s").last().ffill()
    log_price = np.log(resampled)
    state_index = log_price.diff().dropna().index  # aligns with `states`
    state_series = pd.Series(states, index=state_index)

    print("Mapping ticks to bin states...")
    state_per_tick = assign_tick_states(df.index, state_series, DT_SEC)

    prices = df["price"].to_numpy()

    print(f"\n{'state':>6}  {'n_pairs':>10}  {'R_k':>14}  {'Q_k = var-2R':>14}  note")
    for k in sorted(HMM_VARIANCES.keys()):
        r_k, n_pairs = roll_R_for_state(prices, state_per_tick, k)
        var_k = HMM_VARIANCES[k]
        if np.isnan(r_k):
            print(f"  {k:>6}  {n_pairs:>10}  {'insufficient data':>14}  {'--':>14}  "
                  f"< 100 pure pairs, treat with caution")
            continue
        q_k = var_k - 2 * r_k
        note = ""
        if r_k < 0:
            note = "R_k < 0: Roll's model does not hold for this state"
        elif q_k < 0:
            note = "Q_k < 0: R may be overestimated, or var is noise-dominated here"
        print(f"  {k:>6}  {n_pairs:>10}  {r_k:>14.6e}  {q_k:>14.6e}  {note}")

    print("\nSanity check against the independently-derived calm/crisis R values:")
    print("  R (calm baseline, June)   = 3.661341e-11")
    print("  R (crisis, 05-09 to -12)  = 3.175645e-09")
    print("  The states with the most ticks during the June-like calm stretch")
    print("  and the 05-09/-12 crisis stretch should land in roughly the same")
    print("  ballpark as these two independent estimates -- if they don't,")
    print("  investigate before trusting the per-state Q values downstream.")


if __name__ == "__main__":
    main()