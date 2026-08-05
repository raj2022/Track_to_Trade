"""
Refit the regime HMM at K=4 (collapsing the flickering 0/3 pair from the
K=5 fit into one hypothesis), derive per-state R the same way as before,
run the IMM, and compare directly against the K=5 result on two questions:

  1. Does the elevated/extreme persistence result (0.159 / 0.830 / 0.424)
     survive materially intact under K=4?
  2. Does the chattering actually disappear -- i.e. does whichever K=4
     state absorbs the old 0/3 pair show a properly sticky self-transition,
     not the ~58% one that caused the flicker?

If both hold, K=4 is the better choice to feed Phase 2 (simpler, not
chattery, formally slightly worse BIC). If persistence degrades
materially, that's evidence the 5th state was doing real work and K=5
should be kept despite the flicker.

Usage:
    python scripts/k4_recheck.py data/raw/BTCUSDT-aggTrades-2022-05.zip
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_regime_hmm import build_returns, fit_best_hmm, bic, count_params, DT_SEC
from per_state_R import load_trades, assign_tick_states, roll_R_for_state

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.imm import run_imm

K = 4
N_RESTARTS = 5
SEED = 0
PLOTS_DIR = Path("plots")
KNOWN_EVENT_DATE = pd.Timestamp("2022-05-09", tz="UTC")

# Reference values from the K=5 run, for direct comparison.
K5_PERSISTENCE = {"pre": 0.159, "during_early": 0.830, "during_late": 0.424}
K5_STATE_0_SELF_TRANSITION = 0.5812
K5_STATE_3_SELF_TRANSITION = 0.8201


def main():
    if len(sys.argv) != 2:
        print("Usage: python k4_recheck.py <path_to_zip>")
        sys.exit(1)

    PLOTS_DIR.mkdir(exist_ok=True)
    zip_path = Path(sys.argv[1])

    df = load_trades(zip_path)
    returns, scale = build_returns(df["price"], DT_SEC)
    n_obs = len(returns)
    print(f"Fitting K={K} HMM on {n_obs:,} standardized returns...")

    model, ll = fit_best_hmm(returns, K, N_RESTARTS, SEED)
    n_params = count_params(K)
    bic_val = bic(ll, n_params, n_obs)
    print(f"K={K}: log-likelihood={ll:.2f}, BIC={bic_val:.2f}  "
          f"(K=5 BIC was 91589.59 -- K=4 will be formally worse; that's "
          f"expected and not disqualifying on its own)")

    variances = model.covars_.flatten() * (scale ** 2)
    order = np.argsort(variances)
    print(f"\nRegime variances (real units), sorted calm -> volatile:")
    for rank, state in enumerate(order):
        print(f"  state {state} (rank {rank}): variance = {variances[state]:.6e}")

    print(f"\nTransition matrix:")
    print(np.array2string(model.transmat_, precision=4, suppress_small=True))

    calmest_state = order[0]
    self_transition = model.transmat_[calmest_state, calmest_state]
    print(f"\nCalmest state (state {calmest_state}) self-transition: {self_transition:.4f}")
    print(f"For comparison, K=5's flickering pair had self-transitions "
          f"{K5_STATE_0_SELF_TRANSITION:.4f} (state 0) and "
          f"{K5_STATE_3_SELF_TRANSITION:.4f} (state 3).")
    if self_transition > 0.9:
        print("-> Properly sticky. Consistent with the flicker being an artifact "
              "of an unnecessary state split.")
    else:
        print("-> Still not clearly sticky. The chattering may not simply be a "
              "K=5-specific artifact -- worth a closer look before concluding.")

    # Per-state R, same method as per_state_R.py, generalized to K=4.
    states = model.predict(returns)
    resampled = df["price"].resample(f"{DT_SEC}s").last().ffill()
    log_price = np.log(resampled)
    state_index = log_price.diff().dropna().index
    state_series = pd.Series(states, index=state_index)

    print("\nMapping ticks to bin states for per-state R...")
    state_per_tick = assign_tick_states(df.index, state_series, DT_SEC)
    prices = df["price"].to_numpy()

    Q = np.empty(K)
    R = np.empty(K)
    print(f"\n{'state':>6}  {'n_pairs':>10}  {'R_k':>14}  {'Q_k':>14}")
    for k in range(K):
        r_k, n_pairs = roll_R_for_state(prices, state_per_tick, k)
        if np.isnan(r_k) or r_k < 0:
            note = " (clipped to 0)" if not np.isnan(r_k) else " (insufficient data, set to 0)"
            r_k_used = 0.0
        else:
            note = ""
            r_k_used = r_k
        q_k = variances[k] - 2 * r_k_used
        Q[k] = max(q_k, 1e-15)  # guard against a negative Q from a noisy R estimate
        R[k] = r_k_used
        print(f"  {k:>6}  {n_pairs:>10}  {r_k_used:>14.6e}  {Q[k]:>14.6e}{note}")

    # Run the IMM with the K=4 parameters.
    z = log_price.to_numpy()
    index = log_price.index
    result = run_imm(z, Q, R, model.transmat_)

    volatile_states = [s for s in range(K) if variances[s] > np.median(variances)]
    print(f"\nTreating states {volatile_states} as 'elevated/extreme' "
          f"(above-median variance) for the persistence comparison.")
    volatile_prob = result.mode_probs[:, volatile_states].sum(axis=1)

    pre = index < KNOWN_EVENT_DATE
    during_early = (index >= KNOWN_EVENT_DATE) & (index < KNOWN_EVENT_DATE + pd.Timedelta(days=4))
    during_late = (index >= KNOWN_EVENT_DATE + pd.Timedelta(days=4)) & (index < KNOWN_EVENT_DATE + pd.Timedelta(days=10))

    k4_pre = volatile_prob[pre].mean()
    k4_early = volatile_prob[during_early].mean()
    k4_late = volatile_prob[during_late].mean()

    print(f"\n{'':>20}  {'pre':>10}  {'during-early':>14}  {'during-late':>12}")
    print(f"{'K=5 (reference)':>20}  {K5_PERSISTENCE['pre']:>10.3f}  "
          f"{K5_PERSISTENCE['during_early']:>14.3f}  {K5_PERSISTENCE['during_late']:>12.3f}")
    print(f"{'K=4 (this run)':>20}  {k4_pre:>10.3f}  {k4_early:>14.3f}  {k4_late:>12.3f}")

    fig, ax = plt.subplots(figsize=(11, 4))
    for k in range(K):
        ax.plot(index, result.mode_probs[:, k], lw=0.8, label=f"state {k}")
    ax.axvline(KNOWN_EVENT_DATE, color="black", linestyle="--", alpha=0.6)
    ax.set_ylabel("mode probability")
    ax.set_xlabel("Date")
    ax.set_title(f"K=4 recheck — BTCUSDT 2022-05")
    ax.legend(loc="upper right", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = PLOTS_DIR / "k4_recheck_2022-05.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    print("\nDecide: if the persistence numbers are close to the K=5 reference AND")
    print("the calmest state's self-transition is properly sticky, prefer K=4 for")
    print("Phase 2 -- simpler and not chattery. If persistence degrades materially,")
    print("keep K=5 and treat the flicker as a documented, not fixed, limitation.")


if __name__ == "__main__":
    main()