"""
Check whether the microstructure noise variance R is stable across regimes,
as flagged as an open item in notes/microstructure_noise_R_derivation.md.

Computes Roll's estimator on a crisis sub-window (2022-05-09 to 2022-05-12,
the days independently identified as MAD outliers in the daily-RV check)
and compares to the calm-window R already derived.

Usage:
    python scripts/check_R_regime_invariance.py data/raw/BTCUSDT-aggTrades-2022-05.zip
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["aggTradeId", "price", "quantity", "firstTradeId", "lastTradeId",
        "timestamp", "isBuyerMaker", "isBestMatch"]

R_CALM = 3.661341181889e-11  # from notes/microstructure_noise_R_derivation.md

CRISIS_START = "2022-05-09"
CRISIS_END = "2022-05-12"


def load_trades(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        inner_name = z.namelist()[0]
        with z.open(inner_name) as f:
            df = pd.read_csv(f, header=None, names=COLS,
                              usecols=["price", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def roll_noise_variance(price: pd.Series) -> tuple[float, float]:
    """Returns (raw lag-1 autocovariance, R). Same method as
    scripts/roll_estimator.py, applied here to a crisis sub-window."""
    returns = np.diff(np.log(price.to_numpy(dtype=float)))
    cov = float(np.cov(returns[1:], returns[:-1], ddof=1)[0, 1])
    return cov, -cov


def main():
    if len(sys.argv) != 2:
        print("Usage: python check_R_regime_invariance.py <path_to_zip>")
        sys.exit(1)

    df = load_trades(Path(sys.argv[1]))
    crisis = df.loc[CRISIS_START:CRISIS_END]
    print(f"Crisis window {CRISIS_START} to {CRISIS_END}: {len(crisis):,} ticks")

    cov, r_crisis = roll_noise_variance(crisis["price"])
    print(f"\nLag-1 return autocovariance (crisis): {cov:.12e}")
    if cov >= 0:
        print("WARNING: non-negative covariance -- Roll's model does not hold "
              "in this window. Do not trust r_crisis below without investigating.")

    print(f"\nR (calm, previously derived): {R_CALM:.6e}")
    print(f"R (crisis, this window):       {r_crisis:.6e}")
    ratio = r_crisis / R_CALM
    print(f"Ratio (crisis / calm):          {ratio:.2f}x")

    spread_calm_bp = 2 * np.sqrt(R_CALM) * 1e4
    spread_crisis_bp = 2 * np.sqrt(r_crisis) * 1e4 if r_crisis > 0 else float("nan")
    print(f"\nImplied effective spread, calm:   {spread_calm_bp:.3f} bp")
    print(f"Implied effective spread, crisis: {spread_crisis_bp:.3f} bp")

    print("\nDecision guide:")
    print("- If the ratio is close to 1x (say, within ~2x), treat R as shared")
    print("  across IMM hypotheses -- consistent with R representing exchange")
    print("  mechanics rather than regime-dependent behavior.")
    print("- If the ratio is large, R needs to be regime-conditional too, not")
    print("  just Q -- widening spreads during stress is a real, well-documented")
    print("  microstructure phenomenon (liquidity providers widen quotes when")
    print("  uncertain), so a large ratio would not be a surprising or wrong")
    print("  result, just one that changes the IMM's design.")


if __name__ == "__main__":
    main()