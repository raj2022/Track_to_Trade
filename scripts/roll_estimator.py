"""Estimate tick-level microstructure noise with Roll's implied-spread model.

Usage:
    python scripts/roll_estimator.py data/raw/BTCUSDT-aggTrades-2023-06.zip \
        --start 2023-06-13 --end 2023-06-19

        
    python scripts/roll_estimator.py data/raw/BTCUSDT-aggTrades-2023-06.zip \
  --start 2023-06-13 --end 2023-06-19 \
  --exclude-same-timestamp
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


COLS = ["aggTradeId", "price", "quantity", "firstTradeId", "lastTradeId",
        "timestamp", "isBuyerMaker", "isBestMatch"]


def load_trades(zip_path: Path) -> pd.DataFrame:
    """Load trades in chronological order, retaining file order within a millisecond."""
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(f, header=None, names=COLS,
                             usecols=["price", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    # Binance's file order is the trade sequence. A stable sort preserves that
    # sequence for rows that share the same millisecond timestamp.
    return df.set_index("timestamp").sort_index(kind="stable")


def roll_noise_variance(
    price: pd.Series, *, exclude_same_timestamp: bool = False
) -> float:
    """Return Roll's noise variance R = -Cov(r_t, r_{t-1}).

    Returns are consecutive observed aggregate-trade log returns. Zero returns
    remain in the series: they are observed outcomes, rather than resampling
    artifacts. By default, returns between trades sharing a timestamp also
    remain. Set ``exclude_same_timestamp`` for a robustness check that removes
    every covariance observation containing such a return, without joining
    non-adjacent returns together.
    """
    if (price <= 0).any():
        raise ValueError("Log returns require strictly positive prices.")

    returns = np.diff(np.log(price.to_numpy(dtype=float)))
    if len(returns) < 3:
        raise ValueError("At least four prices are required for lag-1 covariance.")

    current_returns = returns[1:]
    lagged_returns = returns[:-1]
    if exclude_same_timestamp:
        # `same_timestamp_return[i]` denotes the return from price[i] to
        # price[i + 1]. A lag-1 covariance observation needs both adjacent
        # returns to cross distinct timestamps.
        same_timestamp_return = price.index[1:] == price.index[:-1]
        usable = (~same_timestamp_return[1:]) & (~same_timestamp_return[:-1])
        current_returns = current_returns[usable]
        lagged_returns = lagged_returns[usable]
        print(f"Lag-1 covariance observations after timestamp exclusion: "
              f"{len(current_returns):,}")

    # This is the usual sample covariance (normalization n - 1) of adjacent
    # returns. Both vectors have one observation per consecutive tick pair.
    lag1_covariance = float(np.cov(current_returns, lagged_returns, ddof=1)[0, 1])
    print(f"Lag-1 return autocovariance: {lag1_covariance:.12e}")

    if lag1_covariance >= 0:
        raise ValueError(
            "Lag-1 covariance is non-negative; Roll's bid-ask-bounce model "
            "does not yield a valid positive noise variance for this sample."
        )

    return -lag1_covariance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--start", default="2023-06-13", help="inclusive UTC date")
    parser.add_argument("--end", default="2023-06-19", help="inclusive UTC date")
    args = parser.parse_args()

    df = load_trades(args.zip_path)
    start = pd.Timestamp(args.start, tz="UTC")
    end_exclusive = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    window = df.loc[(df.index >= start) & (df.index < end_exclusive)]
    if window.empty:
        sys.exit("No trades in the requested date range.")

    print(f"{len(window):,} ticks in window")
    same_ts = window.index.to_series().diff().dt.total_seconds().eq(0)
    print(f"fraction of consecutive pairs sharing a timestamp: {same_ts.mean():.1%}")

    print("\nAll-tick specification (primary):")
    r_all = roll_noise_variance(window["price"])
    print(f"Roll noise variance (R): {r_all:.12e}")

    print("\nSame-timestamp-excluded robustness check:")
    r_clean = roll_noise_variance(window["price"], exclude_same_timestamp=True)
    print(f"Roll noise variance (R): {r_clean:.12e}")
    print(f"Difference from all-tick R: {(r_clean / r_all - 1):+.2%}")


if __name__ == "__main__":
    main()
