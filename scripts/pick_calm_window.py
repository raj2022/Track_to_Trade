"""Inspect and select a calm seven-day window from a Binance aggTrades archive.

The automatic selection is the window with the smallest sum of daily
one-minute realized variances. That score can conceal a single unusually
volatile day, so the script also reports each selected day's share of the
weekly variance. A concentration cutoff is derived from a bootstrap null:
random groups of daily RVs sampled from the month, rather than a hard-coded
share threshold.

Usage:
    python scripts/pick_calm_window.py data/raw/BTCUSDT-aggTrades-2022-04.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# Binance spot aggTrades schema before 2025-01-01 (timestamps are milliseconds).
COLS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
]


def load_prices(zip_path: Path) -> pd.Series:
    """Load trade prices indexed by their UTC timestamp."""
    with zipfile.ZipFile(zip_path) as archive:
        csv_files = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_files) != 1:
            raise ValueError(f"expected one CSV in {zip_path}, found {len(csv_files)}")
        with archive.open(csv_files[0]) as file:
            trades = pd.read_csv(
                file, header=None, names=COLS, usecols=["price", "timestamp"]
            )

    trades["timestamp"] = pd.to_datetime(trades["timestamp"], unit="ms", utc=True)
    return trades.set_index("timestamp")["price"].sort_index()


def daily_metrics(prices: pd.Series) -> pd.DataFrame:
    """Calculate daily one-minute RV and a complementary intraday range."""
    minute_price = prices.resample("1min").last().ffill()
    log_returns = np.log(minute_price).diff()
    daily = pd.DataFrame(
        {
            "rv_1m": log_returns.pow(2).resample("1D").sum(),
            "intraday_range": np.log(
                prices.resample("1D").max() / prices.resample("1D").min()
            ),
        }
    )
    daily.index = daily.index.date
    return daily


def window_summary(daily: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return score and concentration diagnostics for every complete window."""
    rows = []
    for start in range(len(daily) - days + 1):
        window = daily.iloc[start : start + days]
        total_rv = float(window["rv_1m"].sum())
        rows.append(
            {
                "start": window.index[0],
                "end": window.index[-1],
                "total_rv": total_rv,
                "max_daily_rv": float(window["rv_1m"].max()),
                "largest_rv_share": float(window["rv_1m"].max() / total_rv),
                "max_intraday_range": float(window["intraday_range"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("total_rv", ignore_index=True)


def trim_outlier_days(daily_rv: pd.Series, mad_threshold: float = 3.5) -> pd.Series:
    """Drop days whose RV is a robust outlier before it contaminates the null.

    Uses the modified z-score (Iglewicz & Hoya): centers on the median and
    scales by the median absolute deviation, both robust to the very outliers
    being screened for (unlike mean/std, which the outliers themselves skew).
    threshold=3.5 is the standard convention from that method, not a fitted
    value -- if asked to defend it, that's the honest answer: it's a cited
    convention, chosen because it's robust by construction, not because it
    was tuned to produce a particular result here.
    """
    median = np.median(daily_rv.to_numpy())
    mad = np.median(np.abs(daily_rv.to_numpy() - median))
    if mad == 0:
        return daily_rv
    modified_z = 0.6745 * (daily_rv - median) / mad
    return daily_rv[modified_z.abs() <= mad_threshold]


def bootstrap_max_share_null(
    daily_rv: pd.Series, window_len: int = 7, n_draws: int = 5_000, seed: int = 0
) -> np.ndarray:
    """Sample the maximum daily-RV share expected from random daily groupings.

    Drawn from an outlier-trimmed population: the null should represent what
    concentration looks like among genuinely homogeneous "calm" days, not a
    population that already contains the spike days we're trying to screen
    against -- pooling those in inflates the cutoff and makes it too
    permissive, the same conflation-of-populations error as the May
    signature plot.
    """
    calm_pool = trim_outlier_days(daily_rv)
    rng = np.random.default_rng(seed)
    values = calm_pool.to_numpy()
    if len(values) < window_len:
        raise ValueError(
            f"only {len(values)} days remain after outlier trimming, "
            f"need at least {window_len}"
        )
    shares = np.empty(n_draws)
    for draw in range(n_draws):
        sample = values[rng.choice(len(values), size=window_len, replace=False)]
        shares[draw] = sample.max() / sample.sum()
    return shares


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path, help="Binance aggTrades ZIP archive")
    parser.add_argument("--days", type=int, default=7, help="window length (default: 7)")
    parser.add_argument("--top", type=int, default=10, help="number of candidate windows to display (default: 10)")
    parser.add_argument(
        "--null-percentile",
        type=float,
        default=90,
        help="bootstrap-null percentile used for the concentration cutoff (default: 90)",
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=5_000,
        help="number of bootstrap groupings used for the null (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for the bootstrap null (default: 0)"
    )
    args = parser.parse_args()

    if args.days < 2:
        parser.error("--days must be at least 2")
    if not 0 < args.null_percentile < 100:
        parser.error("--null-percentile must be between 0 and 100")
    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive")

    try:
        prices = load_prices(args.zip_path)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        sys.exit(f"Could not load {args.zip_path}: {error}")

    daily = daily_metrics(prices)
    if len(daily) < args.days:
        sys.exit(f"Only {len(daily)} days available; need at least {args.days}.")

    windows = window_summary(daily, args.days)
    calm_pool = trim_outlier_days(daily["rv_1m"])
    trimmed_days = daily.index.difference(calm_pool.index)
    null_shares = bootstrap_max_share_null(
        daily["rv_1m"], args.days, args.bootstrap_draws, args.seed
    )
    cutoff = float(np.percentile(null_shares, args.null_percentile))
    windows["passes_null"] = windows["largest_rv_share"] <= cutoff

    print(f"Loaded {len(prices):,} trades: {prices.index.min()} to {prices.index.max()}")
    print(
        f"\nOutlier trimming (modified z-score, MAD-based, threshold=3.5): "
        f"excluded {len(trimmed_days)} of {len(daily)} days from the null population"
    )
    if len(trimmed_days) > 0:
        print(f"  Excluded days: {sorted(str(d) for d in trimmed_days)}")
    print(
        f"\nBootstrap null: {args.bootstrap_draws:,} random {args.days}-day groupings; "
        f"{args.null_percentile:g}th-percentile largest-RV-share cutoff = {cutoff:.1%}"
    )
    print("\nDaily metrics (log-return units)")
    print(daily.to_string(float_format=lambda value: f"{value:.6f}"))

    print(f"\nTop {min(args.top, len(windows))} {args.days}-day windows by total 1-minute RV")
    print(
        windows.head(args.top).to_string(
            index=False,
            formatters={
                "total_rv": lambda value: f"{value:.6f}",
                "max_daily_rv": lambda value: f"{value:.6f}",
                "largest_rv_share": lambda value: f"{value:.1%}",
                "max_intraday_range": lambda value: f"{value:.2%}",
            },
        )
    )

    passing = windows.loc[windows["passes_null"]]
    if passing.empty:
        print("\nNo candidate passes the bootstrap-derived concentration cutoff.")
        return

    selected_window = passing.iloc[0]
    selected = daily.loc[selected_window["start"] : selected_window["end"]].copy()
    selected["rv_share"] = selected["rv_1m"] / selected_window["total_rv"]
    print(
        "\nLowest-total-RV window that passes the bootstrap-null check: "
        f"{selected_window['start']} through {selected_window['end']}"
    )
    print(selected.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()