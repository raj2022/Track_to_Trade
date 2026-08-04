"""
Derive a rolling, regime-varying process-noise (Q) series for 2022-05,
rather than a single pooled value.

Two steps, deliberately separated:
  1. Re-validate the sampling interval for THIS month. Do not assume the
     2023-06 calm-window plateau (600-3600s) transfers -- trade frequency
     during the LUNA collapse likely differs substantially from ordinary
     trading, which changes where the staleness artifact does or doesn't
     bite.
  2. Compute rolling realized-variance-rate at that validated interval,
     across several candidate rolling-window lengths, so the window length
     itself can be chosen by evidence (does it resolve the known LUNA
     transition without being pure noise during calm stretches) rather than
     picked by feel.

Usage:
    python scripts/rolling_q.py data/raw/BTCUSDT-aggTrades-2022-05.zip
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLS = ["aggTradeId", "price", "quantity", "firstTradeId", "lastTradeId",
        "timestamp", "isBuyerMaker", "isBestMatch"]

# Candidate sampling intervals to re-check staleness at, seconds.
CANDIDATE_DT_SEC = [10, 30, 60, 300, 600, 1800]

# Candidate rolling-window lengths for the local RV-rate estimate.
CANDIDATE_WINDOWS = ["1h", "4h", "12h", "1D", "3D"]

# LUNA/UST depeg began approximately 2022-05-09 (UTC).
KNOWN_EVENT_DATE = pd.Timestamp("2022-05-09", tz="UTC")


def load_trades(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        inner_name = z.namelist()[0]
        with z.open(inner_name) as f:
            df = pd.read_csv(f, header=None, names=COLS,
                              usecols=["price", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def stale_fraction_at_interval(price: pd.Series, interval_sec: int) -> float:
    resampled = price.resample(f"{interval_sec}s").last().dropna()
    if len(resampled) < 2:
        return float("nan")
    return float((resampled.diff().dropna() == 0).mean())


def check_staleness_by_period(df: pd.DataFrame) -> None:
    """
    Staleness can differ a lot between the calm early-May stretch and the
    crisis days -- check both, not just the month as a whole, or a single
    number could hide exactly the kind of pooling error already caught
    twice in this project.
    """
    pre_crash = df.loc[df.index < KNOWN_EVENT_DATE]
    crash = df.loc[df.index >= KNOWN_EVENT_DATE]

    print("Staleness by sampling interval, split at the known LUNA date "
          f"({KNOWN_EVENT_DATE.date()}):\n")
    print(f"{'interval':>10}  {'pre-event stale%':>18}  {'post-event stale%':>18}")
    for interval in CANDIDATE_DT_SEC:
        pre_stale = stale_fraction_at_interval(pre_crash["price"], interval)
        post_stale = stale_fraction_at_interval(crash["price"], interval)
        print(f"  {interval:>7}s  {pre_stale:>17.1%}  {post_stale:>17.1%}")

    print("\nPick the shortest interval where BOTH columns are low (say, "
          "single digits). If pre- and post-event staleness diverge a lot, "
          "that itself is informative -- it means trade frequency is "
          "regime-dependent, which has implications for using a single "
          "fixed dt across the whole rolling series at all.")


def rolling_rv_rate(price: pd.Series, dt_sec: int, window: str) -> pd.Series:
    """
    Rolling realized-variance RATE (per second), at a fixed sampling dt,
    over a rolling window of the given length. This -- not a single-value
    RV over the whole month -- is what should feed a time-varying Q.
    """
    resampled = price.resample(f"{dt_sec}s").last().ffill()
    log_ret = np.log(resampled).diff().dropna()
    sq_ret = log_ret ** 2
    rolling_sum = sq_ret.rolling(window).sum()
    # convert window string to seconds to get a rate
    window_sec = pd.Timedelta(window).total_seconds()
    return rolling_sum / window_sec


def main():
    if len(sys.argv) != 2:
        print("Usage: python rolling_q.py <path_to_zip>")
        sys.exit(1)

    df = load_trades(Path(sys.argv[1]))
    print(f"Loaded {len(df):,} trades, {df.index.min()} -> {df.index.max()}")

    check_staleness_by_period(df)

    dt_sec = int(input("\nEnter the validated dt in seconds (from the table above): "))

    fig, ax = plt.subplots(figsize=(10, 6))
    for window in CANDIDATE_WINDOWS:
        rate = rolling_rv_rate(df["price"], dt_sec, window)
        ax.plot(rate.index, rate.values, label=f"window={window}", alpha=0.8)

    ax.axvline(KNOWN_EVENT_DATE, color="red", linestyle="--",
               label="LUNA depeg (2022-05-09)")
    ax.set_yscale("log")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling realized-variance rate (per second, log scale)")
    ax.set_title(f"Rolling Q candidates — BTCUSDT 2022-05 (dt={dt_sec}s)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = "rolling_q_candidates_2022-05.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    print("\nCompare the candidate window lengths:")
    print("- Too short: the series will be visibly jagged/noisy even before")
    print("  the known event date -- that's estimation noise, not regime signal.")
    print("- Too long: the rise around 2022-05-09 will be smeared out over")
    print("  days instead of tracking the actual transition.")
    print("- Pick the shortest window that's still visibly stable pre-event --")
    print("  that's the bias/variance trade-off, made visible rather than assumed.")


if __name__ == "__main__":
    main()