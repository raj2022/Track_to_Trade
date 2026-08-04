"""
Derive a sampling interval and realized-variance estimate via a
multi-offset-averaged volatility signature plot.

Two fixes over the first version:
  1. Multi-offset averaging: RV at a given interval is computed at several
     bin-phase offsets and averaged, instead of a single fixed-phase
     resample. A single-phase resample lets one large price move land
     arbitrarily inside vs. across a bin boundary, which was producing the
     jagged, non-monotonic plot on the 2022-05 pool. Averaging over offsets
     removes that phase-placement artifact (Zhang-Mykland-Ait-Sahalia style).
  2. --start/--end date filtering: run this on a single, verified-homogeneous
     window (a genuine calm period, or a genuine single-regime slice), not a
     pooled month that mixes calm and crash days -- pooling regimes together
     conflates microstructure noise (roughly regime-invariant) with true
     variance (regime-dependent).

Usage:
    python scripts/signature_plot.py data/raw/BTCUSDT-aggTrades-2023-06.zip \\
        --start 2023-06-13 --end 2023-06-19 --label calm_2023-06-13_2023-06-19
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLS = ["aggTradeId", "price", "quantity", "firstTradeId", "lastTradeId",
        "timestamp", "isBuyerMaker", "isBestMatch"]

CANDIDATE_INTERVALS_SEC = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]

MAX_OFFSETS = 10  # cap on offsets averaged per interval


def load_trades(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        inner_name = z.namelist()[0]
        with z.open(inner_name) as f:
            df = pd.read_csv(f, header=None, names=COLS,
                              usecols=["price", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def report_trade_arrival_stats(df: pd.DataFrame) -> None:
    gaps = df.index.to_series().diff().dt.total_seconds().dropna()
    frac_gt_1s = float((gaps > 1.0).mean())
    print(f"\nTrade arrival diagnostics:")
    print(f"  median inter-trade gap: {gaps.median():.3f}s")
    print(f"  mean inter-trade gap:   {gaps.mean():.3f}s")
    print(f"  fraction of gaps > 1s:  {frac_gt_1s:.1%}")
    if frac_gt_1s > 0.10:
        print("  WARNING: >10% of gaps exceed 1s -- short-interval (1-10s) RV reads")
        print("  are likely biased by stale/repeated prices, not a clean noise signal.")
        print("  Do not treat the short end of the signature plot as a noise-dominated")
        print("  baseline without accounting for this.")


def stale_fraction_at_interval(price: pd.Series, interval_sec: int) -> float:
    """Fraction of resampled bins whose price is unchanged from the prior bin --
    a direct readout of how much a given interval is dominated by repeated,
    stale prices rather than genuine new information."""
    resampled = price.resample(f"{interval_sec}s").last().dropna()
    if len(resampled) < 2:
        return float("nan")
    return float((resampled.diff().dropna() == 0).mean())


def realized_variance_multi_offset(price: pd.Series, interval_sec: int) -> tuple[float, float]:
    """
    Average realized variance over several bin-phase offsets within one
    interval. Returns (mean RV across offsets, std RV across offsets) --
    the std tells you how much a single-offset read could have been off by,
    which is exactly what the naive version silently hid.
    """
    n_offsets = min(MAX_OFFSETS, interval_sec) if interval_sec > 1 else 1
    offset_step = interval_sec / n_offsets

    rv_estimates = []
    for i in range(n_offsets):
        offset_sec = i * offset_step
        resampled = price.resample(
            f"{interval_sec}s", offset=f"{offset_sec}s"
        ).last().dropna()
        log_ret = np.log(resampled).diff().dropna()
        rv_estimates.append(float((log_ret ** 2).sum()))

    return float(np.mean(rv_estimates)), float(np.std(rv_estimates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--start", type=str, default=None,
                         help="inclusive start date, e.g. 2023-06-13")
    parser.add_argument("--end", type=str, default=None,
                         help="inclusive end date, e.g. 2023-06-19")
    parser.add_argument("--label", type=str, default=None,
                         help="label used in the output filename/title")
    args = parser.parse_args()

    print(f"Loading {args.zip_path} ...")
    df = load_trades(args.zip_path)
    print(f"Loaded {len(df):,} trades, {df.index.min()} -> {df.index.max()}")

    if args.start or args.end:
        start = pd.Timestamp(args.start, tz="UTC") if args.start else df.index.min()
        end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1) if args.end else df.index.max()
        df = df.loc[start:end]
        print(f"Filtered to {start.date()} -> {(end - pd.Timedelta(days=1)).date()}: "
              f"{len(df):,} trades")
        if len(df) == 0:
            sys.exit("No trades in the requested date range -- check --start/--end "
                      "fall inside this file's month.")

    label = args.label or (f"{args.start}_{args.end}" if args.start else "full_period")

    report_trade_arrival_stats(df)

    print(f"\n{'interval':>8}  {'RV (mean)':>12}  {'RV (std)':>12}  {'rel.std':>8}  {'stale frac':>10}  n_offsets")
    results = []
    for interval in CANDIDATE_INTERVALS_SEC:
        mean_rv, std_rv = realized_variance_multi_offset(df["price"], interval)
        stale = stale_fraction_at_interval(df["price"], interval)
        n_offsets = min(MAX_OFFSETS, interval) if interval > 1 else 1
        rel_std = std_rv / mean_rv if mean_rv > 0 else float("nan")
        results.append((interval, mean_rv, std_rv, rel_std, stale))
        print(f"  {interval:>6}s  {mean_rv:>12.8f}  {std_rv:>12.8f}  {rel_std:>7.1%}  "
              f"{stale:>9.1%}  {n_offsets}")

    intervals, means, stds, rel_stds, stales = zip(*results)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.errorbar(intervals, means, yerr=stds, marker="o", capsize=3)
    ax1.set_xscale("log")
    ax1.set_ylabel("Realized variance (mean +/- std across offsets)")
    ax1.set_title(f"Volatility signature plot — BTCUSDT {label}")

    ax2.plot(intervals, [r * 100 for r in rel_stds], marker="o", label="relative std (%)")
    ax2.plot(intervals, [s * 100 for s in stales], marker="s", label="stale-price fraction (%)")
    ax2.set_xscale("log")
    ax2.set_xlabel("Sampling interval (seconds, log scale)")
    ax2.set_ylabel("Percent")
    ax2.legend()
    ax2.set_title("Reliability diagnostics — high values here mean don't trust this interval")

    fig.tight_layout()
    out_path = f"signature_plot_{label}.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    print("\nRead the bottom panel before the top one:")
    print("- High stale-price fraction at short intervals confirms the arrival-rate")
    print("  artifact -- those RV reads are biased low, not a genuine low-noise signal.")
    print("- High relative std at long intervals means too few independent windows to")
    print("  trust the mean there.")
    print("- Your plateau candidate must sit where BOTH diagnostics are low, not just")
    print("  where the top panel looks visually flat.")


if __name__ == "__main__":
    main()