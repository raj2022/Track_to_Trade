"""
Check whether the short-window (1h) rolling RV-rate series shows a genuine
~24h periodicity (diurnal trading-session effect), as a candidate
explanation for why longer rolling windows (1D) come out visibly smoother
than shorter ones (1h/4h/12h).

Usage:
    python scripts/diurnal_check.py data/raw/BTCUSDT-aggTrades-2022-05.zip
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLS = ["aggTradeId", "price", "quantity", "firstTradeId", "lastTradeId",
        "timestamp", "isBuyerMaker", "isBestMatch"]

DT_SEC = 60          # validated sampling interval, from the rolling_q step
ROLL_WINDOW = "1h"   # the noisiest / most diurnally-exposed candidate
MAX_LAG_HOURS = 48


def load_trades(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        inner_name = z.namelist()[0]
        with z.open(inner_name) as f:
            df = pd.read_csv(f, header=None, names=COLS,
                              usecols=["price", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def rolling_rv_rate(price: pd.Series, dt_sec: int, window: str) -> pd.Series:
    resampled = price.resample(f"{dt_sec}s").last().ffill()
    log_ret = np.log(resampled).diff().dropna()
    sq_ret = log_ret ** 2
    rolling_sum = sq_ret.rolling(window).sum()
    window_sec = pd.Timedelta(window).total_seconds()
    return rolling_sum / window_sec


def autocorrelation(series: pd.Series, max_lag_bins: int) -> np.ndarray:
    """Simple ACF via numpy, normalized so acf[0] == 1."""
    x = series.dropna().to_numpy()
    x = x - x.mean()
    n = len(x)
    acf = np.empty(max_lag_bins + 1)
    denom = np.dot(x, x)
    for lag in range(max_lag_bins + 1):
        acf[lag] = np.dot(x[: n - lag], x[lag:]) / denom
    return acf


def main():
    if len(sys.argv) != 2:
        print("Usage: python diurnal_check.py <path_to_zip>")
        sys.exit(1)

    df = load_trades(Path(sys.argv[1]))
    print(f"Loaded {len(df):,} trades, {df.index.min()} -> {df.index.max()}")

    rate = rolling_rv_rate(df["price"], DT_SEC, ROLL_WINDOW)

    bins_per_hour = 3600 // DT_SEC
    max_lag_bins = MAX_LAG_HOURS * bins_per_hour
    lag_24h_bins = 24 * bins_per_hour

    print(f"\nComputing ACF of the {ROLL_WINDOW}-window RV-rate series "
          f"out to {MAX_LAG_HOURS}h ({max_lag_bins:,} lags at {DT_SEC}s bins)...")
    acf = autocorrelation(rate, max_lag_bins)

    lag_hours = np.arange(len(acf)) / bins_per_hour

    print(f"\nACF at lag=24h: {acf[lag_24h_bins]:.4f}")
    print(f"ACF at lag=12h: {acf[12 * bins_per_hour]:.4f}  (half-period, for comparison)")
    print(f"ACF at lag=18h: {acf[18 * bins_per_hour]:.4f}  (off-period, for comparison)")
    print(f"ACF at lag=30h: {acf[30 * bins_per_hour]:.4f}  (off-period, for comparison)")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lag_hours, acf)
    ax.axvline(24, color="red", linestyle="--", label="24h")
    ax.axvline(48, color="red", linestyle=":", alpha=0.5, label="48h")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"ACF of {ROLL_WINDOW}-window rolling RV-rate — BTCUSDT 2022-05")
    ax.legend()
    fig.tight_layout()
    out_path = "diurnal_acf_2022-05.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    print("\nInterpretation guide:")
    print("- A genuine diurnal effect shows up as a LOCAL PEAK in the ACF at")
    print("  lag=24h (and a smaller one at 48h) relative to its immediate")
    print("  neighborhood -- not just 'some positive number', since almost any")
    print("  short-range-correlated series has positive ACF near small lags.")
    print("- Compare the 24h value to the 18h/30h off-period values printed above:")
    print("  if 24h is a clear local maximum relative to them, that's real evidence")
    print("  of session-driven periodicity, not just a broad exponential decay that")
    print("  happens to still be nonzero at 24h.")
    print("- If no such peak exists, don't force the diurnal story -- report that")
    print("  the smoothing effect from a 1D window is better explained by")
    print("  averaging horizon alone, not a specific 24h cycle.")


if __name__ == "__main__":
    main()