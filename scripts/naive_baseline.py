"""
Naive baseline: flag "elevated" whenever the current 1-hour realized
variance exceeds the EXPANDING median of all preceding 1-hour RV values.

Deliberately the simplest defensible rule available: no lookback-window
parameter to justify (an expanding median needs none -- it just uses
everything seen so far), no multiplier, no fitted anything. This exists
to answer one question honestly: did the IMM's substantial added
complexity actually earn its keep, or does a nearly parameter-free rule
do comparably well on the one result we have to compare against
(elevated/extreme persistence around the known LUNA event)?

Usage:
    python scripts/naive_baseline.py data/raw/BTCUSDT-aggTrades-2022-05.zip
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLS = ["aggTradeId", "price", "quantity", "firstTradeId", "lastTradeId",
        "timestamp", "isBuyerMaker", "isBestMatch"]

DT_SEC = 60
SHORT_WINDOW = "1h"  # "current" volatility -- matches the noisiest candidate
                       # from the original rolling_q.py comparison, chosen
                       # there for genuine intraday responsiveness
KNOWN_EVENT_DATE = pd.Timestamp("2022-05-09", tz="UTC")
PLOTS_DIR = Path("plots")

# IMM reference result, for direct comparison.
IMM_PERSISTENCE = {"pre": 0.159, "during_early": 0.830, "during_late": 0.424}


def load_trades(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        inner_name = z.namelist()[0]
        with z.open(inner_name) as f:
            df = pd.read_csv(f, header=None, names=COLS,
                              usecols=["price", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def main():
    if len(sys.argv) != 2:
        print("Usage: python naive_baseline.py <path_to_zip>")
        sys.exit(1)

    PLOTS_DIR.mkdir(exist_ok=True)

    df = load_trades(Path(sys.argv[1]))
    print(f"Loaded {len(df):,} trades")

    resampled = df["price"].resample(f"{DT_SEC}s").last().ffill()
    log_ret = np.log(resampled).diff().dropna()
    sq_ret = log_ret ** 2
    window_sec = pd.Timedelta(SHORT_WINDOW).total_seconds()
    current_rv = sq_ret.rolling(SHORT_WINDOW).sum() / window_sec
    current_rv = current_rv.dropna()

    # Expanding median: at each t, the median of every value seen up to
    # and including t-1 (shifted by 1 so the rule never uses today's own
    # value to judge itself -- a real, if easy to miss, leakage risk).
    expanding_median = current_rv.shift(1).expanding().median()

    valid = expanding_median.dropna().index.intersection(current_rv.index)
    current_rv = current_rv.loc[valid]
    expanding_median = expanding_median.loc[valid]

    flagged = current_rv > expanding_median
    print(f"{len(valid):,} steps after warm-up "
          f"(expanding median needs a full {SHORT_WINDOW} window of history "
          f"before its first value)")

    pre = valid < KNOWN_EVENT_DATE
    during_early = (valid >= KNOWN_EVENT_DATE) & (valid < KNOWN_EVENT_DATE + pd.Timedelta(days=4))
    during_late = (valid >= KNOWN_EVENT_DATE + pd.Timedelta(days=4)) & (valid < KNOWN_EVENT_DATE + pd.Timedelta(days=10))

    naive_pre = flagged[pre].mean()
    naive_early = flagged[during_early].mean()
    naive_late = flagged[during_late].mean()

    print(f"\n{'':>20}  {'pre':>10}  {'during-early':>14}  {'during-late':>12}")
    print(f"{'IMM (reference)':>20}  {IMM_PERSISTENCE['pre']:>10.3f}  "
          f"{IMM_PERSISTENCE['during_early']:>14.3f}  {IMM_PERSISTENCE['during_late']:>12.3f}")
    print(f"{'Naive baseline':>20}  {naive_pre:>10.3f}  {naive_early:>14.3f}  {naive_late:>12.3f}")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(valid, flagged.astype(float).rolling("6h").mean(),
             lw=1, label="fraction flagged (6h rolling smoothing, for readability)")
    ax.axvline(KNOWN_EVENT_DATE, color="black", linestyle="--", alpha=0.6,
                label="LUNA depeg")
    ax.set_ylabel("fraction of steps flagged")
    ax.set_xlabel("Date")
    ax.set_title("Naive baseline (1h RV > expanding median) — BTCUSDT 2022-05")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = PLOTS_DIR / "naive_baseline_2022-05.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    print("\nInterpretation guide, don't skip this:")
    print("- By construction, ~50% of ALL steps will be flagged overall (that's")
    print("  what 'above the median' means) -- so the pre-event rate near 0.5 is")
    print("  not itself informative. What matters is whether during-event rates")
    print("  rise CLEARLY above pre-event, and whether that rise persists the way")
    print("  the IMM's did (0.830 early, still 0.424 -- 2.7x pre-event -- late),")
    print("  or whether the naive rule is noisier / less differentiated.")
    print("- If the naive rule's during-vs-pre CONTRAST is comparably clean, that's")
    print("  a real, uncomfortable finding: the IMM's complexity may not be earning")
    print("  its keep on THIS metric alone, and the honest thing to do is say so,")
    print("  not bury it. The IMM's real justification would then need to rest on")
    print("  things the naive rule structurally cannot do at all -- e.g. a")
    print("  calibrated probability, per-regime Q/R for downstream use, or the")
    print("  chi-square-derived detection gate -- not on this one comparison.")


if __name__ == "__main__":
    main()