"""
Run the baseline local-level Kalman filter on 2022-05, using the R and Q
already derived in notes/microstructure_noise_R_derivation.md and
notes/regime_varying_Q_derivation.md as calibrated inputs -- not re-fit,
not re-guessed.

Usage:
    python scripts/run_baseline_filter.py data/raw/BTCUSDT-aggTrades-2022-05.zip \\
        --target-false-alarms-per-day 1
"""

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.kalman import run_local_level_filter, derive_chi2_gate

COLS = ["aggTradeId", "price", "quantity", "firstTradeId", "lastTradeId",
        "timestamp", "isBuyerMaker", "isBestMatch"]

# Derived in notes/microstructure_noise_R_derivation.md.
# All-tick estimate, selected because the filter ingests aggTrades rows
# as-is (sweep bursts included) -- see that note for the alternative
# (same-timestamp-excluded) estimate and the reasoning for this choice.
R_DERIVED = 3.661341181889e-11

# Derived in notes/regime_varying_Q_derivation.md.
DT_SEC = 60
ROLL_WINDOW = "1D"

KNOWN_EVENT_DATE = pd.Timestamp("2022-05-09", tz="UTC")


def load_trades(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        inner_name = z.namelist()[0]
        with z.open(inner_name) as f:
            df = pd.read_csv(f, header=None, names=COLS,
                              usecols=["price", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def build_q_series(price: pd.Series, dt_sec: int, window: str):
    """Same construction as scripts/rolling_q.py's rolling_rv_rate, then
    converted from a rate to a per-step variance (Q_t = rate_t * dt)."""
    resampled = price.resample(f"{dt_sec}s").last().ffill()
    log_price = np.log(resampled)
    log_ret = log_price.diff().dropna()
    sq_ret = log_ret ** 2
    window_sec = pd.Timedelta(window).total_seconds()
    rate = sq_ret.rolling(window).sum() / window_sec
    q_per_step = rate * dt_sec
    return q_per_step, log_price


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--target-false-alarms-per-day", type=float, default=1.0,
                         help="tolerance used to derive the chi-square gate "
                              "(default: 1 false alarm/day on average -- "
                              "change this only with your own justification)")
    args = parser.parse_args()

    df = load_trades(args.zip_path)
    print(f"Loaded {len(df):,} trades, {df.index.min()} -> {df.index.max()}")

    q_series, log_price = build_q_series(df["price"], DT_SEC, ROLL_WINDOW)

    # The rolling window needs a full period of history before Q is defined
    # -- align by dropping the warm-up NaNs from BOTH series together,
    # rather than filling them with a guessed value.
    valid = q_series.dropna().index.intersection(log_price.index)
    q_series = q_series.loc[valid]
    z = log_price.loc[valid]
    print(f"After dropping the {ROLL_WINDOW} warm-up period: "
          f"{len(z):,} steps, starting {z.index.min()}")

    steps_per_day = int(24 * 3600 / DT_SEC)
    alpha, threshold = derive_chi2_gate(args.target_false_alarms_per_day, steps_per_day)
    print(f"\nGate derivation: target={args.target_false_alarms_per_day} false "
          f"alarms/day, steps/day={steps_per_day} -> alpha={alpha:.2e}, "
          f"chi2(1) threshold={threshold:.3f}")

    result = run_local_level_filter(z.to_numpy(), q_series.to_numpy(), R_DERIVED)

    flagged = result.gate_stat > threshold
    print(f"\n{flagged.sum()} of {len(flagged)} steps exceed the gate "
          f"({flagged.sum() / len(flagged) * 100:.3f}% of steps, "
          f"vs. a target of {args.target_false_alarms_per_day / steps_per_day * 100:.3f}%)")

    # Split the flagged fraction into three periods to distinguish
    # "the model is generally wrong" from "the model is only wrong near
    # the known regime transition."
    during_start = KNOWN_EVENT_DATE
    during_end = KNOWN_EVENT_DATE + pd.Timedelta(days=4)  # 05-09 to 05-13
    periods = {
        "pre  (05-01 to 05-09)": z.index < during_start,
        "during (05-09 to 05-13)": (z.index >= during_start) & (z.index < during_end),
        "post (05-13 to 05-31)": z.index >= during_end,
    }
    print("\nFlagged fraction by period:")
    for label, mask in periods.items():
        n_period = mask.sum()
        n_flagged_period = flagged[mask].sum()
        pct = n_flagged_period / n_period * 100 if n_period else float("nan")
        print(f"  {label:>26}: {n_flagged_period:>5} / {n_period:>6} steps "
              f"({pct:.3f}%)")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(z.index, z.values, label="observed log-price", alpha=0.5, lw=0.8)
    ax1.plot(z.index, result.x_filt, label="filtered (Kalman) log-price", lw=1.2)
    ax1.scatter(z.index[flagged], z.values[flagged], color="red", s=10,
                zorder=5, label="gate-flagged steps")
    ax1.axvline(KNOWN_EVENT_DATE, color="black", linestyle="--", alpha=0.6,
                label="LUNA depeg (2022-05-09)")
    ax1.set_ylabel("log(price)")
    ax1.legend(loc="upper right")
    ax1.set_title("Baseline local-level Kalman filter — BTCUSDT 2022-05")

    ax2.plot(z.index, result.gate_stat, lw=0.6, label="gate statistic (y^2/S)")
    ax2.axhline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.2f}")
    ax2.axvline(KNOWN_EVENT_DATE, color="black", linestyle="--", alpha=0.6)
    ax2.set_yscale("log")
    ax2.set_ylabel("gate statistic (log scale)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right")

    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = "baseline_filter_2022-05.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    print("\nThings to actually check, not just eyeball:")
    print("- Do flagged steps cluster around 2022-05-09 through -13, or are")
    print("  they scattered evenly through the month? Scattered flags at a rate")
    print("  well above your target alpha would mean the model is misspecified")
    print("  (Q or R wrong), not that the market is genuinely alarming that often.")
    print("- Compare the flagged fraction above to your target false-alarm rate --")
    print("  if it's far off, that's a model-calibration problem to diagnose")
    print("  before treating any flag as a real signal.")


if __name__ == "__main__":
    main()