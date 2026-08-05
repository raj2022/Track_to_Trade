"""
Run the IMM filter on 2022-05, using the per-regime Q/R derived via
scripts/per_state_R.py and the transition matrix from the BIC-selected
K=5 HMM fit (scripts/fit_regime_hmm.py) -- all fixed, calibrated inputs,
not re-fit here.

Usage:
    python scripts/run_imm_filter.py data/raw/BTCUSDT-aggTrades-2022-05.zip
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.imm import run_imm

COLS = ["aggTradeId", "price", "quantity", "firstTradeId", "lastTradeId",
        "timestamp", "isBuyerMaker", "isBestMatch"]

DT_SEC = 60
KNOWN_EVENT_DATE = pd.Timestamp("2022-05-09", tz="UTC")

PLOTS_DIR = Path("plots")

# Regime parameters, state order [0, 1, 2, 3, 4], from per_state_R.py.
# State 0's R clipped to 0 (raw estimate was -2.73e-12, statistically
# indistinguishable from zero at n=784k pairs -- see notes).
Q = np.array([2.327209e-08, 1.730507e-06, 1.001249e-05, 2.061608e-07, 4.791311e-07])
R = np.array([0.0,          3.003397e-10, 4.530218e-09, 7.573640e-13, 3.619998e-11])

# Transition matrix from the K=5 HMM fit (fit_regime_hmm.py).
TRANSITION = np.array([
    [0.5812, 0.0027, 0.0,    0.3916, 0.0244],
    [0.0,    0.9568, 0.0120, 0.0,    0.0312],
    [0.0,    0.0402, 0.9598, 0.0,    0.0],
    [0.1764, 0.0010, 0.0005, 0.8201, 0.0020],
    [0.0,    0.0163, 0.0003, 0.0082, 0.9752],
])

STATE_LABELS = {0: "calm-thin", 1: "elevated", 2: "extreme", 3: "calm-quiet", 4: "calm-normal"}
VOLATILE_STATES = [1, 2]  # states worth summing for an "elevated or worse" signal


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
        print("Usage: python run_imm_filter.py <path_to_zip>")
        sys.exit(1)

    PLOTS_DIR.mkdir(exist_ok=True)

    df = load_trades(Path(sys.argv[1]))
    print(f"Loaded {len(df):,} trades, {df.index.min()} -> {df.index.max()}")

    resampled = df["price"].resample(f"{DT_SEC}s").last().ffill()
    log_price = np.log(resampled)
    z = log_price.to_numpy()
    index = log_price.index
    print(f"{len(z):,} steps at dt={DT_SEC}s")

    result = run_imm(z, Q, R, TRANSITION)

    volatile_prob = result.mode_probs[:, VOLATILE_STATES].sum(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(index, z, label="observed log-price", alpha=0.5, lw=0.8)
    ax1.plot(index, result.x_filt, label="IMM combined estimate", lw=1.2)
    ax1.axvline(KNOWN_EVENT_DATE, color="black", linestyle="--", alpha=0.6,
                label="LUNA depeg (2022-05-09)")
    ax1.set_ylabel("log(price)")
    ax1.legend(loc="upper right")
    ax1.set_title("IMM filter — BTCUSDT 2022-05")

    for k in range(len(Q)):
        ax2.plot(index, result.mode_probs[:, k], lw=0.8,
                  label=f"state {k} ({STATE_LABELS[k]})")
    ax2.axvline(KNOWN_EVENT_DATE, color="black", linestyle="--", alpha=0.6)
    ax2.set_ylabel("mode probability")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", fontsize=8)

    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = PLOTS_DIR / "imm_filter_2022-05.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    # Direct comparison to the baseline's failure mode: does the
    # "elevated or worse" probability persist through the sustained
    # regime, rather than decaying back down within ~a day like the
    # baseline's continuously-adapting Q did?
    during = (index >= KNOWN_EVENT_DATE) & (index < KNOWN_EVENT_DATE + pd.Timedelta(days=10))
    print(f"\nMean P(state in {{1,2}} = elevated/extreme) during 05-09 to 05-19: "
          f"{volatile_prob[during].mean():.3f}")
    during_early = (index >= KNOWN_EVENT_DATE) & (index < KNOWN_EVENT_DATE + pd.Timedelta(days=4))
    during_late = (index >= KNOWN_EVENT_DATE + pd.Timedelta(days=4)) & (index < KNOWN_EVENT_DATE + pd.Timedelta(days=10))
    print(f"  split: 05-09 to 05-13: {volatile_prob[during_early].mean():.3f}")
    print(f"  split: 05-13 to 05-19: {volatile_prob[during_late].mean():.3f}")
    pre = index < KNOWN_EVENT_DATE
    print(f"Mean P(state in {{1,2}}) pre-event (05-01 to 05-09): "
          f"{volatile_prob[pre].mean():.3f}")
    print("\nCompare this to the baseline filter's finding: a continuously-")
    print("updated Q re-absorbed the anomaly within about a day. Check whether")
    print("this elevated-probability plot instead stays high through the full")
    print("~05-09 to ~05-19 window the HMM decode identified, rather than")
    print("decaying back toward the pre-event level early.")


if __name__ == "__main__":
    main()