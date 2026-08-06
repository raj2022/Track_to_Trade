"""
Run the IMM filter on 2022-05, using the CANONICAL K=4 per-regime Q/R and
transition matrix (scripts/k4_recheck.py) -- all fixed, calibrated inputs,
not re-fit here.

K=4 superseded an earlier K=5 fit: K=5 was BIC-selected but showed a real
flickering artifact (two states with only a 0.58 mutual self-transition
probability). K=4 was tested directly against the actual downstream
persistence result, found comparable-to-better, with a properly sticky
calmest state (0.9745) -- see notes/k4_naive_bipower_phase1_final_closeout.md.

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

# Regime parameters, K=4, state order [0, 1, 2, 3], from k4_recheck.py.
# State 1's R clipped to 0 (raw estimate was near zero at n=4.46M pairs).
Q = np.array([4.668482e-07, 1.229825e-07, 9.774018e-06, 1.648590e-06])
R = np.array([4.247151e-11, 0.0,          4.367791e-09, 2.826219e-10])

# Transition matrix from the K=4 HMM fit (k4_recheck.py).
TRANSITION = np.array([
    [0.9670, 0.0170, 0.0003, 0.0156],
    [0.0247, 0.9745, 0.0004, 0.0004],
    [0.0,    0.0,    0.9600, 0.0400],
    [0.0287, 0.0,    0.0118, 0.9595],
])

# Variance ranks (calm -> volatile): state1 < state0 < state3 < state2.
STATE_LABELS = {0: "calm", 1: "calmest", 2: "extreme", 3: "elevated"}
VOLATILE_STATES = [2, 3]  # above-median variance -- "elevated or worse"


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
    # Reference values from k4_recheck.py's original run: pre=0.165,
    # during-early=0.842, during-late=0.447 -- rerunning should reproduce
    # these closely (same data, same fixed parameters, deterministic).
    during = (index >= KNOWN_EVENT_DATE) & (index < KNOWN_EVENT_DATE + pd.Timedelta(days=10))
    print(f"\nMean P(state in {VOLATILE_STATES} = elevated/extreme) during 05-09 to 05-19: "
          f"{volatile_prob[during].mean():.3f}")
    during_early = (index >= KNOWN_EVENT_DATE) & (index < KNOWN_EVENT_DATE + pd.Timedelta(days=4))
    during_late = (index >= KNOWN_EVENT_DATE + pd.Timedelta(days=4)) & (index < KNOWN_EVENT_DATE + pd.Timedelta(days=10))
    print(f"  split: 05-09 to 05-13: {volatile_prob[during_early].mean():.3f} (reference: 0.842)")
    print(f"  split: 05-13 to 05-19: {volatile_prob[during_late].mean():.3f} (reference: 0.447)")
    pre = index < KNOWN_EVENT_DATE
    print(f"Mean P(state in {VOLATILE_STATES}) pre-event (05-01 to 05-09): "
          f"{volatile_prob[pre].mean():.3f} (reference: 0.165)")
    print("\nCompare this to the baseline filter's finding: a continuously-")
    print("updated Q re-absorbed the anomaly within about a day. Check whether")
    print("this elevated-probability plot instead stays high through the full")
    print("~05-09 to ~05-19 window the HMM decode identified, rather than")
    print("decaying back toward the pre-event level early.")


if __name__ == "__main__":
    main()