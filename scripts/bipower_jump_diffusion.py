"""
Bipower variation (Barndorff-Nielsen & Shephard, 2004/2006) jump/diffusion
decomposition: separates continuous (diffusive) variance from discrete
price jumps within each day, using realized variance (RV, sensitive to
both) against bipower variation (BV, robust to jumps by construction).

RV_t   = sum_i r_i^2
BV_t   = (pi/2) * sum_i |r_i| * |r_{i-1}|          (attenuates jump contribution)
J_t    = max(RV_t - BV_t, 0)                        (jump component, floored at 0)

Statistical significance of the jump on each day is tested via the
ratio-statistic jump test (Huang & Tauchen, 2005; Barndorff-Nielsen &
Shephard, 2006), using realized tripower quarticity (TQ) for the test's
variance estimate -- theta = (pi/2)^2 + pi - 5 is a cited constant from
that derivation, not a fitted or hand-picked value.

Usage:
    python scripts/bipower_jump_diffusion.py data/raw/BTCUSDT-aggTrades-2022-05.zip
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

COLS = ["aggTradeId", "price", "quantity", "firstTradeId", "lastTradeId",
        "timestamp", "isBuyerMaker", "isBestMatch"]

DT_SEC = 60
THETA = (np.pi / 2) ** 2 + np.pi - 5  # cited constant, Barndorff-Nielsen & Shephard
MU1 = np.sqrt(2 / np.pi)              # E|Z|, Z ~ N(0,1) -- used in BV/TQ normalization
KNOWN_EVENT_DATE = pd.Timestamp("2022-05-09", tz="UTC")
PLOTS_DIR = Path("plots")

# Target false-positive rate for the daily jump test, same derivation
# philosophy as the chi-square gate in the baseline filter: pick a
# tolerance, derive alpha, don't state a significance level by convention.
TARGET_FALSE_POSITIVES_PER_MONTH = 1.0


def load_trades(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        inner_name = z.namelist()[0]
        with z.open(inner_name) as f:
            df = pd.read_csv(f, header=None, names=COLS,
                              usecols=["price", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def daily_bipower_decomposition(returns: pd.Series) -> pd.DataFrame:
    """
    Computes RV, BV, jump component, and the jump test z-statistic for
    each calendar day. `returns` must already be at a fixed dt (no gaps
    within a day assumed beyond what ffill in the caller already handled).
    """
    rows = []
    for date, day_returns in returns.groupby(returns.index.date):
        r = day_returns.to_numpy()
        n = len(r)
        if n < 10:
            continue

        rv = np.sum(r ** 2)
        bv = (np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
        jump = max(rv - bv, 0.0)

        # Realized tripower quarticity, for the jump test's variance term.
        mu_43 = MU1 ** (-3)  # normalizing constant, (E|Z|)^-3
        abs_r = np.abs(r)
        tq = n * mu_43 * np.sum(
            abs_r[2:] ** (4 / 3) * abs_r[1:-1] ** (4 / 3) * abs_r[:-2] ** (4 / 3)
        )

        # Ratio jump statistic (Huang & Tauchen 2005 / BNS 2006).
        if bv <= 0 or tq <= 0:
            z_stat = np.nan
        else:
            rj = (rv - bv) / rv
            denom = np.sqrt(THETA * max(1.0, tq / bv ** 2) / n)
            z_stat = rj / denom if denom > 0 else np.nan

        rows.append({"date": pd.Timestamp(date, tz="UTC"), "n": n, "RV": rv,
                       "BV": bv, "jump": jump, "z_stat": z_stat})

    return pd.DataFrame(rows).set_index("date")


def main():
    if len(sys.argv) != 2:
        print("Usage: python bipower_jump_diffusion.py <path_to_zip>")
        sys.exit(1)

    PLOTS_DIR.mkdir(exist_ok=True)

    df = load_trades(Path(sys.argv[1]))
    print(f"Loaded {len(df):,} trades")

    resampled = df["price"].resample(f"{DT_SEC}s").last().ffill()
    log_ret = np.log(resampled).diff().dropna()

    daily = daily_bipower_decomposition(log_ret)

    # Derive the z-statistic threshold from an explicit false-positive
    # tolerance across the month, rather than a stated significance level
    # like 0.05 or 0.01.
    n_days = len(daily)
    alpha = TARGET_FALSE_POSITIVES_PER_MONTH / n_days
    z_threshold = norm.ppf(1 - alpha)
    print(f"\nJump test: target {TARGET_FALSE_POSITIVES_PER_MONTH} false positive/month "
          f"over {n_days} days -> alpha={alpha:.4f}, z-threshold={z_threshold:.3f}")

    daily["jump_flag"] = daily["z_stat"] > z_threshold
    daily["jump_share"] = daily["jump"] / daily["RV"]

    print(f"\n{'date':>12}  {'RV':>12}  {'BV':>12}  {'jump':>12}  "
          f"{'jump_share':>10}  {'z_stat':>8}  {'flag'}")
    for date, row in daily.iterrows():
        flag = "***" if row["jump_flag"] else ""
        print(f"  {date.date()}  {row['RV']:>12.6e}  {row['BV']:>12.6e}  "
              f"{row['jump']:>12.6e}  {row['jump_share']:>9.1%}  "
              f"{row['z_stat']:>8.2f}  {flag}")

    n_flagged = daily["jump_flag"].sum()
    print(f"\n{n_flagged} of {n_days} days flagged as containing a statistically "
          f"significant jump (target was ~{TARGET_FALSE_POSITIVES_PER_MONTH:.0f}/month "
          f"under the null of no jumps).")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(daily.index, daily["RV"], label="RV (total)", marker="o", ms=3)
    ax1.plot(daily.index, daily["BV"], label="BV (diffusion-only)", marker="o", ms=3)
    ax1.axvline(KNOWN_EVENT_DATE, color="black", linestyle="--", alpha=0.6)
    ax1.set_ylabel("daily variance")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_title("Bipower variation jump/diffusion decomposition — BTCUSDT 2022-05")

    ax2.bar(daily.index, daily["jump_share"],
             color=["red" if f else "gray" for f in daily["jump_flag"]])
    ax2.axhline(0, color="black", lw=0.5)
    ax2.axvline(KNOWN_EVENT_DATE, color="black", linestyle="--", alpha=0.6)
    ax2.set_ylabel("jump share of RV")
    ax2.set_xlabel("Date")

    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = PLOTS_DIR / "bipower_jump_diffusion_2022-05.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
    print("\nCheck: do the statistically-flagged days (red bars) cluster around")
    print("2022-05-09 to -12 the way the IMM's extreme state did, or does jump")
    print("activity show up on different days than the IMM flagged as regime")
    print("shifts? A jump (discrete, discontinuous) and a regime SWITCH (sustained")
    print("elevated diffusion) are conceptually different things -- if the flagged")
    print("days don't line up, that's a genuine and useful distinction to write up,")
    print("not a discrepancy to explain away.")


if __name__ == "__main__":
    main()