"""
Walk-forward calibration: refits isotonic regression daily on an
EXPANDING window of prior OOF predictions, applying it only to that day's
predictions -- mirroring the classifier's own purged walk-forward CV
cadence exactly, rather than fitting one static calibration map anywhere
and applying it forward (which failed three separate ways -- see
notes/phase3_calibration_drift.md).

Every calibrated point uses only information strictly prior to it: the
classifier's raw prediction was already walk-forward (purged CV), and now
the calibrator fit is too. No new leakage introduced at this step.

Usage:
    python scripts/calibrate_walk_forward.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

DATA_PATH = Path("data/processed/real_label_oof_predictions.parquet")
PLOTS_DIR = Path("plots")
OUT_PATH = Path("data/processed/real_label_walk_forward_calibrated.parquet")
N_BINS = 10
EPS = 1e-12
CALIBRATION_METHOD = "platt"  # "isotonic" or "platt". Four isotonic attempts
                                # (static time split, static regime split x2,
                                # rolling walk-forward) all made things WORSE.
                                # Common factor: isotonic is flexible/high-
                                # variance and may be overfitting small
                                # calibration windows rather than the window
                                # SHAPE being the problem. Platt scaling
                                # (2-parameter logistic fit) is far more
                                # stable on small, imbalanced samples -- test
                                # this hypothesis directly.
MIN_WARMUP_DAYS = 3  # require at least this many days of prior data before
                       # the first calibration attempt -- an isotonic fit
                       # on too little data (e.g. a single day) would be
                       # unstable, same reasoning as the classifier's own
                       # single-class-fold skip
ROLLING_WINDOW_DAYS = 5  # Defaulting to rolling, not expanding: a synthetic
                          # test with continuous drift showed rolling tracks
                          # it much better (peak |diff/SE| ~3.5 vs ~10.7 for
                          # expanding). 5 days is a starting guess, NOT a
                          # derived value -- try a few window lengths against
                          # the real reliability table/scores and pick based
                          # on what actually works here, the same way window
                          # lengths were derived elsewhere in this project
                          # (e.g. the Q rolling-window comparison). Set to
                          # None to use an expanding window instead.


def log_score(y_true, y_pred):
    p = np.clip(y_pred, EPS, 1 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true, y_pred):
    return float(np.mean((y_pred - y_true) ** 2))


def reliability_table(y_true, y_pred, n_bins=N_BINS):
    bin_edges = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-9
    bin_idx = np.digitize(y_pred, bin_edges[1:-1])
    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        mean_pred = y_pred[mask].mean()
        obs_freq = y_true[mask].mean()
        se = np.sqrt(obs_freq * (1 - obs_freq) / n_b)
        rows.append((mean_pred, obs_freq, n_b, se))
    return rows


def main():
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} not found -- run scripts/save_oof_predictions.py first")

    PLOTS_DIR.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA_PATH).sort_index()
    dates = df.index.normalize()
    unique_days = dates.unique().sort_values()

    calibrated = np.full(len(df), np.nan)
    n_calibrated_days = 0
    n_skipped_warmup = 0

    for day_i, day in enumerate(unique_days):
        if ROLLING_WINDOW_DAYS is None:
            prior_mask = np.asarray(dates < day)
        else:
            window_start = day - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
            prior_mask = np.asarray((dates < day) & (dates >= window_start))
        n_prior_days = day_i  # number of distinct prior days seen so far
        if n_prior_days < MIN_WARMUP_DAYS:
            n_skipped_warmup += 1
            continue

        prior = df[prior_mask]
        if len(prior["y_true"].unique()) < 2:
            continue

        today_mask = np.asarray(dates == day)
        today_raw_pred = df["y_pred_proba"].to_numpy()[today_mask]

        if CALIBRATION_METHOD == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(prior["y_pred_proba"].to_numpy(), prior["y_true"].to_numpy())
            calibrated[today_mask] = iso.predict(today_raw_pred)
        elif CALIBRATION_METHOD == "platt":
            # Platt scaling: logistic regression of true label on the RAW
            # predicted probability's logit -- 2 parameters (slope,
            # intercept), far less prone to overfitting a small calibration
            # window than isotonic's arbitrary monotonic step function.
            p_train = np.clip(prior["y_pred_proba"].to_numpy(), EPS, 1 - EPS)
            logit_train = np.log(p_train / (1 - p_train)).reshape(-1, 1)
            platt = LogisticRegression()
            platt.fit(logit_train, prior["y_true"].to_numpy())

            p_today = np.clip(today_raw_pred, EPS, 1 - EPS)
            logit_today = np.log(p_today / (1 - p_today)).reshape(-1, 1)
            calibrated[today_mask] = platt.predict_proba(logit_today)[:, 1]
        else:
            raise ValueError(f"Unknown CALIBRATION_METHOD: {CALIBRATION_METHOD}")

        n_calibrated_days += 1

    mode_label = (f"{CALIBRATION_METHOD}_" +
                  ("expanding" if ROLLING_WINDOW_DAYS is None else f"rolling_{ROLLING_WINDOW_DAYS}d"))
    print(f"Calibration mode: {mode_label}")

    valid = ~np.isnan(calibrated)
    print(f"{n_calibrated_days} days calibrated, {n_skipped_warmup} skipped "
          f"(warm-up), {len(df) - valid.sum() - n_skipped_warmup} skipped "
          f"(single-class prior)")
    print(f"{valid.sum():,} of {len(df):,} predictions successfully calibrated")

    y_true = df["y_true"].to_numpy()[valid]
    y_raw = df["y_pred_proba"].to_numpy()[valid]
    y_wf_cal = calibrated[valid]

    print(f"\n{'':>20}  {'raw':>10}  {'walk-forward calibrated':>24}")
    print(f"{'log score':>20}  {log_score(y_true, y_raw):>10.4f}  "
          f"{log_score(y_true, y_wf_cal):>24.4f}")
    print(f"{'Brier score':>20}  {brier_score(y_true, y_raw):>10.4f}  "
          f"{brier_score(y_true, y_wf_cal):>24.4f}")

    raw_table = reliability_table(y_true, y_raw)
    cal_table = reliability_table(y_true, y_wf_cal)

    print(f"\nRaw diff/SE per bin:            "
          f"{[f'{(o - p) / s:+.1f}' for p, o, n, s in raw_table]}")
    print(f"Walk-forward calibrated diff/SE: "
          f"{[f'{(o - p) / s:+.1f}' for p, o, n, s in cal_table]}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, table, title in [(axes[0], raw_table, "Raw"), (axes[1], cal_table, "Walk-forward calibrated")]:
        preds = [r[0] for r in table]
        obs = [r[1] for r in table]
        ses = [r[3] for r in table]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.errorbar(preds, obs, yerr=ses, fmt="o-", capsize=3)
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("mean predicted probability")
        ax.set_ylabel("observed frequency")
    fig.tight_layout()
    out_plot = PLOTS_DIR / f"walk_forward_calibration_{mode_label}.png"
    fig.savefig(out_plot, dpi=150)
    print(f"\nSaved plot to {out_plot}")

    out_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred_raw": y_raw,
        "y_pred_calibrated": y_wf_cal,
    }, index=df.index[valid])
    out_path = Path(f"data/processed/real_label_walk_forward_calibrated_{mode_label}.parquet")
    out_df.to_parquet(out_path)
    print(f"Saved to {out_path}")

    print("\nDecide, don't assume:")
    print("- Do diff/SE values drop toward 0 across most bins, unlike every prior")
    print("  static-split attempt (which all made things worse)?")
    print("- Do log score / Brier score improve over raw? If this ALSO fails, the")
    print("  problem may be that even daily refitting is too coarse relative to how")
    print("  fast calibration drifts -- consider a shorter refit cadence.")


if __name__ == "__main__":
    main()
