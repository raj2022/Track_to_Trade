"""
Merges multiple per-window Phase 2 datasets (built by build_phase2_dataset.py
on different, disjoint event windows) into one combined dataset spanning
genuinely diverse market regimes -- not just one crisis month.

Keeps only the columns needed for model comparison (features + real_label).
The per-window null-shift columns don't travel -- they served their purpose
in the already-completed Phase 2 leakage validation and aren't needed here.

Usage:
    python scripts/merge_multi_window_dataset.py
"""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed")
FEATURE_COLS = ["elevated_prob", "elevated_prob_lag1", "rolling_vol_1h"]
OUT_PATH = DATA_DIR / "phase2_dataset_combined.parquet"


def main():
    dataset_files = sorted(DATA_DIR.glob("phase2_dataset_BTCUSDT-aggTrades-*.parquet"))
    if not dataset_files:
        sys.exit(f"No per-window datasets found in {DATA_DIR} -- run "
                  f"build_phase2_dataset.py on each window first.")

    print(f"Found {len(dataset_files)} per-window datasets:")
    frames = []
    for path in dataset_files:
        df = pd.read_parquet(path)[FEATURE_COLS + ["real_label"]]
        window_label = path.stem.replace("phase2_dataset_BTCUSDT-aggTrades-", "")
        print(f"  {window_label}: {len(df):,} rows, "
              f"{df.index.min()} to {df.index.max()}, "
              f"real_label positive rate {df['real_label'].mean():.3f}")
        frames.append(df)

    combined = pd.concat(frames).sort_index()

    # Sanity check: confirm windows are genuinely disjoint (no overlapping
    # timestamps between windows, which would indicate the same window was
    # accidentally included twice, or two windows actually overlap in time).
    if combined.index.duplicated().any():
        n_dup = combined.index.duplicated().sum()
        print(f"\nWARNING: {n_dup} duplicate timestamps found after merging -- "
              f"check for accidental overlap between windows before trusting this dataset.")

    print(f"\nCombined dataset: {len(combined):,} rows across {len(dataset_files)} windows")
    print(f"Overall real_label positive rate: {combined['real_label'].mean():.3f}")
    print(f"Date range: {combined.index.min()} to {combined.index.max()}")

    # Report the calendar gaps between windows explicitly -- these are the
    # boundaries where the purged walk-forward splitter will be overly
    # conservative (safe, but wasteful) rather than precisely correct.
    unique_days = combined.index.normalize().unique().sort_values()
    gaps = unique_days.to_series().diff().dt.days
    boundary_gaps = gaps[gaps > 5]  # a >5-day gap signals a window boundary
    if len(boundary_gaps) > 0:
        print(f"\n{len(boundary_gaps)} window-boundary gaps detected (>5 days):")
        for date, gap in boundary_gaps.items():
            print(f"  {gap:.0f}-day gap ending {date.date()}")

    combined.to_parquet(OUT_PATH)
    print(f"\nSaved to {OUT_PATH}")
    print("\nCheck before trusting this for model comparison:")
    print("- Does the overall positive rate look like a sensible blend of the")
    print("  per-window rates printed above (not dominated by one window due to")
    print("  a row-count imbalance you didn't expect)?")
    print("- Are the reported gaps roughly what you'd expect given the windows")
    print("  chosen (multi-month gaps between disjoint crisis/calm periods)?")


if __name__ == "__main__":
    main()