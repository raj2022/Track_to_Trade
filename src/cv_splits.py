"""
Two cross-validation splitters, built to be compared directly against each
other in the leakage stress test:

  - naive_kfold_splits: standard contiguous k-fold, NO purging or embargo.
    Vulnerable to leakage whenever a training sample's label window
    (forward-looking, length H) or a training sample's feature window
    (backward-looking, length ~H) overlaps a test sample.

  - purged_embargoed_walk_forward_splits: expanding-window, always trains
    on strictly past data, PURGES training samples whose forward label
    window would overlap the test period, and EMBARGOES a further H-step
    buffer immediately after each test period from ever being used in a
    later fold's training set (guards against the backward-looking
    rolling_vol_1h feature leaking test-period information forward).

Both purge and embargo widths equal H -- not independently chosen values,
but each tied to a concrete mechanism (H = label horizon for purge; H =
feature lookback window for embargo, since rolling_vol_1h is a 1h window
and H is also 1h at this dt).
"""

import numpy as np
import pandas as pd


def naive_kfold_splits(index: pd.DatetimeIndex, n_splits: int):
    """Standard contiguous k-fold. No purging, no embargo -- deliberately
    the leaky baseline to compare against."""
    n = len(index)
    fold_bounds = np.linspace(0, n, n_splits + 1, dtype=int)
    all_idx = np.arange(n)
    for i in range(n_splits):
        test_idx = all_idx[fold_bounds[i]:fold_bounds[i + 1]]
        train_idx = np.setdiff1d(all_idx, test_idx)
        yield train_idx, test_idx


def purged_embargoed_walk_forward_splits(index: pd.DatetimeIndex, h: int, embargo: int):
    """
    Expanding-window walk-forward, one test fold per calendar day (after
    an initial warm-up period), with:
      - purge: drop training points t where t + h >= test_start
               (their forward label window would reach into the test set)
      - embargo: after each test fold, the following `embargo` points are
        excluded from ALL FUTURE folds' training sets too (not just the
        current one), since in an expanding window they would otherwise
        eventually be included as training data with contaminated
        backward-looking features.
    """
    n = len(index)
    dates = index.normalize()
    unique_days = dates.unique().sort_values()

    # Need at least a few days of warm-up before the first test fold.
    for day_i in range(1, len(unique_days)):
        test_start_date = unique_days[day_i]
        test_end_date = test_start_date + pd.Timedelta(days=1)

        test_mask = (dates >= test_start_date) & (dates < test_end_date)
        test_idx = np.where(test_mask)[0]
        if len(test_idx) == 0:
            continue

        test_start_pos = test_idx[0]
        test_end_pos = test_idx[-1]

        # Candidate training set: everything strictly before this test fold.
        candidate_train = np.arange(0, test_start_pos)

        # Purge: drop training points whose forward label window (t, t+h]
        # would overlap the test period.
        purge_cutoff = test_start_pos - h
        train_idx = candidate_train[candidate_train < purge_cutoff]

        # Embargo from PRIOR folds: any earlier test fold's embargo zone
        # (test_end + embargo) must also be excluded here, since embargoed
        # points are never safe to train on regardless of which fold.
        # Recompute directly: exclude any point within `embargo` steps
        # after ANY earlier test fold's end. Simplest correct approach:
        # exclude points in [prior_test_end, prior_test_end + embargo) for
        # every earlier day boundary.
        embargo_mask = np.zeros(len(train_idx), dtype=bool)
        for prior_day_i in range(1, day_i):
            prior_test_start_date = unique_days[prior_day_i]
            prior_test_end_date = prior_test_start_date + pd.Timedelta(days=1)
            prior_test_mask = (dates >= prior_test_start_date) & (dates < prior_test_end_date)
            prior_test_idx = np.where(prior_test_mask)[0]
            if len(prior_test_idx) == 0:
                continue
            prior_end_pos = prior_test_idx[-1]
            embargo_start = prior_end_pos + 1
            embargo_end = prior_end_pos + 1 + embargo
            embargo_mask |= (train_idx >= embargo_start) & (train_idx < embargo_end)

        train_idx = train_idx[~embargo_mask]

        if len(train_idx) == 0:
            continue

        yield train_idx, test_idx