"""
Scalar Interacting Multiple Model (IMM) filter -- Blom & Bar-Shalom (1988).

Each of r models shares the same local-level structure (F=1, H=1) as the
baseline single-hypothesis filter, but carries its own FIXED (Q_j, R_j),
derived per-regime rather than continuously adapted. This is the structural
fix motivated in notes/baseline_filter_evaluation.md: a single continuously
updated Q re-absorbs anomalies into its own baseline; discrete competing
hypotheses instead require accumulated evidence to shift probability mass
between regimes.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class IMMResult:
    x_filt: np.ndarray        # combined (mode-averaged) state estimate
    P_filt: np.ndarray        # combined state variance
    mode_probs: np.ndarray    # shape (n, r) -- probability of each mode at each step


def run_imm(
    z: np.ndarray,
    Q: np.ndarray,      # shape (r,) -- fixed per-mode process noise
    R: np.ndarray,      # shape (r,) -- fixed per-mode measurement noise
    transition: np.ndarray,  # shape (r, r) -- Pi[i, j] = P(mode i -> mode j)
    x0: float | None = None,
    P0: float | None = None,
    mu0: np.ndarray | None = None,
) -> IMMResult:
    n = len(z)
    r = len(Q)

    if x0 is None:
        x0 = z[0]
    if P0 is None:
        P0 = float(np.mean(R[R > 0])) if np.any(R > 0) else 1e-8
    if mu0 is None:
        mu0 = np.full(r, 1.0 / r)  # uniform prior over modes -- a flagged,
        # simple default; with a sticky transition matrix this washes out
        # within the first several confident steps, but check that
        # assumption if results near t=0 matter to your conclusions.

    x_mode = np.full(r, x0)
    P_mode = np.full(r, P0)
    mu = mu0.copy()

    x_filt = np.empty(n)
    P_filt = np.empty(n)
    mode_probs = np.empty((n, r))

    for t in range(n):
        # --- 1. Mixing ---
        c = transition.T @ mu  # predicted mode probabilities, shape (r,)
        c = np.maximum(c, 1e-300)  # guard against exact zero before dividing
        W = (transition * mu[:, None]) / c[None, :]  # W[i, j] = P(mode i | mode j is active now)

        x0_mode = W.T @ x_mode  # shape (r,), mixed initial state per model j
        P0_mode = np.empty(r)
        for j in range(r):
            spread = x_mode - x0_mode[j]
            P0_mode[j] = np.sum(W[:, j] * (P_mode + spread ** 2))

        # --- 2. Mode-matched filtering (one KF step per model) ---
        x_pred = x0_mode                  # F = 1
        P_pred = P0_mode + Q

        y = z[t] - x_pred                 # H = 1, per-mode innovation
        S = P_pred + R                    # per-mode innovation variance
        K = P_pred / S

        x_mode = x_pred + K * y
        P_mode = (1 - K) * P_pred

        # Gaussian likelihood of the innovation under each model
        likelihood = np.exp(-0.5 * y ** 2 / S) / np.sqrt(2 * np.pi * S)

        # --- 3. Mode probability update ---
        numer = c * likelihood
        denom = np.sum(numer)
        if denom <= 0 or not np.isfinite(denom):
            # All models found this observation implausible under floating
            # point -- fall back to the predicted (mixing-only) probabilities
            # rather than dividing by ~zero. Worth investigating if this
            # fires often; it shouldn't for well-scaled Q/R.
            mu = c
        else:
            mu = numer / denom

        # --- 4. Combination ---
        x_hat = float(np.sum(mu * x_mode))
        P_hat = float(np.sum(mu * (P_mode + (x_mode - x_hat) ** 2)))

        x_filt[t] = x_hat
        P_filt[t] = P_hat
        mode_probs[t] = mu

    return IMMResult(x_filt, P_filt, mode_probs)