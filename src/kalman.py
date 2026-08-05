"""
Scalar local-level Kalman filter, with a fixed measurement noise R and a
time-varying process noise Q_t series -- both derived quantities from prior
notes, not assumed.

Deliberately scoped to a 1D state (latent log-price only). A local-TREND
model (state = [level, drift]) is a natural extension but is NOT
implemented here, because its process noise has not been derived -- adding
it now would mean introducing an undeclared parameter, exactly the thing
this project is built to avoid. Extend only after deriving Q for the drift
component the same way Q was derived for the level.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KalmanFilter1DResult:
    x_filt: np.ndarray       # filtered state estimate at each step
    P_filt: np.ndarray       # filtered state variance at each step
    innovation: np.ndarray   # y_t = z_t - x_pred_t
    innovation_var: np.ndarray  # S_t = P_pred_t + R_t
    gate_stat: np.ndarray = field(init=False)  # y_t^2 / S_t, ~ chi2(1) under the null

    def __post_init__(self):
        self.gate_stat = self.innovation ** 2 / self.innovation_var


def run_local_level_filter(
    z: np.ndarray,
    Q: np.ndarray,
    R: float,
    x0: float | None = None,
    P0: float | None = None,
) -> KalmanFilter1DResult:
    """
    z:  observed log-price series, shape (n,)
    Q:  process noise PER STEP, shape (n,) -- already derived elsewhere as
        rolling_RV_rate * dt, one value per timestep. Must be aligned to z
        (same index/length) before calling this.
    R:  measurement noise variance, scalar, fixed (Roll's estimator).
    x0: initial state. Defaults to z[0] if not given.
    P0: initial state variance. Defaults to R if not given -- a simple,
        flagged-as-arbitrary starting choice (initial uncertainty about the
        true price is at least as large as one observation's noise). If
        this choice matters to your results, that's worth investigating
        directly (does the filter's behavior in the first ~few hours change
        meaningfully under a very different P0?) rather than assuming it
        washes out.
    """
    n = len(z)
    if len(Q) != n:
        raise ValueError(f"Q length ({len(Q)}) must match z length ({n})")

    x_filt = np.empty(n)
    P_filt = np.empty(n)
    innovation = np.empty(n)
    innovation_var = np.empty(n)

    x = x0 if x0 is not None else z[0]
    P = P0 if P0 is not None else R

    for t in range(n):
        # Predict
        x_pred = x            # F = 1 (random walk)
        P_pred = P + Q[t]

        # Update
        y = z[t] - x_pred     # H = 1
        S = P_pred + R
        K = P_pred / S

        x = x_pred + K * y
        P = (1 - K) * P_pred

        x_filt[t] = x
        P_filt[t] = P
        innovation[t] = y
        innovation_var[t] = S

    return KalmanFilter1DResult(x_filt, P_filt, innovation, innovation_var)


def derive_chi2_gate(target_false_alarms_per_day: float, steps_per_day: int) -> tuple[float, float]:
    """
    Derive the chi-square(1) gating threshold from an explicit tolerance on
    false alarms per day, rather than a stated significance level.

    Returns (alpha_per_step, threshold).
    """
    from scipy.stats import chi2

    alpha = target_false_alarms_per_day / steps_per_day
    threshold = chi2.ppf(1 - alpha, df=1)
    return alpha, threshold