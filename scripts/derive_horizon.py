"""
Derive the Phase 2 label horizon H from the K=4 IMM's own transition
matrix, rather than picking a round number.

Treats the "elevated" states as transient states of an absorbing Markov
chain (absorption = exiting to a calm state) and computes the expected
sojourn time in the elevated set via the fundamental matrix N = (I-Q)^-1,
weighted by the quasi-stationary distribution over the transient states
(not a naive average of the individual states' self-transition-implied
dwell times, which would ignore the fact that one state may only be
reachable/exitable through the other).

Usage:
    python scripts/derive_horizon.py
"""

import numpy as np

# K=4 transition matrix, adopted model (scripts/k4_recheck.py output).
TRANSITION = np.array([
    [0.967,  0.017,  0.0003, 0.0156],
    [0.0247, 0.9745, 0.0004, 0.0004],
    [0.,     0.,     0.96,   0.04],
    [0.0287, 0.,     0.0118, 0.9595],
])

# States treated as "elevated" for the label (above-median variance, per
# scripts/k4_recheck.py's classification).
ELEVATED_STATES = [2, 3]

DT_SEC = 60


def derive_horizon(transition: np.ndarray, elevated_states: list, dt_sec: int) -> dict:
    Q = transition[np.ix_(elevated_states, elevated_states)]
    exit_prob = 1 - Q.sum(axis=1)

    n = len(elevated_states)
    N = np.linalg.inv(np.eye(n) - Q)
    expected_sojourn = N.sum(axis=1)  # steps, per starting state

    # Quasi-stationary distribution: dominant left eigenvector of Q.
    eigvals, eigvecs = np.linalg.eig(Q.T)
    idx = np.argmax(eigvals.real)
    quasi_stationary = eigvecs[:, idx].real
    quasi_stationary = np.abs(quasi_stationary) / np.abs(quasi_stationary).sum()

    H_steps = float(quasi_stationary @ expected_sojourn)

    return {
        "Q": Q,
        "exit_prob": exit_prob,
        "N": N,
        "expected_sojourn_steps": dict(zip(elevated_states, expected_sojourn)),
        "quasi_stationary": dict(zip(elevated_states, quasi_stationary)),
        "H_steps": H_steps,
        "H_minutes": H_steps * dt_sec / 60,
        "H_hours": H_steps * dt_sec / 3600,
    }


def main():
    result = derive_horizon(TRANSITION, ELEVATED_STATES, DT_SEC)

    print(f"Elevated states: {ELEVATED_STATES}")
    print(f"\nSub-transition matrix Q (within elevated set):")
    print(result["Q"])
    print(f"\nPer-step exit probability (1 - row sum):")
    for state, p in zip(ELEVATED_STATES, result["exit_prob"]):
        print(f"  state {state}: {p:.4f}")

    print(f"\nExpected sojourn time in elevated set, by starting state:")
    for state, steps in result["expected_sojourn_steps"].items():
        minutes = steps * DT_SEC / 60
        print(f"  starting from state {state}: {steps:.2f} steps ({minutes:.1f} min)")

    print(f"\nQuasi-stationary distribution over the elevated set:")
    for state, p in result["quasi_stationary"].items():
        print(f"  state {state}: {p:.3f}")

    print(f"\nDerived H = {result['H_steps']:.2f} steps "
          f"= {result['H_minutes']:.1f} minutes "
          f"= {result['H_hours']:.2f} hours")
    print(f"\nRounded for practical use: H = 60 steps (60 minutes / 1 hour) -- "
          f"rounding UP from the exact value, since underestimating H risks a")
    print(f"label window that closes before a real elevated episode has resolved,")
    print(f"a worse failure mode than a slightly generous window.")
    print(f"\nNote: this describes a TYPICAL elevated episode under the fitted")
    print(f"dynamics, not the LUNA event specifically (which persisted for days --")
    print(f"an extreme outlier relative to this baseline). That's intentional: the")
    print(f"label horizon should match the common case across the full dataset,")
    print(f"not be implicitly tuned to one exceptional event.")


if __name__ == "__main__":
    main()