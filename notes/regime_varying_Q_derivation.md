# Regime-varying process noise (Q) derivation — 2022-05

**Date:** Day 3/4
**Question:** What is the process noise Q for the Kalman filter's state
transition model — derived as a time-varying series that resolves the
LUNA/UST regime transition, rather than a single pooled value?

## Why a single pooled Q was rejected from the start

A single Q over all of 2022-05 would repeat the same pooling error already
caught twice in this project (the broken May signature plot; implicitly,
the April calm-window search): averaging a calm period together with a
genuine regime transition conflates a regime-dependent quantity (true
variance) into one number that represents neither state well. Q needed to
be a rolling series from the outset, not a scalar.

## Step 1: re-validate the sampling interval for this month

Did not assume the 2023-06 calm-window plateau (600-3600s) transfers to
May 2022 — trade frequency during a crisis could plausibly differ from
ordinary trading, changing where the staleness artifact bites.

Checked staleness split at the known LUNA depeg date (2022-05-09, UTC):

| interval | pre-event stale% | post-event stale% |
|---|---|---|
| 10s  | 12.4% | 8.6% |
| 30s  | 3.2%  | 2.5% |
| 60s  | 0.9%  | 0.8% |
| 300s | 0.0%  | 0.1% |
| 600s | 0.0%  | 0.0% |

**Finding:** pre- and post-event staleness are close to each other at every
interval — trade frequency did not change dramatically across the regime
shift. This is itself a real check that could have failed (it would have
meant a single fixed dt couldn't honestly apply across the whole series)
and didn't — worth recording as a passed check, not just proceeding
silently.

**dt = 60s selected.** 30s is nominally "single-digit" (3.2%/2.5%) but close
enough to the edge that sub-periods within the month could push above it;
60s is robustly low on both sides (0.9%/0.8%); 300s+ buys nothing further
(already ~0%) while coarsening resolution. Shortest interval that is
*robustly*, not marginally, low on both sides of the event.

Script: `scripts/rolling_q.py`

## Step 2: rolling window length

Computed rolling realized-variance rate at dt=60s across five candidate
window lengths (1h, 4h, 12h, 1D, 3D), plotted against the known LUNA depeg
date, and judged by the same bias/variance logic as the signature plot:
shortest window that is visibly *stable pre-event* while still *tracking
the transition sharply*.

- **1h, 4h:** noise-dominated throughout, including well before the event —
  large jagged spikes even during ordinary trading. Rejected: this is
  estimation noise, not regime signal.
- **12h:** damped relative to 4h but still visibly tracks the same jagged
  pattern at lower amplitude — not genuinely stable pre-event.
- **1D:** first window that is genuinely smooth before 2022-05-09, settling
  into a slow trend rather than oscillating, while still resolving the
  May 9-13 rise clearly.
- **3D:** over-smooths — the peak is visibly delayed to ~May 15-17 and the
  rise/decline is smeared across nearly two weeks, well past when the
  market had stabilized.

**1D selected**, as the shortest window that is stable pre-event rather than
merely less noisy than the noisiest option.

**Property to record, not a flaw:** a trailing 1D rolling window has an
inherent lag — it takes up to a full day of new data to fully reflect a
regime shift, which is part of why the visible peak sits a few days after
the actual depeg date. This is the correct behavior for a quantity meant to
calibrate a real-time filter, which can also only see the past — not
something to "fix" with a centered window.

## Step 3: diurnal-periodicity check (ruling out, not confirming, a hypothesis)

The 1h/4h/12h series showed a roughly periodic wiggle even during the calm
pre-event stretch. Hypothesized this could be a ~24h diurnal
(trading-session) effect, and checked rather than assumed it — computed the
ACF of the 1h rolling RV-rate series out to 48h.

**Result: the 24h hypothesis is ruled out.** The ACF at lag=24h sits on a
monotonic decline from the ~16h region, not at a local peak. Instead, the
ACF shows an unambiguous local maximum at **~8h** (ACF ≈ 0.56, clearly above
its 6h/10h neighbors), with a smaller secondary peak at **~16h** (≈0.42,
consistent with a decaying 2×8h harmonic) and no meaningful structure near
24h or 48h.

**Working hypothesis, not confirmed:** Binance perpetual futures settle
funding every 8 hours (00:00/08:00/16:00 UTC). This is spot data, but
arbitrageurs hedging funding exposure by trading spot against the perpetual
could inject an 8h-periodic component into spot trading intensity even
though the underlying mechanism lives on a different product. Not verified
here — would need to check trading-intensity clustering directly around
those three UTC timestamps before asserting this as fact. Recorded as an
open question, not a conclusion.

**Correction to the Step 2 reasoning:** the 1D window is not smoother
because it specifically cancels a 24h cycle — that explanation is ruled out
by this check. The correct explanation is simpler: 1D is long relative to
the ~8h decorrelation timescale, so it averages over several cycles
regardless of exact period alignment. This is the general
variance-reduction-by-averaging argument, not a period-matching coincidence
— noted here as a correction to the initial, tidier-sounding but wrong
story, not silently folded in as if it were the plan all along.

**Flag for later phases:** if an 8h-periodic component in trading
intensity/volatility is real, a regime classifier that doesn't account for
it could end up partially "detecting" time-of-day structure rather than
genuine regime change — the same conflation-of-populations failure mode
already caught twice in this project (April calm-window pooling; the
May signature plot), in a new form. Worth an explicit check in Phase 2.

## Result

Regime-varying Q for 2022-05: rolling realized-variance rate at dt=60s,
1D trailing window, computed via `scripts/rolling_q.py`. Feeds the filter
as a time-varying process-noise input rather than a single scalar.

## Reproducibility

- Scripts: `scripts/rolling_q.py`, `scripts/diurnal_check.py`
- Commands:
  - `python scripts/rolling_q.py data/raw/BTCUSDT-aggTrades-2022-05.zip` (dt=60 entered at prompt)
  - `python scripts/diurnal_check.py data/raw/BTCUSDT-aggTrades-2022-05.zip`

## Status: Q and R both closed

- **R** (measurement noise): 3.661×10⁻¹¹, Roll's estimator on the verified
  calm window (2023-06-13 to 2023-06-19) — see `microstructure_noise_R_derivation.md`.
- **Q** (process noise): rolling realized-variance rate, dt=60s, 1D window,
  derived on 2022-05 (LUNA) as above.

## Next step

Baseline model (per the Phase 1 revision) and the IMM extension, using
these as the calibrated inputs.
