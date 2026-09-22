"""First-arrival (first-break) travel-time picking for GPR traces.

Strategy: a fast amplitude/envelope threshold gives a coarse onset, then an
Akaike Information Criterion (AIC) picker refines it within a local window.
The AIC minimum marks the split between variance regimes
and lands on the true onset.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def envelope(trace: np.ndarray) -> np.ndarray:
    """Analytic-signal (Hilbert) amplitude envelope of a trace."""
    return np.abs(signal.hilbert(np.asarray(trace, float)))


def quiet_level(env: np.ndarray, win_frac: float = 0.02) -> float:
    """Amplitude of the quietest stretch of the record: the smallest median over
    consecutive windows of ``win_frac`` of the trace.

    Taking the minimum over windows rather than a low quantile of the whole trace
    keeps the estimate independent of how much of the record the arrival occupies.
    A quantile silently fails once the lead-in is shorter than the quantile itself:
    a 50 MHz 'contsine' fills over 90% of its record, putting the 10% quantile
    inside the signal and collapsing its apparent SNR to 7.8, below the noise
    ceiling. The windowed minimum reads 27.9 for the same trace.
    """
    env = np.asarray(env, float)
    win = max(16, int(env.size * win_frac))
    n_win = env.size // win
    if n_win < 2:
        return float(np.quantile(env, 0.1))          # too short to window
    return float(np.median(env[:n_win * win].reshape(n_win, win), axis=1).min())


def trace_snr(env: np.ndarray, win_frac: float = 0.02) -> float:
    """Envelope peak over its quiet level: does this trace hold an arrival?

    Measured over every gprMax waveform type from 50 MHz to 3 GHz:

    * arrival-free noise stays under 8, and it falls with trace length (7.8 at 
      n=500, 5.0 at n=20000) rather than rising;
    * the weakest clean arrival scores 28 (50 MHz contsine), most score 1e2-1e3;
    * a dead trace returns 0.0.

    The zero-background guard returns inf. A Hilbert envelope rarely triggers it,
    since its 1/t tails keep the envelope nonzero, but a discrete impulse can:
    its Hilbert transform vanishes at every even sample offset.
    """
    env = np.asarray(env, float)
    peak = float(env.max())
    if peak == 0.0:
        return 0.0
    background = quiet_level(env, win_frac)
    return np.inf if background == 0.0 else peak / background


def threshold_pick(trace: np.ndarray, frac: float = 0.1, floor: float = 0.0) -> int:
    """Index of the first sample exceeding both ``frac`` * max and ``floor``.

    Runs on ``abs(trace)`` rather than the Hilbert envelope because the analytic
    signal is not causal: a long, energetic arrival leaks envelope amplitude
    backwards across the whole record, and for a source that rings for the rest
    of the trace ('contsine') that leakage reaches 71% of peak at sample 0 --
    tripping an envelope threshold immediately and returning index 0. The
    rectified trace is exactly zero until the wave arrives, for every waveform.

    Rectifying costs the envelope's smoothing, so on a noisy trace a fraction of
    peak alone is crossed by the first noise spike. ``floor`` is the absolute
    amplitude the crossing must also clear; pass the measured noise level.
    """
    s = np.abs(np.asarray(trace, float))
    peak = s.max()
    if peak == 0.0:
        return 0
    over = s > max(frac * peak, floor)
    return int(np.argmax(over)) if over.any() else 0


def aic_curve(x: np.ndarray):
    """Vectorised AIC over all split points k of ``x``. Returns (aic[k], k_values)."""
    x = np.asarray(x, float)
    n = x.size
    k = np.arange(1, n)
    csum = np.cumsum(x)
    csum2 = np.cumsum(x * x)
    kf = k.astype(float)
    mean1 = csum[:-1] / kf
    var1 = np.clip(csum2[:-1] / kf - mean1 ** 2, 1e-30, None)
    rc = (n - k).astype(float)
    rsum = csum[-1] - csum[:-1]
    rsum2 = csum2[-1] - csum2[:-1]
    mean2 = rsum / rc
    var2 = np.clip(rsum2 / rc - mean2 ** 2, 1e-30, None)
    aic = kf * np.log(var1) + rc * np.log(var2)
    return aic, k


def pick_first_break(trace: np.ndarray, dt: float, frac: float = 0.1,
                     margin: int = 4, refine: str = "aic", t0: float = 0.0,
                     min_snr: float = 12.0, win_frac: float = 0.02,
                     floor_k: float = 5.0) -> float:
    """Return first-break time (s), or NaN if the trace carries no arrival.

    ``t0`` is a calibration offset subtracted from the pick. Traces whose
    ``trace_snr`` falls below ``min_snr`` (dead receivers, rays attenuated below
    the noise floor) are rejected rather than picked: on such a trace the AIC
    curve is flat and its minimum is decided by the single-sample variance
    clipped to 1e-30 at k=1 / k=n-1, which would return a confident but
    meaningless near-edge time.

    The default sits above the arrival-free noise ceiling rather than midway,
    deliberately preferring a dropped ray -- which the solvers absorb -- over a
    fabricated traveltime steering the reconstruction. The cost is that arrivals
    buried under more than roughly 20-40% rms noise are discarded too; lower
    ``min_snr`` if real data proves quieter than that.

    ``margin`` is how far past the coarse crossing the AIC
    search may run, in samples, and needs only to be a few. A pick is returned only
    when the onset falls strictly inside the search window, so a record that opens
    part-way through the arrival yields NaN rather than a plausible wrong time.
    """
    trace = np.asarray(trace, float)
    env = envelope(trace)
    if trace_snr(env, win_frac=win_frac) < min_snr:
        return np.nan
    # Rectifying loses the envelope's smoothing, so the coarse crossing must clear
    # the measured noise level too. The envelope of Gaussian noise is Rayleigh, whose
    # peak over n samples grows as sigma*sqrt(2 ln n) -- 3.9 sigma at n=2000, and only
    # 4.6 at n=20000, so one constant covers every record length. But quiet_level is a
    # *minimum* over ~20 window medians, which sits well below the 1.18 sigma Rayleigh
    # median: measured 0.84 sigma at n=2000 (rising with window size: 0.73 at n=500,
    # 1.06 at n=20000). So floor_k = 5 is an effective 4.2 sigma against a 4.0 sigma
    # peak, not the 5.9 the Rayleigh median would suggest, and ~10% of pure-noise
    # traces still cross it. That is tolerable only because the min_snr gate above has
    # already rejected arrival-free traces: this floor guards the marginal band
    # 12 < snr < 50, below which the gate fires and above which frac*peak dominates.
    # Note the coupling: shrinking win_frac adds windows, lowers quiet_level, and
    # silently loosens this floor.
    background = quiet_level(env, win_frac)
    i0 = threshold_pick(trace, frac=frac, floor=floor_k * background)
    if i0 == 0:
        return np.nan          # record opens mid-arrival: no lead-in, no first break
    if refine == "aic" and trace.size > 3:
        # Search the lead-in, not a span centred on the crossing. The onset always
        # precedes i0 -- by one sample for an impulse, by a slow ramp's entire rise
        # for a low-frequency 'contsine' (230 samples at 50 MHz) -- so no fixed
        # symmetric window reaches back over every waveform. Extra noise to the left
        # costs AIC nothing, since noise-then-signal is exactly the split it looks
        # for; it is signal to the *right* that offers competing later splits, so
        # the window stops just past the crossing.
        lo, hi = 1, min(trace.size, i0 + margin)
        if hi - lo > 3:
            aic, k = aic_curve(trace[lo:hi])
            j = int(np.argmin(aic))
            if j == 0 or j == k.size - 1:
                # The minimum is the single-sample variance artifact at the window
                # edge, so the onset is not inside the window -- no usable pick.
                return np.nan
            i0 = lo + int(k[j])
    return i0 * dt - t0


def pick_traveltimes(traces: np.ndarray, dt: float, t0: float = 0.0, **kw) -> np.ndarray:
    """Pick a first-break time for every trace in a [n_src, n_rx, n_time] array.

    Entries for traces with no detectable arrival are NaN; callers must mask them
    (``run_tomography`` drops those rays from the system).
    """
    traces = np.asarray(traces, float)
    n_src, n_rx, _ = traces.shape
    out = np.empty((n_src, n_rx), dtype=float)
    for si in range(n_src):
        for ri in range(n_rx):
            out[si, ri] = pick_first_break(traces[si, ri], dt, t0=t0, **kw)
    return out
