import numpy as np
import pytest

from Inversion.traveltime.picking import (envelope, trace_snr, threshold_pick,
                               pick_first_break, pick_traveltimes)
from conftest import WAVEFORM_TYPES, gprmax_waveform, waveform_chi


def test_threshold_pick_between_onset_and_peak(ricker_trace):
    # The coarse crossing lags the onset (700) -- it needs the wavelet to reach a
    # tenth of peak -- but must land before the main lobe one chi (94 samples) later.
    trace, dt, t_onset = ricker_trace
    i = threshold_pick(trace, frac=0.1)
    chi = waveform_chi("ricker", 1.5e9) / dt
    assert 700 <= i <= 700 + chi


def test_aic_pick_close_to_onset(ricker_trace):
    # AIC refines the coarse crossing back onto the onset itself, well inside the
    # rise time chi that separates onset from main lobe.
    trace, dt, t_onset = ricker_trace
    chi = waveform_chi("ricker", 1.5e9)
    t_pick = pick_first_break(trace, dt, frac=0.1, refine="aic")
    assert abs(t_pick - t_onset) < 0.25 * chi


def test_pick_traveltimes_shape():
    # [n_src, n_rx, n_time] -> [n_src, n_rx]
    rng = np.random.default_rng(0)
    traces = rng.standard_normal((3, 4, 500))
    times = pick_traveltimes(traces, dt=1e-11)
    assert times.shape == (3, 4)


def test_flat_trace_returns_nan():
    # A dead receiver has no first break. Returning a number here would be a
    # silent failure: the AIC endpoint artifact (single-sample variance clipped
    # to 1e-30) makes k=1 win on a flat curve, yielding a bogus near-zero time.
    flat = np.zeros(500)
    assert np.isnan(pick_first_break(flat, dt=1e-11))


def test_pure_noise_returns_nan():
    # No arrival -> homogeneous variance -> no meaningful AIC minimum.
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1e-6, 2000)
    assert np.isnan(pick_first_break(noise, dt=1e-11))


@pytest.mark.parametrize("wave_type", WAVEFORM_TYPES)
def test_snr_gate_keeps_every_waveform_type(wave_type):
    # The gate must key off "is there an arrival", never off the shape or the
    # duration of the source. 'contsine' is the case that matters: it rings for
    # the rest of the record, so a median-based background would sit inside the
    # signal and reject it.
    dt, fc, n = 1e-11, 1.5e9, 2000
    trace = gprmax_waveform(wave_type, np.arange(n) * dt, fc, 700 * dt, dt)
    assert trace_snr(envelope(trace)) > 12.0
    assert np.isfinite(pick_first_break(trace, dt))


def test_snr_gate_is_duration_agnostic():
    # A long-ringing arrival and a one-sample impulse both pass, though they
    # occupy ~65% and ~0.6% of the trace respectively.
    dt, fc, n = 1e-11, 1.5e9, 2000
    t = np.arange(n) * dt
    long_lived = envelope(gprmax_waveform("contsine", t, fc, 700 * dt, dt))
    impulsive = envelope(gprmax_waveform("impulse", t, fc, 700 * dt, dt))
    assert (long_lived > 0.1 * long_lived.max()).mean() > 0.5
    assert (impulsive > 0.1 * impulsive.max()).mean() < 0.05
    assert trace_snr(long_lived) > 12.0
    assert trace_snr(impulsive) > 12.0


def test_snr_gate_keeps_arrival_buried_in_noise(ricker_trace):
    # A real but noisy arrival must survive the gate; only traces with *no*
    # arrival are dropped.
    trace, dt, t_true = ricker_trace
    rng = np.random.default_rng(1)
    noisy = trace + rng.normal(0.0, 0.05 * np.abs(trace).max(), trace.size)
    assert np.isfinite(pick_first_break(noisy, dt))


@pytest.mark.parametrize("fc,dt", [
    (50e6, 1e-10), (50e6, 1e-11),            # dt fixed by the FDTD grid, not by fc
    (100e6, 1e-10), (100e6, 1e-11),
    (250e6, 1e-10), (500e6, 5e-11),          # dt tracking the centre frequency
    (1e9, 2.5e-11), (1.5e9, 1e-11), (3e9, 1e-11), (3e9, 5e-12),
])
@pytest.mark.parametrize("wave_type", WAVEFORM_TYPES)
def test_pick_lag_is_constant_across_frequency(wave_type, fc, dt):
    # What makes the pipeline waveform- and frequency-agnostic is not that the pick
    # equals the onset, but that its offset from the onset is the SAME for every
    # ray -- calibrate_t0 absorbs a constant, not a drift. Regressions covered:
    # a search window wider than the lead-in (250 MHz: 115 samples of drift), and
    # one too narrow to reach back over a slow ramp (50 MHz contsine: 324).
    arrivals = np.linspace(11e-9, 14e-9, 6)          # crosshole traveltime span
    n = int((arrivals.max() + 8 / fc) / dt) + 50
    t = np.arange(n) * dt
    lags = np.array([(pick_first_break(gprmax_waveform(wave_type, t, fc, a, dt), dt) - a) / dt
                     for a in arrivals])
    assert np.isfinite(lags).all()
    assert lags.max() - lags.min() < 2.0


def test_pick_rejected_when_record_opens_mid_arrival(ricker_trace):
    # Cut the lead-in away so the record starts inside the wavelet. There is no
    # first break in it, so reject rather than return a plausible wrong time.
    trace, dt, t_onset = ricker_trace
    truncated = trace[int(t_onset / dt) + 60:]
    assert np.isnan(pick_first_break(truncated, dt))


def test_pick_traveltimes_propagates_nan(ricker_trace):
    trace, dt, _ = ricker_trace
    traces = np.tile(trace, (2, 3, 1))
    traces[1, 2] = 0.0                      # one dead receiver
    times = pick_traveltimes(traces, dt)
    assert np.isnan(times[1, 2])
    assert np.isfinite(np.delete(times.reshape(-1), 1 * 3 + 2)).all()
