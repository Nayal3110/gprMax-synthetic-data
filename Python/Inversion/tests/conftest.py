"""Shared fixtures for the Inversion test-suite.

Run from repo root:
    uv run python -m pytest Python/Inversion/tests/ -v
The loader test additionally needs gprMax importable; that path is added here too.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "Python"))
sys.path.insert(0, str(REPO_ROOT / "gprMax"))


from gprMax.waveforms import Waveform

# Every source shape gprMax can emit, bar 'user' (which needs a caller-supplied
# function). Tests parametrise over this so nothing is tuned to one wavelet.
WAVEFORM_TYPES = [t for t in Waveform.types if t != "user"]


def waveform_chi(wave_type, fc):
    """gprMax's chi coefficient: where the waveform's main lobe sits in its own
    time frame. Zero for the waveforms that simply start at t=0."""
    w = Waveform()
    w.type, w.freq = wave_type, fc
    w.calculate_coefficients()
    return w.chi


def gprmax_waveform(wave_type, time, fc, t_onset=0.0, dt=None, amp=1.0):
    """Sample any gprMax waveform on ``time``, using gprMax's own definition.

    Delegating to ``Waveform`` rather than restating the formulas keeps the tests
    exact by construction and covers every type in ``Waveform.types``.

    ``t_onset`` delays the source's own time origin, which is what a receiver at
    range actually sees: gprMax fires at t=0 and the trace arrives shifted by the
    traveltime, so the first break is at ``t_onset`` and the main lobe follows one
    ``chi`` later. The lead-in before the arrival is therefore the traveltime, and
    never shrinks as fc falls -- placing the *lobe* at the arrival instead would
    push the onset off the front of the record for a long, low-frequency wavelet.
    Samples before the origin are zero, so causality holds for the types gprMax
    defines only for t >= 0 ('sine', 'contsine', 'impulse').
    """
    w = Waveform()
    w.type, w.freq, w.amp = wave_type, fc, amp
    w.calculate_coefficients()
    time = np.asarray(time, float)
    dt = float(time[1] - time[0]) if dt is None else float(dt)
    shifted = time - t_onset
    try:
        # The gaussian/ricker families are pure numpy and evaluate in one shot.
        values = np.broadcast_to(w.calculate_value(shifted, dt), shifted.shape).copy()
    except (ValueError, TypeError):
        # 'sine'/'contsine'/'impulse' branch on scalar time; feed them sample by sample.
        values = np.array([w.calculate_value(float(s), dt) for s in shifted], float)
    return np.where(shifted >= 0.0, values, 0.0)


@pytest.fixture
def ricker_trace():
    """A 2000-sample Ricker arriving at a known time. Returns (trace, dt, t_onset);
    the main lobe sits one chi (~94 samples at 1.5 GHz) after ``t_onset``."""
    dt = 1e-11
    n = 2000
    fc = 1.5e9
    time = np.arange(n) * dt
    t_onset = 700 * dt
    return gprmax_waveform("ricker", time, fc, t_onset, dt), dt, t_onset
