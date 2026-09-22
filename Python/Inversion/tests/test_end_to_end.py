import numpy as np
import pytest

from Inversion.traveltime.geometry import PixelGrid, build_system_matrix
from Inversion.traveltime.loaders import Acquisition, AcquisitionData
from Inversion.traveltime.sirt import eps_to_slowness, slowness_to_eps, forward
from Inversion.traveltime.tomography import calibrate_t0, run_tomography
from Inversion.traveltime.picking import pick_traveltimes
from conftest import WAVEFORM_TYPES, gprmax_waveform

# The source waveform is not part of the inversion model: the picker keys off the
# first break and calibrate_t0 absorbs any constant onset-to-pick lag, so the
# pipeline must recover the same eps map whatever gprMax emitted -- an impulse, a
# wavelet, or a source that keeps ringing for the rest of the record ('contsine').


def _synthetic_data(eps_map, grid, acq, dt=1e-11, n_time=2000, fc=1.5e9, t0=3e-9,
                    wave_type="ricker"):
    """Build traces arriving at t0 + straight-ray traveltime, as gprMax would: the
    source fires once and each receiver sees it delayed, so the first break is at
    the arrival and the main lobe follows one chi later.

    Any gprMax waveform type works: the picker reports a constant onset-to-pick
    lag for a given shape, which ``calibrate_t0`` absorbs along with ``t0``.
    """
    L, rays = build_system_matrix(acq.sources, acq.receivers, grid)
    m_true = eps_to_slowness(eps_map)
    tt = forward(L, m_true)                       # [n_rays]
    n_src, n_rx = acq.sources.shape[0], acq.receivers.shape[0]
    t = np.arange(n_time) * dt
    traces = np.zeros((n_src, n_rx, n_time))
    for ray_idx, (si, ri) in enumerate(rays):
        traces[si, ri] = gprmax_waveform(wave_type, t, fc, t0 + tt[ray_idx], dt)
    return AcquisitionData(acq=acq, traces=traces, dt=dt, component="Ez")


def _geom():
    ys = np.linspace(0.05, 0.95, 6)
    acq = Acquisition(sources=np.column_stack([np.full(6, 0.0), ys]),
                      receivers=np.column_stack([np.full(6, 1.0), ys]))
    grid = PixelGrid(0.0, 1.0, 0.0, 1.0, nx=6, ny=6)
    return acq, grid


def test_calibrate_t0_centers_residuals():
    # calibrate_t0 returns the constant offset that makes picked times match the
    # straight-ray traveltimes through the known-eps medium; subtracting it must
    # center the residuals on the predicted traveltimes (within pick scatter).
    acq, grid = _geom()
    data = _synthetic_data(np.full(grid.n_pixels, 6.0), grid, acq, t0=3e-9)
    times = pick_traveltimes(data.traces, data.dt, frac=0.1)
    t0 = calibrate_t0(times, acq, grid, eps_known=6.0)
    L, _ = build_system_matrix(acq.sources, acq.receivers, grid)
    predicted = (L @ np.full(grid.n_pixels, eps_to_slowness(6.0))).reshape(
        acq.sources.shape[0], acq.receivers.shape[0])
    assert np.allclose(times - t0, predicted, atol=20 * data.dt)


def test_calibrate_t0_ignores_dead_traces():
    # A dead trace picks as NaN; t0 must come from the surviving rays only.
    acq, grid = _geom()
    data = _synthetic_data(np.full(grid.n_pixels, 6.0), grid, acq, t0=3e-9)
    times = pick_traveltimes(data.traces, data.dt, frac=0.1)
    t0_all = calibrate_t0(times, acq, grid, eps_known=6.0)
    times[2, 3] = np.nan
    t0_gap = calibrate_t0(times, acq, grid, eps_known=6.0)
    assert np.isfinite(t0_gap)
    assert t0_gap == pytest.approx(t0_all, abs=5 * data.dt)


def test_run_tomography_drops_dead_traces():
    # Zeroed traces carry no arrival; those rays must be removed from the system
    # rather than fitted to a bogus near-zero traveltime.
    acq, grid = _geom()
    data = _synthetic_data(np.full(grid.n_pixels, 6.0), grid, acq, t0=3e-9)
    data.traces[1, 4] = 0.0
    data.traces[3, 0] = 0.0
    res = run_tomography(data, grid, solver="sirt", n_iter=300, relax=0.2,
                         calibrate_eps=6.0, picker_kwargs={"frac": 0.1})
    assert res["n_dropped"] == 2
    assert res["L"].shape[0] == 36 - 2
    assert np.isnan(res["times"][1, 4])
    assert np.mean(res["eps_map"]) == pytest.approx(6.0, abs=0.5)


def test_run_tomography_all_dead_raises():
    acq, grid = _geom()
    data = _synthetic_data(np.full(grid.n_pixels, 6.0), grid, acq, t0=3e-9)
    data.traces[:] = 0.0
    with pytest.raises(ValueError, match="no usable"):
        run_tomography(data, grid, calibrate_eps=6.0)


@pytest.mark.parametrize("fc,dt,n_time", [
    (50e6, 1e-10, 2200), (250e6, 1e-10, 700), (1e9, 2.5e-11, 1000), (3e9, 1e-11, 1800),
])
@pytest.mark.parametrize("wave_type", ["ricker", "contsine", "impulse"])
def test_run_tomography_across_frequency(wave_type, fc, dt, n_time):
    # Recovery must hold over the GPR band, not just at the 1.5 GHz the picker was
    # first tuned at. contsine and impulse bracket the extremes of source duration.
    acq, grid = _geom()
    data = _synthetic_data(np.full(grid.n_pixels, 6.0), grid, acq, dt=dt, n_time=n_time,
                           fc=fc, t0=3e-9, wave_type=wave_type)
    res = run_tomography(data, grid, solver="sirt", n_iter=300, relax=0.2,
                         calibrate_eps=6.0, picker_kwargs={"frac": 0.1})
    assert res["n_dropped"] == 0
    assert np.mean(res["eps_map"]) == pytest.approx(6.0, abs=0.5)


@pytest.mark.parametrize("wave_type", WAVEFORM_TYPES)
def test_run_tomography_homogeneous(wave_type):
    acq, grid = _geom()
    data = _synthetic_data(np.full(grid.n_pixels, 6.0), grid, acq, t0=3e-9,
                           wave_type=wave_type)
    res = run_tomography(data, grid, solver="sirt", n_iter=300, relax=0.2,
                         calibrate_eps=6.0, picker_kwargs={"frac": 0.1})
    assert np.mean(res["eps_map"]) == pytest.approx(6.0, abs=0.5)
    assert res["eps_map"].shape == (grid.ny, grid.nx)
