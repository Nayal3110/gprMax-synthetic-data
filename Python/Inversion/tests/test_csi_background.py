"""Calibration and background estimation.

The load-bearing tests here are the pair that pin *both* estimators at once:
``test_real_merged_soil_recovers_eps_6`` shows the phasor stack reading 6.0 on
the real panel over its whole band, and
``test_phase_slope_diagnostic_wraps_above_1_ghz`` shows the unwrap-based
diagnostic collapsing on the same data above ~0.9 GHz. Anyone tempted to
"simplify" the stack into an unwrap has to break the second test to do it.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as C_LIGHT

from Inversion.CSI import EPS_MIN, EPS_WATER_MAX
from Inversion.CSI.background import (
    COHERENCE_GATE, Background, coherence, estimate_background, green_2d,
    model_field, offsets, phase_slope_permittivity,
)
from Inversion.CSI.calibration import (
    calibrate_air, calibrate_wavelet, wavelet_estimate,
)
from Inversion.CSI.operators import background_wavenumber
from Inversion.CSI.spectra import Spectra
from Inversion.traveltime.loaders import bscan_geometry


REPO_ROOT = Path(__file__).resolve().parents[3]
MERGED_SOIL = (REPO_ROOT / "gprMax" / "user_models" / "test_B_array_soil"
               / "Simple_soil" / "test_B_array_soil_merged.out")

# Known geometry of the test_B_array runs: 9 sources stepping +0.1 m in y from
# (0.10, 0.10), 9 receivers likewise from (0.90, 0.10). Truth eps = 6.0.
B_ARRAY_ACQ = dict(src_start=(0.1, 0.1), src_step=(0.0, 0.1), n_src=9,
                   rx_start=(0.9, 0.1), rx_step=(0.0, 0.1), n_rx=9)
B_ARRAY_EPS = 6.0


# --------------------------------------------------------------------------
# synthetic panels
# --------------------------------------------------------------------------

def _panel(n_src=5, n_rx=6):
    """A crosshole fan 0.8 m wide, the shape of the test_B_array geometry."""
    src = np.column_stack([np.full(n_src, 0.10), 0.10 + 0.15 * np.arange(n_src)])
    rx = np.column_stack([np.full(n_rx, 0.90), 0.10 + 0.13 * np.arange(n_rx)])
    return src, rx


def _freqs(n=24, lo=2.0e8, hi=1.2e9):
    return np.linspace(lo, hi, n)


def _homogeneous(eps=6.0, sigma=0.0, n_src=5, n_rx=6, n_freq=24):
    """``[n_src, n_rx, n_freq]`` field of a homogeneous panel, unit source."""
    src, rx = _panel(n_src, n_rx)
    freqs = _freqs(n_freq)
    return model_field(offsets(src, rx), freqs, eps, sigma), freqs, src, rx


def _heterogeneous(seed=3, spread=0.5, n_src=5, n_rx=6, n_freq=24):
    """Each pair travels through its own permittivity -- no single eps fits."""
    src, rx = _panel(n_src, n_rx)
    freqs = _freqs(n_freq)
    dists = offsets(src, rx)
    rng = np.random.default_rng(seed)
    eps_pair = 6.0 * (1.0 + spread * rng.standard_normal(dists.shape))
    eps_pair = np.clip(eps_pair, 1.5, 20.0)

    values = np.empty(dists.shape + freqs.shape, dtype=np.complex128)
    for s in range(dists.shape[0]):
        for r in range(dists.shape[1]):
            k = np.array([background_wavenumber(2 * np.pi * f, float(eps_pair[s, r]))
                          for f in freqs])
            values[s, r] = green_2d(k, np.full(freqs.shape, dists[s, r]))
    return values, freqs, src, rx


def _wavelet(freqs, seed=0):
    """A complex per-frequency scale with real phase structure, like a wavelet."""
    rng = np.random.default_rng(seed)
    mag = 1.0 + rng.random(freqs.shape)
    phase = -2.0 * np.pi * freqs * 3e-9 + 0.4 * rng.standard_normal(freqs.shape)
    return mag * np.exp(1j * phase)


# --------------------------------------------------------------------------
# wavelet calibration: exact round trip in all three modes
# --------------------------------------------------------------------------

@pytest.mark.parametrize("per", ["global", "source", "pair"])
def test_wavelet_round_trip_is_exact(per):
    """Apply a known complex scale, recover it, and land back on the model."""
    bg, freqs, _, _ = _homogeneous()
    n_src, n_rx, n_freq = bg.shape
    rng = np.random.default_rng(11)

    if per == "global":
        scale = _wavelet(freqs)
        applied = scale[None, None, :]
    elif per == "source":
        scale = np.stack([_wavelet(freqs, seed=s) for s in range(n_src)])[:, None, :]
        applied = scale
    else:
        scale = (rng.standard_normal(bg.shape) + 1j * rng.standard_normal(bg.shape))
        applied = scale

    meas = applied * bg

    estimated = wavelet_estimate(meas, bg, per=per, freqs=freqs)
    assert estimated.shape == np.squeeze(scale, axis=(0, 1)).shape if per == "global" \
        else estimated.shape == scale.shape
    assert np.allclose(estimated, np.squeeze(scale, axis=(0, 1)) if per == "global"
                       else scale, rtol=0, atol=1e-12)

    cal = calibrate_wavelet(meas, bg, per=per, freqs=freqs)
    assert isinstance(cal, Spectra)
    assert cal.values.dtype == np.complex128
    assert np.allclose(cal.values, bg, rtol=0, atol=1e-14 * np.max(np.abs(bg)))


def test_wavelet_global_uses_one_scale_for_the_whole_array():
    """A per-pair anomaly survives 'global' and is absorbed by 'pair'."""
    bg, freqs, _, _ = _homogeneous()
    meas = bg.copy()
    meas[0, 0] *= 3.0                                   # one rogue pair

    glob = calibrate_wavelet(meas, bg, per="global", freqs=freqs)
    pair = calibrate_wavelet(meas, bg, per="pair", freqs=freqs)

    assert not np.allclose(glob.values[0, 0], bg[0, 0])
    assert np.allclose(pair.values, bg, rtol=0, atol=1e-12 * np.max(np.abs(bg)))


def test_wavelet_calibration_preserves_the_spectra_contract():
    bg, freqs, _, _ = _homogeneous()
    scale = _wavelet(freqs)
    meas = Spectra(values=scale[None, None, :] * bg, freqs=freqs,
                   band=freqs > 4.0e8, component="Ez")

    cal = calibrate_wavelet(meas, Spectra(values=bg, freqs=freqs), per="global")

    assert np.array_equal(cal.freqs, freqs)
    assert np.array_equal(cal.band, meas.band)
    assert cal.component == "Ez"


def test_wavelet_rejects_a_bad_per_mode():
    bg, freqs, _, _ = _homogeneous()
    with pytest.raises(ValueError, match="per must be one of"):
        calibrate_wavelet(bg, bg, per="receiver", freqs=freqs)


# --------------------------------------------------------------------------
# air calibration: analytic by construction
# --------------------------------------------------------------------------

def test_air_calibration_recovers_the_2d_green_function():
    """A 3-D-like measurement over its own air twin returns exactly ``G_2D``.

    Build ``u_meas = H(f) * spread(d) * exp(-j k d)`` with an arbitrary
    instrument response and 3-D spreading, and an air twin with the same
    ``H`` and spreading at the free-space wavenumber. The ratio is
    dimensionless, so the calibrated result must be ``G_2D(k0, d)`` times the
    medium's own extra phase -- and with the medium set to free space, exactly
    ``G_2D(k0, d)``.
    """
    src, rx = _panel()
    freqs = _freqs()
    d = offsets(src, rx)
    k0 = 2.0 * np.pi * freqs / C_LIGHT

    response = (2.0 + 0.5j) * np.exp(-1j * 2 * np.pi * freqs * 1.7e-9)
    spread = 1.0 / d[:, :, None]                       # 3-D geometric spreading
    phase = np.exp(-1j * k0[None, None, :] * d[:, :, None])
    air = response[None, None, :] * spread * phase
    meas = air.copy()

    cal = calibrate_air(meas, air, src, rx, freqs=freqs)

    expected = green_2d(k0[None, None, :], d[:, :, None])
    assert np.allclose(cal.values, expected, rtol=0, atol=1e-15)


def test_air_calibration_carries_the_medium_phase_through():
    """With the panel at eps = 6 the calibrated data is ``G_2D`` times the extra phase."""
    src, rx = _panel()
    freqs = _freqs()
    d = offsets(src, rx)
    k0 = 2.0 * np.pi * freqs / C_LIGHT
    k_soil = k0 * np.sqrt(6.0)

    response = (0.3 - 1.1j) * np.ones_like(freqs)
    spread = 1.0 / d[:, :, None]
    air = response[None, None, :] * spread * np.exp(-1j * k0[None, None, :] * d[:, :, None])
    meas = response[None, None, :] * spread * np.exp(-1j * k_soil[None, None, :] * d[:, :, None])

    cal = calibrate_air(meas, air, src, rx, k0=k0, freqs=freqs)

    expected = (green_2d(k0[None, None, :], d[:, :, None])
                * np.exp(-1j * (k_soil - k0)[None, None, :] * d[:, :, None]))
    assert np.allclose(cal.values, expected, rtol=1e-12, atol=0)


def test_air_calibration_accepts_a_callable_k0():
    src, rx = _panel()
    freqs = _freqs()
    d = offsets(src, rx)
    air = np.ones(d.shape + freqs.shape, dtype=np.complex128)

    a = calibrate_air(air, air, src, rx, k0=lambda f: 2 * np.pi * f / C_LIGHT, freqs=freqs)
    b = calibrate_air(air, air, src, rx, freqs=freqs)
    assert np.allclose(a.values, b.values)


def test_air_calibration_checks_shapes():
    src, rx = _panel()
    freqs = _freqs()
    air = np.ones((len(src), len(rx), len(freqs)), dtype=np.complex128)
    with pytest.raises(ValueError, match="they must match"):
        calibrate_air(air[:, :-1], air, src, rx, freqs=freqs)


# --------------------------------------------------------------------------
# coherence: high on a homogeneous panel, low on a heterogeneous one
# --------------------------------------------------------------------------

def test_coherence_is_one_on_a_homogeneous_panel():
    values, freqs, src, rx = _homogeneous(eps=6.0)
    score = coherence(values, offsets(src, rx), freqs, 6.0, 0.0)
    assert score == pytest.approx(1.0, abs=1e-12)


def test_coherence_ignores_an_unknown_wavelet():
    """A complex scale per frequency -- the whole point of the per-frequency stack."""
    values, freqs, src, rx = _homogeneous(eps=6.0)
    scaled = _wavelet(freqs)[None, None, :] * values
    assert coherence(scaled, offsets(src, rx), freqs, 6.0, 0.0) == pytest.approx(1.0, abs=1e-12)


def test_coherence_falls_off_away_from_the_truth():
    values, freqs, src, rx = _homogeneous(eps=6.0)
    d = offsets(src, rx)
    assert coherence(values, d, freqs, 6.0, 0.0) > coherence(values, d, freqs, 7.0, 0.0)
    assert coherence(values, d, freqs, 6.0, 0.0) > coherence(values, d, freqs, 5.0, 0.0)


def test_estimate_background_recovers_a_synthetic_homogeneous_panel():
    values, freqs, src, rx = _homogeneous(eps=6.0, sigma=0.0)
    bg = estimate_background(values, src, rx, freqs)
    assert bg.eps_bg == pytest.approx(6.0, rel=2e-3)
    assert bg.sigma_bg == pytest.approx(0.0, abs=1e-3)
    assert bg.coherence > 0.999


def test_estimate_background_recovers_a_lossy_panel():
    values, freqs, src, rx = _homogeneous(eps=9.0, sigma=0.02)
    bg = estimate_background(values, src, rx, freqs)
    assert bg.eps_bg == pytest.approx(9.0, rel=2e-3)
    assert bg.coherence > 0.999


def test_coherence_drops_on_a_heterogeneous_panel():
    """No single permittivity explains a panel whose pairs disagree."""
    homo, freqs, src, rx = _homogeneous(eps=6.0)
    hetero, _, _, _ = _heterogeneous()

    good = estimate_background(homo, src, rx, freqs)
    bad = estimate_background(hetero, src, rx, freqs)

    assert good.coherence > 0.99
    assert bad.coherence < COHERENCE_GATE
    assert bad.coherence < good.coherence


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def test_require_coherent_passes_above_the_gate():
    Background(eps_bg=6.0, sigma_bg=0.0, coherence=0.95).require_coherent()


def test_require_coherent_refuses_below_the_gate():
    bg = Background(eps_bg=6.0, sigma_bg=0.0, coherence=0.31)
    with pytest.raises(ValueError, match="below the gate"):
        bg.require_coherent()
    with pytest.raises(ValueError, match="below the gate"):
        bg.require_coherent(threshold=0.4)
    bg.require_coherent(threshold=0.3)          # an explicit lower bar is allowed


def test_a_heterogeneous_panel_refuses_at_the_default_gate():
    hetero, freqs, src, rx = _heterogeneous()
    bg = estimate_background(hetero, src, rx, freqs)
    with pytest.raises(ValueError, match="below the gate"):
        bg.require_coherent()


def test_background_rejects_unphysical_values():
    with pytest.raises(ValueError, match="eps_bg must be"):
        Background(eps_bg=0.5, sigma_bg=0.0, coherence=1.0)
    with pytest.raises(ValueError, match="sigma_bg must be"):
        Background(eps_bg=6.0, sigma_bg=-1e-3, coherence=1.0)


def test_estimate_background_validates_its_ranges():
    values, freqs, src, rx = _homogeneous()
    with pytest.raises(ValueError, match="eps_range must increase"):
        estimate_background(values, src, rx, freqs, eps_range=(9.0, 2.0))
    with pytest.raises(ValueError, match="eps_range must start"):
        estimate_background(values, src, rx, freqs, eps_range=(0.2, 9.0))
    with pytest.raises(ValueError, match="sigma_range"):
        estimate_background(values, src, rx, freqs, sigma_range=(-0.1, 0.5))


# --------------------------------------------------------------------------
# the real panel
# --------------------------------------------------------------------------

def _real_dataset():
    if not MERGED_SOIL.exists():
        pytest.skip(f"dataset not present: {MERGED_SOIL}")
    from Inversion.CSI.io import from_out_files          # test-only h5py path
    return from_out_files(str(MERGED_SOIL), acq=bscan_geometry(**B_ARRAY_ACQ))


def test_real_merged_soil_recovers_eps_6():
    """The acceptance gate: eps within 2% of 6.0 on the real merged B-scan.

    The file was run with sigma = 0 (its A-scan siblings carry the .in file's
    sigma = 0.01), so sigma_bg is expected at the bottom of its range.
    """
    ds = _real_dataset()
    bg = estimate_background(ds.meas, ds.src_pos, ds.rx_pos)

    assert bg.eps_bg == pytest.approx(B_ARRAY_EPS, rel=0.02)
    assert bg.sigma_bg < 5e-3
    assert bg.coherence > 0.99
    bg.require_coherent()


def test_real_merged_soil_is_wrap_free_at_the_top_of_the_band():
    """Restricted to >= 1 GHz -- where the unwrap diagnostic collapses -- the
    phasor stack still reads 6.0. This is what 'wrap-free' buys."""
    ds = _real_dataset()
    high = ds.meas.at(ds.meas.freqs >= 1.0e9)
    bg = estimate_background(high, ds.src_pos, ds.rx_pos)

    assert bg.eps_bg == pytest.approx(B_ARRAY_EPS, rel=0.02)
    assert bg.coherence > 0.99


def test_phase_slope_diagnostic_wraps_above_1_ghz():
    """Pinned numbers: 6.00 / 6.00 / 2.32 / 0.06 on the real 81-pair panel.

    The first two agree with truth, the last two do not -- naive unwrap dies
    once ``beta * delta_d`` exceeds pi between adjacent receivers. Keeping both
    halves asserted is what stops the primary estimator being rewritten as an
    unwrap.
    """
    ds = _real_dataset()

    def eps_at(f):
        return phase_slope_permittivity(ds.meas, ds.src_pos, ds.rx_pos, f)

    assert eps_at(0.625e9) == pytest.approx(6.00, abs=0.01)
    assert eps_at(0.812e9) == pytest.approx(6.00, abs=0.01)

    assert eps_at(1.000e9) == pytest.approx(2.32, abs=0.01)
    assert eps_at(1.500e9) == pytest.approx(0.06, abs=0.01)
    assert eps_at(1.000e9) < 0.5 * B_ARRAY_EPS          # disagrees, loudly
    assert eps_at(1.500e9) < EPS_MIN                    # not even a physical medium


def test_phase_slope_agrees_with_the_stack_where_it_is_valid():
    ds = _real_dataset()
    low = ds.meas.at(ds.meas.freqs <= 0.9e9)
    stacked = estimate_background(low, ds.src_pos, ds.rx_pos)
    slope = phase_slope_permittivity(ds.meas, ds.src_pos, ds.rx_pos, 0.625e9)
    assert slope == pytest.approx(stacked.eps_bg, rel=0.02)


# --------------------------------------------------------------------------
# the two modules stay on numpy/scipy
# --------------------------------------------------------------------------

def test_background_and_calibration_import_nothing_forbidden():
    import ast

    forbidden = {"h5py", "matplotlib", "gprMax", "Tomograph",
                 "Inversion.CSI.truth", "Inversion.CSI.metrics"}
    root = Path(__file__).resolve().parents[1] / "CSI"
    for name in ("background.py", "calibration.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for module in names:
                head = module.split(".")[0]
                assert head not in {"h5py", "matplotlib", "gprMax", "Tomograph"}, \
                    f"{name} imports {module}"
                assert not module.endswith(("truth", "metrics")), \
                    f"{name} imports {module}"
        assert forbidden  # the list is the point of the test


def test_eps_bounds_come_from_the_package_constants():
    assert (EPS_MIN, EPS_WATER_MAX) == (1.0, 90.0)
    values, freqs, src, rx = _homogeneous(eps=6.0)
    default = estimate_background(values, src, rx, freqs)
    explicit = estimate_background(values, src, rx, freqs,
                                   eps_range=(EPS_MIN, EPS_WATER_MAX))
    assert default.eps_bg == pytest.approx(explicit.eps_bg)
