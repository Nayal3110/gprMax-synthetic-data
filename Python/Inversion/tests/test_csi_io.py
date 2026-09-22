"""Data-loading layer: traces -> spectra, and gprMax .out -> CsiDataset.

The load-bearing test here is ``test_from_arrays_matches_from_out_files``: the
real-data door and the synthetic door must produce the same object, or the
synthetic validation is measuring a path the field data never takes.
"""
import ast
from pathlib import Path

import h5py
import numpy as np
import pytest

from Inversion.CSI.io import (
    CsiDataset, MERGED_NEEDS_ACQ, from_arrays, from_out_files,
)
from Inversion.CSI.spectra import (
    Spectra, band_mask, band_profile, select_component, spectra_from_traces,
)
from Inversion.traveltime.loaders import Acquisition, bscan_geometry


REPO_ROOT = Path(__file__).resolve().parents[3]
SOIL_DIR = REPO_ROOT / "gprMax" / "user_models" / "test_B_array_soil" / "Simple_soil"
MERGED_SOIL = SOIL_DIR / "test_B_array_soil_merged.out"
ASCANS_SOIL = [SOIL_DIR / f"test_B_array_soil{i}.out" for i in range(1, 10)]
MERGED_WATER = (REPO_ROOT / "gprMax" / "user_models" / "test_B_array_water"
                / "test_B_array_water_merged.out")

# Known geometry of both test_B_array runs (identical): 9 sources stepping
# +0.1 m in y from (0.10, 0.10), 9 receivers likewise from (0.90, 0.10).
B_ARRAY_ACQ = dict(src_start=(0.1, 0.1), src_step=(0.0, 0.1), n_src=9,
                   rx_start=(0.9, 0.1), rx_step=(0.0, 0.1), n_rx=9)


def _need(*paths):
    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"dataset not present: {p}")


def _b_array_acq():
    return bscan_geometry(**B_ARRAY_ACQ)


# --------------------------------------------------------------------------
# fixtures: synthetic .out files in the house style
# --------------------------------------------------------------------------

def _synthetic_traces(n_src=3, n_rx=2, n_time=256, dt=1e-11, fc=1.5e9):
    """A crude Ricker-ish arrival per pair, distinct for every (src, rx)."""
    t = np.arange(n_time) * dt
    traces = np.zeros((n_src, n_rx, n_time))
    for s in range(n_src):
        for r in range(n_rx):
            t0 = (40 + 7 * s + 11 * r) * dt
            arg = (np.pi * fc * (t - t0)) ** 2
            traces[s, r] = (1.0 + 0.1 * s - 0.05 * r) * (1 - 2 * arg) * np.exp(-arg)
    return traces, dt


def _write_merged(path, traces, dt):
    """Merged B-scan layout: root attrs dt/Iterations/nrx only, rxs/<n>/<comp>
    is [n_time, n_src], receiver attrs empty."""
    n_src, n_rx, n_time = traces.shape
    with h5py.File(path, "w") as f:
        f.attrs["dt"] = dt
        f.attrs["Iterations"] = n_time
        f.attrs["nrx"] = n_rx
        rxs = f.create_group("rxs")
        for r in range(n_rx):
            g = rxs.create_group(f"rx{r + 1}")
            g.create_dataset("Ez", data=traces[:, r, :].T.astype("f4"))
            g.create_dataset("Ex", data=np.zeros((n_time, n_src), "f4"))
            g.create_dataset("Ey", data=(0.01 * traces[:, r, :].T).astype("f4"))
    return path


def _write_ascan_set(directory, traces, dt, src_pos, rx_pos):
    """One per-run A-scan per source, carrying srcs/ and receiver Position attrs."""
    n_src, n_rx, n_time = traces.shape
    paths = []
    for s in range(n_src):
        p = directory / f"scan{s + 1}.out"
        with h5py.File(p, "w") as f:
            f.attrs["dt"] = dt
            f.attrs["Iterations"] = n_time
            f.attrs["dx_dy_dz"] = np.array([8e-4, 8e-4, 8e-4])
            f.attrs["nrx"] = n_rx
            f.attrs["nsrc"] = 1
            g = f.create_group("srcs").create_group("src1")
            g.attrs["Position"] = np.asarray(src_pos[s], float)
            g.attrs["Type"] = "HertzianDipole"
            rxs = f.create_group("rxs")
            for r in range(n_rx):
                gr = rxs.create_group(f"rx{r + 1}")
                gr.attrs["Name"] = f"rx{r + 1}"
                gr.attrs["Position"] = np.asarray(rx_pos[r], float)
                gr.create_dataset("Ez", data=traces[s, r].astype("f4"))
                gr.create_dataset("Ex", data=np.zeros(n_time, "f4"))
                gr.create_dataset("Ey", data=(0.01 * traces[s, r]).astype("f4"))
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------
# spectra.py
# --------------------------------------------------------------------------

def test_single_tone_lands_on_its_bin():
    """A pure tone at an exact bin gives amplitude A*N/2 there and nothing else."""
    n_time, dt, amp = 512, 1e-11, 3.0
    k = 40                                   # bin index -> exactly periodic
    t = np.arange(n_time) * dt
    f0 = k / (n_time * dt)
    trace = amp * np.cos(2 * np.pi * f0 * t)
    sp = spectra_from_traces(trace[None, None, :], dt, component="Ez")

    assert sp.freqs[k] == pytest.approx(f0)
    assert np.abs(sp.values[0, 0, k]) == pytest.approx(amp * n_time / 2)
    others = np.delete(np.abs(sp.values[0, 0]), k)
    assert np.max(others) < 1e-8 * amp * n_time


def test_parseval_holds_for_the_one_sided_transform():
    traces, dt = _synthetic_traces()
    sp = spectra_from_traces(traces, dt, component="Ez")
    n = traces.shape[-1]
    # One-sided: interior bins count twice, DC and Nyquist once.
    weight = np.full(sp.n_freq, 2.0)
    weight[0] = 1.0
    if n % 2 == 0:
        weight[-1] = 1.0
    energy_f = np.sum(weight * np.abs(sp.values[0, 0]) ** 2) / n
    assert energy_f == pytest.approx(np.sum(traces[0, 0] ** 2), rel=1e-10)


def test_spectra_are_complex128_and_one_sided():
    traces, dt = _synthetic_traces(n_time=256)
    sp = spectra_from_traces(traces, dt, component="Ez")
    assert sp.values.dtype == np.complex128
    assert sp.freqs.dtype == np.float64
    assert sp.n_freq == 256 // 2 + 1
    assert np.all(sp.freqs >= 0.0)


def test_band_mask_matches_the_tomograph_helper():
    """The numpy-only restatement must agree with Tomograph's own rule."""
    from Tomograph.functions import _bandwidth_mask

    traces, dt = _synthetic_traces()
    sp = spectra_from_traces(traces, dt, component="Ez")
    profile = band_profile(sp.values)
    assert np.array_equal(sp.band, _bandwidth_mask(profile, -40))
    assert np.array_equal(band_mask(profile, -40), _bandwidth_mask(profile, -40))


def test_band_mask_is_contiguous_and_excludes_dc_tail():
    traces, dt = _synthetic_traces()
    sp = spectra_from_traces(traces, dt, component="Ez")
    assert sp.band.any()
    assert not sp.band.all()          # a 40 dB window never keeps everything
    f_lo, f_hi = sp.band_edges
    assert 0.0 <= f_lo < f_hi
    assert np.array_equal(sp.band_freqs, sp.freqs[sp.band])


def test_band_mask_of_a_silent_record_is_empty():
    sp = spectra_from_traces(np.zeros((1, 1, 64)), 1e-11, component="Ez")
    assert not sp.band.any()


def test_component_auto_selection_prefers_the_strongest():
    traces, dt = _synthetic_traces()
    candidates = {"Ex": np.zeros_like(traces), "Ey": 0.01 * traces, "Ez": traces}
    name, margin = select_component(candidates)
    assert name == "Ez"
    assert margin == pytest.approx(100.0, rel=1e-6)
    assert spectra_from_traces(candidates, dt).component == "Ez"


def test_component_override_is_honoured_and_checked():
    traces, dt = _synthetic_traces()
    candidates = {"Ey": 0.01 * traces, "Ez": traces}
    assert select_component(candidates, "Ey")[0] == "Ey"
    assert spectra_from_traces(candidates, dt, component="Ey").component == "Ey"
    with pytest.raises(KeyError, match="not present"):
        select_component(candidates, "Ex")


def test_all_zero_components_raise():
    with pytest.raises(ValueError, match="identically zero"):
        select_component({"Ex": np.zeros((1, 1, 8)), "Ez": np.zeros((1, 1, 8))})


def test_spectra_at_subsets_the_frequency_axis():
    traces, dt = _synthetic_traces()
    sp = spectra_from_traces(traces, dt, component="Ez")
    sub = sp.at(sp.band)
    assert sub.n_freq == int(sp.band.sum())
    assert sub.band.all()
    assert np.array_equal(sub.freqs, sp.band_freqs)
    assert sub.component == "Ez"


def test_spectra_reject_a_mismatched_frequency_axis():
    with pytest.raises(ValueError, match=r"freqs must be"):
        Spectra(values=np.zeros((1, 1, 4), complex), freqs=np.zeros(3))
    with pytest.raises(ValueError, match=r"n_src, n_rx, n_freq"):
        Spectra(values=np.zeros((1, 4), complex), freqs=np.zeros(4))


# --------------------------------------------------------------------------
# io.py -- merged files
# --------------------------------------------------------------------------

def test_merged_without_acq_raises_a_clear_error(tmp_path):
    traces, dt = _synthetic_traces()
    p = _write_merged(tmp_path / "merged.out", traces, dt)
    with pytest.raises(ValueError) as exc:
        from_out_files(str(p))
    msg = str(exc.value)
    assert msg == MERGED_NEEDS_ACQ.format(path=str(p))
    assert "merged B-scan carries no geometry" in msg
    assert "Pass acq=" in msg
    assert str(p) in msg


def test_merged_with_acq_loads(tmp_path):
    traces, dt = _synthetic_traces(n_src=3, n_rx=2, n_time=256)
    p = _write_merged(tmp_path / "merged.out", traces, dt)
    acq = bscan_geometry((0.1, 0.1), (0.0, 0.1), 3, (0.9, 0.1), (0.0, 0.1), 2)
    ds = from_out_files(str(p), acq=acq)

    assert isinstance(ds, CsiDataset)
    assert ds.s21.shape == (3, 2, 256 // 2 + 1)
    assert ds.meas.component == "Ez"
    assert ds.air is None and ds.background is None
    # the merged transpose must not scramble the (src, rx) axes
    expect = np.fft.rfft(traces, axis=-1)
    assert np.allclose(ds.s21, expect, rtol=1e-5, atol=1e-5 * np.abs(expect).max())


def test_merged_acq_shape_is_checked(tmp_path):
    traces, dt = _synthetic_traces(n_src=3, n_rx=2)
    p = _write_merged(tmp_path / "merged.out", traces, dt)
    wrong = bscan_geometry((0.1, 0.1), (0.0, 0.1), 4, (0.9, 0.1), (0.0, 0.1), 2)
    with pytest.raises(ValueError, match=r"acq.sources is"):
        from_out_files(str(p), acq=wrong)


@pytest.mark.parametrize("path", [MERGED_SOIL, MERGED_WATER])
def test_real_merged_file_loads_9x9(path):
    _need(path)
    ds = from_out_files(str(path), acq=_b_array_acq())
    assert ds.s21.shape[:2] == (9, 9)
    assert ds.s21.dtype == np.complex128
    assert ds.freqs.shape == (ds.s21.shape[2],)
    assert ds.meas.component == "Ez"
    f_lo, f_hi = ds.meas.band_edges
    assert 0.0 < f_lo < f_hi < 1 / (2 * 1.886923469399747e-12)


def test_real_merged_component_auto_selection_picks_ez():
    _need(MERGED_SOIL)
    from Inversion.CSI.io import _read_traces
    traces, _, _ = _read_traces(str(MERGED_SOIL), _b_array_acq())
    name, margin = select_component(traces)
    assert name == "Ez"
    assert margin > 10.0            # z-directed dipole in a 2-D TM scene


# --------------------------------------------------------------------------
# io.py -- per-run A-scan sets
# --------------------------------------------------------------------------

def test_ascan_set_auto_derives_geometry(tmp_path):
    traces, dt = _synthetic_traces(n_src=3, n_rx=2)
    src_pos = [(0.1, 0.1 + 0.1 * s, 0.0) for s in range(3)]
    rx_pos = [(0.9, 0.1 + 0.1 * r, 0.0) for r in range(2)]
    paths = _write_ascan_set(tmp_path, traces, dt, src_pos, rx_pos)

    ds = from_out_files(paths)
    assert ds.src_pos.shape == (3, 2)
    assert ds.rx_pos.shape == (2, 2)
    assert np.allclose(ds.src_pos, [[0.1, 0.1], [0.1, 0.2], [0.1, 0.3]])
    assert np.allclose(ds.rx_pos, [[0.9, 0.1], [0.9, 0.2]])
    assert ds.acq.plane_axes == (0, 1)


def test_real_ascan_set_auto_derives_the_known_geometry():
    _need(*ASCANS_SOIL)
    ds = from_out_files([str(p) for p in ASCANS_SOIL])
    assert ds.src_pos.shape == (9, 2)
    assert ds.rx_pos.shape == (9, 2)
    assert np.allclose(ds.src_pos, np.column_stack(
        [np.full(9, 0.10), 0.10 + 0.1 * np.arange(9)]))
    assert np.allclose(ds.rx_pos, np.column_stack(
        [np.full(9, 0.90), 0.10 + 0.1 * np.arange(9)]))
    assert ds.acq.plane_axes == (0, 1)
    assert ds.meas.component == "Ez"


def test_real_soil_merged_is_a_different_simulation_than_its_ascans():
    """Dataset fact, not a loader property.

    ``test_B_array_soil_merged.out`` is NOT the merge of
    ``test_B_array_soil1..9.out``. Both readers agree on geometry and on
    timing -- every pair's waveform matches after a single scale factor, to
    0.2% -- but the merged file is systematically louder, by exactly
    ``exp(alpha*d)`` with ``alpha = 0.769 Np/m``. That is sigma = 0.01 S/m of
    attenuation removed: the merged file was run with the water scene's
    ``#material: 6 0 1 0 soil``, the per-run A-scans with
    ``test_B_array_soil.in``'s ``#material: 6 0.01 1 0 soil``.

    Consequence: the merged soil file is the sigma-matched background twin of
    the water run, so those two may be differenced; the per-run A-scans may
    not be differenced against the water run.
    """
    _need(MERGED_SOIL, *ASCANS_SOIL)
    a = from_out_files([str(p) for p in ASCANS_SOIL])
    m = from_out_files(str(MERGED_SOIL), acq=_b_array_acq())

    assert a.s21.shape == m.s21.shape
    assert np.allclose(a.src_pos, m.src_pos)
    assert np.allclose(a.rx_pos, m.rx_pos)

    scales = np.empty((9, 9))
    for s in range(9):
        for r in range(9):
            x, y = a.s21[s, r], m.s21[s, r]
            k = np.real(np.vdot(x, y) / np.vdot(x, x))
            scales[s, r] = k
            assert np.linalg.norm(y - k * x) / np.linalg.norm(y) < 0.01

    offsets = np.linalg.norm(a.src_pos[:, None, :] - a.rx_pos[None, :, :], axis=-1)
    alpha = np.log(scales) / offsets
    assert scales.min() > 1.5                       # not the same simulation
    assert alpha.std() < 1e-3                       # a pure attenuation difference
    assert alpha.mean() == pytest.approx(0.769, abs=5e-3)


# --------------------------------------------------------------------------
# the acceptance criterion: both doors, one dataset
# --------------------------------------------------------------------------

def _assert_same_dataset(a: CsiDataset, b: CsiDataset):
    assert np.array_equal(a.s21, b.s21)
    assert a.s21.dtype == b.s21.dtype
    assert np.array_equal(a.freqs, b.freqs)
    assert np.array_equal(a.band, b.band)
    assert a.meas.component == b.meas.component
    assert np.array_equal(a.src_pos, b.src_pos)
    assert np.array_equal(a.rx_pos, b.rx_pos)
    assert a.acq.plane_axes == b.acq.plane_axes
    for label in ("air", "background"):
        ra, rb = getattr(a, label), getattr(b, label)
        assert (ra is None) == (rb is None)
        if ra is not None:
            assert np.array_equal(ra.values, rb.values)
            assert np.array_equal(ra.band, rb.band)
            assert ra.component == rb.component


def test_from_arrays_matches_from_out_files(tmp_path):
    traces, dt = _synthetic_traces(n_src=3, n_rx=2, n_time=256)
    p = _write_merged(tmp_path / "merged.out", traces, dt)
    acq = bscan_geometry((0.1, 0.1), (0.0, 0.1), 3, (0.9, 0.1), (0.0, 0.1), 2)

    from_file = from_out_files(str(p), acq=acq)
    from_tuple = from_arrays(from_file.s21, from_file.freqs,
                             acq.sources, acq.receivers,
                             component=from_file.meas.component)
    _assert_same_dataset(from_file, from_tuple)


def test_from_arrays_matches_from_out_files_on_the_real_scene():
    _need(MERGED_SOIL, MERGED_WATER)
    acq = _b_array_acq()
    from_file = from_out_files(str(MERGED_WATER), background=str(MERGED_SOIL), acq=acq)
    from_tuple = from_arrays(from_file.s21, from_file.freqs,
                             acq.sources, acq.receivers,
                             background=from_file.background.values,
                             component=from_file.meas.component)
    _assert_same_dataset(from_file, from_tuple)


def test_from_arrays_needs_no_references():
    rng = np.random.default_rng(0)
    s21 = rng.normal(size=(4, 3, 16)) + 1j * rng.normal(size=(4, 3, 16))
    ds = from_arrays(s21, np.linspace(1e8, 1e9, 16),
                     rng.normal(size=(4, 2)), rng.normal(size=(3, 2)))
    assert ds.air is None and ds.background is None
    assert ds.meas.component is None
    assert ds.s21.dtype == np.complex128
    assert ds.band.shape == (16,)


def test_from_arrays_rejects_a_mismatched_reference():
    s21 = np.ones((2, 2, 8), complex)
    with pytest.raises(ValueError, match="must match the measurement"):
        from_arrays(s21, np.arange(8.0), np.zeros((2, 2)), np.ones((2, 2)),
                    air=np.ones((2, 3, 8), complex))
    with pytest.raises(ValueError, match=r"freqs must be"):
        from_arrays(s21, np.arange(8.0), np.zeros((2, 2)), np.ones((2, 2)),
                    air=np.ones((2, 2, 9), complex))


def test_references_read_on_the_measurement_component(tmp_path):
    traces, dt = _synthetic_traces(n_src=3, n_rx=2)
    meas = _write_merged(tmp_path / "meas.out", traces, dt)
    ref = _write_merged(tmp_path / "ref.out", 0.5 * traces, dt)
    acq = bscan_geometry((0.1, 0.1), (0.0, 0.1), 3, (0.9, 0.1), (0.0, 0.1), 2)

    ds = from_out_files(str(meas), air=str(ref), background=str(ref),
                        component="Ey", acq=acq)
    assert ds.meas.component == "Ey"
    assert ds.air.component == "Ey"
    assert ds.background.component == "Ey"
    assert np.allclose(ds.air.values, 0.5 * ds.s21, rtol=1e-4,
                       atol=1e-4 * np.abs(ds.s21).max())


# --------------------------------------------------------------------------
# import hygiene: io.py is the only door to h5py / gprMax / Tomograph
# --------------------------------------------------------------------------

CSI_DIR = REPO_ROOT / "Python" / "Inversion" / "CSI"
FORBIDDEN_ROOTS = {"h5py", "gprMax", "Tomograph", "matplotlib", "pylab",
                   "tkinter", "truth", "metrics"}
SCORING_ONLY = {"truth", "metrics"}


def _imported_roots(path: Path) -> set[str]:
    """Top-level names a module imports, read from source rather than executed.

    Relative imports are reported by their bare module name so a
    ``from .truth import ...`` is caught as well as an absolute one.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
            if node.level:                      # from . import X / from .pkg import X
                roots.update(a.name.split(".")[0] for a in node.names)
    return roots


def _csi_modules():
    return sorted(p for p in CSI_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_the_csi_package_has_modules_to_check():
    names = {p.name for p in _csi_modules()}
    assert {"__init__.py", "grid.py", "contrast.py", "csi.py",
            "spectra.py", "io.py"} <= names


@pytest.mark.parametrize("path", _csi_modules(), ids=lambda p: p.name)
def test_only_io_reaches_h5py_gprmax_or_tomograph(path):
    roots = _imported_roots(path)
    if path.name == "io.py":
        assert "h5py" in roots, "io.py is the module that owns the file format"
        assert not (roots & SCORING_ONLY)
        return
    offenders = roots & FORBIDDEN_ROOTS
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} imports {sorted(offenders)}; only "
        "CSI/io.py may reach the file format, and no inversion module may "
        "reach truth/metrics")


def test_no_inversion_module_imports_truth_or_metrics():
    for path in _csi_modules():
        assert not (_imported_roots(path) & SCORING_ONLY), path
