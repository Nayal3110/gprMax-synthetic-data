import numpy as np
import h5py
import pytest

from Inversion.traveltime.loaders import (
    AcquisitionData, bscan_geometry, load_merged_bscan, load_ascan_set,
)


def test_bscan_geometry_reconstruction():
    acq = bscan_geometry(src_start=(0.1, 0.1), src_step=(0.0, 0.1), n_src=9,
                         rx_start=(0.9, 0.1), rx_step=(0.0, 0.1), n_rx=9)
    assert acq.sources.shape == (9, 2)
    assert acq.receivers.shape == (9, 2)
    assert np.allclose(acq.sources[0], [0.1, 0.1])
    assert np.allclose(acq.sources[-1], [0.1, 0.9])
    assert np.allclose(acq.receivers[-1], [0.9, 0.9])


def _write_merged(path, n_time=64, n_src=3, n_rx=2):
    with h5py.File(path, "w") as f:
        f.attrs["dt"] = 1e-11
        f.attrs["Iterations"] = n_time
        f.attrs["nrx"] = n_rx
        rxs = f.create_group("rxs")
        for r in range(1, n_rx + 1):
            g = rxs.create_group(f"rx{r}")
            g.create_dataset("Ez", data=np.random.rand(n_time, n_src).astype("f4"))


def test_load_merged_bscan_shapes(tmp_path):
    p = tmp_path / "merged.out"
    _write_merged(p, n_time=64, n_src=3, n_rx=2)
    acq = bscan_geometry((0.1, 0.1), (0.0, 0.1), 3, (0.9, 0.1), (0.0, 0.1), 2)
    data = load_merged_bscan(str(p), acq, component="Ez")
    assert isinstance(data, AcquisitionData)
    assert data.traces.shape == (3, 2, 64)   # [n_src, n_rx, n_time]
    assert data.dt == pytest.approx(1e-11)


def _write_ascan(path, src_pos, rx_positions, n_time=64):
    with h5py.File(path, "w") as f:
        f.attrs["dt"] = 1e-11
        f.attrs["Iterations"] = n_time
        f.attrs["dx_dy_dz"] = np.array([0.0008, 0.0008, 0.0008])
        f.attrs["nrx"] = len(rx_positions)
        s = f.create_group("srcs").create_group("src1")
        s.attrs["Position"] = np.asarray(src_pos, float)
        s.attrs["Type"] = "HertzianDipole"
        rxs = f.create_group("rxs")
        for i, pos in enumerate(rx_positions, start=1):
            g = rxs.create_group(f"rx{i}")
            g.attrs["Name"] = f"rx{i}"
            g.attrs["Position"] = np.asarray(pos, float)
            g.create_dataset("Ez", data=np.random.rand(n_time).astype("f4"))


def test_load_ascan_set_stacks_sources(tmp_path):
    rxp = [(0.9, 0.1, 0.0), (0.9, 0.2, 0.0)]
    files = []
    for k in range(3):
        p = tmp_path / f"scan{k}.out"
        _write_ascan(p, (0.1, 0.1 + 0.1 * k, 0.0), rxp)
        files.append(str(p))
    data = load_ascan_set(files, component="Ez")
    assert data.traces.shape == (3, 2, 64)
    assert data.acq.sources.shape == (3, 2)
    assert np.allclose(data.acq.sources[1], [0.1, 0.2])
    assert np.allclose(data.acq.receivers[0], [0.9, 0.1])
