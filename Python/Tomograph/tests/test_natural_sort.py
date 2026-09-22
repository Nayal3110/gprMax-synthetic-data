"""Verify that the loaders preserve numeric (natural) order even when h5py
itself iterates alphabetically.

Run from the repo root with:
    PYTHONPATH=gprMax uv run python -m pytest Python/Tomograph/tests/
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / 'Python'))
sys.path.insert(0, str(REPO_ROOT / 'gprMax'))

from Python.Tomograph.functions import get_rxs, get_field_data, _natural_sort_key  # noqa: E402


def _write_rx_group(parent, group_name, rx_name, position, n_time=4):
    g = parent.create_group(group_name)
    g.attrs['Name'] = rx_name
    g.attrs['Position'] = np.asarray(position, dtype=float)
    g.create_dataset('Ez', data=np.zeros(n_time, dtype=float))


@pytest.fixture
def shuffled_h5(tmp_path):
    """An .out-shaped HDF5 with 12 receivers written in alphabetic-but-not-numeric
    order. h5py will iterate keys as rx1, rx10, rx11, rx12, rx2, rx3, ..., rx9
    — exactly the bug the natural-sort fix is supposed to mask."""
    path = tmp_path / 'shuffled.out'
    with h5py.File(path, 'w') as f:
        srcs = f.create_group('srcs')
        s = srcs.create_group('src1')
        s.attrs['Position'] = np.asarray([0.0, 0.0, 0.0])
        s.attrs['Type'] = 'HertzianDipole'

        rxs = f.create_group('rxs')
        for i in [1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9]:
            _write_rx_group(rxs, f'rx{i}', f'rx{i}', (float(i), 0.0, 0.0))

        f.attrs['dx_dy_dz'] = np.asarray([0.01, 0.01, 0.01])
        f.attrs['dt'] = 1e-12
        f.attrs['Iterations'] = 4
    return path


def test_natural_sort_key_orders_numerically():
    raw = ['rx1', 'rx10', 'rx11', 'rx2', 'rx20', 'rx3']
    assert sorted(raw, key=_natural_sort_key) == [
        'rx1', 'rx2', 'rx3', 'rx10', 'rx11', 'rx20',
    ]


def test_get_rxs_returns_natural_order(shuffled_h5):
    with h5py.File(shuffled_h5, 'r') as f:
        rxs = get_rxs(f)
    assert list(rxs.keys()) == [f'rx{i}' for i in range(1, 13)]


def test_get_field_data_returns_natural_order(shuffled_h5):
    with h5py.File(shuffled_h5, 'r') as f:
        measurements = get_field_data(f)
    assert list(measurements.keys()) == [f'rx{i}' for i in range(1, 13)]
