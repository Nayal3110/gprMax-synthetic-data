import numpy as np
import pytest

from Inversion.traveltime.geometry import (
    PixelGrid, ray_cell_lengths, build_system_matrix, select_plane_axes,
)


def test_pixelgrid_basic_dims():
    g = PixelGrid(x_min=0.0, x_max=1.0, y_min=0.0, y_max=2.0, nx=4, ny=8)
    assert g.dx == pytest.approx(0.25)
    assert g.dy == pytest.approx(0.25)
    assert g.n_pixels == 32


def test_ray_total_length_equals_euclidean():
    # Ray fully inside the grid: summed cell lengths == straight-line distance.
    g = PixelGrid(0.0, 1.0, 0.0, 1.0, nx=10, ny=10)
    p1, p2 = (0.05, 0.05), (0.95, 0.85)
    idx, lengths = ray_cell_lengths(p1, p2, g)
    assert lengths.sum() == pytest.approx(np.hypot(0.90, 0.80))
    assert idx.min() >= 0 and idx.max() < g.n_pixels


def test_horizontal_ray_one_row():
    # A horizontal ray through the middle of row 0 crosses every column once.
    g = PixelGrid(0.0, 4.0, 0.0, 2.0, nx=4, ny=2)  # cells are 1.0 x 1.0
    idx, lengths = ray_cell_lengths((0.0, 0.5), (4.0, 0.5), g)
    # row index = floor(0.5/1.0) = 0 -> flat indices 0,1,2,3
    assert sorted(idx.tolist()) == [0, 1, 2, 3]
    assert np.allclose(lengths, 1.0)


def test_zero_length_ray_returns_empty():
    g = PixelGrid(0.0, 1.0, 0.0, 1.0, nx=5, ny=5)
    idx, lengths = ray_cell_lengths((0.3, 0.3), (0.3, 0.3), g)
    assert idx.size == 0 and lengths.size == 0


def test_select_plane_axes_picks_two_varying_axes():
    # x varies, y varies, z constant -> axes (0, 1)
    src = np.array([[1.0, 4.0, 1.0]])
    rx = np.array([[13.0, 1.0, 1.0], [13.0, 7.0, 1.0]])
    assert select_plane_axes(src, rx) == (0, 1)


def test_system_matrix_shape_and_row_sums():
    g = PixelGrid(0.0, 1.0, 0.0, 1.0, nx=8, ny=8)
    sources = np.array([[0.05, 0.05], [0.05, 0.95]])
    receivers = np.array([[0.95, 0.05], [0.95, 0.95]])
    L, rays = build_system_matrix(sources, receivers, g)
    assert L.shape == (4, g.n_pixels)        # 2 src x 2 rx
    assert len(rays) == 4
    # each row sums to the straight-line source->receiver distance
    expected = [np.hypot(rx[0] - s[0], rx[1] - s[1])
                for s in sources for rx in receivers]
    assert np.allclose(np.asarray(L.sum(axis=1)).ravel(), expected)
