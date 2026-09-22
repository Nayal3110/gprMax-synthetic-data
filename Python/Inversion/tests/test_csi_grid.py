"""Grid sizing: cell size follows the wavelength, and the flat pixel order
matches the travel-time grid exactly."""
import math

import numpy as np
import pytest
from scipy.constants import c as C_LIGHT

from Inversion.CSI import EPS_WATER_MAX
from Inversion.CSI.grid import PPW, InversionGrid, pml_cells, size_grid, wavelength
from Inversion.traveltime.geometry import PixelGrid


def crosshole(sep=0.8, n=9, y0=0.1, step=0.1):
    """Two vertical borehole lines separated by ``sep``."""
    ys = y0 + np.arange(n) * step
    src = np.column_stack([np.full(n, 0.1), ys])
    rx = np.column_stack([np.full(n, 0.1 + sep), ys])
    return src, rx


def test_cell_size_is_lambda_min_over_ppw():
    src, rx = crosshole()
    f_max = 1.5e9
    grid = size_grid(src, rx, eps_bg=6.0, f_max=f_max)
    expected = wavelength(f_max, EPS_WATER_MAX) / PPW["five_point"]
    assert grid.dx == pytest.approx(expected, rel=1e-12)
    assert grid.dy == pytest.approx(expected, rel=1e-12)


def test_nine_point_stencil_is_coarser():
    src, rx = crosshole()
    five = size_grid(src, rx, eps_bg=6.0, f_max=1.5e9, stencil="five_point")
    nine = size_grid(src, rx, eps_bg=6.0, f_max=1.5e9, stencil="optimized_nine")
    assert nine.dx > five.dx
    assert nine.n_pixels < five.n_pixels
    # 20 ppw vs 8 ppw is 2.5x in cell size, so 6.25x in unknowns.
    assert nine.dx / five.dx == pytest.approx(PPW["five_point"] / PPW["optimized_nine"])


def test_scale_generic_cell_counts():
    """A 0.5 m panel and a 20 m panel land within 2x on cell count.

    Each is driven at the frequency giving the same L/lambda, which is the
    whole point of sizing in wavelengths rather than metres.
    """
    small_src, small_rx = crosshole(sep=0.5, n=9, y0=0.05, step=0.05)
    big_src, big_rx = crosshole(sep=20.0, n=9, y0=2.0, step=2.0)

    # L/lambda = 16 in the background for both panels.
    f_small = 16 * C_LIGHT / (0.5 * math.sqrt(6.0))
    f_big = 16 * C_LIGHT / (20.0 * math.sqrt(6.0))

    small = size_grid(small_src, small_rx, eps_bg=6.0, f_max=f_small)
    big = size_grid(big_src, big_rx, eps_bg=6.0, f_max=f_big)

    ratio = max(small.n_pixels, big.n_pixels) / min(small.n_pixels, big.n_pixels)
    assert ratio < 2.0, f"cell counts diverge by {ratio:.2f}x"


def test_domain_encloses_acquisition_with_pad():
    src, rx = crosshole()
    eps_bg, f_max = 6.0, 1.5e9
    grid = size_grid(src, rx, eps_bg=eps_bg, f_max=f_max, pad_wavelengths=0.5)
    pad = 0.5 * wavelength(f_max, eps_bg)
    pts = np.vstack([src, rx])
    assert grid.x_min <= pts[:, 0].min() - pad + 1e-12
    assert grid.y_min <= pts[:, 1].min() - pad + 1e-12
    assert grid.x_max >= pts[:, 0].max() + pad - 1e-12
    assert grid.y_max >= pts[:, 1].max() + pad - 1e-12


def test_cell_centers_match_pixelgrid_element_for_element():
    """The two grids must agree exactly, or truth maps and travel-time starts
    would silently land half a cell off."""
    g = InversionGrid(x_min=-0.3, x_max=1.1, y_min=0.0, y_max=0.9, nx=17, ny=11)
    p = PixelGrid(x_min=-0.3, x_max=1.1, y_min=0.0, y_max=0.9, nx=17, ny=11)
    np.testing.assert_array_equal(g.cell_centers(), p.cell_centers())


def test_flat_order_is_row_major():
    g = InversionGrid(x_min=0.0, x_max=2.0, y_min=0.0, y_max=3.0, nx=2, ny=3)
    centers = g.cell_centers()
    assert g.shape == (3, 2)
    for iy in range(g.ny):
        for ix in range(g.nx):
            flat = iy * g.nx + ix
            assert centers[flat, 0] == pytest.approx(g.x_min + (ix + 0.5) * g.dx)
            assert centers[flat, 1] == pytest.approx(g.y_min + (iy + 0.5) * g.dy)


def test_cell_area_and_reshape():
    g = InversionGrid(x_min=0.0, x_max=2.0, y_min=0.0, y_max=3.0, nx=4, ny=6)
    assert g.cell_area == pytest.approx(g.dx * g.dy)
    flat = np.arange(g.n_pixels)
    assert g.reshape(flat).shape == (6, 4)
    assert g.reshape(flat)[2, 3] == 2 * g.nx + 3


def test_pml_depth_floors_at_gprmax_default():
    src, rx = crosshole()
    grid = size_grid(src, rx, eps_bg=6.0, f_max=1.5e9)
    assert pml_cells(grid, eps_bg=6.0, f_max=1.5e9) >= 10


def test_rejects_bad_arguments():
    src, rx = crosshole()
    with pytest.raises(ValueError, match="stencil"):
        size_grid(src, rx, eps_bg=6.0, f_max=1e9, stencil="seven_point")
    with pytest.raises(ValueError, match="f_max"):
        size_grid(src, rx, eps_bg=6.0, f_max=0.0)
    with pytest.raises(ValueError, match="eps_bg"):
        size_grid(src, rx, eps_bg=0.5, f_max=1e9)
