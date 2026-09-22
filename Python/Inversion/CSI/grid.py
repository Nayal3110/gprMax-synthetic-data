"""Inversion grid, sized in wavelengths.

``size_grid`` takes an acquisition (source and receiver positions), a
background permittivity, the top frequency of the band and the stencil name,
and returns an ``InversionGrid``. Cell size follows the shortest wavelength the
model can support, so different panels cost roughly the same to solve.

Nothing in the package carries a length in metres as a tuning
constant, so a new survey at a new scale needs no retuning.

Risk: the cell count grows as the square of points-per-wavelength, and the
5-point stencil needs 20 rather than the usual 10 (see ``PPW``). That is a 4x
penalty in unknowns and roughly 15x in factorization time, and it is the main
reason a 9-point stencil is worth adding later.

Measured ``scipy.sparse.linalg.splu`` budget, complex128, ``permc_spec``
left at the default COLAMD::

    n unknowns   LU factor   memory   9-RHS batched solve
    200**2  = 40k    0.8 s     58 MB          41 ms
    400**2 = 160k   27.5 s    294 MB         1.15 s

``MMD_AT_PLUS_A`` did not finish in 400 s at n = 160k; do not switch to it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.constants import c as C_LIGHT

from . import EPS_WATER_MAX

# Points per wavelength required to keep the discrete phase error tolerable.
#
# The 5-point Helmholtz has relative phase error 1.645 / N**2 per wavelength.
# On an L/lambda = 16 rung, N = 10 accumulates ~95 deg across the panel, which
# is larger than the physical signal that rung exists to sharpen. N = 20 gives
# ~24 deg. The optimized 9-point stencil reaches the same accuracy at 8.
PPW = {"five_point": 20, "optimized_nine": 8}

# gprMax's own default PML depth, used as the floor so a comparison against an
# FDTD run is not confounded by a thinner boundary.
N_PML_MIN = 10


@dataclass(frozen=True)
class InversionGrid:
    """Uniform 2-D grid. Pixels flatten row-major as ``flat = iy * nx + ix``.

    ``x_min``/``y_min`` are the low corner of cell (0, 0); ``nx``/``ny`` count
    cells.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    @property
    def n_pixels(self) -> int:
        return self.nx * self.ny

    @property
    def cell_area(self) -> float:
        """Weight for the discrete inner product ``<a, b> = sum(a * conj(b)) * cell_area``."""
        return self.dx * self.dy

    @property
    def shape(self) -> tuple[int, int]:
        """``(ny, nx)`` -- the shape a flat pixel vector reshapes to."""
        return (self.ny, self.nx)

    def cell_centers(self) -> np.ndarray:
        """``[n_pixels, 2]`` centre coordinates in row-major (flat) order."""
        xs = self.x_min + (np.arange(self.nx) + 0.5) * self.dx
        ys = self.y_min + (np.arange(self.ny) + 0.5) * self.dy
        gx, gy = np.meshgrid(xs, ys)  # both [ny, nx]
        return np.column_stack([gx.ravel(), gy.ravel()])

    def reshape(self, flat: np.ndarray) -> np.ndarray:
        """View a flat pixel vector as an ``[ny, nx]`` image."""
        return np.asarray(flat).reshape(self.ny, self.nx)


def wavelength(freq: float, eps: float) -> float:
    """Wavelength in a medium of relative permittivity ``eps``."""
    return C_LIGHT / (freq * math.sqrt(eps))


def size_grid(src_pos, rx_pos, eps_bg: float, f_max: float,
              eps_max: float | None = None, stencil: str = "five_point",
              pad_wavelengths: float = 0.5) -> InversionGrid:
    """Grid covering the acquisition, resolved for the shortest wavelength.

    ``eps_max`` defaults to ``EPS_WATER_MAX``, which sets the smallest
    wavelength the model has to represent. Pass a larger value only if the
    target can exceed water.

    The domain is the bounding box of sources and receivers padded by
    ``pad_wavelengths`` background wavelengths -- the contrast is confined to
    it, so it must enclose everything the inversion is allowed to explain.
    """
    if stencil not in PPW:
        raise ValueError(f"unknown stencil {stencil!r}; expected one of {sorted(PPW)}")
    if f_max <= 0.0:
        raise ValueError(f"f_max must be positive, got {f_max}")
    if eps_bg < 1.0:
        raise ValueError(f"eps_bg must be >= 1, got {eps_bg}")

    eps_max = EPS_WATER_MAX if eps_max is None else max(eps_max, eps_bg)

    lambda_min = wavelength(f_max, eps_max)
    lambda_bg = wavelength(f_max, eps_bg)
    h = lambda_min / PPW[stencil]

    pts = np.vstack([np.asarray(src_pos, float), np.asarray(rx_pos, float)])
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"positions must be [N, 2], got {pts.shape}")
    pad = pad_wavelengths * lambda_bg

    x_min, y_min = pts.min(axis=0) - pad
    x_max, y_max = pts.max(axis=0) + pad

    # Round the cell count up so the realised cell size is <= h, then let the
    # extent grow to match: an exact cell size matters, an exact extent does not.
    nx = max(1, math.ceil((x_max - x_min) / h))
    ny = max(1, math.ceil((y_max - y_min) / h))
    return InversionGrid(
        x_min=float(x_min), x_max=float(x_min + nx * h),
        y_min=float(y_min), y_max=float(y_min + ny * h),
        nx=int(nx), ny=int(ny),
    )


def pml_cells(grid: InversionGrid, eps_bg: float, f_max: float) -> int:
    """PML depth in cells: a quarter background wavelength, floored at gprMax's 10."""
    lambda_bg = wavelength(f_max, eps_bg)
    return max(N_PML_MIN, math.ceil(0.25 * lambda_bg / grid.dx))
