"""Integral-equation Green operator on a homogeneous background.

Takes an ``InversionGrid``, a background wavenumber and point source/receiver
positions; produces the incident field and the two radiation operators ``G_D``
(grid -> grid) and ``G_S`` (grid -> receivers), each with its exact discrete
adjoint. The kernel is the free-space 2-D TM Green's function sampled at cell
centres, the singular self-cell replaced by its closed-form integral over an
equal-area circular patch.

Strength: the kernel is the analytic Green's function, so midpoint quadrature
is the only approximation -- no outer boundary, no numerical dispersion, and
nothing to tune. Uniform cells make ``G_D`` block-Toeplitz, so it applies by
FFT on a circulant embedding in ``O(N log N)``.

Risk: ``G_S`` is stored dense as ``[n_rx, n_pixels]``, and the FFT workspace is
about four times the grid, so memory tracks the domain rather than the data.

Conventions
-----------
``e^{+j w t}``, outgoing ``e^{-j k r}``, ``G_2D = -(1j/4) H_0^(2)(k_b r)``.
``G_D`` carries the ``k_b**2`` of the contrast-source form, i.e. it maps
``w = chi * u`` straight to a scattered field.
"""
from __future__ import annotations

import numpy as np
from scipy import fft as sp_fft
from scipy.special import hankel2

from ..grid import InversionGrid
from . import background_wavenumber

# Ceiling on the dense receiver matrix, in bytes.
MAX_DATA_MATRIX_BYTES = 1 << 30


def self_cell_term(k_b: complex, cell_area: float) -> complex:
    """``k_b**2 * INT_cell G dA`` over the equal-area disc of radius ``sqrt(dA/pi)``.

    Closed form of the only singular entry in the kernel; a sign slip here is
    invisible until the whole inversion is subtly wrong, so it is pinned
    against quadrature in the tests.
    """
    z = k_b * np.sqrt(cell_area / np.pi)
    return complex(-(0.5j * np.pi * z * hankel2(1, z) + 1.0))


def _radiation_kernel(k_b: complex, r: np.ndarray, cell_area: float) -> np.ndarray:
    """``k_b**2 * G * dA`` at distances ``r``; ``r == 0`` takes the self-cell value."""
    r = np.asarray(r, float)
    out = np.empty(r.shape, dtype=np.complex128)
    off = r != 0.0
    out[off] = -0.25j * k_b**2 * cell_area * hankel2(0, k_b * r[off])
    out[~off] = self_cell_term(k_b, cell_area)
    return out


def _point_kernel(k_b: complex, r: np.ndarray) -> np.ndarray:
    """``G_2D`` at distances ``r`` -- the field of a unit line source."""
    r = np.asarray(r, float)
    if np.any(r == 0.0):
        raise ValueError(
            "a source or receiver sits exactly on a cell centre (or on another "
            "point source); G_2D is singular there -- nudge it off the node"
        )
    return -0.25j * hankel2(0, k_b * r)


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``[len(a), len(b)]`` pairwise distances between two ``[N, 2]`` point sets."""
    return np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])


class IntegralOperator:
    """Free-space Green operator for one angular frequency and one background.

    Satisfies ``GreenOperator``. Every field array carries a leading ``n_src``
    axis; ``domain``/``domain_adjoint`` also accept a bare ``[n_pixels]`` vector
    and hand back the same shape.
    """

    def __init__(self, grid: InversionGrid, omega: float, k_b: complex,
                 src_pos, rx_pos):
        self.grid = grid
        self.omega = float(omega)
        self.k_b = complex(k_b)
        self.src_pos = np.asarray(src_pos, float).reshape(-1, 2)
        self.rx_pos = np.asarray(rx_pos, float).reshape(-1, 2)
        self.n_src = int(self.src_pos.shape[0])
        self.n_rx = int(self.rx_pos.shape[0])
        self._fft_shape, self._kernel_spectrum = self._build_kernel()
        self._data_mat = None

    @classmethod
    def from_background(cls, grid: InversionGrid, omega: float, eps_bg: float,
                        src_pos, rx_pos, sigma_bg: float = 0.0) -> "IntegralOperator":
        """Build from a background permittivity and conductivity instead of ``k_b``."""
        return cls(grid, omega, background_wavenumber(omega, eps_bg, sigma_bg),
                   src_pos, rx_pos)

    # ---- kernel -----------------------------------------------------------

    def _build_kernel(self):
        """FFT of the kernel on a circulant embedding wide enough to avoid wrap."""
        g = self.grid
        shape = (sp_fft.next_fast_len(2 * g.ny - 1),
                 sp_fft.next_fast_len(2 * g.nx - 1))
        my, mx = shape

        # Offsets in wraparound order: 0..n-1 up front, -(n-1)..-1 at the tail.
        # Anything left in the middle is padding the convolution never reads,
        # because both the input support and the output window are n wide.
        oy = np.zeros(my)
        oy[:g.ny] = np.arange(g.ny) * g.dy
        oy[my - g.ny + 1:] = np.arange(-(g.ny - 1), 0) * g.dy
        ox = np.zeros(mx)
        ox[:g.nx] = np.arange(g.nx) * g.dx
        ox[mx - g.nx + 1:] = np.arange(-(g.nx - 1), 0) * g.dx

        r = np.hypot(oy[:, None], ox[None, :])
        kernel = _radiation_kernel(self.k_b, r, g.cell_area)
        return shape, sp_fft.fft2(kernel)

    def _convolve(self, f, spectrum):
        """Zero-pad, multiply in Fourier, restrict -- ``R C Z`` on ``[..., n_pixels]``.

        Passing ``conj(spectrum)`` gives the exact conjugate transpose: the
        circular convolution's adjoint is correlation with the conjugated
        kernel, and zero-pad and restrict are each other's transpose.
        """
        g = self.grid
        f = np.asarray(f, np.complex128)
        img = f.reshape(-1, g.ny, g.nx)
        spec = sp_fft.fft2(img, s=self._fft_shape, axes=(-2, -1))
        out = sp_fft.ifft2(spec * spectrum, axes=(-2, -1))[:, :g.ny, :g.nx]
        return out.reshape(f.shape)

    def _data_matrix(self):
        """``[n_rx, n_pixels]``, built once. Dense by nature -- receivers are off-grid."""
        if self._data_mat is None:
            nbytes = 16 * self.n_rx * self.grid.n_pixels
            if nbytes > MAX_DATA_MATRIX_BYTES:
                raise MemoryError(
                    f"data matrix would need {nbytes / 2**30:.2f} GB "
                    f"({self.n_rx} receivers x {self.grid.n_pixels} pixels, "
                    "complex128); the finite-difference backend is the "
                    "alternative at this size"
                )
            d = _distances(self.rx_pos, self.grid.cell_centers())
            self._data_mat = _radiation_kernel(self.k_b, d, self.grid.cell_area)
        return self._data_mat

    # ---- GreenOperator ----------------------------------------------------

    def incident(self) -> np.ndarray:
        """Incident field on the grid, ``[n_src, n_pixels]``."""
        return _point_kernel(self.k_b, _distances(self.src_pos, self.grid.cell_centers()))

    def incident_at_receivers(self) -> np.ndarray:
        """Incident field at the receivers, ``[n_src, n_rx]``."""
        return _point_kernel(self.k_b, _distances(self.src_pos, self.rx_pos))

    def domain(self, f: np.ndarray) -> np.ndarray:
        """``G_D``: contrast source ``[n_src, n_pixels]`` -> field on the grid."""
        return self._convolve(f, self._kernel_spectrum)

    def domain_adjoint(self, g: np.ndarray) -> np.ndarray:
        """``G_D^H`` under ``<a, b> = sum(a * conj(b)) * cell_area``."""
        return self._convolve(g, np.conj(self._kernel_spectrum))

    def data(self, f: np.ndarray) -> np.ndarray:
        """``G_S``: contrast source ``[n_src, n_pixels]`` -> field at receivers."""
        f = np.asarray(f, np.complex128).reshape(-1, self.grid.n_pixels)
        return f @ self._data_matrix().T

    def data_adjoint(self, d: np.ndarray) -> np.ndarray:
        """``G_S^H``: ``[n_src, n_rx]`` -> ``[n_src, n_pixels]``."""
        d = np.asarray(d, np.complex128).reshape(-1, self.n_rx)
        return d @ np.conj(self._data_matrix())


__all__ = [
    "IntegralOperator",
    "MAX_DATA_MATRIX_BYTES",
    "background_wavenumber",
    "self_cell_term",
]
