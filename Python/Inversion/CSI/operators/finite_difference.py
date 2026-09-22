"""Finite-difference Helmholtz Green operator on a stretched-coordinate grid.

Takes an ``InversionGrid``, a background wavenumber and point source/receiver
positions; produces the incident field and the two radiation operators ``G_D``
(grid -> grid) and ``G_S`` (grid -> receivers), each with its exact discrete
adjoint. A node-centred 5-point stencil is assembled on the inversion grid
padded by a buffer and a complex-stretched PML, factorized once, and applied by
back-substitution with every source as a column of one multi-RHS solve.

Every row ``(i, j)`` is multiplied through by ``sx_i * sy_j``. That makes the
matrix complex-symmetric -- ``A == A.T``, not Hermitian -- so ``A^H = conj(A)``
and

    A^H x = b   <=>   x = conj( A^-1 conj(b) ).

One LU factorization therefore serves both the forward and the adjoint solve,
which is what keeps a CSI iteration at four applications and no refactorization.

Risk: the LU is the cost and it grows super-linearly. Measured here, complex128
with the default COLAMD ordering: 200**2 unknowns factor in 0.8 s and 58 MB,
400**2 in 27.5 s and 294 MB. The cache holds a few of those at a time.

Conventions
-----------
``e^{+j w t}``, outgoing ``e^{-j k r}``, ``G_2D = -(1j/4) H_0^(2)(k_b r)``.
``G_D`` carries the ``k_b**2`` of the contrast-source form. Pixels flatten
row-major, ``flat = iy * nx + ix``.

The discrete 5-point Green's function differs from the analytic Hankel by an
``O(1)`` factor near the source, so the source amplitude is calibrated once at
construction against the Hankel on a ring a few cells out. Every solve carries
that one complex scalar, which puts the operator on the analytic scale.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
from scipy.constants import epsilon_0, mu_0
from scipy.sparse.linalg import splu
from scipy.special import hankel2

from ..grid import InversionGrid, pml_cells
from . import background_wavenumber
from .pml import PML_ORDER, stretch_profile
from .structure import KnownStructure

# Radius of the calibration ring, in cells, and how many points sit on it.
CALIBRATION_RING_CELLS = 5.0
CALIBRATION_RING_POINTS = 16

# Cells of plain background kept between the inversion domain (or the outermost
# antenna) and the PML face, so nothing physical sits in the graded layer.
PAD_MARGIN_CELLS = 4

# Factorizations kept alive across operators. Small on purpose: a single 400**2
# factor is ~300 MB, so an unbounded cache is a memory leak with a nice name.
MAX_CACHED_FACTORIZATIONS = 4
_FACTOR_CACHE: "OrderedDict[tuple, object]" = OrderedDict()


def clear_factor_cache() -> None:
    """Drop every cached LU. Frequency continuation walks away from old rungs."""
    _FACTOR_CACHE.clear()


def _cached_factorization(key: tuple, build):
    """``splu`` result for ``key``, built on first use and reused after."""
    lu = _FACTOR_CACHE.get(key)
    if lu is None:
        lu = build()
        _FACTOR_CACHE[key] = lu
        while len(_FACTOR_CACHE) > MAX_CACHED_FACTORIZATIONS:
            _FACTOR_CACHE.popitem(last=False)
    else:
        _FACTOR_CACHE.move_to_end(key)
    return lu


class FiniteDifferenceOperator:
    """Helmholtz Green operator for one angular frequency and one background.

    Satisfies ``GreenOperator``. Every field array carries a leading ``n_src``
    axis; ``domain``/``domain_adjoint`` also accept a bare ``[n_pixels]`` vector
    and hand back the same shape.

    ``structure`` is a ``KnownStructure`` or ``None``. It enters the ``k_b**2``
    map and the PEC mask only, never ``chi``, and ``None`` -- a homogeneous
    background -- is the configuration a real survey runs in.
    """

    def __init__(self, grid: InversionGrid, omega: float, k_b: complex,
                 src_pos, rx_pos, n_pml: int | None = None,
                 n_pad: int | None = None, kappa_max: float = 1.0,
                 structure: KnownStructure | None = None):
        self.grid = grid
        self.omega = float(omega)
        self.k_b = complex(k_b)
        self.src_pos = np.asarray(src_pos, float).reshape(-1, 2)
        self.rx_pos = np.asarray(rx_pos, float).reshape(-1, 2)
        self.n_src = int(self.src_pos.shape[0])
        self.n_rx = int(self.rx_pos.shape[0])
        self.kappa_max = float(kappa_max)
        self.structure = structure

        # Background permittivity implied by k_b; the PML is graded for it, so
        # deriving it here rather than taking it separately keeps the two from
        # disagreeing.
        eps_c = self.k_b**2 / (self.omega**2 * mu_0 * epsilon_0)
        self.eps_bg = float(eps_c.real)
        self.n_pml = (int(pml_cells(grid, self.eps_bg, self.omega / (2.0 * np.pi)))
                      if n_pml is None else int(n_pml))

        self._layout(n_pad)
        self._assemble()
        self._interpolants()
        self._amp = 1.0 + 0.0j
        self._amp = self._calibrate()
        self._u_inc_nodes: np.ndarray | None = None

    @classmethod
    def from_background(cls, grid: InversionGrid, omega: float, eps_bg: float,
                        src_pos, rx_pos, sigma_bg: float = 0.0,
                        **kwargs) -> "FiniteDifferenceOperator":
        """Build from a background permittivity and conductivity instead of ``k_b``."""
        return cls(grid, omega, background_wavenumber(omega, eps_bg, sigma_bg),
                   src_pos, rx_pos, **kwargs)

    # ---- layout -----------------------------------------------------------

    def _layout(self, n_pad: int | None) -> None:
        """Node lattice: inversion cells, a background buffer, then the PML."""
        g = self.grid
        pts = np.vstack([self.src_pos, self.rx_pos]) if (self.n_src + self.n_rx) \
            else np.empty((0, 2))
        lo = pts.min(axis=0) if len(pts) else np.array([g.x_min, g.y_min])
        hi = pts.max(axis=0) if len(pts) else np.array([g.x_max, g.y_max])

        if n_pad is None:
            pad = [PAD_MARGIN_CELLS + max(0, math.ceil((g.x_min - lo[0]) / g.dx)),
                   PAD_MARGIN_CELLS + max(0, math.ceil((hi[0] - g.x_max) / g.dx)),
                   PAD_MARGIN_CELLS + max(0, math.ceil((g.y_min - lo[1]) / g.dy)),
                   PAD_MARGIN_CELLS + max(0, math.ceil((hi[1] - g.y_max) / g.dy))]
        else:
            pad = [int(n_pad)] * 4
        self.n_pad = tuple(pad)

        self.nx_t = pad[0] + g.nx + pad[1] + 2 * self.n_pml
        self.ny_t = pad[2] + g.ny + pad[3] + 2 * self.n_pml
        self._off_x = self.n_pml + pad[0]
        self._off_y = self.n_pml + pad[2]
        self.x_node = g.x_min + (np.arange(self.nx_t) - self._off_x + 0.5) * g.dx
        self.y_node = g.y_min + (np.arange(self.ny_t) - self._off_y + 0.5) * g.dy

        # Antennas must sit in the interior: the graded layer is not the
        # background, so a source inside it radiates the wrong field.
        box = (self.x_node[self.n_pml], self.x_node[self.nx_t - 1 - self.n_pml],
               self.y_node[self.n_pml], self.y_node[self.ny_t - 1 - self.n_pml])
        if len(pts) and not (np.all(pts[:, 0] >= box[0]) and np.all(pts[:, 0] <= box[1])
                             and np.all(pts[:, 1] >= box[2])
                             and np.all(pts[:, 1] <= box[3])):
            raise ValueError(
                f"a source or receiver lies in or beyond the PML; the interior is "
                f"x in [{box[0]:.4g}, {box[1]:.4g}], y in [{box[2]:.4g}, "
                f"{box[3]:.4g}] -- raise n_pad or drop it to None"
            )

    # ---- assembly ---------------------------------------------------------

    def _assemble(self) -> None:
        """Row-scaled 5-point stencil, PEC nodes removed, factorized once."""
        g = self.grid
        sx_n, sx_h = stretch_profile(self.nx_t, self.n_pml, g.dx, self.omega,
                                     self.eps_bg, self.kappa_max)
        sy_n, sy_h = stretch_profile(self.ny_t, self.n_pml, g.dy, self.omega,
                                     self.eps_bg, self.kappa_max)
        k2_bg = self.k_b**2
        if self.structure is None:
            k2 = np.full((self.ny_t, self.nx_t), k2_bg, dtype=np.complex128)
            pec = np.zeros((self.ny_t, self.nx_t), dtype=bool)
        else:
            k2, pec = self.structure.rasterize(self.x_node, self.y_node,
                                               self.omega, k2_bg)

        ax = 1.0 / sx_h
        ay = 1.0 / sy_h
        # Row (i, j) multiplied through by sx_i * sy_j. The east coefficient at
        # i and the west coefficient at i + 1 are then the same expression, which
        # is what makes A symmetric rather than merely close to it.
        ce = sy_n[:, None] * ax[None, 1:] / g.dx**2
        cw = sy_n[:, None] * ax[None, :-1] / g.dx**2
        cn = sx_n[None, :] * ay[1:, None] / g.dy**2
        cs = sx_n[None, :] * ay[:-1, None] / g.dy**2
        s_prod = sy_n[:, None] * sx_n[None, :]
        # Outside the last node the field is held at zero, so the boundary
        # coefficients stay in the diagonal and never become an off-diagonal.
        diag = -(ce + cw + cn + cs) + s_prod * k2

        n_node = self.nx_t * self.ny_t
        idx = np.arange(n_node).reshape(self.ny_t, self.nx_t)
        rows = [idx.ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel(),
                idx[:-1, :].ravel(), idx[1:, :].ravel()]
        cols = [idx.ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel(),
                idx[1:, :].ravel(), idx[:-1, :].ravel()]
        data = [diag.ravel(), ce[:, :-1].ravel(), cw[:, 1:].ravel(),
                cn[:-1, :].ravel(), cs[1:, :].ravel()]
        a = sp.coo_matrix((np.concatenate(data),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n_node, n_node)).tocsr()

        self._keep = np.flatnonzero(~pec.ravel())
        self._red = np.full(n_node, -1, dtype=np.int64)
        self._red[self._keep] = np.arange(self._keep.size)
        if self._keep.size != n_node:
            a = a[self._keep][:, self._keep]
        self.matrix = a.tocsc()
        self._n_red = int(self._keep.size)
        self._s_prod = s_prod.ravel()[self._keep]

        # Pixel (iy, ix) is node (iy + off_y, ix + off_x). A pixel buried in a
        # PEC body has no unknown; it stays out of both the embed and the
        # restrict, which keeps the two exact transposes of each other.
        piy, pix = np.divmod(np.arange(g.n_pixels), g.nx)
        node = (piy + self._off_y) * self.nx_t + (pix + self._off_x)
        red = self._red[node]
        self._pix_valid = red >= 0
        self._pix_red = red[self._pix_valid]

        key = (g, self.omega, self.k_b, self.nx_t, self.ny_t, self._off_x,
               self._off_y, self.n_pml, self.kappa_max, PML_ORDER, self.structure)
        self._lu = _cached_factorization(key, lambda: splu(self.matrix))

    def _interpolants(self) -> None:
        """Bilinear sampling matrices for the sources and the receivers."""
        self._src_w = self._weights(self.src_pos)
        self._rx_w = self._weights(self.rx_pos)
        # Injecting with the transpose of the sampling matrix makes the discrete
        # system reciprocal, since A is symmetric.
        self._src_rho = np.asarray(self._src_w.todense()) / self.grid.cell_area

    def _weights(self, pts: np.ndarray) -> sp.csr_matrix:
        """``[n_pts, n_unknowns]`` bilinear sampling weights on the node lattice."""
        g = self.grid
        pts = np.asarray(pts, float).reshape(-1, 2)
        n = pts.shape[0]
        fx = (pts[:, 0] - self.x_node[0]) / g.dx
        fy = (pts[:, 1] - self.y_node[0]) / g.dy
        i0 = np.clip(np.floor(fx).astype(np.int64), 0, self.nx_t - 2)
        j0 = np.clip(np.floor(fy).astype(np.int64), 0, self.ny_t - 2)
        tx = fx - i0
        ty = fy - j0
        corners = [(0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)),
                   (0, 1, (1 - tx) * ty), (1, 1, tx * ty)]
        rows = np.concatenate([np.arange(n)] * 4)
        cols = np.concatenate([(j0 + dj) * self.nx_t + (i0 + di)
                               for di, dj, _ in corners])
        data = np.concatenate([w for _, _, w in corners])
        m = sp.coo_matrix((data, (rows, cols)),
                          shape=(n, self.nx_t * self.ny_t)).tocsr()
        return m[:, self._keep].tocsr() if self._n_red != m.shape[1] else m

    # ---- solves -----------------------------------------------------------

    def _solve(self, rho: np.ndarray) -> np.ndarray:
        """Field from a source density on the nodes, ``[n, n_unknowns]`` both ways."""
        b = (-self._amp * self._s_prod)[None, :] * rho
        return np.asarray(self._lu.solve(np.asfortranarray(b.T))).T

    def _solve_adjoint(self, y: np.ndarray) -> np.ndarray:
        """Exact conjugate transpose of ``_solve``, on the same LU.

        ``A`` is complex-symmetric, so ``A^-H y == conj(A^-1 conj(y))`` and the
        forward factorization answers the adjoint solve untouched.
        """
        z = np.asarray(self._lu.solve(np.asfortranarray(np.conj(y).T))).T
        return np.conj((-self._amp * self._s_prod)[None, :] * z)

    def _embed(self, f: np.ndarray) -> np.ndarray:
        """Pixels -> nodes, ``[n, n_pixels]`` -> ``[n, n_unknowns]``."""
        out = np.zeros((f.shape[0], self._n_red), dtype=np.complex128)
        out[:, self._pix_red] = f[:, self._pix_valid]
        return out

    def _restrict(self, y: np.ndarray) -> np.ndarray:
        """Nodes -> pixels; the exact transpose of ``_embed``."""
        out = np.zeros((y.shape[0], self.grid.n_pixels), dtype=np.complex128)
        out[:, self._pix_valid] = y[:, self._pix_red]
        return out

    def _calibrate(self) -> complex:
        """One complex scalar matching the discrete point source to the Hankel.

        A unit line source is solved for at the centre of the inversion domain
        and compared with ``-(1j/4) H_0^(2)(k_b r)`` on a ring of
        ``CALIBRATION_RING_CELLS`` cells. The ring is far enough out that the
        stencil's near-source distortion has decayed and close enough in that
        numerical dispersion has not yet accumulated.
        """
        g = self.grid
        i = int(round((0.5 * (g.x_min + g.x_max) - self.x_node[0]) / g.dx))
        j = int(round((0.5 * (g.y_min + g.y_max) - self.y_node[0]) / g.dy))
        i = int(np.clip(i, self.n_pml, self.nx_t - 1 - self.n_pml))
        j = int(np.clip(j, self.n_pml, self.ny_t - 1 - self.n_pml))
        red = self._red[j * self.nx_t + i]
        if red < 0:
            raise ValueError("the domain centre is a PEC node; cannot calibrate there")

        rho = np.zeros((1, self._n_red), dtype=np.complex128)
        rho[0, red] = 1.0 / g.cell_area
        u = self._solve(rho)

        r_cal = CALIBRATION_RING_CELLS * math.sqrt(g.dx * g.dy)
        ang = np.linspace(0.0, 2.0 * np.pi, CALIBRATION_RING_POINTS, endpoint=False)
        ring = np.column_stack([self.x_node[i] + r_cal * np.cos(ang),
                                self.y_node[j] + r_cal * np.sin(ang)])
        obs = np.asarray(self._weights(ring) @ u[0])
        ref = np.full(ang.shape, -0.25j * hankel2(0, self.k_b * r_cal),
                      dtype=np.complex128)
        return complex(np.vdot(obs, ref) / np.vdot(obs, obs))

    # ---- GreenOperator ----------------------------------------------------

    def _incident_nodes(self) -> np.ndarray:
        if self._u_inc_nodes is None:
            self._u_inc_nodes = self._solve(self._src_rho)
        return self._u_inc_nodes

    def incident(self) -> np.ndarray:
        """Incident field on the grid, ``[n_src, n_pixels]``."""
        return self._restrict(self._incident_nodes())

    def incident_at_receivers(self) -> np.ndarray:
        """Incident field at the receivers, ``[n_src, n_rx]``."""
        return np.asarray(self._rx_w @ self._incident_nodes().T).T

    def domain(self, f: np.ndarray) -> np.ndarray:
        """``G_D``: contrast source ``[n_src, n_pixels]`` -> field on the grid."""
        f = np.asarray(f, np.complex128)
        flat = f.reshape(-1, self.grid.n_pixels)
        out = self.k_b**2 * self._restrict(self._solve(self._embed(flat)))
        return out.reshape(f.shape)

    def domain_adjoint(self, g: np.ndarray) -> np.ndarray:
        """``G_D^H`` under ``<a, b> = sum(a * conj(b)) * cell_area``."""
        g = np.asarray(g, np.complex128)
        flat = g.reshape(-1, self.grid.n_pixels)
        out = np.conj(self.k_b**2) * self._restrict(
            self._solve_adjoint(self._embed(flat)))
        return out.reshape(g.shape)

    def data(self, f: np.ndarray) -> np.ndarray:
        """``G_S``: contrast source ``[n_src, n_pixels]`` -> field at receivers."""
        flat = np.asarray(f, np.complex128).reshape(-1, self.grid.n_pixels)
        u = self._solve(self._embed(flat))
        return self.k_b**2 * np.asarray(self._rx_w @ u.T).T

    def data_adjoint(self, d: np.ndarray) -> np.ndarray:
        """``G_S^H``: ``[n_src, n_rx]`` -> ``[n_src, n_pixels]``."""
        d = np.asarray(d, np.complex128).reshape(-1, self.n_rx)
        y = np.asarray(self._rx_w.T @ d.T).T
        return np.conj(self.k_b**2) * self._restrict(self._solve_adjoint(y))


__all__ = [
    "FiniteDifferenceOperator",
    "CALIBRATION_RING_CELLS",
    "MAX_CACHED_FACTORIZATIONS",
    "PAD_MARGIN_CELLS",
    "background_wavenumber",
    "clear_factor_cache",
]
