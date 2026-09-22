"""Green operator interface. Six methods; the solver sees nothing else.

An operator is built for one angular frequency and one background, and holds
the grid, the background wavenumber and the source and receiver positions. It
answers two questions: what field do the sources produce, and what does a
contrast source on the grid radiate -- onto the grid itself (``domain``, the
operator ``G_D``) and onto the receivers (``data``, the operator ``G_S``).

Batching over sources lives inside the operator. Every array carries a leading
``n_src`` axis, which is what lets a factorized backend do one multi-RHS
back-substitution instead of ``n_src`` separate solves.

Both backends must satisfy the discrete adjoint identity to machine precision
-- not to discretization error -- under::

    <a, b> = sum(a * conj(b)) * grid.cell_area

That two-line test catches almost every operator bug, so it is the first thing
a new backend has to pass.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.constants import epsilon_0, mu_0

from ..grid import InversionGrid


def background_wavenumber(omega: float, eps_bg: float, sigma_bg: float = 0.0) -> complex:
    """``k_b`` for a homogeneous background under ``e^{+j w t}``.

    The principal square root already lands on ``Re k > 0``, ``Im k <= 0``,
    which is the branch on which ``e^{-j k r}`` decays outward. Backends share
    this so a comparison between them cannot differ by a branch choice.
    """
    if omega <= 0.0:
        raise ValueError(f"omega must be positive, got {omega}")
    eps_c = eps_bg - 1j * sigma_bg / (omega * epsilon_0)
    return complex(omega * np.sqrt(mu_0 * epsilon_0 * eps_c))


@runtime_checkable
class GreenOperator(Protocol):
    """One frequency, one background. Shapes are fixed by ``n_src``/``n_rx``/grid."""

    grid: InversionGrid
    omega: float
    k_b: complex
    n_src: int
    n_rx: int

    def incident(self) -> np.ndarray:
        """Incident field on the grid, ``[n_src, n_pix]``."""
        ...

    def incident_at_receivers(self) -> np.ndarray:
        """Incident field at the receivers, ``[n_src, n_rx]``."""
        ...

    def domain(self, f: np.ndarray) -> np.ndarray:
        """``G_D``: contrast source ``[n_src, n_pix]`` -> field on grid ``[n_src, n_pix]``."""
        ...

    def domain_adjoint(self, g: np.ndarray) -> np.ndarray:
        """``G_D^H``, adjoint under the cell-area inner product."""
        ...

    def data(self, f: np.ndarray) -> np.ndarray:
        """``G_S``: contrast source ``[n_src, n_pix]`` -> field at receivers ``[n_src, n_rx]``."""
        ...

    def data_adjoint(self, d: np.ndarray) -> np.ndarray:
        """``G_S^H``: ``[n_src, n_rx]`` -> ``[n_src, n_pix]``."""
        ...


def inner(a: np.ndarray, b: np.ndarray, grid: InversionGrid) -> complex:
    """Discrete inner product ``<a, b>`` weighted by cell area."""
    return complex(np.sum(np.asarray(a) * np.conj(np.asarray(b))) * grid.cell_area)


def adjoint_defect(forward, adjoint, shape_in, shape_out, grid: InversionGrid,
                   seed: int = 0) -> float:
    """Relative defect in ``<forward(a), b> == <a, adjoint(b)>``.

    Returned rather than asserted so tests can report the number they saw. A
    correct backend gives something at the 1e-15 level; anything at 1e-8 or
    above is a bug, not round-off.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape_in) + 1j * rng.standard_normal(shape_in)
    b = rng.standard_normal(shape_out) + 1j * rng.standard_normal(shape_out)
    lhs = inner(forward(a), b, grid)
    rhs = inner(a, adjoint(b), grid)
    scale = max(abs(lhs), abs(rhs), 1e-300)
    return abs(lhs - rhs) / scale


__all__ = ["GreenOperator", "inner", "adjoint_defect"]
