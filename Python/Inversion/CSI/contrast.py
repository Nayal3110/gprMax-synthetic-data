"""Contrast parameterization: real eps'/sigma contrasts in, complex ``chi(w)`` out.

``update_contrast`` takes the total field and the contrast source of every frequency
in a band group and returns the frequency-independent unknown
``xi = eps' - eps'_bg``. Each frequency's ``chi`` follows from ``xi`` through a known
complex scalar ``a(w) = 1 / eps_cb(w)``.

``chi`` is linear in real unknowns with known coefficients, so one ``xi``
serves a whole band group and the update stays closed-form no matter how many
frequencies the group holds.

Risk: the solve is per-pixel and carries no smoothing, so a pixel the fields barely
reach has a vanishing denominator and is pinned to the background. Regularization,
when it lands, is what fills those in.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from scipy.constants import epsilon_0

from . import EPS_MIN, EPS_WATER_MAX
from .grid import InversionGrid


@dataclass
class ContrastModel:
    """``eps'(r) = eps_bg + xi(r)``, ``sigma(r) = sigma_bg + zeta(r)`` on the grid.

    ``xi`` and ``zeta`` are flat ``[n_pixels]`` float64, row-major. ``zeta is None``
    means the conductivity is held at the background, which is the eps'-only mode the
    solver runs in today.
    """

    grid: InversionGrid
    eps_bg: float
    sigma_bg: float = 0.0
    xi: np.ndarray | None = None
    zeta: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = self.grid.n_pixels
        if self.eps_bg < EPS_MIN:
            raise ValueError(f"eps_bg must be >= {EPS_MIN}, got {self.eps_bg}")
        if self.sigma_bg < 0.0:
            raise ValueError(f"sigma_bg must be >= 0, got {self.sigma_bg}")
        self.xi = (np.zeros(n) if self.xi is None
                   else np.array(self.xi, dtype=float).reshape(n))
        if self.zeta is not None:
            self.zeta = np.array(self.zeta, dtype=float).reshape(n)

    @property
    def eps(self) -> np.ndarray:
        """Absolute eps' per pixel."""
        return self.eps_bg + self.xi

    @property
    def sigma(self) -> np.ndarray:
        """Absolute conductivity per pixel."""
        zeta = 0.0 if self.zeta is None else self.zeta
        return self.sigma_bg + zeta

    @property
    def xi_bounds(self) -> tuple[float, float]:
        """``xi`` range that keeps eps' inside ``[EPS_MIN, EPS_WATER_MAX]``."""
        return (EPS_MIN - self.eps_bg, EPS_WATER_MAX - self.eps_bg)

    def eps_cb(self, omega: float) -> complex:
        """Complex background permittivity, ``e^{+jwt}`` convention."""
        return complex(self.eps_bg - 1j * self.sigma_bg / (omega * epsilon_0))

    def a(self, omega: float) -> complex:
        """The known complex coefficient linking ``xi`` to ``chi`` at this frequency."""
        return 1.0 / self.eps_cb(omega)

    def chi(self, omega: float) -> np.ndarray:
        """``chi(r, w) = (xi - 1j*zeta/(w*eps_0)) * a(w)``, ``[n_pixels]`` complex128."""
        num = self.xi.astype(np.complex128)
        if self.zeta is not None:
            num = num - 1j * self.zeta / (omega * epsilon_0)
        return num * self.a(omega)

    def project(self) -> "ContrastModel":
        """Clip to the physical box: eps' in ``[EPS_MIN, EPS_WATER_MAX]``, sigma >= 0."""
        lo, hi = self.xi_bounds
        xi = np.clip(self.xi, lo, hi)
        zeta = None if self.zeta is None else np.maximum(self.zeta, -self.sigma_bg)
        return replace(self, xi=xi, zeta=zeta)


def update_contrast(model: ContrastModel, omegas: Sequence[float],
                    u: Sequence[np.ndarray], w: Sequence[np.ndarray],
                    eta_d: Sequence[float], solve_sigma: bool = False) -> ContrastModel:
    """Closed-form eps'-only contrast update, then bound projection.

    Minimizing ``sum_f eta_d[f] * sum_s ||a_f*xi*u - w||**2`` over real ``xi`` decouples
    per pixel::

        xi = Re( SUM_{f,s} eta_d * conj(a_f*u) * w ) / SUM_{f,s} eta_d * |a_f*u|**2

    ``u`` and ``w`` are ``[n_src, n_pixels]`` per frequency. ``eta_d`` is the domain
    weight the caller froze at the previous ``chi``; the cell-area factor common to
    numerator and denominator cancels and is left out.

    ``solve_sigma`` is the two-unknown extension point. Carrying ``zeta`` as well turns
    the per-pixel scalar into a real 2x2 normal system in ``(xi, zeta)`` -- the same
    residual differentiated against both unknowns, with cross term
    ``Re(SUM eta_d * conj(a_f*u) * (-1j/(w*eps_0)) * a_f*u)`` -- inverted in closed form
    and projected onto ``sigma >= 0``. Not implemented.
    """
    if solve_sigma:
        raise NotImplementedError(
            "two-unknown (eps', sigma) update is an extension point; see the docstring")
    if not (len(omegas) == len(u) == len(w) == len(eta_d)):
        raise ValueError("omegas, u, w and eta_d must have one entry per frequency")

    n_pix = model.grid.n_pixels
    num = np.zeros(n_pix)
    den = np.zeros(n_pix)
    for omega, u_f, w_f, e_f in zip(omegas, u, w, eta_d):
        au = model.a(omega) * np.asarray(u_f)
        if au.shape[-1] != n_pix:
            raise ValueError(f"field has {au.shape[-1]} pixels, grid has {n_pix}")
        num += e_f * np.sum(np.real(np.conj(au) * np.asarray(w_f)), axis=0)
        den += e_f * np.sum(np.abs(au) ** 2, axis=0)

    # A pixel no field reaches has den == 0; leave it at the background rather than
    # dividing. Everything else is a plain per-pixel least-squares ratio.
    lit = den > 0.0
    xi = np.zeros(n_pix)
    xi[lit] = num[lit] / den[lit]
    return replace(model, xi=xi).project()


__all__ = ["ContrastModel", "update_contrast"]
