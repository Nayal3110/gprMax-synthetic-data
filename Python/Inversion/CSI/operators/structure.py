"""Known non-inverted structure for the finite-difference grid.

Takes cylinders in survey coordinates -- a fluid-filled borehole, a metal
casing -- and produces the ``k**2`` map and the PEC node mask the
finite-difference operator assembles from.

Structure enters through the background wavenumber only, so whatever it
explains is taken off the contrast before the inversion starts and the
recovered ``chi`` stays a statement about the ground.

Risk: it is optional and defaults to ``None`` because a real survey rarely
knows a casing's radius or material. A wrong cylinder is a modelling error the
data misfit cannot see, so an absent structure is the safer default.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import epsilon_0, mu_0


@dataclass(frozen=True)
class Cylinder:
    """Infinite circular cylinder at ``(x, y)``, seen in cross-section.

    ``pec`` overrides the material: the nodes it covers leave the system
    entirely, which is the right model for a metal casing.
    """

    x: float
    y: float
    radius: float
    eps: float = 1.0
    sigma: float = 0.0
    pec: bool = False

    def mask(self, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """Nodes inside the cylinder, on meshed coordinates."""
        return np.hypot(gx - self.x, gy - self.y) <= self.radius

    def k2(self, omega: float) -> complex:
        """``w**2 mu_0 eps_0 eps_c`` under ``e^{+j w t}``."""
        eps_c = self.eps - 1j * self.sigma / (omega * epsilon_0)
        return complex(omega**2 * mu_0 * epsilon_0 * eps_c)


@dataclass(frozen=True)
class KnownStructure:
    """A hashable bundle of cylinders; hashable so a factorization can cache on it."""

    cylinders: tuple[Cylinder, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "cylinders", tuple(self.cylinders))

    def rasterize(self, x_node: np.ndarray, y_node: np.ndarray, omega: float,
                  k2_bg: complex) -> tuple[np.ndarray, np.ndarray]:
        """``(k2 [ny, nx] complex128, pec [ny, nx] bool)`` on the node lattice.

        Later cylinders win where they overlap, so a casing laid over a fluid
        column reads in the order it is written.
        """
        shape = (len(y_node), len(x_node))
        k2 = np.full(shape, complex(k2_bg), dtype=np.complex128)
        pec = np.zeros(shape, dtype=bool)
        if not self.cylinders:
            return k2, pec
        gx, gy = np.meshgrid(np.asarray(x_node, float), np.asarray(y_node, float))
        for cyl in self.cylinders:
            inside = cyl.mask(gx, gy)
            if cyl.pec:
                pec |= inside
            else:
                k2[inside] = cyl.k2(omega)
        return k2, pec


def borehole_cylinders(positions, radius: float, eps: float = 1.0,
                       sigma: float = 0.0, pec: bool = False) -> KnownStructure:
    """One cylinder per borehole axis in ``positions [N, 2]``.

    ``radius`` is required. A borehole diameter is not recoverable from S21 and
    is not implied by the antenna positions, so a caller that does not know it
    has to say so rather than inherit a plausible-looking default.
    """
    pts = np.asarray(positions, float).reshape(-1, 2)
    if radius <= 0.0:
        raise ValueError(f"radius must be positive, got {radius}")
    return KnownStructure(tuple(
        Cylinder(x=float(p[0]), y=float(p[1]), radius=float(radius),
                 eps=float(eps), sigma=float(sigma), pec=bool(pec))
        for p in pts
    ))


__all__ = ["Cylinder", "KnownStructure", "borehole_cylinders"]
