"""Complex coordinate stretching for the finite-difference Helmholtz boundary.

``stretch_profile`` takes an axis length in nodes, a PML depth in cells, the
cell size, the angular frequency and the background permittivity, and returns
the stretching factor ``s`` at the nodes and at the half-points the 5-point
stencil differences across.

``sigma_max`` is gprMax's own optimum for the same quartic profile, so a later
FD-versus-FDTD comparison measures the solver rather than the boundary.

Risk: the profile is graded for a propagating wave at the background
wavenumber, so a source or receiver placed inside the graded region sees a
medium that is not the background. Antennas belong in the interior.

Conventions
-----------
``e^{+j w t}``, so ``s = kappa - 1j * sigma / (w * eps_0)`` and an outgoing
``e^{-j k x}`` decays inside the layer for ``Re k > 0``.
"""
from __future__ import annotations

import numpy as np
from scipy.constants import epsilon_0, mu_0

# Free-space wave impedance, written exactly as ``gprMax.constants.z0`` writes
# it -- ``np.sqrt(m0 / e0)`` off the same scipy constants -- so the borrowed
# sigma_max formula lands on the same float without importing gprMax.
Z0 = float(np.sqrt(mu_0 / epsilon_0))

# Polynomial order of the conductivity grading. gprMax's default CFS uses the
# 'quartic' profile, i.e. m = 4 (``gprMax/gprMax/pml.py``, ``CFS.__init__``).
PML_ORDER = 4


def sigma_max(d: float, eps_r: float, mu_r: float = 1.0, m: int = PML_ORDER) -> float:
    """Optimum PML conductivity for a cell size ``d`` in the layer's direction.

    Verbatim from gprMax ``gprMax/gprMax/pml.py``, ``CFS.calculate_sigmamax``
    (v3.1.7, around line 83), which cites http://dx.doi.org/10.1109/8.546249::

        self.sigma.max = (0.8 * (m + 1)) / (z0 * d * np.sqrt(er * mr))

    ``d`` is the cell size, not the layer thickness -- reproducing gprMax's
    reading of it is the whole point of borrowing the formula.
    """
    if d <= 0.0:
        raise ValueError(f"cell size must be positive, got {d}")
    if eps_r <= 0.0 or mu_r <= 0.0:
        raise ValueError(f"eps_r and mu_r must be positive, got {eps_r}, {mu_r}")
    return (0.8 * (m + 1)) / (Z0 * d * np.sqrt(eps_r * mu_r))


def _depth(p: np.ndarray, n: int, n_pml: int, d: float) -> np.ndarray:
    """Normalized depth into the layer at axis coordinate ``p``: 0 inside, 1 at the wall."""
    if n_pml == 0:
        return np.zeros_like(p)
    t = n_pml * d
    into_low = (t - p) / t
    into_high = (p - (n - n_pml) * d) / t
    return np.clip(np.maximum(into_low, into_high), 0.0, 1.0)


def stretch_profile(n: int, n_pml: int, d: float, omega: float, eps_r: float,
                    kappa_max: float = 1.0, m: int = PML_ORDER,
                    mu_r: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """``(s_node [n], s_half [n + 1])`` along one axis of an ``n``-node grid.

    Node ``i`` sits at ``(i + 0.5) * d`` from the wall and half-point ``i`` at
    ``i * d``, so ``s_half[i]`` and ``s_half[i + 1]`` bracket node ``i``. The
    outer ``n_pml`` cells at each end carry the graded layer.

    ``kappa_max`` is 1 in gprMax; a frequency-domain solve often wants 5-15,
    which stretches the real coordinate as well and buys extra absorption of
    grazing and evanescent components at no extra depth.
    """
    n = int(n)
    n_pml = int(n_pml)
    if n_pml < 0:
        raise ValueError(f"n_pml must be >= 0, got {n_pml}")
    if 2 * n_pml >= n:
        raise ValueError(f"two {n_pml}-cell layers do not fit in {n} nodes")
    if omega <= 0.0:
        raise ValueError(f"omega must be positive, got {omega}")
    if kappa_max < 1.0:
        raise ValueError(f"kappa_max must be >= 1, got {kappa_max}")

    s_max = sigma_max(d, eps_r, mu_r, m)

    def stretch(p: np.ndarray) -> np.ndarray:
        graded = _depth(p, n, n_pml, d) ** m
        kappa = 1.0 + (kappa_max - 1.0) * graded
        return kappa - 1j * (s_max * graded) / (omega * epsilon_0)

    p_node = (np.arange(n) + 0.5) * d
    p_half = np.arange(n + 1) * d
    return stretch(p_node), stretch(p_half)


__all__ = ["PML_ORDER", "Z0", "sigma_max", "stretch_profile"]
