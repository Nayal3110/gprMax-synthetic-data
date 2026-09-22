"""Multiplicative total-variation regularization for the contrast update.

Takes the current contrast ``xi``, the grid and the data misfit ``F_D``; produces
the TV weights ``b``, the weighted Laplacian ``L(b)`` and the normalized factor
``F_TV``, and a drop-in replacement for ``update_contrast`` that solves the
regularized normal system::

    F_TV(xi)  = (1/V) * INT_D ( |grad xi|**2 + delta**2 ) * b dv
    b         = 1 / ( |grad xi_prev|**2 + delta**2 )
    Cost      = (F_S + F_D) * F_TV

Every constant is tied to the misfit, so a run has no regularization weight to
tune and the smoothing anneals itself away as the fit improves.

Risk: the contrast step is the first Newton step of the multiplicative cost with
``b`` frozen, not the exact minimizer. It is a descent step on a quadratic that
touches the true functional to first order at ``xi_prev``; if a scene comes back
under-regularized, the upgrade is a CG step on ``xi`` with the cubic line search
that makes the ``b``-dependence exact, not a weight added out front.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from .contrast import ContrastModel
from .grid import InversionGrid


def grad_sq(xi: np.ndarray, grid: InversionGrid) -> np.ndarray:
    """``|grad xi|**2`` per pixel, ``[n_pixels]`` float64, Neumann at the edge.

    Each pixel averages the squared differences on the faces it owns, so a face
    missing at the boundary contributes nothing -- the no-flux condition. Written
    this way the pixel sum ``SUM b * grad_sq`` is exactly the quadratic form of
    ``tv_operator(b)``, which is what makes ``F_TV`` and ``L(b)`` agree to the
    last bit rather than to discretization error.
    """
    img = grid.reshape(np.asarray(xi, float))
    ex = np.diff(img, axis=1) / grid.dx
    ey = np.diff(img, axis=0) / grid.dy
    out = np.zeros_like(img)
    out[:, :-1] += 0.5 * ex ** 2
    out[:, 1:] += 0.5 * ex ** 2
    out[:-1, :] += 0.5 * ey ** 2
    out[1:, :] += 0.5 * ey ** 2
    return out.reshape(-1)


def tv_delta_sq(f_d: float, grid: InversionGrid) -> float:
    """``delta**2 = F_D / cell_area`` -- the gradient scale that still counts as flat.

    Carries the units of ``|grad xi|**2``, one over length squared, which is what
    fixes the divisor as the cell area rather than a multiplier. It shrinks with
    the misfit, so late iterations preserve steps that early ones smooth over.
    """
    f_d = float(f_d)
    if f_d < 0.0:
        raise ValueError(f"F_D must be >= 0, got {f_d}")
    return f_d / grid.cell_area


def tv_weights(xi: np.ndarray, grid: InversionGrid, delta_sq: float) -> np.ndarray:
    """``b = 1 / (|grad xi|**2 + delta**2)``, ``[n_pixels]`` float64.

    Large where the previous contrast is flat and small across its steps, so the
    smoothing it weights is applied along edges and not across them.
    """
    if delta_sq <= 0.0:
        raise ValueError(f"delta_sq must be positive, got {delta_sq}")
    return 1.0 / (grad_sq(xi, grid) + delta_sq)


def tv_factor(xi: np.ndarray, b: np.ndarray, grid: InversionGrid,
              delta_sq: float) -> float:
    """``F_TV(xi)``, normalized so ``F_TV(xi_prev) == 1`` for ``b`` built from ``xi_prev``.

    The normalization is what lets the factor multiply the CSI cost without
    changing its scale, and testing it against 1 is the cheapest check that the
    weights and the functional were built from the same gradient.
    """
    b = np.asarray(b, float).reshape(-1)
    return float(np.mean(b * (grad_sq(xi, grid) + delta_sq)))


def tv_operator(b: np.ndarray, grid: InversionGrid) -> sparse.csr_matrix:
    """``L(b) = -div(b grad(.))``, real symmetric PSD ``[n_pixels, n_pixels]``.

    Assembled as a graph Laplacian over the grid's faces with weight
    ``(b_i + b_j) / (2 h**2)``, so ``x.T L x == SUM_faces w (x_i - x_j)**2``:
    symmetry and positive semi-definiteness hold by construction, and a constant
    field is annihilated exactly, which is the Neumann boundary.
    """
    b = np.asarray(b, float).reshape(grid.n_pixels)
    bi = grid.reshape(b)
    wx = 0.5 * (bi[:, :-1] + bi[:, 1:]) / grid.dx ** 2
    wy = 0.5 * (bi[:-1, :] + bi[1:, :]) / grid.dy ** 2

    idx = np.arange(grid.n_pixels).reshape(grid.shape)
    i = np.concatenate([idx[:, :-1].reshape(-1), idx[:-1, :].reshape(-1)])
    j = np.concatenate([idx[:, 1:].reshape(-1), idx[1:, :].reshape(-1)])
    w = np.concatenate([wx.reshape(-1), wy.reshape(-1)])

    n = grid.n_pixels
    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([i, j, j, i])
    vals = np.concatenate([w, w, -w, -w])
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


def normal_terms(model: ContrastModel, omegas: Sequence[float],
                 u: Sequence[np.ndarray], w: Sequence[np.ndarray],
                 eta_d: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """``(num, den)`` of the per-pixel contrast normal equations, ``den * xi = num``.

    The same two sums ``update_contrast`` forms; kept separate here so the TV term
    can be added to ``den`` and the fields are read once either way::

        num = Re( SUM_{f,s} eta_d * conj(a_f*u) * w )
        den =     SUM_{f,s} eta_d * |a_f*u|**2
    """
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
    return num, den


def update_contrast_tv(model: ContrastModel, omegas: Sequence[float],
                       u: Sequence[np.ndarray], w: Sequence[np.ndarray],
                       eta_d: Sequence[float], f_d: float,
                       cg_iter: int = 15, cg_rtol: float = 1e-8) -> ContrastModel:
    """One frozen-weight Newton step of ``(F_S + F_D) * F_TV``, then bound projection.

    With ``b`` held at ``xi_prev`` the multiplicative cost is quadratic in ``xi``
    and its stationary point is the real symmetric positive-definite system::

        [ diag(den) + lambda * L(b_prev) ] xi = num
        lambda   = F_D(w, xi_prev)
        delta**2 = F_D(w, xi_prev) / cell_area

    ``num`` and ``den`` are the closed-form update's own two sums, so the step
    costs one sparse CG solve and no operator application. ``lambda`` and
    ``delta**2`` both track the misfit and their ratio cancels in the flat
    regions, which pins the smoothing strength there while the edge threshold
    keeps falling -- the regularization anneals without a schedule.

    CG is warm-started from ``xi_prev`` and capped at ``cg_iter``: the system
    changes every outer iteration, so solving it exactly buys nothing.
    """
    num, den = normal_terms(model, omegas, u, w, eta_d)
    xi_prev = np.asarray(model.xi, float)

    f_d = float(f_d)
    if f_d <= 0.0:
        # Nothing to anneal against -- the misfit is already zero, so the
        # regularized system degenerates to the closed-form update.
        lit = den > 0.0
        xi = np.zeros_like(den)
        xi[lit] = num[lit] / den[lit]
        return replace(model, xi=xi).project()

    delta_sq = tv_delta_sq(f_d, model.grid)
    b = tv_weights(xi_prev, model.grid, delta_sq)
    a = sparse.diags(den) + f_d * tv_operator(b, model.grid)

    xi, _ = cg(a.tocsr(), num, x0=xi_prev, rtol=cg_rtol, maxiter=cg_iter)
    return replace(model, xi=np.asarray(xi, float)).project()


__all__ = [
    "grad_sq",
    "normal_terms",
    "tv_delta_sq",
    "tv_factor",
    "tv_operator",
    "tv_weights",
    "update_contrast_tv",
]
