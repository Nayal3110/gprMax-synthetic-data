"""SIRT / SART travel-time inversion for slowness, plus permittivity conversion.

Model: t = L @ m, with m = slowness [s/m] per pixel and L = ray-cell lengths [m].
Velocity v = 1/m; relative permittivity eps_r = (c * m) ** 2 (non-magnetic medium).
"""
from __future__ import annotations

import numpy as np

_MU_0 = np.pi * 4e-7
_EPS_0 = 8.854187817e-12
C = 1.0 / np.sqrt(_MU_0 * _EPS_0)


def slowness_to_eps(m: np.ndarray) -> np.ndarray:
    return (C * np.asarray(m, float)) ** 2


def eps_to_slowness(eps: np.ndarray) -> np.ndarray:
    return np.sqrt(np.asarray(eps, float)) / C


def forward(L, m: np.ndarray) -> np.ndarray:
    return L @ np.asarray(m, float)


def _safe_reciprocal(v: np.ndarray) -> np.ndarray:
    """1/v where v > 0, else 0 (no divide-by-zero warning for empty cells)."""
    out = np.zeros_like(v, dtype=float)
    nz = v > 0
    out[nz] = 1.0 / v[nz]
    return out


def _eps_bounds_to_slowness(bounds):
    if bounds is None:
        return None
    lo, hi = bounds
    return eps_to_slowness(lo), eps_to_slowness(hi)


def sirt(L, t: np.ndarray, m0: np.ndarray, n_iter: int = 300, relax: float = 0.2,
         eps_bounds=(1.0, 81.0)):
    """Simultaneous Iterative Reconstruction Technique.

    Gilbert weighting: m <- m + relax * C_col @ L.T @ R_row @ (t - L m),
    R_row = 1/row_sums (ray length), C_col = 1/col_sums (pixel coverage).
    Returns ``(m, residual_history)``.
    """
    m = np.asarray(m0, float).copy()
    t = np.asarray(t, float)
    row = np.asarray(L.sum(axis=1)).ravel()
    col = np.asarray(L.sum(axis=0)).ravel()
    R = _safe_reciprocal(row)
    Cc = _safe_reciprocal(col)
    s_bounds = _eps_bounds_to_slowness(eps_bounds)
    history = []
    for _ in range(n_iter):
        resid = t - L @ m
        history.append(float(np.linalg.norm(resid)))
        m = m + relax * (Cc * (L.T @ (R * resid)))
        if s_bounds is not None:
            # eps lo->hi maps to slowness lo->hi (monotone), so clip directly
            m = np.clip(m, s_bounds[0], s_bounds[1])
    history.append(float(np.linalg.norm(t - L @ m)))
    return m, history


def sart(L, t: np.ndarray, m0: np.ndarray, blocks, n_iter: int = 100,
         relax: float = 1.0, eps_bounds=(1.0, 81.0)):
    """Simultaneous Algebraic Reconstruction Technique with per-source blocks.

    One iteration sweeps every block (e.g. all rays of one source position),
    applying the SIRT-style weighted update sequentially. ``blocks`` is a list
    of ray-index lists. Returns ``(m, residual_history)`` (full-system residual
    recorded once per sweep).
    """
    m = np.asarray(m0, float).copy()
    t = np.asarray(t, float)
    s_bounds = _eps_bounds_to_slowness(eps_bounds)
    prepared = []
    for b in blocks:
        Lb = L[b]
        row = np.asarray(Lb.sum(axis=1)).ravel()
        col = np.asarray(Lb.sum(axis=0)).ravel()
        prepared.append((Lb, np.asarray(b), _safe_reciprocal(row), _safe_reciprocal(col)))
    history = [float(np.linalg.norm(t - L @ m))]
    for _ in range(n_iter):
        for Lb, b, R, Cc in prepared:
            resid = t[b] - Lb @ m
            m = m + relax * (Cc * (Lb.T @ (R * resid)))
            if s_bounds is not None:
                m = np.clip(m, s_bounds[0], s_bounds[1])
        history.append(float(np.linalg.norm(t - L @ m)))
    return m, history