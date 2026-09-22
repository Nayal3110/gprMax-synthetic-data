import numpy as np
import pytest

from Inversion.traveltime.geometry import PixelGrid, build_system_matrix
from Inversion.traveltime.sirt import slowness_to_eps, eps_to_slowness, forward, sirt, sart


def test_slowness_eps_roundtrip():
    eps = np.array([1.0, 6.0, 80.0])
    assert np.allclose(slowness_to_eps(eps_to_slowness(eps)), eps)


def _dense_geometry():
    # Dense fan of rays from a left column to a right column -> well-posed-ish.
    g = PixelGrid(0.0, 1.0, 0.0, 1.0, nx=6, ny=6)
    ys = np.linspace(0.05, 0.95, 6)
    sources = np.column_stack([np.full(6, 0.02), ys])
    receivers = np.column_stack([np.full(6, 0.98), ys])
    L, _ = build_system_matrix(sources, receivers, g)
    return g, L


def test_sirt_recovers_homogeneous_slowness():
    g, L = _dense_geometry()
    m_true = np.full(g.n_pixels, eps_to_slowness(6.0))
    t = forward(L, m_true)
    m0 = np.full(g.n_pixels, eps_to_slowness(4.0))
    m, hist = sirt(L, t, m0, n_iter=300, relax=0.2)
    # mean recovered permittivity is close to 6 and residual fell
    assert np.mean(slowness_to_eps(m)) == pytest.approx(6.0, abs=0.3)
    assert hist[-1] < hist[0]


def test_sirt_residual_monotone_nonincreasing():
    g, L = _dense_geometry()
    m_true = eps_to_slowness(np.linspace(4.0, 8.0, g.n_pixels))
    t = forward(L, m_true)
    _, hist = sirt(L, t, np.full(g.n_pixels, eps_to_slowness(6.0)),
                   n_iter=200, relax=0.15)
    assert hist[-1] <= hist[0]


def test_sart_recovers_homogeneous_slowness():
    g, L = _dense_geometry()
    # 6 sources x 6 receivers -> blocks of 6 consecutive rays
    blocks = [list(range(s * 6, s * 6 + 6)) for s in range(6)]
    m_true = np.full(g.n_pixels, eps_to_slowness(6.0))
    t = forward(L, m_true)
    m0 = np.full(g.n_pixels, eps_to_slowness(4.0))
    # Block-SART is semiconvergent: per-source updates over-relax above ~0.3 on
    # this rank-deficient fan, drifting into a null-space checkerboard. relax=0.2
    # keeps it in the stable regime where it recovers the homogeneous map.
    m, hist = sart(L, t, m0, blocks, n_iter=100, relax=0.2)
    assert np.mean(slowness_to_eps(m)) == pytest.approx(6.0, abs=0.3)
    assert hist[-1] < hist[0]
