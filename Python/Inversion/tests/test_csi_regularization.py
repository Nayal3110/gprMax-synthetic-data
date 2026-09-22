"""Multiplicative TV: the weights, the Laplacian, and whether it earns its place.

The algebraic tests pin the two identities the whole scheme rests on -- that
``F_TV(xi_prev)`` is exactly 1, and that ``L(b)`` is the true gradient operator of
the functional the weights came from. The last test is the one that justifies the
module: a noisy disc, inverted twice.

Measured on the aperture-limited scene of ``test_csi_integration`` (8 sources, 8
receivers, 1296 pixels) TV does **not** help: 1.105 relative without it against
1.143 with it, both fitting the data at 1e-05 or better. At 0.1 real measurements
per unknown the smoothing prior picks a different wrong map, not the right one, so
that case stays what it is -- an aperture limit, not a regularization gap.
"""
import numpy as np
import pytest
from scipy.constants import c as C_LIGHT
from scipy.sparse.linalg import LinearOperator, gmres

from Inversion.CSI.contrast import ContrastModel, update_contrast
from Inversion.CSI.csi import run_csi
from Inversion.CSI.grid import InversionGrid
from Inversion.CSI.operators.integral import IntegralOperator
from Inversion.CSI.regularization import (grad_sq, tv_delta_sq, tv_factor,
                                          tv_operator, tv_weights,
                                          update_contrast_tv)

EPS_BG = 6.0
LAMBDA_BG = 0.2
RADIUS = 0.30

# Deliberately non-square and unequal in dx/dy: an operator that quietly assumes
# dx == dy or nx == ny still passes on a square grid.
GRID = InversionGrid(x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.7, nx=13, ny=9)


def random_field(seed, grid=GRID, scale=1.0):
    return np.random.default_rng(seed).standard_normal(grid.n_pixels) * scale


# ---- the weights and the functional ------------------------------------------


@pytest.mark.parametrize("seed, delta_sq, scale", [
    (0, 1e-8, 0.1), (1, 1e-4, 10.0), (2, 1.0, 1.0),
    (3, 1e4, 0.1), (4, 1e8, 10.0), (5, 3.7, 1e-3),
])
def test_tv_factor_is_exactly_one_at_the_contrast_the_weights_came_from(
        seed, delta_sq, scale):
    """``F_TV(xi_prev) == 1``. The cheapest check that b and F_TV agree.

    Every term is ``b_p * (grad_sq_p + delta**2)`` with ``b_p`` the reciprocal of
    that same quantity, so the mean is 1 to the last bit whatever the scale of the
    contrast or of ``delta``. A mismatched gradient stencil between the two shows
    up here immediately.
    """
    xi = random_field(seed, scale=scale)
    b = tv_weights(xi, GRID, delta_sq)
    assert abs(tv_factor(xi, b, GRID, delta_sq) - 1.0) < 1e-14


def test_tv_gradient_matches_finite_differences():
    """``dF_TV/dxi == (2/N) * L(b) @ xi`` with ``b`` frozen.

    Central differences on every pixel against the analytic gradient. This is what
    makes the frozen-weight contrast step a Newton step of the real functional
    rather than of something adjacent to it.
    """
    delta_sq = 0.7
    b = tv_weights(random_field(0), GRID, delta_sq)
    L = tv_operator(b, GRID)
    xi = random_field(1)

    analytic = (2.0 / GRID.n_pixels) * (L @ xi)
    h = 1e-6
    numeric = np.empty_like(xi)
    for p in range(GRID.n_pixels):
        e = np.zeros_like(xi)
        e[p] = h
        numeric[p] = (tv_factor(xi + e, b, GRID, delta_sq)
                      - tv_factor(xi - e, b, GRID, delta_sq)) / (2 * h)

    err = np.linalg.norm(numeric - analytic) / np.linalg.norm(analytic)
    assert err < 1e-6, f"gradient check off by {err:.2e}"


def test_delta_shrinks_as_the_misfit_falls():
    """The regularization anneals itself: no schedule, no knob.

    ``delta**2`` is the gradient scale below which a feature is treated as flat, so
    a falling misfit lets progressively weaker steps survive as edges.
    """
    deltas = [tv_delta_sq(f_d, GRID) for f_d in (1.0, 1e-2, 1e-4, 1e-8)]
    assert all(a > b > 0.0 for a, b in zip(deltas, deltas[1:]))
    assert tv_delta_sq(1.0, GRID) == pytest.approx(1.0 / GRID.cell_area)


def test_tv_weights_are_small_across_a_step_and_large_where_flat():
    """The edge-preserving property, stated as a number rather than as intent."""
    img = np.zeros(GRID.shape)
    img[:, GRID.nx // 2:] = 5.0
    b = GRID.reshape(tv_weights(img.reshape(-1), GRID, tv_delta_sq(1e-3, GRID)))
    assert b[:, GRID.nx // 2].max() < 1e-3 * b[:, 0].min()


# ---- the operator ------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tv_operator_is_symmetric_and_positive_semidefinite(seed):
    """Real SPD is what lets the contrast step use CG instead of a general solver."""
    L = tv_operator(tv_weights(random_field(seed), GRID, 0.3), GRID)
    assert L.dtype == np.float64

    rng = np.random.default_rng(seed + 10)
    x, y = rng.standard_normal((2, GRID.n_pixels))
    lhs, rhs = float(x @ (L @ y)), float(y @ (L @ x))
    assert abs(lhs - rhs) / abs(lhs) < 1e-13

    for v in rng.standard_normal((20, GRID.n_pixels)):
        assert float(v @ (L @ v)) >= 0.0


def test_tv_operator_annihilates_a_constant():
    """``L @ 1 == 0`` -- a constant field has no variation, and the Neumann check.

    A boundary term left in would break this first, so it doubles as the boundary
    condition test.
    """
    L = tv_operator(tv_weights(random_field(4), GRID, 0.3), GRID)
    ones = np.ones(GRID.n_pixels)
    assert np.max(np.abs(L @ ones)) < 1e-12 * np.max(np.abs(L.data))


def test_the_operator_is_the_quadratic_form_of_the_functional():
    """``x.T L(b) x == SUM_p b_p * grad_sq_p(x)``.

    The identity that ties ``tv_operator`` to ``grad_sq``, and through it the
    Newton step to the functional the normalization is defined on.
    """
    b = tv_weights(random_field(5), GRID, 0.9)
    x = random_field(6)
    quad = float(x @ (tv_operator(b, GRID) @ x))
    assert quad == pytest.approx(float(np.sum(b * grad_sq(x, GRID))), rel=1e-12)


def test_grad_sq_reads_a_linear_ramp():
    """A ramp of slope 1 in x reads ``|grad| == 1`` in the interior."""
    xs = GRID.cell_centers()[:, 0]
    g = GRID.reshape(grad_sq(xs, GRID))
    assert np.allclose(g[:, 1:-1], 1.0)


# ---- the contrast step -------------------------------------------------------


def synthetic_fields(seed=0, n_src=3, grid=GRID):
    """A ContrastModel plus (omegas, u, w, eta_d) with a known closed-form answer."""
    rng = np.random.default_rng(seed)
    model = ContrastModel(grid=grid, eps_bg=EPS_BG)
    omegas = [2 * np.pi * 1e8]
    u = [rng.standard_normal((n_src, grid.n_pixels))
         + 1j * rng.standard_normal((n_src, grid.n_pixels))]
    w = [rng.standard_normal((n_src, grid.n_pixels))
         + 1j * rng.standard_normal((n_src, grid.n_pixels))]
    return model, omegas, u, w, [1.0]


def test_zero_misfit_falls_back_to_the_closed_form_update():
    """At ``F_D == 0`` there is nothing to regularize against, and the step agrees
    with ``update_contrast`` exactly."""
    model, omegas, u, w, eta_d = synthetic_fields()
    plain = update_contrast(model, omegas, u, w, eta_d)
    tv = update_contrast_tv(model, omegas, u, w, eta_d, f_d=0.0)
    assert np.array_equal(plain.xi, tv.xi)


def test_the_step_fills_a_pixel_no_field_reaches():
    """The closed-form update pins an unlit pixel to the background; TV interpolates.

    ``contrast.py`` names this as the gap regularization exists to close, so it is
    worth holding to.
    """
    model, omegas, u, w, eta_d = synthetic_fields(seed=2)
    dark = GRID.n_pixels // 2
    u[0][:, dark] = 0.0

    plain = update_contrast(model, omegas, u, w, eta_d)
    tv = update_contrast_tv(model, omegas, u, w, eta_d, f_d=1e-2, cg_iter=200)
    neighbours = [dark - 1, dark + 1, dark - GRID.nx, dark + GRID.nx]

    assert plain.xi[dark] == 0.0
    assert abs(tv.xi[dark] - np.mean(tv.xi[neighbours])) < 0.25 * np.std(tv.xi)


def test_the_step_stays_inside_the_permittivity_box():
    """Bound projection survives the CG solve."""
    model, omegas, u, w, eta_d = synthetic_fields(seed=3)
    tv = update_contrast_tv(model, omegas, u, w, eta_d, f_d=1e-3)
    lo, hi = model.xi_bounds
    assert np.all(tv.xi >= lo) and np.all(tv.xi <= hi)


# ---- the solver --------------------------------------------------------------


def scene(n_grid, n_ant, xi_amp=6.0):
    """Ring of sources and interleaved receivers around a dielectric disc."""
    omega = 2 * np.pi * C_LIGHT / (LAMBDA_BG * np.sqrt(EPS_BG))
    grid = InversionGrid(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5,
                         nx=n_grid, ny=n_grid)
    ang = np.linspace(0.0, 2 * np.pi, n_ant, endpoint=False)
    src = np.column_stack([0.9 * np.cos(ang), 0.9 * np.sin(ang)])
    off = ang + np.pi / n_ant
    rx = np.column_stack([0.85 * np.cos(off), 0.85 * np.sin(off)])
    op = IntegralOperator.from_background(grid, omega, EPS_BG, src, rx)

    c = grid.cell_centers()
    xi_true = np.where(np.hypot(c[:, 0], c[:, 1]) <= RADIUS, xi_amp, 0.0)
    return grid, op, omega, xi_true


def exact_forward(op, omega, xi):
    """Scattered field at the receivers for ``xi``, with no solver involved."""
    chi = ContrastModel(grid=op.grid, eps_bg=EPS_BG, xi=xi).chi(omega)
    u_inc = op.incident()
    n = op.grid.n_pixels
    w = []
    for s in range(op.n_src):
        a = LinearOperator((n, n), dtype=np.complex128,
                           matvec=lambda x: x - chi * op.domain(x))
        wf, info = gmres(a, chi * u_inc[s], rtol=1e-13, restart=200, maxiter=300)
        assert info == 0, f"gmres failed for source {s} (info={info})"
        w.append(wf)
    return op.data(np.array(w))


def noisy_disc(n_grid=10, n_ant=32, noise=0.05, seed=3):
    """Disc data at a given fraction of white noise, per real and imaginary part."""
    grid, op, omega, xi_true = scene(n_grid, n_ant)
    f = exact_forward(op, omega, xi_true)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(f.shape) + 1j * rng.standard_normal(f.shape)
    return op, xi_true, f + noise * np.linalg.norm(f) / np.sqrt(2 * f.size) * z


def test_tv_is_off_by_default():
    """The default path is the unregularized one, to the bit.

    An aperture-limited fit is non-unique as a matter of physics, and
    ``test_csi_integration`` pins that. Turning TV on by default would quietly
    rewrite what the default solver means.
    """
    op, _, f_meas = noisy_disc(n_grid=8, n_ant=8, noise=0.1)
    kw = dict(eps_bg=EPS_BG, n_iter=40, heldout_frac=0.0)
    assert np.array_equal(run_csi([op], [f_meas], **kw).xi,
                          run_csi([op], [f_meas], tv=False, **kw).xi)


def test_tv_beats_no_tv_on_noisy_data():
    """The test the module exists for. 5% noise on a disc, 32 sources, 100 pixels.

    Measured: 0.101 relative error without TV against 0.036 with it, at 1500
    iterations and the same data. Both fit the data to the noise floor
    (``F_S`` ~ 1.5e-04), so the gap is entirely the prior refusing to spend the
    contrast on noise -- without TV the reconstruction is still tracking noise down
    at 4000 iterations (0.103), with TV it has settled by 400 (0.037).
    """
    op, xi_true, f_meas = noisy_disc()
    kw = dict(eps_bg=EPS_BG, n_iter=1500, f_s_tol=1e-14, df_tol=0.0,
              heldout_frac=0.0)

    off = run_csi([op], [f_meas], **kw)
    on = run_csi([op], [f_meas], tv=True, **kw)

    err_off = np.linalg.norm(off.xi - xi_true) / np.linalg.norm(xi_true)
    err_on = np.linalg.norm(on.xi - xi_true) / np.linalg.norm(xi_true)
    assert err_on < 0.5 * err_off, f"TV {err_on:.3f} vs plain {err_off:.3f}"
    assert err_on < 0.1, f"TV relative error {err_on:.3f}"


def test_tv_reaches_the_same_answer_when_the_data_is_clean():
    """Regularization that cannot be switched off is a bias; this one can.

    Noise-free and overdetermined, ``delta**2`` and ``lambda`` both follow ``F_D``
    to zero and the TV run recovers the disc as exactly as the plain one -- and in
    fewer iterations, because the smoothing removes the null-space wander the
    unregularized run has to work through.
    """
    grid, op, omega, xi_true = scene(n_grid=10, n_ant=32)
    f_meas = exact_forward(op, omega, xi_true)
    state = run_csi([op], [f_meas], eps_bg=EPS_BG, n_iter=4000, f_s_tol=1e-14,
                    df_tol=0.0, heldout_frac=0.0, tv=True)
    rel = np.linalg.norm(state.xi - xi_true) / np.linalg.norm(xi_true)
    assert rel < 1e-4, f"xi relative error {rel:.2e}"
