"""The solver driven by the real integral-equation operator.

``test_csi_solver`` exercises ``run_csi`` against a dense synthetic operator,
which isolates the CG algebra. These tests close the remaining gap: that the
two fit together, that the gradient is the true gradient of the cost under the
operator's own adjoints, and that the inverse crime holds on real physics.

They also pin the one result that is easy to mistake for a bug -- on a
realistic aperture the recovered ``xi`` is wrong while the data misfit is at
machine zero, because the problem is rank-deficient rather than the solver
broken. See ``test_underdetermined_xi_is_genuinely_non_unique``.
"""
import numpy as np
from scipy.constants import c as C_LIGHT
from scipy.sparse.linalg import LinearOperator, gmres

from Inversion.CSI.contrast import ContrastModel
from Inversion.CSI.csi import run_csi
from Inversion.CSI.grid import InversionGrid
from Inversion.CSI.operators import inner
from Inversion.CSI.operators.integral import IntegralOperator

EPS_BG = 6.0
LAMBDA_BG = 0.2
RADIUS = 0.30


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
    """Scattered field at the receivers for ``xi``, with no solver involved.

    Solves ``(I - chi G_D) w = chi u_inc`` to machine precision, so the data is
    self-consistent to the operator rather than to any iterate of ``run_csi``.
    """
    chi = ContrastModel(grid=op.grid, eps_bg=EPS_BG, sigma_bg=0.0, xi=xi).chi(omega)
    u_inc = op.incident()
    n = op.grid.n_pixels
    w = []
    for s in range(op.n_src):
        a = LinearOperator((n, n), dtype=np.complex128,
                           matvec=lambda x: x - chi * op.domain(x))
        wf, info = gmres(a, chi * u_inc[s], rtol=1e-13, restart=200, maxiter=300)
        assert info == 0, f"gmres failed for source {s} (info={info})"
        w.append(wf)
    return op.data(np.array(w)), np.array(w), chi


def test_gradient_matches_the_line_search_curvature():
    """``B == <v, g>`` for an arbitrary direction, to machine precision.

    The strongest available check that ``g`` really is the cost's gradient
    under the operator's own adjoints. It is exact rather than a finite
    difference, and it is the test that catches an inner-product weighting
    mismatch between ``data_adjoint`` and ``domain_adjoint``: weight one side
    by ``cell_area`` and not the other and the two gradient terms are scaled
    apart, which the exact line search would otherwise hide by still
    descending.
    """
    grid, op, omega, xi_true = scene(n_grid=20, n_ant=8)
    f_meas, _, chi = exact_forward(op, omega, xi_true)

    u_inc = op.incident()
    w = np.zeros_like(u_inc)
    u = u_inc + op.domain(w)
    rho = f_meas - op.data(w)
    r = chi * u - w

    def norm2(a):
        return float(inner(a, a, grid).real)

    eta_s = 1.0 / norm2(f_meas)
    eta_d = 1.0 / norm2(chi * u_inc)
    g = eta_s * op.data_adjoint(rho) + eta_d * (r - op.domain_adjoint(np.conj(chi) * r))

    rng = np.random.default_rng(1)
    v = rng.standard_normal(w.shape) + 1j * rng.standard_normal(w.shape)
    q = chi * op.domain(v) - v
    b = eta_s * inner(op.data(v), rho, grid) - eta_d * inner(q, r, grid)

    assert abs(b - inner(v, g, grid)) / abs(b) < 1e-13


def test_synthesized_data_is_self_consistent():
    """The object equation holds exactly for the w the data was built from."""
    _, op, omega, xi_true = scene(n_grid=20, n_ant=8)
    _, w, chi = exact_forward(op, omega, xi_true)
    u = op.incident() + op.domain(w)
    assert np.linalg.norm(chi * u - w) / np.linalg.norm(w) < 1e-11


def test_inverse_crime_on_the_integral_operator():
    """The step-3 gate: known xi, same operator, no regularization.

    Overdetermined on purpose -- 32 sources and 32 receivers against 100
    pixels, about 20 real measurements per unknown. Measured: F_S reaches
    1e-16 and xi comes back to 4.3e-06 relative in ~3400 iterations.

    Determinacy is the point of the ratio, not incidental. The same solver on
    the same operator at 0.1 measurements per unknown fits the data just as
    well and returns a completely different xi, which is what the companion
    test below pins.
    """
    grid, op, omega, xi_true = scene(n_grid=10, n_ant=32)
    f_meas, _, _ = exact_forward(op, omega, xi_true)

    state = run_csi([op], [f_meas], eps_bg=EPS_BG, n_iter=15000,
                    f_s_tol=1e-16, heldout_frac=0.0)

    rel = np.linalg.norm(state.xi - xi_true) / np.linalg.norm(xi_true)
    assert state.f_s[-1] < 1e-10, f"F_S stalled at {state.f_s[-1]:.2e}"
    assert rel < 1e-5, f"xi relative error {rel:.2e}"


def test_underdetermined_xi_is_genuinely_non_unique():
    """A wrong xi that reproduces the data. Not a bug -- do not 'fix' it.

    At 8 sources and 8 receivers against 1296 pixels (0.1 real measurements
    per unknown) the solver returns an xi that is ~110% wrong against truth,
    yet re-solving the exact forward problem for that xi -- discarding the
    solver's own w entirely -- reproduces the measurements to ~2e-06.

    Two very different permittivity maps therefore generate the same data, so
    no solver could prefer the true one. This is the aperture limit, and it is
    what multiplicative TV regularization exists to break. Recording it here
    so a future reader does not read the large xi error as a solver defect.
    """
    grid, op, omega, xi_true = scene(n_grid=36, n_ant=8)
    f_meas, _, _ = exact_forward(op, omega, xi_true)

    state = run_csi([op], [f_meas], eps_bg=EPS_BG, n_iter=3000,
                    f_s_tol=1e-14, heldout_frac=0.0)

    f_pred, _, _ = exact_forward(op, omega, state.xi)
    misfit = np.linalg.norm(f_pred - f_meas) / np.linalg.norm(f_meas)
    rel = np.linalg.norm(state.xi - xi_true) / np.linalg.norm(xi_true)

    assert rel > 0.5, f"expected a badly wrong xi, got {rel:.3f}"
    assert misfit < 1e-4, f"the wrong xi should still fit the data, got {misfit:.2e}"
    assert 2 * op.n_src * op.n_rx / grid.n_pixels < 0.2
