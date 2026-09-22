"""Integral-equation Green operator.

Checks run in rising order of what they prove: the discrete adjoint identity
and the FFT application against a dense Toeplitz build (internal consistency),
the self-cell closed form and the domain operator against quadrature (the
discretization), and scattering off a dielectric cylinder against the Bessel
series (the physics -- the only test here that could catch a wrong Green's
function).
"""
import numpy as np
import pytest
from scipy.constants import c as C_LIGHT
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import hankel2, jv

from Inversion.CSI.grid import InversionGrid
from Inversion.CSI.operators import GreenOperator, adjoint_defect, integral
from Inversion.CSI.operators.integral import (
    IntegralOperator,
    background_wavenumber,
    self_cell_term,
)

ADJOINT_TOL = 1e-12


def make_op(nx=11, ny=13, n_src=3, n_rx=4, eps_bg=6.0, sigma_bg=0.0, freq=3e8):
    """A small crosshole-shaped operator. ``nx != ny`` on purpose: a square grid
    hides every axis-swap bug in the circulant embedding."""
    grid = InversionGrid(x_min=0.0, x_max=0.55, y_min=0.0, y_max=0.65, nx=nx, ny=ny)
    omega = 2.0 * np.pi * freq
    src = np.column_stack([np.full(n_src, -0.30), np.linspace(0.05, 0.60, n_src)])
    rx = np.column_stack([np.full(n_rx, 0.85), np.linspace(0.05, 0.60, n_rx)])
    return IntegralOperator.from_background(grid, omega, eps_bg, src, rx, sigma_bg)


# --- quadrature references --------------------------------------------------


def _polar_quadrature(k_b, radius, n_rho, weight_phi=None, n_phi=1):
    """``k_b**2 * INT (-1j/4) H_0^(2)(k_b rho) dA`` in polar coordinates.

    ``radius`` is either a scalar (a disc) or ``R(phi)`` on the ``n_phi``
    Gauss-Legendre nodes of ``weight_phi``. Gauss-Legendre nodes are open, so
    the log singularity at ``rho = 0`` is never sampled, and the ``rho`` factor
    in the area element kills it anyway.
    """
    xr, wr = np.polynomial.legendre.leggauss(n_rho)
    radius = np.broadcast_to(np.atleast_1d(radius), (n_phi,))
    rho = radius[:, None] * 0.5 * (xr[None, :] + 1.0)
    w_rho = radius[:, None] * 0.5 * wr[None, :]
    integrand = -0.25j * k_b**2 * hankel2(0, k_b * rho) * rho
    radial = np.sum(integrand * w_rho, axis=1)
    if weight_phi is None:
        return complex(2.0 * np.pi * radial[0])
    return complex(np.sum(radial * weight_phi))


def disc_quadrature(k_b, a, n_rho=600):
    """Brute-force integral over a disc of radius ``a`` centred on the singularity."""
    return _polar_quadrature(k_b, a, n_rho)


def square_quadrature(k_b, h, n_rho=600, n_phi=200):
    """Brute-force 2-D integral over an ``h x h`` square centred on the singularity.

    Eight-fold symmetry reduces it to the 45-degree sector where the boundary
    is ``R(phi) = (h/2)/cos(phi)``, which keeps the polar map smooth.
    """
    xp, wp = np.polynomial.legendre.leggauss(n_phi)
    phi = 0.25 * np.pi * 0.5 * (xp + 1.0)
    w_phi = 8.0 * 0.25 * np.pi * 0.5 * wp
    return _polar_quadrature(k_b, (0.5 * h) / np.cos(phi), n_rho, w_phi, n_phi)


# --- adjoints ---------------------------------------------------------------


def test_satisfies_green_operator_protocol():
    assert isinstance(make_op(), GreenOperator)


@pytest.mark.parametrize("sigma_bg", [0.0, 0.01])
def test_domain_adjoint_defect_is_machine_precision(sigma_bg):
    op = make_op(sigma_bg=sigma_bg)
    n = op.grid.n_pixels
    defect = adjoint_defect(op.domain, op.domain_adjoint,
                            (op.n_src, n), (op.n_src, n), op.grid)
    assert defect < ADJOINT_TOL, f"G_D adjoint defect {defect:.3e}"


@pytest.mark.parametrize("sigma_bg", [0.0, 0.01])
def test_data_adjoint_defect_is_machine_precision(sigma_bg):
    op = make_op(sigma_bg=sigma_bg)
    defect = adjoint_defect(op.data, op.data_adjoint,
                            (op.n_src, op.grid.n_pixels), (op.n_src, op.n_rx), op.grid)
    assert defect < ADJOINT_TOL, f"G_S adjoint defect {defect:.3e}"


def test_domain_adjoint_holds_on_a_non_square_grid():
    """Guards the circulant embedding: nx and ny enter the padding separately."""
    op = make_op(nx=7, ny=23)
    n = op.grid.n_pixels
    defect = adjoint_defect(op.domain, op.domain_adjoint, (2, n), (2, n), op.grid)
    assert defect < ADJOINT_TOL, f"defect {defect:.3e}"


# --- the FFT application itself ---------------------------------------------


def dense_domain_matrix(op):
    """``G_D`` written out entry by entry, straight from the kernel definition."""
    c = op.grid.cell_centers()
    r = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    a = -0.25j * op.k_b**2 * op.grid.cell_area * hankel2(0, op.k_b * np.where(r == 0, 1.0, r))
    np.fill_diagonal(a, self_cell_term(op.k_b, op.grid.cell_area))
    return a


def test_fft_application_matches_dense_toeplitz():
    op = make_op(nx=9, ny=14, n_src=2)
    a = dense_domain_matrix(op)
    rng = np.random.default_rng(3)
    f = rng.standard_normal((op.n_src, op.grid.n_pixels)) + \
        1j * rng.standard_normal((op.n_src, op.grid.n_pixels))
    np.testing.assert_allclose(op.domain(f), f @ a.T, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(op.domain_adjoint(f), f @ np.conj(a), rtol=1e-12, atol=1e-14)


def test_domain_accepts_a_bare_pixel_vector():
    op = make_op(n_src=1)
    f = np.arange(op.grid.n_pixels) + 0j
    np.testing.assert_allclose(op.domain(f), op.domain(f[None, :])[0], rtol=0, atol=0)


# --- the self cell ----------------------------------------------------------


@pytest.mark.parametrize("k_h", [0.05, 0.2, 0.5])
def test_self_cell_matches_disc_quadrature(k_h):
    """The closed form *is* the disc integral, so this must hold to quadrature
    precision -- it is what pins the sign."""
    h = 0.01
    k_b = k_h / h
    a = h / np.sqrt(np.pi)  # equal-area radius of an h x h cell
    closed = self_cell_term(k_b, h * h)
    brute = disc_quadrature(k_b, a)
    assert abs(closed - brute) / abs(brute) < 1e-9, f"{closed} vs {brute}"


@pytest.mark.parametrize("k_h", [0.05, 0.2, 0.5])
def test_self_cell_matches_square_cell_quadrature(k_h):
    """The equal-area patch stands in for the square cell; check the swap is
    small on the quantity that actually enters the matrix.

    Measured 0.25-0.50% over this range of ``k_b * h``, essentially all of it in
    the real part: ``Re G = -Y_0/4`` is the log-singular half, and reshaping the
    cell is exactly what a log integral notices. The smooth half
    ``Im G = -J_0/4`` only sees the second-moment mismatch between a square and
    its equal-area disc, ``(k h)**2 * (1/6 - 1/(2 pi)) / 4 = 0.0019 (k h)**2``,
    so it is checked separately against that scaling.
    """
    h = 0.01
    k_b = k_h / h
    closed = self_cell_term(k_b, h * h)
    brute = square_quadrature(k_b, h)
    rel = abs(closed - brute) / abs(brute)
    assert rel < 1e-2, f"equal-area patch off by {rel:.3e} at k*h = {k_h}"
    rel_imag = abs(closed.imag - brute.imag) / abs(brute.imag)
    assert rel_imag < 4e-3 * k_h**2, f"smooth part off by {rel_imag:.3e}"


def test_self_cell_is_lossy_background_safe():
    """A complex k_b must not flip the branch or blow up."""
    k_b = background_wavenumber(2 * np.pi * 3e8, 6.0, 0.05)
    assert k_b.real > 0 and k_b.imag < 0
    assert np.isfinite(self_cell_term(k_b, 1e-4))


# --- incident field ---------------------------------------------------------


def test_incident_matches_hankel2_directly():
    op = make_op()
    c = op.grid.cell_centers()
    u = op.incident()
    assert u.shape == (op.n_src, op.grid.n_pixels)
    assert u.dtype == np.complex128
    for i, s in enumerate(op.src_pos):
        r = np.hypot(c[:, 0] - s[0], c[:, 1] - s[1])
        np.testing.assert_allclose(u[i], -0.25j * hankel2(0, op.k_b * r), rtol=1e-14)


def test_incident_at_receivers_matches_hankel2_directly():
    op = make_op()
    u = op.incident_at_receivers()
    assert u.shape == (op.n_src, op.n_rx)
    for i, s in enumerate(op.src_pos):
        r = np.hypot(op.rx_pos[:, 0] - s[0], op.rx_pos[:, 1] - s[1])
        np.testing.assert_allclose(u[i], -0.25j * hankel2(0, op.k_b * r), rtol=1e-14)


def test_source_on_a_cell_centre_is_rejected():
    grid = InversionGrid(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=4, ny=4)
    op = IntegralOperator(grid, 2 * np.pi * 3e8, 20.0 + 0j,
                          [[0.125, 0.125]], [[2.0, 0.5]])
    with pytest.raises(ValueError, match="cell centre"):
        op.incident()


# --- the dense data matrix guard --------------------------------------------


def test_data_matrix_refuses_to_exceed_the_ceiling(monkeypatch):
    """The ceiling is lowered rather than the grid raised -- a grid big enough
    to trip 1 GB for real costs a 6000-square FFT just to build the kernel."""
    op = make_op(nx=40, ny=40, n_rx=9)
    monkeypatch.setattr(integral, "MAX_DATA_MATRIX_BYTES", 1000)
    with pytest.raises(MemoryError, match="finite-difference"):
        op.data(np.zeros((op.n_src, op.grid.n_pixels), complex))
    with pytest.raises(MemoryError, match="finite-difference"):
        op.data_adjoint(np.zeros((op.n_src, op.n_rx), complex))


def test_the_real_ceiling_is_about_a_gigabyte():
    assert integral.MAX_DATA_MATRIX_BYTES == pytest.approx(2**30, rel=0.1)


# --- convergence ------------------------------------------------------------


def refined_operator(n, half_width, k_b):
    grid = InversionGrid(x_min=-half_width, x_max=half_width,
                         y_min=-half_width, y_max=half_width, nx=n, ny=n)
    return IntegralOperator(grid, 1.0, k_b, [[10.0, 0.0]], [[10.0, 0.0]])


def test_domain_converges_under_refinement():
    """``G_D`` applied to a uniform contrast source, read at the domain centre.

    The exact answer is the singular integral over the whole square, which the
    polar quadrature above gives to many digits. Odd ``n`` puts a cell centre
    exactly on the evaluation point, so there is no interpolation in the way.
    """
    side = 0.2
    k_b = 2.0 * np.pi / side  # one background wavelength across the domain
    exact = square_quadrature(k_b, side, n_rho=800, n_phi=400)

    errors = []
    for n in (21, 41, 81):
        op = refined_operator(n, 0.5 * side, k_b)
        got = op.domain(np.ones(op.grid.n_pixels, complex))[(n // 2) * n + n // 2]
        errors.append(abs(got - exact) / abs(exact))

    assert errors[1] < errors[0] and errors[2] < errors[1], errors
    # Measured 3.7e-3 -> 9.6e-4 -> 2.4e-4, a factor 3.9 per halving: clean
    # second order, as midpoint quadrature should give away from the singular
    # cell. Demand 10 of the 16 that two halvings of a second-order scheme buy.
    assert errors[0] / errors[2] > 10.0, f"convergence too slow: {errors}"


# --- Mie: the only test of the physics --------------------------------------


def mie_scattered_field(k_b, k_1, radius, src_xy, rx_xy, n_max=50):
    """Scattered field of a dielectric cylinder at the origin, line-source lit.

    Graf's addition theorem writes the line-source field inside ``rho < rho_0``
    as ``sum_n a_n J_n(k_b rho) e^{jn(phi - phi_0)}`` with
    ``a_n = -(1j/4) H_n^(2)(k_b rho_0)``. Continuity of ``E_z`` and ``H_phi``
    (so of ``u`` and ``du/drho``, non-magnetic) fixes the scattering
    coefficients ``b_n``.
    """
    n = np.arange(-n_max, n_max + 1)

    def d(fn, order, z):  # Z_n'(z) = (Z_{n-1} - Z_{n+1}) / 2
        return 0.5 * (fn(order - 1, z) - fn(order + 1, z))

    jb, jbp = jv(n, k_b * radius), d(jv, n, k_b * radius)
    j1, j1p = jv(n, k_1 * radius), d(jv, n, k_1 * radius)
    hb, hbp = hankel2(n, k_b * radius), d(hankel2, n, k_b * radius)
    b = -(k_1 * j1p * jb - k_b * jbp * j1) / (k_1 * j1p * hb - k_b * hbp * j1)

    src_xy = np.asarray(src_xy, float)
    rho_0 = np.hypot(*src_xy)
    phi_0 = np.arctan2(src_xy[1], src_xy[0])
    a = -0.25j * hankel2(n, k_b * rho_0)

    rx_xy = np.asarray(rx_xy, float)
    rho = np.hypot(rx_xy[:, 0], rx_xy[:, 1])
    phi = np.arctan2(rx_xy[:, 1], rx_xy[:, 0])
    terms = (a * b)[None, :] * hankel2(n[None, :], k_b * rho[:, None]) \
        * np.exp(1j * n[None, :] * (phi[:, None] - phi_0))
    assert np.all(np.isfinite(terms)), "mode series overflowed; lower n_max"
    return terms.sum(axis=1)


def cylinder_contrast(grid, radius, chi, oversample=8):
    """Contrast map for a disc, cells weighted by the fraction they cover.

    Sub-cell area weighting rather than a hard in/out test: staircasing the
    boundary is otherwise the dominant error against the analytic series.
    """
    off = (np.arange(oversample) + 0.5) / oversample - 0.5
    c = grid.cell_centers()
    sx = c[:, 0][:, None, None] + (off[None, :, None] * grid.dx)
    sy = c[:, 1][:, None, None] + (off[None, None, :] * grid.dy)
    frac = np.mean(np.hypot(sx, sy) <= radius, axis=(1, 2))
    return chi * frac


def solve_contrast_source(op, chi, u_inc):
    """Solve ``(I - chi G_D) w = chi u_inc`` for the contrast source."""
    n = op.grid.n_pixels
    a = LinearOperator((n, n), dtype=np.complex128,
                       matvec=lambda w: w - chi * op.domain(w))
    w, info = gmres(a, chi * u_inc, rtol=1e-11, restart=120, maxiter=60)
    assert info == 0, f"gmres did not converge (info={info})"
    return w


def test_mie_cylinder_matches_the_bessel_series():
    """Contrast 1, 87 cells across the cylinder. Measured 0.044%.

    The tolerance is set close to that on purpose. At 3% this test would still
    pass with the self-cell term negated (0.88%) or dropped (0.46%); at 0.3% it
    catches both, on top of the gross failures it was written for -- an
    ``H^(1)`` kernel gives 131%, and losing the ``-1`` in the self cell 84%.
    """
    lam_b, eps_bg, eps_cyl = 0.2, 6.0, 12.0
    radius = 0.45 * lam_b
    k_b = 2.0 * np.pi / lam_b
    k_1 = k_b * np.sqrt(eps_cyl / eps_bg)
    freq = C_LIGHT / (lam_b * np.sqrt(eps_bg))

    half = radius + 2.5 * lam_b / 40.0
    n = 88  # ~40 cells per background wavelength, ~28 per interior wavelength
    grid = InversionGrid(x_min=-half, x_max=half, y_min=-half, y_max=half, nx=n, ny=n)

    src = np.array([[0.0, -0.5]])
    ang = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    rx = np.column_stack([0.5 * np.cos(ang), 0.5 * np.sin(ang)])

    op = IntegralOperator.from_background(grid, 2 * np.pi * freq, eps_bg, src, rx)
    assert abs(op.k_b - k_b) / k_b < 1e-12  # from_background agrees with the series

    chi = cylinder_contrast(grid, radius, eps_cyl / eps_bg - 1.0)
    w = solve_contrast_source(op, chi, op.incident()[0])
    got = op.data(w[None, :])[0]

    want = mie_scattered_field(k_b, k_1, radius, src[0], rx)
    rel = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert rel < 3e-3, f"Mie relative error {rel:.3%}"


def test_mie_error_falls_when_the_cylinder_is_better_resolved():
    """The remaining Mie error is discretization, not a wrong operator."""
    lam_b, eps_bg, eps_cyl = 0.2, 6.0, 12.0
    radius = 0.45 * lam_b
    k_b = 2.0 * np.pi / lam_b
    k_1 = k_b * np.sqrt(eps_cyl / eps_bg)
    freq = C_LIGHT / (lam_b * np.sqrt(eps_bg))
    src = np.array([[0.0, -0.5]])
    ang = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    rx = np.column_stack([0.5 * np.cos(ang), 0.5 * np.sin(ang)])
    want = mie_scattered_field(k_b, k_1, radius, src[0], rx)

    errors = []
    for n in (44, 88, 132):
        half = radius * (1.0 + 1.5 / n)
        grid = InversionGrid(x_min=-half, x_max=half, y_min=-half, y_max=half, nx=n, ny=n)
        op = IntegralOperator.from_background(grid, 2 * np.pi * freq, eps_bg, src, rx)
        chi = cylinder_contrast(grid, radius, eps_cyl / eps_bg - 1.0)
        w = solve_contrast_source(op, chi, op.incident()[0])
        got = op.data(w[None, :])[0]
        errors.append(np.linalg.norm(got - want) / np.linalg.norm(want))

    assert errors[1] < errors[0] and errors[2] < errors[1], errors
