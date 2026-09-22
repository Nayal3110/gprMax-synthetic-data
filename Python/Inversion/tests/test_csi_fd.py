"""Finite-difference Helmholtz Green operator.

Checks run in rising order of what they prove: the matrix is complex-symmetric
and the two adjoints are exact (internal consistency, and what licenses the
one-LU trick), the boundary does not reflect (the absorber), the operator
agrees with the integral-equation backend at second order (the discretization,
and the only test here that could catch a wrong stencil), and CSI recovers a
known contrast through it (the operator and the solver together).

``structure=None`` -- a homogeneous background, which is what a real survey
gives -- is the configuration every test covers first. The structured case is
an addition on top, never a precondition.
"""
import numpy as np
import pytest
from scipy.constants import c as C_LIGHT
from scipy.constants import epsilon_0, mu_0

from Inversion.CSI.contrast import ContrastModel
from Inversion.CSI.csi import run_csi
from Inversion.CSI.grid import InversionGrid
from Inversion.CSI.operators import GreenOperator, adjoint_defect
from Inversion.CSI.operators.finite_difference import (
    FiniteDifferenceOperator,
    clear_factor_cache,
)
from Inversion.CSI.operators.integral import IntegralOperator
from Inversion.CSI.operators.pml import Z0, sigma_max, stretch_profile
from Inversion.CSI.operators.structure import (
    Cylinder,
    KnownStructure,
    borehole_cylinders,
)

ADJOINT_TOL = 1e-12
EPS_BG = 6.0
FREQ = 3e8


def make_op(nx=11, ny=13, n_src=3, n_rx=4, eps_bg=EPS_BG, sigma_bg=0.0,
            freq=FREQ, **kwargs):
    """A small crosshole-shaped operator. ``nx != ny`` on purpose: a square grid
    hides every axis-swap bug in the stencil assembly."""
    grid = InversionGrid(x_min=0.0, x_max=0.55, y_min=0.0, y_max=0.65, nx=nx, ny=ny)
    omega = 2.0 * np.pi * freq
    src = np.column_stack([np.full(n_src, -0.30), np.linspace(0.05, 0.60, n_src)])
    rx = np.column_stack([np.full(n_rx, 0.85), np.linspace(0.05, 0.60, n_rx)])
    return FiniteDifferenceOperator.from_background(grid, omega, eps_bg, src, rx,
                                                    sigma_bg, **kwargs)


def casing():
    """A PEC casing beside each borehole, off the antenna line so nothing is buried."""
    return KnownStructure((
        Cylinder(x=-0.36, y=0.32, radius=0.04, pec=True),
        Cylinder(x=0.91, y=0.32, radius=0.04, pec=True),
        Cylinder(x=0.20, y=0.30, radius=0.06, eps=81.0, sigma=0.01),
    ))


# --- PML profile ------------------------------------------------------------


def test_sigma_max_is_gprmaxs_formula():
    """Pinned against ``CFS.calculate_sigmamax``: ``0.8*(m+1)/(Z0*d*sqrt(er*mr))``.

    Copied rather than imported, so the copy has to be checked. A drift here
    would silently make an FD-versus-FDTD boundary comparison meaningless.
    """
    d, er, mr, m = 0.01, 6.0, 1.0, 4
    # gprMax's own z0 is np.sqrt(m0 / e0) off these same scipy constants.
    assert Z0 == float(np.sqrt(mu_0 / epsilon_0))
    assert Z0 == pytest.approx(376.7303134, rel=1e-9)
    assert sigma_max(d, er, mr, m) == pytest.approx(
        (0.8 * (m + 1)) / (Z0 * d * np.sqrt(er * mr)), rel=1e-15)


def test_stretch_is_unity_inside_and_lossy_at_the_wall():
    s_node, s_half = stretch_profile(40, 8, 0.01, 2 * np.pi * FREQ, EPS_BG)
    assert np.allclose(s_node[8:-8], 1.0)
    assert np.allclose(s_half[9:-9], 1.0)
    # e^{+jwt}: an outgoing wave decays only if the imaginary part is negative.
    assert s_node[0].imag < 0.0 and s_node[-1].imag < 0.0
    assert abs(s_node[0]) == pytest.approx(abs(s_node[-1]), rel=1e-12)
    depth = np.abs(s_node[:8].imag)
    assert np.all(np.diff(depth) < 0.0), "the layer must grade towards the wall"


def test_stretch_profile_rejects_a_layer_that_does_not_fit():
    with pytest.raises(ValueError):
        stretch_profile(10, 5, 0.01, 2 * np.pi * FREQ, EPS_BG)


# --- symmetry and adjoints --------------------------------------------------


def test_satisfies_green_operator_protocol():
    assert isinstance(make_op(), GreenOperator)


def _symmetry_residual(a):
    d = abs(a - a.T)
    return (d.max() if d.nnz else 0.0) / abs(a).max()


def test_matrix_is_complex_symmetric():
    """``A == A.T`` exactly. This is what licenses one LU for both solves.

    Row ``(i, j)`` is scaled by ``sx_i * sy_j`` precisely so the east
    coefficient at ``i`` and the west coefficient at ``i + 1`` are the same
    expression. Anything but a bitwise-zero residual means the scaling was
    dropped somewhere and ``domain_adjoint`` is quietly solving the wrong system.
    """
    for op in (make_op(), make_op(structure=casing())):
        resid = _symmetry_residual(op.matrix)
        assert resid == 0.0, f"A - A.T is {resid:.3e} relative"


def test_matrix_is_not_hermitian():
    """The complement to the test above: symmetric, and deliberately not Hermitian."""
    a = make_op().matrix
    assert _symmetry_residual(a) == 0.0
    assert abs(a - a.conj().T).max() / abs(a).max() > 1e-3


@pytest.mark.parametrize("structure", [None, casing()], ids=["plain", "structured"])
def test_domain_adjoint_is_exact(structure):
    op = make_op(structure=structure)
    n = (op.n_src, op.grid.n_pixels)
    defect = adjoint_defect(op.domain, op.domain_adjoint, n, n, op.grid)
    assert defect < ADJOINT_TOL, f"<G_D f, g> - <f, G_D^H g> is {defect:.3e}"


@pytest.mark.parametrize("structure", [None, casing()], ids=["plain", "structured"])
def test_data_adjoint_is_exact(structure):
    op = make_op(structure=structure)
    defect = adjoint_defect(op.data, op.data_adjoint,
                            (op.n_src, op.grid.n_pixels), (op.n_src, op.n_rx),
                            op.grid)
    assert defect < ADJOINT_TOL, f"<G_S f, d> - <f, G_S^H d> is {defect:.3e}"


def test_lossy_background_keeps_the_adjoints_exact():
    """A complex ``k_b`` is where a conjugate slip stops cancelling."""
    op = make_op(sigma_bg=0.02)
    assert op.k_b.imag != 0.0
    n = (op.n_src, op.grid.n_pixels)
    assert adjoint_defect(op.domain, op.domain_adjoint, n, n, op.grid) < ADJOINT_TOL
    assert adjoint_defect(op.data, op.data_adjoint, n, (op.n_src, op.n_rx),
                          op.grid) < ADJOINT_TOL


def test_shapes_and_bare_vector_handling():
    op = make_op()
    n_pix = op.grid.n_pixels
    assert op.incident().shape == (op.n_src, n_pix)
    assert op.incident_at_receivers().shape == (op.n_src, op.n_rx)
    f = np.zeros((op.n_src, n_pix), np.complex128)
    assert op.domain(f).shape == f.shape
    assert op.data(f).shape == (op.n_src, op.n_rx)
    assert op.domain(np.zeros(n_pix, np.complex128)).shape == (n_pix,)


# --- the absorbing boundary -------------------------------------------------


def test_pml_reflection_is_below_minus_40_db():
    """The field on the pixels must not know where the wall is.

    Same cell size, same PML depth, only the distance to the layer changes.
    Whatever the tightly-wrapped domain gets that the roomy one does not is
    reflection, so the difference bounds it from above.
    """
    grid = InversionGrid(x_min=0.0, x_max=0.5, y_min=0.0, y_max=0.5, nx=24, ny=24)
    omega = 2.0 * np.pi * FREQ
    ang = np.linspace(0.0, np.pi, 3)
    src = np.column_stack([0.25 + 0.2 * np.cos(ang), 0.25 + 0.2 * np.sin(ang)])
    rx = np.column_stack([0.25 - 0.2 * np.cos(ang), 0.25 - 0.2 * np.sin(ang)])

    reference = FiniteDifferenceOperator.from_background(
        grid, omega, EPS_BG, src, rx, n_pad=40).incident()
    for pad in (2, 4, 8):
        op = FiniteDifferenceOperator.from_background(grid, omega, EPS_BG, src, rx,
                                                      n_pad=pad)
        rel = np.linalg.norm(op.incident() - reference) / np.linalg.norm(reference)
        level = 20.0 * np.log10(rel)
        assert level < -40.0, f"pad={pad} reflects at {level:.1f} dB"


def test_a_thin_layer_reflects_more_than_a_thick_one():
    """Guards the test above against passing for the wrong reason: if the
    comparison were insensitive to the boundary it would also be insensitive
    to the layer depth."""
    grid = InversionGrid(x_min=0.0, x_max=0.5, y_min=0.0, y_max=0.5, nx=24, ny=24)
    omega = 2.0 * np.pi * FREQ
    src = np.array([[0.25, 0.25]])
    rx = np.array([[0.40, 0.25]])
    ref = FiniteDifferenceOperator.from_background(grid, omega, EPS_BG, src, rx,
                                                   n_pad=40).incident()

    def level(n_pml):
        op = FiniteDifferenceOperator.from_background(grid, omega, EPS_BG, src, rx,
                                                      n_pad=2, n_pml=n_pml)
        return 20.0 * np.log10(
            np.linalg.norm(op.incident() - ref) / np.linalg.norm(ref))

    assert level(3) > level(10) + 20.0


# --- against the integral-equation backend ----------------------------------


def test_matches_the_integral_operator_at_second_order():
    """The headline test: same grid, same ``k_b``, error falling as ``h**2``.

    The integral operator's incident field is the analytic Hankel, so the
    difference is the finite-difference error alone -- numerical dispersion
    accumulated over the source-receiver range. A first-order slope would mean
    the stencil or the source placement is one-sided; a zeroth-order floor
    would mean the two backends disagree on the physics rather than the mesh.

    The PML depth is scaled with the refinement so the boundary does not put a
    floor under the last rung. Measured slope on this machine: 1.91.
    """
    omega = 2.0 * np.pi * FREQ
    src = np.column_stack([np.full(3, -0.17), np.linspace(0.06, 0.44, 3)])
    rx = np.column_stack([np.full(4, 0.67), np.linspace(0.06, 0.44, 4)])

    hs, errs = [], []
    for n in (24, 32, 48, 64):
        grid = InversionGrid(x_min=0.0, x_max=0.5, y_min=0.0, y_max=0.5, nx=n, ny=n)
        fd = FiniteDifferenceOperator.from_background(
            grid, omega, EPS_BG, src, rx, n_pml=int(round(10 * n / 16)))
        ie = IntegralOperator.from_background(grid, omega, EPS_BG, src, rx)
        a, b = fd.incident_at_receivers(), ie.incident_at_receivers()
        hs.append(grid.dx)
        errs.append(np.linalg.norm(a - b) / np.linalg.norm(b))

    order = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
    assert errs[-1] < 1e-2, f"finest grid still off by {errs[-1]:.2e}"
    assert 1.8 <= order <= 2.2, f"observed convergence order {order:.3f}"


def test_radiation_operators_agree_with_the_integral_backend():
    """``G_D`` and ``G_S`` too, not only the incident field.

    Loose on purpose -- 20 points per wavelength is a working resolution, not a
    converged one -- but it is what would catch a missing ``k_b**2`` or a
    cell-area factor, which no adjoint test can see.
    """
    grid = InversionGrid(x_min=0.0, x_max=0.4, y_min=0.0, y_max=0.4, nx=40, ny=40)
    omega = 2.0 * np.pi * FREQ
    src = np.column_stack([np.full(3, -0.15), np.linspace(0.05, 0.35, 3)])
    rx = np.column_stack([np.full(4, 0.55), np.linspace(0.05, 0.35, 4)])
    fd = FiniteDifferenceOperator.from_background(grid, omega, EPS_BG, src, rx,
                                                  n_pml=25)
    ie = IntegralOperator.from_background(grid, omega, EPS_BG, src, rx)

    # A smooth blob: midpoint quadrature in the integral backend is only
    # second-order, so a white-noise contrast source compares the two
    # quadratures rather than the two Green's functions.
    c = grid.cell_centers()
    f = np.exp(-((c[:, 0] - 0.2) ** 2 + (c[:, 1] - 0.2) ** 2) / 0.01)[None, :]
    f = np.broadcast_to(f, (3, grid.n_pixels)).astype(np.complex128)

    for name, a, b in (("G_S", fd.data(f), ie.data(f)),
                       ("G_D", fd.domain(f), ie.domain(f))):
        rel = np.linalg.norm(a - b) / np.linalg.norm(b)
        assert rel < 0.05, f"{name} differs by {rel:.3e}"


# --- known structure --------------------------------------------------------


def test_structure_is_optional_and_changes_the_field():
    """``None`` is the default and the survey case; a cylinder must actually bite."""
    plain = make_op()
    assert plain.structure is None
    assert plain.matrix.shape[0] == plain.nx_t * plain.ny_t

    op = make_op(structure=casing())
    assert op.matrix.shape[0] < op.nx_t * op.ny_t, "PEC nodes were not removed"
    rel = (np.linalg.norm(op.incident_at_receivers() - plain.incident_at_receivers())
           / np.linalg.norm(plain.incident_at_receivers()))
    assert rel > 1e-3, f"the structure barely moved the data ({rel:.2e})"


def test_borehole_cylinders_requires_a_radius():
    """A borehole diameter is not in the data, so it cannot have a default."""
    with pytest.raises(TypeError):
        borehole_cylinders([[0.0, 0.0]])
    s = borehole_cylinders([[0.0, 0.0], [1.0, 0.0]], radius=0.05, eps=81.0)
    assert len(s.cylinders) == 2
    assert all(c.radius == 0.05 for c in s.cylinders)
    assert hash(s) == hash(borehole_cylinders([[0.0, 0.0], [1.0, 0.0]],
                                              radius=0.05, eps=81.0))


def test_rasterize_writes_the_expected_wavenumber():
    omega = 2.0 * np.pi * FREQ
    s = KnownStructure((Cylinder(x=0.0, y=0.0, radius=0.15, eps=81.0, sigma=0.01),))
    x = np.linspace(-0.5, 0.5, 21)
    k2, pec = s.rasterize(x, x, omega, complex(1.0))
    assert not pec.any()
    eps_c = 81.0 - 1j * 0.01 / (omega * epsilon_0)
    assert k2[10, 10] == pytest.approx(omega**2 * mu_0 * epsilon_0 * eps_c, rel=1e-12)
    assert k2[0, 0] == 1.0


# --- plumbing ---------------------------------------------------------------


def test_an_antenna_inside_the_pml_is_rejected():
    grid = InversionGrid(x_min=0.0, x_max=0.5, y_min=0.0, y_max=0.5, nx=12, ny=12)
    with pytest.raises(ValueError, match="PML"):
        FiniteDifferenceOperator.from_background(
            grid, 2.0 * np.pi * FREQ, EPS_BG,
            np.array([[-5.0, 0.25]]), np.array([[0.25, 0.25]]), n_pad=2)


def test_the_factorization_is_reused():
    """Two operators with the same frequency, background, structure and grid must
    share one LU -- the whole reason the backend caches at all."""
    clear_factor_cache()
    a = make_op()
    b = make_op()
    assert a._lu is b._lu
    assert make_op(freq=4e8)._lu is not a._lu


# --- the operator and the solver together -----------------------------------


LAMBDA_BG = 0.4
HALF = 0.2
DISC_RADIUS = 0.12


def crime_scene(n_grid=6, n_ant=19, xi_amp=6.0):
    """Ring of sources and interleaved receivers around a dielectric disc.

    Deliberately overdetermined: ``2 * n_src * n_rx / n_pixels`` is about 20
    real measurements per unknown. At a realistic crosshole aperture the same
    solver on the same operator returns a different xi that fits the data just
    as well, which is the aperture limit rather than an operator defect -- see
    ``test_underdetermined_xi_is_genuinely_non_unique`` in
    ``test_csi_integration.py``.
    """
    omega = 2.0 * np.pi * C_LIGHT / (LAMBDA_BG * np.sqrt(EPS_BG))
    grid = InversionGrid(x_min=-HALF, x_max=HALF, y_min=-HALF, y_max=HALF,
                         nx=n_grid, ny=n_grid)
    ang = np.linspace(0.0, 2.0 * np.pi, n_ant, endpoint=False)
    src = np.column_stack([0.26 * np.cos(ang), 0.26 * np.sin(ang)])
    off = ang + np.pi / n_ant
    rx = np.column_stack([0.24 * np.cos(off), 0.24 * np.sin(off)])
    op = FiniteDifferenceOperator.from_background(grid, omega, EPS_BG, src, rx,
                                                  n_pml=5, n_pad=3)
    c = grid.cell_centers()
    xi = np.where(np.hypot(c[:, 0], c[:, 1]) <= DISC_RADIUS, xi_amp, 0.0)
    return grid, op, omega, xi


def exact_forward(op, omega, xi):
    """Scattered field at the receivers for ``xi``, with no solver involved.

    ``G_D`` is small enough here to build column by column, so the
    Lippmann-Schwinger system is solved densely and the data is self-consistent
    to the operator rather than to any iterate of ``run_csi``.
    """
    n = op.grid.n_pixels
    chi = ContrastModel(grid=op.grid, eps_bg=EPS_BG, xi=xi).chi(omega)
    g_d = op.domain(np.eye(n, dtype=np.complex128)).T
    w = np.linalg.solve(np.eye(n) - chi[:, None] * g_d,
                        (chi[None, :] * op.incident()).T).T
    return op.data(w), w, chi


def test_synthesized_data_is_self_consistent():
    """The object equation holds exactly for the w the data was built from."""
    _, op, omega, xi_true = crime_scene()
    _, w, chi = exact_forward(op, omega, xi_true)
    u = op.incident() + op.domain(w)
    assert np.linalg.norm(chi * u - w) / np.linalg.norm(w) < 1e-11


def test_inverse_crime_with_no_structure():
    """Known xi, same operator, ``structure=None``, no regularization.

    Measured: F_S reaches 2e-10 and xi comes back to 2.5e-04 relative in 800
    iterations at 20 real measurements per unknown.
    """
    grid, op, omega, xi_true = crime_scene()
    assert op.structure is None
    ratio = 2 * op.n_src * op.n_rx / grid.n_pixels
    assert ratio > 20.0, f"only {ratio:.1f} measurements per unknown"

    f_meas, _, _ = exact_forward(op, omega, xi_true)
    state = run_csi([op], [f_meas], eps_bg=EPS_BG, n_iter=800, f_s_tol=1e-16,
                    heldout_frac=0.0)

    rel = np.linalg.norm(state.xi - xi_true) / np.linalg.norm(xi_true)
    assert state.f_s[-1] < 1e-8, f"F_S stalled at {state.f_s[-1]:.2e}"
    assert rel < 1e-3, f"xi relative error {rel:.2e}"
