"""CSI solver algebra, tested against a dense random operator.

The operator here is not physical: ``G_D`` and ``G_S`` are dense random matrices whose
adjoints are exact conjugate transposes under the cell-area inner product. That is
deliberate. It isolates the CG step, the exact line search and the closed-form
contrast update from any Green's-function bug, and a random dense operator mixes every
pixel into every datum, which stresses the line search harder than a physical kernel
whose singular values decay.
"""
import numpy as np
import pytest
from scipy.constants import c as C_LIGHT

from Inversion.CSI import EPS_MIN, EPS_WATER_MAX
from Inversion.CSI.contrast import ContrastModel, update_contrast
from Inversion.CSI.csi import heldout_pairs, run_csi
from Inversion.CSI.grid import InversionGrid
from Inversion.CSI.operators import GreenOperator, adjoint_defect, inner


class DenseOperator:
    """A ``GreenOperator`` whose ``G_D``/``G_S`` are dense random matrices.

    Adjoints are conjugate transposes, which is exactly right here: the cell-area
    weight appears on both sides of ``<G a, b> == <a, G^H b>`` and cancels.
    ``domain``/``data`` act on ``[n_src, n_pix]`` as ``f @ G.T``, so batching over
    sources is a single matmul.
    """

    def __init__(self, grid, freq, eps_bg, src_pos, rx_pos, seed=0, scatter=0.3):
        self.grid = grid
        self.omega = 2.0 * np.pi * freq
        self.k_b = complex(self.omega / C_LIGHT * np.sqrt(eps_bg))
        self.n_src = len(src_pos)
        self.n_rx = len(rx_pos)
        n = grid.n_pixels

        rng = np.random.default_rng(seed)
        gd = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
        # Tame the spectral radius so I - diag(chi) G_D stays well conditioned at the
        # contrasts these tests use; the exact w is obtained by a direct solve.
        gd *= scatter / np.max(np.abs(np.linalg.eigvals(gd)))
        self.gd = gd
        self.gs = ((rng.standard_normal((self.n_rx, n))
                    + 1j * rng.standard_normal((self.n_rx, n))) / np.sqrt(2.0 * n))

        cen = grid.cell_centers()
        self._u_inc = self._greens(np.asarray(src_pos), cen)
        self._u_inc_rx = self._greens(np.asarray(src_pos), np.asarray(rx_pos))

    def _greens(self, a, b):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return np.exp(-1j * self.k_b * d) / np.sqrt(d)

    def incident(self):
        return self._u_inc

    def incident_at_receivers(self):
        return self._u_inc_rx

    def domain(self, f):
        return np.asarray(f) @ self.gd.T

    def domain_adjoint(self, g):
        return np.asarray(g) @ np.conj(self.gd)

    def data(self, f):
        return np.asarray(f) @ self.gs.T

    def data_adjoint(self, d):
        return np.asarray(d) @ np.conj(self.gs)


def toy_scene(nx=6, ny=6, n_src=4, n_rx=9, freqs=(1.5e8, 2.5e8), eps_bg=6.0, seed=3):
    """A 0.6 m square panel with sources left, receivers right, two low rungs."""
    grid = InversionGrid(x_min=0.0, x_max=0.6, y_min=0.0, y_max=0.6, nx=nx, ny=ny)
    src = np.column_stack([np.full(n_src, -0.05), np.linspace(0.05, 0.55, n_src)])
    rx = np.column_stack([np.full(n_rx, 0.65), np.linspace(0.05, 0.55, n_rx)])
    ops = [DenseOperator(grid, f, eps_bg, src, rx, seed=seed + i)
           for i, f in enumerate(freqs)]
    return grid, ops, eps_bg


def block_xi(grid, value=6.0, rows=(2, 3), cols=(2, 3)):
    """A square anomaly, row-major flat."""
    img = np.zeros(grid.shape)
    img[np.ix_(rows, cols)] = value
    return img.ravel()


def add_noise(f_meas, frac=0.05, seed=11):
    """Circular complex Gaussian at ``frac`` of the per-frequency RMS."""
    rng = np.random.default_rng(seed)
    out = []
    for d in f_meas:
        rms = np.sqrt(np.mean(np.abs(d) ** 2))
        out.append(d + frac * rms / np.sqrt(2.0)
                   * (rng.standard_normal(d.shape) + 1j * rng.standard_normal(d.shape)))
    return out


def synthesize(ops, model):
    """``f_meas`` from the exact ``w`` solving ``r = chi*u - w = 0``.

    Anything less -- a Born approximation, a truncated fixed point -- leaves a residual
    the solver would have to explain with the wrong xi, and the inverse-crime test
    would then be measuring the synthesis error rather than the solver.
    """
    f_meas, w_exact = [], []
    for op in ops:
        chi = model.chi(op.omega)
        u_inc = op.incident()
        a = np.eye(op.grid.n_pixels) - chi[:, None] * op.gd
        w = np.linalg.solve(a, (chi * u_inc).T).T
        u = u_inc + op.domain(w)
        assert np.allclose(chi * u - w, 0.0, atol=1e-12 * np.abs(w).max())
        f_meas.append(op.data(w))
        w_exact.append(w)
    return f_meas, w_exact


# --- operator sanity ---------------------------------------------------------------

def test_toy_operator_satisfies_the_protocol_and_its_adjoints():
    grid, ops, _ = toy_scene()
    op = ops[0]
    assert isinstance(op, GreenOperator)
    shp_pix = (op.n_src, grid.n_pixels)
    shp_rx = (op.n_src, op.n_rx)
    assert adjoint_defect(op.domain, op.domain_adjoint, shp_pix, shp_pix, grid) < 1e-13
    assert adjoint_defect(op.data, op.data_adjoint, shp_pix, shp_rx, grid) < 1e-13


# --- contrast parameterization -----------------------------------------------------

def test_chi_is_linear_in_xi_with_a_known_coefficient():
    grid, ops, eps_bg = toy_scene()
    xi = block_xi(grid, 4.0)
    m = ContrastModel(grid=grid, eps_bg=eps_bg, xi=xi)
    om = ops[0].omega
    assert m.a(om) == pytest.approx(1.0 / eps_bg)
    np.testing.assert_allclose(m.chi(om), xi / eps_bg)
    # Lossless background: chi is the same at every frequency, which is the property
    # that lets one xi serve a whole band group.
    np.testing.assert_allclose(m.chi(ops[1].omega), m.chi(om))


def test_lossy_background_makes_chi_frequency_dependent():
    grid, ops, eps_bg = toy_scene()
    m = ContrastModel(grid=grid, eps_bg=eps_bg, sigma_bg=0.01,
                      xi=block_xi(grid, 4.0), zeta=block_xi(grid, 0.005))
    lo, hi = m.chi(ops[0].omega), m.chi(ops[1].omega)
    assert not np.allclose(lo, hi)
    assert np.iscomplexobj(lo)


def test_bounds_project_to_the_physical_box():
    grid, _, eps_bg = toy_scene()
    xi = np.full(grid.n_pixels, 500.0)
    xi[:5] = -500.0
    m = ContrastModel(grid=grid, eps_bg=eps_bg, sigma_bg=0.01,
                      xi=xi, zeta=np.full(grid.n_pixels, -1.0)).project()
    assert m.eps.max() == pytest.approx(EPS_WATER_MAX)
    assert m.eps.min() == pytest.approx(EPS_MIN)
    assert m.sigma.min() >= 0.0


def test_closed_form_contrast_update_is_exact_given_exact_fields():
    grid, ops, eps_bg = toy_scene()
    truth = ContrastModel(grid=grid, eps_bg=eps_bg, xi=block_xi(grid, 6.0))
    _, w = synthesize(ops, truth)
    u = [op.incident() + op.domain(wf) for op, wf in zip(ops, w)]
    start = ContrastModel(grid=grid, eps_bg=eps_bg)
    got = update_contrast(start, [op.omega for op in ops], u, w, [1.0, 1.0])
    np.testing.assert_allclose(got.xi, truth.xi, atol=1e-9)


def test_two_unknown_update_is_an_unimplemented_extension_point():
    grid, ops, eps_bg = toy_scene()
    m = ContrastModel(grid=grid, eps_bg=eps_bg)
    with pytest.raises(NotImplementedError):
        update_contrast(m, [ops[0].omega], [ops[0].incident()],
                        [ops[0].incident()], [1.0], solve_sigma=True)


# --- the solver --------------------------------------------------------------------

def test_inverse_crime_recovers_xi():
    """The milestone gate: same operator forward and inverse, so the only error is
    the solver's own."""
    grid, ops, eps_bg = toy_scene()
    truth = ContrastModel(grid=grid, eps_bg=eps_bg, xi=block_xi(grid, 6.0))
    f_meas, _ = synthesize(ops, truth)

    state = run_csi(ops, f_meas, eps_bg=eps_bg, n_iter=4000,
                    f_s_tol=0.0, df_tol=0.0, heldout_frac=0.0)

    rel = np.linalg.norm(state.xi - truth.xi) / np.linalg.norm(truth.xi)
    assert rel < 1e-6, f"relative xi error {rel:.3e}, F_S = {state.f_s[-1]:.3e}"
    assert state.f_s[-1] < 1e-10


def test_descent_identity_b_equals_gradient_norm():
    """First iteration has gamma_PR = 0, so v == g and B must be ||g||**2 -- real,
    positive, and therefore a descent step."""
    grid, ops, eps_bg = toy_scene()
    truth = ContrastModel(grid=grid, eps_bg=eps_bg, xi=block_xi(grid, 6.0))
    f_meas, _ = synthesize(ops, truth)

    state = run_csi(ops, f_meas, eps_bg=eps_bg, n_iter=1, f_s_tol=0.0, df_tol=0.0)

    b, gg = state.b_history[0], state.grad_norm2_history[0]
    assert gg > 0.0
    assert b.real == pytest.approx(gg, rel=1e-10)
    assert abs(b.imag) < 1e-10 * abs(b.real)


def test_cost_is_monotone_non_increasing():
    """What the exact line search guarantees is that ``F_S + F_D`` never rises.

    ``F_S`` alone is not guaranteed and is not observed to be: the step minimizes the
    sum, so the data term is free to give a little back while the object term pays
    more. Measured here, 19 of 300 iterations raise ``F_S``, worst case +14% relative,
    all of them below ``F_S = 3e-4``. Every 5th sample is monotone, and the run still
    ends 1e-10 down. Do not "fix" this by damping the line search.
    """
    grid, ops, eps_bg = toy_scene()
    truth = ContrastModel(grid=grid, eps_bg=eps_bg, xi=block_xi(grid, 6.0))
    f_meas, _ = synthesize(ops, truth)

    state = run_csi(ops, f_meas, eps_bg=eps_bg, n_iter=300, f_s_tol=0.0, df_tol=0.0,
                    heldout_frac=0.0)

    f_s = np.asarray(state.f_s)
    total = f_s + np.asarray(state.f_d)
    assert np.all(np.diff(total) <= 0.0), f"cost rose by {np.diff(total).max():.3e}"
    assert np.all(np.diff(f_s[::5]) <= 0.0)
    assert f_s[-1] < 1e-8 * f_s[0]


def test_held_out_residual_is_recorded_and_finite():
    grid, ops, eps_bg = toy_scene()
    truth = ContrastModel(grid=grid, eps_bg=eps_bg, xi=block_xi(grid, 6.0))
    f_meas, _ = synthesize(ops, truth)

    state = run_csi(ops, f_meas, eps_bg=eps_bg, n_iter=300, f_s_tol=0.0, df_tol=0.0,
                    heldout_frac=0.2, heldout_seed=1)

    assert len(state.heldout) == len(state.f_s)
    assert np.all(np.isfinite(state.heldout))
    # Pairs the inversion never saw are still explained: this is the field-available
    # quality signal, and on an inverse crime it has to fall like F_S.
    assert state.heldout[-1] < 1e-5 * state.heldout[0]
    assert (~state.heldout_mask).sum() == round(0.2 * ops[0].n_src * ops[0].n_rx)


def test_held_out_pairs_are_excluded_from_the_data_residual():
    grid, ops, eps_bg = toy_scene()
    mask = heldout_pairs(ops[0].n_src, ops[0].n_rx, 0.15, seed=0)
    assert mask.dtype == bool and not mask.all()
    truth = ContrastModel(grid=grid, eps_bg=eps_bg, xi=block_xi(grid, 6.0))
    f_meas, _ = synthesize(ops, truth)

    # Corrupting only the withheld entries must not move the reconstruction.
    spoilt = [d.copy() for d in f_meas]
    for d in spoilt:
        d[~mask] += 1e3
    clean = run_csi(ops, f_meas, eps_bg=eps_bg, n_iter=40, f_s_tol=0.0, df_tol=0.0,
                    heldout_mask=mask)
    dirty = run_csi(ops, spoilt, eps_bg=eps_bg, n_iter=40, f_s_tol=0.0, df_tol=0.0,
                    heldout_mask=mask)
    np.testing.assert_allclose(dirty.xi, clean.xi, rtol=1e-8, atol=1e-10)
    assert dirty.heldout[-1] > clean.heldout[-1]


def test_noise_robustness():
    """5% noise: peak contrast within 20%, centroid within a quarter wavelength.

    The wavelength is the operator's own, ``2*pi/Re(k_b)`` at the lowest frequency in
    the group -- the toy scene has no other length scale that means anything.
    """
    grid, ops, eps_bg = toy_scene()
    truth = ContrastModel(grid=grid, eps_bg=eps_bg, xi=block_xi(grid, 6.0))
    f_meas, _ = synthesize(ops, truth)

    state = run_csi(ops, add_noise(f_meas), eps_bg=eps_bg, n_iter=200,
                    heldout_frac=0.15)

    assert state.eps.min() >= EPS_MIN - 1e-12
    assert state.eps.max() <= EPS_WATER_MAX + 1e-12

    peak = state.xi.max()
    assert abs(peak - truth.xi.max()) / truth.xi.max() < 0.20, f"peak xi {peak:.3f}"

    cen = grid.cell_centers()
    def centroid(x):
        wgt = np.maximum(x, 0.0)
        return (wgt @ cen) / wgt.sum()
    lam = 2.0 * np.pi / ops[0].k_b.real
    err = np.linalg.norm(centroid(state.xi) - centroid(truth.xi))
    assert err < 0.25 * lam, f"centroid off by {err:.3f} m, lambda/4 = {0.25 * lam:.3f}"


def test_warm_start_from_xi_beats_a_cold_start():
    """Continuation hands ``xi`` across a rung, never ``w`` -- which is
    frequency-specific and starts again from backpropagation. So a warm start does not
    begin at zero misfit; it begins ahead, and has to stay ahead."""
    grid, ops, eps_bg = toy_scene()
    truth = ContrastModel(grid=grid, eps_bg=eps_bg, xi=block_xi(grid, 6.0))
    f_meas, _ = synthesize(ops, truth)

    kw = dict(eps_bg=eps_bg, n_iter=30, f_s_tol=0.0, df_tol=0.0, heldout_frac=0.0)
    cold = run_csi(ops, f_meas, **kw)
    warm = run_csi(ops, f_meas, xi0=truth.xi, **kw)

    assert warm.f_s[-1] < cold.f_s[-1]
    err = np.linalg.norm(warm.xi - truth.xi) / np.linalg.norm(truth.xi)
    assert err < np.linalg.norm(cold.xi - truth.xi) / np.linalg.norm(truth.xi)


def test_state_carries_per_frequency_chi_and_w():
    grid, ops, eps_bg = toy_scene()
    truth = ContrastModel(grid=grid, eps_bg=eps_bg, xi=block_xi(grid, 6.0))
    f_meas, _ = synthesize(ops, truth)

    state = run_csi(ops, f_meas, eps_bg=eps_bg, n_iter=5, f_s_tol=0.0, df_tol=0.0)

    assert len(state.chi) == len(state.w) == len(ops)
    for op, chi, w in zip(ops, state.chi, state.w):
        assert chi.shape == (grid.n_pixels,) and chi.dtype == np.complex128
        assert w.shape == (op.n_src, grid.n_pixels) and w.dtype == np.complex128
        np.testing.assert_allclose(chi, state.model.chi(op.omega))
    assert state.xi.dtype == np.float64


def test_stopping_rules_fire():
    grid, ops, eps_bg = toy_scene()
    truth = ContrastModel(grid=grid, eps_bg=eps_bg, xi=block_xi(grid, 6.0))
    f_meas, _ = synthesize(ops, truth)

    capped = run_csi(ops, f_meas, eps_bg=eps_bg, n_iter=7, f_s_tol=0.0, df_tol=0.0)
    assert capped.stop_reason == "max_iter" and capped.n_iter == 7

    loose = run_csi(ops, f_meas, eps_bg=eps_bg, n_iter=4000, f_s_tol=1e-4, df_tol=0.0)
    assert loose.stop_reason == "f_s_tol" and loose.f_s[-1] < 1e-4

    # On an inverse crime the cost falls geometrically for ever, so the stall rule
    # only has anything to bite on once there is noise it cannot fit.
    stalled = run_csi(ops, add_noise(f_meas), eps_bg=eps_bg, n_iter=4000,
                      f_s_tol=0.0, df_tol=1e-6, df_patience=5)
    assert stalled.stop_reason == "df_tol" and stalled.n_iter < 4000


def test_rejects_mismatched_inputs():
    grid, ops, eps_bg = toy_scene()
    truth = ContrastModel(grid=grid, eps_bg=eps_bg, xi=block_xi(grid, 6.0))
    f_meas, _ = synthesize(ops, truth)

    with pytest.raises(ValueError, match="at least one operator"):
        run_csi([], [], eps_bg=eps_bg)
    with pytest.raises(ValueError, match="measured fields"):
        run_csi(ops, f_meas[:1], eps_bg=eps_bg)
    with pytest.raises(ValueError, match="expected"):
        run_csi(ops, [f_meas[0][:, :3], f_meas[1]], eps_bg=eps_bg)


def test_inner_product_weight_matches_the_operator_convention():
    """The solver's norms use the same cell-area weight the adjoints are defined
    under, so alpha is scale-free in the grid."""
    grid, ops, eps_bg = toy_scene()
    a = np.ones((2, grid.n_pixels), dtype=np.complex128)
    assert inner(a, a, grid).real == pytest.approx(2 * grid.n_pixels * grid.cell_area)
