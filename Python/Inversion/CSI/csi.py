"""CSI solver: alternating contrast-source and contrast updates over a band group.

``run_csi`` takes one ``GreenOperator`` per frequency plus the measured scattered
field ``[n_src, n_rx]`` at each, and returns a ``CSIState`` carrying ``xi``, the
per-frequency contrast source ``w`` and ``chi``, the ``F_S``/``F_D`` histories and the
held-out-pair residual history.

The total field is updated algebraically along the search direction rather
than by a forward solve, so an iteration costs exactly four operator applications and
the scattering stays exact at any contrast.

Risk: the cost is non-convex in ``(w, xi)`` jointly. A cold start at short wavelength
settles into a local minimum that looks converged -- ``F_S`` falls, then stalls -- so
the held-out residual, not ``F_S``, is the signal to trust.

Weights are per frequency and averaged over the group, so ``F_S == 1`` at ``w = 0``
whatever the group holds and the stopping thresholds keep one meaning. Both the data
and the domain inner product carry ``grid.cell_area``, matching the weight the
operator adjoints are defined under.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .contrast import ContrastModel, update_contrast
from .grid import InversionGrid
from .operators import GreenOperator, inner
from .regularization import update_contrast_tv


def _norm2(a: np.ndarray, grid: InversionGrid) -> float:
    return float(inner(a, a, grid).real)


@dataclass
class CSIState:
    """Result of one band group. ``w`` and ``chi`` are per frequency, ``xi`` is shared."""

    model: ContrastModel
    w: list[np.ndarray]
    chi: list[np.ndarray]
    f_s: list[float]
    f_d: list[float]
    heldout: list[float]
    heldout_mask: np.ndarray
    b_history: list[complex] = field(default_factory=list)
    grad_norm2_history: list[float] = field(default_factory=list)
    n_iter: int = 0
    stop_reason: str = ""

    @property
    def xi(self) -> np.ndarray:
        return self.model.xi

    @property
    def zeta(self) -> np.ndarray | None:
        return self.model.zeta

    @property
    def eps(self) -> np.ndarray:
        """Absolute eps' per pixel; ``model.grid.reshape`` for an image."""
        return self.model.eps


def heldout_pairs(n_src: int, n_rx: int, frac: float, seed: int = 0) -> np.ndarray:
    """Boolean ``[n_src, n_rx]``, True where the pair is used by the inversion.

    Pairs are drawn without replacement and shared across every frequency, because a
    withheld pair is a physical pair. No attempt is made to keep every source fed --
    at the fractions this is used at (0.1-0.2) that does not happen, and forcing it
    would bias which pairs are testable.
    """
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"heldout_frac must be in [0, 1), got {frac}")
    mask = np.ones((n_src, n_rx), dtype=bool)
    n_hold = int(round(frac * n_src * n_rx))
    if n_hold > 0:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n_src * n_rx)[:n_hold]
        mask.reshape(-1)[idx] = False
    return mask


def _eta_data(f_meas: Sequence[np.ndarray], mask: np.ndarray,
              grid: InversionGrid) -> np.ndarray:
    n_f = len(f_meas)
    eta = np.empty(n_f)
    for i, d in enumerate(f_meas):
        norm = _norm2(mask * d, grid)
        if norm <= 0.0:
            raise ValueError(f"frequency {i} has no measured energy on the used pairs")
        eta[i] = 1.0 / (n_f * norm)
    return eta


def _eta_domain(chi: Sequence[np.ndarray], u_inc: Sequence[np.ndarray],
                grid: InversionGrid) -> np.ndarray:
    """``1 / SUM ||chi * u_inc||**2``, frozen at the chi handed in.

    Falls back to the incident-field norm on the first iteration of a cold start,
    where chi is still identically zero and the nominal weight is infinite.
    """
    n_f = len(chi)
    eta = np.empty(n_f)
    for i, (c, ui) in enumerate(zip(chi, u_inc)):
        norm = _norm2(c * ui, grid)
        if norm <= 0.0:
            norm = _norm2(ui, grid)
        eta[i] = 1.0 / (n_f * norm)
    return eta


def _backpropagate(op: GreenOperator, d: np.ndarray, mask: np.ndarray):
    """Start ``w`` at the scaled adjoint of the data; returns ``(w, G_S w)``.

    ``gamma`` is the exact minimizer of ``||d - gamma*G_S G_S^H d||``, so the initial
    data residual is already the best a single adjoint step can do.
    """
    b = op.data_adjoint(mask * d)
    gb = op.data(b)
    denom = _norm2(mask * gb, op.grid)
    gamma = 0.0 if denom == 0.0 else inner(mask * d, mask * gb, op.grid) / denom
    return gamma * b, gamma * gb


def run_csi(operators: Sequence[GreenOperator], f_meas: Sequence[np.ndarray],
            eps_bg: float, sigma_bg: float = 0.0,
            xi0: np.ndarray | None = None, zeta0: np.ndarray | None = None,
            n_iter: int = 256, f_s_tol: float = 1e-4, df_tol: float = 1e-6,
            df_patience: int = 5, heldout_frac: float = 0.15,
            heldout_seed: int = 0, heldout_mask: np.ndarray | None = None,
            tv: bool = False, tv_cg_iter: int = 15) -> CSIState:
    """Contrast Source Inversion over one band group.

    ``operators[i]`` and ``f_meas[i]`` describe the same frequency; ``xi`` is shared
    across them. ``xi0`` warm-starts from a coarser rung -- pass ``xi``, never ``w``,
    which is frequency-specific.

    Per outer iteration, in this order: freeze ``eta_D`` at the current ``chi``, form
    the residuals, take one Polak-Ribiere CG step on ``w`` with an exact line search,
    then update ``xi`` in closed form. Four operator applications, no more::

        g     = eta_S*G_S^H rho + eta_D*( r - G_D^H( conj(chi)*r ) )
        v     = g + gamma_PR * v_prev
        p     = G_S v ;  q = chi*(G_D v) - v
        A     = eta_S*SUM||p||**2 + eta_D*SUM||q||**2
        B     = eta_S*SUM<p,rho>  - eta_D*SUM<q,r>
        alpha = conj(B) / A

    ``B == <v, g>``, so on the first iteration -- and after any restart -- ``B`` is
    ``||g||**2`` and the step descends. The conjugate on ``alpha`` follows from
    ``inner`` conjugating its second argument.

    ``tv`` swaps the closed-form contrast update for the multiplicative
    total-variation step of ``regularization.update_contrast_tv``. The ``w`` step is
    untouched: at fixed ``xi`` the TV factor is 1 and its ``w``-derivative vanishes,
    which is what makes the multiplicative form cost one sparse solve per iteration.
    Off by default, and off it runs the identical arithmetic it always did -- an
    aperture-limited fit is non-unique on purpose, and only a caller asking for TV
    should have that broken.

    Stops on ``F_S < f_s_tol``, on ``|dF|/F < df_tol`` sustained ``df_patience``
    iterations, or at ``n_iter``.
    """
    ops = list(operators)
    if not ops:
        raise ValueError("need at least one operator")
    n_f = len(ops)
    if len(f_meas) != n_f:
        raise ValueError(f"{n_f} operators but {len(f_meas)} measured fields")

    grid = ops[0].grid
    n_src, n_rx = ops[0].n_src, ops[0].n_rx
    data = []
    for i, (op, d) in enumerate(zip(ops, f_meas)):
        # InversionGrid is a frozen dataclass, so compare by value: operators for
        # different frequencies are built separately and need not be the same object.
        if (op.n_src, op.n_rx) != (n_src, n_rx) or op.grid != grid:
            raise ValueError(f"operator {i} disagrees with operator 0 on grid or shape")
        d = np.asarray(d, dtype=np.complex128)
        if d.shape != (n_src, n_rx):
            raise ValueError(f"f_meas[{i}] is {d.shape}, expected {(n_src, n_rx)}")
        data.append(d)

    mask = (heldout_pairs(n_src, n_rx, heldout_frac, heldout_seed)
            if heldout_mask is None else np.asarray(heldout_mask, bool))
    if mask.shape != (n_src, n_rx):
        raise ValueError(f"heldout_mask is {mask.shape}, expected {(n_src, n_rx)}")
    held = ~mask
    held_energy = sum(float(np.sum(np.abs(d[held]) ** 2)) for d in data)

    omegas = [op.omega for op in ops]
    model = ContrastModel(grid=grid, eps_bg=eps_bg, sigma_bg=sigma_bg,
                          xi=xi0, zeta=zeta0).project()
    chi = [model.chi(om) for om in omegas]

    u_inc = [op.incident() for op in ops]
    eta_s = _eta_data(data, mask, grid)

    # Setup: three operator applications per frequency, none of them repeated in the
    # loop. G_S w falls out of the backpropagation, so rho starts without a fourth.
    w, rho_full, u = [], [], []
    for op, d, ui in zip(ops, data, u_inc):
        w_f, gsw = _backpropagate(op, d, mask)
        w.append(w_f)
        rho_full.append(d - gsw)
        u.append(ui + op.domain(w_f))

    if xi0 is None:
        # Cold start: one contrast update off the backpropagated w, so eta_D has a
        # non-degenerate chi to freeze at on iteration 0.
        model = update_contrast(model, omegas, u, w, _eta_domain(chi, u_inc, grid))
        chi = [model.chi(om) for om in omegas]

    state = CSIState(model=model, w=w, chi=chi, f_s=[], f_d=[], heldout=[],
                     heldout_mask=mask)
    g_prev: list[np.ndarray] | None = None
    v: list[np.ndarray] | None = None
    gg_prev = 0.0
    f_prev = None
    stalled = 0

    for it in range(n_iter):
        eta_d = _eta_domain(chi, u_inc, grid)
        rho = [mask * rf for rf in rho_full]
        r = [c * uf - wf for c, uf, wf in zip(chi, u, w)]

        g = [es * op.data_adjoint(rf)
             + ed * (rf_obj - op.domain_adjoint(np.conj(c) * rf_obj))
             for op, es, ed, rf, rf_obj, c
             in zip(ops, eta_s, eta_d, rho, r, chi)]
        gg = sum(_norm2(gf, grid) for gf in g)
        state.grad_norm2_history.append(gg)
        if gg == 0.0:
            state.stop_reason = "zero_gradient"
            break

        if g_prev is None or gg_prev == 0.0:
            gamma_pr = 0.0
        else:
            gamma_pr = sum(inner(gf, gf - gp, grid).real
                           for gf, gp in zip(g, g_prev)) / gg_prev
        v = list(g) if v is None or gamma_pr == 0.0 else \
            [gf + gamma_pr * vf for gf, vf in zip(g, v)]

        def line_search(direction):
            """The two forward applications and the scalars they feed: ``p = G_S v``
            unmasked, ``G_D v``, then ``A`` and ``B``."""
            pf_full = [op.data(vf) for op, vf in zip(ops, direction)]
            gv = [op.domain(vf) for op, vf in zip(ops, direction)]
            pf = [mask * x for x in pf_full]
            qf = [c * x - vf for c, x, vf in zip(chi, gv, direction)]
            a = sum(es * _norm2(x, grid) + ed * _norm2(y, grid)
                    for es, ed, x, y in zip(eta_s, eta_d, pf, qf))
            b = sum(es * inner(x, ro_s, grid) - ed * inner(y, ro_d, grid)
                    for es, ed, x, ro_s, y, ro_d in zip(eta_s, eta_d, pf, rho, qf, r))
            return pf_full, gv, a, b

        p_full, gdv, a_num, b_num = line_search(v)
        if b_num.real <= 0.0 and gamma_pr != 0.0:
            # PR direction stopped descending; restart on steepest descent, where
            # B == ||g||**2 > 0 makes the step a descent step by construction.
            v = list(g)
            p_full, gdv, a_num, b_num = line_search(v)
        state.b_history.append(b_num)

        if a_num <= 0.0:
            state.stop_reason = "zero_curvature"
            break

        # The step sends rho -> rho - alpha*p and r -> r + alpha*q, so the cost along v
        # is const - 2*Re(alpha*B) + |alpha|**2*A, stationary at conj(B)/A. The
        # conjugate is not cosmetic: drop it and alpha carries the wrong phase, which
        # diverges within ~20 iterations.
        alpha = np.conj(b_num) / a_num
        w = [wf + alpha * vf for wf, vf in zip(w, v)]
        u = [uf + alpha * gf for uf, gf in zip(u, gdv)]
        rho_full = [rf - alpha * pf for rf, pf in zip(rho_full, p_full)]

        f_s = sum(es * _norm2(mask * rf, grid) for es, rf in zip(eta_s, rho_full))
        if tv:
            # F_D(w_n, xi_{n-1}): the new contrast source against the contrast the
            # TV weights were built from. It is both the regularization weight and
            # the edge threshold, so it is measured before xi moves.
            f_d_pre = sum(ed * _norm2(c * uf - wf, grid)
                          for ed, c, uf, wf in zip(eta_d, chi, u, w))
            model = update_contrast_tv(model, omegas, u, w, eta_d, f_d_pre,
                                       cg_iter=tv_cg_iter)
        else:
            model = update_contrast(model, omegas, u, w, eta_d)
        chi = [model.chi(om) for om in omegas]
        f_d = sum(ed * _norm2(c * uf - wf, grid)
                  for ed, c, uf, wf in zip(eta_d, chi, u, w))

        state.f_s.append(f_s)
        state.f_d.append(f_d)
        state.heldout.append(
            float(sum(np.sum(np.abs(rf[held]) ** 2) for rf in rho_full) / held_energy)
            if held_energy > 0.0 else float("nan"))
        g_prev, gg_prev = g, gg
        state.n_iter = it + 1

        total = f_s + f_d
        if f_s < f_s_tol:
            state.stop_reason = "f_s_tol"
            break
        if f_prev is not None and total > 0.0 and abs(total - f_prev) / total < df_tol:
            stalled += 1
            if stalled >= df_patience:
                state.stop_reason = "df_tol"
                break
        else:
            stalled = 0
        f_prev = total
    else:
        state.stop_reason = "max_iter"

    state.model, state.w, state.chi = model, w, chi
    return state


__all__ = ["CSIState", "run_csi", "heldout_pairs"]
