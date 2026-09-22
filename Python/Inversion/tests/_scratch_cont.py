"""Scratch: does continuation beat a cold start at the top rung? Not a test."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Python"))

import numpy as np
from scipy.constants import c as C_LIGHT

from Inversion.CSI.contrast import ContrastModel
from Inversion.CSI.continuation import frequency_ladder, run_continuation
from Inversion.CSI.csi import heldout_pairs, run_csi
from Inversion.CSI.grid import InversionGrid
from Inversion.CSI.operators.integral import IntegralOperator

EPS_BG = 6.0


def scene(L=0.5, n=24, n_ant=10, xi_amp=14.0, radius=0.08, geom="ring"):
    half = L / 2
    grid = InversionGrid(x_min=-half, x_max=half, y_min=-half, y_max=half, nx=n, ny=n)
    if geom == "ring":
        ang = np.linspace(0, 2 * np.pi, n_ant, endpoint=False)
        src = np.column_stack([0.62 * L * np.cos(ang), 0.62 * L * np.sin(ang)])
        off = ang + np.pi / n_ant
        rx = np.column_stack([0.62 * L * np.cos(off), 0.62 * L * np.sin(off)])
    else:
        y = np.linspace(-0.45 * L, 0.45 * L, n_ant)
        src = np.column_stack([np.full(n_ant, -0.62 * L), y])
        rx = np.column_stack([np.full(n_ant, 0.62 * L), y + 0.013])
    c = grid.cell_centers()
    xi = np.where(np.hypot(c[:, 0] - 0.06 * L, c[:, 1] + 0.05 * L) <= radius, xi_amp, 0.0)
    return grid, src, rx, xi


def synth(op, xi):
    model = ContrastModel(grid=op.grid, eps_bg=EPS_BG, xi=xi)
    chi = model.chi(op.omega)
    n = op.grid.n_pixels
    gd = op.domain(np.eye(n, dtype=complex)).T          # columns G_D e_i
    a = np.eye(n) - chi[:, None] * gd
    u_inc = op.incident()
    w = np.linalg.solve(a, (chi * u_inc).T).T
    assert np.allclose(chi * (u_inc + op.domain(w)) - w, 0, atol=1e-9 * np.abs(w).max())
    return op.data(w)


def run(L=0.5, n=24, n_ant=10, xi_amp=14.0, radius=0.08, geom="ring",
        ratios=(1, 2, 4), n_per_group=2, it=40, noise=0.0, seed=5):
    grid, src, rx, xi_true = scene(L, n, n_ant, xi_amp, radius, geom)
    f1 = C_LIGHT / (L * np.sqrt(EPS_BG))
    freqs = np.arange(1, 41) * (f1 / 4)
    band = np.ones(freqs.shape, bool)
    lad = frequency_ladder(L, EPS_BG, freqs, band, ratios=ratios, n_per_group=n_per_group)
    make = lambda f: IntegralOperator.from_background(
        grid, 2 * np.pi * f, EPS_BG, src, rx)

    idx = lad.indices
    s21 = np.zeros((len(src), len(rx), freqs.size), complex)
    for i in idx:
        s21[:, :, i] = synth(make(freqs[i]), xi_true)
    if noise:
        rng = np.random.default_rng(seed)
        for i in idx:
            d = s21[:, :, i]
            rms = np.sqrt(np.mean(np.abs(d) ** 2))
            s21[:, :, i] = d + noise * rms / np.sqrt(2) * (
                rng.standard_normal(d.shape) + 1j * rng.standard_normal(d.shape))

    mask = heldout_pairs(len(src), len(rx), 0.1, 0)
    t0 = time.time()
    res = run_continuation(lad, s21, make, EPS_BG, heldout_mask=mask, n_iter=it)
    t1 = time.time()

    top = lad.groups[-1]
    ops = [make(float(f)) for f in top.freqs]
    fm = [s21[:, :, int(i)] for i in top.indices]
    cold = run_csi(ops, fm, EPS_BG, heldout_mask=mask, n_iter=it * len(lad.groups))
    t2 = time.time()

    def err(xi):
        return np.linalg.norm(xi - xi_true) / np.linalg.norm(xi_true)

    print(f"geom={geom} L={L} n={n} ant={n_ant} xi={xi_amp} ratios={lad.ratios} "
          f"npg={n_per_group} it={it} noise={noise}")
    print(f"  ladder err {err(res.xi):.4f}  cold err {err(cold.xi):.4f}  "
          f"iters {res.n_iter}/{cold.n_iter}  t {t1-t0:.1f}s/{t2-t1:.1f}s")
    print(f"  per-group err {[round(err(s.xi), 4) for s in res.states]}  "
          f"F_S {[f'{s.f_s[-1]:.2e}' for s in res.states]}  cold F_S {cold.f_s[-1]:.2e}")
    print(f"  heldout ladder {res.states[-1].heldout[-1]:.3e} cold {cold.heldout[-1]:.3e}")
    return res, cold


if __name__ == "__main__":
    run()
    run(geom="cross")
    run(xi_amp=30.0)
    run(xi_amp=30.0, geom="cross")
