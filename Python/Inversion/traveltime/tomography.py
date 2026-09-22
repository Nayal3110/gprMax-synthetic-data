"""End-to-end crosshole travel-time tomography orchestration.

Pipeline: pick first-break times -> (optional) calibrate source-time offset t0
against a known-permittivity baseline -> build the straight-ray system matrix ->
solve for slowness with SIRT or SART -> convert to an eps_r image.
"""
from __future__ import annotations

import numpy as np

from Inversion.traveltime.geometry import PixelGrid, build_system_matrix
from Inversion.traveltime.picking import pick_traveltimes
from Inversion.traveltime.sirt import sirt, sart, eps_to_slowness, slowness_to_eps


def calibrate_t0(times: np.ndarray, acq, grid: PixelGrid, eps_known: float) -> float:
    """Estimate the constant source-time offset so picked times match a known eps_r.

    t0 = mean(picked - straight_ray_traveltime_at(eps_known)), over picked rays
    only — NaN entries (traces the picker rejected) are excluded.
    """
    L, _ = build_system_matrix(acq.sources, acq.receivers, grid)
    m = np.full(grid.n_pixels, eps_to_slowness(eps_known))
    predicted = (L @ m).reshape(acq.sources.shape[0], acq.receivers.shape[0])
    return float(np.nanmean(times - predicted))


def run_tomography(data, grid: PixelGrid, solver: str = "sirt", n_iter: int = 300,
                   relax: float = 0.2, m0=None, t0: float = 0.0,
                   calibrate_eps=None, eps_bounds=(1.0, 81.0),
                   picker_kwargs=None):
    """Run the full pipeline on an AcquisitionData. Returns a result dict."""
    picker_kwargs = picker_kwargs or {}
    acq = data.acq

    times = pick_traveltimes(data.traces, data.dt, **picker_kwargs)
    if not np.isfinite(times).any():
        raise ValueError("no usable first-break picks: every trace was rejected "
                         "by the picker (check min_snr / component / data)")
    if calibrate_eps is not None:
        t0 = calibrate_t0(times, acq, grid, eps_known=calibrate_eps)
    times = times - t0

    L, rays = build_system_matrix(acq.sources, acq.receivers, grid)
    t_vec = times.reshape(-1)                       # source-major, matches ray order

    # Traces with no detectable arrival pick as NaN; drop those rays entirely
    # rather than let a fabricated traveltime steer the solution.
    keep = np.isfinite(t_vec)
    n_dropped = int((~keep).sum())
    if n_dropped:
        L = L[keep]
        rays = [r for r, k in zip(rays, keep) if k]
        t_vec = t_vec[keep]

    if m0 is None:
        # homogeneous start from the median apparent slowness (t / straight distance)
        dist = np.array([np.hypot(*(acq.receivers[ri] - acq.sources[si]))
                         for si, ri in rays])
        app = t_vec / np.where(dist > 0, dist, np.nan)
        m0 = np.full(grid.n_pixels, np.nanmedian(app))
    else:
        m0 = np.asarray(m0, float)

    if solver == "sirt":
        m, hist = sirt(L, t_vec, m0, n_iter=n_iter, relax=relax, eps_bounds=eps_bounds)
    elif solver == "sart":
        # Group the surviving rays by source; derived from ``rays`` rather than
        # sliced arithmetically so dropped rays can't shift the block boundaries.
        by_src = {}
        for i, (si, _) in enumerate(rays):
            by_src.setdefault(si, []).append(i)
        blocks = [by_src[si] for si in sorted(by_src)]
        m, hist = sart(L, t_vec, m0, blocks, n_iter=n_iter, relax=relax,
                       eps_bounds=eps_bounds)
    else:
        raise ValueError(f"unknown solver {solver!r}")

    eps_map = slowness_to_eps(m).reshape(grid.ny, grid.nx)
    return {"eps_map": eps_map, "slowness": m, "times": times, "t0": t0,
            "residual": hist, "L": L, "rays": rays, "grid": grid,
            "n_dropped": n_dropped}


def plot_tomogram(eps_map, grid: PixelGrid, truth_circle=None, title="eps_r tomogram",
                  ax=None):
    """imshow the eps_r map in physical coordinates; optionally overlay a truth
    circle given as (xc, yc, radius)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(eps_map, origin="lower",
                   extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max],
                   aspect="equal", cmap="viridis")
    ax.figure.colorbar(im, ax=ax, label=r"$\varepsilon_r$")
    if truth_circle is not None:
        xc, yc, r = truth_circle
        ax.add_patch(plt.Circle((xc, yc), r, fill=False, color="red", lw=1.5))
    ax.set(xlabel="x [m]", ylabel="y [m]", title=title)
    return ax
