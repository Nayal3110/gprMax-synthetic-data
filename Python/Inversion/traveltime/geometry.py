"""2-D pixel grid and straight-ray geometry.

Pixels are flattened row-major as ``flat = iy * nx + ix`` so an image reshapes
as ``arr.reshape(ny, nx)`` with ix the column (x) and iy the row (y).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class PixelGrid:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    @property
    def n_pixels(self) -> int:
        return self.nx * self.ny

    @classmethod
    def covering(cls, points: np.ndarray, nx: int, ny: int, pad: float = 0.0) -> "PixelGrid":
        """Grid covering the bounding box of ``points`` ([N, 2]) plus ``pad`` margin."""
        p = np.asarray(points, float)
        return cls(
            x_min=p[:, 0].min() - pad, x_max=p[:, 0].max() + pad,
            y_min=p[:, 1].min() - pad, y_max=p[:, 1].max() + pad,
            nx=nx, ny=ny,
        )

    def cell_centers(self) -> np.ndarray:
        """[n_pixels, 2] centre coordinates, row-major (matches flat indexing)."""
        xs = self.x_min + (np.arange(self.nx) + 0.5) * self.dx
        ys = self.y_min + (np.arange(self.ny) + 0.5) * self.dy
        gx, gy = np.meshgrid(xs, ys)  # both [ny, nx]
        return np.column_stack([gx.ravel(), gy.ravel()])


def ray_cell_lengths(p1, p2, grid: PixelGrid):
    """Length of the straight segment p1->p2 within each grid cell (Siddon).

    Returns ``(flat_indices, lengths)`` for cells the ray actually crosses
    (cells outside the grid are dropped). Both arrays are 1-D, same length.
    """
    x0, y0 = float(p1[0]), float(p1[1])
    x1, y1 = float(p2[0]), float(p2[1])
    total = np.hypot(x1 - x0, y1 - y0)
    if total == 0.0:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=float)
 
    alphas = [0.0, 1.0]
    if x1 != x0:
        planes = grid.x_min + np.arange(grid.nx + 1) * grid.dx
        a = (planes - x0) / (x1 - x0)
        alphas.extend(a[(a > 0.0) & (a < 1.0)].tolist())
    if y1 != y0:
        planes = grid.y_min + np.arange(grid.ny + 1) * grid.dy
        a = (planes - y0) / (y1 - y0)
        alphas.extend(a[(a > 0.0) & (a < 1.0)].tolist())

    alphas = np.unique(np.asarray(alphas, dtype=float))
    seg = np.diff(alphas) * total
    mid = 0.5 * (alphas[:-1] + alphas[1:])
    mx = x0 + mid * (x1 - x0)
    my = y0 + mid * (y1 - y0)
    ix = np.floor((mx - grid.x_min) / grid.dx).astype(np.intp)
    iy = np.floor((my - grid.y_min) / grid.dy).astype(np.intp)
    inside = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny) & (seg > 0)
    flat = iy[inside] * grid.nx + ix[inside]
    return flat, seg[inside]


def select_plane_axes(sources: np.ndarray, receivers: np.ndarray) -> tuple[int, int]:
    """Pick the two coordinate axes with the largest spread (the tomographic plane)."""
    pts = np.vstack([np.asarray(sources, float), np.asarray(receivers, float)])
    spread = pts.max(axis=0) - pts.min(axis=0)
    order = np.argsort(spread)[::-1]
    return tuple(sorted(order[:2].tolist()))


def build_system_matrix(sources_2d: np.ndarray, receivers_2d: np.ndarray, grid: PixelGrid):
    """Build the [n_rays, n_pixels] CSR length matrix for every source->receiver pair.

    Ray order is source-major: (s0,r0), (s0,r1), ..., (s1,r0), ...
    Returns ``(L_csr, rays)`` where ``rays`` is a list of (src_idx, rx_idx) tuples.
    """
    sources_2d = np.asarray(sources_2d, float)
    receivers_2d = np.asarray(receivers_2d, float)
    rows, cols, data, rays = [], [], [], []
    r = 0
    for si, s in enumerate(sources_2d):
        for ri, rx in enumerate(receivers_2d):
            idx, lengths = ray_cell_lengths(s, rx, grid)
            rows.extend([r] * idx.size)
            cols.extend(idx.tolist())
            data.extend(lengths.tolist())
            rays.append((si, ri))
            r += 1
    L = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(rays), grid.n_pixels), dtype=float
    )
    return L, rays