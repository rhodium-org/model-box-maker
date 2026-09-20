"""The cradle as a heightmap: the lowest surface of the grown model over each cell.

Implements model-box-maker REQ-0008 (the cavity is the drop path of the model
grown by the clearance) and REQ-0009 (rounding to the grid only ever enlarges
the cavity). Every step here rounds outward:

* a cell is *touched* when a triangle's projection really intersects the cell
  square (an exact separating-axis test), and the cell takes, for every such
  triangle, the lower of the triangle's plane over the whole cell and the
  triangle's own lowest point, which is at or below the surface anywhere over
  the cell (triangles are first split to a few cells so a steep one cannot
  drag its neighbours far down);
* growing by the clearance is a grey erosion with a spherical structuring
  function measured square-to-square, so a cell is lowered by every model cell
  a ball of that radius could reach from it;
* the outline is then grown by one more cell and each cell drops by half its
  own diagonal, as REQ-0009 states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage as ndi

BIG = 1.0e9
CHUNK = 1_000_000
SUBDIVIDE_FACTOR = 4  # triangles are split to at most this many cells per edge


@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    pitch: float
    nx: int
    ny: int

    def x_edge(self, i) -> np.ndarray | float:
        return self.x0 + np.asarray(i) * self.pitch

    def y_edge(self, j) -> np.ndarray | float:
        return self.y0 + np.asarray(j) * self.pitch


@dataclass
class Cavity:
    """The cavity of one pose, in pose coordinates (model rotated, not yet shifted)."""

    grid: Grid
    mask: np.ndarray  # (nx, ny) bool, cells of the cavity
    floor: np.ndarray  # (nx, ny) float, cradle height where mask, BIG elsewhere
    touched: np.ndarray  # (nx, ny) bool, cells the model itself projects onto
    zmax: float  # highest point of the model in this pose

    @property
    def floor_min(self) -> float:
        return float(self.floor[self.mask].min())

    @property
    def floor_max(self) -> float:
        return float(self.floor[self.mask].max())

    def bbox(self) -> tuple[float, float, float, float]:
        """Outer cell edges of the mask: (x0, y0, x1, y1)."""
        ii, jj = np.nonzero(self.mask)
        g = self.grid
        return (
            float(g.x_edge(ii.min())), float(g.y_edge(jj.min())),
            float(g.x_edge(ii.max() + 1)), float(g.y_edge(jj.max() + 1)),
        )

    def cavity_volume(self, rim_z: float) -> float:
        """Empty space between the cradle floor and the rim over the footprint (REQ-0004)."""
        depth = rim_z - self.floor[self.mask]
        return float(depth.sum() * self.grid.pitch ** 2)

    def rim_z(self, top_space: float) -> float:
        return self.zmax + top_space


def subdivide(vertices: np.ndarray, faces: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Split triangles until no edge is longer than a few cells; returns (vertices, faces).

    The per-cell plane minimum in ``lower_envelope`` is exact for any triangle
    size; splitting only bounds how far a steep triangle's lowest point can
    drag the cells along its edges.
    """
    max_edge = SUBDIVIDE_FACTOR * pitch
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    v2, f2 = trimesh.remesh.subdivide_to_size(v, f, max_edge=max_edge, max_iter=64)
    return np.asarray(v2, dtype=np.float64), np.asarray(f2, dtype=np.int64)


def posed_triangles(vertices: np.ndarray, faces: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate the vertices (one matrix product) and gather triangles as (n, 3, 3)."""
    rotated = np.asarray(vertices, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T
    return rotated[faces]


def _tri_square_overlap(ax, ay, bx, by, cx, cy, sx, sy, half):
    """Exact 2D overlap of triangles (columns a, b, c) with squares centred at (sx, sy)."""
    tx0, ty0 = ax - sx, ay - sy
    tx1, ty1 = bx - sx, by - sy
    tx2, ty2 = cx - sx, cy - sy
    separated = (np.maximum(np.maximum(tx0, tx1), tx2) < -half) | (np.minimum(np.minimum(tx0, tx1), tx2) > half)
    separated |= (np.maximum(np.maximum(ty0, ty1), ty2) < -half) | (np.minimum(np.minimum(ty0, ty1), ty2) > half)
    for (px, py), (qx, qy) in (((tx0, ty0), (tx1, ty1)), ((tx1, ty1), (tx2, ty2)), ((tx2, ty2), (tx0, ty0))):
        nx_ = -(qy - py)
        ny_ = qx - px
        p0 = tx0 * nx_ + ty0 * ny_
        p1 = tx1 * nx_ + ty1 * ny_
        p2 = tx2 * nx_ + ty2 * ny_
        r = half * (np.abs(nx_) + np.abs(ny_))
        separated |= (np.maximum(np.maximum(p0, p1), p2) < -r) | (np.minimum(np.minimum(p0, p1), p2) > r)
    return ~separated


def lower_envelope(tri: np.ndarray, pitch: float, pad: float, exact: bool) -> tuple[Grid, np.ndarray, np.ndarray]:
    """Per-cell lowest z of the triangle soup ``tri`` (n,3,3), rounding outward.

    For every (triangle, cell) pair the cell takes the lower of the triangle's
    plane over the whole cell square and the triangle's own lowest point, which
    is at or below the surface anywhere the triangle covers the cell. With
    ``exact`` a pair counts only when the triangle's projection really meets
    the cell square; without it the triangle's bounding box is used, which is
    faster and still rounds outward. Returns the grid, the touched mask and L
    (BIG where untouched).
    """
    ax, ay, az = tri[:, 0, 0], tri[:, 0, 1], tri[:, 0, 2]
    bx, by, bz = tri[:, 1, 0], tri[:, 1, 1], tri[:, 1, 2]
    cx, cy, cz = tri[:, 2, 0], tri[:, 2, 1], tri[:, 2, 2]
    lox = np.minimum(np.minimum(ax, bx), cx)
    hix = np.maximum(np.maximum(ax, bx), cx)
    loy = np.minimum(np.minimum(ay, by), cy)
    hiy = np.maximum(np.maximum(ay, by), cy)
    zmin = np.minimum(np.minimum(az, bz), cz)
    x0 = float(np.floor((lox.min() - pad) / pitch) * pitch)
    y0 = float(np.floor((loy.min() - pad) / pitch) * pitch)
    nx = int(np.ceil((hix.max() + pad - x0) / pitch)) + 1
    ny = int(np.ceil((hiy.max() + pad - y0) / pitch)) + 1
    grid = Grid(x0, y0, pitch, nx, ny)

    i0 = np.clip(np.floor((lox - x0) / pitch).astype(np.int64), 0, nx - 1)
    i1 = np.clip(np.floor((hix - x0) / pitch).astype(np.int64), 0, nx - 1)
    j0 = np.clip(np.floor((loy - y0) / pitch).astype(np.int64), 0, ny - 1)
    j1 = np.clip(np.floor((hiy - y0) / pitch).astype(np.int64), 0, ny - 1)

    # plane of each triangle: z = az + gx (x - ax) + gy (y - ay); steep or degenerate planes fall
    # back to the triangle's lowest point over every cell they touch
    e1x, e1y, e2x, e2y = bx - ax, by - ay, cx - ax, cy - ay
    det = e1x * e2y - e1y * e2x
    size = np.hypot(e1x, e1y) * np.hypot(e2x, e2y)
    steep = np.abs(det) <= 1e-9 * np.maximum(size, 1e-30)
    safe = np.where(steep, 1.0, det)
    gx = np.where(steep, 0.0, ((bz - az) * e2y - (cz - az) * e1y) / safe)
    gy = np.where(steep, 0.0, (e1x * (cz - az) - e2x * (bz - az)) / safe)
    slope = np.abs(gx) + np.abs(gy)

    counts = (i1 - i0 + 1) * (j1 - j0 + 1)
    nj = j1 - j0 + 1
    L = np.full((nx, ny), BIG)
    half = pitch / 2.0
    cum = np.cumsum(counts)
    n = len(tri)
    start = 0
    while start < n:
        limit = cum[start] - counts[start] + CHUNK
        end = int(min(n, max(start + 1, np.searchsorted(cum, limit, side="right"))))
        c = counts[start:end]
        total = int(c.sum())
        tri_idx = start + np.repeat(np.arange(end - start), c)
        offsets = np.repeat(np.cumsum(c) - c, c)
        local = np.arange(total) - offsets
        njt = nj[tri_idx]
        ii = i0[tri_idx] + local // njt
        jj = j0[tri_idx] + local % njt
        sx = x0 + (ii + 0.5) * pitch
        sy = y0 + (jj + 0.5) * pitch
        t = tri_idx
        if exact:
            keep = _tri_square_overlap(ax[t], ay[t], bx[t], by[t], cx[t], cy[t], sx, sy, half)
            t, ii, jj, sx, sy = t[keep], ii[keep], jj[keep], sx[keep], sy[keep]
        zc = az[t] + gx[t] * (sx - ax[t]) + gy[t] * (sy - ay[t]) - slope[t] * half
        z = np.where(steep[t], zmin[t], np.maximum(zc, zmin[t]))
        _scatter_min(L, ii, jj, z)
        start = end
    touched = L < BIG / 2
    return grid, touched, L


def _scatter_min(L: np.ndarray, ii: np.ndarray, jj: np.ndarray, z: np.ndarray) -> None:
    if len(ii) == 0:
        return
    flat = ii * L.shape[1] + jj
    order = np.argsort(flat, kind="stable")
    flat_s = flat[order]
    z_s = z[order]
    starts = np.flatnonzero(np.r_[True, flat_s[1:] != flat_s[:-1]])
    mins = np.minimum.reduceat(z_s, starts)
    cells = flat_s[starts]
    view = L.ravel()
    view[cells] = np.minimum(view[cells], mins)


def grow_by_clearance(L: np.ndarray, pitch: float, clearance: float) -> np.ndarray:
    """Lower envelope of the model grown by a ball of radius ``clearance``.

    The structuring function is measured between cell squares, not centres, so
    the result is at or below the true grown surface anywhere over each cell.
    """
    r = int(np.ceil(clearance / pitch)) + 1
    di, dj = np.mgrid[-r:r + 1, -r:r + 1]
    gap = np.hypot(np.maximum(np.abs(di) - 1, 0), np.maximum(np.abs(dj) - 1, 0)) * pitch
    footprint = gap <= clearance + 1e-12
    structure = np.zeros(gap.shape)
    structure[footprint] = np.sqrt(np.maximum(clearance ** 2 - gap[footprint] ** 2, 0.0))
    return ndi.grey_erosion(L, footprint=footprint, structure=structure, mode="constant", cval=BIG)


def grow_one_cell(Lc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grow the outline by one cell (REQ-0009) and fill diagonal pinches.

    New cells take the lowest value of their neighbours. A pinch, two cells
    touching only at a corner, is filled so the cavity walls form clean loops.
    """
    mask = Lc < BIG / 2
    out = Lc.copy()
    ring = ndi.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool)) & ~mask
    nb = ndi.minimum_filter(Lc, size=3, mode="constant", cval=BIG)
    out[ring] = nb[ring]
    mask = mask | ring
    for _ in range(64):
        m = mask
        a = m[:-1, :-1] & m[1:, 1:] & ~m[1:, :-1] & ~m[:-1, 1:]
        b = m[1:, :-1] & m[:-1, 1:] & ~m[:-1, :-1] & ~m[1:, 1:]
        if not (a.any() or b.any()):
            break
        nb = ndi.minimum_filter(out, size=3, mode="constant", cval=BIG)
        fill = np.zeros_like(m)
        fill[1:, :-1] |= a
        fill[:-1, 1:] |= a
        fill[:-1, :-1] |= b
        fill[1:, 1:] |= b
        fill &= ~mask
        out[fill] = nb[fill]
        mask = mask | fill
    out[~mask] = BIG
    return mask, out


def cavity_for(tri: np.ndarray, pitch: float, clearance: float, exact: bool) -> Cavity:
    """The cavity of the posed triangle soup ``tri`` at grid pitch ``pitch``."""
    cells = int(np.ceil(clearance / pitch)) + 4
    grid, touched, L = lower_envelope(tri, pitch, pad=cells * pitch, exact=exact)
    Lc = grow_by_clearance(L, pitch, clearance)
    mask, grown = grow_one_cell(Lc)
    floor = np.where(mask, grown - pitch / np.sqrt(2.0), BIG)
    return Cavity(grid=grid, mask=mask, floor=floor, touched=touched, zmax=float(tri[:, :, 2].max()))
