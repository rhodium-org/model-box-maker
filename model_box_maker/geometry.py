"""Solid geometry of the base and the lid, built with manifold3d.

Implements model-box-maker REQ-0010 (solid floor of the stated thickness),
REQ-0011 (no wall thinner than the wall thickness), REQ-0012 (seen from above
the box is cut back to a wall no thicker than --wall-max), REQ-0013 (lid over
a lip, stopped by the rim), REQ-0014 (the lid never reaches the model),
REQ-0015 (lattice walls cut along the outline) and REQ-0021 (the cut-back box
still stands).

Box coordinates: the base underside is z = 0 and the closed box's outer
footprint has its minimum corner at (0, 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import manifold3d as m3d
import numpy as np
from scipy.spatial import cKDTree

from .raster import BIG, Cavity
from .spec import BoxSpec

ARC_SEGMENTS = 24  # per quarter circle of a rounded rectangle
CIRCLE_SEGMENTS = 48  # per full circle for manifold3d's round offsets: 0.011 mm chord error at r = 5 mm
m3d.set_circular_segments(CIRCLE_SEGMENTS)
WEB = 1.2  # mm of solid material between lattice holes (three nozzle lines)
LATTICE_BLIND = 0.5  # mm a hole cuts past the greatest wall so it opens into the cavity
LATTICE_OUTSIDE = 1.0  # mm a hole prism starts outside the wall
LATTICE_BESIDE = 1.0  # mm beyond the hole's width and cut depth that count as "beside" it
CAP_ABOVE_RIM = 2.0  # mm the cavity solid rises above the rim before subtraction
TIP_MIN_DEG = 20.0  # REQ-0021: below this the rectangle is kept if it does better
BEND_SKIP_DEG = 30.0  # REQ-0015: no hole where the outline turns more than this across it
SIMPLIFY_EPS = 0.02  # mm tolerance when tidying offset outlines


# ----------------------------------------------------------------- rounded rectangles


def rounded_rect(x0: float, y0: float, x1: float, y1: float, r: float,
                 segments: int = ARC_SEGMENTS) -> np.ndarray:
    """Counter-clockwise polygon of an axis-aligned rectangle with rounded corners."""
    r = min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0)
    if r <= 1e-9:
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    pts = []
    corners = [
        (x1 - r, y1 - r, 0.0),
        (x0 + r, y1 - r, 90.0),
        (x0 + r, y0 + r, 180.0),
        (x1 - r, y0 + r, 270.0),
    ]
    for cx, cy, start in corners:
        for k in range(segments + 1):
            a = np.radians(start + 90.0 * k / segments)
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts, dtype=np.float64)


def rounded_rect_sdf(px: np.ndarray, py: np.ndarray, x0: float, y0: float, x1: float, y1: float,
                     r: float) -> np.ndarray:
    """Signed distance to the rounded rectangle boundary; negative inside."""
    hx = (x1 - x0) / 2.0 - r
    hy = (y1 - y0) / 2.0 - r
    qx = np.abs(px - (x0 + x1) / 2.0) - hx
    qy = np.abs(py - (y0 + y1) / 2.0) - hy
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside - r


def fit_lip_rect(cavity: Cavity, wall: float, r_lip: float) -> tuple[float, float, float, float]:
    """The smallest rounded rectangle keeping every outline point ``wall`` inside (REQ-0011).

    Starts from the cavity's bounding box grown by the wall and, when the corner
    radius would cut inside that thickness, grows all four sides equally until
    every corner of every boundary cell is at least ``wall`` inside.
    """
    x0, y0, x1, y1 = cavity.bbox()
    rect = (x0 - wall, y0 - wall, x1 + wall, y1 + wall)
    if r_lip <= 1e-9:
        return rect
    g = cavity.grid
    from scipy import ndimage as ndi
    boundary = cavity.mask & ~ndi.binary_erosion(cavity.mask, structure=np.ones((3, 3), bool), border_value=0)
    ii, jj = np.nonzero(boundary)
    corners_x = np.concatenate([g.x_edge(ii), g.x_edge(ii + 1), g.x_edge(ii), g.x_edge(ii + 1)])
    corners_y = np.concatenate([g.y_edge(jj), g.y_edge(jj), g.y_edge(jj + 1), g.y_edge(jj + 1)])

    def ok(e: float) -> bool:
        d = rounded_rect_sdf(corners_x, corners_y, rect[0] - e, rect[1] - e, rect[2] + e, rect[3] + e, r_lip)
        return bool(d.max() <= -wall + 1e-9)

    if ok(0.0):
        return rect
    lo, hi = 0.0, r_lip
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if ok(mid):
            hi = mid
        else:
            lo = mid
    e = hi
    return (rect[0] - e, rect[1] - e, rect[2] + e, rect[3] + e)


# ----------------------------------------------------------------- the cavity outline


def cavity_loops(cavity: Cavity) -> list[np.ndarray]:
    """Boundary loops of the cavity mask along cell edges, in pose coordinates.

    Outer loops run counter-clockwise and holes clockwise (the mask is always
    on the left of travel). Collinear runs are merged. Diagonal pinches were
    filled when the mask was grown, so every corner has one way out.
    """
    mask = cavity.mask
    g = cavity.grid
    nx, ny = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    starts, ends = [], []
    i, j = np.nonzero(mask & ~padded[1:nx + 1, 0:ny])  # -Y neighbour missing
    starts.append(np.stack([i, j], 1)); ends.append(np.stack([i + 1, j], 1))
    i, j = np.nonzero(mask & ~padded[2:nx + 2, 1:ny + 1])  # +X
    starts.append(np.stack([i + 1, j], 1)); ends.append(np.stack([i + 1, j + 1], 1))
    i, j = np.nonzero(mask & ~padded[1:nx + 1, 2:ny + 2])  # +Y
    starts.append(np.stack([i + 1, j + 1], 1)); ends.append(np.stack([i, j + 1], 1))
    i, j = np.nonzero(mask & ~padded[0:nx, 1:ny + 1])  # -X
    starts.append(np.stack([i, j + 1], 1)); ends.append(np.stack([i, j], 1))
    s = np.vstack(starts)
    e = np.vstack(ends)
    stride = ny + 2
    s_key = (s[:, 0] * stride + s[:, 1]).tolist()
    e_key = (e[:, 0] * stride + e[:, 1]).tolist()
    by_start = {k: idx for idx, k in enumerate(s_key)}
    if len(by_start) != len(s_key):
        raise RuntimeError("internal error: cavity outline has a pinch")
    visited = np.zeros(len(s), dtype=bool)
    loops: list[np.ndarray] = []
    for first in range(len(s)):
        if visited[first]:
            continue
        idx = first
        corners = []
        while not visited[idx]:
            visited[idx] = True
            corners.append(s[idx])
            idx = by_start[e_key[idx]]
        pts = np.array(corners, dtype=np.float64)
        # merge collinear runs: keep a corner only where the direction changes
        d_in = pts - np.roll(pts, 1, axis=0)
        d_out = np.roll(pts, -1, axis=0) - pts
        turn = (d_in[:, 0] * d_out[:, 1] - d_in[:, 1] * d_out[:, 0]) != 0
        pts = pts[turn]
        loops.append(np.column_stack([g.x_edge(pts[:, 0]), g.y_edge(pts[:, 1])]))
    return loops


def signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def section_polygons(section: m3d.CrossSection) -> list[np.ndarray]:
    return [np.asarray(p, dtype=np.float64) for p in section.to_polygons()]


def boundary_distance(points: np.ndarray, section: m3d.CrossSection) -> np.ndarray:
    """Unsigned distance from each point to the nearest boundary of the section."""
    pts = np.asarray(points, dtype=np.float64)[:, :2]
    best = np.full(len(pts), np.inf)
    for poly in section_polygons(section):
        a = poly
        b = np.roll(poly, -1, axis=0)
        ab = b - a
        length2 = np.maximum((ab ** 2).sum(axis=1), 1e-18)
        for start in range(0, len(pts), 512):
            p = pts[start:start + 512]
            ap = p[:, None, :] - a[None, :, :]
            t = np.clip((ap * ab[None, :, :]).sum(axis=2) / length2[None, :], 0.0, 1.0)
            closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
            d = np.linalg.norm(p[:, None, :] - closest, axis=2).min(axis=1)
            best[start:start + 512] = np.minimum(best[start:start + 512], d)
    return best


def contains(points: np.ndarray, section: m3d.CrossSection) -> np.ndarray:
    """Even-odd point-in-polygon over every loop of the section."""
    pts = np.asarray(points, dtype=np.float64)[:, :2]
    inside = np.zeros(len(pts), dtype=bool)
    for poly in section_polygons(section):
        a = poly
        b = np.roll(poly, -1, axis=0)
        for start in range(0, len(pts), 512):
            p = pts[start:start + 512]
            px, py = p[:, 0][:, None], p[:, 1][:, None]
            ay, by = a[:, 1][None, :], b[:, 1][None, :]
            ax, bx = a[:, 0][None, :], b[:, 0][None, :]
            crosses = (ay > py) != (by > py)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_at = ax + (py - ay) * (bx - ax) / (by - ay)
            hit = crosses & (px < x_at)
            inside[start:start + 512] ^= (hit.sum(axis=1) % 2 == 1)
    return inside


def signed_distance(points: np.ndarray, section: m3d.CrossSection) -> np.ndarray:
    d = boundary_distance(points, section)
    return np.where(contains(points, section), -d, d)


def centroid(section: m3d.CrossSection) -> np.ndarray:
    total = 0.0
    acc = np.zeros(2)
    for poly in section_polygons(section):
        x, y = poly[:, 0], poly[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cross = x * yn - xn * y
        area = 0.5 * cross.sum()
        if abs(area) < 1e-12:
            continue
        acc += np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / 6.0
        total += area
    return acc / total if abs(total) > 1e-12 else np.array(section.bounds()[:2]) * 0.0


def tipping_angle(section: m3d.CrossSection, height: float) -> float:
    """Degrees the closed box can tilt before its mid-height centre leaves the support (REQ-0021)."""
    c = centroid(section)
    hull = section.hull()
    d = float(boundary_distance(c[None, :], hull)[0])
    return float(np.degrees(np.arctan2(d, height / 2.0)))


# ----------------------------------------------------------------- the exterior


@dataclass
class Exterior:
    """Plan-view profiles: lip, wall below the lip (also the lid), and the skirt's inside."""

    lip: m3d.CrossSection
    body: m3d.CrossSection
    skirt_inner: m3d.CrossSection
    kind: str  # "cut back" or "rectangle"
    reason: str
    footprint_area: float
    rectangle_area: float
    tipping_deg: float
    tipping_cut_deg: float
    tipping_rect_deg: float
    warnings: list[str] = field(default_factory=list)

    def translated(self, dx: float, dy: float) -> "Exterior":
        return Exterior(
            lip=self.lip.translate([dx, dy]), body=self.body.translate([dx, dy]),
            skirt_inner=self.skirt_inner.translate([dx, dy]), kind=self.kind, reason=self.reason,
            footprint_area=self.footprint_area, rectangle_area=self.rectangle_area,
            tipping_deg=self.tipping_deg, tipping_cut_deg=self.tipping_cut_deg,
            tipping_rect_deg=self.tipping_rect_deg, warnings=list(self.warnings),
        )


def build_exterior(cavity: Cavity, lip_rect: tuple[float, float, float, float], spec: BoxSpec,
                   height: float) -> Exterior:
    """REQ-0012 and REQ-0021 in pose coordinates.

    The lip profile is the rounded rectangle cut back to within
    ``wall_max - body_offset`` of the cavity outline; the wall below the lip and
    the lid outline are that profile grown by the body offset, so the wall is
    never thicker than --wall-max nor thinner than --wall. The rectangle is
    kept when the cut-back would reach the whole rectangle anyway, would split
    the box, or would tip below TIP_MIN_DEG where the rectangle does not.
    """
    off = spec.body_offset
    rect = m3d.CrossSection([rounded_rect(*lip_rect, spec.lip_radius)])
    rect_area = rect.area()
    outer = [loop for loop in cavity_loops(cavity) if signed_area(loop) > 0]
    cap = m3d.CrossSection(outer).offset(spec.wall_max - off, m3d.JoinType.Round).simplify(SIMPLIFY_EPS)
    # only near-duplicate crossing vertices are removed here: the rectangle's sides must stay exactly
    # the wall from the cavity, so no real simplification
    cut = (rect ^ cap).simplify(1e-4)
    cut_area = cut.area()
    # the rectangle grows with mitre joins, which keep a sharp corner sharp and grow a rounded one
    # by exactly the offset; a cut-back profile grows with round joins, which never exceed the
    # offset at the crossings between flat and arc (a mitre there pokes past --wall-max)
    body_rect = rect.offset(off, m3d.JoinType.Miter)
    body_cut = cut.offset(off, m3d.JoinType.Round)
    tip_rect = tipping_angle(body_rect, height)
    tip_cut = tipping_angle(body_cut, height) if cut_area > 1e-9 else 0.0

    if cut_area >= rect_area * (1.0 - 1e-6):
        kind, reason, lip, body = "rectangle", f"--wall-max {spec.wall_max:g} mm reaches the whole rectangle", rect, body_rect
    elif len(cut.decompose()) != 1:
        kind, reason, lip, body = "rectangle", "the cut-back outline would split the box into separate pieces", rect, body_rect
    elif tip_cut < TIP_MIN_DEG <= tip_rect:
        kind, reason, lip, body = ("rectangle", f"the cut-back outline would tip at {tip_cut:.0f} degrees; "
                                   f"the rectangle tips at {tip_rect:.0f}", rect, body_rect)
    else:
        kind, reason, lip, body = "cut back", f"wall bounded by --wall-max {spec.wall_max:g} mm", cut, body_cut
    tipping = tip_rect if kind == "rectangle" else tip_cut
    warnings = []
    if tipping < TIP_MIN_DEG:
        warnings.append(f"warning: the box tips at only {tipping:.0f} degrees; consider another orientation")
    join = m3d.JoinType.Miter if kind == "rectangle" else m3d.JoinType.Round
    return Exterior(
        lip=lip, body=body, skirt_inner=lip.offset(spec.fit, join),
        kind=kind, reason=reason, footprint_area=float(body.area()), rectangle_area=float(body_rect.area()),
        tipping_deg=tipping, tipping_cut_deg=tip_cut, tipping_rect_deg=tip_rect, warnings=warnings,
    )


# ----------------------------------------------------------------- layout


@dataclass
class Layout:
    """Where everything sits in box coordinates."""

    spec: BoxSpec
    lip: tuple[float, float, float, float]  # bounds of the lip profile (x0, y0, x1, y1)
    body: tuple[float, float, float, float]  # bounds of the wall below the lip, also the lid's outline
    r_lip: float
    r_body: float
    rim_z: float
    shoulder_z: float
    translation: np.ndarray  # pose coordinates -> box coordinates
    floor_z: float  # z of the lowest cradle point (= spec.floor)
    cradle_top_z: float  # highest point of the cradle surface
    model_top_z: float
    exterior: Exterior  # profiles in box coordinates
    cavity_section: m3d.CrossSection  # the cavity outline (outer loops) in box coordinates
    lattice: list[dict] = field(default_factory=list)  # holes cut: outline point, outward normal, centre z

    @property
    def size(self) -> tuple[float, float, float]:
        return (self.body[2] - self.body[0], self.body[3] - self.body[1], self.rim_z + self.spec.lid_plate)

    def lip_section(self) -> m3d.CrossSection:
        return self.exterior.lip

    def body_section(self) -> m3d.CrossSection:
        return self.exterior.body

    def skirt_inner_section(self) -> m3d.CrossSection:
        return self.exterior.skirt_inner

    def mirrored_for_print(self, section: m3d.CrossSection) -> m3d.CrossSection:
        """The lid prints plate down and is turned over about X to seat, so print its mirror in y."""
        yc = (self.body[1] + self.body[3]) / 2.0
        return section.translate([0.0, -yc]).mirror([0.0, 1.0]).translate([0.0, yc])

    def wall_below_lip(self) -> float:
        return self.spec.wall + self.spec.body_offset


def layout_for(cavity: Cavity, lip_rect: tuple[float, float, float, float], rim_z_pose: float,
               spec: BoxSpec, exterior: Exterior) -> Layout:
    bx0, by0, bx1, by1 = exterior.body.bounds()
    tx = -bx0
    ty = -by0
    tz = spec.floor - cavity.floor_min
    ext = exterior.translated(tx, ty)
    lip = ext.lip.bounds()
    body = ext.body.bounds()
    rim_z = rim_z_pose + tz
    outer = [loop + np.array([tx, ty]) for loop in cavity_loops(cavity) if signed_area(loop) > 0]
    return Layout(
        spec=spec,
        lip=(float(lip[0]), float(lip[1]), float(lip[2]), float(lip[3])),
        body=(float(body[0]), float(body[1]), float(body[2]), float(body[3])),
        r_lip=spec.lip_radius,
        r_body=spec.corner_radius,
        rim_z=rim_z,
        shoulder_z=rim_z - spec.lip,
        translation=np.array([tx, ty, tz]),
        floor_z=spec.floor,
        cradle_top_z=cavity.floor_max + tz,
        model_top_z=cavity.zmax + tz,
        exterior=ext,
        cavity_section=m3d.CrossSection(outer),
    )


# ----------------------------------------------------------------- the cavity solid


def heightmap_solid(cavity: Cavity, translation: np.ndarray, z_cap: float) -> tuple[np.ndarray, np.ndarray]:
    """A closed solid above the cradle: the cavity to subtract from the base.

    Vertices sit at cell corners and take the lowest value of the cells around
    them, so the surface lies at or below every cell's value (REQ-0009). Walls
    rise from the outline to ``z_cap``; a cap closes the top.
    """
    g = cavity.grid
    mask = cavity.mask
    floor = cavity.floor
    nx, ny = mask.shape
    used = np.zeros((nx + 1, ny + 1), dtype=bool)
    val = np.full((nx + 1, ny + 1), BIG)
    cell_val = np.where(mask, floor, BIG)
    for di in (0, 1):
        for dj in (0, 1):
            used[di:nx + di, dj:ny + dj] |= mask
            np.minimum(val[di:nx + di, dj:ny + dj], cell_val, out=val[di:nx + di, dj:ny + dj])
    n_used = int(used.sum())
    ids = np.full((nx + 1, ny + 1), -1, dtype=np.int64)
    ids[used] = np.arange(n_used)
    ci, cj = np.nonzero(used)
    tx, ty, tz = (float(t) for t in translation)
    bottom = np.column_stack([g.x_edge(ci) + tx, g.y_edge(cj) + ty, val[used] + tz])
    top = np.column_stack([g.x_edge(ci) + tx, g.y_edge(cj) + ty, np.full(n_used, z_cap)])
    vertices = np.vstack([bottom, top])
    ids_b = ids
    ids_t = np.where(used, ids + n_used, -1)

    mi, mj = np.nonzero(mask)
    c00 = ids_b[mi, mj]
    c10 = ids_b[mi + 1, mj]
    c01 = ids_b[mi, mj + 1]
    c11 = ids_b[mi + 1, mj + 1]
    t00 = ids_t[mi, mj]
    t10 = ids_t[mi + 1, mj]
    t01 = ids_t[mi, mj + 1]
    t11 = ids_t[mi + 1, mj + 1]
    faces = [
        np.column_stack([c00, c01, c10]), np.column_stack([c10, c01, c11]),  # cradle, normal -Z
        np.column_stack([t00, t10, t01]), np.column_stack([t10, t11, t01]),  # cap, normal +Z
    ]

    def open_side(shift_i: int, shift_j: int) -> tuple[np.ndarray, np.ndarray]:
        """Mask cells whose neighbour at (shift_i, shift_j) is outside the mask."""
        padded = np.pad(mask, 1, constant_values=False)
        neighbour = padded[1 + shift_i:1 + shift_i + nx, 1 + shift_j:1 + shift_j + ny]
        return np.nonzero(mask & ~neighbour)

    # -X wall: corners (i, j) -> (i, j+1); normal -X
    i, j = open_side(-1, 0)
    a_b, a_t, b_b, b_t = ids_b[i, j], ids_t[i, j], ids_b[i, j + 1], ids_t[i, j + 1]
    faces += [np.column_stack([a_b, a_t, b_b]), np.column_stack([b_b, a_t, b_t])]
    # +X wall: corners (i+1, j) -> (i+1, j+1); normal +X
    i, j = open_side(1, 0)
    a_b, a_t, b_b, b_t = ids_b[i + 1, j], ids_t[i + 1, j], ids_b[i + 1, j + 1], ids_t[i + 1, j + 1]
    faces += [np.column_stack([a_b, b_b, a_t]), np.column_stack([b_b, b_t, a_t])]
    # -Y wall: corners (i, j) -> (i+1, j); normal -Y
    i, j = open_side(0, -1)
    a_b, a_t, b_b, b_t = ids_b[i, j], ids_t[i, j], ids_b[i + 1, j], ids_t[i + 1, j]
    faces += [np.column_stack([a_b, b_b, a_t]), np.column_stack([b_b, b_t, a_t])]
    # +Y wall: corners (i, j+1) -> (i+1, j+1); normal +Y
    i, j = open_side(0, 1)
    a_b, a_t, b_b, b_t = ids_b[i, j + 1], ids_t[i, j + 1], ids_b[i + 1, j + 1], ids_t[i + 1, j + 1]
    faces += [np.column_stack([a_b, a_t, b_b]), np.column_stack([b_b, a_t, b_t])]

    return vertices, np.vstack(faces).astype(np.int64)


# ----------------------------------------------------------------- manifold helpers


def to_manifold(vertices: np.ndarray, faces: np.ndarray) -> m3d.Manifold:
    mesh = m3d.Mesh64(vert_properties=np.ascontiguousarray(vertices, dtype=np.float64),
                      tri_verts=np.ascontiguousarray(faces, dtype=np.uint64))
    solid = m3d.Manifold(mesh)
    status = solid.status()
    if status != m3d.Error.NoError:
        raise RuntimeError(f"internal error: built a non-manifold solid ({status})")
    return solid


def from_manifold(solid: m3d.Manifold) -> tuple[np.ndarray, np.ndarray]:
    mesh = solid.to_mesh64()
    v = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    f = np.asarray(mesh.tri_verts, dtype=np.int64)
    return v, f


def extrude(polygons: list[np.ndarray], height: float, z0: float = 0.0) -> m3d.Manifold:
    """Extrude a polygon (with optional holes, given clockwise) from z0 up by height."""
    section = m3d.CrossSection([np.ascontiguousarray(p, dtype=np.float64) for p in polygons])
    return extrude_section(section, height, z0)


def extrude_section(section: m3d.CrossSection, height: float, z0: float = 0.0) -> m3d.Manifold:
    solid = m3d.Manifold.extrude(section, float(height))
    if z0:
        solid = solid.translate([0.0, 0.0, float(z0)])
    return solid


# ----------------------------------------------------------------- bodies


def build_base(cavity: Cavity, layout: Layout) -> m3d.Manifold:
    """Body and lip prisms minus the cavity solid, minus lattice holes if asked (REQ-0010..0015)."""
    spec = layout.spec
    body = extrude_section(layout.body_section(), layout.shoulder_z)
    lip = extrude_section(layout.lip_section(), layout.rim_z)
    v, f = heightmap_solid(cavity, layout.translation, layout.rim_z + CAP_ABOVE_RIM)
    hollow = to_manifold(v, f)
    base = (body + lip) - hollow
    if spec.walls != "solid":
        holes = lattice_holes(layout, cavity)
        if holes is not None:
            base = base - holes
    if base.status() != m3d.Error.NoError:
        raise RuntimeError(f"internal error: base boolean failed ({base.status()})")
    return base


def build_lid(layout: Layout) -> m3d.Manifold:
    """The lid the way it prints: plate on z = 0, skirt rising from it (REQ-0013, REQ-0016).

    Printed plate down, the lid is turned over about X to seat, so it is built
    from the mirror image of the base's outline.
    """
    spec = layout.spec
    outline = layout.mirrored_for_print(layout.body_section())
    inner = layout.mirrored_for_print(layout.skirt_inner_section())
    # one prism for plate and skirt together, then the skirt's inside removed above the plate: a
    # union of two prisms sharing their outer faces leaves sliver triangles at the seam
    block = extrude_section(outline, spec.lid_plate + spec.skirt_height)
    core = extrude_section(inner, spec.skirt_height + 1.0, z0=spec.lid_plate)
    lid = block - core
    if lid.status() != m3d.Error.NoError:
        raise RuntimeError(f"internal error: lid boolean failed ({lid.status()})")
    return lid


def seated_lid_transform(layout: Layout) -> np.ndarray:
    """4x4 transform taking the printed lid to its seated pose in box coordinates."""
    spec = layout.spec
    rx180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    t = np.array([0.0, layout.body[1] + layout.body[3], layout.rim_z + spec.lid_plate])
    m = np.eye(4)
    m[:3, :3] = rx180
    m[:3, 3] = t
    return m


# ----------------------------------------------------------------- lattice


def hole_polygon(pattern: str, max_hole: float) -> np.ndarray:
    """One hole in wall coordinates (u along the wall, v up), centred at the origin.

    Both shapes stay printable without support (REQ-0017): the flat-top hexagon
    bridges a span shorter than its width and its upper edges lean 30 degrees
    from vertical; the diamond's upper edges lean under 40 degrees.
    """
    if pattern == "hex":
        rc = max_hole / 2.0  # vertex-to-vertex width equals max_hole; flat top and bottom
        pts = [(rc * np.cos(np.radians(60 * k)), rc * np.sin(np.radians(60 * k))) for k in range(6)]
        return np.array(pts, dtype=np.float64)
    if pattern == "diamond":
        h = max_hole / 2.0
        w = 0.8 * max_hole / 2.0
        return np.array([(w, 0.0), (0.0, h), (-w, 0.0), (0.0, -h)], dtype=np.float64)
    raise ValueError(pattern)


def cradle_beside(cells_xy: np.ndarray, cells_z: np.ndarray, tree: cKDTree, p: np.ndarray, n: np.ndarray,
                  half_width: float, depth: float) -> float | None:
    """Highest cradle point in the window of cavity cells directly behind a place on the wall.

    The window is ``half_width`` either way along the wall and ``depth`` inward
    from the outline point ``p`` (outward normal ``n``). None when no cavity cell
    lies there, which happens only when a kept rectangle stands off the cavity.
    """
    near = tree.query_ball_point(p, float(np.hypot(half_width, depth)))
    if not near:
        return None
    q = cells_xy[near] - p
    t = np.array([-n[1], n[0]])
    along = q @ t
    inward = -(q @ n)
    sel = (np.abs(along) <= half_width) & (inward >= -0.5) & (inward <= depth)
    if not sel.any():
        return None
    return float(cells_z[near][sel].max())


def _outer_loop(section: m3d.CrossSection) -> np.ndarray:
    polys = section_polygons(section)
    loop = max(polys, key=lambda p: abs(signed_area(p)))
    return loop if signed_area(loop) > 0 else loop[::-1]


def lattice_holes(layout: Layout, cavity: Cavity) -> m3d.Manifold | None:
    """Hole prisms cut inward along the outline's normal, out of every band (REQ-0015)."""
    spec = layout.spec
    poly = hole_polygon(spec.walls, spec.max_hole)
    hole_w = float(np.ptp(poly[:, 0]))
    hole_h = float(np.ptp(poly[:, 1]))
    z_hi = layout.shoulder_z - spec.band
    z_floor = layout.floor_z + spec.band
    if z_hi - z_floor < hole_h:
        return None
    loop = _outer_loop(layout.body_section())
    seg = np.roll(loop, -1, axis=0) - loop
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-9
    loop, seg, seg_len = loop[keep], seg[keep], seg_len[keep]
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    tangents = seg / seg_len[:, None]
    su = hole_w + WEB
    sz = hole_h + WEB
    n_along = int(np.floor(total / su))
    if n_along < 1:
        return None
    pitch_s = total / n_along

    # cradle height beside each place along the wall, from the cavity cells nearby
    g = cavity.grid
    mi, mj = np.nonzero(cavity.mask)
    cells_xy = np.column_stack([g.x_edge(mi) + g.pitch / 2 + layout.translation[0],
                                g.y_edge(mj) + g.pitch / 2 + layout.translation[1]])
    cells_z = cavity.floor[mi, mj] + layout.translation[2]
    tree = cKDTree(cells_xy)
    half_width = hole_w / 2.0 + LATTICE_BESIDE
    look_in = spec.wall_max + LATTICE_BLIND + LATTICE_BESIDE

    def at(s: float) -> tuple[np.ndarray, np.ndarray]:
        s = s % total
        k = int(np.searchsorted(cum, s, side="right") - 1)
        k = min(max(k, 0), len(seg) - 1)
        return loop[k] + (s - cum[k]) * tangents[k], tangents[k]

    depth = spec.wall_max + LATTICE_BLIND + LATTICE_OUTSIDE
    reach = hole_w / 2.0 + spec.wall_max + LATTICE_BLIND
    prisms: list[m3d.Manifold] = []
    row = 0
    zc = z_hi - hole_h / 2.0
    while zc - hole_h / 2.0 >= z_floor - 1e-9:
        offset = pitch_s / 2.0 if row % 2 else 0.0
        for k in range(n_along):
            s = offset + k * pitch_s
            p, t = at(s)
            # a hole's prism reaches wall_max inward, so a bend that far along the wall on either
            # side would let two prisms meet inside a corner: measure the turn over that reach
            _, t_before = at(s - reach)
            _, t_after = at(s + reach)
            turn = np.degrees(np.arccos(np.clip(np.dot(t_before, t_after), -1.0, 1.0)))
            if turn > BEND_SKIP_DEG:
                continue
            n = np.array([t[1], -t[0]])  # outward for a counter-clockwise loop
            beside = cradle_beside(cells_xy, cells_z, tree, p, n, half_width, look_in)
            z_lo = (layout.floor_z if beside is None else beside) + spec.band
            if zc - hole_h / 2.0 < z_lo - 1e-9:
                continue
            layout.lattice.append({"point": p.copy(), "normal": n.copy(), "z": float(zc), "z_low": float(z_lo)})
            prism = m3d.Manifold.extrude(m3d.CrossSection([poly]), depth)
            m = np.array([[t[0], 0.0, n[0], 0.0], [t[1], 0.0, n[1], 0.0], [0.0, 1.0, 0.0, 0.0]])
            start = p - n * (spec.wall_max + LATTICE_BLIND)
            m[0, 3], m[1, 3], m[2, 3] = start[0], start[1], zc
            prisms.append(prism.transform(m))
        row += 1
        zc -= sz
    if not prisms:
        return None
    return m3d.Manifold.compose(prisms)
