"""Solid geometry of the base and the lid, built with manifold3d.

Implements model-box-maker REQ-0010 (solid floor of the stated thickness),
REQ-0011 (no wall thinner than the wall thickness), REQ-0012 (cuboid with
rounded vertical edges), REQ-0013 (lid over a lip, stopped by the rim),
REQ-0014 (the lid never reaches the model) and REQ-0015 (lattice walls).

Box coordinates: the base underside is z = 0 and the closed box's outer
footprint has its minimum corner at (0, 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import manifold3d as m3d
import numpy as np

from .raster import BIG, Cavity
from .spec import BoxSpec

ARC_SEGMENTS = 24  # per quarter circle
WEB = 1.2  # mm of solid material between lattice holes (three nozzle lines)
LATTICE_BLIND = 0.5  # mm a hole cuts past the nominal wall so it opens into the cavity
CAP_ABOVE_RIM = 2.0  # mm the cavity solid rises above the rim before subtraction


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


# ----------------------------------------------------------------- layout


@dataclass
class Layout:
    """Where everything sits in box coordinates."""

    spec: BoxSpec
    lip: tuple[float, float, float, float]  # lip outer profile rectangle (x0, y0, x1, y1)
    body: tuple[float, float, float, float]  # wall below the lip, also the lid's outline
    r_lip: float
    r_body: float
    rim_z: float
    shoulder_z: float
    translation: np.ndarray  # pose coordinates -> box coordinates
    floor_z: float  # z of the lowest cradle point (= spec.floor)
    cradle_top_z: float  # highest point of the cradle surface
    model_top_z: float

    @property
    def size(self) -> tuple[float, float, float]:
        return (self.body[2] - self.body[0], self.body[3] - self.body[1], self.rim_z + self.spec.lid_plate)

    def body_polygon(self) -> np.ndarray:
        return rounded_rect(*self.body, self.r_body)

    def lip_polygon(self) -> np.ndarray:
        return rounded_rect(*self.lip, self.r_lip)

    def skirt_inner_polygon(self) -> np.ndarray:
        f = self.spec.fit
        x0, y0, x1, y1 = self.lip
        return rounded_rect(x0 - f, y0 - f, x1 + f, y1 + f, self.r_lip + f)

    def wall_below_lip(self) -> float:
        return self.spec.wall + self.spec.body_offset


def layout_for(cavity: Cavity, lip_rect: tuple[float, float, float, float], rim_z_pose: float,
               spec: BoxSpec) -> Layout:
    off = spec.body_offset
    tx = -(lip_rect[0] - off)
    ty = -(lip_rect[1] - off)
    tz = spec.floor - cavity.floor_min
    lip = (lip_rect[0] + tx, lip_rect[1] + ty, lip_rect[2] + tx, lip_rect[3] + ty)
    body = (lip[0] - off, lip[1] - off, lip[2] + off, lip[3] + off)
    rim_z = rim_z_pose + tz
    return Layout(
        spec=spec,
        lip=lip,
        body=body,
        r_lip=spec.lip_radius,
        r_body=spec.corner_radius,
        rim_z=rim_z,
        shoulder_z=rim_z - spec.lip,
        translation=np.array([tx, ty, tz]),
        floor_z=spec.floor,
        cradle_top_z=cavity.floor_max + tz,
        model_top_z=cavity.zmax + tz,
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
    solid = m3d.Manifold.extrude(section, float(height))
    if z0:
        solid = solid.translate([0.0, 0.0, float(z0)])
    return solid


# ----------------------------------------------------------------- bodies


def build_base(cavity: Cavity, layout: Layout) -> m3d.Manifold:
    """Body and lip prisms minus the cavity solid, minus lattice holes if asked (REQ-0010..0015)."""
    spec = layout.spec
    body = extrude([layout.body_polygon()], layout.shoulder_z)
    lip = extrude([layout.lip_polygon()], layout.rim_z)
    v, f = heightmap_solid(cavity, layout.translation, layout.rim_z + CAP_ABOVE_RIM)
    hollow = to_manifold(v, f)
    base = (body + lip) - hollow
    if spec.walls != "solid":
        holes = lattice_holes(layout)
        if holes is not None:
            base = base - holes
    if base.status() != m3d.Error.NoError:
        raise RuntimeError(f"internal error: base boolean failed ({base.status()})")
    return base


def build_lid(layout: Layout) -> m3d.Manifold:
    """The lid the way it prints: plate on z = 0, skirt rising from it (REQ-0013, REQ-0016)."""
    spec = layout.spec
    plate = extrude([layout.body_polygon()], spec.lid_plate)
    skirt = extrude([layout.body_polygon(), layout.skirt_inner_polygon()[::-1]], spec.skirt_height,
                    z0=spec.lid_plate)
    lid = plate + skirt
    if lid.status() != m3d.Error.NoError:
        raise RuntimeError(f"internal error: lid boolean failed ({lid.status()})")
    return lid


def seated_lid_transform(layout: Layout) -> np.ndarray:
    """4x4 transform taking the printed lid to its seated pose in box coordinates."""
    spec = layout.spec
    depth = layout.body[3] - layout.body[1]
    rx180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    t = np.array([0.0, layout.body[1] * 2 + depth, layout.rim_z + spec.lid_plate])
    m = np.eye(4)
    m[:3, :3] = rx180
    m[:3, 3] = t
    return m


# ----------------------------------------------------------------- lattice


def hole_polygon(pattern: str, max_hole: float) -> np.ndarray:
    """One hole in panel coordinates (u along the panel, z up), centred at the origin.

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


def lattice_holes(layout: Layout) -> m3d.Manifold | None:
    """Hole prisms through the four wall panels, staying out of the solid bands (REQ-0015)."""
    spec = layout.spec
    poly = hole_polygon(spec.walls, spec.max_hole)
    hole_w = float(np.ptp(poly[:, 0]))
    hole_h = float(np.ptp(poly[:, 1]))
    z_lo = layout.cradle_top_z + spec.band
    z_hi = layout.shoulder_z - spec.band
    if z_hi - z_lo < hole_h:
        return None
    depth = layout.wall_below_lip() + LATTICE_BLIND
    x0, y0, x1, y1 = layout.body
    inset = layout.r_body + spec.band
    prisms: list[m3d.Manifold] = []
    panels = [
        # (u axis start, u axis end, function mapping (u, z) polygon to a 3D prism)
        ("-x", y0 + inset, y1 - inset),
        ("+x", y0 + inset, y1 - inset),
        ("-y", x0 + inset, x1 - inset),
        ("+y", x0 + inset, x1 - inset),
    ]
    su = hole_w + WEB
    sz = hole_h + WEB
    for side, u_lo, u_hi in panels:
        if u_hi - u_lo < hole_w:
            continue
        centres = _stagger(u_lo, u_hi, z_lo, z_hi, hole_w, hole_h, su, sz)
        for uc, zc in centres:
            section = poly + np.array([uc, zc])
            prism = m3d.Manifold.extrude(m3d.CrossSection([section]), depth + 1.0)
            # extrude gives the polygon in the XY plane rising along +Z: rotate so the
            # polygon's (u, z) becomes the panel plane and the extrusion runs through the wall
            if side == "-x":
                prism = prism.rotate([90.0, 0.0, 90.0]).translate([x0 - 1.0, 0.0, 0.0])
            elif side == "+x":
                prism = prism.rotate([90.0, 0.0, -90.0]).translate([x1 + 1.0, 0.0, 0.0])
            elif side == "-y":
                prism = prism.rotate([90.0, 0.0, 0.0]).translate([0.0, y0 - 1.0, 0.0])
            else:
                prism = prism.rotate([90.0, 0.0, 180.0]).translate([0.0, y1 + 1.0, 0.0])
            prisms.append(prism)
    if not prisms:
        return None
    return m3d.Manifold.compose(prisms)


def _stagger(u_lo, u_hi, z_lo, z_hi, hole_w, hole_h, su, sz) -> list[tuple[float, float]]:
    """Staggered rows of hole centres that keep every hole wholly inside the zone."""
    centres = []
    n_rows = int(np.floor((z_hi - z_lo - hole_h) / sz)) + 1
    z_start = z_lo + hole_h / 2.0 + ((z_hi - z_lo - hole_h) - (n_rows - 1) * sz) / 2.0
    for row in range(n_rows):
        zc = z_start + row * sz
        offset = su / 2.0 if row % 2 else 0.0
        n_cols = int(np.floor((u_hi - u_lo - hole_w - offset) / su)) + 1
        if n_cols <= 0:
            continue
        u_start = u_lo + hole_w / 2.0 + offset + ((u_hi - u_lo - hole_w - offset) - (n_cols - 1) * su) / 2.0
        for col in range(n_cols):
            centres.append((float(u_start + col * su), float(zc)))
    return centres
