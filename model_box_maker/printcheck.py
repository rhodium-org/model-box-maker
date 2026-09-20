"""Does a body print without support? (model-box-maker REQ-0017, CON-0001)

A downward-facing face may lean at most 45 degrees from vertical unless it lies
on the bed. A horizontal downward face is a bridge and is allowed when the
span it bridges is 5 mm or less.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

LEAN_LIMIT_DEG = 45.0
BRIDGE_LIMIT = 5.0  # mm
HORIZONTAL_DEG = 89.0
BED_TOL = 1e-4


@dataclass
class Violation:
    face: int
    reason: str


def check_printable(vertices: np.ndarray, faces: np.ndarray, bed_z: float = 0.0) -> list[Violation]:
    """Every face that would need support in the orientation given."""
    mesh = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces, np.int64), process=False)
    normals = mesh.face_normals
    lean = np.degrees(np.arcsin(np.clip(np.abs(normals[:, 2]), 0.0, 1.0)))
    down = normals[:, 2] < -1e-9
    z = mesh.vertices[mesh.faces][:, :, 2]
    on_bed = (np.abs(z - bed_z) <= BED_TOL).all(axis=1)
    area = mesh.area_faces
    violations: list[Violation] = []

    steep = down & ~on_bed & (lean > LEAN_LIMIT_DEG + 1e-6) & (lean <= HORIZONTAL_DEG) & (area > 1e-9)
    for idx in np.flatnonzero(steep):
        violations.append(Violation(int(idx), f"downward face leans {lean[idx]:.1f} degrees from vertical"))

    flat = down & ~on_bed & (lean > HORIZONTAL_DEG) & (area > 1e-9)
    if flat.any():
        flat_idx = np.flatnonzero(flat)
        sub = mesh.submesh([flat_idx], append=True, repair=False)
        components = trimesh.graph.connected_components(sub.face_adjacency, nodes=np.arange(len(sub.faces)))
        for comp in components:
            comp = np.asarray(comp)
            pts = sub.vertices[np.unique(sub.faces[comp])][:, :2]
            span = _min_width(pts)
            if span > BRIDGE_LIMIT + 1e-6:
                violations.append(Violation(int(flat_idx[comp[0]]), f"bridge spans {span:.2f} mm"))
    return violations


def _min_width(pts: np.ndarray) -> float:
    """Smallest caliper width of a 2D point set."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return float(np.linalg.norm(np.ptp(pts, axis=0))) if len(pts) else 0.0
    try:
        hull = ConvexHull(pts)
    except QhullError:
        d = np.ptp(pts, axis=0)
        return float(min(d))
    p = pts[hull.vertices]
    edges = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(edges, axis=1)
    best = np.inf
    for e, ln in zip(edges, lengths):
        if ln < 1e-12:
            continue
        n = np.array([-e[1], e[0]]) / ln
        width = float(np.ptp(p @ n))
        best = min(best, width)
    return float(best)
