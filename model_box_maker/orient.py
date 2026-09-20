"""Choosing how the model lies in its box.

Implements model-box-maker REQ-0004 (automatic orientation: candidates, yaw for
the smallest footprint, outer volume first, cavity volume within the tie band,
then the lowest box), REQ-0005 (--orientation keep or RX,RY,RZ used exactly),
REQ-0007 (--max-outer drops poses that fit no way round) and NFR-0001 (screen
at a coarse pitch, finish at the real one).
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from . import raster
from .geometry import Exterior, build_exterior, fit_lip_rect
from .spec import BoxSpec

HULL_FACE_CAP = 30
HULL_MERGE_DEG = 8.0
DOWN_MERGE_DEG = 0.5
SCREEN_PITCH = 1.0
FINALIST_CAP = 4


class OrientationError(ValueError):
    """No pose fits the size limit; the message names the smallest box achieved."""


@dataclass
class Candidate:
    label: str
    down: np.ndarray  # unit vector in model coordinates that will point to -Z
    rotation: np.ndarray = field(default=None)  # set once the yaw is chosen


@dataclass
class Evaluation:
    """One pose, scored. Lengths in mm, volumes in mm^3, pose coordinates."""

    label: str
    rotation: np.ndarray
    cavity: raster.Cavity
    lip_rect: tuple[float, float, float, float]
    rim_z: float
    size: tuple[float, float, float]  # closed box W, D, H
    outer_volume: float
    cavity_volume: float
    fits: bool = True
    stage: str = "final"
    index: int = 0  # position in the candidate list; the saved pose is 0
    exterior: Exterior = None  # plan profiles in pose coordinates (REQ-0012, REQ-0021)

    @property
    def height(self) -> float:
        return self.size[2]

    def euler_xyz(self) -> list[float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # gimbal lock only changes which of two equal answers is printed
            return [float(a) for a in Rotation.from_matrix(self.rotation).as_euler("xyz", degrees=True)]


# ----------------------------------------------------------------- rotations


def rotation_taking(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rotation matrix R with R @ a parallel to b (Rodrigues)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / s ** 2)


def rot_z(deg: float) -> np.ndarray:
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


def rotation_from_angles(rx: float, ry: float, rz: float) -> np.ndarray:
    """About X, then Y, then Z (REQ-0005)."""
    return Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()


# ----------------------------------------------------------------- candidates


def candidate_downs(vertices: np.ndarray) -> tuple[list[Candidate], np.ndarray]:
    """The model as saved, every convex-hull face down, each principal axis vertical.

    Also returns the hull vertices, enough to find every pose's footprint.
    """
    downs: list[Candidate] = [Candidate("saved", np.array([0.0, 0.0, -1.0]))]
    hull_points = vertices
    try:
        hull = ConvexHull(vertices)
        hull_points = vertices[hull.vertices]
        normals = hull.equations[:, :3]
        pts = vertices[hull.simplices]
        areas = 0.5 * np.linalg.norm(np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]), axis=1)
        order = np.argsort(-areas, kind="stable")
        kept: list[np.ndarray] = []
        cos_merge = np.cos(np.radians(HULL_MERGE_DEG))
        for idx in order:
            n = normals[idx] / np.linalg.norm(normals[idx])
            if kept and float(np.max(np.asarray(kept) @ n)) > cos_merge:
                continue
            kept.append(n)
            downs.append(Candidate(f"hull face {len(kept)}", n))
            if len(kept) >= HULL_FACE_CAP:
                break
    except QhullError:
        pass
    centred = vertices - vertices.mean(axis=0)
    cov = centred.T @ centred / max(len(vertices), 1)
    _, vectors = np.linalg.eigh(cov)
    for rank, k in enumerate((2, 1, 0)):
        e = vectors[:, k]
        e = e * np.sign(e[np.argmax(np.abs(e))] or 1.0)
        downs.append(Candidate(f"principal axis {rank + 1} down", e))
        downs.append(Candidate(f"principal axis {rank + 1} up", -e))
    unique: list[Candidate] = []
    cos_same = np.cos(np.radians(DOWN_MERGE_DEG))
    for cand in downs:
        d = cand.down / np.linalg.norm(cand.down)
        if any(float(np.dot(d, u.down)) > cos_same for u in unique):
            continue
        cand.down = d
        unique.append(cand)
    return unique, hull_points


def min_footprint_yaw(xy: np.ndarray) -> float:
    """Degrees to turn about Z so the smallest bounding rectangle is axis-aligned.

    The rectangle's long side ends along X. Ties go to the smallest angle.
    """
    pts = np.asarray(xy, float)
    try:
        hull = ConvexHull(pts)
        pts = pts[hull.vertices]
    except QhullError:
        return 0.0
    edges = np.roll(pts, -1, axis=0) - pts
    angles = np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2)
    best = None
    for theta in sorted(set(np.round(angles, 12).tolist())):
        c, s = np.cos(-theta), np.sin(-theta)
        rx = pts[:, 0] * c - pts[:, 1] * s
        ry = pts[:, 0] * s + pts[:, 1] * c
        w = float(np.ptp(rx))
        d = float(np.ptp(ry))
        key = (round(w * d, 9), theta)
        if best is None or key < best[0]:
            best = (key, theta, w, d)
    _, theta, w, d = best
    if w < d:
        theta += np.pi / 2
    return float(np.degrees(-theta))


# ----------------------------------------------------------------- scoring


def evaluate(mesh: tuple[np.ndarray, np.ndarray], rotation: np.ndarray, spec: BoxSpec, pitch: float,
             exact: bool, label: str, stage: str = "final") -> Evaluation:
    posed = raster.posed_triangles(mesh[0], mesh[1], rotation)
    cavity = raster.cavity_for(posed, pitch, spec.clearance, exact)
    rim_z = cavity.rim_z(spec.top_space)
    lip_rect = fit_lip_rect(cavity, spec.wall, spec.lip_radius)
    height = spec.floor + (rim_z - cavity.floor_min) + spec.lid_plate
    exterior = build_exterior(cavity, lip_rect, spec, height)
    bx0, by0, bx1, by1 = exterior.body.bounds()
    width = bx1 - bx0
    depth = by1 - by0
    return Evaluation(
        label=label,
        rotation=rotation,
        cavity=cavity,
        lip_rect=lip_rect,
        rim_z=rim_z,
        size=(float(width), float(depth), float(height)),
        outer_volume=float(exterior.footprint_area * height),
        cavity_volume=cavity.cavity_volume(rim_z),
        stage=stage,
        exterior=exterior,
    )


def fits_limit(size: tuple[float, float, float], limit: tuple[float, float, float] | None) -> bool:
    """True when some pairing of the three sides lies within the limit (REQ-0007)."""
    if limit is None:
        return True
    for perm in itertools.permutations(size):
        if all(p <= q + 1e-9 for p, q in zip(perm, limit)):
            return True
    return False


def pick(evaluations: list[Evaluation], spec: BoxSpec) -> tuple[Evaluation, list[Evaluation], str]:
    """Apply the rule of REQ-0004 to fitting evaluations; returns winner, ties, reason."""
    fitting = [e for e in evaluations if e.fits]
    best = min(e.outer_volume for e in fitting)
    band = best * (1.0 + spec.size_tolerance / 100.0)
    ties = [e for e in fitting if e.outer_volume <= band + 1e-9]
    ordered = sorted(
        range(len(ties)),
        key=lambda i: (round(ties[i].cavity_volume, 6), round(ties[i].height, 6), ties[i].index, i),
    )
    winner = ties[ordered[0]]
    if len(ties) == 1:
        reason = "smallest outer volume of every candidate pose"
    else:
        reason = (f"{len(ties)} poses lie within {spec.size_tolerance:g} percent of the smallest "
                  f"outer volume; this one has the smallest cavity volume of them")
        same_cavity = [e for e in ties if abs(e.cavity_volume - winner.cavity_volume) < 1e-6]
        if len(same_cavity) > 1:
            reason += " and is the lowest box"
    return winner, ties, reason


@dataclass
class Choice:
    winner: Evaluation
    evaluations: list[Evaluation]
    reason: str
    mode: str


def choose(vertices: np.ndarray, faces: np.ndarray, spec: BoxSpec, mode: str,
           angles: tuple[float, float, float] | None,
           max_outer: tuple[float, float, float] | None) -> Choice:
    """Pick the pose. ``mode`` is 'auto', 'keep' or 'angles'."""
    pitch = spec.pitch
    if mode != "auto":
        rotation = np.eye(3) if mode == "keep" else rotation_from_angles(*angles)
        mesh = raster.subdivide(vertices, faces, pitch)
        label = "as saved" if mode == "keep" else "given angles"
        ev = evaluate(mesh, rotation, spec, pitch, exact=True, label=label)
        ev.fits = fits_limit(ev.size, max_outer)
        if not ev.fits:
            raise OrientationError(_limit_message([ev], max_outer))
        return Choice(ev, [ev], "orientation fixed by --orientation", mode)

    screen_pitch = max(pitch, SCREEN_PITCH)
    mesh_screen = raster.subdivide(vertices, faces, screen_pitch)
    candidates, hull_points = candidate_downs(vertices)
    screened: list[Evaluation] = []
    for index, cand in enumerate(candidates):
        r_face = rotation_taking(cand.down, [0.0, 0.0, -1.0])
        xy = (hull_points @ r_face.T)[:, :2]
        yaw = min_footprint_yaw(xy)
        cand.rotation = rot_z(yaw) @ r_face
        ev = evaluate(mesh_screen, cand.rotation, spec, screen_pitch, exact=False,
                      label=cand.label, stage="screened")
        ev.fits = fits_limit(ev.size, max_outer)
        ev.index = index
        screened.append(ev)
    if not any(e.fits for e in screened):
        raise OrientationError(_limit_message(screened, max_outer))

    fitting = [e for e in screened if e.fits]
    best = min(e.outer_volume for e in fitting)
    band = best * (1.0 + spec.size_tolerance / 100.0) * 1.05
    shortlist = sorted(
        (e for e in fitting if e.outer_volume <= band),
        key=lambda e: (round(e.outer_volume, 6), round(e.cavity_volume, 6), round(e.height, 6), e.index),
    )[:FINALIST_CAP]

    mesh_fine = mesh_screen if screen_pitch == pitch else raster.subdivide(vertices, faces, pitch)
    finals = []
    for e in shortlist:
        fine = evaluate(mesh_fine, e.rotation, spec, pitch, exact=False, label=e.label, stage="final")
        fine.fits = fits_limit(fine.size, max_outer)
        fine.index = e.index
        finals.append(fine)
    if not any(f.fits for f in finals):
        raise OrientationError(_limit_message(finals, max_outer))
    chosen, _, reason = pick(finals, spec)
    # the winner is rasterised once more with the exact cell test: that cavity is the one built
    winner = evaluate(mesh_fine, chosen.rotation, spec, pitch, exact=True, label=chosen.label, stage="final")
    winner.fits = chosen.fits
    winner.index = chosen.index
    finals = [winner if f is chosen else f for f in finals]
    others = [e for e in screened if e.label not in {f.label for f in finals}]
    others.sort(key=lambda e: (round(e.outer_volume, 6), e.index))
    ordered = [winner] + [f for f in finals if f is not winner] + others
    return Choice(winner, ordered, reason, "auto")


def _limit_message(evaluations: list[Evaluation], limit) -> str:
    smallest = min(evaluations, key=lambda e: e.outer_volume)
    w, d, h = smallest.size
    return (f"no orientation fits within --max-outer {limit[0]:g},{limit[1]:g},{limit[2]:g} mm; "
            f"the smallest box achieved is {w:.1f} x {d:.1f} x {h:.1f} mm ({smallest.label})")
