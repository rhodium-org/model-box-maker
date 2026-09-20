"""Orientation: model-box-maker TEST-0007..TEST-0013."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from model_box_maker import BoxSpec, OrientationError, make_box
from model_box_maker.mesh_io import load_model
from model_box_maker.orient import evaluate, rotation_from_angles
from model_box_maker import raster


def test_0007_a_bowl_is_cradled_mouth_down(run_box):
    """TEST-0007: the shell's rim ends lowest and the cradle rises into a dome under the hollow."""
    result, _ = run_box("bowl")
    rest = result.model_at_rest()
    # the rim ring is the set of points at |xy| between 17 and 20 in the shell's own frame; in the
    # chosen pose the model's lowest points must be on the rim, not the dome apex
    z = rest[:, 2]
    lowest = rest[z <= z.min() + 0.5]
    radial = np.hypot(lowest[:, 0] - rest[:, 0].mean(), lowest[:, 1] - rest[:, 1].mean())
    assert radial.min() > 15.0, "the lowest points are not on the rim"
    cavity = result.cavity()
    g = cavity.grid
    cx, cy = rest[:, 0].mean() - result.layout.translation[0], rest[:, 1].mean() - result.layout.translation[1]
    i = int((cx - g.x0) / g.pitch)
    j = int((cy - g.y0) / g.pitch)
    dome = cavity.floor[i, j]
    assert cavity.mask[i, j]
    assert dome - cavity.floor_min >= 18.5 - result.spec.clearance - 1.0


def _axis_aligned_rotations():
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                m[row, col] = s
            if np.linalg.det(m) > 0:
                mats.append(m)
    assert len(mats) == 24
    return mats


@pytest.mark.parametrize("name", ["bowl", "wedge", "t_bracket", "mushroom"])
def test_0008_automatic_beats_every_axis_aligned_pose(run_box, fixture_path, name):
    """TEST-0008: the winner's outer volume is no larger than each axis-aligned pose, or ties with a smaller cavity."""
    result, _ = run_box(name)
    spec = result.spec
    winner = result.choice.winner
    model = load_model(fixture_path(name))
    mesh = raster.subdivide(model.vertices, model.faces, spec.pitch)
    band = 1.0 + spec.size_tolerance / 100.0
    for rot in _axis_aligned_rotations():
        other = evaluate(mesh, rot, spec, spec.pitch, exact=True, label="axis")
        assert (winner.outer_volume <= other.outer_volume * (1 + 1e-6)
                or (winner.outer_volume <= other.outer_volume * band
                    and winner.cavity_volume <= other.cavity_volume * (1 + 1e-6)))


def test_0009_a_plate_lies_flat(run_box):
    """TEST-0009: a 100x100x3 plate saved on edge is laid flat and the box is under 15 mm tall."""
    result, _ = run_box("plate_on_edge")
    rest = result.model_at_rest()
    assert np.ptp(rest[:, 2]) < 3.5
    assert result.layout.size[2] < 15.0


def test_0010_the_footprint_follows_the_model(run_box):
    """TEST-0010: a bar turned 30 degrees is boxed with its long side along a box side."""
    result, _ = run_box("bar30")
    rest = result.model_at_rest()
    extents = np.ptp(rest[:, :2], axis=0)
    assert abs(extents[0] - 80.0) < 1e-3 and abs(extents[1] - 10.0) < 1e-3, extents
    spec = result.spec
    w, d, _ = result.layout.size
    margin = 2 * (spec.clearance + spec.wall + spec.skirt + spec.fit)
    assert 80.0 + margin - 1e-6 <= w <= 80.0 + margin + 6 * spec.pitch + 1e-6
    assert 10.0 + margin - 1e-6 <= d <= 10.0 + margin + 6 * spec.pitch + 1e-6


def test_0011_a_fixed_orientation_is_used_exactly(run_box):
    """TEST-0011: keep reports identity; 90,0,0 reports that rotation; neither turns about the vertical."""
    keep, _ = run_box("bar30", orientation="keep", pitch=0.5)
    assert np.allclose(keep.choice.winner.rotation, np.eye(3), atol=1e-9)
    assert keep.report["orientation_mode"] == "keep"
    rest = keep.model_at_rest()
    extents = np.ptp(rest[:, :2], axis=0)
    assert extents[0] > 60 and extents[1] > 30, "keep must not turn the bar straight"
    given, _ = run_box("bar30", orientation="angles", angles=(90.0, 0.0, 0.0), pitch=0.5)
    expected = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    assert np.allclose(given.choice.winner.rotation, expected, atol=1e-6)
    assert np.allclose(given.report["rotation_matrix"], expected, atol=1e-6)
    assert given.report["orientation_mode"] == "angles"


def test_0012_the_report_is_complete_and_matches_the_geometry(run_box):
    """TEST-0012: every field of REQ-0006 is present and the transform places the model in the cavity."""
    result, paths = run_box("t_bracket")
    with open(paths[2], encoding="utf-8") as fh:
        report = json.load(fh)
    for key in ("input", "scale", "orientation_mode", "rotation_matrix", "rotation_euler_xyz_deg",
                "translation_mm", "outer_size_mm", "outer_volume_mm3", "cavity_volume_mm3",
                "free_volume_fraction", "options", "files", "candidates", "winner_reason"):
        assert key in report, key
    assert len(report["rotation_matrix"]) == 3 and len(report["candidates"]) >= 1
    assert report["free_volume_fraction"] is not None
    assert all(name in report["options"] for name in result.spec.as_dict())
    rot = np.array(report["rotation_matrix"])
    t = np.array(report["translation_mm"])
    placed = result.model.vertices @ rot.T + t
    cavity = result.cavity()
    g = cavity.grid
    tx, ty, tz = result.layout.translation
    i = np.floor((placed[:, 0] - tx - g.x0) / g.pitch).astype(int)
    j = np.floor((placed[:, 1] - ty - g.y0) / g.pitch).astype(int)
    assert cavity.mask[i, j].all(), "a vertex lies outside the cavity outline"
    floor_under = cavity.floor[i, j] + tz
    assert (placed[:, 2] - floor_under >= result.spec.clearance - result.spec.pitch - 1e-9).all()


def test_0013_a_size_limit_changes_the_choice_or_fails_plainly(run_box, fixture_path):
    """TEST-0013: a limit only the flat pose meets lays the plate flat; an impossible limit exits with the smallest box."""
    result, _ = run_box("plate_on_edge", max_outer=(120.0, 120.0, 20.0), pitch=0.5)
    assert result.layout.size[2] <= 20.0
    model = load_model(fixture_path("plate_on_edge"))
    with pytest.raises(OrientationError) as excinfo:
        make_box(model, BoxSpec(pitch=0.5), max_outer=(50.0, 50.0, 50.0))
    message = str(excinfo.value)
    assert "smallest box" in message and "x" in message
