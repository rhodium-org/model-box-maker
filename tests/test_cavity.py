"""Cavity, floor and walls: model-box-maker TEST-0014..TEST-0020."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from model_box_maker.geometry import rounded_rect_sdf
from tests.conftest import intersects, manifold_of, translated, trimesh_of


def _surface_samples(result, count=2000, seed=0):
    rest = trimesh.Trimesh(result.model_at_rest(), result.model.faces, process=False)
    return rest.sample(count, seed=seed)


def test_0014_the_clearance_holds_everywhere(run_box):
    """TEST-0014: 2000 surface points at rest are at least clearance - pitch from the base body."""
    result, _ = run_box("mushroom")
    pts = _surface_samples(result)
    base = trimesh_of(result.base)
    _, dist, _ = trimesh.proximity.closest_point(base, pts)
    assert dist.min() >= result.spec.clearance - result.spec.pitch - 1e-6, dist.min()


def test_0015_the_model_lifts_straight_out(run_box):
    """TEST-0015: lifting the model by 0.5, 2, 10 mm and above the rim never meets the base."""
    result, _ = run_box("mushroom")
    base = manifold_of(result.base)
    rest = (result.model_at_rest(), result.model.faces)
    lifts = [0.5, 2.0, 10.0, result.layout.rim_z - result.layout.floor_z + 5.0]
    for lift in lifts:
        moved = manifold_of(translated(rest, [0.0, 0.0, lift]))
        assert not intersects(base, moved), lift


@pytest.mark.parametrize("pitch", [0.25, 0.5, 1.0])
def test_0016_a_thin_fin_keeps_its_slot(run_box, pitch):
    """TEST-0016: a 0.6 mm fin hanging below a block still clears the base at three pitches."""
    result, _ = run_box("fin_block", orientation="keep", pitch=pitch)
    pts = _surface_samples(result, 3000)
    base = trimesh_of(result.base)
    _, dist, _ = trimesh.proximity.closest_point(base, pts)
    assert dist.min() >= result.spec.clearance - pitch - 1e-6, (pitch, dist.min())
    rest = result.model_at_rest()
    assert np.ptp(rest[:, 2]) > 17.0, "the fin must hang below the block in the kept pose"


def test_0017_a_coarser_grid_never_cuts_inward(run_box):
    """TEST-0017: the coarse outline contains the fine outline, and over the model's own footprint the
    coarse cradle floor is at or below the fine one."""
    fine, _ = run_box("t_bracket", orientation="keep", pitch=0.25)
    coarse, _ = run_box("t_bracket", orientation="keep", pitch=1.0)
    fc, cc = fine.cavity(), coarse.cavity()
    tz_f, tz_c = fine.layout.translation[2], coarse.layout.translation[2]
    ii, jj = np.nonzero(fc.mask)
    x = fc.grid.x_edge(ii) + fc.grid.pitch / 2
    y = fc.grid.y_edge(jj) + fc.grid.pitch / 2
    ci = np.floor((x - cc.grid.x0) / cc.grid.pitch).astype(int)
    cj = np.floor((y - cc.grid.y0) / cc.grid.pitch).astype(int)
    inside = (ci >= 0) & (ci < cc.mask.shape[0]) & (cj >= 0) & (cj < cc.mask.shape[1])
    assert inside.all() and cc.mask[ci, cj].all(), "a fine cavity cell lies outside the coarse outline"
    # compare floors in pose coordinates (both poses are the kept one) over the model's footprint
    footprint = fc.touched[ii, jj]
    assert footprint.sum() > 1000
    assert (cc.floor[ci, cj][footprint] <= fc.floor[ii, jj][footprint] + 1e-9).all()
    assert abs(tz_f - fine.spec.floor + fc.floor_min) < 1e-9 and abs(tz_c - coarse.spec.floor + cc.floor_min) < 1e-9


def test_0018_the_floor_is_exactly_the_floor_thickness_at_its_thinnest(run_box):
    """TEST-0018: lowest cradle point is floor above the underside; rays meet solid with no gap."""
    result, _ = run_box("t_bracket")
    base = trimesh_of(result.base)
    cavity = result.cavity()
    tz = result.layout.translation[2]
    assert abs((cavity.floor_min + tz) - result.spec.floor) < 0.01
    assert abs(base.bounds[0][2]) < 1e-6
    rng = np.random.default_rng(0)
    ii, jj = np.nonzero(cavity.mask)
    pick = rng.choice(len(ii), 200, replace=False)
    g = cavity.grid
    x = g.x_edge(ii[pick]) + g.pitch / 2 + result.layout.translation[0]
    y = g.y_edge(jj[pick]) + g.pitch / 2 + result.layout.translation[1]
    origins = np.column_stack([x, y, np.full(len(pick), -1.0)])
    directions = np.tile([0.0, 0.0, 1.0], (len(pick), 1))
    locations, ray_ids, _ = base.ray.intersects_location(origins, directions, multiple_hits=True)
    for k in range(len(pick)):
        hits = np.sort(locations[ray_ids == k][:, 2])
        assert len(hits) >= 2, "ray missed the base"
        assert abs(hits[0]) < 1e-6
        # the second crossing is the cradle surface: never above the cell's recorded floor (the mesh
        # takes the lowest of the surrounding cells) and never below the floor thickness
        floor_here = cavity.floor[ii[pick[k]], jj[pick[k]]] + tz
        assert result.spec.floor - 1e-6 <= hits[1] <= floor_here + 1e-6, (hits[:2], floor_here)


@pytest.mark.parametrize("radius", [0.0, 3.0, 10.0])
def test_0019_the_wall_is_never_thinner_than_the_wall_thickness(run_box, radius):
    """TEST-0019: the outline to the outer face is at least wall - 0.01 all round and at the lip."""
    result, _ = run_box("cube", corner_radius=radius, pitch=0.5)
    cavity = result.cavity()
    layout = result.layout
    tx, ty, _ = layout.translation
    from scipy import ndimage as ndi
    boundary = cavity.mask & ~ndi.binary_erosion(cavity.mask, structure=np.ones((3, 3), bool), border_value=0)
    ii, jj = np.nonzero(boundary)
    g = cavity.grid
    xs = np.concatenate([g.x_edge(ii), g.x_edge(ii + 1), g.x_edge(ii), g.x_edge(ii + 1)]) + tx
    ys = np.concatenate([g.y_edge(jj), g.y_edge(jj), g.y_edge(jj + 1), g.y_edge(jj + 1)]) + ty
    at_lip = -rounded_rect_sdf(xs, ys, *layout.lip, layout.r_lip)
    below_lip = -rounded_rect_sdf(xs, ys, *layout.body, layout.r_body)
    assert at_lip.min() >= result.spec.wall - 0.01
    assert below_lip.min() >= result.spec.wall - 0.01


def test_0020_the_outside_is_a_rounded_cuboid(run_box):
    """TEST-0020: rounded rectangle outline with the corner radius, sharp at zero; flat underside and lid top."""
    for radius in (0.0, 3.0):
        result, _ = run_box("cube", corner_radius=radius, pitch=0.5)
        base = trimesh_of(result.base)
        lid = trimesh_of(result.lid)
        w, d, _ = result.layout.size
        # the outline: every underside vertex that also belongs to a side face lies on the rounded
        # rectangle (a flat face may keep interior vertices from the body-and-lip union)
        v = base.vertices
        side = np.abs(base.face_normals[:, 2]) < 0.5
        on_side = np.zeros(len(v), dtype=bool)
        on_side[np.unique(base.faces[side])] = True
        ring = v[(np.abs(v[:, 2]) < 1e-6) & on_side]
        assert len(ring) >= 4
        sdf = rounded_rect_sdf(ring[:, 0], ring[:, 1], 0.0, 0.0, w, d, radius)
        assert np.abs(sdf).max() < 0.02, radius
        if radius == 0:
            corners = [(0, 0), (w, 0), (0, d), (w, d)]
            for cx, cy in corners:
                assert np.any(np.hypot(v[:, 0] - cx, v[:, 1] - cy) < 1e-6), "sharp corner missing"
        assert abs(base.bounds[0][2]) < 1e-9 and abs(lid.bounds[0][2]) < 1e-9
        underside = base.vertices[np.abs(base.vertices[:, 2]) < 1e-6]
        assert len(underside) >= 4
        top = lid.face_normals[:, 2] < -0.999
        assert np.allclose(lid.vertices[np.unique(lid.faces[top])][:, 2], 0.0, atol=1e-6)
