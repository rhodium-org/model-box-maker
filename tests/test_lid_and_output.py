"""Lid, lattice, files, printability, preview, determinism, CLI, NFRs:
model-box-maker TEST-0021..TEST-0032."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

import numpy as np
import pytest
import trimesh
import xml.etree.ElementTree as etree

from model_box_maker import printcheck
from model_box_maker.geometry import (LATTICE_BESIDE, LATTICE_BLIND, LATTICE_OUTSIDE, boundary_distance,
                                      cradle_beside, hole_polygon, section_polygons, signed_distance)
from model_box_maker.mesh_io import load_body
from scipy.spatial import cKDTree
from tests.conftest import ROOT, densify, intersects, manifold_of, translated, trimesh_of


def _seated(result):
    return result.seated_lid()


@pytest.mark.parametrize("name,kwargs", [("cube", {}), ("l_bracket_standing", {"orientation": "keep"})])
def test_0021_the_lid_seats_on_the_rim_and_stops_there(run_box, name, kwargs):
    """TEST-0021: plate underside on the rim plane, skirt the fit gap from the lip, flush outside, and blocked below."""
    result, _ = run_box(name, pitch=0.5, **kwargs)
    spec, layout = result.spec, result.layout
    lid_v, lid_f = _seated(result)
    lid = trimesh.Trimesh(lid_v, lid_f, process=False)
    # plate underside: the lowest face pointing down that spans the footprint sits on the rim plane
    down = lid.face_normals[:, 2] < -0.999
    plate_z = lid.vertices[np.unique(lid.faces[down])][:, 2]
    inner_plate = plate_z[np.abs(plate_z - layout.rim_z) < 0.5]
    assert len(inner_plate) and np.abs(inner_plate - layout.rim_z).max() < 0.01
    # skirt inner face is fit from the lip; outer face flush with the body wall
    v = lid.vertices
    skirt_bottom = layout.rim_z - spec.skirt_height
    skirt_band = v[np.abs(v[:, 2] - skirt_bottom) < 1e-6]
    d_lip = signed_distance(skirt_band, layout.lip_section())
    d_body = np.abs(signed_distance(skirt_band, layout.body_section()))
    inner = skirt_band[np.abs(d_lip - spec.fit) < 0.05]
    outer = skirt_band[d_body < 0.05]
    assert abs(skirt_bottom - (layout.shoulder_z + spec.fit)) < 1e-9
    assert len(inner) > 8 and len(outer) > 8
    assert len(inner) + len(outer) >= len(skirt_band) - 2
    # lowering the seated lid by 0.5 mm makes it intersect the base
    base = manifold_of(result.base)
    assert intersects(base, manifold_of(translated((lid_v, lid_f), [0, 0, -0.5])))
    assert not intersects(base, manifold_of((lid_v, lid_f)))


def test_0022_the_lid_lifts_straight_off(run_box):
    """TEST-0022: the seated lid raised by 0.5 mm, 3 mm and the lip height never meets the base."""
    result, _ = run_box("cube", pitch=0.5)
    base = manifold_of(result.base)
    seated = _seated(result)
    for lift in (0.5, 3.0, result.spec.lip):
        assert not intersects(base, manifold_of(translated(seated, [0, 0, lift]))), lift


def test_0023_the_seated_lid_clears_the_model(run_box):
    """TEST-0023: seated lid to model distance is at least top space - 0.05; lid below the rim only outside the lip."""
    result, _ = run_box("mushroom")
    spec, layout = result.spec, result.layout
    lid_v, lid_f = _seated(result)
    lid = trimesh.Trimesh(lid_v, lid_f, process=False)
    rest = trimesh.Trimesh(result.model_at_rest(), result.model.faces, process=False)
    pts = rest.sample(2000, seed=1)
    _, dist, _ = trimesh.proximity.closest_point(lid, pts)
    assert dist.min() >= spec.top_space - 0.05
    below = lid_v[lid_v[:, 2] < layout.rim_z - 1e-6]
    assert len(below)
    outside = signed_distance(below, layout.lip_section())
    assert outside.min() >= spec.fit - 0.02


def _cavity_cells(result):
    cav = result.cavity()
    g = cav.grid
    mi, mj = np.nonzero(cav.mask)
    t = result.layout.translation
    xy = np.column_stack([g.x_edge(mi) + g.pitch / 2 + t[0], g.y_edge(mj) + g.pitch / 2 + t[1]])
    return xy, cav.floor[mi, mj] + t[2], cKDTree(xy)


def _removed_pieces(solid, patterned):
    removed = manifold_of(solid.base) - manifold_of(patterned.base)
    pieces = []
    for piece in removed.decompose():
        mesh = piece.to_mesh64()
        pieces.append(np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3])
    return removed, pieces


@pytest.mark.parametrize("pattern", ["hex", "diamond"])
def test_0024_a_lattice_takes_material_only_from_the_panels(run_box, pattern):
    """TEST-0024: patterned within solid; removed material within wall-max plus the cut depth of the cavity,
    above the cradle beside it plus the band, below the lip band; holes within max-hole; printable."""
    solid, _ = run_box("cube60", walls="solid", pitch=0.5)
    patterned, _ = run_box("cube60", walls=pattern, pitch=0.5)
    spec, layout = patterned.spec, patterned.layout
    assert (manifold_of(patterned.base) - manifold_of(solid.base)).volume() < 1e-6, "patterned base sticks out"
    removed, pieces = _removed_pieces(solid, patterned)
    assert removed.volume() > 100.0 and len(pieces) >= 8
    poly = hole_polygon(pattern, spec.max_hole)
    hole_w, hole_h = float(np.ptp(poly[:, 0])), float(np.ptp(poly[:, 1]))
    cells_xy, cells_z, tree = _cavity_cells(patterned)
    holes = layout.lattice
    assert len(holes) >= len(pieces)
    reach = spec.wall_max + LATTICE_BLIND + LATTICE_OUTSIDE + 0.05
    for pv in pieces:
        assert boundary_distance(pv, layout.cavity_section).max() <= reach
        assert pv[:, 2].max() <= layout.shoulder_z - spec.band + 1e-6
        c = pv.mean(axis=0)
        hole = min(holes, key=lambda h: np.hypot(*(h["point"] - c[:2])))
        p, n = hole["point"], hole["normal"]
        beside = cradle_beside(cells_xy, cells_z, tree, p, n, hole_w / 2 + LATTICE_BESIDE,
                               spec.wall_max + LATTICE_BLIND + LATTICE_BESIDE)
        z_low = (layout.floor_z if beside is None else beside) + spec.band
        assert pv[:, 2].min() >= z_low - 1e-6
        t = np.array([-n[1], n[0]])
        assert np.ptp(pv[:, :2] @ t) <= spec.max_hole + 0.1
        assert np.ptp(pv[:, 2]) <= spec.max_hole + 0.1
    assert printcheck.check_printable(*patterned.base) == []


def test_0035_a_tall_cradle_on_one_side_does_not_stop_holes_elsewhere(run_box):
    """TEST-0035: a post with an arm from its top: holes beside the post where the cradle is low, none beside
    the arm where the cradle rises, each cut along the normal."""
    solid, _ = run_box("flag", orientation="keep", pitch=0.5)
    patterned, _ = run_box("flag", orientation="keep", pitch=0.5, walls="hex")
    spec, layout = patterned.spec, patterned.layout
    t = layout.translation
    _, pieces = _removed_pieces(solid, patterned)
    assert pieces, "no holes were cut"
    centres = np.array([pv.mean(axis=0) for pv in pieces])
    pole_far_x = 20.0 + t[0]
    arm_start_x = 20.0 + t[0] + 10.0
    assert (centres[:, 0] < pole_far_x).sum() >= 2, "no holes beside the post"
    assert not (centres[:, 0] > arm_start_x).any(), "a hole beside the arm"
    for pv, hole in zip(pieces, [min(layout.lattice, key=lambda h: np.hypot(*(h["point"] - pv.mean(axis=0)[:2]))) for pv in pieces]):
        n = hole["normal"]
        along_n = np.ptp(pv[:, :2] @ n)
        assert spec.wall + spec.body_offset - 0.1 <= along_n <= spec.wall_max + LATTICE_BLIND + 0.1
        assert np.ptp(pv[:, :2] @ np.array([-n[1], n[0]])) <= spec.max_hole + 0.1


def test_0025_solid_is_the_default(run_box):
    """TEST-0025: no --walls equals --walls solid byte for byte; hex removes volume."""
    default, paths_default = run_box("cube60", pitch=0.5)
    solid, paths_solid = run_box("cube60", walls="solid", pitch=0.5)
    hexed, _ = run_box("cube60", walls="hex", pitch=0.5)
    assert open(paths_default[0], "rb").read() == open(paths_solid[0], "rb").read()
    assert abs(trimesh_of(hexed.base).volume) < abs(trimesh_of(solid.base).volume)


def test_0026_two_closed_bodies_each_the_way_it_prints(run_box, fixture_path, tmp_path, cli):
    """TEST-0026: watertight bodies; base opening up on z=0; lid plate on z=0 with the skirt rising; 3MF too."""
    result, paths = run_box("t_bracket")
    base = load_body(paths[0])
    lid = load_body(paths[1])
    for body in (base, lid):
        assert body.is_watertight and body.is_winding_consistent and body.volume > 0
        assert abs(body.bounds[0][2]) < 1e-6
    rim = result.layout.rim_z
    top_faces = base.face_normals[:, 2] > 0.999
    assert np.any(np.abs(base.vertices[np.unique(base.faces[top_faces])][:, 2] - rim) < 1e-6)
    inside = base.contains(np.array([[result.layout.size[0] / 2, result.layout.size[1] / 2, rim - 0.5]]))
    assert not inside[0], "the cavity is not open at the rim"
    assert lid.bounds[1][2] > result.spec.lid_plate + 1e-6
    assert abs(lid.bounds[1][2] - (result.spec.lid_plate + result.spec.skirt_height)) < 1e-6
    out = tmp_path / "threemf"
    proc = cli(fixture_path("cube"), "-o", str(out), "--format", "3mf", "--pitch", "0.5")
    assert proc.returncode == 0, proc.stderr
    for name in ("cube-base.3mf", "cube-lid.3mf"):
        body = load_body(str(out / name))
        assert body.is_watertight and body.volume > 0


def test_0027_no_downward_face_steeper_than_45_degrees_off_the_bed(run_box):
    """TEST-0027: no offending face on either body; lattice bridges within 5 mm."""
    for name, kwargs in (("mushroom", {}), ("cube60", {"walls": "hex", "pitch": 0.5}), ("cube60", {"walls": "diamond", "pitch": 0.5})):
        result, _ = run_box(name, **kwargs)
        for body in (result.base, result.lid):
            assert printcheck.check_printable(*body) == []
    # the checker itself catches a real overhang
    v = np.array([[0, 0, 5], [10, 0, 5], [0, 10, 5], [0, 0, 6], [10, 0, 6], [0, 10, 6]], dtype=float)
    f = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]])
    assert printcheck.check_printable(v, f), "a 10 mm bridge slab off the bed must be reported"


def test_0028_the_preview_holds_base_lid_and_model(run_box):
    """TEST-0028: three named objects; the model object matches the report transform; lid wholly above the base."""
    result, paths = run_box("t_bracket")
    preview = paths[3]
    with zipfile.ZipFile(preview) as zf:
        root = etree.fromstring(zf.read("3D/3dmodel.model"))
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    names = [obj.get("name") for obj in root.findall(".//m:object", ns)]
    assert names == ["base", "lid", "model"]
    scene = trimesh.load(preview)
    geoms = {k: v for k, v in scene.geometry.items()}
    assert set(geoms) == {"base", "lid", "model"}
    expected = trimesh.Trimesh(result.model_at_rest(), result.model.faces, process=False)
    got = geoms["model"]
    assert np.allclose(np.sort(got.vertices, axis=0), np.sort(expected.vertices, axis=0), atol=1e-4)
    assert geoms["lid"].bounds[0][2] > geoms["base"].bounds[1][2]


def test_0029_a_rerun_is_byte_identical(fixture_path, tmp_path, cli):
    """TEST-0029: two runs produce identical hashes for every file."""
    hashes = []
    for k in range(2):
        out = tmp_path / f"run{k}"
        proc = cli(fixture_path("mushroom"), "-o", str(out), "--preview", "--pitch", "0.5")
        assert proc.returncode == 0, proc.stderr
        hashes.append({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())})
    assert hashes[0] == hashes[1]
    assert len(hashes[0]) == 4


def test_0030_exit_codes_and_messages(fixture_path, tmp_path, cli):
    """TEST-0030: 0 with paths and a pose line; 2 for a missing model; 1 for a refused model; nothing left behind."""
    out = tmp_path / "good"
    proc = cli(fixture_path("cube"), "-o", str(out), "--pitch", "0.5")
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 4 and lines[-1].startswith("pose:") and "box" in lines[-1]
    for line in lines[:3]:
        assert os.path.exists(line)
    proc = cli()
    assert proc.returncode == 2 and "usage" in proc.stderr
    bad = tmp_path / "bad.stl"
    bad.write_bytes(b"not a mesh at all")
    out2 = tmp_path / "bad-out"
    proc = cli(str(bad), "-o", str(out2))
    assert proc.returncode == 1
    assert len([ln for ln in proc.stderr.splitlines() if ln.strip()]) == 1
    assert not out2.exists() or not any(out2.iterdir())


def test_0031_200000_triangles_within_a_minute(fixture_path, tmp_path, cli):
    """TEST-0031 (NFR-0001): a 200 000-triangle, 150 mm fixture at default options in under 60 s."""
    path = fixture_path("torus200k")
    mesh = trimesh.load(path)
    assert len(mesh.faces) == 200_000 and abs(np.ptp(mesh.vertices[:, 0]) - 150.0) < 1e-6
    started = time.time()
    proc = cli(path, "-o", str(tmp_path / "big"))
    elapsed = time.time() - started
    assert proc.returncode == 0, proc.stderr
    assert elapsed < 60.0, elapsed


@pytest.mark.skipif(not os.environ.get("MODEL_BOX_MAKER_INSTALL_TEST"),
                    reason="set MODEL_BOX_MAKER_INSTALL_TEST=1 to build a fresh venv (needs the network)")
def test_0032_installs_into_a_fresh_environment(tmp_path):
    """TEST-0032 (NFR-0002): pip install into a clean venv with wheels only, then --help runs."""
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    pip = venv / "bin" / "pip"
    subprocess.run([str(pip), "install", "--only-binary", ":all:", ROOT], check=True)
    proc = subprocess.run([str(venv / "bin" / "model-box-maker"), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0 and "--clearance" in proc.stdout
