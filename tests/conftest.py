"""Fixtures: synthetic models built with trimesh/manifold3d, and a cached box runner.

Every test names the graph item it implements (model-box-maker TEST-xxxx).
"""

from __future__ import annotations

import os
import subprocess
import sys

import manifold3d as m3d
import numpy as np
import pytest
import trimesh

from model_box_maker import BoxSpec, make_box, write_outputs
from model_box_maker.mesh_io import load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _manifold_to_trimesh(solid: m3d.Manifold) -> trimesh.Trimesh:
    mesh = solid.to_mesh64()
    v = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    f = np.asarray(mesh.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(v, f, process=True)


def rotated(mesh: trimesh.Trimesh, axis: str, degrees: float) -> trimesh.Trimesh:
    m = mesh.copy()
    vec = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(degrees), vec))
    return m


def build_fixture(name: str) -> trimesh.Trimesh:
    """Synthetic models, all in millimetres."""
    if name == "cube":
        return trimesh.creation.box([30.0, 20.0, 10.0])
    if name == "unit_cube":
        return trimesh.creation.box([1.0, 1.0, 1.0])
    if name == "cube2":
        return trimesh.creation.box([2.0, 2.0, 2.0])
    if name == "cube60":
        return trimesh.creation.box([60.0, 60.0, 60.0])
    if name == "cube500":
        return trimesh.creation.box([500.0, 500.0, 500.0])
    if name == "bowl":
        # hemispherical shell, outer 20, wall 1.5, open towards +Z as saved
        shell = m3d.Manifold.sphere(20.0, 96) - m3d.Manifold.sphere(18.5, 96)
        half = shell.trim_by_plane([0.0, 0.0, -1.0], 0.0)  # keep z <= 0
        return _manifold_to_trimesh(half)
    if name == "wedge":
        section = np.array([[0.0, 0.0], [40.0, 0.0], [0.0, 20.0]])
        return _manifold_to_trimesh(m3d.Manifold.extrude(m3d.CrossSection([section]), 25.0))
    if name == "t_bracket":
        flange = m3d.Manifold.cube([40.0, 8.0, 30.0], True)
        web = m3d.Manifold.cube([8.0, 30.0, 30.0], True).translate([0.0, 19.0, 0.0])
        return _manifold_to_trimesh(flange + web)
    if name == "mushroom":
        cap = m3d.Manifold.cylinder(6.0, 15.0, 15.0, 96, True).translate([0.0, 0.0, 15.0])
        stem = m3d.Manifold.cylinder(25.0, 3.0, 3.0, 48, True).translate([0.0, 0.0, 0.0])
        return _manifold_to_trimesh(cap + stem)
    if name == "plate_on_edge":
        plate = trimesh.creation.box([100.0, 100.0, 3.0])
        return rotated(plate, "x", 90.0)
    if name == "bar30":
        bar = trimesh.creation.box([80.0, 10.0, 10.0])
        return rotated(bar, "z", 30.0)
    if name == "fin_block":
        block = m3d.Manifold.cube([20.0, 20.0, 10.0], True)
        fin = m3d.Manifold.cube([0.6, 15.0, 8.0], True).translate([0.0, 0.0, -9.0])
        return _manifold_to_trimesh(block + fin)
    if name == "torus200k":
        return trimesh.creation.torus(major_radius=50.0, minor_radius=25.0,
                                      major_sections=400, minor_sections=250)
    raise KeyError(name)


_FIXTURE_PATHS: dict[str, str] = {}
_RESULTS: dict = {}


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("fixtures")


@pytest.fixture(scope="session")
def fixture_path(fixture_dir):
    def _path(name: str, ext: str = "stl") -> str:
        key = f"{name}.{ext}"
        if key not in _FIXTURE_PATHS:
            mesh = build_fixture(name)
            path = str(fixture_dir / key)
            if ext == "stl_ascii":
                path = str(fixture_dir / f"{name}-ascii.stl")
                mesh.export(path, file_type="stl_ascii")
            else:
                mesh.export(path)
            _FIXTURE_PATHS[key] = path
        return _FIXTURE_PATHS[key]
    return _path


@pytest.fixture(scope="session")
def run_box(fixture_path, fixture_dir):
    """Run the pipeline once per (fixture, options) and cache the result."""
    def _run(name: str, ext: str = "stl", orientation: str = "auto", angles=None,
             max_outer=None, scale: float = 1.0, **spec_kwargs):
        key = (name, ext, orientation, angles, max_outer, scale, tuple(sorted(spec_kwargs.items())))
        if key not in _RESULTS:
            model = load_model(fixture_path(name, ext), scale)
            spec = BoxSpec(**spec_kwargs)
            result = make_box(model, spec, orientation=orientation, angles=angles, max_outer=max_outer)
            out = fixture_dir / f"out-{len(_RESULTS)}"
            paths = write_outputs(result, str(out), name, preview=True)
            _RESULTS[key] = (result, paths)
        return _RESULTS[key]
    return _run


@pytest.fixture(scope="session")
def cli():
    def _cli(*args: str, cwd: str | None = None):
        cmd = [sys.executable, "-m", "model_box_maker.cli", *args]
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": ROOT})
    return _cli


def trimesh_of(vf) -> trimesh.Trimesh:
    v, f = vf
    return trimesh.Trimesh(np.asarray(v), np.asarray(f), process=False)


def manifold_of(vf) -> m3d.Manifold:
    v, f = vf
    return m3d.Manifold(m3d.Mesh64(vert_properties=np.ascontiguousarray(v, dtype=np.float64),
                                   tri_verts=np.ascontiguousarray(f, dtype=np.uint64)))


def intersects(a: m3d.Manifold, b: m3d.Manifold) -> bool:
    return (a ^ b).volume() > 1e-6


def translated(vf, offset):
    v, f = vf
    return np.asarray(v) + np.asarray(offset, dtype=float), f
