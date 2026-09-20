"""Input handling: model-box-maker TEST-0001..TEST-0006."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import trimesh

from model_box_maker import BoxSpec, SpecError
from model_box_maker.mesh_io import ModelError, load_model, units_warning
from model_box_maker.spec import dimension_names


def _volume(vf):
    return abs(trimesh.Trimesh(*vf, process=False).volume)


def test_0001_every_supported_format_gives_the_same_box(run_box):
    """TEST-0001: STL binary, STL ASCII, OBJ, 3MF and PLY agree within 0.1 percent."""
    results = {ext: run_box("t_bracket", ext=ext, pitch=0.5) for ext in ("stl", "stl_ascii", "obj", "3mf", "ply")}
    reference, _ = results["stl"]
    ref_volume = _volume(reference.base)
    for ext, (result, _) in results.items():
        assert abs(_volume(result.base) - ref_volume) <= 1e-3 * ref_volume, ext
        assert np.allclose(result.choice.winner.rotation, reference.choice.winner.rotation, atol=1e-3), ext


def test_0002_a_broken_model_is_refused_before_anything_is_written(tmp_path, cli):
    """TEST-0002: empty file, NaN vertex, zero-height triangles each exit 1 with one line."""
    empty = tmp_path / "empty.stl"
    empty.write_bytes(b"")
    nan_obj = tmp_path / "nan.obj"
    nan_obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 nan\nf 1 2 3\n")
    flat = tmp_path / "flat.stl"
    tri = trimesh.Trimesh([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]], [[0, 1, 2], [1, 3, 2]], process=False)
    tri.export(str(flat))
    for bad in (empty, nan_obj, flat):
        out = tmp_path / f"out-{bad.stem}"
        proc = cli(str(bad), "-o", str(out))
        assert proc.returncode == 1, (bad, proc.stderr)
        lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
        assert len(lines) == 1 and bad.name in lines[0], proc.stderr
        assert not out.exists() or not any(out.iterdir())


def test_0003_scale_is_applied_before_everything_else(run_box):
    """TEST-0003: a 1-unit cube with --scale 25.4 gets a cavity 25.4 + 2c wide, at most three pitches over per side."""
    result, _ = run_box("unit_cube", scale=25.4, orientation="keep", pitch=0.25)
    x0, y0, x1, y1 = result.cavity().bbox()
    nominal = 25.4 + 2 * result.spec.clearance
    for width in (x1 - x0, y1 - y0):
        assert width >= nominal - 1e-9
        assert width <= nominal + 6 * result.spec.pitch + 1e-9


def test_0004_suspicious_sizes_warn_and_continue(fixture_path, cli, tmp_path):
    """TEST-0004: 2 mm and 500 mm cubes warn on stderr and still produce a box; 60 mm does not warn."""
    for name, expect in (("cube2", True), ("cube500", True), ("cube60", False)):
        model = load_model(fixture_path(name))
        warning = units_warning(model)
        assert (warning is not None) == expect, name
        if expect:
            assert "units" in warning
    proc = cli(fixture_path("cube2"), "-o", str(tmp_path / "w"), "--pitch", "0.5")
    assert proc.returncode == 0, proc.stderr
    assert "units" in proc.stderr
    assert (tmp_path / "w" / "cube2-base.stl").exists()


def test_0005_help_names_every_dimension_and_its_default(cli):
    """TEST-0005: --help lists each dimension option of REQ-0003 with its default in mm."""
    proc = cli("--help")
    assert proc.returncode == 0
    defaults = BoxSpec()
    for name in dimension_names():
        option = BoxSpec.option_name(name)
        assert option in proc.stdout, option
        value = getattr(defaults, name)
        assert f"default: {value:g}" in proc.stdout or f"default: {value}" in proc.stdout, option
    assert "(mm)" in proc.stdout


def test_0006_a_bad_dimension_fails_fast(cli, fixture_path, tmp_path):
    """TEST-0006: negative clearance, zero floor, 0.4 mm wall each exit 2 naming the option, quickly, writing nothing."""
    cases = [("--clearance", "-1"), ("--floor", "0"), ("--wall", "0.4")]
    for option, value in cases:
        out = tmp_path / f"out{option}"
        started = time.time()
        # the model path does not exist: a refusal that came after trying to read it would be exit 1
        proc = cli(str(tmp_path / "no-such-model.stl"), "-o", str(out), option, value)
        elapsed = time.time() - started
        assert proc.returncode == 2, (option, proc.stderr)
        assert elapsed < 1.0, elapsed
        assert option in proc.stderr
        assert not out.exists()
    proc = cli(fixture_path("cube"), "-o", str(tmp_path / "never"), "--skirt", "0.4")
    assert proc.returncode == 2 and "--skirt" in proc.stderr and not (tmp_path / "never").exists()
    with pytest.raises(SpecError):
        BoxSpec(skirt=0.4)
