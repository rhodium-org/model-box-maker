"""The pipeline: model in, base and lid and report out.

Implements model-box-maker INT-0001 end to end: REQ-0006 (the report),
REQ-0016 (two files, each the way up it prints), REQ-0017 (refuse a body that
needs support), REQ-0018 (--preview), REQ-0019 (deterministic files) and
REQ-0020 (never a partial set of files).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from . import geometry, mesh_io, orient, printcheck
from .mesh_io import Model, ModelError
from .orient import Choice, OrientationError
from .spec import BoxSpec

PREVIEW_LIFT = 10.0  # mm of daylight between base rim and the lifted lid in a preview


@dataclass
class BoxResult:
    model: Model
    spec: BoxSpec
    choice: Choice
    layout: geometry.Layout
    base: tuple[np.ndarray, np.ndarray]
    lid: tuple[np.ndarray, np.ndarray]
    report: dict = field(default_factory=dict)

    # ----- geometry helpers used by the preview and the tests

    def model_transform(self) -> np.ndarray:
        """4x4: model file coordinates (already scaled) -> rest pose in box coordinates."""
        m = np.eye(4)
        m[:3, :3] = self.choice.winner.rotation
        m[:3, 3] = self.layout.translation
        return m

    def model_at_rest(self) -> np.ndarray:
        return self.model.vertices @ self.choice.winner.rotation.T + self.layout.translation

    def seated_lid(self) -> tuple[np.ndarray, np.ndarray]:
        t = geometry.seated_lid_transform(self.layout)
        v, f = self.lid
        v2 = v @ t[:3, :3].T + t[:3, 3]
        return v2, f

    def cavity(self):
        return self.choice.winner.cavity


def make_box(model: Model, spec: BoxSpec, orientation: str = "auto",
             angles: tuple[float, float, float] | None = None,
             max_outer: tuple[float, float, float] | None = None) -> BoxResult:
    """Choose the pose, build both bodies, check them, assemble the report."""
    mode = "angles" if angles is not None else orientation
    choice = orient.choose(model.vertices, model.faces, spec, mode, angles, max_outer)
    winner = choice.winner
    layout = geometry.layout_for(winner.cavity, winner.lip_rect, winner.rim_z, spec)

    base_solid = geometry.build_base(winner.cavity, layout)
    lid_solid = geometry.build_lid(layout)
    base = mesh_io.canonical(*geometry.from_manifold(base_solid))
    lid = mesh_io.canonical(*geometry.from_manifold(lid_solid))

    for name, (v, f) in (("base", base), ("lid", lid)):
        bad = printcheck.check_printable(v, f)
        if bad:
            first = bad[0]
            raise RuntimeError(f"the {name} would need support: face {first.face} {first.reason} "
                               f"({len(bad)} such faces); refusing to write it")

    result = BoxResult(model=model, spec=spec, choice=choice, layout=layout, base=base, lid=lid)
    result.report = build_report(result, max_outer)
    return result


def build_report(result: BoxResult, max_outer) -> dict:
    """The report of REQ-0006, without paths (added when files are written)."""
    winner = result.choice.winner
    layout = result.layout
    spec = result.spec
    size = layout.size
    outer_volume = size[0] * size[1] * size[2]
    cavity_volume = winner.cavity_volume
    free_fraction = None
    if result.model.volume is not None and cavity_volume > 0:
        free_fraction = max(0.0, (cavity_volume - result.model.volume) / cavity_volume)
    candidates = []
    for ev in result.choice.evaluations[:10]:
        candidates.append({
            "label": ev.label,
            "stage": ev.stage,
            "rotation_euler_xyz_deg": [round(a, 4) for a in ev.euler_xyz()],
            "outer_size_mm": [round(s, 3) for s in ev.size],
            "outer_volume_mm3": round(ev.outer_volume, 1),
            "cavity_volume_mm3": round(ev.cavity_volume, 1),
            "fits_limit": bool(ev.fits),
            "chosen": ev is winner,
        })
    options = spec.as_dict()
    options["max_outer"] = list(max_outer) if max_outer else None
    return {
        "tool": "model-box-maker",
        "input": os.path.basename(result.model.path),
        "scale": result.model.scale,
        "model_watertight": result.model.watertight,
        "model_volume_mm3": None if result.model.volume is None else round(result.model.volume, 2),
        "model_triangles": int(len(result.model.faces)),
        "orientation_mode": result.choice.mode,
        "rotation_matrix": [[round(float(x), 9) for x in row] for row in winner.rotation],
        "rotation_euler_xyz_deg": [round(a, 6) for a in winner.euler_xyz()],
        "translation_mm": [round(float(t), 6) for t in layout.translation],
        "outer_size_mm": [round(s, 3) for s in size],
        "outer_volume_mm3": round(outer_volume, 1),
        "cavity_volume_mm3": round(cavity_volume, 1),
        "free_volume_fraction": None if free_fraction is None else round(free_fraction, 4),
        "rim_z_mm": round(layout.rim_z, 4),
        "cradle_lowest_z_mm": round(layout.floor_z, 4),
        "cradle_highest_z_mm": round(layout.cradle_top_z, 4),
        "model_top_z_mm": round(layout.model_top_z, 4),
        "options": options,
        "winner_reason": result.choice.reason,
        "candidates": candidates,
        "files": [],
    }


def write_outputs(result: BoxResult, out_dir: str, stem: str, fmt: str = "stl",
                  preview: bool = False) -> list[str]:
    """Write base, lid, report and optional preview; all or nothing (REQ-0020)."""
    os.makedirs(out_dir, exist_ok=True)
    ext = ".3mf" if fmt == "3mf" else ".stl"
    targets = {
        "base": os.path.join(out_dir, f"{stem}-base{ext}"),
        "lid": os.path.join(out_dir, f"{stem}-lid{ext}"),
        "report": os.path.join(out_dir, f"{stem}-report.json"),
    }
    if preview:
        targets["preview"] = os.path.join(out_dir, f"{stem}-preview.3mf")
    result.report["files"] = [os.path.basename(p) for p in targets.values()]
    temps = {k: v + f".tmp-{os.getpid()}" for k, v in targets.items()}
    try:
        _write_body(temps["base"], fmt, "base", *result.base)
        _write_body(temps["lid"], fmt, "lid", *result.lid)
        if preview:
            lid_v, lid_f = result.seated_lid()
            lid_v = lid_v + np.array([0.0, 0.0, result.spec.lip + PREVIEW_LIFT])
            mesh_io.write_3mf(temps["preview"], [
                ("base", *result.base),
                ("lid", lid_v, lid_f),
                ("model", result.model_at_rest(), result.model.faces),
            ])
        with open(temps["report"], "w", encoding="utf-8") as fh:
            json.dump(result.report, fh, indent=2, sort_keys=False)
            fh.write("\n")
        for key in targets:
            os.replace(temps[key], targets[key])
    except BaseException:
        for path in temps.values():
            if os.path.exists(path):
                os.remove(path)
        raise
    return list(targets.values())


def _write_body(path: str, fmt: str, name: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    if fmt == "3mf":
        mesh_io.write_3mf(path, [(name, vertices, faces)])
    else:
        mesh_io.write_stl(path, vertices, faces)


__all__ = ["BoxResult", "ModelError", "OrientationError", "make_box", "write_outputs", "build_report"]
