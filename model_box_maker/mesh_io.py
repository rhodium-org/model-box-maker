"""Reading a model and writing bodies.

Implements model-box-maker REQ-0001 (common mesh files, refused when broken),
REQ-0002 (millimetres, --scale, units warning), REQ-0016 (closed bodies as STL
or 3MF) and REQ-0019 (byte-identical output: canonical vertex and face order,
fixed zip timestamps, no timestamps anywhere).
"""

from __future__ import annotations

import io
import os
import struct
import zipfile
from dataclasses import dataclass

import numpy as np
import trimesh

SUPPORTED = (".stl", ".obj", ".3mf", ".ply")
UNITS_SMALL = 5.0  # mm
UNITS_LARGE = 300.0  # mm


class ModelError(ValueError):
    """The model file cannot be used. The message is one line naming the fault."""


@dataclass
class Model:
    vertices: np.ndarray  # (n, 3) float64, already scaled
    faces: np.ndarray  # (m, 3) int64
    path: str
    scale: float
    watertight: bool
    volume: float | None  # None unless watertight

    @property
    def extents(self) -> np.ndarray:
        return self.vertices.max(axis=0) - self.vertices.min(axis=0)

    def trimesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(self.vertices, self.faces, process=False)


def load_model(path: str, scale: float = 1.0) -> Model:
    """Read a mesh file, scale it, and refuse anything a box cannot be made for."""
    if not (scale > 0):
        raise ModelError(f"scale must be greater than zero, got {scale}")
    if not os.path.isfile(path):
        raise ModelError(f"cannot read {path}: no such file")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        raise ModelError(f"cannot read {path}: expected one of {', '.join(SUPPORTED)}")
    if os.path.getsize(path) == 0:
        raise ModelError(f"cannot read {path}: the file is empty")
    try:
        raw = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:  # trimesh raises many types
        raise ModelError(f"cannot read {path}: {_one_line(exc)}") from exc
    if not isinstance(raw, trimesh.Trimesh):
        raise ModelError(f"cannot read {path}: no mesh found in the file")
    vertices = np.asarray(raw.vertices, dtype=np.float64)
    faces = np.asarray(raw.faces, dtype=np.int64)
    if len(faces) == 0:
        raise ModelError(f"{path} holds no triangles")
    if not np.isfinite(vertices).all():
        raise ModelError(f"{path} has a vertex that is not a finite number")
    vertices = vertices * float(scale)
    used = np.zeros(len(vertices), dtype=bool)
    used[faces.ravel()] = True
    extents = vertices[used].max(axis=0) - vertices[used].min(axis=0)
    for axis, name in enumerate("XYZ"):
        if extents[axis] <= 1e-9:
            raise ModelError(f"{path} has zero extent along {name}; it is not a solid shape")
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    if len(mesh.faces) == 0:
        raise ModelError(f"{path} holds no usable triangles")
    watertight = bool(mesh.is_watertight and mesh.is_winding_consistent)
    volume = float(abs(mesh.volume)) if watertight else None
    return Model(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        path=path,
        scale=float(scale),
        watertight=watertight,
        volume=volume,
    )


def units_warning(model: Model) -> str | None:
    """REQ-0002: a suspicious size is reported, not refused."""
    longest = float(model.extents.max())
    if longest < UNITS_SMALL:
        return (f"warning: the model is only {longest:.2f} mm across; "
                "check the units (use --scale if the file is not in millimetres)")
    if longest > UNITS_LARGE:
        return (f"warning: the model is {longest:.1f} mm across; "
                "check the units (use --scale if the file is not in millimetres)")
    return None


def _one_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return text[0] if text else exc.__class__.__name__


# ----------------------------------------------------------------- canonical form


def canonical(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Order vertices and faces so equal geometry gives equal bytes (REQ-0019).

    Winding is preserved: each face is rotated to start at its smallest index,
    never reversed.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0:
        return vertices.reshape(0, 3), faces.reshape(0, 3)
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    remap = np.empty(len(vertices), dtype=np.int64)
    remap[order] = np.arange(len(vertices))
    v = vertices[order]
    f = remap[faces]
    if len(f):
        start = f.argmin(axis=1)
        idx = (start[:, None] + np.arange(3)[None, :]) % 3
        f = np.take_along_axis(f, idx, axis=1)
        forder = np.lexsort((f[:, 2], f[:, 1], f[:, 0]))
        f = f[forder]
    return v, f


# ----------------------------------------------------------------- writers


def write_stl(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Binary STL with a fixed header; no timestamp, no library banner."""
    v, f = canonical(vertices, faces)
    tri = v[f]  # (m, 3, 3)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normals, axis=1)
    length[length == 0] = 1.0
    normals = normals / length[:, None]
    record = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    data = np.zeros(len(f), dtype=record)
    data["n"] = normals.astype(np.float32)
    data["v"] = tri.astype(np.float32)
    header = b"model-box-maker binary STL".ljust(80, b"\0")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(f)))
        fh.write(data.tobytes())


def write_3mf(path: str, objects: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    """A minimal 3MF holding one named mesh object per body, deterministic bytes.

    ``objects`` is a list of (name, vertices, faces). Coordinates are written in
    millimetres in the order given; every zip entry carries the same fixed date.
    """
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n',
        " <resources>\n",
    ]
    for index, (name, vertices, faces) in enumerate(objects, start=1):
        v, f = canonical(vertices, faces)
        parts.append(f'  <object id="{index}" name="{_xml_escape(name)}" type="model">\n   <mesh>\n    <vertices>\n')
        parts.extend(f'     <vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>\n' for x, y, z in v)
        parts.append("    </vertices>\n    <triangles>\n")
        parts.extend(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>\n' for a, b, c in f)
        parts.append("    </triangles>\n   </mesh>\n  </object>\n")
    parts.append(" </resources>\n <build>\n")
    parts.extend(f'  <item objectid="{index}"/>\n' for index in range(1, len(objects) + 1))
    parts.append(" </build>\n</model>\n")
    model_xml = "".join(parts).encode("utf-8")

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        "</Types>\n"
    ).encode("utf-8")
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        ' <Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
        "</Relationships>\n"
    ).encode("utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry_name, payload in (
            ("[Content_Types].xml", content_types),
            ("_rels/.rels", rels),
            ("3D/3dmodel.model", model_xml),
        ):
            info = zipfile.ZipInfo(entry_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, payload)
    with open(path, "wb") as fh:
        fh.write(buffer.getvalue())


def _xml_escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def load_body(path: str) -> trimesh.Trimesh:
    """Load one written body back (used by tests and the preview check)."""
    mesh = trimesh.load(path, force="mesh", process=False)
    return trimesh.Trimesh(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64), process=True)
