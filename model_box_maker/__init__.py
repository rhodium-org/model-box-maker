"""model-box-maker: a printable two-part postal box shaped to one 3D model.

The requirements graph in ``idd/`` is the specification; module docstrings cite
the items they implement as ``model-box-maker <UID>``.

The geometry modules pull in numpy, scipy, trimesh and manifold3d, which take
about a second to import. They are loaded on first use so that option
validation and ``--help`` answer at once (REQ-0003, TEST-0006).
"""

from __future__ import annotations

import importlib

from .spec import BoxSpec, SpecError

__version__ = "0.1.0"

_LAZY = {
    "BoxResult": ".box",
    "make_box": ".box",
    "write_outputs": ".box",
    "ModelError": ".mesh_io",
    "OrientationError": ".orient",
}

__all__ = [
    "BoxSpec",
    "SpecError",
    "BoxResult",
    "ModelError",
    "OrientationError",
    "make_box",
    "write_outputs",
    "__version__",
]


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name, __name__), name)
