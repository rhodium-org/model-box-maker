"""Command line: ``model-box-maker MODEL [options]`` (model-box-maker REQ-0020).

Exit 0 on success, 2 on a usage error (argparse and refused dimension values,
REQ-0003), 1 on any other failure with a single line on stderr.

The geometry modules are imported only after the options have been accepted,
so a refused value or --help never waits for numpy, trimesh and manifold3d to
load (TEST-0006).
"""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .spec import WALL_PATTERNS, BoxSpec, SpecError

DIMENSION_HELP = {
    "clearance": "space kept around the model in every direction",
    "wall": "wall thickness at the lip, the thinnest wall anywhere",
    "wall_max": "greatest wall thickness below the lip, seen from above; the box is cut back to it "
                "(larger than the box keeps the full rectangle)",
    "floor": "floor thickness under the lowest point of the cradle",
    "top_space": "space between the highest point of the model and the lid",
    "lip": "height of the lip the lid closes over",
    "lid_plate": "thickness of the lid plate",
    "skirt": "thickness of the lid skirt",
    "fit": "gap between the skirt and the lip",
    "corner_radius": "radius of the vertical outer edges (0 for sharp)",
    "pitch": "grid pitch the cradle is computed on",
}


def build_parser() -> argparse.ArgumentParser:
    defaults = BoxSpec()
    parser = argparse.ArgumentParser(
        prog="model-box-maker",
        description="Make a printable two-part postal box shaped to one 3D model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", help="model file: STL (binary or ASCII), OBJ, 3MF or PLY, in millimetres")
    parser.add_argument("-o", "--out", default=".", help="directory to write into")
    parser.add_argument("--stem", default=None, help="output file stem (default: the model's file name)")
    parser.add_argument("--format", choices=("stl", "3mf"), default="stl", help="format of the base and lid files")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply every coordinate first (25.4 for a file authored in inches)")
    parser.add_argument("--orientation", default="auto",
                        help="auto, keep (the file's Z up, used exactly), or RX,RY,RZ degrees about X then Y then Z")
    parser.add_argument("--max-outer", default=None, metavar="A,B,C",
                        help="largest closed box allowed, three sides in mm in any order")
    parser.add_argument("--size-tolerance", type=float, default=defaults.size_tolerance, metavar="PCT",
                        help="poses within this percent of the smallest outer volume tie; the tie goes to the "
                             "smallest cavity volume")
    dims = parser.add_argument_group("dimensions (mm)")
    for name, text in DIMENSION_HELP.items():
        dims.add_argument(BoxSpec.option_name(name), type=float, default=getattr(defaults, name), help=text)
    lattice = parser.add_argument_group("walls")
    lattice.add_argument("--walls", choices=WALL_PATTERNS, default=defaults.walls,
                         help="solid walls, or cut a hex or diamond lattice into the wall panels")
    lattice.add_argument("--max-hole", type=float, default=defaults.max_hole, help="widest lattice hole (mm)")
    lattice.add_argument("--band", type=float, default=defaults.band,
                         help="solid band kept beside the lip, the floor and every vertical edge (mm)")
    parser.add_argument("--preview", action="store_true",
                        help="also write <stem>-preview.3mf with base, lifted lid and the model at rest")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def parse_orientation(text: str, parser: argparse.ArgumentParser):
    if text in ("auto", "keep"):
        return text, None
    parts = text.split(",")
    if len(parts) != 3:
        parser.error("--orientation must be auto, keep, or three angles RX,RY,RZ in degrees")
    try:
        return "angles", tuple(float(p) for p in parts)
    except ValueError:
        parser.error("--orientation angles must be numbers, e.g. 90,0,0")


def parse_limit(text: str | None, parser: argparse.ArgumentParser):
    if text is None:
        return None
    parts = text.split(",")
    try:
        values = tuple(float(p) for p in parts)
    except ValueError:
        values = ()
    if len(values) != 3 or any(v <= 0 for v in values):
        parser.error("--max-outer must be three positive lengths A,B,C in mm")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        spec = BoxSpec(
            clearance=args.clearance, wall=args.wall, wall_max=args.wall_max, floor=args.floor,
            top_space=args.top_space,
            lip=args.lip, lid_plate=args.lid_plate, skirt=args.skirt, fit=args.fit,
            corner_radius=args.corner_radius, pitch=args.pitch, size_tolerance=args.size_tolerance,
            walls=args.walls, max_hole=args.max_hole, band=args.band,
        )
    except SpecError as exc:
        parser.error(str(exc))
    if not (args.scale > 0):
        parser.error(f"--scale: must be greater than zero, got {args.scale}")
    mode, angles = parse_orientation(args.orientation, parser)
    limit = parse_limit(args.max_outer, parser)
    stem = args.stem or os.path.splitext(os.path.basename(args.model))[0]

    from .box import make_box, write_outputs
    from .mesh_io import ModelError, load_model, units_warning
    from .orient import OrientationError

    try:
        model = load_model(args.model, args.scale)
        warning = units_warning(model)
        if warning:
            print(warning, file=sys.stderr)
        result = make_box(model, spec, orientation=mode, angles=angles, max_outer=limit)
        for line in result.warnings:
            print(line, file=sys.stderr)
        paths = write_outputs(result, args.out, stem, fmt=args.format, preview=args.preview)
    except (ModelError, OrientationError, RuntimeError, OSError) as exc:
        print(f"model-box-maker: {_one_line(exc)}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    w, d, h = result.layout.size
    euler = result.choice.winner.euler_xyz()
    print(f"pose: {result.choice.winner.label} (X {euler[0]:.1f}, Y {euler[1]:.1f}, Z {euler[2]:.1f} deg); "
          f"box {w:.1f} x {d:.1f} x {h:.1f} mm, outline {result.layout.exterior.kind}")
    return 0


def _one_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return text[0] if text else exc.__class__.__name__


if __name__ == "__main__":
    sys.exit(main())
