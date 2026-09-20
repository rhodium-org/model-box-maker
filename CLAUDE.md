# model-box-maker

A Python tool that turns a 3D model file into a printable two-part postal box
shaped to that model. Read `README.md` first for the design; the requirements
graph in `idd/` is the specification the code is built and tested against.

## Working here

- This repo is IDD-managed; the graph is at `idd/`. Run `tl -C idd context`
  before changing anything, change the graph only through the `tl` CLI, and
  keep `tl -C idd check` at 0 errors.
- Every commit cites the item it serves, written as `model-box-maker <UID>`.
- Amend or add the requirement before changing the behaviour, never after.
  Amending a ratified item makes it `ratified-stale`; say so in the hand-over
  and never ratify on a human's behalf.
- Item text is plain, short technical sentences. Avoid `: ` inside a YAML
  plain scalar; the CLI quotes for you when you go through `tl new`/`tl amend`.
- Code: `python3 -m venv .venv && .venv/bin/pip install -e .[test]`, then
  `.venv/bin/python -m pytest -q` (about four minutes). Test functions are
  named `test_<NNNN>_...` after their TEST item; module docstrings name the
  REQ/NFR items they implement. Keep both true when you change either.
- Outputs are deterministic and derive only from the input model and the
  options; never hand-edit a generated STL, 3MF or report, and never put a
  timestamp or an absolute path into one.
- Geometry is built with manifold3d from closed solids; the cradle is a
  heightmap that only rounds outward (REQ-0009). If a boolean fails the tool
  raises rather than writing a body.
- Plan-view profiles are manifold3d CrossSections (`geometry.build_exterior`):
  the lip is the rounded rectangle intersected with the cavity outline offset
  by `wall_max - skirt - fit`; body and skirt are offsets of the lip. Never
  simplify the cut profile beyond removing coincident vertices: the rectangle's
  sides are exactly the wall from the cavity.
- The printed lid is the mirror image (in y) of the base outline; the seating
  transform turns it over about X. Keep the two consistent or an asymmetric
  outline will not fit.
- Export cleans meshes to the file's precision (`mesh_io.clean_for_export`);
  compare geometry in memory with `result.base`, but check files with
  `load_body`, which is what a slicer sees.
