# model-box-maker

A Python tool that turns a 3D model file into a printable two-part postal box
shaped to that model. Read `README.md` first for the design; the requirements
graph in `idd/` is the specification the code is built and tested against.

## Working here

- This repo is IDD-managed; the graph is at `idd/`. Run `tl -C idd context`
  before changing anything, change the graph only through the `tl` CLI, and
  keep `tl -C idd check` at 0 errors.
- Every commit cites the item it serves, written as `model-box-maker <UID>`.
- Generate or amend the requirement before implementing it, never after.
- Every item is `origin: ai` and sits at `proposed` until a human ratifies it.
  `check --strict` reports each unratified item; that red is the expected
  state for an unread item, not a defect to silence. Never ratify on a
  human's behalf.
- Item text is plain, short technical sentences. Avoid `: ` inside a YAML
  plain scalar; the CLI quotes for you when you go through `tl new`/`tl amend`.
- When code exists: outputs are deterministic and derive only from the input
  model and the options; never hand-edit a generated STL, 3MF or report.
