# model-box-maker

Take a 3D model file, get back a printable two-part box for posting it. The
model drops into the base from above, the lid closes over a lip, and the pair
protects the model in the post. Two STL (or 3MF) files come out, each already
the way up it prints, plus a report that explains the box.

```bash
pipx install git+https://github.com/rhodium-org/model-box-maker.git
model-box-maker figurine.stl -o out --preview
```

Pure Python; every dependency ships binary wheels (numpy, scipy, trimesh,
manifold3d, rtree, networkx, lxml). No CAD kernel or external program.

**Status: implemented and verified against the requirements graph in
`idd/`.** All items are ratified and the code is tested against every
requirement (`python -m pytest`, 35 tests, one function per TEST item).
`tl -C idd check --strict` names anything still waiting on a signature.

## How the box is made

1. **Pose.** The model is placed in a candidate orientation: the file as saved,
   each face of its convex hull downward, and each principal axis vertical.
   Every candidate is turned about the vertical so its footprint is smallest.
2. **Grow.** The posed model is offset outward by the clearance (1 mm by
   default) so that a flex in the box never bears on the model.
3. **Sweep.** The cavity is every point the grown model passes through when it
   is lifted straight up out of the box. The walls are therefore vertical, the
   floor follows the underside of the model, and the model always lifts
   straight out. This is the "subtract the model as it drops in" idea, computed
   as a heightmap on a grid that only ever rounds outward: each cell takes the
   lowest point of the grown model over the whole cell, less half a cell
   diagonal, and the outline is grown by one cell.
4. **Cut back the outside.** Seen from above, the box starts as the smallest
   rounded rectangle around the cavity and is cut back wherever the wall would
   be thicker than `--wall-max` (5 mm by default), so the outside follows the
   model at a bounded distance while never coming nearer than `--wall`. The
   floor and the lid top stay flat: a conformal top or underside would need
   support material to print. The tool then checks that the box still stands:
   from the footprint's convex hull it works out the angle the closed box can
   tilt before it tips, and below 20 degrees it keeps the full rectangle if
   that does better, saying so in the report.
5. **Score.** Candidates are screened at a 1 mm pitch and the best few
   re-scored at the real pitch. Each is scored by the volume of the closed box
   it needs (footprint times height). Poses within 5 % of the smallest are
   ties, and a tie goes to the smallest cavity volume, which is the empty space
   around the model. That puts hollows and overhangs downward, so a thin bowl
   goes in mouth down over a dome and a mushroom goes stem down into a well,
   while a flat plate lies flat.
6. **Build.** With manifold3d: the base is the outline prism plus a narrower
   lip prism, minus a closed "cavity solid" whose underside is the cradle
   heightmap. The lid is a flat plate with a skirt that slides over the lip on
   the outside and lands flush with the wall; nothing on it reaches below the
   rim except that skirt, so the lid cannot touch the model however hard it is
   pressed. Because the lid prints plate down and is turned over to seat, it is
   built from the mirror image of the outline. No hinge, latch or snap; tape
   holds the lid for posting.
7. **Check.** Every downward face of both bodies must lie on the bed or lean no
   more than 45 degrees from vertical; a lattice bridge may span 5 mm. A body
   that fails is refused rather than written.
8. **Optionally open the walls.** `--walls hex` or `--walls diamond` cuts a
   lattice into the wall, each hole inward along the wall's own normal, leaving
   solid material wherever the outline bends sharply, around the lip, and from
   the floor up to the cradle beside that hole plus a band.

## Command line

```text
model-box-maker MODEL [-o DIR] [--stem NAME] [--format stl|3mf] [--scale N]
                [--orientation auto|keep|RX,RY,RZ] [--max-outer A,B,C]
                [--size-tolerance 5]
                [--clearance 1.0] [--wall 2.4] [--wall-max 5.0] [--floor 3.0]
                [--top-space 2.0] [--lip 6.0] [--lid-plate 2.0] [--skirt 1.6]
                [--fit 0.2] [--corner-radius 3.0] [--pitch 0.25]
                [--walls solid|hex|diamond] [--max-hole 6] [--band 4]
                [--preview]
```

Outputs: `<stem>-base.stl`, `<stem>-lid.stl`, `<stem>-report.json`, and with
`--preview` a `<stem>-preview.3mf` holding base, lifted lid and the model at
rest as three named objects. Exit 0 on success, 2 on a usage error (including
a refused dimension), 1 on any other failure with one line on stderr, and
never a partial set of files. Runs are byte-identical for the same input and
options.

The report records the rotation (matrix and X, Y, Z angles), the translation
that places the model, the closed box size and volume, the outline used and
why (cut back, or the rectangle kept), the footprint and rectangle areas, the
tipping angle, the cavity volume, the free volume fraction (watertight
models), every option used, and for an automatic orientation the ten best
candidates with their scores and why the winner won. To take a runner-up,
re-run with `--orientation RX,RY,RZ` using its angles.

## Material

Measured on a Grim Reaper duck (84k triangles, tilted pose, 75 × 70 × 59 mm),
with printed material estimated at two perimeters and 15 % infill:

| Box | Base STL | Estimated print |
|---|---|---|
| rectangle (`--wall-max 100`) | 181 cm³ | ~67 cm³, 83 g |
| cut back (default, 5 mm) | 134 cm³ | ~56 cm³, 69 g |

The rest of the material is the slicer's infill of the solid cradle mass
under the model and the shells; slice the base at 8 to 10 % gyroid and the
lid solid. Boxy models gain nothing from the cut-back; round or diagonal ones
gain most. `--wall 1.6 --lid-plate 1.6` shaves a little more at the cost of
stiffness.

## Things worth knowing

- **Lateral slack.** The cavity can be up to three grid pitches wider than the
  grown model on each side (0.75 mm at the default pitch): the cell the
  outline falls in, the cell the clearance reaches into, and the one cell of
  growth REQ-0009 asks for. Use `--pitch 0.1` for a tighter fit at the cost of
  time and file size.
- **Lid fit.** The skirt stands `--fit` (0.2 mm) off the lip all round and is
  that much shorter than the lip, so the plate lands on the rim first. Widen
  it to 0.3 if a print is tight; a slicer's XY compensation changes it too.
- **The tie-break can stand a thick slab on edge.** Cavity volume counts the
  clearance and top-space layers over the whole footprint, which favours a
  small footprint. A 100 × 100 × 20 mm slab lands within the 5 % band both
  ways and is boxed on edge. `--size-tolerance 0` makes outer volume decide
  alone; `--orientation keep` fixes it by hand.
- **File size.** The cradle is a triangulated heightmap, two triangles per
  cell, so a 100 × 100 mm footprint at 0.25 mm gives a base of several
  hundred thousand triangles. Slicers cope; `--pitch 0.5` quarters it.
- **Speed.** A 200 000-triangle, 150 mm model takes about 10 s at the default
  pitch on an idle desktop CPU and about 40 s on a busy one (NFR-0001 asks for
  under a minute). A million-triangle scan takes a couple of minutes and a
  gigabyte. Small models take a second or two.

## The requirements graph

The graph lives in `idd/` and is managed with
[throughline](https://github.com/rhodium-org/throughline) (`tl`).

| Register | Prefix | What it holds |
|---|---|---|
| `idd/vision` | `INT` | the one intent the tool serves |
| `idd/constraints` | `CON` | the printer the parts must print on |
| `idd/assumptions` | `ASM` | model files are in millimetres |
| `idd/risks` | `RISK` | the seven ways a box fails its model |
| `idd/non-goals` | `NG` | no hinge; not a packaging designer; no undercut |
| `idd/requirements` | `REQ` | twenty-one behaviours, one per thing that could be wanted differently |
| `idd/nonfunctional` | `NFR` | runtime bound; installs with pipx |
| `idd/tests` | `TEST` | thirty-five checks, at least one per requirement; `tests/` implements them one function per item |

```bash
tl -C idd context                 # the generated brief for this graph; read it first
tl -C idd check                   # the gate; 0 errors
tl -C idd query --type requirement
tl -C idd trace RISK-0001 --direction in
```

Change the graph only through the `tl` CLI. Every commit cites the item it
serves, written as `model-box-maker <UID>`. Module docstrings cite the items
they implement; test functions are named after theirs.

## Developing

```bash
python3 -m venv .venv && .venv/bin/pip install -e .[test]
.venv/bin/python -m pytest -q            # about three minutes; one 200k-triangle timing test
MODEL_BOX_MAKER_INSTALL_TEST=1 .venv/bin/python -m pytest -q tests/test_lid_and_output.py -k 0032
```

## Licence

Apache-2.0. Copyright 2026 Henry J Grech-Cini. Created by Dr Henry J Grech-Cini.
