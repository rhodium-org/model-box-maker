# model-box-maker

Take a 3D model file, get back a printable two-part box for posting it. The
model drops into the base from above, the lid closes over a lip, and the pair
protects the model in the post. Two STL (or 3MF) files come out, each already
the way up it prints, plus a report that explains the box.

**Status: specification only.** This repository holds the requirements graph
the tool will be built and tested against (`idd/`). No code has been written
yet; the graph comes first so the design can be read and argued with before
anything is implemented. Every item is AI-authored (`origin: ai`) and sits at
`proposed`.

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
   as a heightmap on a grid that only ever rounds outward.
4. **Score.** Each candidate is scored by the outer volume of the box it needs.
   Poses within 5 % of the smallest are ties, and a tie goes to the smallest
   cavity volume, which is the empty space around the model. That puts hollows
   and overhangs downward, so a bowl goes in mouth down over a dome and a
   mushroom goes stem down into a well, while a flat plate still lies flat.
5. **Build.** The base is a solid heightmap from its underside up to the cradle,
   inside a rounded-cornered cuboid wall. The top of the wall is stepped in to
   form a lip; the lid is a flat plate with a skirt that slides over that lip
   and stops on the rim, flush outside. Nothing on the lid reaches below the
   rim except the skirt, which lies outside the wall, so the lid cannot touch
   the model however hard it is pressed. There is no hinge, latch or snap; tape
   holds the lid for posting.
6. **Optionally open the walls.** `--walls hex` or `--walls diamond` cuts a
   lattice into the four wall panels to save filament and time, leaving solid
   bands at every vertical edge, around the lip and through the floor.

Every dimension is an option with a stated default, and the orientation can be
fixed by hand (`--orientation keep` or `--orientation RX,RY,RZ`) when the
automatic choice is wrong for a particular model. A `--max-outer A,B,C` limit
holds the box inside a postal size band or a printer bed.

## Planned command line

Option names below are a proposal; the requirements fix the behaviours and the
defaults, not the spellings.

```text
model-box-maker MODEL [--out DIR] [--format stl|3mf] [--scale N]
                [--orientation auto|keep|RX,RY,RZ] [--max-outer A,B,C]
                [--size-tolerance 5]
                [--clearance 1.0] [--wall 2.4] [--floor 3.0] [--top-space 2.0]
                [--lip 6.0] [--lid-plate 2.0] [--skirt 1.6] [--fit 0.2]
                [--corner-radius 3.0] [--pitch 0.25]
                [--walls solid|hex|diamond] [--max-hole 6] [--band 4]
                [--preview]
```

Outputs: `<stem>-base.stl`, `<stem>-lid.stl`, `<stem>-report.json`, and with
`--preview` a `<stem>-preview.3mf` holding base, lid and model as three named
objects. Exit 0 on success, 2 on a usage error, 1 on any other failure, and
never a partial set of files.

## The requirements graph

The graph lives in `idd/` and is managed with
[throughline](https://github.com/rhodium-org/throughline) (`tl`).

| Register | Prefix | What it holds |
|---|---|---|
| `idd/vision` | `INT` | the one intent the tool serves |
| `idd/constraints` | `CON` | the printer the parts must print on |
| `idd/assumptions` | `ASM` | model files are in millimetres |
| `idd/risks` | `RISK` | the six ways a box fails its model |
| `idd/non-goals` | `NG` | no hinge; not a packaging designer; no undercut |
| `idd/requirements` | `REQ` | twenty behaviours, one per thing that could be wanted differently |
| `idd/nonfunctional` | `NFR` | runtime bound; installs with pipx |
| `idd/tests` | `TEST` | thirty-two checks, at least one per requirement |

```bash
tl -C idd context                 # the generated brief for this graph; read it first
tl -C idd check                   # the gate; 0 errors, unratified items are warnings
tl -C idd query --type requirement
tl -C idd trace RISK-0001 --direction in
```

Change the graph only through the `tl` CLI. Every commit cites the item it
serves, written as `model-box-maker <UID>`.
