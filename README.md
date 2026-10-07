# moldgen

Turn a 3D model into a casting mold you can 3D print.

moldgen takes a mesh (STL, OBJ, PLY, OFF, 3MF, glTF/GLB), finds the direction in
which a rigid mold releases the part best, splits it along a flat or curved
parting surface, adds side pieces where the two halves alone would lock on the
part, and builds
print-ready mold pieces with a pour sprue and funnel, air vents and
registration keys. Defaults (sprue and vent size, shrinkage compensation,
release agent, print-material temperature limits) come from presets for common
casting materials such as polyurethane resin, epoxy, plaster, wax, soap,
chocolate and low-melt alloys.

It runs as a command-line tool, a local browser GUI, or a Python library.

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Lion4re/automated_3d_mold_generator
cd automated_3d_mold_generator
pip install ".[repair]"
```

The `repair` extra adds [pymeshfix](https://github.com/pyvista/pymeshfix) for
fixing badly broken meshes. Clean meshes do not need it.

## Quick start

```bash
moldgen make models/Pawn.stl
```

This writes `molds/Pawn/` with the mold pieces as STL files (already oriented
for printing, parting face up), `INSTRUCTIONS.txt` with print, assembly, pouring
and demolding notes, and `report.json` with every value that was used.

Inspect a model before committing to a print:

```bash
moldgen analyze models/Knight.stl
```

Open the graphical interface:

```bash
moldgen gui
```

Run `moldgen` with no arguments for a short guided setup in the terminal.

## Command line

```text
moldgen make MODEL... [options]     generate molds
moldgen analyze MODEL               show parting candidates and recommended settings
moldgen materials                   list casting and print material presets
moldgen gui                         open the browser interface
```

Common options for `make`:

| Option | Meaning | Default |
|---|---|---|
| `-m, --material KEY` | Casting material preset (`moldgen materials`) | `resin` |
| `-p, --print-material KEY` | Material the mold is printed in | `pla` |
| `-d, --direction AXIS` | Demolding direction: `auto`, `x`, `-z`, ... or a vector such as `0,1,1` | `auto` |
| `--parting-offset MM` | Parting plane position along the direction | least undercut |
| `--pieces auto\|2\|4` | `auto` adds side pieces where needed; `2` and `4` force a fixed split | `auto` |
| `--max-pieces N` | Most pieces `auto` may use (2-10) | `6` |
| `--flat-parting` | Always split the halves with a flat plane | curved where it helps |
| `--wall MM` | Wall thickness around the part | from part size |
| `--sprue MM`, `--vent MM` | Channel diameters | from material |
| `--keys N`, `--clearance MM` | Registration keys and their fit clearance | `4`, `0.2` |
| `--shrinkage PCT` | Override the material's shrinkage compensation | from material |
| `--units mm\|cm\|m\|in`, `--scale F` | Input units and scale | `mm`, `1.0` |
| `-o, --out DIR` | Output folder; each model gets a subfolder | `molds` |
| `--json` | Machine-readable output | |

Several models can be processed at once (`moldgen make models/*.stl`). Run
`moldgen make --help` for the full list.

The exit code is 0 when every mold was generated, 1 when at least one model
failed, and 2 for invalid arguments.

## Graphical interface

`moldgen gui` starts a local server and opens it in your browser (built on
[viser](https://viser.studio)). Nothing leaves your machine.

1. Drop a model onto the page.
2. Faces are coloured by how they release: grey releases cleanly, amber has
   little draft, red is an undercut that will lock the mold.
3. Accept the suggested parting plane, choose another axis, or drag the plane.
4. Pick the casting and print materials and adjust wall, keys and pieces.
5. Generate, inspect the exploded pieces and any warnings, then download a zip.

## Python API

```python
from moldgen import MoldConfig, generate_mold

result = generate_mold("models/Rook.stl", MoldConfig(material="plaster", pieces=4))
for warning in result.warnings:
    print(warning)
result.save("molds/Rook")
```

## How it works

1. **Load and repair.** The mesh is converted to millimetres. A mesh that is
   already a valid solid is used unchanged; otherwise duplicate and degenerate
   elements are removed, winding is made consistent, holes are closed, and the
   result is checked again. Repair never silently changes the volume: changes of
   more than 3 % produce a warning.
2. **Parting direction.** Candidate directions (the axes, the principal axes and
   a sampled hemisphere, refined locally) are scored by the surface area that
   cannot be released by either half, using an occlusion-aware ray test rather
   than face normals alone, then by area with too little draft. The parting plane
   is placed where the fewest faces are trapped. Where a flat plane would still
   trap part of the model, the halves are split along a curved surface that
   follows the part's outline instead
   ([docs/curved-parting-design.md](docs/curved-parting-design.md)).
3. **Mold block.** The block is aligned with the minimum-area rectangle around
   the part, so it uses as little filament as possible. The cavity is enlarged to
   compensate for the material's shrinkage.
4. **Gating.** The mold is poured with the parting plane vertical. The sprue
   enters at the widest, highest section of the part and ends in a pour funnel.
   Vents are placed at local high points where air would be trapped.
5. **Side pieces.** Where the two halves cannot release part of the surface
   (a side hole, the inside of a handle), side pieces are added. Each comes off
   in its own straight direction before the halves. Every piece must release
   every part of the cast it can touch on its way out, and after building, each
   piece is slid out as an exact solid to confirm it does not catch. Areas that
   no piece can release are filled and reported. See
   [docs/multi-piece-design.md](docs/multi-piece-design.md).
6. **Keys.** Conical registration keys with a clearance fit are placed on the
   parting face and on the cut faces of side pieces, clear of the cavity and the
   channels.
7. **Pieces.** All booleans run on [manifold3d](https://github.com/elalish/manifold),
   so every piece is a closed solid. Each piece is laid flat for printing.

## Limitations

- Side pieces are cut with flat planes; only the main parting surface curves.
  Shapes that lock in every direction, such as a chain link or an arm wrapped
  around a body, cannot be released by any rigid mold. moldgen
  fills those spots and reports how much, so you can decide whether to use a
  flexible silicone mold instead.
- Draft is analysed and reported, not added to the part.
- Sealed internal voids cannot be reproduced; the cast is solid there.

## Related tools

Many browser apps, Blender add-ons and small scripts now produce a two-piece,
plane-split mold from an STL. Few search the parting direction automatically,
analyse undercuts with occlusion, and derive gating and release settings from
the casting material, and none of those that do are open source under a
permissive license. [docs/prior-art.md](docs/prior-art.md) surveys about 55
open-source projects, the commercial tools and the academic literature, with
sources.

## Development

```bash
pip install -e ".[dev,repair]"
python -m pytest            # add -m "not slow" to skip the full-pipeline runs
ruff check src tests && ruff format --check src tests
```

## Background

moldgen started as a bachelor's thesis ([report_thesis.pdf](report_thesis.pdf))
and was rewritten as a package in 2026. Version 1.0 replaces the earlier
`makeMold.py` script. Its command-line flags changed, for example
`--split_axis z` is now `--direction z` and `--mold_pieces 4` is `--pieces 4`.

This work is part of the [CRAEFT project](https://craeft.eu), "Craft
Understanding, Education, Training, and Preservation for Posterity and
Prosperity", funded by the European Union's Horizon Europe programme (grant
agreement No 101094349) and coordinated by the Foundation for Research and
Technology Hellas (FORTH).

## License

MIT. See [LICENSE](LICENSE).
