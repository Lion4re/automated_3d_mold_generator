# moldgen

Turn a 3D model into a casting mold you can 3D print.

moldgen takes a mesh (STL, OBJ, PLY, OFF, 3MF, glTF/GLB), finds the direction in
which a rigid two-piece mold releases the part best, places the parting plane,
and builds print-ready mold pieces with a pour sprue and funnel, air vents and
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
| `--pieces 2\|4` | Split each half once more for 4 pieces | `2` |
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
   is placed where the fewest faces are trapped.
3. **Mold block.** The block is aligned with the minimum-area rectangle around
   the part, so it uses as little filament as possible. The cavity is enlarged to
   compensate for the material's shrinkage.
4. **Gating.** The mold is poured with the parting plane vertical. The sprue
   enters at the widest, highest section of the part and ends in a pour funnel.
   Vents are placed at local high points where air would be trapped.
5. **Keys.** Conical registration keys with a clearance fit are placed on the
   parting face, clear of the cavity and the channels. Four-piece molds get keys
   on the second seam too.
6. **Pieces.** All booleans run on [manifold3d](https://github.com/elalish/manifold),
   so every piece is a closed solid. Each piece is laid flat for printing.

## Limitations

- The parting surface is a plane. Parts with undercuts in every direction
  (for example a figure with arms wrapped around its body) cannot be released
  by a rigid two- or four-piece mold; moldgen reports the undercut share so you
  can decide to use a flexible silicone mold instead.
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
