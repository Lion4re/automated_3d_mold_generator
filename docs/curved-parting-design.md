# Curved parting surface

The line between the two halves no longer has to be a flat plane. When a
flat cut would trap part of the model, the halves are split along a curved
surface that follows the part's outline. Parts that already split cleanly on a
plane keep exactly the flat mold they had before. The code is in
`src/moldgen/surface.py`, and the rest of the pipeline uses it through
`CastFaces`, `core_half` and `pipeline._layout_pieces`.

## The surface

In the mold frame the top half comes off along +Z and the bottom half along
-Z. The parting surface is a height field `z = h(x, y)`: the top half is
everything above it, the bottom half everything below.

**Why any height field works.** The top only moves up and the bottom only down,
so neither can run into the other whatever shape `h` has. The removal argument
of the multi-piece design (`docs/multi-piece-design.md`) carries over
unchanged.

**What a good surface needs.** On each vertical line the surface must run
through the cast, between its lowest and highest point. Then the mold material
above the cast is in the top and the material below it is in the bottom, and
both slide off. What can still lock is material between two layers of the cast
on one vertical line. That is a true undercut for this pull, which no parting
surface can release; side pieces or filling handle it.

## Fitting it

`fit_surface` works on a grid covering the mold block, with a spacing of about
0.4 % of the block's diagonal:

1. **Spans.** For every grid point, the lowest and highest point of the cast
   above it (`parting.column_spans`, the same grid ray query the release test
   uses). The cast includes the sprue, funnel and vents.
2. **Keep the plane if it works.** If z = 0 passes through every span (to within
   0.1 mm), the surface is that plane. Parts that split cleanly on a plane are
   therefore unchanged.
3. **Otherwise follow the middle.** Over the part, the surface runs through the
   middle of each span. Towards the part's outline a span shrinks to a point,
   so the middle meets the outline exactly, at the silhouette, which is where a
   parting line belongs. Along the sprue, funnel and vents beyond the part, the
   surface is held at the channel's height so each channel stays split down its
   axis.
4. **Smooth elsewhere.** Everywhere else the surface is harmonic, as smooth as
   possible. It is solved by red-black over-relaxation, from a coarse grid up
   to the fine one, with a flat slope at the block's edges.

## Using it

- **Classifying faces.** A face belongs to the top or the bottom by its
  vertices' heights relative to the surface. A face the surface runs through
  touches both halves and must release both ways, so `CastFaces` subdivides
  those faces three more times. That keeps the strip along the outline narrow.
- **Sweep check.** The check for faces a half can run into measures each path
  down (or up) to the surface rather than to z = 0.
- **Side pieces.** Caps limited to one side of z = 0 would cut across a curved
  surface, so with one only whole-block and boxed caps are planned.
- **Choosing curved or flat.** The pipeline fits the surface after the gating
  is placed. It uses the surface only if it releases more than the plane: for
  `--pieces 2` by locked area, and for `--pieces auto` by locked area and then
  number of pieces, after a quick side-piece plan for both. `--flat-parting`
  (`MoldConfig.parting_surface = "flat"`) turns it off. Four-piece molds stay
  flat.
- **Splitting.** The halves are cut with a closed solid under the surface
  (`PartingSurface.below`), built from the same grid triangles that
  `height()` interpolates on. Filling extrudes locked faces down (or up) to the
  surface.
- **Keys.** Registration keys go on parts of the surface that slope less than
  about 19 degrees (rise over run 0.35), away from the cast and the side
  pieces. Each key's cone starts at the highest point of the surface under it,
  so it stands out by its full height all round, on a straight post that
  reaches below the lowest point, so it is anchored all round
  (`pipeline._stand_on`). The post is clipped to mold material. The same
  applies to keys on the curved cuts of side pieces.
- **Removal check.** The safety net that slides every finished piece out
  covers curved halves too.

## Results

The test tube runs along a curve that leaves every plane (`tests/test_surface.py`):
- **Flat plane:** 8.3 % of the surface undercut.
- **Curved surface:** 0.1 % with two pieces.
- **Auto mode:** picks the curved two-piece mold instead of adding side pieces.

The sample chess models and the multi-piece test parts are unchanged.

## Known limits

- **One height field per pull.** Undercuts that lie between two layers of the
  cast on one vertical line are not helped. The chess pieces split across
  their axis are an example; they still need side pieces.
- **Grid resolution.** The surface is resolved at the grid spacing. Near very
  steep or very thin parts of the outline a narrow strip can stay locked; it
  is reported, and in auto mode side pieces or filling take care of it.
- Side pieces can be curved too; see `docs/multi-piece-design.md`.
