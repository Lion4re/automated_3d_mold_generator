# Multi-piece rigid molds

`--pieces auto` (the default) adds side pieces wherever the two halves of a
classic mold cannot release the part. Parts that already work as two pieces get
exactly the mold they got before. The code is in `src/moldgen/pieces.py`, and
the piece assembly is in `pipeline._layout_pieces`.

Decisions made with the project owner: organic parts first, CAD parts too.
Areas no piece can release are filled in the cavity and reported. The layout is
automatic, and the GUI preview colours the part by piece.

## Piece model

All geometry is in the mold frame. The top half is pulled along +Z, the bottom
half along -Z, and the parting plane is `z = 0`.

The block is taken apart in a fixed order:

1. **Side pieces (caps)** `k = 1..K`. Cap `k` has a pull direction `d_k`. Its
   *reach* is the convex region where all of these hold:
   - `p . d_k >= o_k` (its cut plane);
   - optionally `z >= 0` or `z <= 0`, which limits the cap to the top or
     bottom half. This is allowed only when `d_k` does not point into the
     other half.
   - optionally four *side planes* with normals perpendicular to `d_k`, which
     box the cap in around one feature.

   The cap takes whatever is still left of the block inside its reach.
2. **Top and bottom**: what remains, split at `z = 0`. If no cast face touches
   the leftover top block, it is printed as part of the bottom piece.

## Why the pieces come apart

Pieces are removed in the order above. Moving along `d_k` never leaves the
cap's reach: the cut plane is crossed in the right direction, side planes are
parallel to the motion, and a half limit is never crossed. The reach holds
only the cap itself and space that earlier caps left empty, so a cap cannot hit
another piece. The top and bottom move into `z > 0` and `z < 0`, which are
equally empty by then.

The cast is what remains to hit. The cast here is the cavity plus the sprue,
funnel and vent solids, since those fill with material too. A piece may slide
through space an earlier piece vacated, so it must release every cast face in
the space it sweeps, not only the faces it touches:

- **cap `k`**: every face in its reach;
- **top / bottom**: every face it owns, plus every face it can run into on the
  way out. That is a face whose straight path down (up) to the parting plane
  enters the half's region before meeting the cast. These paths are cast with
  a grid-based vertical ray query and cached per cast.

"Releases" means the occlusion-aware release test of the parting analysis
(`parting.releasable`), with its 0.1 mm tolerance. The analysis runs on the
cast surface subdivided to edges of at most 5 % of its size, so a cut plane can
take part of a long CAD triangle.

After building, `pipeline._removal_warnings` slides every piece along its pull
as an exact manifold3d solid. If one overlaps the cast or a piece still in
place, the user is warned. This is a safety net for the planner and has not
triggered on the test parts.

## Choosing the caps

1. The main direction is the best two-piece direction, unless another of the
   four best two-piece candidates leaves less locked area once caps are added
   (`choose_main_direction`).
2. The pour side is chosen in the same way among the four options
   (`pipeline._gating_for_pieces`). The sprue lies in the parting plane, and a
   cap crossing it must slide off the sprue stub as well.
3. Greedy cover, one cap at a time:
   - **Candidates**: about 160 directions on a Fibonacci sphere plus the axes,
     times three extents (whole, top half, bottom half), times "open" or
     "boxed around one of the four largest locked patches".
   - **Floor**: for each candidate, every face the direction cannot release
     sets the lowest allowed plane. Locked faces wholly beyond the floor are
     released.
   - **Pruning**: a normals-only pass gives an upper bound on each candidate's
     gain. Exact ray tests run only for faces leaning less than about 9 degrees,
     and only on the most promising candidates.
   - **Choice**: the best candidate is refined in 5 and 2.5 degree steps. The
     best few are then compared by the locked area they actually leave, since a
     cap can put faces in the path of the halves. A cap that would hold less
     than 0.5 % of the part's bounding box in mold material is skipped: such a
     piece is impractical to print and handle, so the spot is filled instead.
4. **Plane position.** The cut plane goes halfway between the floor and the
   lowest released face, at most 2 % of the part size from that face, so it
   never runs through the cast's own vertices. It is also kept at least that
   far from parallel cut planes and from the parting plane. Where it would
   nearly meet an opposite cap's plane it overlaps it instead, because that
   space is already empty. Otherwise both would leave a zero-thickness sliver.

## Curved side pieces

With `side_piece_cuts = "auto"` (the default; `--flat-side-pieces` turns it
off), a side piece can be cut with a height field along its pull instead of a
plane: the piece takes everything with `dot(p, d) >= g(u, v)`, where `(u, v)`
are coordinates across the pull (`surface.CutSurface`).

- **Why it still comes apart.** Moving along `d` raises `dot(p, d)` and leaves
  `(u, v)` unchanged, so the piece never leaves its reach, exactly as with a
  plane. The rule is the same: every cast face in the reach must release
  along `d`.
- **The cut** (`surface.fit_cut`). On each line along the pull that meets the
  cast, the cut runs just inside the cast's far end. The piece therefore takes
  all mold beyond the cast on that line and nothing else. On lines that miss
  the cast it is smooth (harmonic). It is then held behind the vertices of the
  faces the piece should take, and in front of any face in its reach that the
  pull cannot release, and refitted, up to three rounds.
- **No global floor.** A flat cut must clear the highest blocking face anywhere
  in its box. A curved cut only has to clear blocking faces on their own lines.
  Inside a ring handle, two opposite curved pieces meet exactly at the hole's
  narrowest point, with no strip left over.
- **Search.**
  - Curved pieces are boxed in around one locked patch. An open one would take
    every bit of mold the cast shows along its pull.
  - They are tried along the axes and along each patch's own axes: the
    direction its faces are most nearly parallel to (a hole's axis) and its
    average facing direction.
  - Each candidate is also tried together with its opposite piece around the
    same patch, and candidates are compared by the locked area released per
    piece. A one-at-a-time greedy choice otherwise prefers oblique pieces that
    free a little more at first and leave the rest harder to reach.
- **Checks.** The planner tests faces against the cut at their vertices. The
  test for what a half sweeps past samples each path against curved pieces.
  Both are close but not exact, so the finished pieces are slid out as exact
  solids as before.
- **Choosing curved or flat.** The search is greedy, so when the plan uses any
  curved cut, the mold is also built with flat side pieces only, and the
  better one is kept. The ranking is: no catching piece, then less locked
  area, then fewer pieces, then less filling (`pipeline._mold_score`). If a
  layout without curved side pieces still catches and used a curved parting
  surface, it is rebuilt with a flat one. Auto mode is therefore never worse
  than flat cuts.
- **Keys** go on the stretches of the cut that run through mold rather than
  along the cast, where the cut slopes gently. There must be mold behind them
  that no earlier piece took.

Results:

| Part | Flat side pieces | Curved side pieces |
|---|---|---|
| Knight split across its axis | 5 pieces, 0.06 cm³ filled | 3 pieces, nothing filled |
| Ball with two ring handles | 4 pieces, 0.16 cm³ filled | 4 pieces, nothing filled |

On the pawn, the bishop and the drilled blocks the flat layout was as good or
better and was kept.

## Locked areas

Faces still locked after the caps are filled:
- **Prisms.** Each locked face that leans against its half's pull is extruded
  to the parting plane. The prism is clipped to that half's region and added to
  the cavity.
- **Islands.** Mold material that filling seals inside the cast would float
  free of every piece, so it joins the cavity as well.

The rule is then re-checked on the filled cast. The report gives the locked
share before filling, the volume added and the share still flagged after it.

## Keys

- **Open side pieces** carry male keys on their cut face, pointing back along
  `-d_k`. The pieces behind get the sockets, so the side piece slides off its
  keys and nothing protrudes from the pieces still in place.
- **Boxed side pieces** sit in a pocket of the halves, which locates them, so
  missing keys there are not reported.
- **Main seam**: the halves keep their keys on the part of `z = 0` that no cap
  took. Two keys are considered enough there.
- **Clearance from other seams**: thin slabs around every other cut plane are
  key obstacles. `keys.plan_keys_on_plane` places keys on any plane and region.

## Printing and output

- **Printing:** each piece is rotated so that the opposite of its pull points
  up, then rested on the bed. Its cut face is up and the cavity opens upwards.
- **Pieces:** `MoldPiece.pull` gives the pull direction. `MoldResult.pieces` is
  in removal order.
- **Layout:** `MoldResult.layout` gives the caps, the piece releasing each part
  face (for the GUI), and the locked and filled figures.
- **Instructions:** `INSTRUCTIONS.txt` lists the assembly order and the
  removal order with each piece's direction. `report.json` includes the layout.

## Tests

`tests/test_pieces.py` covers the following:
- the planner frees a cross-drilled block that locks in two pieces;
- parts that release in two pieces get no caps and an identical mold;
- a piece limit leads to filling;
- plane placement;
- an independent sweep check that slides each piece out and measures the
  overlap with the cast and the remaining pieces.

## Known limits

- **Greedy order.** Caps are planned one (or one opposite pair) at a time, so
  some shapes get more pieces than an optimal layout would.
- **Flat cuts in holes.** Where two flat caps meet in the narrowest part of a
  curved hole, a strip about one face wide can be left. It is filled and shows
  up as thin flash in the hole. Curved caps avoid this when they are chosen.
- **Over-filling.** Prism filling extrudes to the parting plane, so it can
  over-fill below a locked face when the cast has a hole underneath. The
  reported filled volume shows this.
- **Gating.** The sprue and vents are planned for the two halves first. The
  planner only chooses among the four pour sides; it does not reroute the sprue
  around side pieces.
- **Speed.** Parts that need side pieces take about 20 to 70 s on a laptop,
  because a layout with curved side pieces is compared with one with flat
  side pieces (`--flat-side-pieces` skips that). Parts that need no side
  pieces cost at most half a second more than the two-piece mold.
