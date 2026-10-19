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

- **Flat cuts.** Cuts are flat and caps are peeled in a fixed, greedy order.
  Where two caps meet in the narrowest part of a curved hole, a strip about one
  face wide can be left. It is filled and shows up as thin flash in the hole.
- **Over-filling.** Prism filling extrudes to the parting plane, so it can
  over-fill below a locked face when the cast has a hole underneath. The
  reported filled volume shows this.
- **Gating.** The sprue and vents are planned for the two halves first. The
  planner only chooses among the four pour sides; it does not reroute the sprue
  around side pieces.
- **Speed.** Parts that need side pieces take about 5 to 15 s on a laptop.
  Parts that do not need them cost at most half a second more than the
  two-piece mold.
