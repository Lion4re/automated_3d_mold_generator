"""Multi-piece layouts: side pieces ("caps") taken off the mold block first.

See ``docs/multi-piece-design.md`` for the full argument. In the mold frame
(main pull +Z for the top half, -Z for the bottom half, parting plane z == 0)
the block is divided in a fixed order:

1. cap ``k`` takes what is still left of the block inside its convex reach:
   beyond its cut plane ``dot(p, d_k) >= o_k``, optionally only in the top or
   bottom half, and optionally inside side planes parallel to ``d_k`` that box
   it in around one feature;
2. what remains is split at z == 0 into the top and bottom halves.

Pieces come off in that order. Moving along ``d_k`` never leaves the cap's
reach, which holds only cap ``k`` and space earlier caps left empty, so the
only thing a piece can hit is the cast. A piece must therefore release every
cast face in the space it sweeps: for a cap that is its whole reach, for the
top and bottom halves it is everything above or below their region.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import trimesh
from manifold3d import Manifold, OpType
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from moldgen.booleans import to_manifold, to_trimesh
from moldgen.parting import (
    NORMAL_EPS,
    DirectionScore,
    PartingResult,
    mold_frame,
    ray_hit_distances,
    releasable,
    release_tolerance_for,
)

logger = logging.getLogger(__name__)

UP = np.array([0.0, 0.0, 1.0])

CANDIDATE_DIRECTIONS = 160
"""Directions sampled on the sphere when searching for a cap, plus the six axes."""

QUICK_DIRECTIONS = 60
"""Directions sampled when comparing main pull directions, where speed matters more."""

MAIN_CANDIDATES = 4
"""Best two-piece directions compared as the main pull of a multi-piece mold."""

REFINE_STEPS_DEG = (5.0, 2.5)
"""Angular steps of the local search around the best sampled cap direction."""

MIN_GAIN_FRACTION = 5e-4
"""A cap must release at least this share of the cast's surface area to be worth a piece."""

MIN_PIECE_FRACTION = 0.005
"""Side pieces smaller than this share of the part's bounding box are not worth printing;
the spot they would release is filled instead."""

SHORTLIST = 3
"""Caps compared by the locked area they really leave, after the fast estimate."""

PATCHES = 4
"""Largest locked patches that get a boxed-in cap of their own."""

PATCH_RELEASE_SHARE = 0.5
"""A direction is tried for a patch only if face normals alone release this share of it."""

BOX_MARGIN_FRACTION = 0.1
"""How far (times the cast size) a boxed-in cap reaches past the patch it releases."""

PLANE_EPSILON = 1e-6
"""Distance (times the cast size) within which a face counts as lying on a cap plane."""

MAX_PLANE_SHIFT_FRACTION = 0.02
"""Largest gap (times the cast size) left between a cap plane and the lowest face it takes.

The plane goes halfway between the faces it must leave out and the faces it
takes, up to this gap, so it never runs through the cast's own vertices:
that would leave slivers of zero thickness in the pieces."""

MAX_EDGE_FRACTION = 0.05
"""Longest triangle edge (times the cast size) in the analysis mesh.

A cut plane takes whole faces only, so a long triangle (a hole wall in a CAD
export, say) would keep every cap from cutting through it.
"""

LEAN_PREFILTER = 0.15
"""Faces leaning against a pull by more than this (as -cos of the angle, about 9 degrees)
are taken as blocked without a ray test. Faces leaning less may still release within
the release tolerance, so they are tested."""

PARALLEL_COS = float(np.cos(np.radians(1.0)))
"""Cut planes whose normals are within this angle (as a cosine) count as parallel."""

SIDES = (0, 1, -1)
"""Cap extents: the whole remaining block, its top half, its bottom half."""

Halfspace = tuple[np.ndarray, float]
"""``(n, c)`` for the half-space ``dot(p, n) >= c``."""


@dataclass
class Cap:
    """A side piece: what is left of the block inside its reach."""

    direction: np.ndarray
    """Unit pull direction in the mold frame; the cut plane is perpendicular to it."""
    offset: float
    """The cut plane is ``dot(p, direction) == offset``; the cap lies beyond it."""
    side: int = 0
    """1 or -1 limits the cap to the top or bottom half; 0 lets it span both."""
    bounds: tuple[Halfspace, ...] = ()
    """Side planes parallel to ``direction`` that box the cap in around one feature."""

    def planes(self) -> list[Halfspace]:
        """The cut plane and the side planes (the half limit is kept separately)."""
        return [(self.direction, self.offset), *self.bounds]

    def halfspaces(self) -> list[Halfspace]:
        """Every half-space of the cap's reach, the half limit included."""
        limit = [(self.side * UP, 0.0)] if self.side else []
        return [*self.planes(), *limit]


@dataclass
class PieceLayout:
    caps: list[Cap] = field(default_factory=list)
    """Side pieces in removal order; the top and bottom halves come off after them."""
    face_region: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    """Per cavity face: the piece that releases it (caps in order, then top, then
    bottom), or -1 where no piece can and the cavity was filled."""
    locked_fraction: float = 0.0
    """Share of the cast surface that no piece could release, before filling."""
    filled_volume: float = 0.0
    """Volume (mm^3) added to the cavity to fill locked areas."""
    remaining_locked_fraction: float = 0.0
    """Share of the cast surface still locked after filling; normally 0."""
    top_needed: bool = True
    """False when no cast face touches what is left of the top half after the caps,
    so that leftover block is printed as part of the bottom piece."""


class CastFaces:
    """The cast surface used for the analysis, subdivided to a maximum edge length.

    Faces of one solid that lie inside another (where the sprue meets the
    cavity) are not part of the cast's surface; they still block rays but
    carry no release requirement.
    """

    def __init__(self, cavity: trimesh.Trimesh, gating: list[trimesh.Trimesh]) -> None:
        solids = [cavity, *gating]
        joined = trimesh.util.concatenate(solids) if gating else cavity
        vertices, faces, source = trimesh.remesh.subdivide_to_size(
            joined.vertices, joined.faces, MAX_EDGE_FRACTION * joined.scale, return_index=True
        )
        self.mesh = trimesh.Trimesh(vertices, faces, process=False)
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.normals = self.mesh.face_normals
        self.source = np.asarray(source, dtype=np.int64)
        """Input face (cavity faces first, then gating) each analysis face came from."""
        self.cavity_faces = len(cavity.faces)
        solid_of_face = np.searchsorted(
            np.cumsum([len(s.faces) for s in solids]), self.source, side="right"
        )
        self.internal = np.zeros(len(faces), dtype=bool)
        centres = self.mesh.triangles_center
        for i in range(len(solids)):
            for j, other in enumerate(solids):
                if j == i:
                    continue
                lo, hi = other.bounds
                near = np.flatnonzero(
                    (solid_of_face == i) & np.all((centres >= lo) & (centres <= hi), axis=1)
                )
                if len(near):
                    self.internal[near] |= other.contains(centres[near])
        self.box_volume = float(np.prod(self.mesh.extents))
        self.area = np.where(self.internal, 0.0, self.mesh.area_faces)
        self.total_area = float(self.area.sum())
        self.tolerance = release_tolerance_for(self.mesh)
        self.plane_tolerance = 0.5 * PLANE_EPSILON * self.mesh.scale
        """Faces this close to a cap plane count as lying on it (for example the
        faces left where a filled region is clipped by the plane)."""
        low, high = self.span(UP)
        flat = (high <= self.tolerance) & (low >= -self.tolerance)
        facing_up = self.normals[:, 2] >= 0
        # Crossings of z == 0 shallower than the release tolerance do not count, and a
        # face lying in the plane belongs to the side it faces, as in the two-piece analysis.
        self.top = (high > self.tolerance) | (flat & facing_up)
        self.bottom = (low < -self.tolerance) | (flat & ~facing_up)
        self._solids = solids
        self._solid: Manifold | None = None
        self._released: dict[tuple[float, ...], np.ndarray] = {}
        self._vertical: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def vertical_reach(self, sign: int) -> tuple[np.ndarray, np.ndarray]:
        """Points just outside each face, and how far each can move towards z == 0.

        The distance stops at the parting plane or at the first cast surface
        in the way, whichever comes first (``sign`` is the half: 1 top, -1 bottom).
        """
        if sign not in self._vertical:
            origins = self.mesh.triangles_center + self.tolerance * self.normals
            length = np.maximum(sign * origins[:, 2], 0.0)
            hit = ray_hit_distances(self.mesh, origins, -sign * UP)
            self._vertical[sign] = (origins, np.minimum(length, hit))
        return self._vertical[sign]

    def solid(self) -> Manifold:
        """The cast as one manifold3d solid."""
        if self._solid is None:
            self._solid = Manifold.batch_boolean(
                [to_manifold(s, "the cast") for s in self._solids], OpType.Add
            )
        return self._solid

    def released(self, direction: np.ndarray) -> np.ndarray:
        key = tuple(np.round(direction, 12))
        if key not in self._released:
            self._released[key] = releasable(self.mesh, direction)
        return self._released[key]

    def span(self, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lowest and highest vertex of each face along ``direction``."""
        heights = (self.vertices @ direction)[self.faces]
        return heights.min(axis=1), heights.max(axis=1)

    def locked_area(self, locked: np.ndarray) -> float:
        """Share of the cast surface covered by the ``locked`` faces."""
        return float(self.area[locked].sum()) / self.total_area if self.total_area else 0.0

    def side(self, side: int) -> np.ndarray:
        """Faces in the top half (1), the bottom half (-1) or anywhere (0)."""
        if side > 0:
            return self.top
        if side < 0:
            return self.bottom
        return np.ones(len(self.faces), dtype=bool)

    def reaches(self, planes: list[Halfspace]) -> np.ndarray:
        """Faces with some part strictly inside all ``planes`` (checked plane by plane)."""
        reach = np.ones(len(self.faces), dtype=bool)
        for normal, value in planes:
            reach &= self.span(normal)[1] > value + self.plane_tolerance
        return reach

    def within(self, planes: list[Halfspace]) -> np.ndarray:
        """Faces wholly inside all ``planes``."""
        inside = np.ones(len(self.faces), dtype=bool)
        for normal, value in planes:
            inside &= self.span(normal)[0] >= value - self.plane_tolerance
        return inside


def core_sides(cast: CastFaces, caps: list[Cap]) -> tuple[np.ndarray, np.ndarray]:
    """Faces that still touch the top and the bottom half after the caps are taken."""
    in_top, in_bottom = cast.top.copy(), cast.bottom.copy()
    for cap in caps:
        taken = cast.within(cap.planes())
        if cap.side >= 0:
            in_top &= ~taken
        if cap.side <= 0:
            in_bottom &= ~taken
    return in_top, in_bottom


def owners(cast: CastFaces, caps: list[Cap]) -> np.ndarray:
    """(faces, len(caps) + 2) flags: the pieces whose mold material each cast face touches."""
    owned = np.zeros((len(cast.faces), len(caps) + 2), dtype=bool)
    for k, cap in enumerate(caps):
        in_top, in_bottom = core_sides(cast, caps[:k])
        owned[:, k] = _on_side(in_top, in_bottom, cap.side) & cast.reaches(cap.planes())
    owned[:, -2], owned[:, -1] = core_sides(cast, caps)
    return owned


def requirements(cast: CastFaces, caps: list[Cap]) -> np.ndarray:
    """(faces, len(caps) + 2) flags: the pieces each cast face must release.

    A piece must release every face its sweep can reach, not only the faces
    it touches: it may slide through space an earlier piece left empty. A
    cap never leaves its reach, so it must release every face there. The top
    and bottom halves must release every face above (below) their region.
    """
    required = np.zeros((len(cast.faces), len(caps) + 2), dtype=bool)
    for k, cap in enumerate(caps):
        required[:, k] = cast.side(cap.side) & cast.reaches(cap.planes())
    in_top, in_bottom = core_sides(cast, caps)
    required[:, -2] = in_top | _swept(cast, caps, 1)
    required[:, -1] = in_bottom | _swept(cast, caps, -1)
    return required & ~cast.internal[:, None]


def locked_faces(cast: CastFaces, caps: list[Cap]) -> np.ndarray:
    """Cast faces that some piece which must release them cannot release."""
    required = requirements(cast, caps)
    pulls = [cap.direction for cap in caps] + [UP, -UP]
    locked = np.zeros(len(cast.faces), dtype=bool)
    for k, pull in enumerate(pulls):
        locked |= required[:, k] & ~cast.released(pull)
    return locked


def plan_layout(
    cavity: trimesh.Trimesh,
    gating: list[trimesh.Trimesh],
    block_bounds: np.ndarray,
    max_pieces: int,
    *,
    caps: list[Cap] | None = None,
) -> tuple[PieceLayout, trimesh.Trimesh]:
    """Plan side pieces for ``cavity`` and return the layout and the cavity to cut.

    ``caps`` skips the search when the caps are already known. Areas no piece
    can release are filled; the returned cavity then differs from ``cavity``.
    """
    cast = CastFaces(cavity, gating)
    if caps is None:
        caps = plan_caps(cast, max_pieces - 2)
    locked = locked_faces(cast, caps)
    layout = PieceLayout(
        caps=caps,
        face_region=face_regions(cast, caps, locked),
        locked_fraction=cast.locked_area(locked),
    )
    cut = cavity
    final = cast
    if locked.any():
        cut, layout.filled_volume = fill_locked(cavity, cast, caps, block_bounds)
        cut, island_volume = _absorb_islands(cut, gating, block_bounds)
        layout.filled_volume += island_volume
        final = CastFaces(cut, gating) if layout.filled_volume > 0 else cast
        layout.remaining_locked_fraction = final.locked_area(locked_faces(final, caps))
    if caps:
        in_top, _ = core_sides(final, caps)
        layout.top_needed = bool((in_top & ~final.internal).any())
    return layout, cut


def choose_main_direction(
    mesh: trimesh.Trimesh, parting: PartingResult, max_pieces: int
) -> DirectionScore | None:
    """The two-piece candidate that leaves the least locked area once side pieces are added.

    Ties go to fewer side pieces, then to the better two-piece score. Returns
    None when the chosen two-piece direction already releases the part.
    """
    if not parting.candidates or parting.undercut_fraction == 0.0:
        return None
    best_key, best = None, None
    for rank, score in enumerate(parting.candidates[:MAIN_CANDIDATES]):
        cavity = mesh.copy().apply_transform(mold_frame(mesh, score.direction, score.offset))
        cast = CastFaces(cavity, [])
        caps = plan_caps(cast, max_pieces - 2, directions=QUICK_DIRECTIONS)
        locked = cast.locked_area(locked_faces(cast, caps))
        key = (round(locked, 3), len(caps), rank)
        logger.debug(
            "main direction %s: %.2f%% locked with %d caps",
            score.direction,
            100 * locked,
            len(caps),
        )
        if best_key is None or key < best_key:
            best_key, best = key, score
        if locked == 0.0 and not caps:
            break
    return best


class _Frame:
    """Per-direction extents of every face, shared by all options along that direction."""

    def __init__(self, cast: CastFaces, direction: np.ndarray) -> None:
        self.direction = direction
        self.low, self.high = cast.span(direction)
        self.normal_ok = cast.normals @ direction >= -LEAN_PREFILTER
        helper = np.eye(3)[np.argmin(np.abs(direction))]
        e1 = np.cross(direction, helper)
        e1 /= np.linalg.norm(e1)
        self.axes = (e1, np.cross(direction, e1))
        self.vertex_heights = [cast.vertices @ axis for axis in self.axes]
        face_heights = [h[cast.faces] for h in self.vertex_heights]
        self.face_low = [h.min(axis=1) for h in face_heights]
        self.face_high = [h.max(axis=1) for h in face_heights]


@dataclass
class _Option:
    bound: float
    frame: _Frame
    side: int
    patch: tuple[np.ndarray, np.ndarray] | None
    """Faces and vertices of the locked patch a boxed-in cap is fitted around, or
    None for a cap open on all sides but its cut plane."""
    avoid: list[float] = field(default_factory=list)
    """Positions along the direction of earlier parallel cut planes (see ``_parallel_planes``)."""


def plan_caps(
    cast: CastFaces, max_caps: int, *, directions: int = CANDIDATE_DIRECTIONS
) -> list[Cap]:
    """Greedily add the caps that release the most locked area, up to ``max_caps``."""
    caps: list[Cap] = []
    locked = locked_faces(cast, caps)
    min_gain = MIN_GAIN_FRACTION * cast.total_area
    candidates = _sphere_directions(directions)
    while len(caps) < max_caps and cast.area[locked].sum() >= min_gain:
        in_top, in_bottom = core_sides(cast, caps)
        patches = [(mask, np.unique(cast.faces[mask])) for mask in _patches(cast, locked)]
        options = []
        for direction in candidates:
            frame = _Frame(cast, direction)
            avoid = _parallel_planes(caps, direction)
            usable = [
                patch
                for patch in patches
                if cast.area[patch[0] & frame.normal_ok].sum()
                >= PATCH_RELEASE_SHARE * cast.area[patch[0]].sum()
            ]
            for side in SIDES:
                if side * direction[2] < -NORMAL_EPS:
                    continue  # a half-limited cap may not move into the other half
                for patch in [None, *usable]:
                    option = _Option(0.0, frame, side, patch, avoid)
                    option.bound, _ = _gain(cast, option, in_top, in_bottom, locked)
                    if option.bound >= min_gain:
                        options.append(option)
        options.sort(key=lambda option: -option.bound)
        shortlist: list[tuple[float, Cap, _Option]] = []
        for option in options:
            if len(shortlist) == SHORTLIST and option.bound <= shortlist[-1][0]:
                break
            gain, cap = _gain(cast, option, in_top, in_bottom, locked, exact=True)
            if cap is not None and gain >= min_gain:
                shortlist = sorted([*shortlist, (gain, cap, option)], key=lambda item: -item[0])
                shortlist = shortlist[:SHORTLIST]
        if not shortlist:
            break
        best_gain, best, best_option = shortlist[0]
        for step in REFINE_STEPS_DEG:
            for direction in _around(best.direction, step):
                if best.side * direction[2] < -NORMAL_EPS:
                    continue
                option = _Option(
                    0.0,
                    _Frame(cast, direction),
                    best.side,
                    best_option.patch,
                    _parallel_planes(caps, direction),
                )
                gain, cap = _gain(cast, option, in_top, in_bottom, locked, exact=True)
                if gain > best_gain:
                    best_gain, best = gain, cap
        # A cap can also lock faces in the paths of the pieces after it, so
        # judge the shortlist by the locked area that is actually left.
        locked_area = cast.area[locked].sum()
        true_gain, choice, choice_locked = 0.0, None, locked
        for cap in [best, *(cap for _, cap, _ in shortlist[1:])]:
            if _piece_volume(cast, caps, cap) < MIN_PIECE_FRACTION * cast.box_volume:
                continue
            after = locked_faces(cast, [*caps, cap])
            gain = locked_area - cast.area[after].sum()
            if gain > true_gain:
                true_gain, choice, choice_locked = gain, cap, after
        if choice is None or true_gain < min_gain:
            break
        caps.append(choice)
        locked = choice_locked
        logger.debug(
            "cap %d along %s (side %d, %d side planes) releases %.1f mm^2",
            len(caps),
            np.round(choice.direction, 3),
            choice.side,
            len(choice.bounds),
            true_gain,
        )
    return caps


def fill_locked(
    cavity: trimesh.Trimesh,
    cast: CastFaces,
    caps: list[Cap],
    block_bounds: np.ndarray,
) -> tuple[trimesh.Trimesh, float]:
    """Grow the cavity so the top and bottom halves release their locked faces.

    Each locked face that leans against its half's pull is extruded to the
    parting plane, clipped to that half's region of the block. Returns the
    new cavity and the volume added.
    """
    # ponytail: extruding to the parting plane over-fills when the cast has a hole
    # below a locked face; stop the extrusion at the first cast surface if that matters.
    required = requirements(cast, caps)
    locked = locked_faces(cast, caps)
    solid = to_manifold(cavity, "the cavity")
    before = solid.volume()
    for sign, column in ((1, -2), (-1, -1)):
        leaning = locked & required[:, column] & (sign * cast.normals[:, 2] < 0)
        prisms = []
        for triangle in cast.mesh.triangles[leaning]:
            flat = triangle.copy()
            flat[:, 2] = 0.0
            prism = Manifold.hull_points(np.vstack([triangle, flat]))
            if prism.volume() > 0:
                prisms.append(prism)
        if prisms:
            half = core_half(block_bounds, caps, sign)
            solid = solid + (Manifold.batch_boolean(prisms, OpType.Add) ^ half)
    if solid.volume() <= before:
        return cavity, 0.0
    return to_trimesh(solid), float(solid.volume() - before)


def block_solid(block_bounds: np.ndarray) -> Manifold:
    lo, hi = np.asarray(block_bounds, dtype=float)
    return Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


def trim(solid: Manifold, halfspaces: list[Halfspace]) -> Manifold:
    """``solid`` cut down to the intersection of ``halfspaces``."""
    for normal, value in halfspaces:
        solid = solid.trim_by_plane(tuple(np.asarray(normal, dtype=float)), float(value))
    return solid


def cap_region(cap: Cap, remaining: Manifold) -> Manifold:
    """The part of ``remaining`` that ``cap`` takes."""
    return trim(remaining, cap.halfspaces())


def core_half(block_bounds: np.ndarray, caps: list[Cap], sign: int) -> Manifold:
    """The region of the top (``sign == 1``) or bottom half left after the caps."""
    block = block_solid(block_bounds)
    region = block.trim_by_plane((0.0, 0.0, float(sign)), 0.0)
    reach = [trim(block, cap.halfspaces()) for cap in caps if cap.side in (0, sign)]
    if reach:
        region = region - Manifold.batch_boolean(reach, OpType.Add)
    return region


def face_regions(cast: CastFaces, caps: list[Cap], locked: np.ndarray) -> np.ndarray:
    """For each cavity face, the first piece it touches, or -1 if any of it is locked."""
    owned = owners(cast, caps)
    first = np.argmax(owned, axis=1)
    first[~owned.any(axis=1)] = len(caps)
    first[locked] = -1
    on_cavity = cast.source < cast.cavity_faces
    regions = np.full(cast.cavity_faces, len(caps), dtype=np.int64)
    # Fancy assignment keeps the last write, so write in reverse to keep the first.
    regions[cast.source[on_cavity][::-1]] = first[on_cavity][::-1]
    regions[cast.source[on_cavity & locked]] = -1
    return regions


def _absorb_islands(
    cavity: trimesh.Trimesh, gating: list[trimesh.Trimesh], block_bounds: np.ndarray
) -> tuple[trimesh.Trimesh, float]:
    """Add to the cavity any mold material that filling has sealed inside the cast.

    Such an island would float free of every piece, so the cast takes its place.
    """
    cast = to_manifold(cavity, "the cavity")
    body = block_solid(block_bounds) - Manifold.batch_boolean(
        [cast, *(to_manifold(g, "a gating channel") for g in gating)], OpType.Add
    )
    lo, hi = np.asarray(block_bounds, dtype=float)
    margin = PLANE_EPSILON * float(np.linalg.norm(hi - lo))
    islands = []
    for part in body.decompose():
        box = np.asarray(part.bounding_box()).reshape(2, 3)
        if np.all(box[0] > lo + margin) and np.all(box[1] < hi - margin):
            islands.append(part)
    if not islands:
        return cavity, 0.0
    grown = cast + Manifold.batch_boolean(islands, OpType.Add)
    return to_trimesh(grown), float(sum(part.volume() for part in islands))


def _swept(cast: CastFaces, caps: list[Cap], sign: int) -> np.ndarray:
    """Faces the top (``sign == 1``) or bottom half can run into on its way out.

    The half moves along ``sign * Z``. Its material can reach a face if,
    starting just outside the face and moving towards the parting plane, one
    enters the half's region (the block minus the caps' reach) before meeting
    the cast.
    """
    if not any(cap.side in (0, sign) for cap in caps):
        # The half's region is the whole half, which core_sides already covers.
        return np.zeros(len(cast.faces), dtype=bool)
    origins, reach = cast.vertical_reach(sign)
    return _enters_core(origins, reach, caps, sign, cast) & ~cast.internal


def _enters_core(
    points: np.ndarray, length: np.ndarray, caps: list[Cap], sign: int, cast: CastFaces
) -> np.ndarray:
    """Whether the segment from each point a ``length`` towards z == 0 leaves the caps' reach.

    Each cap's reach covers one stretch of the segment; the segment enters the
    half's region where those stretches leave a gap.
    """
    eps = cast.plane_tolerance
    stretches = []
    for cap in caps:
        if cap.side not in (0, sign):
            continue
        # The point p - sign * t * Z is inside the plane dot(x, n) >= c when a - b * t >= 0.
        start, end = np.zeros(len(points)), length.copy()
        for normal, value in cap.planes():
            a = points @ normal - value
            b = sign * normal[2]
            if abs(b) < 1e-12:
                end = np.where(a >= 0, end, -np.inf)
            elif b > 0:
                end = np.minimum(end, a / b)
            else:
                start = np.maximum(start, a / b)
        stretches.append((start, end))
    reach = np.zeros(len(points))
    gap = np.zeros(len(points), dtype=bool)
    if stretches:
        starts = np.column_stack([s for s, _ in stretches])
        ends = np.column_stack([e for _, e in stretches])
        order = np.argsort(starts, axis=1)
        starts = np.take_along_axis(starts, order, axis=1)
        ends = np.take_along_axis(ends, order, axis=1)
        for j in range(starts.shape[1]):
            valid = ends[:, j] > starts[:, j]
            gap |= valid & (starts[:, j] > reach + eps)
            reach = np.where(valid, np.maximum(reach, ends[:, j]), reach)
    return (length > eps) & (gap | (reach < length - eps))


def _patches(cast: CastFaces, locked: np.ndarray) -> list[np.ndarray]:
    """The largest edge-connected patches of locked faces, as face masks."""
    pairs = cast.mesh.face_adjacency
    both = locked[pairs[:, 0]] & locked[pairs[:, 1]]
    n = len(cast.faces)
    graph = coo_matrix(
        (np.ones(int(both.sum()), dtype=np.int8), (pairs[both, 0], pairs[both, 1])), (n, n)
    )
    _, labels = connected_components(graph, directed=False)
    areas = np.bincount(labels, weights=np.where(locked, cast.area, 0.0))
    biggest = np.argsort(-areas)[:PATCHES]
    return [locked & (labels == label) for label in biggest if areas[label] > 0]


def _on_side(in_top: np.ndarray, in_bottom: np.ndarray, side: int) -> np.ndarray:
    if side > 0:
        return in_top
    if side < 0:
        return in_bottom
    return in_top | in_bottom


def _gain(
    cast: CastFaces,
    option: _Option,
    in_top: np.ndarray,
    in_bottom: np.ndarray,
    locked: np.ndarray,
    *,
    exact: bool = False,
) -> tuple[float, Cap | None]:
    """Locked area a cap for ``option`` releases, and the smallest such cap.

    The cap may not take any face its direction cannot release, which sets
    the lowest allowed cut plane. Locked faces wholly beyond it are released,
    and the plane then moves up to the lowest of those so the cap takes no
    more of the cast than it needs. A boxed-in cap only reaches faces inside
    its side planes. Without ``exact`` only face normals are tested, which
    gives an upper bound on the gain.
    """
    frame = option.frame
    direction = frame.direction
    reachable = cast.side(option.side) & ~cast.internal
    inside_box = np.ones(len(cast.faces), dtype=bool)
    bounds: list[Halfspace] = []
    if option.patch is not None:
        faces, vertices = option.patch
        if not (faces & frame.normal_ok).any():
            return 0.0, None
        margin = BOX_MARGIN_FRACTION * cast.mesh.scale
        tol = cast.plane_tolerance
        for axis, heights, low, high in zip(
            frame.axes, frame.vertex_heights, frame.face_low, frame.face_high, strict=True
        ):
            lo = float(heights[vertices].min()) - margin
            hi = float(heights[vertices].max()) + margin
            bounds += [(axis, lo), (-axis, -hi)]
            reachable &= (high > lo + tol) & (low < hi - tol)
            inside_box &= (low >= lo - tol) & (high <= hi + tol)
    released = frame.normal_ok
    # The cap must release every face in its reach, owned by an earlier cap or not.
    floor = _highest(frame.high, reachable & ~released)
    if exact:
        # Faces at or below the normal-only floor cannot change the result.
        query = reachable & released & (frame.high > floor)
        if query.any():
            released = releasable(cast.mesh, direction, faces=query)
            floor = _highest(frame.high, reachable & ~released)
    freed = locked & _on_side(in_top, in_bottom, option.side) & released & inside_box
    freed &= frame.low >= floor
    if option.side:
        freed &= ~_on_side(in_top, in_bottom, -option.side)  # still held by the other half
    if not freed.any():
        return 0.0, None
    lowest = float(frame.low[freed].min())
    offset = _place_plane(floor, lowest, option.avoid, MAX_PLANE_SHIFT_FRACTION * cast.mesh.scale)
    cap = Cap(direction=direction, offset=offset, side=option.side, bounds=tuple(bounds))
    return float(cast.area[freed].sum()), cap


def _piece_volume(cast: CastFaces, caps: list[Cap], cap: Cap) -> float:
    """Mold material ``cap`` would take within the cast's bounding box (a lower bound)."""
    box = block_solid(cast.mesh.bounds)
    taken = [trim(box, earlier.halfspaces()) for earlier in caps]
    region = trim(box, cap.halfspaces()) - cast.solid()
    if taken:
        region = region - Manifold.batch_boolean(taken, OpType.Add)
    return float(region.volume())


def _parallel_planes(caps: list[Cap], direction: np.ndarray) -> list[float]:
    """Positions along ``direction`` of cut planes parallel to a new cap's plane.

    Includes the planes of earlier caps facing the same or the opposite way
    and, for a vertical direction, the parting plane.
    """
    planes = [0.0] if abs(direction[2]) >= PARALLEL_COS else []
    for cap in caps:
        alignment = float(cap.direction @ direction)
        if alignment >= PARALLEL_COS:
            planes.append(cap.offset)
        elif alignment <= -PARALLEL_COS:
            planes.append(-cap.offset)
    return planes


def _place_plane(floor: float, lowest: float, avoid: list[float], gap: float) -> float:
    """Cut plane position between ``floor`` (exclusive) and ``lowest`` (inclusive).

    Halfway between the two, at most ``gap`` below ``lowest``; then moved off
    any parallel plane in ``avoid`` that it would nearly meet, preferring to
    overlap the earlier piece (whose space is already empty) over leaving a
    thin slab between the two planes.
    """
    offset = lowest - min((lowest - floor) / 2.0, gap)
    for plane in avoid:
        if abs(offset - plane) < gap:
            if plane - gap > floor:
                offset = plane - gap
            elif plane + gap <= lowest:
                offset = plane + gap
            else:
                # No room for a full gap: take the end of the range farthest from the plane.
                offset = lowest if lowest - plane >= plane - floor else (floor + plane) / 2.0
    return offset


def _highest(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].max()) if mask.any() else -np.inf


def _sphere_directions(count: int) -> np.ndarray:
    """The six axis directions, then ``count`` near-uniform unit vectors (Fibonacci sphere)."""
    i = np.arange(count) + 0.5
    z = 1.0 - 2.0 * i / count
    phi = np.pi * (1.0 + 5**0.5) * i
    r = np.sqrt(1.0 - z * z)
    sphere = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return np.vstack([np.eye(3), -np.eye(3), sphere])


def _around(direction: np.ndarray, step_deg: float) -> np.ndarray:
    """Eight directions ``step_deg`` away from ``direction``."""
    helper = np.eye(3)[np.argmin(np.abs(direction))]
    e1 = np.cross(direction, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    angle = np.radians(step_deg)
    turns = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    tilt = np.cos(turns)[:, None] * e1 + np.sin(turns)[:, None] * e2
    out = np.cos(angle) * direction + np.sin(angle) * tilt
    return out / np.linalg.norm(out, axis=1, keepdims=True)
