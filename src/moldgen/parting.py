"""Parting direction search, parting plane placement and undercut analysis.

Conventions
-----------
Input frame
    The coordinate frame of the loaded part (millimetres).
Parting plane
    The plane ``dot(p, direction) == offset`` in the input frame. The mold
    halves separate along ``+direction`` (top half) and ``-direction`` (bottom half).
Mold frame
    ``to_mold`` maps the input frame to the mold frame, in which the parting
    plane is ``z == 0``, ``+Z`` is ``direction`` and the X/Y axes are aligned
    with the minimum-area rectangle around the part's projection onto the
    plane, so an axis-aligned box around the part wastes as little material as
    possible.

Release test
------------
A face lying above the plane must be released by the top half, one below by
the bottom half, and a face the plane cuts by both, since each half holds part
of it. Crossings shallower than the release tolerance do not count as cuts.
The tolerance is capped at a small share of the part size, so it never
swallows the features of a tiny part.
A face is released by a half if a ray along that half's pull, starting just
outside the face (lifted along its normal by the release tolerance), leaves
the part. On a closed, consistently oriented mesh the first surface such a ray
meets faces the ray, so only those triangles can block it. That includes the
face itself: a face leaning against the pull blocks its own ray once the lean
across the face exceeds the tolerance. Interference shallower than the
tolerance, such as tessellation noise on a flat base, is ignored. Without
occlusion only the normal is tested.

All rays of one half are parallel, so the ray test is a 2D point location in
the plane perpendicular to the pull: triangles are binned into a uniform grid
and each ray is tested against the triangles of its cell.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

logger = logging.getLogger(__name__)

FACE_OK = 0
FACE_LOW_DRAFT = 1
FACE_UNDERCUT = 2

NORMAL_EPS = float(np.sin(np.radians(0.05)))
"""A face counts as facing against the pull only beyond this tilt, so exactly vertical walls are not undercuts."""

RELEASE_TOLERANCE_MM = 0.1
"""Undercuts shallower than this are ignored: about one FDM layer, below what a printed mold resolves."""

MAX_TOLERANCE_SHARE = 0.01
"""Cap on the release tolerance (times the part size), so it cannot swallow the features of tiny parts."""

RAY_NUDGE = 1e-5
"""Minimum lift of ray origins off their face (times the part size), against self-hits from round-off."""

PLANE_TOLERANCE = 1e-6
"""Minimum plane tolerance (times the part size), so faces lying in the plane may go with either half."""

FACING_EPS = 1e-6
"""Occluders must face the ray at least this much; grazing triangles cannot block it on their own."""

BARYCENTRIC_TOLERANCE = 1e-9
"""Rays through a triangle edge or vertex count as hits."""

GRID_CELL_SCALE = 2.0
"""Grid cell edge as a multiple of sqrt(footprint area / triangle count); tuned on the sample models."""

PAIR_BATCH = 500_000
"""Upper bound on (ray, triangle) pairs tested at once, which bounds peak memory."""

COARSE_MAX_FACES = 20_000
"""Faces sampled (area-weighted) for the normal-only direction scan of large meshes."""

COARSE_SEED = 0
FIBONACCI_DIRECTIONS = 200
"""Hemisphere samples of the direction scan, roughly 10 degrees apart."""

DOMINANT_NORMALS = 6
"""Largest flat regions whose normals join the direction scan: the faces of a box."""

NORMAL_GROUP_DECIMALS = 6
"""Faces whose normals agree to this many decimals count as one flat region."""

COARSE_BINS = 512
"""Plane positions tried per direction in the scan, evenly spaced over the part's extent."""

COARSE_DIRECTION_BATCH = 64
"""Directions scored per vectorised step of the scan, which bounds peak memory."""

REFINE_COUNT = 8
"""Directions re-scored with occlusion on the full mesh, besides the three axes."""

CANDIDATE_SEPARATION_DEG = 15.0
"""Minimum angle between re-scored directions, so the candidate list offers real alternatives."""

LOCAL_SEARCH_START_DEG = 5.0
"""First ring radius of the local search: half the spacing of the hemisphere samples."""
LOCAL_SEARCH_ROUNDS = 3
LOCAL_SEARCH_RING = 8

EQUAL_UNDERCUT_TOLERANCE = 0.001
"""Undercut fractions this close are ties: axes win them, then the lower low-draft fraction."""

DUPLICATE_DIRECTION_DEG = 2.0
"""Refined directions closer than this to an axis or to each other are dropped as duplicates."""

AXIS_ALIGNED_COS = 1.0 - 1e-9

COST_ROUNDOFF = 1e-9
"""Plane positions whose undercut area is within this share of the total area of the minimum tie."""

FOOTPRINT_MIN_GAIN = 0.01
"""Rotate the mold frame about Z only if that shrinks the footprint rectangle by more than this share."""


@dataclass
class DirectionScore:
    direction: np.ndarray
    offset: float
    undercut_fraction: float
    low_draft_fraction: float


@dataclass
class PartingResult:
    direction: np.ndarray
    """Unit demolding direction of the top half, input frame."""
    offset: float
    """Plane position: ``dot(p, direction) == offset`` (mm)."""
    to_mold: np.ndarray
    """4x4 homogeneous transform from the input frame to the mold frame."""
    face_class: np.ndarray
    """Per-face classification of the analysed mesh: FACE_OK, FACE_LOW_DRAFT or FACE_UNDERCUT."""
    undercut_fraction: float
    """Share of the surface area that cannot be released by either half."""
    low_draft_fraction: float
    """Share of the surface area with less draft than the threshold (but no undercut)."""
    candidates: list[DirectionScore] = field(default_factory=list)
    """Best scoring directions, best first. Includes the chosen direction."""

    @property
    def from_mold(self) -> np.ndarray:
        return np.linalg.inv(self.to_mold)


@dataclass
class _FaceSample:
    """Area-weighted face sample for the normal-only direction scan."""

    normals: np.ndarray
    triangles: np.ndarray
    weights: np.ndarray


@dataclass
class _Evaluation:
    direction: np.ndarray
    offset: float
    face_class: np.ndarray
    undercut_fraction: float
    low_draft_fraction: float

    def score(self) -> DirectionScore:
        return DirectionScore(
            self.direction.copy(), self.offset, self.undercut_fraction, self.low_draft_fraction
        )


def classify_faces(
    mesh: trimesh.Trimesh,
    direction: np.ndarray,
    offset: float,
    *,
    draft_threshold_deg: float = 1.0,
    occlusion: bool = True,
    release_tolerance: float = RELEASE_TOLERANCE_MM,
) -> np.ndarray:
    """Classify each face for a planar two-part mold.

    A face above the plane must be releasable by moving the top half along
    ``+direction``; a face below must be releasable along ``-direction``.
    Without ``occlusion``, releasable means the face normal does not point
    against the pull. With it, a ray along the pull from just outside the
    face must leave the part, which ignores interference shallower than
    ``release_tolerance`` (mm, see the module docstring). Faces lying
    in the plane may be released by either half; faces the plane cuts must
    be released by both. Released faces with less draft than
    ``draft_threshold_deg`` along their pull are low-draft.
    """
    d = _unit(direction)
    above, below = _placement(mesh, d, offset, release_tolerance)
    up_ok, down_ok = _release(
        mesh, d, occlusion, release_tolerance, test_up=above | ~below, test_down=below | ~above
    )
    return _classify(mesh.face_normals @ d, above, below, up_ok, down_ok, draft_threshold_deg)


def best_offset(
    mesh: trimesh.Trimesh,
    direction: np.ndarray,
    *,
    occlusion: bool = True,
    release_tolerance: float = RELEASE_TOLERANCE_MM,
) -> float:
    """Return the plane position along ``direction`` with the least undercut area.

    Among equally good positions, the middle of the widest optimal interval is
    returned. The result lies strictly inside the part's extent along
    ``direction`` whenever the part has any thickness along it.
    """
    d = _unit(direction)
    up_ok, down_ok = _release(mesh, d, occlusion, release_tolerance)
    return _optimal_offset(mesh, d, up_ok, down_ok, release_tolerance)


def mold_frame(mesh: trimesh.Trimesh, direction: np.ndarray, offset: float) -> np.ndarray:
    """Return the 4x4 transform from the input frame to the mold frame (see module docstring)."""
    tilt = _rotation_to_z(_unit(direction))
    xy = mesh.vertices @ tilt[:2].T
    hull = _hull_points(xy)
    angle = _min_area_rectangle_angle(hull)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    spin = np.array([[cos_a, sin_a, 0.0], [-sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
    rotation = spin @ tilt
    footprint = hull @ spin[:2, :2].T
    centre = (footprint.min(axis=0) + footprint.max(axis=0)) / 2.0

    to_mold = np.eye(4)
    to_mold[:3, :3] = rotation
    to_mold[:3, 3] = [-centre[0], -centre[1], -float(offset)]
    return to_mold


def analyze_parting(
    mesh: trimesh.Trimesh,
    direction: np.ndarray | None = None,
    offset: float | None = None,
    *,
    draft_threshold_deg: float = 1.0,
    release_tolerance: float = RELEASE_TOLERANCE_MM,
) -> PartingResult:
    """Pick (or evaluate) the demolding direction and parting plane for ``mesh``.

    ``direction=None`` searches for the direction with the least undercut
    area; ``offset=None`` places the plane with :func:`best_offset`. An
    ``offset`` without a ``direction`` is rejected because it has no meaning
    before the direction is known.
    """
    if len(mesh.faces) == 0 or mesh.area <= 0.0:
        raise ValueError("cannot analyse a mesh without surface area")
    if direction is None:
        if offset is not None:
            raise ValueError("a parting offset needs an explicit direction")
        ranked = _search_directions(mesh, draft_threshold_deg, release_tolerance)
        chosen = ranked[0]
        logger.info(
            "Parting direction %s, offset %.3f mm, undercut %.2f %%",
            np.round(chosen.direction, 4),
            chosen.offset,
            100.0 * chosen.undercut_fraction,
        )
    else:
        chosen = _evaluate(mesh, _unit(direction), draft_threshold_deg, release_tolerance, offset)
        ranked = [chosen]

    return PartingResult(
        direction=chosen.direction,
        offset=chosen.offset,
        to_mold=mold_frame(mesh, chosen.direction, chosen.offset),
        face_class=chosen.face_class,
        undercut_fraction=chosen.undercut_fraction,
        low_draft_fraction=chosen.low_draft_fraction,
        candidates=[e.score() for e in ranked[:REFINE_COUNT]],
    )


def releasable(
    mesh: trimesh.Trimesh,
    direction: np.ndarray,
    *,
    faces: np.ndarray | None = None,
    release_tolerance: float = RELEASE_TOLERANCE_MM,
) -> np.ndarray:
    """Per-face flags: whether mold material on the face slides off along ``direction``.

    Uses the occlusion-aware release test described in the module docstring.
    With a boolean ``faces`` mask, rays are cast only for those faces; the
    others are judged by their normal alone, which can only err towards
    "released".
    """
    up, _ = _release(
        mesh,
        _unit(direction),
        True,
        release_tolerance,
        test_up=faces,
        test_down=np.zeros(len(mesh.faces), dtype=bool),
    )
    return up


def release_tolerance_for(
    mesh: trimesh.Trimesh, release_tolerance: float = RELEASE_TOLERANCE_MM
) -> float:
    """The release tolerance in effect for ``mesh`` (mm), capped for very small parts."""
    return _tolerance(mesh, release_tolerance)


def _unit(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"direction must be a non-zero finite 3-vector, got {vector!r}")
    return v / norm


def _face_span(mesh: trimesh.Trimesh, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest vertex height of each face along ``direction``."""
    heights = mesh.triangles @ direction
    return heights.min(axis=1), heights.max(axis=1)


def _tolerance(mesh: trimesh.Trimesh, release_tolerance: float) -> float:
    """The release tolerance in effect for ``mesh`` (mm), see ``MAX_TOLERANCE_SHARE``."""
    scale = mesh.scale
    return float(np.clip(release_tolerance, PLANE_TOLERANCE * scale, MAX_TOLERANCE_SHARE * scale))


def _placement(
    mesh: trimesh.Trimesh, direction: np.ndarray, offset: float, release_tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face flags: above the plane, below it (both for faces in it, neither for cut faces)."""
    lo, hi = _face_span(mesh, direction)
    tolerance = _tolerance(mesh, release_tolerance)
    return lo >= offset - tolerance, hi <= offset + tolerance


def _classify(
    normal_dot: np.ndarray,
    above: np.ndarray,
    below: np.ndarray,
    up_ok: np.ndarray,
    down_ok: np.ndarray,
    draft_threshold_deg: float,
) -> np.ndarray:
    in_plane = above & below
    released = np.select(
        [in_plane, above, below], [up_ok | down_ok, up_ok, down_ok], up_ok & down_ok
    )
    draft = np.select(
        [in_plane, above, below],
        [np.abs(normal_dot), normal_dot, -normal_dot],
        -np.abs(normal_dot),
    )
    face_class = np.full(len(normal_dot), FACE_OK, dtype=np.int8)
    face_class[draft < np.sin(np.radians(draft_threshold_deg))] = FACE_LOW_DRAFT
    face_class[~released] = FACE_UNDERCUT
    return face_class


def _release(
    mesh: trimesh.Trimesh,
    direction: np.ndarray,
    occlusion: bool,
    tolerance: float,
    *,
    test_up: np.ndarray | None = None,
    test_down: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face flags: released by pulling along ``+direction`` / ``-direction``.

    With ``occlusion``, rays are cast only for the faces selected by
    ``test_up`` / ``test_down`` (default: all); the other faces keep the
    normal-only verdict.
    """
    normal_dot = mesh.face_normals @ direction
    up_ok = normal_dot >= -NORMAL_EPS
    down_ok = normal_dot <= NORMAL_EPS
    if not occlusion:
        return up_ok, down_ok

    frame = _ray_frame(direction)
    local_tris = mesh.triangles @ frame.T
    lift = max(_tolerance(mesh, tolerance), RAY_NUDGE * mesh.scale)
    start = (mesh.triangles_center + lift * mesh.face_normals) @ frame.T
    for sign, ok, test in ((1.0, up_ok, test_up), (-1.0, down_ok, test_down)):
        query = np.flatnonzero(np.ones(len(ok), dtype=bool) if test is None else test)
        if len(query) == 0:
            continue
        occluder = sign * normal_dot < -FACING_EPS
        tris = local_tris[occluder]
        tris[..., 2] *= sign
        planes = _barycentric_planes(tris)
        points = start[query]
        points[:, 2] *= sign

        # A face leaning against the pull is itself an occluder; when its own
        # triangle blocks the lifted ray the grid query can be skipped.
        own = np.full(len(ok), -1)
        own[occluder] = np.arange(len(tris))
        own = own[query]
        leaning = own >= 0
        blocked = np.zeros(len(query), dtype=bool)
        blocked[leaning] = _inside(planes[own[leaning]], points[leaning, :2])
        rest = ~blocked
        blocked[rest] = _blocked(tris, planes, points[rest])
        ok[query] = ~blocked
    return up_ok, down_ok


def _ray_frame(direction: np.ndarray) -> np.ndarray:
    """Rotation whose rows are two unit vectors perpendicular to ``direction``, then ``direction``."""
    helper = np.eye(3)[np.argmin(np.abs(direction))]
    e1 = np.cross(direction, helper)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(direction, e1), direction])


def _blocked(tris: np.ndarray, planes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """For rays from ``points`` along local +Z, whether any triangle in ``tris`` is hit.

    Both arrays are in a frame whose Z axis is the ray direction; ``tris`` is
    (T, 3, 3), ``planes`` its :func:`_barycentric_planes` and ``points`` is (Q, 3).
    """
    return np.isfinite(_first_hit(tris, planes, points))


def ray_hit_distances(
    mesh: trimesh.Trimesh, origins: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Distance from each origin along ``direction`` to the first face it enters (inf if none).

    The origins must lie outside the closed ``mesh``, so the first face a ray
    meets faces against it; only those faces are tested.
    """
    direction = _unit(direction)
    frame = _ray_frame(direction)
    facing = mesh.face_normals @ direction < -FACING_EPS
    tris = mesh.triangles[facing] @ frame.T
    return _first_hit(tris, _barycentric_planes(tris), np.asarray(origins) @ frame.T)


def _first_hit(tris: np.ndarray, planes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """For rays from ``points`` along local +Z, the distance to the nearest triangle hit.

    Same frame and arguments as :func:`_blocked`; inf where nothing is hit.
    """
    hit = np.full(len(points), np.inf)
    for q, height in _column_heights(tris, planes, points, above_only=True):
        gap = height - points[q, 2]
        ahead = gap > 0
        np.minimum.at(hit, q[ahead], gap[ahead])
    return hit


def column_spans(mesh: trimesh.Trimesh, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest point of ``mesh`` on the vertical line through each ``xy`` point.

    Both are inf / -inf where the line misses the mesh.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    low = np.full(len(xy), np.inf)
    high = np.full(len(xy), -np.inf)
    tris = mesh.triangles[np.abs(mesh.face_normals[:, 2]) > FACING_EPS]
    points = np.column_stack([xy, np.zeros(len(xy))])
    for q, height in _column_heights(tris, _barycentric_planes(tris), points, above_only=False):
        np.minimum.at(low, q, height)
        np.maximum.at(high, q, height)
    return low, high


def points_inside(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Whether each point lies inside the closed ``mesh``: a ray up from it crosses it oddly often."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    crossings = np.zeros(len(points), dtype=np.int64)
    tris = mesh.triangles[np.abs(mesh.face_normals[:, 2]) > FACING_EPS]
    for q, height in _column_heights(tris, _barycentric_planes(tris), points, above_only=True):
        np.add.at(crossings, q[height > points[q, 2]], 1)
    return crossings % 2 == 1


def _column_heights(
    tris: np.ndarray, planes: np.ndarray, points: np.ndarray, *, above_only: bool
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield, in batches, each point that lies over a triangle and the triangle's height there.

    ``tris``, ``planes`` and ``points`` are as for :func:`_blocked`. Triangles are
    binned into a uniform grid over their XY footprints, so each point is only
    tested against the triangles of its own cell. With ``above_only``, cells
    whose triangles all lie below a point are skipped for that point.
    """
    if len(tris) == 0 or len(points) == 0:
        return

    uv = tris[:, :, :2]
    lo = uv.min(axis=1)
    hi = uv.max(axis=1)
    origin = lo.min(axis=0)
    span = np.maximum(hi.max(axis=0) - origin, np.finfo(float).tiny)
    # The second bound keeps a very elongated footprint to at most T cells along its length.
    cell = max(GRID_CELL_SCALE * np.sqrt(span[0] * span[1] / len(tris)), span.max() / len(tris))
    shape = np.maximum(np.ceil(span / cell).astype(np.int64), 1)

    first = np.minimum(((lo - origin) / cell).astype(np.int64), shape - 1)
    last = np.minimum(((hi - origin) / cell).astype(np.int64), shape - 1)
    rows = last[:, 1] - first[:, 1] + 1
    tri_ids, slot = _expand((last[:, 0] - first[:, 0] + 1) * rows)
    cells = (first[tri_ids, 0] + slot // rows[tri_ids]) * shape[1] + (
        first[tri_ids, 1] + slot % rows[tri_ids]
    )
    n_cells = int(np.prod(shape))
    cell_tris = tri_ids[np.argsort(cells)]
    cell_count = np.bincount(cells, minlength=n_cells)
    cell_start = np.cumsum(cell_count) - cell_count

    rel = points[:, :2] - origin
    queries = np.flatnonzero(np.all((rel >= 0.0) & (rel <= span), axis=1))
    q_cell_2d = np.minimum((rel[queries] / cell).astype(np.int64), shape - 1)
    q_cell = q_cell_2d[:, 0] * shape[1] + q_cell_2d[:, 1]
    if above_only:
        cell_top = np.full(n_cells, -np.inf)
        np.maximum.at(cell_top, cells, tris[:, :, 2].max(axis=1)[tri_ids])
        reachable = cell_top[q_cell] > points[queries, 2]
        queries = queries[reachable]
        q_cell = q_cell[reachable]
    if len(queries) == 0:
        return

    pair_count = cell_count[q_cell]
    splits = np.searchsorted(
        np.cumsum(pair_count), np.arange(PAIR_BATCH, pair_count.sum(), PAIR_BATCH)
    )
    for chunk in np.split(np.arange(len(queries)), splits):
        if len(chunk) == 0:
            continue
        owner, slot = _expand(pair_count[chunk])
        q = queries[chunk][owner]
        pair_planes = planes[cell_tris[cell_start[q_cell[chunk]][owner] + slot]]
        inside = _inside(pair_planes, points[q, :2])
        q = q[inside]
        h = pair_planes[inside, 2]
        yield q, h[:, 0] * points[q, 0] + h[:, 1] * points[q, 1] + h[:, 2]


def _inside(planes: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Whether each point of ``uv`` lies in the projected triangle of the matching ``planes`` entry."""
    l1 = planes[:, 0, 0] * uv[:, 0] + planes[:, 0, 1] * uv[:, 1] + planes[:, 0, 2]
    l2 = planes[:, 1, 0] * uv[:, 0] + planes[:, 1, 1] * uv[:, 1] + planes[:, 1, 2]
    return (
        (l1 >= -BARYCENTRIC_TOLERANCE)
        & (l2 >= -BARYCENTRIC_TOLERANCE)
        & (l1 + l2 <= 1.0 + BARYCENTRIC_TOLERANCE)
    )


def _expand(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give item ``i`` ``counts[i]`` slots; return each slot's item and its index within the item."""
    owner = np.repeat(np.arange(len(counts)), counts)
    slot = np.arange(len(owner)) - np.repeat(np.cumsum(counts) - counts, counts)
    return owner, slot


def _barycentric_planes(tris: np.ndarray) -> np.ndarray:
    """(T, 3, 3) affine maps from local (u, v, 1) to (lambda1, lambda2, height on the triangle)."""
    p0 = tris[:, 0, :2]
    e1 = tris[:, 1, :2] - p0
    e2 = tris[:, 2, :2] - p0
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    l1 = np.stack([e2[:, 1], -e2[:, 0], e2[:, 0] * p0[:, 1] - e2[:, 1] * p0[:, 0]], axis=1)
    l2 = np.stack([-e1[:, 1], e1[:, 0], e1[:, 1] * p0[:, 0] - e1[:, 0] * p0[:, 1]], axis=1)
    l1 /= det[:, None]
    l2 /= det[:, None]
    w = tris[:, :, 2]
    height = (w[:, 1] - w[:, 0])[:, None] * l1 + (w[:, 2] - w[:, 0])[:, None] * l2
    height[:, 2] += w[:, 0]
    return np.stack([l1, l2, height], axis=1)


def _optimal_offset(
    mesh: trimesh.Trimesh,
    direction: np.ndarray,
    up_ok: np.ndarray,
    down_ok: np.ndarray,
    release_tolerance: float,
) -> float:
    """Exact minimiser of the undercut area over plane positions (sorting and prefix sums).

    Faces releasable only upwards must lie above the plane and faces
    releasable only downwards below it; the cost is the area of those that
    do not. It changes only where the plane passes such a face's lowest or
    highest vertex (shifted by the tolerance), so it is evaluated at those
    positions and halfway between them.
    """
    lo, hi = _face_span(mesh, direction)
    bottom, top = lo.min(), hi.max()
    if top - bottom <= PLANE_TOLERANCE * mesh.scale:
        return float((bottom + top) / 2.0)

    tolerance = _tolerance(mesh, release_tolerance)
    area = mesh.area_faces
    up_only = up_ok & ~down_ok
    down_only = down_ok & ~up_ok
    # An up-only face costs once the plane rises past lo + tolerance; a down-only face costs
    # until the plane reaches hi - tolerance.
    rise = np.sort(lo[up_only] + tolerance)
    fall = np.sort(hi[down_only] - tolerance)
    up_cum = np.concatenate([[0.0], np.cumsum(area[up_only][np.argsort(lo[up_only])])])
    down_cum = np.concatenate([[0.0], np.cumsum(area[down_only][np.argsort(hi[down_only])])])

    breaks = np.unique(np.clip(np.concatenate([rise, fall, [bottom, top]]), bottom, top))
    positions = np.empty(2 * len(breaks) - 1)
    positions[0::2] = breaks
    positions[1::2] = (breaks[:-1] + breaks[1:]) / 2.0
    cost = up_cum[np.searchsorted(rise, positions, side="left")] + (
        down_cum[-1] - down_cum[np.searchsorted(fall, positions, side="right")]
    )
    optimal = cost <= cost.min() + COST_ROUNDOFF * area.sum()

    edges = np.diff(np.concatenate([[False], optimal, [False]]).astype(np.int8))
    run_start = np.flatnonzero(edges == 1)
    run_end = np.flatnonzero(edges == -1) - 1
    widest = int(np.argmax(positions[run_end] - positions[run_start]))
    return float((positions[run_start[widest]] + positions[run_end[widest]]) / 2.0)


def _evaluate(
    mesh: trimesh.Trimesh,
    direction: np.ndarray,
    draft_threshold_deg: float,
    release_tolerance: float,
    offset: float | None = None,
) -> _Evaluation:
    up_ok, down_ok = _release(mesh, direction, True, release_tolerance)
    if offset is None:
        offset = _optimal_offset(mesh, direction, up_ok, down_ok, release_tolerance)
    above, below = _placement(mesh, direction, offset, release_tolerance)
    face_class = _classify(
        mesh.face_normals @ direction, above, below, up_ok, down_ok, draft_threshold_deg
    )
    area = mesh.area_faces
    total = area.sum()
    return _Evaluation(
        direction=direction,
        offset=float(offset),
        face_class=face_class,
        undercut_fraction=float(area[face_class == FACE_UNDERCUT].sum() / total),
        low_draft_fraction=float(area[face_class == FACE_LOW_DRAFT].sum() / total),
    )


def _search_directions(
    mesh: trimesh.Trimesh, draft_threshold_deg: float, release_tolerance: float
) -> list[_Evaluation]:
    """Score candidate directions and return the re-scored ones, best first.

    A normal-only scan over many directions picks a few distinct promising
    ones; those and the three axes are then evaluated with occlusion on the
    full mesh.
    """
    sample = _coarse_sample(mesh)
    axes = np.eye(3)
    pool = np.vstack(
        [
            axes,
            _inertia_axes(mesh),
            _dominant_normals(mesh),
            _hemisphere_directions(FIBONACCI_DIRECTIONS),
        ]
    )
    pool = np.array([_canonical(d) for d in pool])
    coarse = _coarse_undercut(sample, pool)
    logger.debug("Direction scan: best normal-only undercut %.2f %%", 100.0 * coarse.min())

    picked = _distinct(pool[np.argsort(coarse, kind="stable")], CANDIDATE_SEPARATION_DEG)
    others = np.array([d for d in picked[:REFINE_COUNT] if not _is_axis(d)]).reshape(-1, 3)
    directions = _distinct(
        np.vstack([axes, _local_search(sample, others)]), DUPLICATE_DIRECTION_DEG
    )
    evaluations = [_evaluate(mesh, d, draft_threshold_deg, release_tolerance) for d in directions]
    return _rank(evaluations)


def _distinct(directions: np.ndarray, min_angle_deg: float) -> list[np.ndarray]:
    """Greedily keep directions (as lines) at least ``min_angle_deg`` from every kept one."""
    min_cos = np.cos(np.radians(min_angle_deg))
    kept: list[np.ndarray] = []
    for d in directions:
        if all(abs(d @ k) < min_cos for k in kept):
            kept.append(d)
    return kept


def _rank(evaluations: list[_Evaluation]) -> list[_Evaluation]:
    """Order evaluations by preference: least undercut, axes among ties, then less low draft."""
    best = min(e.undercut_fraction for e in evaluations)
    ties = [e for e in evaluations if e.undercut_fraction <= best + EQUAL_UNDERCUT_TOLERANCE]
    ties = [e for e in ties if _is_axis(e.direction)] or ties
    chosen = min(ties, key=lambda e: (e.low_draft_fraction, e.undercut_fraction))
    rest = sorted(
        (e for e in evaluations if e is not chosen),
        key=lambda e: (e.undercut_fraction, e.low_draft_fraction),
    )
    return [chosen, *rest]


def _coarse_sample(mesh: trimesh.Trimesh) -> _FaceSample:
    """All faces, or an area-weighted sample of ``COARSE_MAX_FACES`` for large meshes."""
    area = mesh.area_faces
    if len(area) <= COARSE_MAX_FACES:
        return _FaceSample(mesh.face_normals, mesh.triangles, area / area.sum())
    rng = np.random.default_rng(COARSE_SEED)
    idx = rng.choice(len(area), size=COARSE_MAX_FACES, p=area / area.sum())
    return _FaceSample(
        mesh.face_normals[idx],
        mesh.triangles[idx],
        np.full(COARSE_MAX_FACES, 1.0 / COARSE_MAX_FACES),
    )


def _coarse_undercut(sample: _FaceSample, directions: np.ndarray) -> np.ndarray:
    """Normal-only undercut fraction per direction, at the best of ``COARSE_BINS - 1`` plane positions.

    The planes are evenly spaced inside the part's extent. A face is allowed
    to cross the plane by less than one spacing, which stands in for the
    release tolerance and lets faces that meet at a vertex ring go to
    different halves.
    """
    result = np.empty(len(directions))
    for start in range(0, len(directions), COARSE_DIRECTION_BATCH):
        batch = directions[start : start + COARSE_DIRECTION_BATCH]
        count = len(batch)
        heights = sample.triangles @ batch.T
        face_lo = heights.min(axis=1)
        face_hi = heights.max(axis=1)
        normal_dot = sample.normals @ batch.T
        low = face_lo.min(axis=0)
        step = np.maximum(face_hi.max(axis=0) - low, np.finfo(float).tiny) / COARSE_BINS
        w = np.broadcast_to(sample.weights[:, None], normal_dot.shape)
        # Plane k sits at low + k * step. An up-only face must lie above it, so it costs for
        # k > ceil(lo / step); a down-only face must lie below it and costs for k < floor(hi / step).
        up_level = np.clip(np.ceil((face_lo - low) / step), 0, COARSE_BINS).astype(np.int64)
        down_level = np.clip(np.floor((face_hi - low) / step), 0, COARSE_BINS).astype(np.int64)
        offsets = np.arange(count) * (COARSE_BINS + 1)
        size = count * (COARSE_BINS + 1)
        up = np.bincount(
            (up_level + offsets).ravel(), (w * (normal_dot > NORMAL_EPS)).ravel(), size
        )
        down = np.bincount(
            (down_level + offsets).ravel(), (w * (normal_dot < -NORMAL_EPS)).ravel(), size
        )
        up = np.cumsum(up.reshape(count, -1), axis=1)
        down = np.cumsum(down.reshape(count, -1), axis=1)
        # cost[:, k - 1] for the planes k = 1 .. COARSE_BINS - 1
        cost = up[:, :-2] + (down[:, -1:] - down[:, 1:-1])
        result[start : start + count] = cost.min(axis=1)
    return result


def _local_search(sample: _FaceSample, directions: np.ndarray) -> np.ndarray:
    """Improve each direction by a pattern search on rings of shrinking radius (normal-only score)."""
    best = directions.copy()
    if len(best) == 0:
        return best
    best_cost = _coarse_undercut(sample, best)
    phases = np.linspace(0.0, 2.0 * np.pi, LOCAL_SEARCH_RING, endpoint=False)
    ring_offsets = np.column_stack([np.cos(phases), np.sin(phases)])
    angle = np.radians(LOCAL_SEARCH_START_DEG)
    rows = np.arange(len(best))
    for _ in range(LOCAL_SEARCH_ROUNDS):
        frames = np.array([_ray_frame(d) for d in best])
        rings = np.cos(angle) * frames[:, None, 2] + np.sin(angle) * np.einsum(
            "rk,mkj->mrj", ring_offsets, frames[:, :2]
        )
        costs = _coarse_undercut(sample, rings.reshape(-1, 3)).reshape(len(best), -1)
        pick = costs.argmin(axis=1)
        better = costs[rows, pick] < best_cost
        best[better] = rings[rows, pick][better]
        best_cost[better] = costs[rows, pick][better]
        angle /= 2.0
    return np.array([_canonical(d) for d in best])


def _hemisphere_directions(count: int) -> np.ndarray:
    """Fibonacci lattice on the upper hemisphere (each line through the origin once)."""
    i = np.arange(count)
    z = (i + 0.5) / count
    r = np.sqrt(1.0 - z**2)
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _inertia_axes(mesh: trimesh.Trimesh) -> np.ndarray:
    if not mesh.is_volume:
        return np.empty((0, 3))
    return np.asarray(mesh.principal_inertia_vectors, dtype=float)


def _dominant_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Normals of the flat regions with the most area, which a CAD part is usually best pulled along."""
    normals, group = np.unique(
        np.round(mesh.face_normals, NORMAL_GROUP_DECIMALS), axis=0, return_inverse=True
    )
    area = np.bincount(group.ravel(), weights=mesh.area_faces, minlength=len(normals))
    top = np.argsort(-area, kind="stable")[:DOMINANT_NORMALS]
    return normals[top[np.linalg.norm(normals[top], axis=1) > 0.5]]


def _canonical(direction: np.ndarray) -> np.ndarray:
    """Pick the sign of a demolding line so its largest component is positive."""
    d = _unit(direction)
    return -d if d[np.argmax(np.abs(d))] < 0 else d


def _is_axis(direction: np.ndarray) -> bool:
    return bool(np.max(np.abs(direction)) >= AXIS_ALIGNED_COS)


def _rotation_to_z(direction: np.ndarray) -> np.ndarray:
    """Smallest rotation taking ``direction`` to +Z (a half turn about X for -Z)."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(direction, z)
    sin_a = np.linalg.norm(axis)
    cos_a = float(direction @ z)
    if sin_a < 1e-12:  # +-Z up to round-off; the axis below would be undefined
        return np.eye(3) if cos_a > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1.0 - cos_a) / sin_a**2)


def _hull_points(xy: np.ndarray) -> np.ndarray:
    try:
        return xy[ConvexHull(xy).vertices]
    except QhullError:
        # Collinear or degenerate projection: every point is on the hull.
        return xy


def _min_area_rectangle_angle(hull: np.ndarray) -> float:
    """Rotation angle (radians, within +-45 degrees) aligning the minimum-area rectangle with X/Y.

    Returns 0 when the rotation would shrink the axis-aligned footprint only marginally.
    """
    if len(hull) < 3:
        return 0.0
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.mod(np.arctan2(edges[:, 1], edges[:, 0]) + np.pi / 4, np.pi / 2) - np.pi / 4
    angles = np.concatenate([[0.0], angles])
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    x = hull[:, 0][None] * cos_a[:, None] + hull[:, 1][None] * sin_a[:, None]
    y = -hull[:, 0][None] * sin_a[:, None] + hull[:, 1][None] * cos_a[:, None]
    areas = np.ptp(x, axis=1) * np.ptp(y, axis=1)
    best = int(np.argmin(areas))
    if areas[best] >= (1.0 - FOOTPRINT_MIN_GAIN) * areas[0]:
        return 0.0
    return float(angles[best])
