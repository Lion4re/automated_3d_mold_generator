"""Pour sprue, pour funnel and air vents.

All geometry is in the mold frame (see :mod:`moldgen.parting`): the parting
plane is ``z == 0``. To pour, the assembled mold is stood up so that ``up``
(one of +X, -X, +Y, -Y) points against gravity. The sprue runs inside the
parting plane, so each half carries half of the channel and both halves
print without supports and can be cleaned after casting.

Pour direction
--------------
Every candidate ``up`` is scored and the highest score wins::

    score = 2 * entry_fit + bulk - sum over traps of (0.05 + prominence / length)

``entry_fit`` (0 to 1) measures how well the part's cross-section in the
parting plane accepts the sprue: the chord one sprue radius below the top
should be at least 1.5 sprue diameters wide. ``bulk`` (0 to 1) is the height of
the centre of mass along ``up`` relative to the part's ``length`` along ``up``;
feeding the bulkiest end lets the sprue top up the section that shrinks most
while curing and leaves the sprue stub where it is easiest to trim (a chess
piece is poured base up). Each air trap that needs a vent costs a fixed
amount plus its prominence relative to the part length.

Air traps
---------
Air rises to local maxima (along ``up``) of the part surface. They are found
with topographic persistence on the vertex graph: sweeping a level downwards,
a peak's prominence is its height minus the level at which its region merges
with a region holding a higher peak. Only peaks with an upward surface normal
and a prominence above ``max(0.5 mm, vent radius)`` count. Vertices around the
sprue entry drain through the sprue and act as the highest peak.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import manifold3d
import numpy as np
import shapely
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Polygon

from moldgen.booleans import to_manifold

logger = logging.getLogger(__name__)

POUR_DIRECTIONS = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
"""Candidate pour ``up`` directions, in tie-break order."""
POUR_DIRECTIONS.setflags(write=False)

FACE_OVERSHOOT = 1.0
"""How far channels reach past the block's outer face (mm), so cuts are clean."""
MIN_CAVITY_OVERLAP = 1.0
"""Depth (mm) the sprue reaches into the cavity, unless capped by ``MAX_OVERLAP_FRACTION``."""
MAX_OVERLAP_FRACTION = 0.5
"""Largest fraction of the cavity's depth under the entry that the sprue reaches into, so it
cannot punch through a section thinner than twice ``MIN_CAVITY_OVERLAP``."""
MIN_SPRUE_RADIUS = 1.0
"""Smallest sprue radius (mm) when a narrow entry or a thin block forces a thinner sprue."""
SPRUE_RADIUS_STEPS = 16
"""Number of radii tried from the requested sprue radius down to ``MIN_SPRUE_RADIUS``."""
ENTRY_COMFORT = 1.5
"""Entry chord width, in sprue diameters, that earns a full ``entry_fit``."""
SPRUE_DRAIN_REACH = 1.0
"""Peaks within the sprue radius plus this distance (mm) of the sprue drain through it."""

MIN_PROMINENCE = 0.5
"""Smallest prominence (mm) of a peak that gets a vent."""
PLATEAU_TOLERANCE = 0.05
"""Height band (mm) below a peak treated as the same flat top when centring its vent."""
MIN_VENT_OVERLAP = 0.5
"""Minimum depth (mm) a vent reaches into the cavity."""
VENT_SNAP_RADII = 0.75
"""Vents of peaks within this many vent radii of ``z == 0`` move into the parting plane."""
VENT_MERGE_DIAMETERS = 3.0
"""Peaks closer than this many vent diameters to a kept vent share it."""
MAX_VENTS = 8
"""Most vents placed; less prominent traps beyond this are counted in ``GatingPlan.unvented``."""

FUNNEL_MOUTH_RATIO = 2.25
"""Preferred funnel mouth radius as a multiple of the sprue radius."""
FUNNEL_HALF_ANGLE_DEG = 30.0
"""Angle between the funnel wall and its axis."""
FUNNEL_MIN_WALL = 1.0
"""Material (mm) kept between the funnel and the cavity or the block edges, and between
the sprue and the block's faces above and below the parting plane."""
FUNNEL_MIN_LIP = 0.5
"""A funnel whose mouth is not at least this much (mm) wider than the sprue is skipped."""
FUNNEL_THROAT_EXTENSION = 0.5
"""How far (mm) the funnel solid continues into the sprue past its throat."""

ENTRY_WEIGHT = 2.0
"""Weight of ``entry_fit`` in the pour score."""
BULK_WEIGHT = 1.0
"""Weight of ``bulk`` in the pour score."""
TRAP_WEIGHT = 1.0
"""Weight of the air trap cost in the pour score."""
TRAP_BASE_PENALTY = 0.05
"""Fixed cost of each air trap, on top of its relative prominence."""

CHANNEL_SECTIONS = 32
"""Facets around sprue and vent cylinders."""
FUNNEL_SECTIONS = 48
"""Facets around the funnel cone."""
PROBE_SECTIONS = 64
"""Facets around the cylinder that measures the cavity clearance under the funnel."""
PROBE_EXTENSION = 1.0
"""How far (mm) probe lines reach past the cross-section so they cross it completely."""
SLICE_INSET = 1e-6
"""How far (mm) inside a part touching ``z == 0`` its parting section is sliced."""
SURFACE_TOLERANCE = 1e-3
"""Distance (mm) within which a point counts as lying on a face."""
CHORD_TIE_TOLERANCE = 1e-6
"""Chords whose widths differ by less than this (mm) count as equally wide."""

_Z = np.array([0.0, 0.0, 1.0])


@dataclass
class Channel:
    """A straight cylindrical channel from ``start`` to ``end``."""

    start: np.ndarray
    end: np.ndarray
    radius: float

    def solid(self, sections: int = CHANNEL_SECTIONS) -> trimesh.Trimesh:
        start = np.asarray(self.start, dtype=float)
        axis = np.asarray(self.end, dtype=float) - start
        length = float(np.linalg.norm(axis))
        profile = [(0.0, 0.0), (self.radius, 0.0), (self.radius, length), (0.0, length)]
        return solid_of_revolution(profile, start, axis / length, sections)


@dataclass
class Funnel:
    """A cone cut into the outer face of the mold around the sprue inlet."""

    center: np.ndarray
    """Centre of the funnel mouth, on the outer face of the mold block."""
    axis: np.ndarray
    """Unit vector pointing out of the mold (equal to the pour ``up`` direction)."""
    mouth_radius: float
    throat_radius: float
    depth: float

    def solid(self, sections: int = FUNNEL_SECTIONS) -> trimesh.Trimesh:
        """The cone, extended past the face and into the sprue so boolean cuts are clean."""
        slope = (self.mouth_radius - self.throat_radius) / self.depth
        throat_extension = FUNNEL_THROAT_EXTENSION
        if slope > 0:
            throat_extension = min(throat_extension, 0.5 * self.throat_radius / slope)
        bottom = -self.depth - throat_extension
        profile = [
            (0.0, bottom),
            (self.throat_radius - throat_extension * slope, bottom),
            (self.mouth_radius + FACE_OVERSHOOT * slope, FACE_OVERSHOOT),
            (0.0, FACE_OVERSHOOT),
        ]
        return solid_of_revolution(profile, self.center, self.axis, sections)


@dataclass
class PourScore:
    """Evaluation of one pour direction (see the module docstring)."""

    up: np.ndarray
    score: float
    entry_fit: float
    bulk: float
    traps: int
    """Air traps that need a vent."""
    trap_height: float
    """Summed prominence of those traps (mm)."""


@dataclass
class GatingPlan:
    up: np.ndarray
    """Unit vector that must point up while pouring (mold frame)."""
    sprue: Channel
    funnel: Funnel | None = None
    vents: list[Channel] = field(default_factory=list)
    scores: list[PourScore] = field(default_factory=list)
    """Evaluated pour directions, best first."""
    unvented: int = 0
    """Air traps left without a vent because of ``MAX_VENTS``."""

    def solids(self) -> list[trimesh.Trimesh]:
        """Watertight solids to subtract from the mold block."""
        solids = [self.sprue.solid()]
        if self.funnel is not None:
            solids.append(self.funnel.solid())
        solids.extend(vent.solid() for vent in self.vents)
        return solids


@dataclass
class _PourOption:
    score: PourScore
    sprue: Channel
    entry: np.ndarray
    traps: list[tuple[int, float]]


def plan_gating(
    part: trimesh.Trimesh,
    block_bounds: np.ndarray,
    *,
    sprue_diameter: float,
    vent_diameter: float,
    funnel: bool = True,
    vents: bool = True,
    up: np.ndarray | None = None,
) -> GatingPlan:
    """Choose the pour orientation and lay out sprue, funnel and vents.

    ``part`` is the part in the mold frame and ``block_bounds`` the (2, 3)
    bounds of the solid mold block around it. ``up`` forces the pour
    direction (one of +X, -X, +Y, -Y) instead of scoring all four.
    """
    block_bounds = np.asarray(block_bounds, dtype=float)
    if block_bounds.shape != (2, 3):
        raise ValueError(f"block_bounds must have shape (2, 3), got {block_bounds.shape}")
    if not (sprue_diameter > 0 and vent_diameter > 0):
        raise ValueError("Sprue and vent diameters must be positive")
    if part.body_count > 1:
        # The pipeline reports the cavities this leaves unfilled.
        logger.info(
            "Part has %d separate bodies; only bodies the sprue reaches will fill", part.body_count
        )

    solid = to_manifold(part, "the part")
    section = _parting_section(solid)
    sprue_radius = _fit_sprue_radius(0.5 * sprue_diameter, block_bounds)
    vent_radius = 0.5 * vent_diameter
    threshold = max(MIN_PROMINENCE, vent_radius)
    directions = POUR_DIRECTIONS if up is None else [_pour_direction(up)]
    options = [
        _evaluate_pour(part, section, block_bounds, direction, sprue_radius, threshold)
        for direction in directions
    ]
    options.sort(key=lambda option: -option.score.score)
    best = options[0]
    for option in options:
        logger.debug("Pour direction %s: %s", option.score.up, option.score)

    direction = best.score.up
    top_face = float(np.max(block_bounds @ direction))
    vent_channels, unvented = (
        _plan_vents(part, direction, best, top_face, vent_radius) if vents else ([], 0)
    )
    pour_funnel = (
        _plan_funnel(solid, block_bounds, direction, best.sprue, top_face, vent_channels)
        if funnel
        else None
    )
    return GatingPlan(
        up=direction,
        sprue=best.sprue,
        funnel=pour_funnel,
        vents=vent_channels,
        scores=[option.score for option in options],
        unvented=unvented,
    )


def solid_of_revolution(
    profile: list[tuple[float, float]] | np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    sections: int,
) -> trimesh.Trimesh:
    """Revolve a ``(radius, height)`` profile about ``axis`` through ``origin``.

    The profile must start and end on the axis and run upwards so normals face
    outwards. For axes lying in the parting plane, two ring vertices lie
    exactly in ``z == 0`` so both mold halves get identical half-channels.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    reference = _Z if abs(axis @ _Z) < 0.9 else np.array([1.0, 0.0, 0.0])
    radial = np.cross(reference, axis)
    radial /= np.linalg.norm(radial)
    frame = np.eye(4)
    frame[:3, 0] = radial
    frame[:3, 1] = np.cross(axis, radial)
    frame[:3, 2] = axis
    frame[:3, 3] = origin
    return trimesh.creation.revolve(
        np.asarray(profile, dtype=float), sections=sections, transform=frame
    )


def cross_section_to_shapely(section: manifold3d.CrossSection) -> MultiPolygon:
    """Convert a manifold3d cross-section to shapely polygons with holes."""
    polygons = []
    for component in section.decompose():
        contours = [Polygon(contour) for contour in component.to_polygons() if len(contour) >= 3]
        if not contours:
            continue
        # The outer contour of a component encloses all of its holes.
        outer = max(contours, key=lambda contour: contour.area)
        holes = [contour.exterior.coords for contour in contours if contour is not outer]
        polygons.append(Polygon(outer.exterior.coords, holes))
    return MultiPolygon(polygons)


def _pour_direction(up: np.ndarray) -> np.ndarray:
    up = np.asarray(up, dtype=float).reshape(3)
    matches = np.flatnonzero(np.all(np.isclose(POUR_DIRECTIONS, up / np.linalg.norm(up)), axis=1))
    if len(matches) != 1:
        raise ValueError(f"Pour direction must be one of +X, -X, +Y, -Y, got {up}")
    return POUR_DIRECTIONS[matches[0]]


def _fit_sprue_radius(radius: float, block_bounds: np.ndarray) -> float:
    """Cap the sprue radius so ``FUNNEL_MIN_WALL`` remains above and below the channel."""
    room = min(-block_bounds[0, 2], block_bounds[1, 2]) - FUNNEL_MIN_WALL
    if room < min(radius, MIN_SPRUE_RADIUS):
        raise ValueError(
            f"The mold block is too thin around the parting plane for a sprue: it needs "
            f"{min(radius, MIN_SPRUE_RADIUS) + FUNNEL_MIN_WALL:.1f} mm above and below it. "
            "Increase the wall thickness."
        )
    if room < radius:
        logger.info("Sprue narrowed to %.1f mm to fit the mold block", 2.0 * room)
        return float(room)
    return radius


def _parting_section(solid: manifold3d.Manifold) -> MultiPolygon:
    """The part's cross-section in the parting plane."""
    z_min, z_max = solid.bounding_box()[2::3]
    if not z_min <= 0.0 <= z_max:
        raise ValueError("Part does not reach the parting plane z == 0")
    # A part resting on the plane is sliced just inside so its flat face counts.
    slice_height = float(np.clip(0.0, z_min + SLICE_INSET, z_max - SLICE_INSET))
    section = cross_section_to_shapely(solid.slice(slice_height))
    if section.is_empty:
        raise ValueError("Part has no cross-section in the parting plane z == 0")
    return section


def _side(up: np.ndarray) -> np.ndarray:
    """In-plane unit vector perpendicular to ``up``."""
    return np.cross(_Z, up)


def _chords(region: MultiPolygon, line: LineString) -> list[LineString]:
    parts = shapely.get_parts(region.intersection(line))
    return [part for part in parts if part.geom_type == "LineString" and part.length > 0]


def _column(region: MultiPolygon, x: float) -> list[LineString]:
    """Vertical chords of ``region`` at ``x`` (local frame)."""
    _, bottom, _, top = region.bounds
    return _chords(region, LineString([(x, top + PROBE_EXTENSION), (x, bottom - PROBE_EXTENSION)]))


def _column_top(region: MultiPolygon, x: float) -> float:
    return max((chord.bounds[3] for chord in _column(region, x)), default=-np.inf)


def _widest_chord(region: MultiPolygon, height: float) -> tuple[float, float]:
    """Width and centre of the longest horizontal chord at ``height`` (local frame).

    Of equally wide chords, the one whose column through its centre reaches highest wins.
    """
    x_min, _, x_max, _ = region.bounds
    probe = LineString([(x_min - PROBE_EXTENSION, height), (x_max + PROBE_EXTENSION, height)])
    chords = _chords(region, probe)
    if not chords:
        return 0.0, 0.5 * (x_min + x_max)
    longest = max(chord.length for chord in chords)
    ties = [chord for chord in chords if chord.length >= longest - CHORD_TIE_TOLERANCE]
    widest = max(ties, key=lambda chord: _column_top(region, chord.centroid.x))
    return widest.length, widest.centroid.x


def _evaluate_pour(
    part: trimesh.Trimesh,
    section: MultiPolygon,
    block_bounds: np.ndarray,
    up: np.ndarray,
    sprue_radius: float,
    threshold: float,
) -> _PourOption:
    """Place the sprue for pour direction ``up`` and score that direction.

    The sprue enters the top of the widest column of the parting section and
    reaches ``max(MIN_CAVITY_OVERLAP, radius)`` into it, but never more than
    ``MAX_OVERLAP_FRACTION`` of the column's length, so on a thin section the
    overlap is shallower than ``MIN_CAVITY_OVERLAP`` rather than cutting through.
    """
    side = _side(up)
    # Local 2D frame: x along ``side``, y along ``up``.
    local = affinity.affine_transform(section, [side[0], side[1], up[0], up[1], 0.0, 0.0])
    _, bottom, _, top = local.bounds
    # Probing deeper than half the section would measure its far side, not its top.
    max_probe = 0.5 * (top - bottom)

    entry_width, _ = _widest_chord(local, top - min(sprue_radius, max_probe))
    entry_fit = min(1.0, entry_width / (ENTRY_COMFORT * 2.0 * sprue_radius))

    floor = min(sprue_radius, MIN_SPRUE_RADIUS)
    for radius in np.linspace(sprue_radius, floor, SPRUE_RADIUS_STEPS):
        width, center = _widest_chord(local, top - min(radius, max_probe))
        if width >= 2.0 * radius:
            break
    radius = float(radius)

    column = _column(local, center)
    if not column:
        raise ValueError("Part cross-section in the parting plane is too thin for a sprue")
    entry_chord = max(column, key=lambda chord: chord.bounds[3])
    entry_height = entry_chord.bounds[3]
    overlap = min(max(MIN_CAVITY_OVERLAP, radius), MAX_OVERLAP_FRACTION * entry_chord.length)

    top_face = float(np.max(block_bounds @ up))
    entry = center * side + entry_height * up
    sprue = Channel(
        start=center * side + (top_face + FACE_OVERSHOOT) * up,
        end=entry - overlap * up,
        radius=radius,
    )

    vertices = part.vertices
    heights = vertices @ up
    offset = vertices - entry
    axis_distance = np.hypot(offset @ side, offset[:, 2])
    drains = (axis_distance <= radius + SPRUE_DRAIN_REACH) & (
        heights >= entry_height - overlap - SPRUE_DRAIN_REACH
    )
    # Large faces may have no vertex near the sprue; the face it enters through drains too.
    drains[part.faces[_faces_at(part, entry)]] = True
    traps = _air_traps(part, heights, drains, threshold)
    if traps:
        # A local maximum facing down is the roof of a dent in the underside, not an air trap.
        normals = part.vertex_normals
        traps = [(vertex, prominence) for vertex, prominence in traps if normals[vertex] @ up > 0]

    length = float(np.ptp(heights))
    bulk = float((part.center_mass @ up - heights.min()) / length)
    trap_height = sum(prominence for _, prominence in traps)
    trap_cost = len(traps) * TRAP_BASE_PENALTY + trap_height / length
    score = ENTRY_WEIGHT * entry_fit + BULK_WEIGHT * bulk - TRAP_WEIGHT * trap_cost
    return _PourOption(
        score=PourScore(
            up=up.copy(),
            score=float(score),
            entry_fit=float(entry_fit),
            bulk=bulk,
            traps=len(traps),
            trap_height=float(trap_height),
        ),
        sprue=sprue,
        entry=entry,
        traps=traps,
    )


def _faces_at(
    part: trimesh.Trimesh, point: np.ndarray, tolerance: float = SURFACE_TOLERANCE
) -> np.ndarray:
    """Indices of the faces within ``tolerance`` of a point on the surface."""
    face_z = part.vertices[:, 2][part.faces]
    near = np.flatnonzero(
        (face_z.min(axis=1) <= point[2] + tolerance) & (face_z.max(axis=1) >= point[2] - tolerance)
    )
    triangles = part.vertices[part.faces[near]]
    near = near[
        np.all(
            (triangles.min(axis=1) - tolerance <= point)
            & (point <= triangles.max(axis=1) + tolerance),
            axis=1,
        )
    ]
    if len(near) > 0:
        closest = trimesh.triangles.closest_point(
            part.vertices[part.faces[near]], np.tile(point, (len(near), 1))
        )
        near = near[np.linalg.norm(closest - point, axis=1) <= tolerance]
    if len(near) == 0:
        raise ValueError(f"Point {point} is not on the part surface")
    return near


def _air_traps(
    part: trimesh.Trimesh, heights: np.ndarray, drains: np.ndarray, threshold: float
) -> list[tuple[int, float]]:
    """Return ``(peak vertex, prominence)`` of every peak above ``threshold``, most prominent first.

    Each vertex first flows to its highest neighbour until it reaches a local
    maximum, which contracts the vertex graph to one node per basin without
    changing any prominence. Basins are then joined across their highest
    shared edge, highest first, with a union-find; the lower peak dies at each
    join. ``drains`` vertices form one basin above every peak. Equal heights
    are ordered by vertex index.
    """
    count = len(heights)
    level = heights.astype(float)
    # Any level above every peak makes the drain basin the eldest.
    level[drains] = heights.max() + 1.0
    by_rank = np.lexsort((np.arange(count), level))
    rank = np.empty(count, dtype=np.int64)
    rank[by_rank] = np.arange(count)

    edges = part.edges_unique
    highest = rank.copy()
    np.maximum.at(highest, edges[:, 0], rank[edges[:, 1]])
    np.maximum.at(highest, edges[:, 1], rank[edges[:, 0]])
    basin = by_rank[highest]
    while True:
        jumped = basin[basin]
        if np.array_equal(jumped, basin):
            break
        basin = jumped
    basin[np.isin(basin, basin[drains])] = basin[np.flatnonzero(drains)[0]]

    first, second = basin[edges[:, 0]], basin[edges[:, 1]]
    crossing = first != second
    low = np.minimum(first, second)[crossing]
    high = np.maximum(first, second)[crossing]
    saddle = np.minimum(level[edges[:, 0]], level[edges[:, 1]])[crossing]
    order = np.argsort(-saddle, kind="stable")
    _, highest_edge = np.unique(low[order] * count + high[order], return_index=True)
    order = order[np.sort(highest_edge)]

    parent: dict[int, int] = {}

    def find(node: int) -> int:
        root = node
        while parent.get(root, root) != root:
            root = parent[root]
        while node != root:
            parent[node], node = root, parent[node]
        return root

    traps = []
    for a, b, saddle_level in zip(
        low[order].tolist(), high[order].tolist(), saddle[order].tolist(), strict=True
    ):
        elder, younger = find(a), find(b)
        if elder == younger:
            continue
        if rank[elder] < rank[younger]:
            elder, younger = younger, elder
        prominence = float(level[younger]) - saddle_level
        if prominence >= threshold:
            traps.append((younger, prominence))
        parent[younger] = elder
    traps.sort(key=lambda trap: -trap[1])
    return traps


def _plateau_point(
    part: trimesh.Trimesh, heights: np.ndarray, vertex: int, up: np.ndarray
) -> np.ndarray:
    """Centre of the flat top around a peak vertex, kept on the part surface."""
    inside = np.flatnonzero(heights >= heights[vertex] - PLATEAU_TOLERANCE)
    local = np.full(len(heights), -1)
    local[inside] = np.arange(len(inside))
    edges = local[part.edges_unique]
    edges = edges[np.all(edges >= 0, axis=1)]
    graph = sparse.coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(inside), len(inside))
    )
    _, labels = connected_components(graph, directed=False)
    members = inside[labels == labels[local[vertex]]]

    faces = part.faces[np.all(np.isin(part.faces, members), axis=1)]
    if len(faces) == 0:
        return np.array(part.vertices[vertex], dtype=float)
    triangles = part.vertices[faces]
    face_centers = triangles.mean(axis=1)
    areas = trimesh.triangles.area(triangles)
    center = areas @ face_centers / areas.sum()
    # The centre of a non-convex plateau may fall outside it; check in projection along ``up``.
    flat = triangles - np.outer(triangles.reshape(-1, 3) @ up, up).reshape(triangles.shape)
    point = center - (center @ up) * up
    # Faces seen edge-on along ``up`` have no projected area and no barycentric frame.
    valid = trimesh.triangles.area(flat) > 1e-12
    if np.any(valid):
        barycentric = trimesh.triangles.points_to_barycentric(
            flat[valid], np.tile(point, (int(valid.sum()), 1))
        )
        # The small negative bound accepts points on a shared edge despite rounding.
        if np.any(np.all(barycentric >= -1e-9, axis=1)):
            return center
    return face_centers[np.argmin(np.linalg.norm(face_centers - center, axis=1))]


def _plan_vents(
    part: trimesh.Trimesh,
    up: np.ndarray,
    option: _PourOption,
    top_face: float,
    vent_radius: float,
) -> tuple[list[Channel], int]:
    """Vents for the air traps of ``option`` and the number of traps left without one."""
    heights = part.vertices @ up
    sprue_reach = option.sprue.radius + vent_radius + SPRUE_DRAIN_REACH
    overlap = max(MIN_VENT_OVERLAP, vent_radius)
    positions: list[np.ndarray] = []
    unvented = 0
    for vertex, _ in option.traps:
        position = _plateau_point(part, heights, vertex, up)
        if abs(position[2]) <= VENT_SNAP_RADII * vent_radius:
            position[2] = 0.0
        offset = position - option.entry
        if np.linalg.norm(offset - (offset @ up) * up) < sprue_reach:
            continue
        if any(
            np.linalg.norm(position - kept) < VENT_MERGE_DIAMETERS * 2.0 * vent_radius
            for kept in positions
        ):
            continue
        if len(positions) == MAX_VENTS:
            unvented += 1
            continue
        positions.append(position)
    if unvented:
        logger.info(
            "%d air traps have no vent; only the %d most prominent are vented", unvented, MAX_VENTS
        )
    channels = [
        Channel(
            start=position - overlap * up,
            end=position + (top_face + FACE_OVERSHOOT - position @ up) * up,
            radius=vent_radius,
        )
        for position in positions
    ]
    return channels, unvented


def _plan_funnel(
    solid: manifold3d.Manifold,
    block_bounds: np.ndarray,
    up: np.ndarray,
    sprue: Channel,
    top_face: float,
    vents: list[Channel],
) -> Funnel | None:
    side = _side(up)
    center = (sprue.start @ side) * side + top_face * up
    lateral = np.abs(side)
    edge_room = min(
        center @ lateral - block_bounds[0] @ lateral,
        block_bounds[1] @ lateral - center @ lateral,
        -block_bounds[0, 2],
        block_bounds[1, 2],
    )
    limits = [FUNNEL_MOUTH_RATIO * sprue.radius, edge_room - FUNNEL_MIN_WALL]
    for vent in vents:
        offset = vent.end - center
        limits.append(np.linalg.norm(offset - (offset @ up) * up) - vent.radius - FUNNEL_MIN_WALL)
    mouth = min(limits)
    if mouth - sprue.radius < FUNNEL_MIN_LIP:
        logger.debug("No room for a pour funnel")
        return None

    slope = np.tan(np.radians(FUNNEL_HALF_ANGLE_DEG))
    # Highest point of the cavity under the funnel, widened by the wall it must keep.
    reach = mouth + FUNNEL_MIN_WALL
    span = float(np.ptp(block_bounds @ up)) + 2.0 * FACE_OVERSHOOT
    column = solid_of_revolution(
        [(0.0, 0.0), (reach, 0.0), (reach, span), (0.0, span)],
        center + FACE_OVERSHOOT * up,
        -up,
        PROBE_SECTIONS,
    )
    under = solid ^ to_manifold(column)
    if under.is_empty():
        clearance = span
    else:
        corners = np.reshape(under.bounding_box(), (2, 3))
        clearance = top_face - float(np.max(corners @ up))
    depth = min(
        (mouth - sprue.radius) / slope,
        clearance - FUNNEL_MIN_WALL - FUNNEL_THROAT_EXTENSION,
    )
    mouth = sprue.radius + depth * slope
    if mouth - sprue.radius < FUNNEL_MIN_LIP:
        logger.debug("Cavity too close to the mold face for a pour funnel")
        return None
    return Funnel(
        center=center,
        axis=up.copy(),
        mouth_radius=float(mouth),
        throat_radius=sprue.radius,
        depth=float(depth),
    )
