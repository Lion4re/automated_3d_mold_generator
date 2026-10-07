"""Registration keys that align mold pieces.

Each key is a male bump on one piece and a matching socket, enlarged by the
clearance, on the mating piece. Keys sit on a flat mating face (axis-aligned
for the halves, any orientation for side pieces) and are kept clear of the
cavity, the gating channels and other seams.

Key shape
---------
A truncated cone with 45 degree walls. With the parting face printed up, the
bump narrows and the socket widens towards the nozzle, so neither needs
support. No surface is steeper than the usual 45 degree overhang limit, so the
same key also prints without support on the side faces of a four-piece mold,
where a hemisphere would leave a near-flat overhang at its rim. The taper
guides the pieces into place when the mold is closed, and the flat top prints
cleanly where a shallow dome would show layer steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import shapely
import trimesh

from moldgen.booleans import to_manifold
from moldgen.gating import cross_section_to_shapely, solid_of_revolution

logger = logging.getLogger(__name__)

KEY_WALL_ANGLE_DEG = 45.0
"""Wall angle measured from the mating face."""
KEY_HEIGHT_RATIO = 0.5
"""Key height as a fraction of its base radius."""
SOCKET_EXTRA_DEPTH = 0.3
"""Extra socket depth (mm) beyond the clearance, so the faces seat before the key bottoms out."""
BOOLEAN_OVERLAP = 0.3
"""How far (mm) key and socket solids reach behind the face so unions and cuts are clean."""
KEY_SECTIONS = 48
"""Facets around each key and socket solid."""
KEY_SPACING_RADII = 3.0
"""Minimum centre distance between keys, in key radii."""
CANDIDATE_COUNT = 4000
"""Approximate number of grid points searched for key positions."""
CENTROID_WEIGHT = 0.5
"""Preference for key positions far from the face centre, which resist rotation best."""
BUFFER_QUAD_SEGMENTS = 16
"""Segments per quarter circle when offsetting obstacle shadows."""
FLAT_TOLERANCE = 1e-6
"""Largest extent (mm) of ``face_bounds`` along the normal still treated as flat."""

_COT = 1.0 / np.tan(np.radians(KEY_WALL_ANGLE_DEG))
_WALL_OFFSET = 1.0 / np.sin(np.radians(KEY_WALL_ANGLE_DEG))


@dataclass
class KeyPlan:
    normal: np.ndarray
    """Unit plane normal. Male keys belong to the piece on the ``-normal`` side
    and protrude towards ``+normal``; sockets are cut into the ``+normal`` piece."""
    positions: np.ndarray
    """(N, 3) key centres lying on the mating face."""
    radius: float
    clearance: float

    @property
    def height(self) -> float:
        """How far a key protrudes from the face (mm)."""
        return KEY_HEIGHT_RATIO * self.radius

    @property
    def socket_depth(self) -> float:
        """How deep sockets reach into the ``+normal`` piece (mm)."""
        return self.height + self.clearance + SOCKET_EXTRA_DEPTH

    @property
    def footprint_radius(self) -> float:
        """Largest radius of any key or socket solid, overlap included (mm)."""
        return self._socket_radius(-BOOLEAN_OVERLAP)

    def male_solids(self, *, sections: int = KEY_SECTIONS) -> list[trimesh.Trimesh]:
        """Bumps to add to the ``-normal`` piece."""
        top = self.height
        profile = [
            (0.0, -BOOLEAN_OVERLAP),
            (self._key_radius(-BOOLEAN_OVERLAP), -BOOLEAN_OVERLAP),
            (self._key_radius(top), top),
            (0.0, top),
        ]
        return [
            solid_of_revolution(profile, center, self.normal, sections) for center in self.positions
        ]

    def female_solids(self, *, sections: int = KEY_SECTIONS) -> list[trimesh.Trimesh]:
        """Sockets to subtract from the ``+normal`` piece: the bump grown by the clearance."""
        self._check_size()
        top = self.socket_depth
        profile = [
            (0.0, -BOOLEAN_OVERLAP),
            (self._socket_radius(-BOOLEAN_OVERLAP), -BOOLEAN_OVERLAP),
            (self._socket_radius(top), top),
            (0.0, top),
        ]
        return [
            solid_of_revolution(profile, center, self.normal, sections) for center in self.positions
        ]

    def _check_size(self) -> None:
        if self._socket_radius(self.socket_depth) <= 0:
            raise ValueError(
                f"Key radius {self.radius} mm is too small for a {self.clearance} mm clearance"
            )

    def _key_radius(self, height: float) -> float:
        return self.radius - height * _COT

    def _socket_radius(self, height: float) -> float:
        return self._key_radius(height) + self.clearance * _WALL_OFFSET


def plan_keys(
    obstacles: list[trimesh.Trimesh],
    face_bounds: np.ndarray,
    normal: np.ndarray,
    *,
    count: int,
    radius: float,
    clearance: float,
    margin: float,
) -> KeyPlan:
    """Place up to ``count`` keys on a rectangular mating face.

    ``face_bounds`` is a (2, 3) axis-aligned box that is flat along ``normal``
    (for example ``[[x0, y0, 0], [x1, y1, 0]]`` for the plane ``z == 0``).
    ``obstacles`` are solids the keys and sockets must stay at least
    ``margin`` away from (the part and the gating channels). Keys are spread
    as far apart as possible. Fewer keys are returned if they do not fit.
    """
    normal = np.asarray(normal, dtype=float).reshape(3)
    axis = int(np.argmax(np.abs(normal)))
    unit = np.zeros(3)
    unit[axis] = np.sign(normal[axis])
    if not np.allclose(normal / np.linalg.norm(normal), unit):
        raise ValueError(f"Key face normal must be axis-aligned, got {normal}")
    face_bounds = np.asarray(face_bounds, dtype=float)
    if face_bounds.shape != (2, 3):
        raise ValueError(f"face_bounds must have shape (2, 3), got {face_bounds.shape}")
    if abs(face_bounds[1, axis] - face_bounds[0, axis]) > FLAT_TOLERANCE:
        raise ValueError("face_bounds must be flat along the normal")
    if count < 0 or radius <= 0 or clearance < 0 or margin < 0:
        raise ValueError("count, clearance and margin must be non-negative and radius positive")

    plan = KeyPlan(
        normal=unit, positions=np.empty((0, 3)), radius=float(radius), clearance=float(clearance)
    )
    plan._check_size()
    if count == 0:
        return plan

    in_plane = [(axis + 1) % 3, (axis + 2) % 3]
    reach = plan.footprint_radius + margin
    low = face_bounds.min(axis=0)[in_plane] + reach
    high = face_bounds.max(axis=0)[in_plane] - reach
    allowed = shapely.box(*low, *high) if np.all(low < high) else shapely.Polygon()
    # Cyclic axis permutation (no mirroring) that moves the face normal onto Z.
    rows = np.eye(3)[[(axis + 1) % 3, (axis + 2) % 3, axis]]
    _place(plan, obstacles, rows, float(face_bounds[0, axis]), allowed, count, margin)
    return plan


def plan_keys_on_plane(
    obstacles: list[trimesh.Trimesh],
    normal: np.ndarray,
    offset: float,
    region: shapely.Geometry,
    *,
    count: int,
    radius: float,
    clearance: float,
    margin: float,
) -> KeyPlan:
    """Place up to ``count`` keys on the plane ``dot(p, normal) == offset``.

    ``region`` is the mating face in the coordinates of :func:`plane_basis`
    (``u = dot(p, rows[0])``, ``v = dot(p, rows[1])``). Male keys protrude
    along ``normal``, as for :func:`plan_keys`; ``normal`` may point anywhere.
    """
    rows = plane_basis(normal)
    if count < 0 or radius <= 0 or clearance < 0 or margin < 0:
        raise ValueError("count, clearance and margin must be non-negative and radius positive")
    plan = KeyPlan(
        normal=rows[2], positions=np.empty((0, 3)), radius=float(radius), clearance=float(clearance)
    )
    plan._check_size()
    if count == 0:
        return plan
    allowed = region.buffer(-(plan.footprint_radius + margin))
    _place(plan, obstacles, rows, float(offset), allowed, count, margin)
    return plan


def plane_basis(normal: np.ndarray) -> np.ndarray:
    """Right-handed rows ``(e1, e2, n)``: two in-plane unit vectors, then the unit normal."""
    n = np.asarray(normal, dtype=float).reshape(3)
    n = n / np.linalg.norm(n)
    helper = np.eye(3)[np.argmin(np.abs(n))]
    e1 = np.cross(helper, n)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(n, e1), n])


def _place(
    plan: KeyPlan,
    obstacles: list[trimesh.Trimesh],
    rows: np.ndarray,
    plane: float,
    allowed: shapely.Geometry,
    count: int,
    margin: float,
) -> None:
    """Fill ``plan.positions`` with up to ``count`` keys in ``allowed``, clear of ``obstacles``.

    ``rows`` maps mold coordinates to face coordinates whose third axis is
    the face normal; the face lies at ``plane`` along that axis.
    """
    reach = plan.footprint_radius + margin
    if obstacles and not allowed.is_empty:
        shadows = [
            _shadow(obstacle, rows, plane, plan.socket_depth + margin) for obstacle in obstacles
        ]
        # Grow the buffer so its polygonal arcs stay outside the true round offset.
        grow = reach / np.cos(np.pi / (4 * BUFFER_QUAD_SEGMENTS))
        forbidden = shapely.union_all(shadows).buffer(grow, quad_segs=BUFFER_QUAD_SEGMENTS)
        allowed = allowed.difference(forbidden)

    spacing = max(KEY_SPACING_RADII * plan.radius, 2.0 * plan.footprint_radius + margin)
    points = _spread_points(allowed, count, spacing)
    plan.positions = points @ rows[:2] + plane * rows[2]
    if len(plan.positions) < count:
        # The pipeline reports this to the user.
        logger.info(
            "Only %d of %d registration keys fit on the mating face", len(plan.positions), count
        )


def _shadow(
    obstacle: trimesh.Trimesh, rows: np.ndarray, plane: float, half_thickness: float
) -> shapely.Geometry:
    """Projection onto the face of the part of ``obstacle`` within ``half_thickness`` of it."""
    solid = to_manifold(obstacle).transform(np.column_stack([rows, np.zeros(3)]))
    slab = solid.trim_by_plane((0.0, 0.0, 1.0), plane - half_thickness).trim_by_plane(
        (0.0, 0.0, -1.0), -(plane + half_thickness)
    )
    return cross_section_to_shapely(slab.project())


def _spread_points(region: shapely.Geometry, count: int, spacing: float) -> np.ndarray:
    """Pick up to ``count`` points in ``region`` at least ``spacing`` apart, spread out.

    Farthest-point sampling over a grid plus the region's boundary vertices,
    starting from the candidate farthest from the region's centroid. Each
    step takes the candidate maximising its distance to the chosen points plus
    ``CENTROID_WEIGHT`` times its distance to the centroid, which favours the
    corners of a rectangular face over points midway along its edges.
    """
    if region.is_empty:
        return np.empty((0, 2))
    x_min, y_min, x_max, y_max = region.bounds
    step = max(np.sqrt((x_max - x_min) * (y_max - y_min) / CANDIDATE_COUNT), 1e-3)
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max + step, step), np.arange(y_min, y_max + step, step)
    )
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid = grid[shapely.contains_xy(region, grid[:, 0], grid[:, 1])]
    candidates = np.vstack([shapely.get_coordinates(region.boundary), grid])

    outwards = np.linalg.norm(candidates - np.array(region.centroid.coords[0]), axis=1)
    chosen = [int(np.argmax(outwards))]
    distance = np.linalg.norm(candidates - candidates[chosen[0]], axis=1)
    while len(chosen) < count:
        score = np.where(distance >= spacing, distance + CENTROID_WEIGHT * outwards, -np.inf)
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            break
        chosen.append(best)
        distance = np.minimum(distance, np.linalg.norm(candidates - candidates[best], axis=1))
    return candidates[chosen]
