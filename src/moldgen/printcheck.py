"""Checks a print would reveal, done on the pieces before printing them.

Each piece is checked as it will lie on the build plate (``MoldPiece.print_mesh``):

- does it fit the printer's bed;
- does it rest on enough flat area to stay put while printing;
- how much of it overhangs steeply enough to need supports;
- where its walls are thinner than the printer can make solid;
- whether it is small enough to be fragile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

OVERHANG_ANGLE_DEG = 45.0
"""Faces facing down more steeply than this from vertical need supports."""

MIN_WALL_FDM_MM = 1.2
"""Thinnest wall a filament printer makes solid: about three 0.4 mm lines."""

MIN_WALL_RESIN_MM = 0.8
"""Thinnest wall a resin printer makes reliably."""

BED_CONTACT_TOLERANCE_MM = 0.05
"""Faces this close to the lowest point, facing down, rest on the bed."""

MIN_CONTACT_SHARE = 0.15
"""A piece resting on less than this share of its footprint may tip or come loose."""

THICKNESS_SAMPLES = 4000
"""Faces sampled (by area) to measure wall thickness."""

WALL_COS = 0.5
"""The far side of a wall faces back within 60 degrees of the near side."""

FLAT_FACE_CANDIDATES = 6
"""Largest flat regions tried as the face to print a piece on."""

OVERHANG_WARN_MM2 = 50.0
THIN_WARN_MM2 = 20.0
"""Overhang or thin area above which a piece gets a warning."""

FRAGILE_VOLUME_MM3 = 500.0
FRAGILE_SIZE_MM = 4.0
"""A piece under this volume, or this thin in any direction, is easy to break."""


@dataclass
class PieceCheck:
    name: str
    size_mm: np.ndarray
    """Size as printed (x, y, z)."""
    fits_bed: bool
    contact_mm2: float
    """Flat area resting on the bed."""
    overhang_mm2: float
    """Area facing down steeply enough to need supports."""
    thin_mm2: float
    """Area where the wall is thinner than the printer can make solid."""
    min_wall_mm: float
    """Thinnest wall found (inf if no wall was measured)."""
    fragile: bool

    def warnings(self, bed: tuple[float, float, float], min_wall: float) -> list[str]:
        label = f"The {self.name.replace('_', ' ')} piece"
        notes = []
        if not self.fits_bed:
            size = " x ".join(f"{v:.0f}" for v in self.size_mm)
            bed_size = " x ".join(f"{v:.0f}" for v in bed)
            notes.append(
                f"{label} ({size} mm) does not fit a {bed_size} mm bed; use more pieces, "
                "a smaller scale or a larger printer."
            )
        footprint = float(self.size_mm[0] * self.size_mm[1])
        if footprint > 0 and self.contact_mm2 < MIN_CONTACT_SHARE * footprint:
            notes.append(f"{label} rests on little flat area; use a brim so it stays put.")
        if self.overhang_mm2 > OVERHANG_WARN_MM2:
            notes.append(
                f"{label} has {self.overhang_mm2:.0f} mm² of steep overhang and needs supports "
                "there; check the cavity side is free of them."
            )
        if self.thin_mm2 > THIN_WARN_MM2:
            notes.append(
                f"{label} is thinner than the {min_wall:.1f} mm this printer makes solid over "
                f"{self.thin_mm2:.0f} mm², at feather edges where a cut meets the cast at a "
                "shallow angle or in narrow gaps of the part; these spots may chip or print "
                "incompletely."
            )
        if self.fragile:
            notes.append(f"{label} is very small and may break; handle it with care.")
        return notes


def check_piece(
    name: str, mesh: trimesh.Trimesh, bed: tuple[float, float, float], min_wall: float
) -> PieceCheck:
    """Check one piece, given as it lies on the bed (lowest point at z == 0)."""
    size = mesh.extents
    footprint = np.sort(size[:2])
    fits = bool(np.all(footprint <= np.sort(bed[:2])) and size[2] <= bed[2])
    normals = mesh.face_normals
    centres_z = mesh.triangles_center[:, 2]
    low = float(mesh.bounds[0, 2])
    on_bed = (normals[:, 2] < -0.99) & (centres_z <= low + BED_CONTACT_TOLERANCE_MM)
    steep = normals[:, 2] < -np.cos(np.radians(OVERHANG_ANGLE_DEG))
    areas = mesh.area_faces
    thin_area, min_wall_found = _thin_walls(mesh, min_wall)
    return PieceCheck(
        name=name,
        size_mm=size,
        fits_bed=fits,
        contact_mm2=float(areas[on_bed].sum()),
        overhang_mm2=float(areas[steep & ~on_bed].sum()),
        thin_mm2=thin_area,
        min_wall_mm=min_wall_found,
        fragile=bool(abs(mesh.volume) < FRAGILE_VOLUME_MM3 or size.min() < FRAGILE_SIZE_MM),
    )


def _thin_walls(mesh: trimesh.Trimesh, min_wall: float) -> tuple[float, float]:
    """Area thinner than ``min_wall`` and the thinnest wall, from rays cast inwards.

    A ray from just inside each sampled face, against its normal, meets the
    other side of the wall; the distance is the wall's thickness there. Only
    hits on a far side that faces roughly back count: near a sharp edge the ray
    meets the neighbouring face at an angle, which is an edge, not a thin wall.
    """
    areas = mesh.area_faces
    total = float(areas.sum())
    if total == 0:
        return 0.0, float("inf")
    count = min(THICKNESS_SAMPLES, len(mesh.faces))
    rng = np.random.default_rng(0)
    faces = rng.choice(len(mesh.faces), size=count, replace=False, p=areas / total)
    normals = mesh.face_normals[faces]
    origins = mesh.triangles_center[faces] - 1e-4 * normals
    hits, rays, hit_faces = mesh.ray.intersects_location(origins, -normals, multiple_hits=False)
    if len(rays) == 0:
        return 0.0, float("inf")
    thickness = np.full(count, np.inf)
    distance = np.linalg.norm(np.asarray(hits).reshape(-1, 3) - origins[rays], axis=1)
    facing_back = np.einsum("ij,ij->i", mesh.face_normals[hit_faces], normals[rays]) < -WALL_COS
    thickness[rays[facing_back]] = distance[facing_back]
    thin = thickness < min_wall
    sampled = areas[faces]
    # Scale the thin share of the sampled area up to the whole surface.
    return float(sampled[thin].sum() / sampled.sum() * total), float(thickness.min())


def best_print_up(mesh: trimesh.Trimesh, preferred: np.ndarray) -> np.ndarray:
    """The direction to point up when printing ``mesh``: the least area needing supports.

    Candidates are ``preferred`` (for a mold piece, the side facing against its
    pull, so the cavity opens upwards) and the opposites of the piece's largest
    flat faces, so that face lies on the bed. Ties go to ``preferred``.
    """
    preferred = np.asarray(preferred, dtype=float) / np.linalg.norm(preferred)
    candidates = [preferred]
    for normal in _flat_faces(mesh):
        candidates.append(-normal)
    scores = [_support_area(mesh, up) for up in candidates]
    best = int(np.argmin(np.round(scores, 1)))  # first wins ties: the preferred side
    return candidates[best]


def _flat_faces(mesh: trimesh.Trimesh, count: int = FLAT_FACE_CANDIDATES) -> list[np.ndarray]:
    """Normals of the largest flat regions of ``mesh``, largest first."""
    normals = np.round(mesh.face_normals, 3)
    unique, inverse = np.unique(normals, axis=0, return_inverse=True)
    area = np.bincount(inverse.ravel(), weights=mesh.area_faces)
    order = np.argsort(-area)[:count]
    return [unique[i] / np.linalg.norm(unique[i]) for i in order if area[i] > 0]


def _support_area(mesh: trimesh.Trimesh, up: np.ndarray) -> float:
    """Area needing supports with ``up`` pointing up; infinite if it would not sit flat."""
    heights = mesh.vertices @ up
    face_heights = heights[mesh.faces].mean(axis=1)
    facing = mesh.face_normals @ up
    on_bed = (facing < -0.99) & (face_heights <= heights.min() + BED_CONTACT_TOLERANCE_MM)
    areas = mesh.area_faces
    footprint = _footprint_area(mesh, up)
    if footprint > 0 and areas[on_bed].sum() < MIN_CONTACT_SHARE * footprint:
        return float("inf")
    steep = facing < -np.cos(np.radians(OVERHANG_ANGLE_DEG))
    return float(areas[steep & ~on_bed].sum())


def _footprint_area(mesh: trimesh.Trimesh, up: np.ndarray) -> float:
    """Area of the rectangle around the piece seen from above with ``up`` pointing up."""
    helper = np.eye(3)[np.argmin(np.abs(up))]
    e1 = np.cross(up, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    flat = mesh.vertices @ np.column_stack([e1, e2])
    return float(np.prod(np.ptp(flat, axis=0)))
