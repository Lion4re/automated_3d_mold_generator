"""Pure helpers behind the GUI: settings mapping, viewer geometry and text.

Nothing in this module imports viser, so all of it can be unit-tested
without a browser. Lengths are millimetres; colours are 0-255 RGB tuples.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from moldgen.config import MAX_AUTO_PIECES, MIN_AUTO_PIECES, MoldConfig
from moldgen.materials import MATERIALS, PRINT_MATERIALS, compatibility_warnings
from moldgen.meshio import SUPPORTED_SUFFIXES
from moldgen.parting import FACE_LOW_DRAFT, FACE_OK, FACE_UNDERCUT

if TYPE_CHECKING:
    from moldgen.parting import PartingResult
    from moldgen.pipeline import PreparedPart
    from moldgen.surface import PartingSurface

RGB = tuple[int, int, int]

FACE_COLORS: dict[int, RGB] = {
    FACE_OK: (200, 203, 208),
    FACE_LOW_DRAFT: (226, 158, 42),
    FACE_UNDERCUT: (204, 62, 52),
}
FACE_LABELS: dict[int, str] = {
    FACE_OK: "Releases cleanly",
    FACE_LOW_DRAFT: "Low draft",
    FACE_UNDERCUT: "Undercut",
}
FACE_KEYS: dict[int, str] = {FACE_OK: "ok", FACE_LOW_DRAFT: "low_draft", FACE_UNDERCUT: "undercut"}

# Muted tones, one per piece (up to MAX_AUTO_PIECES), kept clear of the amber and red
# analysis colours so those still stand out on the part and inside the mold.
PIECE_COLORS: tuple[RGB, ...] = (
    (116, 141, 173),
    (139, 163, 126),
    (152, 139, 176),
    (108, 154, 152),
    (106, 106, 88),
    (136, 88, 124),
    (70, 118, 82),
    (118, 196, 172),
    (70, 106, 118),
    (70, 100, 148),
)
FILLED_COLOR: RGB = FACE_COLORS[FACE_UNDERCUT]
FILLED = -1
"""Face label for areas no piece can release, where the cavity was filled."""
PLANE_COLOR: RGB = (56, 110, 182)

AUTO_DIRECTION = "Auto (best)"
AXIS_DIRECTIONS: dict[str, tuple[float, float, float]] = {
    "X axis": (1.0, 0.0, 0.0),
    "Y axis": (0.0, 1.0, 0.0),
    "Z axis": (0.0, 0.0, 1.0),
}
UNIT_LABELS: dict[str, str] = {
    "mm": "Millimetres",
    "cm": "Centimetres",
    "m": "Metres",
    "in": "Inches",
}
AUTO_PIECES = "Automatic"
PIECE_OPTIONS: dict[str, int | str] = {AUTO_PIECES: "auto", "2 pieces": 2, "4 pieces": 4}
MAX_PIECES_RANGE = (MIN_AUTO_PIECES, MAX_AUTO_PIECES)
PARTING_SURFACE_OPTIONS: dict[str, str] = {"Curved where needed": "auto", "Flat": "flat"}


# ---------------------------------------------------------------------------
# Settings


@dataclass
class MoldSettings:
    """Values of the GUI controls that feed :class:`MoldConfig`.

    The "auto" checkboxes are kept separate from the numbers so the number
    fields can show the automatic value without overriding it.
    """

    units: str = "mm"
    scale: float = 1.0
    repair: bool = True
    material: str = "resin"
    print_material: str = "pla"
    pieces: int | str = "auto"
    max_pieces: int = 6
    parting_surface: str = "auto"
    wall_auto: bool = True
    wall_thickness: float = 10.0
    keys: int = 4
    clearance: float = 0.25
    sprue_auto: bool = True
    sprue_diameter: float = 6.0
    funnel: bool = True
    vents: bool = True
    shrinkage_override: bool = False
    shrinkage_percent: float = 0.5
    draft_threshold_deg: float = 1.0
    orient_for_print: bool = True

    @classmethod
    def from_config(cls, config: MoldConfig | None = None) -> MoldSettings:
        """GUI defaults that mirror ``config`` (``MoldConfig()`` when omitted)."""
        config = config or MoldConfig()
        material = MATERIALS.get(config.material) or next(iter(MATERIALS.values()))
        print_material = config.print_material
        if print_material not in PRINT_MATERIALS:
            print_material = next(iter(PRINT_MATERIALS))
        shrinkage = material.linear_shrinkage if config.shrinkage is None else config.shrinkage
        return cls(
            units=config.units,
            scale=config.scale,
            repair=config.repair,
            material=material.key,
            print_material=print_material,
            pieces=config.pieces,
            max_pieces=config.max_pieces,
            parting_surface=config.parting_surface,
            wall_auto=config.wall_thickness is None,
            wall_thickness=config.wall_thickness or material.min_wall_mm,
            keys=config.keys,
            clearance=config.clearance,
            sprue_auto=config.sprue_diameter is None,
            sprue_diameter=config.sprue_diameter or material.sprue_diameter_mm,
            funnel=config.funnel,
            vents=config.vents,
            shrinkage_override=config.shrinkage is not None,
            shrinkage_percent=round(100.0 * shrinkage, 3),
            draft_threshold_deg=config.draft_threshold_deg,
            orient_for_print=config.orient_for_print,
        )


def to_config(settings: MoldSettings, parting: PartingResult | None = None) -> MoldConfig:
    """Build a :class:`MoldConfig`; with ``parting`` the chosen plane is recorded in it.

    Raises :class:`moldgen.config.ConfigError` for invalid values.
    """
    if parting is not None:
        direction: str | tuple[float, ...] = tuple(float(v) for v in parting.direction)
        offset: float | None = float(parting.offset)
    else:
        direction, offset = "auto", None
    config = MoldConfig(
        material=settings.material,
        print_material=settings.print_material,
        units=settings.units,  # type: ignore[arg-type]
        scale=float(settings.scale),
        direction=direction,
        parting_offset=offset,
        pieces=settings.pieces if settings.pieces == "auto" else int(settings.pieces),
        max_pieces=int(settings.max_pieces),
        parting_surface=settings.parting_surface,
        wall_thickness=None if settings.wall_auto else float(settings.wall_thickness),
        shrinkage=float(settings.shrinkage_percent) / 100.0
        if settings.shrinkage_override
        else None,
        sprue_diameter=None if settings.sprue_auto else float(settings.sprue_diameter),
        funnel=bool(settings.funnel),
        vents=bool(settings.vents),
        keys=int(settings.keys),
        clearance=float(settings.clearance),
        draft_threshold_deg=float(settings.draft_threshold_deg),
        repair=bool(settings.repair),
        orient_for_print=bool(settings.orient_for_print),
    )
    config.validate()
    return config


def material_options() -> dict[str, str]:
    """Dropdown label -> casting material key."""
    return _unique_labels((m.name, m.key) for m in MATERIALS.values())


def print_material_options() -> dict[str, str]:
    """Dropdown label -> print material key."""
    return _unique_labels((p.name, p.key) for p in PRINT_MATERIALS.values())


def _unique_labels(pairs: Any) -> dict[str, str]:
    options: dict[str, str] = {}
    for label, key in pairs:
        if label in options:
            label = f"{label} ({key})"
        options[label] = key
    return options


def label_for(options: Mapping[str, Any], value: Any) -> str:
    """Inverse lookup in a label -> value mapping (first label if not found)."""
    for label, candidate in options.items():
        if candidate == value:
            return label
    return next(iter(options))


def auto_wall_estimate(extents: Sequence[float], material_key: str) -> float:
    """Wall thickness the pipeline will pick automatically, estimated from the part size."""
    from moldgen.pipeline import MAX_AUTO_WALL_MM, WALL_FRACTION_OF_EXTENT

    material = MATERIALS[material_key]
    typical = float(np.median(np.asarray(extents, dtype=float)))
    return float(np.clip(WALL_FRACTION_OF_EXTENT * typical, material.min_wall_mm, MAX_AUTO_WALL_MM))


# ---------------------------------------------------------------------------
# Parting directions


@dataclass(frozen=True)
class DirectionChoice:
    label: str
    direction: np.ndarray
    offset: float | None
    """Best plane position for this direction when already known."""


def direction_label(vector: Sequence[float]) -> str:
    """Short label: +X, -Z and so on for axis directions, else the rounded vector."""
    vec = np.asarray(vector, dtype=float)
    axis = int(np.argmax(np.abs(vec)))
    if abs(vec[axis]) > 0.999:
        return ("+" if vec[axis] > 0 else "-") + "XYZ"[axis]
    return "(" + ", ".join(f"{round(float(v), 2) + 0.0:.2f}" for v in vec) + ")"


def direction_choices(analysis: PartingResult) -> list[DirectionChoice]:
    """Dropdown entries: the automatic choice, the ranked candidates and the three axes."""
    choices = [
        DirectionChoice(AUTO_DIRECTION, np.asarray(analysis.direction, float), analysis.offset)
    ]
    for rank, candidate in enumerate(analysis.candidates, start=1):
        label = (
            f"{rank}. {direction_label(candidate.direction)}, "
            f"{candidate.undercut_fraction:.1%} undercut"
        )
        choices.append(
            DirectionChoice(label, np.asarray(candidate.direction, float), candidate.offset)
        )
    for label, vector in AXIS_DIRECTIONS.items():
        choices.append(DirectionChoice(label, np.asarray(vector, float), None))
    return choices


def surface_fractions(areas: np.ndarray, face_class: np.ndarray) -> tuple[float, float]:
    """Area-weighted (undercut, low draft) fractions of a face classification."""
    areas = np.asarray(areas, dtype=float)
    total = float(areas.sum())
    if total <= 0:
        return 0.0, 0.0
    undercut = float(areas[face_class == FACE_UNDERCUT].sum()) / total
    low_draft = float(areas[face_class == FACE_LOW_DRAFT].sum()) / total
    return undercut, low_draft


def projection_range(vertices: np.ndarray, direction: Sequence[float]) -> tuple[float, float]:
    """Smallest and largest ``dot(p, direction)`` over the vertices."""
    proj = np.asarray(vertices, dtype=float) @ np.asarray(direction, dtype=float)
    return float(proj.min()), float(proj.max())


def nice_step(span: float, divisions: int = 200) -> float:
    """A 1/2/5 x 10^k step that splits ``span`` into roughly ``divisions`` parts."""
    if not span > 0 or not math.isfinite(span):
        return 0.1
    raw = span / divisions
    power = 10.0 ** math.floor(math.log10(raw))
    for factor in (1.0, 2.0, 5.0, 10.0):
        if raw <= factor * power:
            return factor * power
    return 10.0 * power


def step_precision(step: float) -> int:
    """Decimal places needed to show values that are multiples of ``step``."""
    if not step > 0:
        return 2
    return max(0, -math.floor(math.log10(step) + 1e-9))


def slider_range(lo: float, hi: float, divisions: int = 200) -> tuple[float, float, float]:
    """Round ``[lo, hi]`` outwards to a nice step; returns ``(min, max, step)``."""
    step = nice_step(hi - lo, divisions)
    digits = step_precision(step)
    return (
        round(math.floor(lo / step) * step, digits),
        round(math.ceil(hi / step) * step, digits),
        step,
    )


def plane_basis(direction: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed orthonormal ``(u, v, d)`` with ``d`` along ``direction``."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, d)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v, d


@dataclass(frozen=True)
class PlaneFrame:
    """Placement of the parting plane quad in the part frame."""

    position: np.ndarray
    """Centre of the quad, on the plane."""
    wxyz: np.ndarray
    """Rotation taking local +Z to the parting direction."""
    half_size: tuple[float, float]
    """Half extents of the quad along its local X and Y."""


def plane_frame(
    vertices: np.ndarray, direction: Sequence[float], offset: float, margin: float = 0.12
) -> PlaneFrame:
    """Quad covering the part's projection onto the plane, grown by ``margin`` of its size."""
    u, v, d = plane_basis(direction)
    pts = np.asarray(vertices, dtype=float)
    pu, pv = pts @ u, pts @ v
    lo = np.array([pu.min(), pv.min()])
    hi = np.array([pu.max(), pv.max()])
    pad = max(margin * float((hi - lo).max()), 1.0)
    centre = (lo + hi) / 2
    half = (hi - lo) / 2 + pad
    rotation = np.eye(4)
    rotation[:3, :3] = np.column_stack([u, v, d])
    return PlaneFrame(
        position=d * offset + u * centre[0] + v * centre[1],
        wxyz=np.asarray(trimesh.transformations.quaternion_from_matrix(rotation), float),
        half_size=(float(half[0]), float(half[1])),
    )


def quad_mesh(half_size: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and faces of a rectangle in the local XY plane."""
    hx, hy = half_size
    vertices = np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return vertices, faces


def quad_outline(half_size: tuple[float, float]) -> np.ndarray:
    """(4, 2, 3) line segments along the edge of :func:`quad_mesh`."""
    corners, _ = quad_mesh(half_size)
    return np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)


def surface_mesh(
    surface: PartingSurface, block_bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and faces of a parting surface, clipped to the block's XY bounds (mold frame).

    The grid is triangulated like :meth:`PartingSurface.below`: each cell is split
    along its (i, j)-(i+1, j+1) diagonal.
    """
    nx, ny = surface.heights.shape
    vertices = np.column_stack([surface.nodes(), surface.heights.ravel()])
    index = np.arange(nx * ny).reshape(nx, ny)
    a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
    c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    lo, hi = np.asarray(block_bounds, dtype=float)
    for normal, origin in (((1, 0, 0), lo), ((-1, 0, 0), hi), ((0, 1, 0), lo), ((0, -1, 0), hi)):
        vertices, faces = trimesh.intersections.slice_faces_plane(
            vertices, faces, np.asarray(normal, dtype=float), origin
        )[:2]
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.uint32)


def offset_from_position(position: Sequence[float], direction: Sequence[float]) -> float:
    return float(np.dot(np.asarray(position, float), np.asarray(direction, float)))


# ---------------------------------------------------------------------------
# Meshes for display


def shading_split(mesh: trimesh.Trimesh, crease_deg: float = 35.0) -> tuple[np.ndarray, np.ndarray]:
    """Duplicate vertices along sharp edges so the viewer's smooth shading keeps creases.

    Face order is preserved, so per-face data (like the face classes) still
    lines up with the returned faces.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)
    n_corners = 3 * len(faces)
    if n_corners == 0:
        return vertices.astype(np.float32), faces.astype(np.uint32)
    pairs = np.asarray(mesh.face_adjacency, dtype=np.int64)
    smooth = np.asarray(mesh.face_adjacency_angles) < np.radians(crease_deg)
    pairs = pairs[smooth]
    edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)[smooth]
    rows, cols = [], []
    for j in range(2):
        shared = edges[:, j][:, None]
        corner_a = pairs[:, 0] * 3 + np.argmax(faces[pairs[:, 0]] == shared, axis=1)
        corner_b = pairs[:, 1] * 3 + np.argmax(faces[pairs[:, 1]] == shared, axis=1)
        rows.append(corner_a)
        cols.append(corner_b)
    row = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    col = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    graph = coo_matrix((np.ones(len(row), dtype=np.int8), (row, col)), shape=(n_corners,) * 2)
    _, labels = connected_components(graph, directed=False)
    corner_vertex = faces.reshape(-1)
    # Corners that share a label always share the original vertex, so any representative works.
    new_vertices = np.zeros((labels.max() + 1, 3), dtype=float)
    new_vertices[labels] = vertices[corner_vertex]
    return new_vertices.astype(np.float32), labels.reshape(-1, 3).astype(np.uint32)


def submesh(
    vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Faces selected by ``mask`` with only the vertices they use."""
    picked = np.asarray(faces)[np.asarray(mask, dtype=bool)]
    used, inverse = np.unique(picked.reshape(-1), return_inverse=True)
    return (
        np.asarray(vertices)[used].astype(np.float32),
        inverse.reshape(-1, 3).astype(np.uint32),
    )


def explode_directions(piece_bounds: Sequence[np.ndarray], block_bounds: np.ndarray) -> np.ndarray:
    """Unit vectors (mold frame) along which each piece moves away from the others.

    A piece moves along every axis on which its bounding box sits clearly to
    one side of the block centre, so halves separate along the parting
    direction and quarter pieces also move apart across the second seam.
    """
    block = np.asarray(block_bounds, dtype=float)
    centre = block.mean(axis=0)
    half = np.maximum((block[1] - block[0]) / 2, 1e-9)
    out = np.zeros((len(piece_bounds), 3))
    for i, bounds in enumerate(piece_bounds):
        rel = (np.asarray(bounds, float).mean(axis=0) - centre) / half
        vec = np.where(np.abs(rel) > 0.2, np.sign(rel), 0.0)
        if not vec.any():
            vec = rel
        norm = np.linalg.norm(vec)
        out[i] = vec / norm if norm > 1e-12 else 0.0
    return out


def piece_color(index: int) -> RGB:
    """Colour of the piece at ``index`` in removal order."""
    return PIECE_COLORS[index % len(PIECE_COLORS)]


def face_pieces(
    face_region: np.ndarray, side_pieces: int, top_needed: bool, piece_names: Sequence[str]
) -> np.ndarray:
    """Map :attr:`PieceLayout.face_region` to indices into ``piece_names`` (or ``FILLED``).

    Regions are numbered side pieces first, then top, then bottom; when the
    top is not needed its region belongs to the bottom piece.
    """
    regions = [f"side_{k + 1}" for k in range(side_pieces)]
    regions += ["top" if top_needed else "bottom", "bottom"]
    names = list(piece_names)
    # The trailing FILLED entry is what region -1 indexes. A piece that came out
    # empty releases nothing, so its region cannot hold faces in practice.
    lookup = [names.index(name) if name in names else FILLED for name in regions] + [FILLED]
    return np.asarray(lookup, dtype=np.int64)[np.asarray(face_region, dtype=np.int64)]


def legend_rows(info: Mapping[str, Any]) -> list[tuple[str, str, RGB]]:
    """(name, pull direction, colour) per piece of :func:`moldgen.report.summary`.

    A "filled" row with the share of the surface follows when locked areas were filled.
    """
    rows = [(p["name"], p["pull_label"], piece_color(i)) for i, p in enumerate(info["pieces"])]
    layout = info.get("layout")
    if layout and layout["locked_fraction"] > 0:
        rows.append(("filled", _percent(layout["locked_fraction"]), FILLED_COLOR))
    return rows


def scene_floor(min_z: Sequence[float], offsets: np.ndarray | None = None) -> float:
    """Lowest point of the displayed geometry, given each object's lowest z and displacement."""
    z = np.asarray(min_z, dtype=float)
    if offsets is not None:
        z = z + np.asarray(offsets, dtype=float)[:, 2]
    return float(z.min())


def grid_spacing(extent: float) -> tuple[float, float]:
    """(cell, section) sizes for the ground grid, roughly 8 cells across the part."""
    if not extent > 0 or not math.isfinite(extent):
        return 10.0, 50.0
    cell = nice_step(extent, 8)
    mantissa = round(cell / 10.0 ** math.floor(math.log10(cell)))
    return cell, cell * (10 if mantissa == 5 else 5)


@dataclass(frozen=True)
class CameraPose:
    position: np.ndarray
    look_at: np.ndarray
    near: float
    far: float


VIEW_DIRECTION = np.array([0.95, -1.35, 0.85]) / np.linalg.norm([0.95, -1.35, 0.85])


def side_view_direction(axis: Sequence[float], keep: float = 0.2) -> np.ndarray:
    """Viewing direction mostly across ``axis``, so a gap opened along it is visible."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    view = VIEW_DIRECTION - (1.0 - keep) * float(VIEW_DIRECTION @ a) * a
    if view[2] < 0.25:
        view[2] = 0.25
    return view / np.linalg.norm(view)


def camera_pose(
    bounds: np.ndarray,
    fov_rad: float = math.radians(50.0),
    view_direction: Sequence[float] | None = None,
) -> CameraPose:
    """A three-quarter view (front right by default) that fits ``bounds`` on screen."""
    b = np.asarray(bounds, dtype=float)
    centre = b.mean(axis=0)
    radius = max(float(np.linalg.norm(b[1] - b[0])) / 2, 1.0)
    distance = 1.3 * radius / math.sin(fov_rad / 2)
    view = VIEW_DIRECTION if view_direction is None else np.asarray(view_direction, float)
    view = view / np.linalg.norm(view)
    return CameraPose(
        position=centre + view * distance,
        look_at=centre,
        near=max(distance / 500.0, 0.05),
        far=distance * 50.0,
    )


# ---------------------------------------------------------------------------
# Text


def format_length(value: float) -> str:
    return f"{value:.1f}" if abs(value) < 1000 else f"{value:.0f}"


def format_size(extents: Sequence[float]) -> str:
    return " x ".join(format_length(float(v)) for v in extents) + " mm"


TEXT_STYLE = "font-size:0.875em;line-height:1.45;padding:0 0.75em 0.5em"
DIMMED = "color:var(--mantine-color-dimmed)"
WARNING_STYLE = "color:#a15c07"
NUMERIC = "font-variant-numeric:tabular-nums"


def info_html(
    rows: Sequence[tuple[str, str]] = (),
    *,
    title: str | None = None,
    intro: str | None = None,
    notes: Sequence[str] = (),
    warnings: Sequence[str] = (),
) -> str:
    """A compact block of label/value rows with optional title, notes and warnings.

    All text is HTML-escaped.
    """
    parts = [f'<div style="{TEXT_STYLE}">']
    if title:
        parts.append(f'<div style="font-weight:600;margin-bottom:0.2em">{_esc(title)}</div>')
    if intro:
        parts.append(f'<div style="{DIMMED};margin-bottom:0.35em">{_esc(intro)}</div>')
    if rows:
        parts.append(
            '<div style="display:grid;grid-template-columns:auto 1fr;'
            'column-gap:0.75em;row-gap:0.15em">'
        )
        for label, value in rows:
            parts.append(f'<span style="{DIMMED}">{_esc(label)}</span><span>{_esc(value)}</span>')
        parts.append("</div>")
    for note in notes:
        parts.append(f'<div style="{DIMMED};margin-top:0.35em">{_esc(note)}</div>')
    for warning in warnings:
        parts.append(f'<div style="{WARNING_STYLE};margin-top:0.35em">{_esc(warning)}</div>')
    parts.append("</div>")
    return "".join(parts)


def text_html(text: str) -> str:
    """A short paragraph in the panel's text style."""
    return f'<div style="{TEXT_STYLE}">{_esc(text)}</div>'


def part_html(part: PreparedPart, units: str) -> str:
    """Name, size, volume and repair report of a loaded part."""
    mesh = part.mesh
    rows = [
        ("Size", format_size(mesh.extents)),
        ("Volume", f"{float(mesh.volume) / 1000.0:.2f} cm³"),
        ("Triangles", f"{len(mesh.faces):,}"),
    ]
    if units != "mm":
        rows.append(("File units", f"{UNIT_LABELS.get(units, units)}, converted to mm"))
    report = part.repair
    if report.actions:
        rows.append(("Repairs", "; ".join(report.actions)))
    elif report.was_watertight:
        rows.append(("Mesh", "Closed solid, no repair needed"))
    return info_html(rows, title=part.name, warnings=part.warnings)


def empty_part_html() -> str:
    formats = ", ".join(s.lstrip(".").upper() for s in SUPPORTED_SUFFIXES)
    return info_html(intro=f"Open a model to begin. Supported formats: {formats}.")


def material_html(material_key: str, print_key: str) -> str:
    """Description, key numbers, notes and temperature warnings for a material pair."""
    material = MATERIALS[material_key]
    printing = PRINT_MATERIALS[print_key]
    if material.pour_temp_c is None:
        pour = "Room temperature"
    else:
        low, high = material.pour_temp_c
        pour = f"{low:.0f} to {high:.0f} °C"
    rows = [
        ("Shrinkage", f"{100 * material.linear_shrinkage:.2g} %"),
        ("Pour at", pour),
    ]
    notes = [_sentence(material.notes)] if material.notes else []
    if printing.notes:
        notes.append(f"{printing.name}: {_sentence(printing.notes)}")
    return info_html(
        rows,
        intro=_sentence(material.description),
        notes=notes,
        warnings=compatibility_warnings(material, printing),
    )


def readout_html(undercut: float | None, low_draft: float | None) -> str:
    """Colour legend with the share of the surface in each class."""
    if undercut is None or low_draft is None:
        values = dict.fromkeys(FACE_COLORS, "-")
    else:
        ok = max(0.0, 1.0 - undercut - low_draft)
        values = {
            FACE_OK: _percent(ok),
            FACE_LOW_DRAFT: _percent(low_draft),
            FACE_UNDERCUT: _percent(undercut),
        }
    caption = (
        f'<div style="{DIMMED};font-size:0.8em;padding:0.3em 0.75em 0.1em">'
        "For a flat cut at the plane shown. Generate to see the result of the actual "
        "mold, which may curve or add side pieces.</div>"
    )
    return caption + swatch_html(
        [
            (FACE_LABELS[c], values[c], FACE_COLORS[c])
            for c in (FACE_OK, FACE_LOW_DRAFT, FACE_UNDERCUT)
        ]
    )


def swatch_html(rows: Sequence[tuple[str, str, RGB]]) -> str:
    """Colour legend: a swatch, a label and a right-aligned value per row."""
    cells = []
    for label, value, (r, g, b) in rows:
        cells.append(
            f'<span style="width:0.75em;height:0.75em;border-radius:2px;'
            f'background:rgb({r},{g},{b})"></span>'
            f"<span>{_esc(label)}</span>"
            f'<span style="text-align:right;{NUMERIC}">{_esc(value)}</span>'
        )
    return (
        '<div style="display:grid;grid-template-columns:auto 1fr auto;align-items:center;'
        'column-gap:0.5em;row-gap:0.3em;font-size:0.875em;padding:0.1em 0.75em 0.4em">'
        + "".join(cells)
        + "</div>"
    )


def _percent(fraction: float) -> str:
    return f"{100 * fraction:.1f} %"


def summary_html(info: Mapping[str, Any]) -> str:
    """Readable digest of :func:`moldgen.report.summary` with a per-piece table."""
    mold = info["mold"]
    parting = info["parting"]
    sprue = f"{mold['sprue_diameter_mm']:.1f} mm" + (", with funnel" if mold["funnel"] else "")
    layout = info.get("layout")
    if parting.get("surface") == "curved":
        split = f"curved surface, up to {parting['surface_rise_mm']:.1f} mm from flat"
    else:
        split = "flat plane"
    # What can still lock in this mold, after the cut, any side pieces and any filling.
    if not layout:
        undercut = parting["undercut_fraction"]
    elif layout["filled_volume_cm3"] > 0:
        undercut = layout["remaining_locked_fraction"]
    else:
        undercut = layout["locked_fraction"]
    rows = [
        ("Outer size", format_size(mold["outer_size_mm"])),
        ("Wall", f"{mold['wall_thickness_mm']:.1f} mm"),
        ("Parting", f"{parting['direction_label']}, {split}"),
        ("Undercut", f"{_percent(undercut)} of the surface"),
        ("Sprue", sprue),
        ("Air vents", str(mold["vents"])),
        ("Keys", f"{mold['keys']}, {mold['key_clearance_mm']:.2f} mm clearance"),
        ("Cast volume", f"{info['material']['cast_volume_cm3']:.2f} cm³"),
    ]
    if layout and layout["side_pieces"]:
        rows.append(("Removal order", ", ".join(p["name"] for p in info["pieces"])))
    if layout and layout["filled_volume_cm3"] > 0:
        rows.append(
            (
                "Filled",
                f"{_percent(layout['locked_fraction'])} of the surface, "
                f"{layout['filled_volume_cm3']:.2f} cm³ added to the cast",
            )
        )
    table = [
        '<div style="display:grid;grid-template-columns:1fr auto auto;column-gap:0.75em;'
        f'row-gap:0.15em;margin-top:0.6em;{NUMERIC}">',
        f'<span style="{DIMMED}">Piece</span>',
        f'<span style="{DIMMED}">Print size (mm)</span>',
        f'<span style="{DIMMED};text-align:right">Mass</span>',
    ]
    for piece in info["pieces"]:
        size = " x ".join(f"{v:.0f}" for v in piece["print_size_mm"])
        table += [
            f"<span>{_esc(piece['name'])}</span>",
            f"<span>{size}</span>",
            f'<span style="text-align:right">{piece["approx_mass_g"]:.0f} g</span>',
        ]
    table.append("</div>")
    block = info_html(rows)
    return block[: -len("</div>")] + "".join(table) + "</div>"


def warnings_html(warnings: Sequence[str]) -> str:
    return info_html(title="Warnings", warnings=warnings) if warnings else ""


def download_name(part_name: str) -> str:
    """Safe file name for the zip download."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", part_name).strip("._") or "part"
    return f"{stem}_mold.zip"


def _sentence(text: str) -> str:
    text = text.strip()
    return text[:1].upper() + text[1:]


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)
