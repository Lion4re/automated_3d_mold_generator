"""End-to-end mold generation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import manifold3d
import numpy as np
import shapely
import shapely.affinity
import trimesh

from moldgen import booleans
from moldgen.config import MoldConfig
from moldgen.gating import POUR_DIRECTIONS, GatingPlan, cross_section_to_shapely, plan_gating
from moldgen.keys import KeyPlan, plan_keys, plan_keys_on_plane, plane_basis
from moldgen.materials import (
    Material,
    PrintMaterial,
    compatibility_warnings,
    get_material,
    get_print_material,
)
from moldgen.meshio import load_mesh, units_warning
from moldgen.parting import PartingResult, analyze_parting
from moldgen.pieces import (
    QUICK_DIRECTIONS,
    UP,
    Cap,
    CastFaces,
    PieceLayout,
    block_solid,
    cap_region,
    choose_main_direction,
    core_half,
    face_regions,
    locked_faces,
    plan_caps,
    plan_layout,
    trim,
)
from moldgen.repair import RepairReport, repair_mesh
from moldgen.surface import KEY_MAX_SLOPE, PartingSurface, fit_surface

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]

UNDERCUT_WARNING_FRACTION = 0.005
WALL_FRACTION_OF_EXTENT = 0.2
MAX_AUTO_WALL_MM = 30.0
KEY_RADIUS_FRACTION_OF_WALL = 0.3
MIN_KEY_RADIUS_MM = 1.5
MAX_KEY_RADIUS_MM = 5.0
KEY_MARGIN_FRACTION_OF_WALL = 0.25
MIN_KEY_MARGIN_MM = 1.0
MAX_KEY_MARGIN_MM = 3.0
SECONDARY_KEYS_PER_SEAM = 2
SIDE_PIECE_KEYS = 2
SEAM_SLAB_HALF_MM = 0.05
"""Half thickness of the slabs that keep keys of one seam off the other seams."""
NOTICEABLE_FILL_MM3 = 10.0
"""Filling smaller than this (0.01 cm³) is reported in the summary but not warned about."""
SLIVER_VOLUME_FRACTION = 1e-6
"""Shells of the mold body smaller than this share of its volume are numerical slivers."""
REMOVAL_STEPS_FRACTION = (0.01, 0.05, 0.2, 1.0)
"""Distances (times the block size) each piece is slid along its pull to check removal."""
REMOVAL_OVERLAP_FRACTION = 1e-4
"""Overlap (times the piece volume) tolerated when sliding, for the release tolerance."""
SLICE_INSET = 1e-5
"""How far (times the block size) from a seam its two sides are sliced to find the mating face."""


class MoldError(RuntimeError):
    """A problem with the input or settings that prevents building a usable mold."""


@dataclass
class PreparedPart:
    """A loaded, unit-converted and repaired part, ready for analysis."""

    name: str
    mesh: trimesh.Trimesh
    repair: RepairReport
    warnings: list[str] = field(default_factory=list)


@dataclass
class MoldPiece:
    name: str
    mesh: trimesh.Trimesh
    """The piece in its assembled position (mold frame)."""
    print_transform: np.ndarray
    """Places the piece on the build plate with its parting face up."""
    pull: np.ndarray = field(default_factory=lambda: np.zeros(3))
    """Direction the piece is pulled off the cast (mold frame)."""

    def print_mesh(self) -> trimesh.Trimesh:
        return self.mesh.copy().apply_transform(self.print_transform)


@dataclass
class MoldResult:
    config: MoldConfig
    material: Material
    print_material: PrintMaterial
    part: PreparedPart
    parting: PartingResult
    cavity: trimesh.Trimesh
    """The part as it sits in the mold: mold frame, shrinkage compensation applied."""
    shrink_scale: float
    wall_thickness: float
    block_bounds: np.ndarray
    gating: GatingPlan
    keys: list[KeyPlan]
    pieces: list[MoldPiece]
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    layout: PieceLayout | None = None
    """Side pieces and locked-area report when ``config.pieces == "auto"``."""
    surface: PartingSurface | None = None
    """The curved parting surface between the halves, or None for the plane z == 0."""

    def save(self, out_dir: str | Path) -> list[Path]:
        from moldgen.report import save_result

        return save_result(self, out_dir)


class _Stages:
    """Times pipeline stages and forwards progress to an optional callback."""

    def __init__(self, callback: ProgressCallback | None, total: int) -> None:
        self.callback = callback
        self.total = total
        self.done = 0
        self.timings: dict[str, float] = {}
        self._current: str | None = None
        self._started = 0.0

    def start(self, name: str) -> None:
        self._finish_current()
        self._current = name
        self._started = time.perf_counter()
        log.info("%s", name)
        if self.callback:
            self.callback(name, self.done / self.total)

    def finish(self) -> None:
        self._finish_current()
        if self.callback:
            self.callback("Done", 1.0)

    def _finish_current(self) -> None:
        if self._current is not None:
            self.timings[self._current] = time.perf_counter() - self._started
            self.done += 1
            self._current = None


def prepare_part(
    source: str | Path | trimesh.Trimesh,
    config: MoldConfig | None = None,
    *,
    name: str | None = None,
) -> PreparedPart:
    """Load (if needed), convert units and repair the part."""
    config = config or MoldConfig()
    config.validate()
    if isinstance(source, trimesh.Trimesh):
        from moldgen.config import UNIT_TO_MM

        mesh = source.copy()
        mesh.apply_scale(UNIT_TO_MM[config.units] * config.scale)
        part_name = name or "part"
    else:
        mesh = load_mesh(source, units=config.units, scale=config.scale)
        part_name = name or Path(source).stem

    warnings: list[str] = []
    if (hint := units_warning(mesh)) is not None:
        warnings.append(hint)

    if config.repair:
        mesh, report = repair_mesh(mesh)
    else:
        report = _check_without_repair(mesh)
    if not report.manifold_ok:
        reason = report.warnings[-1] if report.warnings else "it is not a closed solid."
        raise MoldError(f"Cannot build a mold from {part_name}. {reason.rstrip('.')}.")
    warnings.extend(report.warnings)
    return PreparedPart(name=part_name, mesh=mesh, repair=report, warnings=warnings)


def _check_without_repair(mesh: trimesh.Trimesh) -> RepairReport:
    report = RepairReport(was_watertight=mesh.is_watertight, is_watertight=mesh.is_watertight)
    try:
        solid = booleans.to_manifold(mesh, "the part")
    except booleans.BooleanError as exc:
        report.warnings.append(str(exc))
        return report
    report.volume = float(solid.volume())
    report.manifold_ok = report.volume > 0
    if not report.manifold_ok:
        report.warnings.append("The part encloses no volume.")
    return report


def auto_wall_thickness(
    cavity: trimesh.Trimesh, material: Material, key_clearance: float | None = None
) -> float:
    """Wall thickness that scales with the part but respects the material's minimum.

    With ``key_clearance`` the wall is also wide enough for registration keys
    beside a part that fills the whole parting face, such as a box.
    """
    typical = float(np.median(cavity.extents))
    wall = float(np.clip(WALL_FRACTION_OF_EXTENT * typical, material.min_wall_mm, MAX_AUTO_WALL_MM))
    if key_clearance is not None:
        smallest_key = MIN_KEY_RADIUS_MM + _key_overhead(key_clearance) + MIN_KEY_MARGIN_MM
        wall = max(wall, 2.0 * smallest_key)
    return wall


def auto_key_size(wall: float, clearance: float) -> tuple[float, float]:
    """Return the key radius and the key margin to the cavity for a given wall.

    Both grow with the wall, but are kept small enough that a key fits in the
    rim of width ``wall`` around the cavity, when the wall allows it.
    """
    radius = float(
        np.clip(KEY_RADIUS_FRACTION_OF_WALL * wall, MIN_KEY_RADIUS_MM, MAX_KEY_RADIUS_MM)
    )
    margin = float(
        np.clip(KEY_MARGIN_FRACTION_OF_WALL * wall, MIN_KEY_MARGIN_MM, MAX_KEY_MARGIN_MM)
    )
    # A key at the middle of the rim needs footprint + margin on both sides.
    budget = wall / 2.0 - _key_overhead(clearance)
    margin = float(np.clip(budget - radius, MIN_KEY_MARGIN_MM, margin))
    radius = float(np.clip(budget - margin, MIN_KEY_RADIUS_MM, radius))
    return radius, margin


def _key_overhead(clearance: float) -> float:
    """How far a key's socket footprint reaches beyond the key radius."""
    plan = KeyPlan(np.zeros(3), np.empty((0, 3)), radius=1.0, clearance=clearance)
    return plan.footprint_radius - 1.0


def generate_mold(
    source: str | Path | trimesh.Trimesh | PreparedPart,
    config: MoldConfig | None = None,
    *,
    parting: PartingResult | None = None,
    progress: ProgressCallback | None = None,
) -> MoldResult:
    """Build all mold pieces for ``source``.

    ``parting`` lets interactive callers reuse an analysis they already ran
    (it must have been computed on the prepared part's mesh).
    """
    config = config or MoldConfig()
    config.validate()
    material = get_material(config.material)
    print_material = get_print_material(config.print_material)
    stages = _Stages(progress, total=7 if config.pieces == 2 else 8)

    stages.start("Loading and repairing the part")
    part = source if isinstance(source, PreparedPart) else prepare_part(source, config)
    warnings = list(part.warnings)
    warnings.extend(compatibility_warnings(material, print_material))

    stages.start("Choosing the parting plane")
    if parting is None:
        parting = analyze_parting(
            part.mesh,
            config.direction_vector(),
            config.parting_offset,
            draft_threshold_deg=config.draft_threshold_deg,
        )
    if config.pieces == "auto" and config.direction == "auto" and config.parting_offset is None:
        main = choose_main_direction(part.mesh, parting, config.max_pieces)
        if main is not None and not np.allclose(main.direction, parting.direction):
            parting = analyze_parting(
                part.mesh,
                main.direction,
                main.offset,
                draft_threshold_deg=config.draft_threshold_deg,
            )
    if config.pieces == 4 and parting.undercut_fraction > UNDERCUT_WARNING_FRACTION:
        warnings.append(
            f"{parting.undercut_fraction:.1%} of the surface is undercut for this parting plane; "
            "the cast may lock in the mold. Try another direction, more pieces or a flexible "
            "casting material."
        )

    stages.start("Building the mold block")
    shrinkage = material.linear_shrinkage if config.shrinkage is None else config.shrinkage
    shrink_scale = 1.0 / (1.0 - shrinkage)
    cavity = part.mesh.copy().apply_transform(parting.to_mold)
    centre = cavity.bounds.mean(axis=0)
    pivot = np.array([centre[0], centre[1], 0.0])
    cavity.apply_transform(trimesh.transformations.scale_matrix(shrink_scale, origin=pivot))
    key_clearance = config.clearance if config.keys > 0 else None
    wall = config.wall_thickness or auto_wall_thickness(cavity, material, key_clearance)
    block_bounds = np.array([cavity.bounds[0] - wall, cavity.bounds[1] + wall])
    block = trimesh.creation.box(bounds=block_bounds)

    stages.start("Placing the sprue and vents")

    def place_gating(up: np.ndarray | None = None) -> GatingPlan:
        return plan_gating(
            cavity,
            block_bounds,
            sprue_diameter=config.sprue_diameter or material.sprue_diameter_mm,
            vent_diameter=config.vent_diameter or material.vent_diameter_mm,
            funnel=config.funnel,
            vents=config.vents,
            up=up,
        )

    try:
        gating = place_gating()
    except ValueError as exc:
        raise MoldError(f"Cannot place the sprue: {str(exc).rstrip('.')}.") from exc

    layout = None
    surface = None
    if config.pieces == "auto":
        stages.start("Planning the mold pieces")
        gating = _gating_for_pieces(cavity, gating, place_gating, config.max_pieces)
        surface = _curved_surface(cavity, gating, block_bounds, config)
        layout, cavity = plan_layout(
            cavity, gating.solids(), block_bounds, config.max_pieces, surface=surface
        )
        warnings.extend(_layout_warnings(layout))
    elif config.pieces == 2:
        surface = _curved_surface(cavity, gating, block_bounds, config)
        undercut = parting.undercut_fraction
        if surface is not None:
            cast = CastFaces(cavity, gating.solids(), surface)
            locked = locked_faces(cast, [])
            layout = PieceLayout(
                face_region=face_regions(cast, [], locked), locked_fraction=cast.locked_area(locked)
            )
            undercut = layout.locked_fraction
        if undercut > UNDERCUT_WARNING_FRACTION:
            warnings.append(
                f"{undercut:.1%} of the surface is undercut for this parting; the cast may lock "
                "in the mold. Try --pieces auto, another direction or a flexible casting material."
            )
    if gating.unvented:
        warnings.append(
            f"{gating.unvented} air pockets have no vent and may leave bubbles in the cast"
        )
    gating_solids = gating.solids()

    stages.start("Cutting the cavity and splitting the mold")
    body = booleans.difference(block, [cavity, *gating_solids])
    if _enclosed_voids(body):
        warnings.append(
            "Part of the cavity is not reachable from the sprue (for example a separate body), "
            "so it will not fill when pouring."
        )
    stages.start("Adding registration keys")
    key_radius, key_margin = auto_key_size(wall, config.clearance)
    if config.key_diameter:
        key_radius = config.key_diameter / 2
    obstacles = [cavity, *gating_solids]
    if (layout is not None and layout.caps) or surface is not None:
        pieces, key_plans = _layout_pieces(
            body,
            layout or PieceLayout(),
            obstacles,
            block_bounds,
            config,
            key_radius,
            key_margin,
            warnings,
            surface,
        )
    else:
        pieces, key_plans = _halves(
            body, gating, obstacles, block_bounds, config, key_radius, key_margin, warnings, stages
        )

    stages.start("Checking the result")
    for piece in pieces:
        if piece.mesh.is_empty or not piece.mesh.is_watertight:
            raise MoldError(f"Mold piece {piece.name!r} came out broken (not a closed solid).")

    stages.finish()
    return MoldResult(
        config=config,
        material=material,
        print_material=print_material,
        part=part,
        parting=parting,
        cavity=cavity,
        shrink_scale=shrink_scale,
        wall_thickness=wall,
        block_bounds=block_bounds,
        gating=gating,
        keys=key_plans,
        pieces=pieces,
        warnings=warnings,
        timings=stages.timings,
        layout=layout,
        surface=surface,
    )


def _halves(
    body: trimesh.Trimesh,
    gating: GatingPlan,
    obstacles: list[trimesh.Trimesh],
    block_bounds: np.ndarray,
    config: MoldConfig,
    key_radius: float,
    key_margin: float,
    warnings: list[str],
    stages: _Stages,
) -> tuple[list[MoldPiece], list[KeyPlan]]:
    """Split ``body`` into the classic two halves, or four pieces with ``config.pieces == 4``."""
    top, bottom = booleans.split_by_plane(body, np.array([0.0, 0.0, 1.0]), 0.0)

    secondary_axis = _secondary_axis(gating) if config.pieces == 4 else None
    secondary_offset = (
        float(block_bounds.mean(axis=0)[secondary_axis]) if secondary_axis is not None else 0.0
    )
    main_obstacles = list(obstacles)
    if secondary_axis is not None:
        main_obstacles.append(
            _seam_slab(block_bounds, secondary_axis, secondary_offset, key_radius)
        )

    key_plans: list[KeyPlan] = []
    if config.keys > 0:
        main_face = np.array([[*block_bounds[0][:2], 0.0], [*block_bounds[1][:2], 0.0]])
        main_keys = plan_keys(
            main_obstacles,
            main_face,
            np.array([0.0, 0.0, 1.0]),
            count=config.keys,
            radius=key_radius,
            clearance=config.clearance,
            margin=key_margin,
        )
        _check_key_count(main_keys, config.keys, "the parting face", warnings)
        bottom = booleans.union([bottom, *main_keys.male_solids()])
        top = booleans.difference(top, main_keys.female_solids())
        key_plans.append(main_keys)

    if secondary_axis is None:
        pieces = [
            MoldPiece("top", top, _print_transform(top, flip=True), pull=UP),
            MoldPiece("bottom", bottom, _print_transform(bottom, flip=False), pull=-UP),
        ]
    else:
        stages.start("Splitting into four pieces")
        normal = np.zeros(3)
        normal[secondary_axis] = 1.0
        pieces = []
        for half_name, half, flip in (("top", top, True), ("bottom", bottom, False)):
            plus, minus = booleans.split_by_plane(half, normal, secondary_offset)
            if config.keys > 0:
                seam = _seam_face(block_bounds, secondary_axis, secondary_offset, upper=flip)
                seam_keys = plan_keys(
                    obstacles,
                    seam,
                    normal,
                    count=SECONDARY_KEYS_PER_SEAM,
                    radius=key_radius,
                    clearance=config.clearance,
                    margin=key_margin,
                )
                _check_key_count(
                    seam_keys, SECONDARY_KEYS_PER_SEAM, f"the {half_name} seam", warnings
                )
                minus = booleans.union([minus, *seam_keys.male_solids()])
                plus = booleans.difference(plus, seam_keys.female_solids())
                key_plans.append(seam_keys)
            axis_name = "xy"[secondary_axis]
            pull = UP if flip else -UP
            for sign, piece in (("-", minus), ("+", plus)):
                pieces.append(
                    MoldPiece(
                        f"{half_name}_{axis_name}{sign}",
                        piece,
                        _print_transform(piece, flip),
                        pull=pull,
                    )
                )

    return pieces, key_plans


def _layout_pieces(
    body: trimesh.Trimesh,
    layout: PieceLayout,
    obstacles: list[trimesh.Trimesh],
    block_bounds: np.ndarray,
    config: MoldConfig,
    key_radius: float,
    key_margin: float,
    warnings: list[str],
    surface: PartingSurface | None = None,
) -> tuple[list[MoldPiece], list[KeyPlan]]:
    """Cut ``body`` into the side pieces of ``layout`` and the two halves, in removal order.

    Each side piece carries male keys on its cut face pointing back along its
    pull, and the pieces behind it get the sockets, so it slides off its keys
    and leaves nothing sticking out of the pieces that are still in place.
    The pieces stay manifold3d solids until the end: converting to a mesh and
    back can fuse sheets that touch along an edge.
    """
    caps = layout.caps
    rest = booleans.to_manifold(body, "the mold body")
    remaining = block_solid(block_bounds)
    inset = SLICE_INSET * float(np.linalg.norm(block_bounds[1] - block_bounds[0]))
    solids: list[tuple[str, manifold3d.Manifold, np.ndarray]] = []
    key_plans: list[KeyPlan] = []
    for k, cap in enumerate(caps):
        piece = cap_region(cap, rest)
        rest = rest - piece
        if config.keys > 0:
            rows = plane_basis(-cap.direction)
            behind = trim(remaining, [(-cap.direction, -cap.offset), *cap.halfspaces()[1:]])
            face = _section(cap_region(cap, remaining), rows, -cap.offset - inset).intersection(
                _section(behind, rows, -cap.offset + inset)
            )
            if face.area == 0:
                # The cut plane lies in space earlier pieces took: nothing behind to key to.
                remaining = remaining - cap_region(cap, remaining)
                solids.append((f"side_{k + 1}", piece, cap.direction))
                continue
            # Only the side piece has bumps; the pieces behind just get sockets, so a
            # socket may cross a later seam and needs no clearance from it.
            plan = plan_keys_on_plane(
                obstacles,
                -cap.direction,
                -cap.offset,
                face,
                count=SIDE_PIECE_KEYS,
                radius=key_radius,
                clearance=config.clearance,
                margin=key_margin,
            )
            if not cap.bounds:
                # A boxed-in side piece sits in a pocket of the halves, which locates it.
                _check_key_count(plan, SIDE_PIECE_KEYS, f"side piece {k + 1}", warnings)
            if len(plan.positions):
                piece = piece + _joined(plan.male_solids())
                rest = rest - _joined(plan.female_solids())
                key_plans.append(plan)
        remaining = remaining - cap_region(cap, remaining)
        solids.append((f"side_{k + 1}", piece, cap.direction))

    if surface is None:
        top = rest.trim_by_plane((0.0, 0.0, 1.0), 0.0)
        bottom = rest.trim_by_plane((0.0, 0.0, -1.0), 0.0)
    else:
        below = surface.below(float(block_bounds[0][2]) - surface.cell)
        top, bottom = rest - below, rest ^ below
    if not layout.top_needed:
        # No cast face touches the leftover top block, and the bottom comes off
        # last, so the two can be one piece.
        top, bottom = manifold3d.Manifold(), bottom + top
    elif config.keys > 0 and surface is not None and not (top.is_empty() or bottom.is_empty()):
        plan = _surface_keys(surface, caps, block_bounds, config, key_radius, key_margin)
        _check_key_count(plan, min(config.keys, 2), "the parting surface", warnings)
        if len(plan.positions):
            bottom = bottom + _joined(plan.male_solids())
            top = top - _joined(plan.female_solids())
            key_plans.append(plan)
    elif config.keys > 0 and not (top.is_empty() or bottom.is_empty()):
        rows = plane_basis(UP)
        face = _section(core_half(block_bounds, caps, 1), rows, inset).intersection(
            _section(core_half(block_bounds, caps, -1), rows, -inset)
        )
        # The bumps stand up into the top half; keep them clear of the side pieces,
        # which slide in over them during assembly.
        seams = [_slab(block_bounds, n, c) for cap in caps for n, c in cap.planes()]
        plan = plan_keys_on_plane(
            obstacles + seams,
            UP,
            0.0,
            face,
            count=config.keys,
            radius=key_radius,
            clearance=config.clearance,
            margin=key_margin,
        )
        # Side pieces leave less of the parting face; two keys are enough to align the halves.
        _check_key_count(plan, min(config.keys, 2), "the parting face", warnings)
        if len(plan.positions):
            bottom = bottom + _joined(plan.male_solids())
            top = top - _joined(plan.female_solids())
            key_plans.append(plan)
    solids += [("top", top, UP), ("bottom", bottom, -UP)]
    block_volume = float(np.prod(block_bounds[1] - block_bounds[0]))
    solids = [item for item in solids if item[1].volume() > SLIVER_VOLUME_FRACTION * block_volume]
    warnings.extend(_removal_warnings(solids, obstacles, block_bounds))

    pieces = []
    for name, solid, pull in solids:
        mesh = booleans.to_trimesh(solid)
        pieces.append(MoldPiece(name, mesh, _print_transform_along(mesh, pull), pull=pull))
    return pieces, key_plans


def _removal_warnings(
    solids: list[tuple[str, manifold3d.Manifold, np.ndarray]],
    cast: list[trimesh.Trimesh],
    block_bounds: np.ndarray,
) -> list[str]:
    """Slide each piece out in removal order and report any that catch on something.

    A safety net for the planner: the cast and the pieces still in place
    must not overlap a piece at any point along its pull.
    """
    cast_solid = _joined(cast)
    size = float(np.linalg.norm(block_bounds[1] - block_bounds[0]))
    warnings = []
    for i, (name, solid, pull) in enumerate(solids):
        others = manifold3d.Manifold.batch_boolean(
            [cast_solid, *(s for _, s, _ in solids[i + 1 :])], manifold3d.OpType.Add
        )
        allowed = REMOVAL_OVERLAP_FRACTION * solid.volume()
        for distance in REMOVAL_STEPS_FRACTION:
            moved = solid.translate(tuple(np.asarray(pull) * distance * size))
            if (moved ^ others).volume() > allowed:
                warnings.append(
                    f"The {name.replace('_', ' ')} piece may catch on the cast or on another "
                    "piece when it is removed; check it before printing."
                )
                break
    return warnings


def _joined(meshes: list[trimesh.Trimesh]) -> manifold3d.Manifold:
    return manifold3d.Manifold.batch_boolean(
        [booleans.to_manifold(mesh, "a key") for mesh in meshes], manifold3d.OpType.Add
    )


def _gating_for_pieces(
    cavity: trimesh.Trimesh,
    gating: GatingPlan,
    place_gating: Callable[[np.ndarray], GatingPlan],
    max_pieces: int,
) -> GatingPlan:
    """Pick the pour side whose sprue and vents leave the side pieces the least locked area.

    The sprue lies in the parting plane, so a side piece that crosses it must
    also slide off the sprue stub. The default pour side is kept unless
    another one lets the pieces release more of the part.
    """
    best_key, best = None, gating
    options: list[GatingPlan | np.ndarray] = [gating]
    options += [up for up in POUR_DIRECTIONS if not np.allclose(up, gating.up)]
    for rank, option in enumerate(options):
        if not isinstance(option, GatingPlan):
            try:
                option = place_gating(option)
            except ValueError:
                continue
        cast = CastFaces(cavity, option.solids())
        caps = plan_caps(cast, max_pieces - 2, directions=QUICK_DIRECTIONS)
        locked = cast.locked_area(locked_faces(cast, caps))
        key = (round(locked, 3), len(caps), rank)
        if best_key is None or key < best_key:
            best_key, best = key, option
        if locked == 0.0:
            break
    return best


def _curved_surface(
    cavity: trimesh.Trimesh, gating: GatingPlan, block_bounds: np.ndarray, config: MoldConfig
) -> PartingSurface | None:
    """A curved parting surface if it releases more of the part than the plane z == 0.

    With side pieces, the comparison counts the locked area left after a quick
    plan of the side pieces, then the number of pieces: a curved surface rules
    out side pieces limited to one half, so it is not better by default.
    """
    if config.parting_surface != "auto":
        return None
    solids = gating.solids()
    candidate = fit_surface(trimesh.util.concatenate([cavity, *solids]), block_bounds, gating)
    if candidate.flat:
        return None
    scores = []
    for surface in (None, candidate):
        cast = CastFaces(cavity, solids, surface)
        caps = []
        if config.pieces == "auto":
            caps = plan_caps(cast, config.max_pieces - 2, directions=QUICK_DIRECTIONS)
        scores.append((round(cast.locked_area(locked_faces(cast, caps)), 3), len(caps)))
    return candidate if scores[1] < scores[0] else None


def _surface_keys(
    surface: PartingSurface,
    caps: list[Cap],
    block_bounds: np.ndarray,
    config: MoldConfig,
    key_radius: float,
    key_margin: float,
) -> KeyPlan:
    """Keys on the gently sloping parts of a curved parting surface, away from the cast.

    Each key stands at the surface's height where it is placed; the slope
    limit keeps the key's base inside the bottom piece all round.
    """
    usable = (surface.slope() <= KEY_MAX_SLOPE) & ~surface.covered
    points = np.column_stack([surface.nodes(), surface.heights.ravel()])
    for cap in caps:
        reach = np.ones(len(points), dtype=bool)
        for normal, value in cap.halfspaces():
            reach &= points @ normal >= value
        usable &= ~reach.reshape(usable.shape)
    cells = usable[:-1, :-1] & usable[1:, :-1] & usable[:-1, 1:] & usable[1:, 1:]
    x0, y0 = surface.origin
    boxes = []
    for i, row in enumerate(cells):
        # One box per run of usable cells along the row keeps the union small.
        edges = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(np.int8), [0]])))
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            boxes.append(
                shapely.box(
                    x0 + i * surface.cell,
                    y0 + start * surface.cell,
                    x0 + (i + 1) * surface.cell,
                    y0 + end * surface.cell,
                )
            )
    lo, hi = np.asarray(block_bounds, dtype=float)
    region = shapely.union_all(boxes).intersection(shapely.box(lo[0], lo[1], hi[0], hi[1]))
    rows = plane_basis(UP)
    region = shapely.affinity.affine_transform(
        region, [rows[0, 0], rows[0, 1], rows[1, 0], rows[1, 1], 0.0, 0.0]
    )
    plan = plan_keys_on_plane(
        [],
        UP,
        0.0,
        region,
        count=config.keys,
        radius=key_radius,
        clearance=config.clearance,
        margin=key_margin,
    )
    if len(plan.positions):
        plan.positions[:, 2] = surface.height(plan.positions[:, :2])
    return plan


def _layout_warnings(layout: PieceLayout) -> list[str]:
    warnings = []
    if layout.filled_volume >= NOTICEABLE_FILL_MM3:
        warnings.append(
            f"{layout.locked_fraction:.1%} of the surface cannot be released by any piece; the "
            f"cavity was filled there, adding {layout.filled_volume / 1000.0:.2f} cm³ to the cast."
        )
    return warnings


def _section(solid: manifold3d.Manifold, rows: np.ndarray, height: float) -> shapely.Geometry:
    """Cross-section of ``solid`` at ``dot(p, rows[2]) == height``, in ``rows`` coordinates."""
    local = solid.transform(np.column_stack([rows, np.zeros(3)]))
    return cross_section_to_shapely(local.slice(height))


def _slab(block_bounds: np.ndarray, normal: np.ndarray, offset: float) -> trimesh.Trimesh:
    """A thin slab of the block around the plane ``dot(p, normal) == offset``."""
    normal = np.asarray(normal, dtype=float)
    slab = (
        block_solid(block_bounds)
        .trim_by_plane(tuple(normal), offset - SEAM_SLAB_HALF_MM)
        .trim_by_plane(tuple(-normal), -(offset + SEAM_SLAB_HALF_MM))
    )
    return booleans.to_trimesh(slab)


def _print_transform_along(mesh: trimesh.Trimesh, pull: np.ndarray) -> np.ndarray:
    """Turn the side facing against ``pull`` up, then rest the piece on z == 0."""
    transform = trimesh.geometry.align_vectors(-np.asarray(pull, dtype=float), [0.0, 0.0, 1.0])
    vertices = trimesh.transform_points(mesh.vertices, transform)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return trimesh.transformations.translation_matrix(shift) @ transform


def _enclosed_voids(body: trimesh.Trimesh) -> bool:
    """True if the mold body has internal surfaces, i.e. a void not open to the outside.

    Shells without volume (slivers where a channel just touches a block face)
    do not count.
    """
    shells = body.split(only_watertight=False)
    with np.errstate(divide="ignore", invalid="ignore"):  # trimesh divides by a zero volume
        volumes = [abs(shell.volume) for shell in shells]
    return sum(volume > SLIVER_VOLUME_FRACTION * body.volume for volume in volumes) > 1


def _secondary_axis(gating: GatingPlan) -> int:
    """Axis of the second cut for four-piece molds.

    The cut runs parallel to the pour direction, so the extra seam is vertical
    while pouring and every piece keeps part of the sprue opening.
    """
    up_axis = int(np.argmax(np.abs(gating.up[:2])))
    return 1 - up_axis


def _seam_slab(
    block_bounds: np.ndarray, axis: int, offset: float, half_width: float
) -> trimesh.Trimesh:
    """Thin box around the secondary seam that main keys must avoid."""
    lo, hi = block_bounds.copy()
    lo[axis], hi[axis] = offset - half_width, offset + half_width
    return trimesh.creation.box(bounds=np.array([lo, hi]))


def _seam_face(block_bounds: np.ndarray, axis: int, offset: float, *, upper: bool) -> np.ndarray:
    """Face bounds of the secondary seam within the top (z > 0) or bottom half."""
    lo, hi = block_bounds.copy()
    lo[axis] = hi[axis] = offset
    if upper:
        lo[2] = 0.0
    else:
        hi[2] = 0.0
    return np.array([lo, hi])


def _check_key_count(plan: KeyPlan, requested: int, where: str, warnings: list[str]) -> None:
    placed = len(plan.positions)
    if placed < requested:
        warnings.append(
            f"Only {placed} of {requested} registration keys fit on {where}; "
            "increase the wall thickness or reduce the key diameter."
        )


def _print_transform(mesh: trimesh.Trimesh, flip: bool) -> np.ndarray:
    """Rotate so the parting face (z == 0) points up, then rest the piece on z == 0."""
    transform = np.eye(4)
    if flip:
        transform = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    bounds = trimesh.transform_points(mesh.bounds, transform)
    lo, hi = bounds.min(axis=0), bounds.max(axis=0)
    shift = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return trimesh.transformations.translation_matrix(shift) @ transform
