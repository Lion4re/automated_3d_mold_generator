"""End-to-end mold generation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from moldgen import booleans
from moldgen.config import MoldConfig
from moldgen.gating import GatingPlan, plan_gating
from moldgen.keys import KeyPlan, plan_keys
from moldgen.materials import (
    Material,
    PrintMaterial,
    compatibility_warnings,
    get_material,
    get_print_material,
)
from moldgen.meshio import load_mesh, units_warning
from moldgen.parting import PartingResult, analyze_parting
from moldgen.repair import RepairReport, repair_mesh

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
    if parting.undercut_fraction > UNDERCUT_WARNING_FRACTION:
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
    try:
        gating = plan_gating(
            cavity,
            block_bounds,
            sprue_diameter=config.sprue_diameter or material.sprue_diameter_mm,
            vent_diameter=config.vent_diameter or material.vent_diameter_mm,
            funnel=config.funnel,
            vents=config.vents,
        )
    except ValueError as exc:
        raise MoldError(f"Cannot place the sprue: {str(exc).rstrip('.')}.") from exc
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
    top, bottom = booleans.split_by_plane(body, np.array([0.0, 0.0, 1.0]), 0.0)

    stages.start("Adding registration keys")
    key_radius, key_margin = auto_key_size(wall, config.clearance)
    if config.key_diameter:
        key_radius = config.key_diameter / 2
    obstacles = [cavity, *gating_solids]
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
            MoldPiece("top", top, _print_transform(top, flip=True)),
            MoldPiece("bottom", bottom, _print_transform(bottom, flip=False)),
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
            pieces.append(
                MoldPiece(f"{half_name}_{axis_name}-", minus, _print_transform(minus, flip))
            )
            pieces.append(
                MoldPiece(f"{half_name}_{axis_name}+", plus, _print_transform(plus, flip))
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
    )


def _enclosed_voids(body: trimesh.Trimesh) -> bool:
    """True if the mold body has internal surfaces, i.e. a void not open to the outside."""
    components = trimesh.graph.connected_components(
        body.face_adjacency, nodes=np.arange(len(body.faces)), min_len=1
    )
    return len(components) > 1


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
