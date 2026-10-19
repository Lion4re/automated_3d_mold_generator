import logging

import numpy as np
import pytest
import trimesh

from moldgen.booleans import to_manifold
from moldgen.gating import (
    FUNNEL_MIN_WALL,
    FUNNEL_MOUTH_RATIO,
    FUNNEL_THROAT_EXTENSION,
    MAX_OVERLAP_FRACTION,
    MAX_VENTS,
    MIN_SPRUE_RADIUS,
    GatingPlan,
    plan_gating,
)

SPRUE_DIAMETER = 6.0
VENT_DIAMETER = 1.5


def _block_bounds(part: trimesh.Trimesh, wall: float = 8.0) -> np.ndarray:
    return part.bounds + np.array([[-wall], [wall]])


def _plan(part: trimesh.Trimesh, wall: float = 8.0, **kwargs) -> GatingPlan:
    return plan_gating(
        part,
        _block_bounds(part, wall),
        sprue_diameter=SPRUE_DIAMETER,
        vent_diameter=VENT_DIAMETER,
        **kwargs,
    )


def _union(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.boolean.union(meshes, engine="manifold")


@pytest.fixture
def lying_pawn(pawn: trimesh.Trimesh) -> trimesh.Trimesh:
    """The pawn lying along X with its base towards -X, centred on the origin."""
    part = pawn.copy()
    part.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    part.apply_translation(-part.bounds.mean(axis=0))
    return part


@pytest.fixture
def lying_cylinder() -> trimesh.Trimesh:
    """Cylinder with its axis along X, centred on the origin."""
    mesh = trimesh.creation.cylinder(radius=12.0, height=50.0, sections=64)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    return mesh


@pytest.fixture
def u_shape() -> trimesh.Trimesh:
    """A U in the XY plane opening towards +Y, 10 mm thick, centred on z == 0."""
    bar = trimesh.creation.box([40.0, 10.0, 10.0])
    bar.apply_translation([0.0, -15.0, 0.0])
    arms = []
    for x in (-15.0, 15.0):
        arm = trimesh.creation.box([10.0, 40.0, 10.0])
        arm.apply_translation([x, 0.0, 0.0])
        arms.append(arm)
    return _union([bar, *arms])


def _dumbbell(lower_sphere_z: float = 0.0) -> trimesh.Trimesh:
    """Two balls joined by a rod along the XY diagonal.

    Whichever of +X, -X, +Y, -Y points up, one ball sits lower than the other
    and traps air under its top.
    """
    first = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    first.apply_translation([-20.0, -20.0, lower_sphere_z])
    second = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
    second.apply_translation([20.0, 20.0, 0.0])
    rod = trimesh.creation.cylinder(radius=3.0, segment=[[-20.0, -20.0, 0.0], [20.0, 20.0, 0.0]])
    return _union([first, second, rod])


def _comb(teeth: int, spiked: bool = False) -> trimesh.Trimesh:
    """A wide sprue tooth at x == 0 and ``teeth`` narrow teeth along +Y, each lower than the last.

    With ``spiked`` the last tooth ends in two 1 mm spikes 3 mm apart instead of a flat top.
    """
    centers = 14.0 + 8.0 * np.arange(teeth)
    bar = trimesh.creation.box(bounds=[[-6.0, -6.0, -5.0], [centers[-1] + 2.5, 0.0, 5.0]])
    parts = [bar, trimesh.creation.box(bounds=[[-6.0, 0.0, -5.0], [6.0, 30.0, 5.0]])]
    for i, x in enumerate(centers):
        height = 20.0 - 0.5 * i
        parts.append(trimesh.creation.box(bounds=[[x - 2.5, 0.0, -5.0], [x + 2.5, height, 5.0]]))
    if spiked:
        for dx in (-1.5, 1.5):
            spike = trimesh.creation.box([1.5, 2.0, 10.0])
            spike.apply_translation([centers[-1] + dx, height, 0.0])
            parts.append(spike)
    return _union(parts)


def _part_named(name: str, request: pytest.FixtureRequest) -> trimesh.Trimesh:
    if name == "dumbbell":
        return _dumbbell()
    return request.getfixturevalue(name)


def test_pawn_is_poured_base_up(lying_pawn: trimesh.Trimesh) -> None:
    x, radial = lying_pawn.vertices[:, 0], np.hypot(*lying_pawn.vertices[:, 1:].T)
    assert radial[x < -20].max() > radial[x > 20].max(), "the base should lie towards -X"

    plan = _plan(lying_pawn)

    np.testing.assert_allclose(plan.up, [-1.0, 0.0, 0.0])
    assert plan.scores[0].score > plan.scores[1].score
    assert plan.vents == []


@pytest.mark.parametrize("name", ["lying_pawn", "lying_cylinder", "sphere", "u_shape", "dumbbell"])
def test_sprue_connects_cavity_to_outside(name: str, request: pytest.FixtureRequest) -> None:
    part = _part_named(name, request)
    bounds = _block_bounds(part)
    plan = _plan(part)
    sprue = plan.sprue

    assert sprue.start[2] == 0.0 and sprue.end[2] == 0.0
    direction = (sprue.start - sprue.end) / np.linalg.norm(sprue.start - sprue.end)
    np.testing.assert_allclose(direction, plan.up, atol=1e-12)
    assert sprue.start @ plan.up > np.max(bounds @ plan.up)

    block = trimesh.creation.box(bounds=bounds)
    closed = trimesh.boolean.difference([block, part], engine="manifold")
    assert closed.body_count == 2, "without gating the cavity is an enclosed void"
    mold = trimesh.boolean.difference([block, part, *plan.solids()], engine="manifold")
    assert mold.body_count == 1, "the cavity must open to the outside"

    sprue_solid = sprue.solid()
    overlap = trimesh.boolean.intersection([sprue_solid, part], engine="manifold")
    assert overlap.volume > 1.0
    assert _union([part, sprue_solid]).body_count == 1


@pytest.mark.parametrize("name", ["lying_pawn", "lying_cylinder", "sphere", "u_shape", "dumbbell"])
def test_gating_solids_are_watertight(name: str, request: pytest.FixtureRequest) -> None:
    plan = _plan(_part_named(name, request))
    solids = plan.solids()
    assert len(solids) == 1 + (plan.funnel is not None) + len(plan.vents)
    for solid in solids:
        assert solid.is_watertight
        assert solid.is_winding_consistent
        assert solid.volume > 0


@pytest.mark.parametrize("wall", [8.0, 3.0])
def test_funnel_never_breaks_into_cavity(lying_pawn: trimesh.Trimesh, wall: float) -> None:
    bounds = _block_bounds(lying_pawn, wall)
    plan = _plan(lying_pawn, wall)
    funnel = plan.funnel
    assert funnel is not None
    assert funnel.mouth_radius > plan.sprue.radius
    assert funnel.center @ plan.up == pytest.approx(np.max(bounds @ plan.up))

    gap = to_manifold(funnel.solid()).min_gap(to_manifold(lying_pawn), 10.0)
    assert gap >= FUNNEL_MIN_WALL - 1e-6

    # Each half of the block carries half of the funnel and it stays inside the face.
    assert funnel.center[2] == 0.0
    side = np.abs(np.cross([0.0, 0.0, 1.0], plan.up))
    assert (
        bounds[0] @ side + funnel.mouth_radius
        < funnel.center @ side
        < bounds[1] @ side - funnel.mouth_radius
    )
    assert funnel.mouth_radius < bounds[1, 2]


def test_funnel_skipped_when_wall_is_too_thin(lying_pawn: trimesh.Trimesh) -> None:
    plan = _plan(lying_pawn, wall=2.0)
    assert plan.funnel is None
    assert len(plan.solids()) == 1 + len(plan.vents)


def test_funnel_disabled(sphere: trimesh.Trimesh) -> None:
    assert _plan(sphere, funnel=False).funnel is None


def test_funnel_narrowed_by_block_edge() -> None:
    """The sprue enters an L's upright, 7 mm from the block's +X face."""
    bar = trimesh.creation.box(bounds=[[-20.0, -10.0, -5.0], [20.0, 0.0, 5.0]])
    upright = trimesh.creation.box(bounds=[[10.0, 0.0, -5.0], [20.0, 30.0, 5.0]])
    part = _union([bar, upright])
    bounds = _block_bounds(part)
    bounds[1, 0] = part.bounds[1, 0] + 2.0

    plan = plan_gating(
        part, bounds, sprue_diameter=SPRUE_DIAMETER, vent_diameter=VENT_DIAMETER, up=[0, 1, 0]
    )

    funnel = plan.funnel
    assert funnel is not None
    assert funnel.center[0] == pytest.approx(15.0)
    assert funnel.mouth_radius == pytest.approx(bounds[1, 0] - 15.0 - FUNNEL_MIN_WALL)
    assert funnel.mouth_radius < FUNNEL_MOUTH_RATIO * plan.sprue.radius


def test_funnel_made_shallow_by_cavity_below() -> None:
    part = trimesh.creation.box([40.0, 20.0, 10.0])
    bounds = _block_bounds(part)
    bounds[1, 0] = part.bounds[1, 0] + 4.0

    plan = plan_gating(
        part, bounds, sprue_diameter=SPRUE_DIAMETER, vent_diameter=VENT_DIAMETER, up=[1, 0, 0]
    )

    funnel = plan.funnel
    assert funnel is not None
    assert funnel.depth == pytest.approx(4.0 - FUNNEL_MIN_WALL - FUNNEL_THROAT_EXTENSION)
    assert funnel.mouth_radius < FUNNEL_MOUTH_RATIO * plan.sprue.radius
    gap = to_manifold(funnel.solid()).min_gap(to_manifold(part), 10.0)
    assert gap >= FUNNEL_MIN_WALL - 1e-6


def test_sphere_needs_no_vents(sphere: trimesh.Trimesh) -> None:
    plan = _plan(sphere)
    assert plan.vents == []
    assert all(score.traps == 0 for score in plan.scores)


def test_u_shape_pours_into_the_bar_without_vents(u_shape: trimesh.Trimesh) -> None:
    plan = _plan(u_shape)
    np.testing.assert_allclose(plan.up, [0.0, -1.0, 0.0])
    assert plan.vents == []


def test_u_shape_poured_open_end_up_vents_the_other_arm(u_shape: trimesh.Trimesh) -> None:
    bounds = _block_bounds(u_shape)
    plan = _plan(u_shape, up=[0.0, 1.0, 0.0])

    assert len(plan.vents) == 1
    vent = plan.vents[0]
    assert np.sign(vent.start[0]) == -np.sign(plan.sprue.start[0])
    assert vent.start[2] == 0.0 and vent.end[2] == 0.0, "flat arm tip centred on the plane"
    assert u_shape.contains([vent.start])[0]
    assert vent.end @ plan.up > np.max(bounds @ plan.up)
    np.testing.assert_allclose(
        vent.end - vent.start, (vent.end - vent.start) @ plan.up * plan.up, atol=1e-9
    )


def test_dent_in_the_underside_is_not_an_air_trap() -> None:
    """The roof of a dent faces down: it is a height maximum, but mold material fills it."""
    box = trimesh.creation.box([30.0, 20.0, 20.0])
    dent = trimesh.creation.icosphere(subdivisions=3, radius=6.0)
    dent.apply_translation([-15.0, 0.0, 0.0])
    part = trimesh.boolean.difference([box, dent], engine="manifold")

    plan = _plan(part, up=[1.0, 0.0, 0.0])

    assert plan.scores[0].traps == 0
    assert plan.vents == []


def test_vents_disabled(u_shape: trimesh.Trimesh) -> None:
    assert _plan(u_shape, up=[0.0, 1.0, 0.0], vents=False).vents == []


def test_dumbbell_vents_the_lower_ball_in_every_orientation() -> None:
    part = _dumbbell()
    plan = _plan(part)
    assert all(score.traps == 1 for score in plan.scores)
    assert len(plan.vents) == 1
    vent = plan.vents[0]
    lower_ball = np.array([-20.0, -20.0, 0.0]) if plan.up.sum() > 0 else np.array([20.0, 20.0, 0.0])
    top_of_ball = lower_ball + 10.0 * plan.up
    assert np.linalg.norm(vent.start + VENT_DIAMETER / 2 * plan.up - top_of_ball) < 0.5


@pytest.mark.parametrize(("offset", "snapped"), [(0.4, True), (4.0, False)])
def test_vents_snap_into_parting_plane_only_when_close(offset: float, snapped: bool) -> None:
    part = _dumbbell(lower_sphere_z=offset)
    plan = _plan(part, up=[1.0, 0.0, 0.0])
    assert len(plan.vents) == 1
    vent_z = plan.vents[0].start[2]
    assert vent_z == (0.0 if snapped else pytest.approx(offset, abs=0.1))
    vent_overlap = trimesh.boolean.intersection([plan.vents[0].solid(), part], engine="manifold")
    assert vent_overlap.volume > 0.1


def test_vents_capped_and_rest_reported(caplog: pytest.LogCaptureFixture) -> None:
    part = _comb(MAX_VENTS + 1)
    with caplog.at_level(logging.INFO, logger="moldgen.gating"):
        plan = _plan(part, up=[0.0, 1.0, 0.0])

    assert plan.scores[0].traps == MAX_VENTS + 1
    assert len(plan.vents) == MAX_VENTS
    assert plan.unvented == 1
    lowest_tooth = 14.0 + 8.0 * MAX_VENTS
    assert all(abs(vent.start[0] - lowest_tooth) > 2.5 for vent in plan.vents)
    assert "1 air traps have no vent" in caplog.text


def test_close_peaks_share_a_vent_and_do_not_count_as_unvented(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The spiked tooth's two peaks make MAX_VENTS + 1 traps, but need only MAX_VENTS vents."""
    part = _comb(MAX_VENTS, spiked=True)
    with caplog.at_level(logging.WARNING, logger="moldgen.gating"):
        plan = _plan(part, up=[0.0, 1.0, 0.0])

    assert plan.scores[0].traps == MAX_VENTS + 1
    assert len(plan.vents) == MAX_VENTS
    assert plan.unvented == 0
    spiked_tooth = 14.0 + 8.0 * (MAX_VENTS - 1)
    assert sum(abs(vent.start[0] - spiked_tooth) <= 2.5 for vent in plan.vents) == 1
    assert caplog.text == ""


def test_vent_on_ring_shaped_top_reaches_the_part() -> None:
    """A tube's flat top is a ring whose centre is not on the part."""
    bar = trimesh.creation.box([40.0, 10.0, 10.0])
    bar.apply_translation([0.0, -15.0, 0.0])
    arm = trimesh.creation.box([10.0, 40.0, 10.0])
    arm.apply_translation([-15.0, 0.0, 0.0])
    tube = trimesh.creation.annulus(r_min=2.0, r_max=5.0, height=40.0, sections=48)
    tube.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    tube.apply_translation([15.0, 0.0, 0.0])
    part = _union([bar, arm, tube])
    vertices = part.vertices.copy()

    plan = _plan(part, up=[0.0, 1.0, 0.0])

    np.testing.assert_array_equal(part.vertices, vertices)
    assert plan.sprue.start[0] < 0, "the solid arm offers the wider entry"
    assert len(plan.vents) == 1
    vent_overlap = trimesh.boolean.intersection([plan.vents[0].solid(), part], engine="manifold")
    assert vent_overlap.volume > 0.1


def test_parting_plane_splits_channels_into_equal_halves(sphere: trimesh.Trimesh) -> None:
    bounds = _block_bounds(sphere)
    plan = _plan(sphere)
    block = trimesh.creation.box(bounds=bounds)
    mold = to_manifold(
        trimesh.boolean.difference([block, sphere, *plan.solids()], engine="manifold")
    )

    top, bottom = mold.split_by_plane((0.0, 0.0, 1.0), 0.0)
    assert top.volume() == pytest.approx(bottom.volume(), rel=1e-6)
    assert top.volume() + bottom.volume() == pytest.approx(mold.volume(), rel=1e-9)


def test_narrow_entry_gets_thinner_sprue() -> None:
    cone = trimesh.creation.cone(radius=10.0, height=40.0, sections=64)
    cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    cone.apply_translation(-cone.bounds.mean(axis=0))

    auto = _plan(cone)
    np.testing.assert_allclose(auto.up, [-1.0, 0.0, 0.0])
    assert auto.sprue.radius == SPRUE_DIAMETER / 2

    tip = _plan(cone, up=[1.0, 0.0, 0.0])
    assert MIN_SPRUE_RADIUS <= tip.sprue.radius < SPRUE_DIAMETER / 2
    assert tip.scores[0].entry_fit < 1.0


def test_sprue_fits_between_block_faces_above_and_below() -> None:
    plate = trimesh.creation.box([60.0, 30.0, 2.0])
    bounds = plate.bounds + np.array([[-8.0, -8.0, -1.5], [8.0, 8.0, 1.5]])

    plan = plan_gating(plate, bounds, sprue_diameter=SPRUE_DIAMETER, vent_diameter=VENT_DIAMETER)

    assert plan.sprue.radius == pytest.approx(bounds[1, 2] - FUNNEL_MIN_WALL)


def test_rejects_block_too_thin_for_any_sprue() -> None:
    plate = trimesh.creation.box([60.0, 30.0, 1.0])
    bounds = plate.bounds + np.array([[-8.0, -8.0, -0.8], [8.0, 8.0, 0.8]])
    with pytest.raises(ValueError, match="too thin"):
        plan_gating(plate, bounds, sprue_diameter=SPRUE_DIAMETER, vent_diameter=VENT_DIAMETER)


def test_sprue_overlap_capped_on_thin_section() -> None:
    """A 1 mm deep section would be cut through by the full overlap."""
    slab = trimesh.creation.box([1.0, 20.0, 10.0])

    plan = _plan(slab, up=[1.0, 0.0, 0.0])

    assert plan.sprue.radius == SPRUE_DIAMETER / 2
    assert plan.sprue.end[0] == pytest.approx(0.5 - MAX_OVERLAP_FRACTION * 1.0)
    assert plan.sprue.end[0] > slab.bounds[0, 0]


def test_equally_wide_entries_prefer_the_higher() -> None:
    bar = trimesh.creation.box(bounds=[[-20.0, -6.0, -4.0], [20.0, 0.0, 4.0]])
    high = trimesh.creation.box(bounds=[[-9.0, 0.0, -4.0], [-1.0, 20.0, 4.0]])
    low = trimesh.creation.box(bounds=[[2.0, 0.0, -4.0], [10.0, 19.5, 4.0]])

    plan = _plan(_union([bar, high, low]), up=[0.0, 1.0, 0.0])

    assert plan.sprue.start[0] == pytest.approx(-5.0)


def test_rejects_part_away_from_parting_plane(sphere: trimesh.Trimesh) -> None:
    part = sphere.copy()
    part.apply_translation([0.0, 0.0, 25.0])
    with pytest.raises(ValueError, match="parting plane"):
        _plan(part)


def test_rejects_vertical_pour_direction(sphere: trimesh.Trimesh) -> None:
    with pytest.raises(ValueError, match="Pour direction"):
        _plan(sphere, up=[0.0, 0.0, 1.0])
