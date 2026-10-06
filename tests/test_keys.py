import logging

import numpy as np
import pytest
import trimesh
from scipy.spatial.distance import pdist

from moldgen.booleans import to_manifold
from moldgen.gating import Channel, plan_gating
from moldgen.keys import BOOLEAN_OVERLAP, KEY_SECTIONS, SOCKET_EXTRA_DEPTH, KeyPlan, plan_keys

RADIUS = 4.0
CLEARANCE = 0.25
MARGIN = 2.0
FACE = np.array([[-50.0, -40.0, 0.0], [50.0, 40.0, 0.0]])
UP = np.array([0.0, 0.0, 1.0])


def _keys(
    obstacles: list[trimesh.Trimesh], face: np.ndarray = FACE, normal=UP, count: int = 4
) -> KeyPlan:
    return plan_keys(
        obstacles, face, normal, count=count, radius=RADIUS, clearance=CLEARANCE, margin=MARGIN
    )


def _assert_clear(plan: KeyPlan, obstacles: list[trimesh.Trimesh], margin: float = MARGIN) -> None:
    for key in plan.male_solids() + plan.female_solids():
        key_solid = to_manifold(key)
        for obstacle in obstacles:
            assert key_solid.min_gap(to_manifold(obstacle), 2 * margin) >= margin - 1e-6


def _assert_spaced(plan: KeyPlan) -> None:
    if len(plan.positions) > 1:
        assert pdist(plan.positions).min() >= 3 * RADIUS - 1e-9


@pytest.fixture
def lying_pawn(pawn: trimesh.Trimesh) -> trimesh.Trimesh:
    part = pawn.copy()
    part.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    part.apply_translation(-part.bounds.mean(axis=0))
    return part


def _frustum_volume(bottom_radius: float, top_radius: float, height: float, sections: int) -> float:
    """Volume of a frustum with regular polygonal cross-sections inscribed in the given circles."""
    polygon = 0.5 * sections * np.sin(2 * np.pi / sections)
    bottom, top = polygon * bottom_radius**2, polygon * top_radius**2
    return height / 3 * (bottom + np.sqrt(bottom * top) + top)


def test_keys_spread_to_face_corners(sphere: trimesh.Trimesh) -> None:
    plan = _keys([sphere])

    assert plan.positions.shape == (4, 3)
    np.testing.assert_array_equal(plan.positions[:, 2], 0.0)
    inset = plan.footprint_radius + MARGIN
    assert np.all(plan.positions[:, :2] >= FACE[0, :2] + inset - 1e-9)
    assert np.all(plan.positions[:, :2] <= FACE[1, :2] - inset + 1e-9)
    quadrants = {tuple(np.sign(position[:2])) for position in plan.positions}
    assert len(quadrants) == 4
    _assert_spaced(plan)
    _assert_clear(plan, [sphere])


def test_count_is_respected(sphere: trimesh.Trimesh) -> None:
    plan = _keys([sphere], count=2)
    assert len(plan.positions) == 2
    np.testing.assert_allclose(plan.positions[0], -plan.positions[1], atol=1e-9)

    assert len(_keys([sphere], count=0).positions) == 0


def test_keys_avoid_part_and_gating_channels(lying_pawn: trimesh.Trimesh) -> None:
    block = lying_pawn.bounds + np.array([[-12.0], [12.0]])
    gating = plan_gating(lying_pawn, block, sprue_diameter=6.0, vent_diameter=1.5)
    obstacles = [lying_pawn, *gating.solids()]
    face = block.copy()
    face[:, 2] = 0.0

    plan = _keys(obstacles, face)

    assert len(plan.positions) == 4
    np.testing.assert_array_equal(plan.positions[:, 2], 0.0)
    _assert_spaced(plan)
    _assert_clear(plan, obstacles)


def test_keys_move_away_from_channel_through_a_corner(sphere: trimesh.Trimesh) -> None:
    channel = Channel(start=np.zeros(3), end=np.array([60.0, 50.0, 0.0]), radius=3.0).solid()
    free = _keys([sphere])
    corner = free.positions[np.argmax(free.positions @ [1.0, 1.0, 0.0])]

    plan = _keys([sphere, channel])

    assert len(plan.positions) == 4
    assert np.linalg.norm(plan.positions - corner, axis=1).min() > 1.0
    _assert_spaced(plan)
    _assert_clear(plan, [sphere, channel])


def test_fewer_keys_when_face_is_crowded(
    sphere: trimesh.Trimesh, caplog: pytest.LogCaptureFixture
) -> None:
    face = np.array([[-30.0, -30.0, 0.0], [30.0, 30.0, 0.0]])
    with caplog.at_level(logging.INFO, logger="moldgen.keys"):
        plan = _keys([sphere], face, count=8)

    assert 0 < len(plan.positions) < 8
    assert "registration keys fit" in caplog.text
    _assert_spaced(plan)
    _assert_clear(plan, [sphere])


def test_no_keys_when_part_covers_face(caplog: pytest.LogCaptureFixture) -> None:
    slab = trimesh.creation.box([95.0, 75.0, 10.0])
    with caplog.at_level(logging.INFO, logger="moldgen.keys"):
        plan = _keys([slab])
    assert plan.positions.shape == (0, 3)
    assert plan.male_solids() == [] and plan.female_solids() == []
    assert "registration keys fit" in caplog.text


def test_obstacles_beyond_key_reach_are_ignored() -> None:
    far = trimesh.creation.box([95.0, 75.0, 10.0])
    far.apply_translation([0.0, 0.0, 5.0 + RADIUS + MARGIN + 1.0])
    assert len(_keys([far]).positions) == 4


def test_male_key_fits_socket_with_clearance() -> None:
    plan = KeyPlan(normal=UP, positions=np.zeros((1, 3)), radius=RADIUS, clearance=CLEARANCE)
    (male,) = plan.male_solids()
    (female,) = plan.female_solids()

    assert plan.height > 0 and plan.socket_depth == pytest.approx(
        plan.height + CLEARANCE + SOCKET_EXTRA_DEPTH
    )
    assert male.bounds[1, 2] == pytest.approx(plan.height)
    assert female.bounds[1, 2] == pytest.approx(plan.socket_depth)
    assert male.bounds[0, 2] == pytest.approx(-BOOLEAN_OVERLAP)

    interference = trimesh.boolean.difference([male, female], engine="manifold")
    assert interference.is_empty or abs(interference.volume) < 1e-9

    shell = trimesh.boolean.difference([female, male], engine="manifold")
    wall_offset = CLEARANCE * np.sqrt(2.0)
    expected = _frustum_volume(
        RADIUS + BOOLEAN_OVERLAP + wall_offset,
        RADIUS + wall_offset - plan.socket_depth,
        plan.socket_depth + BOOLEAN_OVERLAP,
        KEY_SECTIONS,
    ) - _frustum_volume(
        RADIUS + BOOLEAN_OVERLAP, RADIUS - plan.height, plan.height + BOOLEAN_OVERLAP, KEY_SECTIONS
    )
    assert shell.volume == pytest.approx(expected, rel=1e-6)

    # Points on the male's sloped wall sit one clearance inside the socket wall.
    points, faces = trimesh.sample.sample_surface(male, 2000, seed=0)
    on_wall = (
        (np.abs(male.face_normals[faces] @ UP) < 0.9)
        & (points[:, 2] > 0.05)
        & (points[:, 2] < plan.height - 0.05)
    )
    gap = trimesh.proximity.signed_distance(female, points[on_wall])
    np.testing.assert_allclose(gap, CLEARANCE, atol=0.01)


def test_key_solids_are_watertight(sphere: trimesh.Trimesh) -> None:
    plan = _keys([sphere])
    for solid in plan.male_solids() + plan.female_solids():
        assert solid.is_watertight
        assert solid.is_winding_consistent
        assert solid.volume > 0


@pytest.mark.parametrize(
    ("normal", "face"),
    [
        ([-1.0, 0.0, 0.0], [[30.0, -40.0, -20.0], [30.0, 40.0, 20.0]]),
        ([0.0, 0.0, -1.0], FACE),
        ([0.0, 1.0, 0.0], [[-40.0, 22.0, -20.0], [40.0, 22.0, 20.0]]),
        ([0.0, -1.0, 0.0], [[-40.0, 22.0, -20.0], [40.0, 22.0, 20.0]]),
    ],
    ids=["-X", "-Z", "+Y", "-Y"],
)
def test_keys_follow_face_normal(sphere: trimesh.Trimesh, normal: list, face: list) -> None:
    normal, face = np.array(normal), np.array(face)
    plan = _keys([sphere], face, normal)

    assert len(plan.positions) == 4
    level = face[0] @ normal
    np.testing.assert_allclose(plan.positions @ normal, level)
    for male in plan.male_solids():
        heights = male.vertices @ normal
        assert heights.max() == pytest.approx(level + plan.height)
        assert heights.min() == pytest.approx(level - BOOLEAN_OVERLAP)
    for female in plan.female_solids():
        assert (female.vertices @ normal).max() == pytest.approx(level + plan.socket_depth)
    _assert_spaced(plan)
    _assert_clear(plan, [sphere])


def test_rejects_tilted_normal(sphere: trimesh.Trimesh) -> None:
    with pytest.raises(ValueError, match="axis-aligned"):
        _keys([sphere], normal=[0.0, 0.6, 0.8])


def test_rejects_face_not_flat_along_normal(sphere: trimesh.Trimesh) -> None:
    face = FACE.copy()
    face[1, 2] = 1.0
    with pytest.raises(ValueError, match="flat"):
        _keys([sphere], face)
