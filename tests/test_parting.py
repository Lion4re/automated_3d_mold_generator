"""Tests for parting direction search, plane placement and undercut analysis."""

import numpy as np
import pytest
import trimesh

from moldgen.parting import (
    FACE_LOW_DRAFT,
    FACE_OK,
    FACE_UNDERCUT,
    _Evaluation,
    _rank,
    analyze_parting,
    best_offset,
    classify_faces,
    mold_frame,
)

Z = np.array([0.0, 0.0, 1.0])
X = np.array([1.0, 0.0, 0.0])
NO_UNDERCUT = 1e-3


def _transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _random_directions(rng: np.random.Generator, count: int) -> np.ndarray:
    v = rng.normal(size=(count, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _side_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    """Faces of a Z-axis cylinder's mantle."""
    return np.abs(mesh.face_normals[:, 2]) < 0.5


def test_sphere_releases_in_any_direction_through_its_centre(sphere, rng):
    centre = sphere.bounds.mean(axis=0)
    for d in [*np.eye(3), *_random_directions(rng, 5)]:
        result = analyze_parting(sphere, d)
        assert result.undercut_fraction < NO_UNDERCUT
        assert result.offset == pytest.approx(centre @ d, abs=0.5)


def test_auto_direction_prefers_an_axis_among_equals(sphere):
    result = analyze_parting(sphere)
    assert np.max(np.abs(result.direction)) == pytest.approx(1.0)


def test_spool_split_across_its_axle_traps_a_disc(spool):
    assert analyze_parting(spool, Z).undercut_fraction > 0.05


def test_spool_auto_direction_splits_through_the_axle(spool):
    result = analyze_parting(spool)
    assert abs(result.direction[2]) < 0.1
    assert result.undercut_fraction < NO_UNDERCUT


def test_mushroom_plane_sits_between_cap_underside_and_top(mushroom):
    result = analyze_parting(mushroom, Z)
    assert 30.0 < result.offset < 38.0
    assert result.undercut_fraction < NO_UNDERCUT
    assert best_offset(mushroom, Z) == pytest.approx(result.offset)


def test_occlusion_flags_faces_shadowed_by_an_overhang(mushroom):
    stem = _side_faces(mushroom) & (np.linalg.norm(mushroom.triangles_center[:, :2], axis=1) < 6.0)
    normal_only = classify_faces(mushroom, Z, 5.0, occlusion=False)
    with_rays = classify_faces(mushroom, Z, 5.0)
    assert np.all(normal_only[stem] == FACE_LOW_DRAFT)
    assert np.all(with_rays[stem] == FACE_UNDERCUT)


def test_torus_lying_flat_is_split_horizontally(torus):
    result = analyze_parting(torus)
    assert abs(result.direction[2]) > 0.95
    assert result.undercut_fraction < NO_UNDERCUT


def test_torus_split_across_its_hole_has_undercut(torus):
    assert analyze_parting(torus, X).undercut_fraction > 0.05


def test_pawn_is_split_through_its_axis(pawn):
    result = analyze_parting(pawn)
    assert abs(result.direction[2]) < 0.1
    assert result.undercut_fraction < 0.01
    assert result.face_class.shape == (len(pawn.faces),)
    assert 1 < len(result.candidates) <= 8
    np.testing.assert_allclose(result.candidates[0].direction, result.direction)
    np.testing.assert_array_equal(
        classify_faces(pawn, result.direction, result.offset), result.face_class
    )


def test_cylinder_split_along_its_axis_has_no_undercut(cylinder):
    face_class = classify_faces(cylinder, X, 0.0)
    side = _side_faces(cylinder)
    assert np.all(face_class[side] != FACE_UNDERCUT)
    assert np.any(face_class[side] == FACE_OK)
    assert np.all(face_class[~side] == FACE_LOW_DRAFT)


def test_walls_parallel_to_the_pull_are_low_draft(cylinder):
    face_class = classify_faces(cylinder, Z, 20.0)
    side = _side_faces(cylinder)
    assert np.all(face_class[side] == FACE_LOW_DRAFT)
    assert np.all(face_class[~side] == FACE_OK)


def test_faces_on_the_plane_may_go_with_either_half(cylinder):
    face_class = classify_faces(cylinder, Z, 0.0)
    bottom = cylinder.face_normals[:, 2] < -0.5
    assert np.all(face_class[bottom] == FACE_OK)


def test_release_tolerance_ignores_tessellation_noise(cylinder, rng):
    vertices = cylinder.vertices.copy()
    rim = np.isclose(vertices[:, 2], 0.0) & (np.linalg.norm(vertices[:, :2], axis=1) > 1.0)
    vertices[rim, 2] += rng.uniform(-0.03, 0.03, rim.sum())
    noisy = trimesh.Trimesh(vertices, cylinder.faces, process=False)
    assert analyze_parting(noisy, X).undercut_fraction < NO_UNDERCUT
    assert analyze_parting(noisy, X, release_tolerance=0.0).undercut_fraction > 0.01


def test_explicit_direction_and_offset_are_used_as_given(cylinder):
    result = analyze_parting(cylinder, [0.0, 0.0, 2.0], 10.0)
    np.testing.assert_allclose(result.direction, Z)
    assert result.offset == 10.0
    assert len(result.candidates) == 1


def test_mold_frame_is_rigid_and_puts_the_plane_at_z0(mushroom, rng):
    offset = 12.0
    for d in [Z, -Z, X, *_random_directions(rng, 4)]:
        to_mold = mold_frame(mushroom, d, offset)
        rotation = to_mold[:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        np.testing.assert_allclose(to_mold[3], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(rotation @ d, Z, atol=1e-12)

        on_plane = offset * d + np.cross(d, rng.normal(size=3))
        assert _transform(to_mold, on_plane[None])[0, 2] == pytest.approx(0.0, abs=1e-9)
        xy = _transform(to_mold, mushroom.vertices)[:, :2]
        np.testing.assert_allclose(xy.min(axis=0) + xy.max(axis=0), 0.0, atol=1e-9)
        np.testing.assert_array_equal(mold_frame(mushroom, d, offset), to_mold)


def test_mold_frame_aligns_the_footprint_with_x_and_y():
    box = trimesh.creation.box(extents=[40.0, 10.0, 5.0])
    box.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30.0), Z))
    xy = _transform(mold_frame(box, Z, 0.0), box.vertices)[:, :2]
    np.testing.assert_allclose(sorted(np.ptp(xy, axis=0)), [10.0, 40.0], atol=1e-9)


def test_mold_frame_does_not_spin_round_parts(cylinder):
    np.testing.assert_allclose(mold_frame(cylinder, Z, 20.0)[:3, :3], np.eye(3), atol=1e-12)


def test_result_transforms_are_inverse(spool):
    result = analyze_parting(spool, X)
    np.testing.assert_allclose(result.from_mold @ result.to_mold, np.eye(4), atol=1e-12)


def test_offset_without_direction_is_rejected(sphere):
    with pytest.raises(ValueError, match="direction"):
        analyze_parting(sphere, offset=0.0)


def test_zero_direction_is_rejected(sphere):
    with pytest.raises(ValueError, match="direction"):
        classify_faces(sphere, np.zeros(3), 0.0)


def test_faces_cut_by_the_plane_must_release_both_ways():
    # Pulled along the body diagonal, the plane through the centre cuts four of the six faces.
    box = trimesh.creation.box(extents=[40.0, 20.0, 10.0])
    assert analyze_parting(box, [1.0, 1.0, 1.0], 0.0).undercut_fraction > 0.5


def test_auto_direction_follows_a_rotated_box():
    box = trimesh.creation.box(extents=[40.0, 20.0, 10.0])
    rotation = trimesh.transformations.random_rotation_matrix(np.random.default_rng(0).random(3))
    box.apply_transform(rotation)
    result = analyze_parting(box)
    assert result.undercut_fraction < NO_UNDERCUT
    assert np.max(np.abs(rotation[:3, :3].T @ result.direction)) == pytest.approx(1.0, abs=1e-6)


def _rotated(mesh: trimesh.Trimesh, seed: int) -> tuple[trimesh.Trimesh, np.ndarray]:
    rotation = trimesh.transformations.random_rotation_matrix(np.random.default_rng(seed).random(3))
    return mesh.copy().apply_transform(rotation), rotation[:3, :3]


def test_auto_direction_follows_a_rotated_torus(torus):
    rotated, rotation = _rotated(torus, 1)
    result = analyze_parting(rotated)
    assert abs(rotation[:, 2] @ result.direction) == pytest.approx(1.0, abs=1e-6)
    assert result.low_draft_fraction < NO_UNDERCUT


def test_auto_direction_follows_a_rotated_bracket():
    plate = trimesh.creation.box(bounds=[[0.0, 0.0, 0.0], [40.0, 20.0, 5.0]])
    wall = trimesh.creation.box(bounds=[[0.0, 0.0, 0.0], [5.0, 20.0, 30.0]])
    bracket = trimesh.boolean.union([plate, wall], engine="manifold")
    assert abs(analyze_parting(bracket).direction[2]) == pytest.approx(1.0)
    rotated, rotation = _rotated(bracket, 2)
    result = analyze_parting(rotated)
    assert abs(rotation[:, 2] @ result.direction) == pytest.approx(1.0, abs=1e-6)
    assert result.undercut_fraction < NO_UNDERCUT


def test_an_axis_must_not_win_with_more_undercut():
    def evaluation(direction, undercut, low_draft):
        return _Evaluation(np.asarray(direction, float), 0.0, np.zeros(0), undercut, low_draft)

    axis = evaluation(Z, 0.004, 0.1)
    oblique = evaluation([0.6, 0.0, 0.8], 0.0, 0.2)
    assert _rank([axis, oblique])[0] is oblique
    near_tie = evaluation(Z, 1e-4, 0.3)
    assert _rank([near_tie, oblique])[0] is near_tie


def test_tiny_parts_are_analysed_like_large_ones(spool):
    tiny = spool.copy().apply_scale(0.1 / spool.extents.max())
    assert analyze_parting(tiny, Z).undercut_fraction > 0.05
    assert abs(analyze_parting(tiny).direction[2]) < 0.1
