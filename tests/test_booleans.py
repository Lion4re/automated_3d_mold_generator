import math
import time

import numpy as np
import pytest
import trimesh

from moldgen.booleans import (
    BooleanError,
    difference,
    intersection,
    split_by_plane,
    to_manifold,
    to_trimesh,
    union,
)


def _box(extents, center=(0.0, 0.0, 0.0)) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents)
    box.apply_translation(center)
    return box


def _open_box() -> trimesh.Trimesh:
    box = _box([10, 10, 10])
    box.update_faces(np.arange(1, len(box.faces)))
    return box


def test_box_minus_sphere_matches_analytic_volume():
    box = _box([40, 40, 40])
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=10.0)

    result = difference(box, [sphere])

    assert result.is_watertight
    assert result.volume == pytest.approx(box.volume - sphere.volume, rel=1e-9)
    analytic = 40.0**3 - 4.0 / 3.0 * math.pi * 10.0**3
    assert result.volume == pytest.approx(analytic, rel=1e-3)
    assert not result.contains([[0.0, 0.0, 0.0]])[0]


def test_union_and_intersection_of_overlapping_boxes():
    a = _box([10, 10, 10])
    b = _box([10, 10, 10], center=(5, 0, 0))

    joined = union([a, b])
    common = intersection(a, b)

    assert joined.is_watertight and common.is_watertight
    assert joined.volume == pytest.approx(1500.0)
    assert common.volume == pytest.approx(500.0)
    np.testing.assert_allclose(common.bounds, [[0, -5, -5], [5, 5, 5]])


def test_empty_results_are_empty_meshes():
    a = _box([10, 10, 10])
    far = _box([10, 10, 10], center=(100, 0, 0))

    for result in (intersection(a, far), union([]), difference(a, [_box([30, 30, 30])])):
        assert isinstance(result, trimesh.Trimesh)
        assert len(result.faces) == 0


def test_difference_without_tools_returns_base():
    base = _box([10, 20, 30])
    assert difference(base, []).volume == pytest.approx(base.volume)


@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_split_box_by_axis_plane(scale):
    box = _box([10, 20, 30], center=(0, 0, 15))  # z from 0 to 30
    normal = np.array([0.0, 0.0, scale])

    positive, negative = split_by_plane(box, normal, 10.0 * scale)  # plane z == 10

    assert positive.is_watertight and negative.is_watertight
    assert positive.volume == pytest.approx(10 * 20 * 20)
    assert negative.volume == pytest.approx(10 * 20 * 10)
    assert positive.bounds[0, 2] == pytest.approx(10.0)
    assert negative.bounds[1, 2] == pytest.approx(10.0)


def test_split_by_oblique_plane_puts_material_on_the_right_side():
    box = _box([20, 20, 20])
    normal = np.array([1.0, 1.0, 0.0])
    offset = 4.0

    positive, negative = split_by_plane(box, normal, offset)

    assert positive.is_watertight and negative.is_watertight
    assert positive.volume + negative.volume == pytest.approx(box.volume)
    assert np.all(positive.vertices @ normal >= offset - 1e-9)
    assert np.all(negative.vertices @ normal <= offset + 1e-9)


def test_split_plane_missing_the_mesh_gives_an_empty_side():
    box = _box([10, 10, 10])
    positive, negative = split_by_plane(box, np.array([0, 0, 1]), 50.0)
    assert len(positive.faces) == 0
    assert negative.volume == pytest.approx(box.volume)


def test_split_rejects_zero_normal():
    with pytest.raises(ValueError, match="normal"):
        split_by_plane(_box([10, 10, 10]), np.zeros(3), 0.0)


def test_open_mesh_raises_boolean_error():
    with pytest.raises(BooleanError, match=r"not a closed solid.*repair"):
        difference(_box([30, 30, 30]), [_open_box()])
    with pytest.raises(BooleanError, match="not a closed solid"):
        split_by_plane(_open_box(), np.array([0, 0, 1]), 0.0)


def test_inside_out_mesh_raises_boolean_error():
    inverted = _box([10, 10, 10])
    inverted.invert()
    with pytest.raises(BooleanError, match="inside out"):
        union([inverted])


def test_unmerged_triangle_soup_is_accepted():
    box = _box([10, 20, 30])
    soup = trimesh.Trimesh(
        vertices=box.triangles.reshape(-1, 3),
        faces=np.arange(len(box.faces) * 3).reshape(-1, 3),
        process=False,
    )
    assert to_manifold(soup).volume() == pytest.approx(box.volume)


def test_round_trip_keeps_double_precision():
    box = _box([10, 10, 10], center=(12345.678901, -0.000001234, 7.0))
    restored = to_trimesh(to_manifold(box))
    np.testing.assert_allclose(restored.bounds, box.bounds, rtol=0, atol=1e-9)


def test_many_small_tools_are_subtracted_quickly():
    base = _box([100, 100, 100])
    grid = np.linspace(-40, 40, 5)
    centers = np.array([[x, y, z] for x in grid for y in grid for z in grid[:2]])
    tools = []
    for center in centers:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=3.0)
        sphere.apply_translation(center)
        tools.append(sphere)
    assert len(tools) == 50

    start = time.perf_counter()
    result = difference(base, tools)
    elapsed = time.perf_counter() - start

    assert result.is_watertight
    assert result.volume == pytest.approx(base.volume - sum(t.volume for t in tools))
    assert elapsed < 2.0


def test_mold_block_around_pawn(pawn):
    block = _box(pawn.extents + 20.0, center=pawn.bounding_box.centroid)

    cavity = difference(block, [pawn])
    top, bottom = split_by_plane(cavity, np.array([0, 0, 1]), float(pawn.centroid[2]))

    assert cavity.is_watertight
    assert cavity.volume == pytest.approx(block.volume - pawn.volume, rel=1e-9)
    assert top.is_watertight and bottom.is_watertight
    assert top.volume + bottom.volume == pytest.approx(cavity.volume, rel=1e-9)
