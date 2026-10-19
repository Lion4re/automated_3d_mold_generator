import numpy as np
import pytest
import trimesh

from moldgen.parting import points_inside
from moldgen.pieces import Cap, _snap_to_block
from moldgen.printcheck import MIN_WALL_FDM_MM, best_print_up, check_piece

BED = (220.0, 220.0, 250.0)


def test_solid_block_has_no_print_problems():
    block = trimesh.creation.box(bounds=[[0, 0, 0], [40, 30, 20]])

    check = check_piece("top", block, BED, MIN_WALL_FDM_MM)

    assert check.fits_bed and not check.fragile
    assert check.contact_mm2 == 1200.0
    assert check.overhang_mm2 == 0.0 and check.thin_mm2 == 0.0
    assert check.warnings(BED, MIN_WALL_FDM_MM) == []


def test_thin_plate_and_oversized_piece_are_reported():
    plate = trimesh.creation.box(bounds=[[0, 0, 0], [300, 40, 0.6]])

    check = check_piece("side_1", plate, BED, MIN_WALL_FDM_MM)
    notes = " ".join(check.warnings(BED, MIN_WALL_FDM_MM))

    assert not check.fits_bed and check.min_wall_mm < MIN_WALL_FDM_MM
    assert check.thin_mm2 > 0.9 * plate.area
    assert "does not fit" in notes and "thinner than" in notes


def test_piece_on_an_edge_is_turned_onto_a_face():
    # A wedge whose preferred side up leaves it standing on its sharp edge.
    wedge = trimesh.Trimesh(
        [[0, 0, 0], [20, 0, 0], [0, 0, 20], [0, 30, 0], [20, 30, 0], [0, 30, 20]],
        [[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2], [1, 2, 5], [1, 5, 4]],
    )
    wedge.fix_normals()
    edge_down = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)

    up = best_print_up(wedge, edge_down)

    assert not np.allclose(up, edge_down)


def test_ray_through_a_shared_edge_counts_one_crossing():
    box = trimesh.creation.box([10.0, 10.0, 10.0])  # the centre's ray meets the top diagonal

    assert points_inside(box, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])).tolist() == [
        True,
        False,
    ]


def test_thin_area_does_not_depend_on_tessellation():
    plate = trimesh.creation.box(bounds=[[0, 0, 0], [80, 80, 0.6]])
    # Small triangles on the thin faces, large ones on the thick edges, and more
    # faces than are measured, so the thin area is estimated from a sample.
    fine = plate
    for _ in range(6):
        fine = fine.subdivide(np.flatnonzero(np.abs(fine.face_normals[:, 2]) > 0.9))

    coarse_check = check_piece("plate", plate, BED, MIN_WALL_FDM_MM)
    fine_check = check_piece("plate", fine, BED, MIN_WALL_FDM_MM)

    assert fine_check.thin_mm2 == pytest.approx(coarse_check.thin_mm2, rel=0.02)


def test_points_inside_matches_trimesh(sphere, rng):
    points = rng.uniform(-25, 25, size=(3000, 3))

    assert np.array_equal(points_inside(sphere, points), sphere.contains(points))


def test_side_plane_close_to_the_block_face_is_dropped():
    block = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
    x = np.array([1.0, 0.0, 0.0])
    cap = Cap(np.array([0.0, 1.0, 0.0]), 80.0, bounds=((x, 1.0), (-x, -50.0)))

    snapped = _snap_to_block(cap, block, np.array([[50.0, 90.0, 50.0]]))
    blocked = _snap_to_block(cap, block, np.array([[0.5, 90.0, 50.0]]))

    assert len(snapped.bounds) == 1 and snapped.bounds[0][1] == -50.0
    assert blocked is cap  # cast in the slab: the cap would have to release it too
