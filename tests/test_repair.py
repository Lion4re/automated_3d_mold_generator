import sys

import numpy as np
import pytest
import trimesh
from trimesh.exceptions import ExceptionWrapper

from moldgen.booleans import to_manifold
from moldgen.repair import RepairReport, repair_mesh


def _without_faces(mesh: trimesh.Trimesh, drop: np.ndarray) -> trimesh.Trimesh:
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[drop] = False
    return trimesh.Trimesh(mesh.vertices.copy(), mesh.faces[keep].copy(), process=False)


def _with_patch_hole(sphere: trimesh.Trimesh) -> trimesh.Trimesh:
    """Sphere with a cap of adjacent faces removed: a hole far larger than a quad."""
    near_pole = np.linalg.norm(sphere.triangles_center - [0, 0, 20.0], axis=1) < 5.0
    holed = _without_faces(sphere, np.flatnonzero(near_pole))
    holed.remove_unreferenced_vertices()
    return holed


def _assert_solid(mesh: trimesh.Trimesh, report: RepairReport) -> None:
    assert report.is_watertight and mesh.is_watertight
    assert report.manifold_ok
    assert report.volume > 0
    assert to_manifold(mesh).volume() == pytest.approx(report.volume)


def test_watertight_pawn_is_unchanged(pawn):
    repaired, report = repair_mesh(pawn)

    assert report.actions == []
    assert report.warnings == []
    assert report.was_watertight and report.is_watertight and report.manifold_ok
    assert report.bodies == 1
    assert report.volume == pytest.approx(pawn.volume)
    np.testing.assert_array_equal(repaired.vertices, pawn.vertices)
    np.testing.assert_array_equal(repaired.faces, pawn.faces)


def test_sphere_with_missing_faces_is_closed(sphere):
    holed = _without_faces(sphere, np.array([0, 400, 800, 1600, 3200]))

    repaired, report = repair_mesh(holed)

    _assert_solid(repaired, report)
    assert not report.was_watertight
    # trimesh fills these holes when networkx is installed, pymeshfix otherwise.
    assert report.actions
    assert report.volume == pytest.approx(sphere.volume, rel=1e-3)


def test_large_hole_is_closed_with_fans_without_pymeshfix(sphere, monkeypatch):
    monkeypatch.setitem(sys.modules, "pymeshfix", None)

    repaired, report = repair_mesh(_with_patch_hole(sphere))

    _assert_solid(repaired, report)
    assert "closed 1 larger hole with triangle fans" in report.actions
    assert report.volume == pytest.approx(sphere.volume, rel=1e-2)


def test_large_hole_is_closed_with_pymeshfix(sphere):
    pytest.importorskip("pymeshfix")

    repaired, report = repair_mesh(_with_patch_hole(sphere))

    _assert_solid(repaired, report)
    assert any("pymeshfix" in action for action in report.actions)
    assert report.volume == pytest.approx(sphere.volume, rel=1e-2)


def test_holes_are_closed_when_networkx_is_missing(sphere, monkeypatch):
    # trimesh's own hole filling needs networkx, which is not a hard dependency.
    monkeypatch.setattr(trimesh.repair, "nx", ExceptionWrapper(ImportError("no networkx")))
    monkeypatch.setitem(sys.modules, "pymeshfix", None)

    repaired, report = repair_mesh(_without_faces(sphere, np.array([0, 400, 800])))

    _assert_solid(repaired, report)
    assert report.volume == pytest.approx(sphere.volume, rel=1e-6)


def test_non_convex_hole_is_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymeshfix", None)
    box = trimesh.creation.box([40, 40, 40]).subdivide_to_size(2.0)
    box.merge_vertices()
    x, y, z = box.triangles_center.T
    on_top = np.isclose(z, 20.0)
    bottom_bar = (np.abs(x) < 15) & (y > -15) & (y < -5)
    left_bar = (x > -15) & (x < -5) & (np.abs(y) < 15)
    l_shape = bottom_bar | left_bar
    holed = _without_faces(box, np.flatnonzero(on_top & l_shape))

    repaired, report = repair_mesh(holed)

    _assert_solid(repaired, report)
    assert report.volume == pytest.approx(40.0**3)


def test_inverted_normals_are_fixed(sphere):
    inverted = sphere.copy()
    inverted.invert()
    assert inverted.volume < 0

    repaired, report = repair_mesh(inverted)

    _assert_solid(repaired, report)
    assert report.actions == ["turned 1 inside-out body right side out"]
    assert repaired.volume == pytest.approx(sphere.volume)


def test_inconsistent_winding_is_fixed(sphere):
    faces = sphere.faces.copy()
    faces[::7] = faces[::7, ::-1]
    scrambled = trimesh.Trimesh(sphere.vertices.copy(), faces, process=False)
    assert not scrambled.is_winding_consistent

    repaired, report = repair_mesh(scrambled)

    _assert_solid(repaired, report)
    assert any(action.startswith("fixed inconsistent winding") for action in report.actions)
    assert repaired.volume == pytest.approx(sphere.volume)


def test_duplicate_and_degenerate_faces_are_removed():
    box = trimesh.creation.box([10, 10, 10])
    faces = np.vstack([box.faces, box.faces[:3], [[0, 0, 1], [2, 3, 3]]])
    messy = trimesh.Trimesh(box.vertices.copy(), faces, process=False)

    repaired, report = repair_mesh(messy)

    _assert_solid(repaired, report)
    assert "removed 2 degenerate faces" in report.actions
    assert "removed 3 duplicate faces" in report.actions
    assert len(repaired.faces) == 12
    assert report.volume == pytest.approx(1000.0)


def test_duplicate_vertices_are_merged():
    box = trimesh.creation.box([10, 10, 10])
    soup = trimesh.Trimesh(
        vertices=box.triangles.reshape(-1, 3),
        faces=np.arange(36).reshape(-1, 3),
        process=False,
    )

    repaired, report = repair_mesh(soup)

    _assert_solid(repaired, report)
    assert report.actions == ["merged 28 duplicate vertices"]
    assert len(repaired.vertices) == 8


def test_unreferenced_and_non_finite_vertices_are_removed():
    box = trimesh.creation.box([10, 10, 10])
    vertices = np.vstack([box.vertices, [[np.nan, 0, 0], [50, 50, 50]]])
    faces = np.vstack([box.faces, [[8, 0, 1]]])
    dirty = trimesh.Trimesh(vertices, faces, process=False)

    repaired, report = repair_mesh(dirty)

    _assert_solid(repaired, report)
    assert report.actions[:2] == [
        "removed 1 face with NaN or infinite coordinates",
        "removed 1 unreferenced vertex",
    ]
    assert np.isfinite(repaired.vertices).all()


def test_debris_is_removed_and_main_body_kept(pawn):
    debris = trimesh.creation.icosphere(subdivisions=1, radius=0.3)
    debris.apply_translation(pawn.bounds[1] + 10.0)

    repaired, report = repair_mesh(trimesh.util.concatenate([pawn, debris]))

    _assert_solid(repaired, report)
    assert report.bodies == 1
    assert any("tiny loose piece" in action for action in report.actions)
    assert report.volume == pytest.approx(pawn.volume)


def test_separate_bodies_are_kept_with_a_warning(sphere):
    other = sphere.copy()
    other.apply_translation([100, 0, 0])

    repaired, report = repair_mesh(trimesh.util.concatenate([sphere, other]))

    _assert_solid(repaired, report)
    assert report.actions == []
    assert report.bodies == 2
    assert any("2 separate bodies" in warning for warning in report.warnings)


def test_overlapping_bodies_are_merged():
    a = trimesh.creation.box([10, 10, 10])
    b = trimesh.creation.box([10, 10, 10])
    b.apply_translation([5, 0, 0])

    repaired, report = repair_mesh(trimesh.util.concatenate([a, b]))

    _assert_solid(repaired, report)
    assert report.actions == ["merged overlapping bodies (2 became 1)"]
    assert report.bodies == 1
    assert report.volume == pytest.approx(1500.0)


def test_flat_sheet_is_reported_as_not_a_solid(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymeshfix", None)
    sheet = trimesh.Trimesh(
        [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], [[0, 1, 2], [0, 2, 3]], process=False
    )

    _, report = repair_mesh(sheet)

    assert not report.manifold_ok
    assert any("encloses no volume" in warning for warning in report.warnings)


def test_unrepairable_mesh_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymeshfix", None)
    box = trimesh.creation.box([10, 10, 10])
    a, b = box.edges_unique[0]
    # A large fin hanging off one edge makes that edge shared by three faces.
    finned = trimesh.Trimesh(
        np.vstack([box.vertices, [[0, 0, 30]]]),
        np.vstack([box.faces, [[a, b, 8]]]),
        process=False,
    )

    _, report = repair_mesh(finned)

    assert not report.manifold_ok
    assert any("not a closed solid" in warning for warning in report.warnings)
    assert any("moldgen[repair]" in warning for warning in report.warnings)


def test_empty_mesh_is_reported():
    repaired, report = repair_mesh(trimesh.Trimesh())
    assert len(repaired.faces) == 0
    assert not report.manifold_ok
    assert report.warnings


def test_input_mesh_is_not_mutated(sphere):
    broken = _without_faces(sphere, np.array([0, 10, 20]))
    broken.invert()
    broken = trimesh.Trimesh(
        np.vstack([broken.vertices, broken.vertices[:5]]),
        np.vstack([broken.faces, broken.faces[:4]]),
        process=False,
    )
    vertices_before = broken.vertices.copy()
    faces_before = broken.faces.copy()

    repaired, report = repair_mesh(broken)

    assert repaired is not broken
    assert report.actions
    np.testing.assert_array_equal(broken.vertices, vertices_before)
    np.testing.assert_array_equal(broken.faces, faces_before)
    assert not broken.is_watertight


def test_faces_with_missing_vertices_are_dropped():
    box = trimesh.creation.box([10, 10, 10])
    faces = np.vstack([box.faces, [[0, 1, 50]]])
    dirty = trimesh.Trimesh(box.vertices, faces, process=False)

    repaired, report = repair_mesh(dirty)

    _assert_solid(repaired, report)
    assert report.actions[0] == "removed 1 face referencing missing vertices"
    assert report.volume == pytest.approx(1000.0)
