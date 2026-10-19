from pathlib import Path

import numpy as np
import pytest
import trimesh

from moldgen.meshio import (
    MeshExportError,
    MeshLoadError,
    export_mesh,
    load_mesh,
    units_warning,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SAMPLE_MODELS = sorted(MODELS_DIR.glob("*.stl"))


def test_sample_models_exist():
    assert len(SAMPLE_MODELS) >= 6


@pytest.mark.parametrize("path", SAMPLE_MODELS, ids=lambda p: p.stem)
def test_load_sample_models(path):
    mesh = load_mesh(path)

    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.is_watertight
    assert mesh.volume > 0
    assert 20.0 < mesh.extents.max() < 150.0
    assert units_warning(mesh) is None


def test_obj_round_trip(tmp_path, sphere):
    written = export_mesh(sphere, tmp_path / "sphere.obj")
    loaded = load_mesh(written)

    assert written == tmp_path / "sphere.obj"
    assert len(loaded.faces) == len(sphere.faces)
    assert loaded.is_watertight
    assert loaded.volume == pytest.approx(sphere.volume, rel=1e-6)
    np.testing.assert_allclose(loaded.bounds, sphere.bounds, atol=1e-6)


def test_export_defaults_to_binary_stl_and_creates_directories(tmp_path, cylinder):
    written = export_mesh(cylinder, tmp_path / "nested" / "dir" / "part")

    assert written == tmp_path / "nested" / "dir" / "part.stl"
    # Binary STL: 80-byte header, uint32 count, 50 bytes per triangle.
    assert written.stat().st_size == 84 + 50 * len(cylinder.faces)
    assert load_mesh(written).volume == pytest.approx(cylinder.volume, rel=1e-5)


def test_export_rejects_unknown_suffix(tmp_path, sphere):
    with pytest.raises(MeshExportError, match=r"\.xyz"):
        export_mesh(sphere, tmp_path / "part.xyz")
    assert not (tmp_path / "part.xyz").exists()


@pytest.mark.parametrize(("units", "factor"), [("m", 1000.0), ("cm", 10.0), ("in", 25.4)])
def test_units_are_converted_to_millimetres(tmp_path, units, factor):
    box = trimesh.creation.box([0.03, 0.02, 0.01])
    path = export_mesh(box, tmp_path / "box.stl")

    mesh = load_mesh(path, units=units)

    np.testing.assert_allclose(mesh.extents, np.array([0.03, 0.02, 0.01]) * factor, rtol=1e-6)


def test_scale_is_applied_after_unit_conversion(tmp_path):
    path = export_mesh(trimesh.creation.box([0.03, 0.02, 0.01]), tmp_path / "box.stl")
    mesh = load_mesh(path, units="m", scale=2.0)
    np.testing.assert_allclose(mesh.extents, [60.0, 40.0, 20.0], rtol=1e-6)


def test_scene_is_flattened_with_transforms(tmp_path):
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box([2, 2, 2]), node_name="a")
    scene.add_geometry(
        trimesh.creation.box([2, 2, 2]),
        node_name="b",
        transform=trimesh.transformations.translation_matrix([10, 0, 0]),
    )
    path = tmp_path / "scene.glb"
    scene.export(path)

    mesh = load_mesh(path)

    np.testing.assert_allclose(mesh.bounds, [[-1, -1, -1], [11, 1, 1]])
    assert mesh.body_count == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises(MeshLoadError, match="not found"):
        load_mesh(tmp_path / "missing.stl")


def test_unsupported_suffix_raises(tmp_path):
    path = tmp_path / "part.fbx"
    path.write_bytes(b"not a mesh")
    with pytest.raises(MeshLoadError, match=r"Unsupported file type '\.fbx'"):
        load_mesh(path)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.stl"
    path.write_bytes(b"")
    with pytest.raises(MeshLoadError, match="no triangles"):
        load_mesh(path)


def test_corrupt_file_raises(tmp_path):
    path = tmp_path / "broken.ply"
    path.write_text("this is not a ply file")
    with pytest.raises(MeshLoadError, match=r"not a valid \.ply file"):
        load_mesh(path)


def test_point_cloud_raises(tmp_path):
    path = tmp_path / "points.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
        "0 0 0\n1 0 0\n0 1 0\n"
    )
    with pytest.raises(MeshLoadError, match="point cloud"):
        load_mesh(path)


def test_invalid_units_and_scale_raise(tmp_path, sphere):
    path = export_mesh(sphere, tmp_path / "sphere.stl")
    with pytest.raises(ValueError, match="units"):
        load_mesh(path, units="ft")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Scale"):
        load_mesh(path, scale=0.0)


def test_units_warning_for_tiny_part():
    tiny = trimesh.creation.box([0.03, 0.02, 0.05])
    message = units_warning(tiny)
    assert message is not None
    assert "metres, centimetres or inches" in message


def test_units_warning_for_huge_part():
    huge = trimesh.creation.box([2000.0, 500.0, 500.0])
    message = units_warning(huge)
    assert message is not None
    assert "check the units" in message


def test_units_warning_ignores_mesh_without_faces():
    points = trimesh.Trimesh(vertices=[[0, 0, 0], [0.01, 0, 0]], faces=np.zeros((0, 3), int))
    assert "metres" in units_warning(points)


def test_corrupt_gltf_names_the_file(tmp_path):
    path = tmp_path / "broken.gltf"
    path.write_text("{not json")
    with pytest.raises(MeshLoadError, match=r"broken\.gltf is not a valid \.gltf file"):
        load_mesh(path)


def test_export_into_a_file_path_raises_export_error(tmp_path):
    blocker = tmp_path / "taken"
    blocker.write_text("")
    with pytest.raises(MeshExportError, match="Could not write"):
        export_mesh(trimesh.creation.box(), blocker / "part.stl")


def test_no_units_warning_for_pawn(pawn):
    assert units_warning(pawn) is None
