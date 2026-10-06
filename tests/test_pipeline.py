import json
import zipfile
from io import BytesIO
from itertools import combinations

import numpy as np
import pytest
import trimesh

from moldgen import MoldConfig, MoldError, booleans, generate_mold
from moldgen.report import summary, zip_result


def _intersection_volume(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    # Pieces touch along the parting plane, which can leave a zero-volume sliver;
    # manifold3d measures that without trimesh's centre-of-mass division.
    return float((booleans.to_manifold(a) ^ booleans.to_manifold(b)).volume())


def _assert_valid_pieces(result, expected: int) -> None:
    assert len(result.pieces) == expected
    for piece in result.pieces:
        assert piece.mesh.is_watertight, piece.name
        assert piece.mesh.volume > 0, piece.name
        printed = piece.print_mesh()
        assert printed.bounds[0][2] == pytest.approx(0.0, abs=1e-6), piece.name
    for a, b in combinations(result.pieces, 2):
        overlap = _intersection_volume(a.mesh, b.mesh)
        assert overlap < 1e-3 * min(a.mesh.volume, b.mesh.volume), (a.name, b.name, overlap)


def _assert_pieces_clear_of_cavity(result) -> None:
    for piece in result.pieces:
        overlap = _intersection_volume(piece.mesh, result.cavity)
        assert overlap < 1e-3 * result.cavity.volume, (piece.name, overlap)


def test_sphere_two_piece(sphere):
    result = generate_mold(sphere, MoldConfig(material="resin"))
    _assert_valid_pieces(result, 2)
    _assert_pieces_clear_of_cavity(result)
    assert result.parting.undercut_fraction < 0.005
    assert len(result.gating.vents) == 0
    assert sum(len(plan.positions) for plan in result.keys) >= 2


def test_spool_is_split_through_its_axle(spool):
    result = generate_mold(spool, MoldConfig())
    assert abs(result.parting.direction[2]) < 0.1
    assert result.parting.undercut_fraction < 0.005
    _assert_valid_pieces(result, 2)


def test_forced_direction_reports_undercut(spool):
    result = generate_mold(spool, MoldConfig(direction="z"))
    assert result.parting.undercut_fraction > 0.05
    assert any("undercut" in warning for warning in result.warnings)


def test_cavity_is_open_to_the_outside(sphere):
    result = generate_mold(sphere, MoldConfig())
    assert not any("not reachable" in warning for warning in result.warnings)
    assembled = trimesh.boolean.union([p.mesh for p in result.pieces], engine="manifold")
    block = trimesh.creation.box(bounds=result.block_bounds)
    void = trimesh.boolean.difference([block, assembled], engine="manifold")
    cavity_bodies = [
        b for b in void.split(only_watertight=False) if b.volume > 0.5 * result.cavity.volume
    ]
    assert len(cavity_bodies) == 1
    # The channel void reaches the outer face of the block, i.e. it is open to the outside.
    assert np.isclose(cavity_bodies[0].bounds, result.block_bounds, atol=1e-3).any()


def test_shrinkage_compensation_enlarges_cavity(cylinder):
    result = generate_mold(cylinder, MoldConfig(shrinkage=0.02))
    expected = cylinder.volume * (1 / 0.98) ** 3
    assert result.cavity.volume == pytest.approx(expected, rel=1e-3)


def test_hot_material_in_pla_warns(cylinder):
    from moldgen.materials import MATERIALS

    hot = [m.key for m in MATERIALS.values() if m.pour_temp_c and m.pour_temp_c[1] > 60]
    if not hot:
        pytest.skip("no hot-pour material preset")
    result = generate_mold(cylinder, MoldConfig(material=hot[0], print_material="pla"))
    assert any("soften" in warning for warning in result.warnings)


def test_open_surface_is_rejected():
    plane = trimesh.Trimesh(
        vertices=[[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], faces=[[0, 1, 2], [0, 2, 3]]
    )
    with pytest.raises(MoldError):
        generate_mold(plane, MoldConfig())


def test_report_and_zip(cylinder, tmp_path):
    result = generate_mold(cylinder, MoldConfig())
    files = result.save(tmp_path)
    names = {path.name for path in files}
    assert {"report.json", "INSTRUCTIONS.txt"} <= names
    assert sum(name.endswith(".stl") for name in names) == 2
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["mold"]["pieces"] == 2
    json.dumps(summary(result))
    with zipfile.ZipFile(BytesIO(zip_result(result))) as archive:
        assert len(archive.namelist()) == 4


@pytest.mark.slow
def test_pawn_four_piece(pawn):
    result = generate_mold(pawn, MoldConfig(pieces=4))
    _assert_valid_pieces(result, 4)
    _assert_pieces_clear_of_cavity(result)
    assert abs(result.parting.direction[2]) < 0.1


@pytest.mark.slow
def test_pawn_two_piece(pawn):
    result = generate_mold(pawn, MoldConfig())
    _assert_valid_pieces(result, 2)
    _assert_pieces_clear_of_cavity(result)
    assert sum(len(plan.positions) for plan in result.keys) == 4


def test_keys_fit_beside_a_part_that_fills_the_parting_face():
    box = trimesh.creation.box([40.0, 20.0, 10.0])

    result = generate_mold(box, MoldConfig(keys=4))

    assert len(result.keys[0].positions) == 4
    assert not any("registration keys" in warning for warning in result.warnings)
