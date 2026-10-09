import numpy as np
import pytest
import trimesh

from moldgen import MoldConfig, booleans, generate_mold
from moldgen.config import ConfigError
from moldgen.pieces import CastFaces, _place_plane, locked_faces, plan_caps
from moldgen.report import instructions, summary


def _cross_drilled() -> trimesh.Trimesh:
    """A block with a through-hole along X above z == 0 and one along Y below it.

    No single pull direction releases both holes, so a two-piece mold locks.
    """
    block = trimesh.creation.box([40.0, 30.0, 30.0])
    holes = []
    for axis, z in (([0.0, 1.0, 0.0], 9.0), ([1.0, 0.0, 0.0], -9.0)):
        hole = trimesh.creation.cylinder(radius=3.0, height=60.0, sections=48)
        hole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, axis))
        hole.apply_translation([0.0, 0.0, z])
        holes.append(hole)
    return booleans.difference(block, holes)


@pytest.fixture(scope="module")
def drilled_mold():
    return generate_mold(_cross_drilled(), MoldConfig(pieces="auto"))


def test_caps_release_what_two_halves_cannot():
    cast = CastFaces(_cross_drilled(), [])
    assert cast.locked_area(locked_faces(cast, [])) > 0.1

    caps = plan_caps(cast, 4)

    assert 1 <= len(caps) <= 4
    assert cast.locked_area(locked_faces(cast, caps)) == 0.0


def test_part_that_releases_in_two_pieces_gets_no_caps(sphere):
    cast = CastFaces(sphere, [])
    assert plan_caps(cast, 4) == []


def test_auto_mold_of_a_simple_part_matches_the_two_piece_mold(pawn):
    auto = generate_mold(pawn, MoldConfig(pieces="auto"))
    two = generate_mold(pawn, MoldConfig(pieces=2))

    assert auto.layout is not None and auto.layout.caps == []
    assert [p.name for p in auto.pieces] == [p.name for p in two.pieces]
    for a, b in zip(auto.pieces, two.pieces, strict=True):
        assert a.mesh.volume == pytest.approx(b.mesh.volume, rel=1e-9)


@pytest.mark.slow
def test_side_pieces_come_off_without_catching(drilled_mold):
    result = drilled_mold
    assert len(result.pieces) > 2
    assert result.layout.remaining_locked_fraction < 0.005
    assert not any("may catch" in warning for warning in result.warnings)

    cast = booleans.to_manifold(booleans.union([result.cavity, *result.gating.solids()]), "cast")
    solids = [booleans.to_manifold(piece.mesh, piece.name) for piece in result.pieces]
    size = float(np.linalg.norm(result.block_bounds[1] - result.block_bounds[0]))
    for i, piece in enumerate(result.pieces):
        assert piece.mesh.is_watertight
        others = cast
        for later in solids[i + 1 :]:
            others = others + later
        for distance in (0.5, 3.0, size):
            moved = solids[i].translate(tuple(piece.pull * distance))
            assert (moved ^ others).volume() < 1e-3 * solids[i].volume(), (piece.name, distance)


@pytest.mark.slow
def test_report_lists_the_removal_order(drilled_mold):
    info = summary(drilled_mold)
    text = instructions(drilled_mold)

    assert info["layout"]["side_pieces"] == len(drilled_mold.layout.caps)
    assert all(piece["pull_label"] for piece in info["pieces"])
    order = ", ".join(f"{p['name']} along {p['pull_label']}" for p in info["pieces"])
    assert order in text


def test_piece_limit_fills_what_cannot_be_released():
    result = generate_mold(_cross_drilled(), MoldConfig(pieces="auto", max_pieces=2))

    assert result.layout.caps == []
    assert result.layout.filled_volume > 0
    assert len(result.pieces) == 2
    assert any("was filled" in warning for warning in result.warnings)


def test_cut_plane_keeps_clear_of_parallel_planes():
    # Halfway point near the earlier plane at 0: overlap it by a full gap.
    assert _place_plane(-10.0, 1.0, [0.0], gap=2.0) == pytest.approx(-2.0)
    # No room to overlap: leave a full gap on the other side.
    assert _place_plane(-0.5, 2.5, [0.0], gap=2.0) == pytest.approx(2.0)
    # No room either way: the end of the range farthest from the plane.
    assert _place_plane(-1.0, 1.5, [0.0], gap=2.0) == pytest.approx(1.5)
    # No parallel plane nearby: halfway, at most one gap below the lowest face.
    assert _place_plane(-10.0, 10.0, [], gap=2.0) == pytest.approx(8.0)


@pytest.mark.parametrize(
    ("pieces", "max_pieces", "valid"),
    [("auto", 6, True), (2, 6, True), (4, 6, True), (3, 6, False), ("auto", 1, False)],
)
def test_piece_settings_are_validated(pieces, max_pieces, valid):
    config = MoldConfig(pieces=pieces, max_pieces=max_pieces)
    if valid:
        config.validate()
    else:
        with pytest.raises(ConfigError):
            config.validate()


def test_part_side_pieces_cannot_release_gets_a_two_piece_mold(monkeypatch):
    from moldgen import pieces, pipeline

    # Through-holes along X and along Y: no single pull releases both.
    block = trimesh.creation.box([30.0, 30.0, 20.0])
    holes = [
        trimesh.creation.cylinder(radius=4.0, height=40.0, transform=turn)
        for turn in (
            trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]),
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]),
        )
    ]
    drilled = booleans.difference(block, holes)
    monkeypatch.setattr(pieces, "RIGID_LIMIT_LOCKED", 0.0)
    monkeypatch.setattr(pipeline, "RIGID_LIMIT_LOCKED", 0.0)

    result = generate_mold(drilled, MoldConfig())

    assert len(result.pieces) == 2 and result.layout is None
    assert "a rigid mold cannot release this part" in result.warnings[0]
    assert not any("--pieces auto" in warning for warning in result.warnings)
