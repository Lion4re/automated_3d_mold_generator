import numpy as np
import pytest
import trimesh

from moldgen import MoldConfig, booleans, generate_mold
from moldgen.config import ConfigError
from moldgen.keys import plane_basis
from moldgen.pieces import CastFaces, PieceLayout, _build_curved, locked_faces, plan_caps
from moldgen.pipeline import MoldResult, _mold_score
from moldgen.surface import fit_cut

MODELS = "models"


def _ring_with_hole_along_x() -> trimesh.Trimesh:
    ring = trimesh.creation.torus(major_radius=9.0, minor_radius=2.5)
    ring.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    return ring


def _two_handles() -> trimesh.Trimesh:
    """A ball with ring handles at right angles: one hole along X, one along Y."""
    body = trimesh.creation.icosphere(subdivisions=4, radius=15.0)
    rings = []
    for axis, shift in (([1, 0, 0], [20.0, 0, 0]), ([0, 1, 0], [0, 20.0, 0])):
        ring = trimesh.creation.torus(major_radius=9.0, minor_radius=2.5)
        ring.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, axis))
        ring.apply_translation(shift)
        rings.append(ring)
    return booleans.union([body, *rings])


def test_cut_runs_past_the_cast_and_across_free_lines():
    ring = _ring_with_hole_along_x()
    rows = plane_basis(np.array([1.0, 0.0, 0.0]))
    uv = ring.vertices @ rows[:2].T
    cut = fit_cut(ring, rows, np.array([uv.min(axis=0) - 3, uv.max(axis=0) + 3]), 0.4)

    assert cut.beyond(np.array([[3.0, 9.0, 0.0]]))[0] > 0  # past the tube along +X
    assert cut.beyond(np.array([[0.0, 9.0, 0.0]]))[0] < 0  # inside the tube
    assert cut.solid(30.0).status().name == "NoError"


def test_cap_solid_matches_its_reach():
    ring = _ring_with_hole_along_x()
    cast = CastFaces(ring, [])
    y, z = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    box = ((y, -12.0), (-y, -12.0), (z, -12.0), (-z, -12.0))  # |y|, |z| <= 12
    cap = _build_curved(cast, np.array([1.0, 0.0, 0.0]), 0, box)
    assert cap is not None and cap.cut is not None

    lo, hi = np.array([-8.0, -15.0, -15.0]), np.array([8.0, 15.0, 15.0])
    region = cap.region(booleans.to_manifold(trimesh.creation.box(bounds=[lo, hi]), "box"))
    points = np.random.default_rng(1).uniform(lo, hi, size=(2000, 3))

    inside = booleans.to_trimesh(region).contains(points)

    # The solid the piece is cut from and the reach the analysis uses agree; points
    # within a grid cell of the curved cut may fall either way.
    assert np.mean(inside == cap.contains(points)) > 0.98


@pytest.mark.slow
def test_curved_caps_release_both_handle_holes():
    cast = CastFaces(_two_handles(), [])
    caps = plan_caps(cast, 4)

    assert any(cap.cut is not None for cap in caps)
    assert cast.locked_area(locked_faces(cast, caps)) < 0.01


@pytest.mark.slow
def test_auto_keeps_curved_side_pieces_when_they_are_better():
    knight = f"{MODELS}/Knight.stl"
    curved = generate_mold(knight, MoldConfig(direction="z"))
    flat = generate_mold(knight, MoldConfig(direction="z", side_piece_cuts="flat"))

    assert not curved.pieces_catch
    assert _mold_score(curved) <= _mold_score(flat)
    assert all(piece.mesh.is_watertight for piece in curved.pieces)


def test_better_mold_wins():
    def result(catch: bool, locked: float, pieces: int) -> MoldResult:
        stub = MoldResult.__new__(MoldResult)
        stub.pieces_catch = catch
        stub.layout = PieceLayout(locked_fraction=locked)
        stub.pieces = [None] * pieces
        return stub

    assert _mold_score(result(False, 0.01, 6)) < _mold_score(result(True, 0.0, 2))
    assert _mold_score(result(False, 0.0, 6)) < _mold_score(result(False, 0.01, 2))
    assert _mold_score(result(False, 0.0, 3)) < _mold_score(result(False, 0.0, 4))


def test_side_piece_cut_setting_is_validated():
    MoldConfig(side_piece_cuts="flat").validate()
    with pytest.raises(ConfigError):
        MoldConfig(side_piece_cuts="wavy").validate()
