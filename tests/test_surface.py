import numpy as np
import pytest
import shapely
import trimesh

from moldgen import MoldConfig, booleans, generate_mold
from moldgen.config import ConfigError
from moldgen.gating import Channel, GatingPlan
from moldgen.surface import fit_surface


def _curved_tube() -> trimesh.Trimesh:
    """A tube along a curve that leaves every plane: no flat cut splits it cleanly.

    Each vertical line meets it at most once, so a curved surface can.
    """
    t = np.linspace(-1.0, 1.0, 60)
    path = np.column_stack([40.0 * t, 12.0 * np.sin(2.0 * t), 10.0 * t**2 - 3.0])
    return trimesh.creation.sweep_polygon(shapely.Point(0, 0).buffer(4.0, quad_segs=16), path)


def _block(mesh: trimesh.Trimesh, wall: float = 6.0) -> np.ndarray:
    return np.array([mesh.bounds[0] - wall, mesh.bounds[1] + wall])


def test_plane_is_kept_when_it_runs_through_the_cast(sphere):
    assert fit_surface(sphere, _block(sphere)).flat


def test_surface_follows_the_tube_inside_the_cast():
    tube = _curved_tube()
    surface = fit_surface(tube, _block(tube))

    assert not surface.flat
    t = np.linspace(-0.9, 0.9, 40)
    centre = np.column_stack([40.0 * t, 12.0 * np.sin(2.0 * t), 10.0 * t**2 - 3.0])
    assert np.all(np.abs(surface.height(centre[:, :2]) - centre[:, 2]) < 4.0)


def test_solid_below_the_surface_is_closed():
    tube = _curved_tube()
    block = _block(tube)
    surface = fit_surface(tube, block)

    below = surface.below(float(block[0, 2]) - 1.0)

    assert below.status().name == "NoError"
    lo, hi = block
    footprint = (hi[0] - lo[0] + 2 * surface.cell) * (hi[1] - lo[1] + 2 * surface.cell)
    assert below.volume() > 0.5 * footprint


def test_surface_is_held_at_channel_height_beyond_the_part(sphere):
    # Raised so the plane z == 0 misses the sphere near its rim and the surface must curve.
    raised = sphere.copy().apply_translation([0.0, 0.0, 6.0])
    block = np.array([[-30.0, -26.0, -20.0], [30.0, 26.0, 32.0]])
    sprue = Channel(start=np.array([20.0, 0.0, 2.0]), end=np.array([30.0, 0.0, 2.0]), radius=2.0)
    gating = GatingPlan(up=np.array([1.0, 0.0, 0.0]), sprue=sprue)

    surface = fit_surface(raised, block, gating)

    assert not surface.flat
    path = np.column_stack([np.linspace(22.0, 29.0, 8), np.zeros(8)])
    assert np.allclose(surface.height(path), 2.0)


def test_curved_two_piece_mold_releases_the_tube():
    flat = generate_mold(_curved_tube(), MoldConfig(pieces=2, parting_surface="flat"))
    curved = generate_mold(_curved_tube(), MoldConfig(pieces=2))

    assert flat.surface is None and flat.parting.undercut_fraction > 0.05
    assert curved.surface is not None
    assert curved.layout.locked_fraction < 0.005
    assert [piece.name for piece in curved.pieces] == ["top", "bottom"]
    assert all(piece.mesh.is_watertight for piece in curved.pieces)
    assert not any("catch" in warning for warning in curved.warnings)
    assert len(curved.keys[0].positions) >= 2


@pytest.mark.slow
def test_curved_halves_slide_apart():
    result = generate_mold(_curved_tube(), MoldConfig(pieces=2))
    cast = booleans.to_manifold(booleans.union([result.cavity, *result.gating.solids()]), "cast")
    top, bottom = (booleans.to_manifold(piece.mesh, piece.name) for piece in result.pieces)
    for distance in (0.5, 3.0, 60.0):
        moved = top.translate((0.0, 0.0, distance))
        assert (moved ^ (cast + bottom)).volume() < 1e-3 * top.volume()
        moved = bottom.translate((0.0, 0.0, -distance))
        assert (moved ^ cast).volume() < 1e-3 * bottom.volume()


def test_parting_surface_setting_is_validated():
    MoldConfig(parting_surface="flat").validate()
    with pytest.raises(ConfigError):
        MoldConfig(parting_surface="wavy").validate()
