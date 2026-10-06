from pathlib import Path

import numpy as np
import pytest
import trimesh

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@pytest.fixture
def sphere() -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=4, radius=20.0)


@pytest.fixture
def cylinder() -> trimesh.Trimesh:
    """Upright cylinder (axis along Z), 30 mm diameter, 40 mm tall, base on z = 0."""
    mesh = trimesh.creation.cylinder(radius=15.0, height=40.0, sections=64)
    mesh.apply_translation([0, 0, 20.0])
    return mesh


@pytest.fixture
def mushroom() -> trimesh.Trimesh:
    """A wide cap on a thin stem along Z.

    Moldable along Z only if the parting plane sits at the underside of the cap (z = 30).
    """
    stem = trimesh.creation.cylinder(radius=5.0, height=30.0, sections=48)
    stem.apply_translation([0, 0, 15.0])
    cap = trimesh.creation.cylinder(radius=15.0, height=8.0, sections=48)
    cap.apply_translation([0, 0, 34.0])
    return trimesh.boolean.union([stem, cap], engine="manifold")


@pytest.fixture
def spool() -> trimesh.Trimesh:
    """Two discs joined by a thin axle along Z.

    Splitting across the axle (direction Z) always traps one disc; splitting
    through the axle (direction X or Y) is free of undercuts.
    """
    parts = []
    for z in (3.0, 27.0):
        disc = trimesh.creation.cylinder(radius=15.0, height=6.0, sections=64)
        disc.apply_translation([0, 0, z])
        parts.append(disc)
    axle = trimesh.creation.cylinder(radius=5.0, height=20.0, sections=48)
    axle.apply_translation([0, 0, 15.0])
    parts.append(axle)
    return trimesh.boolean.union(parts, engine="manifold")


@pytest.fixture
def torus() -> trimesh.Trimesh:
    """Torus lying flat (hole axis along Z)."""
    return trimesh.creation.torus(
        major_radius=20.0, minor_radius=6.0, major_sections=64, minor_sections=32
    )


@pytest.fixture
def pawn() -> trimesh.Trimesh:
    return trimesh.load_mesh(MODELS_DIR / "Pawn.stl")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
