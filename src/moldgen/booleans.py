"""Robust mesh booleans backed by manifold3d.

Meshes cross the boundary as ``manifold3d.Mesh64`` (float64 positions), so
no precision is lost converting to and from trimesh. An operand made of
several overlapping shells is treated as their union. Every result is a
closed, consistently oriented solid. A result with no volume (for example
the intersection of two disjoint solids) is returned as an empty
``trimesh.Trimesh`` with zero vertices and faces, never as ``None``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import manifold3d
import numpy as np
import trimesh
from manifold3d import Manifold, OpType

logger = logging.getLogger(__name__)

_REPAIR_HINT = "try enabling repair"


class BooleanError(RuntimeError):
    """Raised when a boolean operation cannot be carried out."""


def to_manifold(
    mesh: trimesh.Trimesh,
    label: str = "the mesh",
    *,
    suggest_repair: bool = True,
    merge_vertices: bool = True,
) -> Manifold:
    """Convert ``mesh`` to a ``Manifold``, raising :class:`BooleanError` if it is not a valid solid.

    ``label`` names the mesh in error messages (for example "the part");
    ``suggest_repair`` appends advice to enable mesh repair. With
    ``merge_vertices`` a surface stored with duplicated vertices is accepted
    by merging coincident vertices. A mesh without faces converts to an
    empty ``Manifold``.
    """
    if len(mesh.faces) == 0:
        return Manifold()
    # manifold3d only accepts writeable C-ordered arrays of these exact dtypes,
    # and trimesh often hands out read-only views, so always copy.
    mesh64 = manifold3d.Mesh64(
        vert_properties=np.array(mesh.vertices, dtype=np.float64, order="C"),
        tri_verts=np.array(mesh.faces, dtype=np.uint64, order="C"),
    )
    solid = Manifold(mesh64)
    if merge_vertices and solid.status() == manifold3d.Error.NotManifold and mesh64.merge():
        # The surface may be closed but stored with duplicated vertices
        # (a triangle soup); merging coincident vertices is lossless.
        logger.debug("%s needed vertex merging to become manifold", label)
        solid = Manifold(mesh64)
    _check_status(solid, label, suggest_repair=suggest_repair)
    if solid.volume() < 0:
        raise BooleanError(
            _message(label, "is inside out (its faces point inwards)", suggest_repair)
        )
    return solid


def to_trimesh(solid: Manifold) -> trimesh.Trimesh:
    """Convert a ``Manifold`` to a ``trimesh.Trimesh`` (empty if the manifold is empty)."""
    out = solid.to_mesh64()
    vertices = np.asarray(out.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(out.tri_verts, dtype=np.int64)
    # process=False keeps manifold3d's topology: merging coincident vertices
    # could fuse distinct sheets that touch along an edge.
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def difference(base: trimesh.Trimesh, tools: Iterable[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Subtract every mesh in ``tools`` from ``base`` in a single batched operation."""
    operands = [_operand(base, "the base mesh")]
    operands.extend(_tool_manifolds(tools))
    if len(operands) == 1:
        return to_trimesh(operands[0])
    return _run(operands, OpType.Subtract, "difference")


def union(meshes: Iterable[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Return the union of ``meshes`` (empty if no meshes are given)."""
    operands = [_operand(mesh, f"mesh {i + 1}") for i, mesh in enumerate(meshes)]
    return _run(operands, OpType.Add, "union")


def intersection(a: trimesh.Trimesh, b: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return the volume shared by ``a`` and ``b`` (empty if they do not overlap)."""
    operands = [_operand(a, "the first mesh"), _operand(b, "the second mesh")]
    return _run(operands, OpType.Intersect, "intersection")


def split_by_plane(
    mesh: trimesh.Trimesh, normal: np.ndarray, offset: float
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Cut ``mesh`` with the plane ``dot(p, normal) == offset``.

    Returns ``(positive_side, negative_side)``, both closed solids;
    ``positive_side`` is where ``dot(p, normal) > offset``. ``normal`` need
    not be a unit vector. A side that holds no material is an empty mesh.
    """
    normal = np.asarray(normal, dtype=np.float64).reshape(-1)
    length = float(np.linalg.norm(normal)) if normal.shape == (3,) else 0.0
    if not np.isfinite(length) or length == 0.0:
        raise ValueError(f"Plane normal must be a finite, non-zero 3-vector, got {normal!r}")
    if not np.isfinite(offset):
        raise ValueError(f"Plane offset must be finite, got {offset!r}")
    solid = _operand(mesh, "the mesh to split")
    # manifold3d normalises the normal and measures the offset along the unit
    # normal, so rescale the offset to keep the plane dot(p, normal) == offset.
    positive, negative = solid.split_by_plane(tuple(normal / length), float(offset) / length)
    _check_status(positive, "the split result", suggest_repair=False)
    _check_status(negative, "the split result", suggest_repair=False)
    return to_trimesh(positive), to_trimesh(negative)


def _tool_manifolds(tools: Iterable[trimesh.Trimesh]) -> list[Manifold]:
    solids = (_operand(tool, f"tool mesh {i + 1}") for i, tool in enumerate(tools))
    return [solid for solid in solids if not solid.is_empty()]


def _operand(mesh: trimesh.Trimesh, label: str) -> Manifold:
    """Convert ``mesh`` and merge any of its shells that overlap.

    Models assembled from several parts often store overlapping closed
    shells in one mesh. manifold3d accepts them, but booleans then keep the
    internal walls, so such shells are united first. Meshes with internal
    voids (inward-facing shells) are left as they are.
    """
    return merge_overlapping_shells(to_manifold(mesh, label))


def merge_overlapping_shells(solid: Manifold) -> Manifold:
    """Return ``solid`` with overlapping shells united, or ``solid`` itself if none overlap.

    Solids with internal voids (inward-facing shells) are returned as they are.
    """
    shells = solid.decompose()
    if len(shells) < 2 or any(shell.volume() < 0 for shell in shells):
        return solid
    boxes = np.array([shell.bounding_box() for shell in shells])  # (min xyz, max xyz)
    tree = trimesh.util.bounds_tree(boxes)
    # Each box intersects itself, so a single hit means the shell is apart from the rest.
    if all(len(list(tree.intersection(box))) == 1 for box in boxes):
        return solid
    return Manifold.batch_boolean(shells, OpType.Add)


def _run(operands: list[Manifold], op: OpType, name: str) -> trimesh.Trimesh:
    result = Manifold.batch_boolean(operands, op)
    _check_status(result, f"the {name} result", suggest_repair=False)
    return to_trimesh(result)


def _check_status(solid: Manifold, label: str, *, suggest_repair: bool) -> None:
    status = solid.status()
    if status == manifold3d.Error.NoError:
        return
    if status == manifold3d.Error.NotManifold:
        detail = (
            "is not a closed solid (not manifold): it has holes or edges shared by "
            "more than two faces"
        )
    elif status == manifold3d.Error.NonFiniteVertex:
        detail = "has vertices with NaN or infinite coordinates"
    else:
        detail = f"could not be converted to a solid (manifold3d reported {status.name})"
    raise BooleanError(_message(label, detail, suggest_repair))


def _message(label: str, detail: str, suggest_repair: bool) -> str:
    text = f"{label[:1].upper()}{label[1:]} {detail}"
    return f"{text}; {_REPAIR_HINT}" if suggest_repair else text
