"""Mesh cleanup so the part can be used as a boolean operand.

A mesh that manifold3d already accepts as a solid, with every shell facing
outwards or enclosing a void, is returned unchanged apart from dropping
faces with NaN coordinates; separate or touching bodies are never welded.
Anything else is repaired on a copy: drop unreferenced, duplicate and
degenerate elements; remove tiny loose debris; make the winding consistent;
fill holes (trimesh for small holes, then pymeshfix if installed, then
centroid fans); turn inside-out bodies right side out. The result is then
checked with manifold3d again.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import trimesh
from manifold3d import Manifold
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from moldgen.booleans import BooleanError, merge_overlapping_shells, to_manifold, to_trimesh

logger = logging.getLogger(__name__)

DEBRIS_AREA_FRACTION = 1e-3
"""Loose bodies with less surface area than this share of the largest body are discarded."""

VOLUME_CHANGE_WARNING = 0.03
"""Warn when repair changes a measurable enclosed volume by more than this fraction."""

# A closed surface whose volume is below this fraction of its bounding cube is flat.
_FLAT_VOLUME_FRACTION = 1e-9
# A pymeshfix result whose bounding-box diagonal or surface area shrinks by
# more than this fraction has deleted part of the model and is rejected.
_REBUILD_SHRINK_TOLERANCE = 0.05
# An inward-facing shell is a void when at least this share of it lies inside an outer shell.
_VOID_CONTAINED_FRACTION = 0.999

_PYMESHFIX_HINT = "installing the optional repair extra (pip install 'moldgen[repair]') may help"
_EDITOR_HINT = "fix the model in a mesh editor (for example Blender's 3D-Print Toolbox)"


@dataclass
class RepairReport:
    was_watertight: bool
    """Whether the input surface was closed."""
    is_watertight: bool
    """Whether the repaired surface is closed."""
    actions: list[str] = field(default_factory=list)
    """One short description per change that was made, in order."""
    bodies: int = 1
    """Number of separate solid bodies (internal voids are not counted)."""
    volume: float = 0.0
    """Enclosed volume of the repaired mesh; 0.0 when it is not a valid solid."""
    warnings: list[str] = field(default_factory=list)
    """When ``manifold_ok`` is False the last warning explains why."""
    manifold_ok: bool = False
    """True when manifold3d accepts the repaired mesh as a closed solid, so booleans will work."""


def repair_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, RepairReport]:
    """Return a cleaned copy of ``mesh`` and a report of what was done.

    The input is never modified. Only geometry is copied; colours and
    texture coordinates are dropped.
    """
    vertices = np.array(mesh.vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.array(mesh.faces, dtype=np.int64).reshape(-1, 3)
    in_range = ((faces >= 0) & (faces < len(vertices))).all(axis=1)
    work = trimesh.Trimesh(vertices=vertices, faces=faces[in_range], process=False)
    report = RepairReport(was_watertight=work.is_watertight, is_watertight=False)
    actions = report.actions

    if not in_range.all():
        report.was_watertight = False
        actions.append(
            f"removed {_count(int((~in_range).sum()), 'face')} referencing missing vertices"
        )
    _remove_non_finite(work, actions)
    _remove_unreferenced_vertices(work, actions)
    if _is_clean_solid(work):
        reference_volume = None
        _remove_debris(work, actions)
    else:
        reference_volume = _measurable_volume(work)
        _merge_duplicate_vertices(work, actions)
        _remove_bad_faces(work, actions)
        _remove_debris(work, actions)
        _fix_winding(work, actions)
        if not work.is_watertight:
            work = _fill_holes(work, actions)
        _fix_inversion(work, actions)
    work = _merge_overlapping_bodies(work, actions)

    _finish_report(work, report, reference_volume)
    return work, report


def _remove_non_finite(work: trimesh.Trimesh, actions: list[str]) -> None:
    finite = np.isfinite(work.vertices).all(axis=1)
    if finite.all():
        return
    keep = finite[work.faces].all(axis=1)
    if keep.all():
        actions.append(
            f"removed {_count(int((~finite).sum()), 'vertex', 'vertices')} "
            "with NaN or infinite coordinates"
        )
    else:
        actions.append(
            f"removed {_count(int((~keep).sum()), 'face')} with NaN or infinite coordinates"
        )
        work.update_faces(keep)
    work.update_vertices(finite)


def _is_clean_solid(work: trimesh.Trimesh) -> bool:
    """Whether manifold3d accepts ``work`` as is and every shell faces outwards."""
    solid, problem = _check_solid(work, "the part")
    if solid is None or problem is not None:
        return False
    return all(shell.volume() > 0 for shell in solid.decompose())


def _measurable_volume(work: trimesh.Trimesh) -> float | None:
    if len(work.faces) and work.is_watertight and work.is_winding_consistent:
        return abs(float(work.volume))
    return None


def _remove_unreferenced_vertices(work: trimesh.Trimesh, actions: list[str]) -> None:
    referenced = np.zeros(len(work.vertices), dtype=bool)
    referenced[work.faces] = True
    unused = int((~referenced).sum())
    if unused:
        work.remove_unreferenced_vertices()
        actions.append(f"removed {_count(unused, 'unreferenced vertex', 'unreferenced vertices')}")


def _merge_duplicate_vertices(work: trimesh.Trimesh, actions: list[str]) -> None:
    before = len(work.vertices)
    work.merge_vertices()
    merged = before - len(work.vertices)
    if merged:
        actions.append(f"merged {_count(merged, 'duplicate vertex', 'duplicate vertices')}")


def _remove_bad_faces(work: trimesh.Trimesh, actions: list[str]) -> None:
    faces = work.faces
    repeated = (
        (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    )
    remove = repeated
    zero_area = ~work.nondegenerate_faces() & ~repeated
    # A zero-area sliver often closes a T-junction; dropping it would open the surface.
    if zero_area.any() and _open_edge_count(faces[~(repeated | zero_area)]) <= _open_edge_count(
        faces[~repeated]
    ):
        remove = repeated | zero_area
    changed = bool(remove.any())
    if changed:
        work.update_faces(~remove)
        actions.append(f"removed {_count(int(remove.sum()), 'degenerate face')}")

    unique = work.unique_faces()
    if not unique.all():
        work.update_faces(unique)
        actions.append(f"removed {_count(int((~unique).sum()), 'duplicate face')}")
        changed = True
    if changed:
        work.remove_unreferenced_vertices()


def _remove_debris(work: trimesh.Trimesh, actions: list[str]) -> None:
    labels, count = _face_labels(work)
    if count < 2:
        return
    areas = np.bincount(labels, weights=work.area_faces, minlength=count)
    debris = areas < DEBRIS_AREA_FRACTION * areas.max()
    if not debris.any():
        return
    work.update_faces(~debris[labels])
    work.remove_unreferenced_vertices()
    actions.append(
        f"removed {_count(int(debris.sum()), 'tiny loose piece')} "
        f"(under {DEBRIS_AREA_FRACTION:.1%} of the largest body's surface area)"
    )


def _fix_winding(work: trimesh.Trimesh, actions: list[str]) -> None:
    """Make neighbouring faces agree on orientation, flipping as few faces as possible.

    Works on a double cover of the face graph: node ``f`` is face ``f`` as
    stored and node ``f + n`` is face ``f`` flipped. Two faces sharing an edge
    in opposite directions agree, so ``f`` links to ``g``; if they traverse it
    in the same direction one must flip, so ``f`` links to ``g + n``. Each
    orientable patch becomes two mirror components; keeping the one that
    holds more as-stored faces gives the fewest flips.
    """
    n = len(work.faces)
    first, second = _shared_edge_pairs(work)
    if len(first) == 0:
        return
    edges = work.edges
    f, g = first // 3, second // 3
    same_direction = edges[first, 0] == edges[second, 0]
    rows = np.concatenate((f, f + n))
    cols = np.concatenate((np.where(same_direction, g + n, g), np.where(same_direction, g, g + n)))
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(2 * n, 2 * n))
    _, labels = connected_components(graph, directed=False)
    as_stored, flipped = labels[:n], labels[n:]
    orientable = as_stored != flipped  # equal labels: a Moebius-like patch, left as is
    patch = np.minimum(as_stored, flipped)
    in_lower = as_stored == patch
    size = 2 * n
    lower_count = np.bincount(patch[orientable & in_lower], minlength=size)
    upper_count = np.bincount(patch[orientable & ~in_lower], minlength=size)
    flip_lower = lower_count < upper_count
    flip = orientable & np.where(in_lower, flip_lower[patch], ~flip_lower[patch])
    if flip.any():
        work.faces = np.where(flip[:, None], work.faces[:, ::-1], work.faces)
        actions.append(f"fixed inconsistent winding on {_count(int(flip.sum()), 'face')}")


def _merge_overlapping_bodies(work: trimesh.Trimesh, actions: list[str]) -> trimesh.Trimesh:
    """Unite closed bodies that overlap, so the volume and body count are not double counted."""
    solid, problem = _check_solid(work, "the part")
    if solid is None or problem is not None:
        return work
    merged = merge_overlapping_shells(solid)
    if merged is solid:
        return work
    before, after = len(solid.decompose()), len(merged.decompose())
    actions.append(f"merged overlapping bodies ({before} became {after})")
    return to_trimesh(merged)


def _fix_inversion(work: trimesh.Trimesh, actions: list[str]) -> None:
    """Flip closed shells that face inwards, unless they are voids inside another shell."""
    labels, count = _face_labels(work)
    if count == 0:
        return
    closed = np.bincount(labels, weights=~_closed_faces(work), minlength=count) == 0
    tri = work.triangles
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    volume = np.bincount(labels, weights=signed, minlength=count)
    inward = np.flatnonzero(closed & (volume < 0))
    if len(inward) == 0:
        return
    outward = np.flatnonzero(closed & (volume > 0))
    to_flip = [i for i in inward if not _is_void(work, labels, i, outward)]
    if not to_flip:
        return
    flip = np.isin(labels, to_flip)
    work.faces = np.where(flip[:, None], work.faces[:, ::-1], work.faces)
    actions.append(
        f"turned {_count(len(to_flip), 'inside-out body', 'inside-out bodies')} right side out"
    )


def _is_void(work: trimesh.Trimesh, labels: np.ndarray, shell: int, outer: np.ndarray) -> bool:
    inner_mesh = work.submesh([np.flatnonzero(labels == shell)], append=True)
    inner_mesh.invert()
    for candidate in outer:
        outer_mesh = work.submesh([np.flatnonzero(labels == candidate)], append=True)
        if not (
            np.all(outer_mesh.bounds[0] <= inner_mesh.bounds[0])
            and np.all(inner_mesh.bounds[1] <= outer_mesh.bounds[1])
        ):
            continue
        try:
            inner = to_manifold(inner_mesh, merge_vertices=False)
            shared = (to_manifold(outer_mesh, merge_vertices=False) ^ inner).volume()
        except BooleanError as exc:
            logger.debug("cannot test shell %d for being a void: %s", shell, exc)
            continue
        if shared >= _VOID_CONTAINED_FRACTION * inner.volume():
            return True
    return False


def _fill_holes(work: trimesh.Trimesh, actions: list[str]) -> trimesh.Trimesh:
    holes = _count_holes(work)
    try:
        trimesh.repair.fill_holes(work)
    except ImportError as exc:
        # trimesh needs networkx for this; the fan fill below covers all holes anyway.
        logger.debug("trimesh hole filling unavailable: %s", exc)
    remaining = _count_holes(work)
    if remaining < holes:
        actions.append(f"filled {_count(holes - remaining, 'small hole')}")
    if work.is_watertight:
        return work

    work = _repair_open_bodies_with_pymeshfix(work, actions)
    if work.is_watertight:
        return work
    return _fill_holes_with_fans(work, actions)


def _repair_open_bodies_with_pymeshfix(
    work: trimesh.Trimesh, actions: list[str]
) -> trimesh.Trimesh:
    try:
        import pymeshfix
    except ImportError:
        return work

    labels, count = _face_labels(work)
    bodies = [work.submesh([np.flatnonzero(labels == i)], append=True) for i in range(count)]
    rebuilt = 0
    for i, body in enumerate(bodies):
        if body.is_watertight:
            continue
        # pymeshfix keeps only the largest component by default; each call
        # here gets exactly one body, so nothing else may be thrown away.
        with _native_output_silenced():
            vertices, faces = pymeshfix.clean_from_arrays(
                np.array(body.vertices, dtype=np.float64, order="C"),
                np.array(body.faces, dtype=np.int32, order="C"),
                remove_smallest_components=False,
            )
        candidate = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if _is_plausible_rebuild(body, candidate):
            bodies[i] = candidate
            rebuilt += 1
    if not rebuilt:
        return work
    actions.append(f"rebuilt {_count(rebuilt, 'open body', 'open bodies')} with pymeshfix")
    return trimesh.util.concatenate(bodies)


def _is_plausible_rebuild(original: trimesh.Trimesh, rebuilt: trimesh.Trimesh) -> bool:
    """Accept a pymeshfix result only if it is a valid solid of about the same shape."""
    if len(rebuilt.faces) == 0 or not (rebuilt.is_watertight and rebuilt.is_winding_consistent):
        return False
    if rebuilt.volume < 0:
        rebuilt.invert()
    keep = 1.0 - _REBUILD_SHRINK_TOLERANCE
    if np.linalg.norm(rebuilt.extents) < keep * np.linalg.norm(original.extents):
        return False
    if rebuilt.area < keep * original.area:
        return False
    _, problem = _check_solid(rebuilt, "the pymeshfix result")
    return problem is None


def _fill_holes_with_fans(work: trimesh.Trimesh, actions: list[str]) -> trimesh.Trimesh:
    """Close every boundary loop with a fan of triangles around the loop's centroid.

    A centroid fan is valid for any hole that is star-shaped around its
    centroid, which covers most holes in scanned or exported models.
    """
    boundary = _boundary_edges(work)
    if len(boundary) == 0:
        return work
    loop_of_vertex, loops = _boundary_loops(boundary, len(work.vertices))
    on_boundary = loop_of_vertex >= 0
    weights = np.bincount(loop_of_vertex[on_boundary], minlength=loops)
    centroids = (
        np.column_stack(
            [
                np.bincount(loop_of_vertex[on_boundary], work.vertices[on_boundary, axis], loops)
                for axis in range(3)
            ]
        )
        / weights[:, None]
    )
    # A boundary edge u->v of a consistently wound surface is matched by a new
    # face v->u->centroid, so the fan inherits the surrounding winding.
    u, v = boundary[:, 0], boundary[:, 1]
    fan = np.column_stack((v, u, len(work.vertices) + loop_of_vertex[u]))
    filled = trimesh.Trimesh(
        vertices=np.vstack((work.vertices, centroids)),
        faces=np.vstack((work.faces, fan)),
        process=False,
    )
    closed = loops - _count_holes(filled)
    if closed <= 0:
        return work
    actions.append(f"closed {_count(closed, 'larger hole')} with triangle fans")
    return filled


def _finish_report(
    work: trimesh.Trimesh, report: RepairReport, reference_volume: float | None
) -> None:
    report.is_watertight = work.is_watertight
    report.bodies = _face_labels(work)[1]
    label = "the repaired part" if report.actions else "the part"
    solid, problem = _check_solid(work, label)
    if solid is None or problem is not None:
        hint = _EDITOR_HINT if _pymeshfix_available() else _PYMESHFIX_HINT
        report.warnings.append(f"{problem}, so no mold can be built from it; {hint}")
        return

    report.manifold_ok = True
    report.volume = solid.volume()
    shell_volumes = np.array([shell.volume() for shell in solid.decompose()])
    report.bodies = int((shell_volumes > 0).sum())
    voids = int((shell_volumes < 0).sum())
    if reference_volume:
        change = report.volume / reference_volume - 1.0
        if abs(change) > VOLUME_CHANGE_WARNING:
            report.warnings.append(
                f"Repair changed the enclosed volume from {reference_volume:.6g} to "
                f"{report.volume:.6g} ({change:+.1%}); check the repaired part before casting"
            )
    if report.bodies > 1:
        report.warnings.append(
            f"The part consists of {report.bodies} separate bodies; all of them are kept "
            "and cast in the same mold"
        )
    if voids:
        report.warnings.append(
            f"The part encloses {_count(voids, 'sealed internal void')}, which a mold cannot "
            "reproduce; the cast will be solid there"
        )


def _check_solid(mesh: trimesh.Trimesh, label: str) -> tuple[Manifold | None, str | None]:
    """Return manifold3d's view of ``mesh`` as a solid, or a reason why it is not one."""
    if len(mesh.faces) == 0:
        return None, "No usable faces are left; the file holds no solid"
    try:
        solid = to_manifold(mesh, label, suggest_repair=False, merge_vertices=False)
    except BooleanError as exc:
        return None, str(exc)
    if solid.volume() <= _FLAT_VOLUME_FRACTION * float(np.max(mesh.extents)) ** 3:
        return solid, (
            f"{label[:1].upper()}{label[1:]} encloses no volume "
            "(it is a flat or open surface, not a solid)"
        )
    return solid, None


@contextlib.contextmanager
def _native_output_silenced() -> Iterator[None]:
    """Discard text that native code writes straight to the stdout and stderr descriptors.

    pymeshfix prints progress messages from C++ that Python cannot capture.
    The redirection is process wide, so keep the block short.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    try:
        saved = (os.dup(1), os.dup(2))
    except OSError:  # no standard descriptors, for example in a windowed app
        saved = None
    if saved is None:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


def _pymeshfix_available() -> bool:
    try:
        import pymeshfix  # noqa: F401
    except ImportError:
        return False
    return True


def _face_labels(mesh: trimesh.Trimesh) -> tuple[np.ndarray, int]:
    """Label each face with the index of its edge-connected body; return labels and body count."""
    n = len(mesh.faces)
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    pairs = mesh.face_adjacency
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), (n, n))
    count, labels = connected_components(graph, directed=False)
    return labels, count


def _shared_edge_pairs(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Indices into ``mesh.edges`` of the two half-edges of every edge used by exactly two faces."""
    pairs = np.asarray(trimesh.grouping.group_rows(mesh.edges_sorted, require_count=2))
    pairs = pairs.reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


def _closed_faces(mesh: trimesh.Trimesh) -> np.ndarray:
    """Faces whose every edge is shared with exactly one other face, in opposite direction."""
    first, second = _shared_edge_pairs(mesh)
    edges = mesh.edges
    opposite = edges[first, 0] == edges[second, 1]
    good = np.zeros(len(edges), dtype=bool)
    good[first[opposite]] = True
    good[second[opposite]] = True
    return good.reshape(-1, 3).all(axis=1)


def _boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """Directed edges, as wound in their face, that belong to only one face."""
    single = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    return mesh.edges[single]


def _boundary_loops(boundary: np.ndarray, vertex_count: int) -> tuple[np.ndarray, int]:
    """Label each vertex with its boundary loop (-1 off the boundary); return labels and count."""
    graph = coo_matrix(
        (np.ones(len(boundary), dtype=np.int8), (boundary[:, 0], boundary[:, 1])),
        (vertex_count, vertex_count),
    )
    _, labels = connected_components(graph, directed=False)
    on_boundary = np.zeros(vertex_count, dtype=bool)
    on_boundary[boundary.ravel()] = True
    loop_ids, compact = np.unique(labels[on_boundary], return_inverse=True)
    loop_of_vertex = np.full(vertex_count, -1, dtype=np.int64)
    loop_of_vertex[on_boundary] = compact
    return loop_of_vertex, len(loop_ids)


def _count_holes(mesh: trimesh.Trimesh) -> int:
    boundary = _boundary_edges(mesh)
    if len(boundary) == 0:
        return 0
    return _boundary_loops(boundary, len(mesh.vertices))[1]


def _open_edge_count(faces: np.ndarray) -> int:
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def _count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else plural or singular + 's'}"
