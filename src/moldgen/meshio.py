"""Loading and saving meshes."""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path
from typing import NoReturn

import numpy as np
import trimesh

from moldgen.config import SUPPORTED_SUFFIXES, UNIT_TO_MM, Units

__all__ = [
    "EXPORT_SUFFIXES",
    "SUPPORTED_SUFFIXES",
    "MeshExportError",
    "MeshLoadError",
    "export_mesh",
    "load_mesh",
    "units_warning",
]

EXPORT_SUFFIXES: tuple[str, ...] = (".stl", ".obj", ".ply", ".off", ".3mf", ".glb")
"""Single-file formats :func:`export_mesh` can write (``.gltf`` needs side files)."""

SMALL_PART_MM = 2.0
"""Parts smaller than this were most likely modelled in metres or centimetres."""

LARGE_PART_MM = 1500.0
"""Parts larger than this exceed any desktop printer and suggest a unit mix-up."""

# Exceptions trimesh's parsers raise for malformed or truncated files.
_PARSE_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    KeyError,
    IndexError,
    EOFError,
    struct.error,
    zipfile.BadZipFile,
)


class MeshLoadError(ValueError):
    """Raised when a file cannot be read as a triangle mesh."""


class MeshExportError(ValueError):
    """Raised when a mesh cannot be written in the requested format."""


def load_mesh(path: str | Path, units: Units = "mm", scale: float = 1.0) -> trimesh.Trimesh:
    """Load a mesh file, merge scenes into one mesh and convert to millimetres.

    Multi-object files (scenes) are flattened with their transforms applied.
    The result is a new geometry-only mesh with duplicate vertices merged.
    """
    if units not in UNIT_TO_MM:
        raise ValueError(f"Unknown units {units!r}; use one of {', '.join(UNIT_TO_MM)}")
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"Scale must be a positive number, got {scale!r}")

    path = Path(path).expanduser()
    if not path.exists():
        raise MeshLoadError(f"File not found: {path}")
    if not path.is_file():
        raise MeshLoadError(f"Not a file: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise MeshLoadError(
            f"Unsupported file type {suffix or '(no extension)'!r} for {path.name}; "
            f"supported types are {', '.join(SUPPORTED_SUFFIXES)}"
        )

    try:
        loaded = trimesh.load(path, file_type=suffix.lstrip("."))
    except ImportError as exc:
        raise MeshLoadError(
            f"Reading {suffix} files needs an optional dependency that is not installed ({exc})"
        ) from exc
    except FileNotFoundError as exc:
        # The file itself exists, so the parser could not find a file it refers
        # to (a glTF buffer or texture), or it gave up on unreadable JSON.
        raise MeshLoadError(
            f"{path.name} is not a valid {suffix} file or refers to a missing file "
            f"({Path(exc.filename or str(exc)).name})"
        ) from exc
    except OSError as exc:
        raise MeshLoadError(f"Could not read {path}: {exc.strerror or exc}") from exc
    except _PARSE_ERRORS as exc:
        raise MeshLoadError(f"{path.name} is not a valid {suffix} file ({exc})") from exc

    mesh = _as_single_mesh(loaded, path.name)
    factor = UNIT_TO_MM[units] * scale
    # A fresh mesh drops visuals and merges vertices by position only, so
    # texture seams in OBJ/glTF files do not split the surface open.
    return trimesh.Trimesh(vertices=mesh.vertices * factor, faces=mesh.faces, process=True)


def units_warning(mesh: trimesh.Trimesh) -> str | None:
    """Return a warning when the part's size suggests the wrong units were assumed."""
    if len(mesh.vertices) == 0:
        return None
    size = float(np.ptp(mesh.vertices, axis=0).max())
    if size < SMALL_PART_MM:
        return (
            f"The part is only {size:.3g} mm across, so the file is probably in metres, "
            "centimetres or inches; set the units (m, cm or in) to match."
        )
    if size > LARGE_PART_MM:
        return (
            f"The part is {size:.0f} mm across, too big for a desktop printer; check the "
            "units, or reduce the scale."
        )
    return None


def export_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Write ``mesh`` to ``path`` and return the path written.

    The format follows the suffix; a path without a suffix gets ``.stl``.
    STL files are binary. Parent directories are created as needed.
    """
    path = Path(path).expanduser()
    if not path.suffix:
        path = path.with_suffix(".stl")
    suffix = path.suffix.lower()
    if suffix not in EXPORT_SUFFIXES:
        raise MeshExportError(
            f"Cannot export to {suffix!r}; supported types are {', '.join(EXPORT_SUFFIXES)}"
        )
    # Serialize before touching the file system so a failure leaves no partial file.
    try:
        data = mesh.export(file_type=suffix.lstrip("."))
    except ImportError as exc:
        raise MeshExportError(
            f"Writing {suffix} files needs an optional dependency that is not installed ({exc})"
        ) from exc
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            path.write_bytes(data)
    except OSError as exc:
        raise MeshExportError(f"Could not write {path}: {exc.strerror or exc}") from exc
    return path


def _as_single_mesh(loaded: object, name: str) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Scene):
        geometry = list(loaded.geometry.values())
        if geometry and not any(isinstance(g, trimesh.Trimesh) for g in geometry):
            _raise_not_a_surface(geometry[0], name)
        mesh = loaded.to_mesh()
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        _raise_not_a_surface(loaded, name)
    if len(mesh.faces) == 0:
        raise MeshLoadError(f"{name} contains no triangles (the file is empty or unreadable)")
    return mesh


def _raise_not_a_surface(geometry: object, name: str) -> NoReturn:
    if isinstance(geometry, trimesh.PointCloud):
        raise MeshLoadError(
            f"{name} contains only points (a point cloud), not a surface; "
            "reconstruct a triangle mesh from it first"
        )
    raise MeshLoadError(f"{name} contains {type(geometry).__name__} data, not a triangle mesh")
