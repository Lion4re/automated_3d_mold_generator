"""User-facing configuration for mold generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

Units = Literal["mm", "cm", "m", "in"]

UNIT_TO_MM: dict[str, float] = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}

SUPPORTED_SUFFIXES: tuple[str, ...] = (".stl", ".obj", ".ply", ".off", ".3mf", ".glb", ".gltf")
"""Model file types that can be loaded."""

AXIS_VECTORS: dict[str, tuple[float, float, float]] = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
    "-x": (-1.0, 0.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "-z": (0.0, 0.0, -1.0),
}


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


@dataclass
class MoldConfig:
    """All knobs for a mold generation run. Lengths are in millimetres.

    ``None`` means "derive automatically" (usually from the material preset
    and the part size).
    """

    material: str = "resin"
    """Casting material preset key, see :mod:`moldgen.materials`."""

    print_material: str = "pla"
    """Material the mold itself will be printed in; used for temperature checks."""

    units: Units = "mm"
    """Units of the input file. Everything is converted to millimetres on load."""

    scale: float = 1.0
    """Extra uniform scale applied to the part after unit conversion."""

    direction: str | Sequence[float] = "auto"
    """Demolding direction: "auto", an axis name ("x", "-z", ...) or a 3-vector."""

    parting_offset: float | None = None
    """Parting plane position along ``direction`` in the input frame (mm).
    ``None`` picks the position with the least undercut area."""

    pieces: int = 2
    """2 for a classic two-part mold, 4 to also split each half once more."""

    wall_thickness: float | None = None
    """Minimum distance from the part's bounding box to the outside of the mold."""

    shrinkage: float | None = None
    """Linear shrinkage of the casting material as a fraction (0.01 = 1 %).
    The cavity is enlarged to compensate. Overrides the material preset."""

    sprue_diameter: float | None = None
    funnel: bool = True
    vents: bool = True
    vent_diameter: float | None = None

    keys: int = 4
    """Number of registration keys on the main parting face."""

    key_diameter: float | None = None
    clearance: float = 0.2
    """Gap per side (mm) between mating key surfaces. 0.2 suits most FDM printers;
    use about 0.1 for resin (SLA) printers."""

    draft_threshold_deg: float = 2.0
    """Faces with less draft than this are reported as low-draft (2 degrees is a common minimum)."""

    repair: bool = True
    orient_for_print: bool = True
    """Rotate each exported piece so its parting face points up (no supports needed)."""

    extra: dict[str, object] = field(default_factory=dict)
    """Free-form values for experimental options; ignored by the core pipeline."""

    def direction_vector(self) -> np.ndarray | None:
        """Return the requested demolding direction as a unit vector, or None for auto."""
        d = self.direction
        if isinstance(d, str):
            key = d.strip().lower()
            if key == "auto":
                return None
            if key not in AXIS_VECTORS:
                raise ConfigError(
                    f"Unknown direction {d!r}; use auto, x, y, z, -x, -y, -z or a vector"
                )
            return np.asarray(AXIS_VECTORS[key], dtype=float)
        vec = np.asarray(d, dtype=float).reshape(-1)
        if vec.shape != (3,) or not np.all(np.isfinite(vec)) or np.linalg.norm(vec) < 1e-9:
            raise ConfigError(f"Direction vector must be three finite numbers, got {d!r}")
        return vec / np.linalg.norm(vec)

    def validate(self) -> None:
        """Raise :class:`ConfigError` if any value is out of range."""
        from moldgen.materials import MATERIALS, PRINT_MATERIALS

        if self.material not in MATERIALS:
            raise ConfigError(
                f"Unknown material {self.material!r}; choose one of {', '.join(sorted(MATERIALS))}"
            )
        if self.print_material not in PRINT_MATERIALS:
            raise ConfigError(
                f"Unknown print material {self.print_material!r}; "
                f"choose one of {', '.join(sorted(PRINT_MATERIALS))}"
            )
        if self.units not in UNIT_TO_MM:
            raise ConfigError(f"Unknown units {self.units!r}; use mm, cm, m or in")
        if not self.scale > 0:
            raise ConfigError("Scale must be positive")
        if self.pieces not in (2, 4):
            raise ConfigError("Pieces must be 2 or 4")
        if self.wall_thickness is not None and not self.wall_thickness > 0:
            raise ConfigError("Wall thickness must be positive")
        if self.shrinkage is not None and not -0.05 <= self.shrinkage < 0.2:
            raise ConfigError(
                "Shrinkage must be a fraction between -0.05 and 0.2 (e.g. 0.01 for 1 %)"
            )
        for name in ("sprue_diameter", "vent_diameter", "key_diameter"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ConfigError(f"{name.replace('_', ' ').capitalize()} must be positive")
        if not 0 <= self.keys <= 8:
            raise ConfigError("Keys must be between 0 and 8")
        if not 0 <= self.clearance <= 2:
            raise ConfigError("Clearance must be between 0 and 2 mm")
        if not 0 <= self.draft_threshold_deg < 45:
            raise ConfigError("Draft threshold must be between 0 and 45 degrees")
        self.direction_vector()

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if not isinstance(self.direction, str):
            data["direction"] = [float(v) for v in self.direction]
        return data
