"""Generate 3D-printable casting molds from 3D models."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "1.0.0.dev0"

_EXPORTS = {
    "ConfigError": "moldgen.config",
    "MoldConfig": "moldgen.config",
    "MATERIALS": "moldgen.materials",
    "PRINT_MATERIALS": "moldgen.materials",
    "Material": "moldgen.materials",
    "PrintMaterial": "moldgen.materials",
    "PartingResult": "moldgen.parting",
    "analyze_parting": "moldgen.parting",
    "MoldError": "moldgen.pipeline",
    "MoldPiece": "moldgen.pipeline",
    "MoldResult": "moldgen.pipeline",
    "PreparedPart": "moldgen.pipeline",
    "generate_mold": "moldgen.pipeline",
    "prepare_part": "moldgen.pipeline",
}

__all__ = [
    "MATERIALS",
    "PRINT_MATERIALS",
    "ConfigError",
    "Material",
    "MoldConfig",
    "MoldError",
    "MoldPiece",
    "MoldResult",
    "PartingResult",
    "PreparedPart",
    "PrintMaterial",
    "__version__",
    "analyze_parting",
    "generate_mold",
    "prepare_part",
]

if TYPE_CHECKING:
    from moldgen.config import ConfigError, MoldConfig
    from moldgen.materials import MATERIALS, PRINT_MATERIALS, Material, PrintMaterial
    from moldgen.parting import PartingResult, analyze_parting
    from moldgen.pipeline import (
        MoldError,
        MoldPiece,
        MoldResult,
        PreparedPart,
        generate_mold,
        prepare_part,
    )


def __getattr__(name: str) -> Any:
    # Importing the geometry stack takes about a second, so it is deferred
    # until something from it is actually used.
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'moldgen' has no attribute {name!r}")
