"""The ``moldgen`` command-line interface.

``moldgen make`` builds molds, ``moldgen analyze`` inspects a model without
writing files, ``moldgen materials`` lists the presets and ``moldgen gui``
starts the browser interface. Running ``moldgen`` alone in a terminal starts a
short guided setup.
"""

from __future__ import annotations

import glob
import logging
import math
import re
import shlex
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
import typer
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from moldgen import __version__
from moldgen.config import AXIS_VECTORS, SUPPORTED_SUFFIXES, ConfigError, MoldConfig, Units
from moldgen.materials import (
    MATERIALS,
    PRINT_MATERIALS,
    compatibility_warnings,
    get_material,
    get_print_material,
)

# The geometry modules import trimesh and manifold3d, which take seconds to load
# on a cold start. They are imported inside the functions that do the work, so
# help, --version and 'materials' stay fast.
if TYPE_CHECKING:
    from moldgen.parting import DirectionScore, PartingResult
    from moldgen.pipeline import PreparedPart, ProgressCallback

ISSUES_URL = "https://github.com/Lion4re/automated_3d_mold_generator/issues"
DEFAULT_OUT = Path("molds")
FORMATS = ", ".join(SUPPORTED_SUFFIXES)
MAX_CANDIDATES_SHOWN = 5
MAX_MODELS_LISTED = 20

_DEFAULTS = MoldConfig()

# Choice types are built from the presets so new presets appear automatically.
MaterialKey = Enum("MaterialKey", {key: key for key in MATERIALS}, type=str)
PrintMaterialKey = Enum("PrintMaterialKey", {key: key for key in PRINT_MATERIALS}, type=str)
Pieces = Literal["auto", "2", "4"]

_DEFAULT_MATERIAL = MaterialKey(_DEFAULTS.material)
_DEFAULT_PRINT_MATERIAL = PrintMaterialKey(_DEFAULTS.print_material)

_INPUT = "Input"
_MATERIAL = "Material"
_PARTING = "Parting"
_MOLD = "Mold"
_OUTPUT = "Output"


def _option(*names: str, help: str, panel: str, default: str | None = None, **kwargs: Any) -> Any:
    """typer.Option in a help panel; ``default`` replaces the shown default (e.g. "auto")."""
    return typer.Option(
        *names, help=help, rich_help_panel=panel, show_default=default or True, **kwargs
    )


MaterialOption = Annotated[
    MaterialKey,
    _option(
        "--material",
        "-m",
        metavar="KEY",
        help="Casting material; see 'moldgen materials' for the keys.",
        panel=_MATERIAL,
    ),
]
PrintMaterialOption = Annotated[
    PrintMaterialKey,
    _option(
        "--print-material",
        "-p",
        metavar="KEY",
        help="Material the mold is printed in; see 'moldgen materials'.",
        panel=_MATERIAL,
    ),
]
DirectionOption = Annotated[
    str,
    _option(
        "--direction",
        "-d",
        metavar="AXIS",
        help="Demolding direction: auto, x, y, z, -x, -y, -z or a vector such as 0,1,1.",
        panel=_PARTING,
    ),
]
UnitsOption = Annotated[Units, _option("--units", help="Units of the model file.", panel=_INPUT)]
ScaleOption = Annotated[
    float, _option("--scale", help="Scale factor applied after unit conversion.", panel=_INPUT)
]
NoRepairOption = Annotated[
    bool, _option("--no-repair", help="Use the mesh as is, without automatic repair.", panel=_INPUT)
]
JsonOption = Annotated[
    bool, _option("--json", help="Print the result as JSON instead of text.", panel=_OUTPUT)
]
VerboseOption = Annotated[
    bool, _option("--verbose", "-v", help="Show debug logging and tracebacks.", panel=_OUTPUT)
]

app = typer.Typer(
    name="moldgen",
    add_completion=False,
    invoke_without_command=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=(
        "Examples:\n\n"
        "moldgen make pawn.stl\n\n"
        "moldgen make models/*.stl --pieces 4 --out molds\n\n"
        "moldgen analyze pawn.stl"
    ),
)


class CliError(Exception):
    """An expected problem with a one-line message and an optional hint."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.hint = hint


@dataclass
class Ui:
    """Console output for one command, honouring --quiet, --json and --verbose."""

    quiet: bool = False
    as_json: bool = False
    verbose: bool = False
    out: Console = field(
        default_factory=lambda: Console(highlight=False, markup=False, emoji=False)
    )
    err: Console = field(
        default_factory=lambda: Console(stderr=True, highlight=False, markup=False, emoji=False)
    )

    @property
    def chatty(self) -> bool:
        """True when the normal human-readable output should be printed."""
        return not (self.quiet or self.as_json)

    def warning(self, message: str, subject: str | None = None) -> None:
        if self.as_json:
            return
        prefix = f"Warning ({subject}): " if subject else "Warning: "
        console = self.out if self.chatty else self.err
        console.print(Text(prefix + message, style="yellow"), soft_wrap=True)

    def error(self, message: str, hint: str | None = None) -> None:
        self.err.print(Text(f"Error: {message}", style="red"), soft_wrap=True)
        if hint:
            self.err.print(Text(f"  {hint}"), soft_wrap=True)

    @contextmanager
    def progress(self, label: str) -> Iterator[ProgressCallback | None]:
        """Show a progress bar driven by the pipeline's stage callback."""
        if not (self.chatty and self.err.is_terminal):
            yield None
            return
        columns = (
            SpinnerColumn(),
            TextColumn("{task.fields[label]}", style="bold"),
            TextColumn("{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
        )
        with Progress(*columns, console=self.err, transient=True) as progress:
            task = progress.add_task("Starting", total=1.0, label=label)

            def update(stage: str, fraction: float) -> None:
                progress.update(task, description=stage, completed=fraction)

            yield update

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        if self.chatty and self.err.is_terminal:
            with self.err.status(message):
                yield
        else:
            yield


@contextmanager
def _logging_to(
    console: Console, verbose: bool, *, info_from: Sequence[str] = ()
) -> Iterator[None]:
    """Send ``moldgen`` log records to ``console`` while a command runs.

    Shows warnings and errors, or everything with ``verbose``. Loggers named in
    ``info_from`` also show INFO records.
    """
    root = logging.getLogger("moldgen")
    levels = {root: logging.DEBUG if verbose else logging.WARNING}
    for name in info_from:
        levels[logging.getLogger(name)] = logging.DEBUG if verbose else logging.INFO
    previous = {logger: logger.level for logger in levels}
    from rich.logging import RichHandler

    handler = RichHandler(console=console, show_time=verbose, show_path=verbose)
    for logger, level in levels.items():
        logger.setLevel(level)
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        for logger, level in previous.items():
            logger.setLevel(level)


def _explain(exc: BaseException) -> tuple[str, str | None]:
    """Return a one-line message and an optional hint for ``exc``."""
    if isinstance(exc, CliError):
        return str(exc), exc.hint
    if isinstance(exc, ConfigError):
        return str(exc), "Run 'moldgen make --help' to see the valid values."
    # Errors from the geometry stack; it is already loaded when one of them is raised.
    from moldgen.booleans import BooleanError
    from moldgen.meshio import MeshLoadError
    from moldgen.pipeline import MoldError

    if isinstance(exc, MeshLoadError):
        return str(exc), f"Check that the file is a valid mesh. Supported formats: {FORMATS}."
    if isinstance(exc, MoldError):
        return str(exc), None
    if isinstance(exc, BooleanError):
        return str(exc), "Run again with -v for details."
    if isinstance(exc, OSError) and exc.strerror:
        where = f" {exc.filename}" if exc.filename else ""
        return f"Could not access{where}: {exc.strerror}", None
    detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
    return (
        f"Unexpected error ({detail})",
        f"Run again with -v for the full traceback, and please report it at {ISSUES_URL}",
    )


def _report_error(ui: Ui, exc: BaseException, model: Path | None = None) -> str:
    """Print ``exc`` as a red line plus hint (and a traceback with -v); return the message.

    In batch runs ``model`` names the file the error belongs to, unless the message
    already mentions it.
    """
    message, hint = _explain(exc)
    named = model is not None and (model.name in message or model.stem in message.split())
    prefix = f"{model}: " if model is not None and not named else ""
    ui.error(prefix + message, hint)
    if ui.verbose:
        from rich.traceback import Traceback

        ui.err.print(Traceback.from_exception(type(exc), exc, exc.__traceback__))
    return message


def _check_model(path: Path) -> Path:
    if not path.exists():
        raise CliError(f"File not found: {path}", f"Check the path. Supported formats: {FORMATS}.")
    if path.is_dir():
        raise CliError(f"{path} is a folder, not a model file", f"Supported formats: {FORMATS}.")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        kind = f"'{path.suffix}'" if path.suffix else "without an extension"
        raise CliError(
            f"Unsupported file type {kind}: {path}", f"Export the model as one of: {FORMATS}."
        )
    return path


def _key(choice: Enum | str) -> str:
    return choice.value if isinstance(choice, Enum) else choice


def _parse_direction(text: str) -> str | tuple[float, float, float]:
    """Turn 'auto', 'x', '+x', '-z' or '0,1,1' into a MoldConfig direction."""
    key = text.strip().lower().removeprefix("+")
    if key == "auto" or key in AXIS_VECTORS:
        return key
    try:
        values = tuple(float(part) for part in key.replace(",", " ").split())
    except ValueError:
        values = ()
    if len(values) != 3 or not all(math.isfinite(v) for v in values) or not any(values):
        raise typer.BadParameter(
            f"{text!r} is not a direction; use auto, x, y, z, -x, -y, -z or three numbers "
            "such as 0,1,1",
            param_hint="'--direction'",
        )
    return (values[0], values[1], values[2])


def _parse_bed(text: str) -> tuple[float, float, float]:
    """Turn '220x220x250' (or '220,220,250') into a build volume in mm."""
    try:
        values = tuple(float(part) for part in re.split(r"[x, ]+", text.strip().lower()))
    except ValueError:
        values = ()
    if len(values) != 3:
        raise typer.BadParameter(
            f"{text!r} is not a build volume; give width, depth and height in mm, "
            "such as 220x220x250",
            param_hint="'--bed'",
        )
    return (values[0], values[1], values[2])


def _direction_text(vector: Sequence[float] | np.ndarray) -> str:
    """'+Z' / '-X' for axis directions, otherwise 'x,y,z'; accepted back by --direction."""
    v = np.asarray(vector, dtype=float)
    axis = int(np.argmax(np.abs(v)))
    if abs(v[axis]) > 0.999:
        return ("+" if v[axis] > 0 else "-") + "XYZ"[axis]
    return ",".join(f"{c:.3f}" for c in v)


def _build_config(
    *,
    material: Enum | str = _DEFAULTS.material,
    print_material: Enum | str = _DEFAULTS.print_material,
    bed: str | None = None,
    units: Units = _DEFAULTS.units,
    scale: float = _DEFAULTS.scale,
    direction: str = "auto",
    parting_offset: float | None = None,
    pieces: int | str = _DEFAULTS.pieces,
    max_pieces: int = _DEFAULTS.max_pieces,
    parting_surface: str = _DEFAULTS.parting_surface,
    side_piece_cuts: str = _DEFAULTS.side_piece_cuts,
    wall: float | None = None,
    shrinkage_percent: float | None = None,
    sprue: float | None = None,
    vent: float | None = None,
    funnel: bool = True,
    vents: bool = True,
    keys: int = _DEFAULTS.keys,
    key_diameter: float | None = None,
    clearance: float = _DEFAULTS.clearance,
    repair: bool = True,
    orient: bool = True,
) -> MoldConfig:
    """Map command-line values onto a validated MoldConfig; invalid values are usage errors."""
    config = MoldConfig(
        material=_key(material),
        print_material=_key(print_material),
        units=units,
        scale=scale,
        direction=_parse_direction(direction),
        parting_offset=parting_offset,
        pieces=int(pieces) if str(pieces).isdigit() else pieces,
        max_pieces=max_pieces,
        parting_surface=parting_surface,
        side_piece_cuts=side_piece_cuts,
        wall_thickness=wall,
        shrinkage=None if shrinkage_percent is None else shrinkage_percent / 100.0,
        sprue_diameter=sprue,
        vent_diameter=vent,
        funnel=funnel,
        vents=vents,
        keys=keys,
        key_diameter=key_diameter,
        clearance=clearance,
        repair=repair,
        orient_for_print=orient,
        bed_size_mm=_DEFAULTS.bed_size_mm if bed is None else _parse_bed(bed),
    )
    try:
        config.validate()
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return config


def _count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _size(values: Sequence[float]) -> str:
    return " x ".join(f"{v:.1f}" for v in values) + " mm"


def _mm(value: float) -> str:
    # Adding 0.0 turns -0.0 into 0.0, so a plane at -0.001 is not shown as -0.00.
    return f"{round(value, 2) + 0.0:.2f} mm"


def _details_table() -> Table:
    table = Table(box=None, show_header=False, pad_edge=True, padding=(0, 3, 0, 1))
    table.add_column(style="dim")
    table.add_column()
    return table


# make


@dataclass
class _Outcome:
    model: Path
    out_dir: Path
    summary: dict[str, Any] | None = None
    files: list[Path] = field(default_factory=list)
    seconds: float = 0.0
    error: str | None = None

    def as_json(self) -> dict[str, Any]:
        if self.summary is None:
            return {"model": str(self.model), "error": self.error}
        return {
            "model": str(self.model),
            "output_dir": str(self.out_dir),
            "files": [str(path) for path in self.files],
            **self.summary,
        }


def _has_files(folder: Path) -> bool:
    return folder.is_dir() and any(folder.iterdir())


def _remove_previous_outputs(out_dir: Path, name: str) -> None:
    """Delete files from an earlier run so stale pieces do not linger next to new ones."""
    stale = [
        *out_dir.glob(f"{glob.escape(name)}_*.stl"),
        out_dir / "report.json",
        out_dir / "INSTRUCTIONS.txt",
    ]
    for path in stale:
        if path.is_file():
            path.unlink()


def _run_make(ui: Ui, models: Sequence[Path], config: MoldConfig, out: Path, force: bool) -> int:
    """Generate one mold per model, continuing past failures. Returns the exit code."""
    models = list(dict.fromkeys(models))
    seen: dict[str, Path] = {}
    for model in models:
        if (other := seen.setdefault(model.stem, model)) is not model:
            raise typer.BadParameter(
                f"{other} and {model} would both be saved to {out / model.stem}; "
                "run them separately with different --out folders",
                param_hint="MODEL",
            )

    batch = len(models) > 1
    outcomes = [_Outcome(model, out / model.stem) for model in models]
    pending: list[_Outcome] = []
    for outcome in outcomes:
        try:
            _check_model(outcome.model)
        except CliError as exc:
            outcome.error = _report_error(ui, exc)
        else:
            pending.append(outcome)

    busy = [str(o.out_dir) for o in pending if _has_files(o.out_dir)]
    if busy and not force:
        where = busy[0] if len(busy) == 1 else f"{len(busy)} output folders ({', '.join(busy)})"
        ui.error(
            f"{where} already {'contains' if len(busy) == 1 else 'contain'} files",
            "Add --force to overwrite earlier results, or choose another folder with --out.",
        )
        return 1

    for outcome in pending:
        try:
            _make_one(ui, outcome, config, force=force)
        except Exception as exc:  # one bad model must not stop the batch
            outcome.error = _report_error(ui, exc, outcome.model if batch else None)
            continue
        _print_mold(ui, outcome, batch=batch)

    if ui.as_json:
        entries = [outcome.as_json() for outcome in outcomes]
        ui.out.print_json(data=entries if batch else entries[0])
    elif batch:
        _print_batch(ui, outcomes)
    return 1 if any(outcome.error is not None for outcome in outcomes) else 0


def _make_one(ui: Ui, outcome: _Outcome, config: MoldConfig, *, force: bool) -> None:
    started = time.perf_counter()
    with ui.progress(outcome.model.name) as on_progress:
        # Imported here so the slow first import already shows the progress bar.
        from moldgen.pipeline import generate_mold
        from moldgen.report import summary

        result = generate_mold(outcome.model, config, progress=on_progress)
        if on_progress:
            on_progress("Saving files", 1.0)
        if force and outcome.out_dir.is_dir():
            _remove_previous_outputs(outcome.out_dir, result.part.name)
        outcome.files = result.save(outcome.out_dir)
    outcome.summary = summary(result)
    outcome.seconds = time.perf_counter() - started


def _print_mold(ui: Ui, outcome: _Outcome, *, batch: bool) -> None:
    info = outcome.summary
    assert info is not None
    if not ui.chatty:
        for warning in info["warnings"]:
            ui.warning(warning, subject=str(outcome.model) if batch else None)
        return

    part, mold, parting = info["part"], info["mold"], info["parting"]
    out = ui.out
    out.print()
    out.print(
        Text.assemble(
            (part["name"], "bold"),
            f": {mold['pieces']}-piece mold for {info['material']['name']}, "
            f"printed in {info['print_material']['name']}",
        )
    )
    pieces = Table(box=box.SIMPLE_HEAD)
    pieces.add_column("Piece")
    pieces.add_column("Print size")
    pieces.add_column("Approx. mass", justify="right")
    for piece in info["pieces"]:
        pieces.add_row(
            piece["name"], _size(piece["print_size_mm"]), f"{piece['approx_mass_g']:.0f} g"
        )
    out.print(pieces)

    pouring = f"{mold['sprue_diameter_mm']:g} mm sprue"
    if mold["funnel"]:
        pouring += " with funnel"
    pouring += f", {_count(mold['vents'], 'vent')}"
    keys = f"{mold['keys']}, {mold['key_clearance_mm']:g} mm clearance" if mold["keys"] else "none"
    scale = info["material"]["shrink_compensation_scale"]
    if abs(scale - 1.0) < 1e-9:
        shrinkage = "no compensation"
    else:
        change = (scale - 1.0) * 100
        shrinkage = f"cavity {'enlarged' if change > 0 else 'reduced'} by {abs(change):.2f}%"

    details = _details_table()
    details.add_row("Mold size", _size(mold["outer_size_mm"]))
    details.add_row("Wall thickness", f"{mold['wall_thickness_mm']:.1f} mm")
    plane = f"{parting['direction_label']}, plane at {_mm(parting['offset_mm'])}"
    if parting["surface"] == "curved":
        plane = (
            f"{parting['direction_label']}, curved surface "
            f"(up to {parting['surface_rise_mm']:.1f} mm from flat)"
        )
    layout = info["layout"]
    if layout and layout["side_pieces"]:
        details.add_row("Parting", plane)
        released = f"{layout['side_pieces']}, removed first in the order listed above"
        if layout["filled_volume_cm3"] > 0:
            released += f"; {layout['locked_fraction']:.1%} of the surface filled"
        details.add_row("Side pieces", released)
    elif layout:
        details.add_row("Parting", f"{plane}, {layout['locked_fraction']:.1%} undercut")
    else:
        details.add_row("Parting", f"{plane}, {parting['undercut_fraction']:.1%} undercut")
    details.add_row("Pouring", pouring)
    details.add_row("Keys", keys)
    details.add_row("Cast volume", f"{info['material']['cast_volume_cm3']:.1f} cm³ plus sprue")
    details.add_row("Shrinkage", shrinkage)
    out.print(details)
    out.print()

    for warning in info["warnings"]:
        ui.warning(warning)
    out.print(
        Text(
            f"Saved {_count(len(outcome.files), 'file')} to {outcome.out_dir} in {outcome.seconds:.1f} s"
        ),
        soft_wrap=True,
    )
    out.print(
        Text(f"Next: follow {outcome.out_dir / 'INSTRUCTIONS.txt'} to print and cast the mold."),
        soft_wrap=True,
    )


def _print_batch(ui: Ui, outcomes: Sequence[_Outcome]) -> None:
    if not ui.chatty:
        return
    table = Table(box=box.SIMPLE_HEAD, title="Summary", title_justify="left")
    table.add_column("Model")
    table.add_column("Result")
    table.add_column("Details", overflow="fold")
    for outcome in outcomes:
        if outcome.summary is None:
            table.add_row(str(outcome.model), Text("failed", style="red"), outcome.error or "")
        else:
            pieces = _count(len(outcome.summary["pieces"]), "piece")
            warnings = _count(len(outcome.summary["warnings"]), "warning")
            table.add_row(
                str(outcome.model),
                Text("done", style="green"),
                f"{outcome.out_dir} ({pieces}, {warnings})",
            )
    ui.out.print()
    ui.out.print(table)
    failed = sum(outcome.error is not None for outcome in outcomes)
    done = len(outcomes) - failed
    tail = f", {failed} failed." if failed else "."
    ui.out.print(f"{done} of {_count(len(outcomes), 'mold')} generated{tail}")


# analyze


def _score_json(score: DirectionScore, selected: bool) -> dict[str, Any]:
    return {
        "direction": [round(float(v), 4) for v in score.direction],
        "label": _direction_text(score.direction),
        "offset_mm": round(float(score.offset), 3),
        "undercut_fraction": round(float(score.undercut_fraction), 4),
        "low_draft_fraction": round(float(score.low_draft_fraction), 4),
        "selected": selected,
    }


def _analysis(
    model: Path, part: PreparedPart, parting: PartingResult, config: MoldConfig
) -> dict[str, Any]:
    """JSON-serialisable analysis of a prepared part and its parting options."""
    from moldgen.parting import DirectionScore
    from moldgen.pipeline import UNDERCUT_WARNING_FRACTION, auto_key_size, auto_wall_thickness

    material = get_material(config.material)
    print_material = get_print_material(config.print_material)
    chosen = DirectionScore(
        parting.direction, parting.offset, parting.undercut_fraction, parting.low_draft_fraction
    )

    def is_chosen(score: DirectionScore) -> bool:
        return bool(np.allclose(score.direction, chosen.direction, atol=1e-6)) and math.isclose(
            score.offset, chosen.offset, abs_tol=1e-6
        )

    rows = list(parting.candidates[:MAX_CANDIDATES_SHOWN])
    if not any(is_chosen(score) for score in rows):
        rows.append(chosen)

    cavity = part.mesh.copy().apply_transform(parting.to_mold)
    key_clearance = config.clearance if config.keys > 0 else None
    wall = auto_wall_thickness(cavity, material, key_clearance)
    warnings = [*part.warnings, *compatibility_warnings(material, print_material)]
    if parting.undercut_fraction > UNDERCUT_WARNING_FRACTION:
        if config.direction == "auto":
            where = "even in the best direction"
            advice = "Consider a flexible casting material or splitting the model."
        else:
            where = f"in direction {_direction_text(parting.direction)}"
            advice = "Leave out --direction to search for a better one."
        warnings.append(
            f"{parting.undercut_fraction:.1%} of the surface is undercut {where}; "
            f"the cast may lock in a rigid mold. {advice}"
        )

    command = ["moldgen", "make", Path(model).as_posix()]
    if config.material != _DEFAULTS.material:
        command += ["--material", config.material]
    if config.print_material != _DEFAULTS.print_material:
        command += ["--print-material", config.print_material]
    if config.units != _DEFAULTS.units:
        command += ["--units", config.units]
    if config.scale != _DEFAULTS.scale:
        command += ["--scale", f"{config.scale:g}"]
    if config.direction != "auto":
        command += [f"--direction={_direction_text(parting.direction)}"]
    if not config.repair:
        command += ["--no-repair"]

    repair = part.repair
    return {
        "part": {
            "name": part.name,
            "file": str(model),
            "size_mm": [round(float(v), 2) for v in part.mesh.extents],
            "volume_cm3": round(float(part.mesh.volume) / 1000.0, 3),
            "triangles": len(part.mesh.faces),
            "bodies": int(repair.bodies),
            "closed_before_repair": bool(repair.was_watertight),
            "repair_actions": list(repair.actions),
        },
        "parting": _score_json(chosen, True),
        "candidates": [_score_json(score, is_chosen(score)) for score in rows],
        "draft_threshold_deg": config.draft_threshold_deg,
        "recommended": {
            "material": material.key,
            "print_material": print_material.key,
            "direction": _direction_text(parting.direction),
            "parting_offset_mm": round(float(parting.offset), 3),
            "wall_thickness_mm": round(wall, 1),
            "sprue_diameter_mm": material.sprue_diameter_mm,
            "vent_diameter_mm": material.vent_diameter_mm,
            "key_diameter_mm": round(2 * auto_key_size(wall, config.clearance)[0], 1),
            "shrinkage_percent": round(material.linear_shrinkage * 100, 3),
        },
        "warnings": warnings,
        "command": shlex.join(command),
    }


def _print_analysis(ui: Ui, data: dict[str, Any], repaired: bool) -> None:
    part, rec = data["part"], data["recommended"]
    out = ui.out
    out.print(Text(part["file"], style="bold"))
    details = _details_table()
    details.add_row("Size", _size(part["size_mm"]))
    details.add_row("Volume", f"{part['volume_cm3']:.2f} cm³")
    closed = "closed" if part["closed_before_repair"] else "not closed before repair"
    details.add_row(
        "Mesh",
        f"{part['triangles']:,} triangles, {_count(part['bodies'], 'body', 'bodies')}, {closed}",
    )
    if repaired:
        details.add_row("Repair", "; ".join(part["repair_actions"]) or "nothing to fix")
    else:
        details.add_row("Repair", "skipped")
    out.print(details)
    out.print()

    table = Table(box=box.SIMPLE_HEAD, title="Parting directions, best first", title_justify="left")
    table.add_column("Direction")
    table.add_column("Plane at", justify="right")
    table.add_column("Undercut", justify="right")
    table.add_column(f"Low draft (<{data['draft_threshold_deg']:g}°)", justify="right")
    table.add_column("")
    for row in data["candidates"]:
        table.add_row(
            row["label"],
            _mm(row["offset_mm"]),
            f"{row['undercut_fraction']:.1%}",
            f"{row['low_draft_fraction']:.1%}",
            "selected" if row["selected"] else "",
            style="bold" if row["selected"] else None,
        )
    out.print(table)

    material = MATERIALS[rec["material"]].name
    printed = PRINT_MATERIALS[rec["print_material"]].name
    out.print(Text(f"Recommended settings for {material}, printed in {printed}", style="bold"))
    settings = _details_table()
    settings.add_row("Direction", f"{rec['direction']}, plane at {_mm(rec['parting_offset_mm'])}")
    settings.add_row("Wall thickness", f"{rec['wall_thickness_mm']:.1f} mm")
    settings.add_row(
        "Sprue / vent", f"{rec['sprue_diameter_mm']:g} mm / {rec['vent_diameter_mm']:g} mm"
    )
    settings.add_row("Key diameter", f"{rec['key_diameter_mm']:.1f} mm")
    settings.add_row(
        "Shrinkage", f"{rec['shrinkage_percent']:g}% (the cavity is enlarged to match)"
    )
    out.print(settings)
    out.print()

    for warning in data["warnings"]:
        ui.warning(warning)
    out.print(Text(f"Next: {data['command']}"), soft_wrap=True)


# guided mode


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _find_models() -> list[Path]:
    found: list[Path] = []
    for folder in (Path("."), Path("models")):
        if folder.is_dir():
            found += sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            )
    return found[:MAX_MODELS_LISTED]


def _ask_model(console: Console) -> Path:
    found = _find_models()
    if found:
        console.print("Models found here:")
        for number, path in enumerate(found, 1):
            console.print(f"  {number}. {path}")
        question, default = "Model (number or path)", "1"
    else:
        question, default = "Path to the model file", None
    while True:
        answer = Prompt.ask(question, console=console, default=default) or ""
        answer = answer.strip()
        if answer.isdigit() and 1 <= int(answer) <= len(found):
            return found[int(answer) - 1]
        if not answer:
            continue
        try:
            return _check_model(Path(answer).expanduser())
        except CliError as exc:
            console.print(Text(str(exc), style="red"))


def _ask_material(console: Console) -> str:
    keys = list(MATERIALS)
    if len(keys) == 1:
        return keys[0]
    console.print("Casting materials:")
    for number, key in enumerate(keys, 1):
        console.print(f"  {number}. {MATERIALS[key].name} ({key})")
    default = str(keys.index(_DEFAULTS.material) + 1) if _DEFAULTS.material in keys else "1"
    while True:
        answer = (Prompt.ask("Material", console=console, default=default) or "").strip().lower()
        if answer.isdigit() and 1 <= int(answer) <= len(keys):
            return keys[int(answer) - 1]
        if answer in MATERIALS:
            return answer
        console.print(Text(f"Enter a number from 1 to {len(keys)}.", style="red"))


def _guided(ui: Ui) -> int:
    """Ask for a model, material and piece count, then run ``make`` with automatic settings."""
    console = ui.out
    console.print(
        Text("moldgen guided setup", style="bold"),
        Text("Press Enter to accept the value in brackets, Ctrl+C to quit."),
        sep="\n",
    )
    console.print()
    model = _ask_model(console)
    material = _ask_material(console)
    pieces = Prompt.ask("Pieces", console=console, choices=["auto", "2", "4"], default="auto")
    config = _build_config(material=material, pieces=pieces)
    out_dir = DEFAULT_OUT / model.stem

    force = False
    if _has_files(out_dir):
        force = Confirm.ask(
            Text(f"{out_dir} already contains files. Overwrite them?"),
            console=console,
            default=False,
        )
        if not force:
            console.print("Nothing was generated.")
            return 1
    command = [
        "moldgen",
        "make",
        Path(model).as_posix(),
        "--material",
        material,
        "--pieces",
        pieces,
    ]
    if force:
        command.append("--force")
    console.print(Text(f"Equivalent command: {shlex.join(command)}"), soft_wrap=True)
    if not Confirm.ask("Generate the mold now?", console=console, default=True):
        console.print("Nothing was generated.")
        return 1
    return _run_make(ui, [model], config, DEFAULT_OUT, force)


# commands


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"moldgen {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_show_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Turn a 3D model into a 3D-printable casting mold.

    Run moldgen without arguments in a terminal for a short guided setup.
    """
    if ctx.invoked_subcommand is not None:
        return
    if _is_interactive():
        ui = Ui()
        with _logging_to(ui.err, verbose=False):
            raise typer.Exit(_guided(ui))
    typer.echo(ctx.get_help(), nl=False)


@app.command(
    epilog=(
        "Examples:\n\n"
        "moldgen make pawn.stl\n\n"
        "moldgen make pawn.stl --material resin --pieces 4 --shrinkage 1.5\n\n"
        "moldgen make models/*.stl --out molds --force"
    )
)
def make(
    models: Annotated[
        list[Path], typer.Argument(metavar="MODEL...", help=f"Model files ({FORMATS}).")
    ],
    material: MaterialOption = _DEFAULT_MATERIAL,
    print_material: PrintMaterialOption = _DEFAULT_PRINT_MATERIAL,
    bed: Annotated[
        str,
        _option(
            "--bed",
            help="Printer build volume in mm as WIDTHxDEPTHxHEIGHT; pieces are checked against it.",
            panel=_MATERIAL,
        ),
    ] = "x".join(f"{v:g}" for v in _DEFAULTS.bed_size_mm),
    shrinkage: Annotated[
        float | None,
        _option(
            "--shrinkage",
            help="Linear shrinkage of the cast in percent, e.g. 1.5. Overrides the preset.",
            panel=_MATERIAL,
            default="from material",
            min=-5.0,
            max=20.0,
        ),
    ] = None,
    direction: DirectionOption = "auto",
    parting_offset: Annotated[
        float | None,
        _option(
            "--parting-offset",
            help="Parting plane position along the direction, in mm.",
            panel=_PARTING,
            default="least undercut",
        ),
    ] = None,
    pieces: Annotated[
        Pieces,
        _option(
            "--pieces",
            help="auto adds side pieces where the two halves cannot release the part; "
            "2 for a two-part mold; 4 to split each half again.",
            panel=_PARTING,
        ),
    ] = str(_DEFAULTS.pieces),
    max_pieces: Annotated[
        int,
        _option("--max-pieces", help="Most pieces --pieces auto may use (2-10).", panel=_PARTING),
    ] = _DEFAULTS.max_pieces,
    flat_side_pieces: Annotated[
        bool,
        _option(
            "--flat-side-pieces",
            help="Cut side pieces with flat planes only; faster for parts that need them.",
            panel=_PARTING,
        ),
    ] = False,
    flat_parting: Annotated[
        bool,
        _option(
            "--flat-parting",
            help="Always split the halves with a flat plane, even where a curved surface "
            "would release more of the part.",
            panel=_PARTING,
        ),
    ] = False,
    wall: Annotated[
        float | None,
        _option(
            "--wall", help="Wall thickness around the part, in mm.", panel=_MOLD, default="auto"
        ),
    ] = None,
    sprue: Annotated[
        float | None,
        _option(
            "--sprue", help="Pour channel diameter, in mm.", panel=_MOLD, default="from material"
        ),
    ] = None,
    vent: Annotated[
        float | None,
        _option("--vent", help="Air vent diameter, in mm.", panel=_MOLD, default="from material"),
    ] = None,
    no_funnel: Annotated[
        bool, _option("--no-funnel", help="Leave out the pour funnel.", panel=_MOLD)
    ] = False,
    no_vents: Annotated[
        bool, _option("--no-vents", help="Leave out the air vents.", panel=_MOLD)
    ] = False,
    keys: Annotated[
        int, _option("--keys", help="Registration keys on the parting face (0-8).", panel=_MOLD)
    ] = _DEFAULTS.keys,
    key_diameter: Annotated[
        float | None,
        _option("--key-diameter", help="Key diameter, in mm.", panel=_MOLD, default="auto"),
    ] = None,
    clearance: Annotated[
        float, _option("--clearance", help="Gap per side between mating keys, in mm.", panel=_MOLD)
    ] = _DEFAULTS.clearance,
    units: UnitsOption = _DEFAULTS.units,
    scale: ScaleOption = _DEFAULTS.scale,
    no_repair: NoRepairOption = False,
    out: Annotated[
        Path,
        _option("--out", "-o", help="Output folder; each model gets a subfolder.", panel=_OUTPUT),
    ] = DEFAULT_OUT,
    force: Annotated[
        bool,
        _option(
            "--force", "-f", help="Overwrite earlier results in the output folder.", panel=_OUTPUT
        ),
    ] = False,
    no_orient: Annotated[
        bool,
        _option(
            "--no-orient",
            help="Export pieces in their assembled position, not laid out for printing.",
            panel=_OUTPUT,
        ),
    ] = False,
    json_output: JsonOption = False,
    quiet: Annotated[
        bool, _option("--quiet", "-q", help="Only print warnings and errors.", panel=_OUTPUT)
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Generate a casting mold for each MODEL.

    Each model gets a subfolder of the output folder with print-ready STL pieces, report.json and INSTRUCTIONS.txt. Settings left at auto come from the part size and the material preset.
    """
    ui = Ui(quiet=quiet, as_json=json_output, verbose=verbose)
    with _logging_to(ui.err, verbose):
        config = _build_config(
            material=material,
            print_material=print_material,
            bed=bed,
            units=units,
            scale=scale,
            direction=direction,
            parting_offset=parting_offset,
            pieces=pieces,
            max_pieces=max_pieces,
            parting_surface="flat" if flat_parting else "auto",
            side_piece_cuts="flat" if flat_side_pieces else "auto",
            wall=wall,
            shrinkage_percent=shrinkage,
            sprue=sprue,
            vent=vent,
            funnel=not no_funnel,
            vents=not no_vents,
            keys=keys,
            key_diameter=key_diameter,
            clearance=clearance,
            repair=not no_repair,
            orient=not no_orient,
        )
        code = _run_make(ui, models, config, out, force)
    if code:
        raise typer.Exit(code)


@app.command()
def analyze(
    model: Annotated[
        Path, typer.Argument(metavar="MODEL", help="Model file to inspect.", show_default=False)
    ],
    material: MaterialOption = _DEFAULT_MATERIAL,
    print_material: PrintMaterialOption = _DEFAULT_PRINT_MATERIAL,
    units: UnitsOption = _DEFAULTS.units,
    scale: ScaleOption = _DEFAULTS.scale,
    no_repair: NoRepairOption = False,
    direction: DirectionOption = "auto",
    json_output: JsonOption = False,
    verbose: VerboseOption = False,
) -> None:
    """Inspect a model and recommend mold settings. Writes no files.

    Shows the part size and repairs, the best parting directions with their undercut and low-draft share, and the settings 'moldgen make' would use.
    """
    ui = Ui(as_json=json_output, verbose=verbose)
    with _logging_to(ui.err, verbose):
        config = _build_config(
            material=material,
            print_material=print_material,
            units=units,
            scale=scale,
            direction=direction,
            repair=not no_repair,
        )
        try:
            _check_model(model)
            with ui.status(f"Analyzing {model.name}"):
                # Imported here so the slow first import already shows the spinner.
                from moldgen.parting import analyze_parting
                from moldgen.pipeline import prepare_part

                part = prepare_part(model, config)
                parting = analyze_parting(
                    part.mesh,
                    config.direction_vector(),
                    config.parting_offset,
                    draft_threshold_deg=config.draft_threshold_deg,
                )
            data = _analysis(model, part, parting, config)
        except Exception as exc:  # report any failure as a single line, see _explain
            _report_error(ui, exc)
            raise typer.Exit(1) from None
    if ui.as_json:
        ui.out.print_json(data=data)
    else:
        _print_analysis(ui, data, repaired=config.repair)


@app.command()
def materials() -> None:
    """List the casting and print material presets."""
    console = Console(highlight=False, markup=False, emoji=False)
    casting = Table(
        box=box.SIMPLE_HEAD, title="Casting materials (--material)", title_justify="left"
    )
    casting.add_column("Material")
    casting.add_column("Key", no_wrap=True)
    casting.add_column("Shrinkage", justify="right", no_wrap=True)
    casting.add_column("Pour temp.", no_wrap=True)
    casting.add_column("Sprue / vent", no_wrap=True)
    for m in MATERIALS.values():
        pour = (
            "room temp."
            if m.pour_temp_c is None
            else f"{m.pour_temp_c[0]:.0f}-{m.pour_temp_c[1]:.0f} °C"
        )
        sizes = f"{m.sprue_diameter_mm:g} / {m.vent_diameter_mm:g} mm"
        casting.add_row(m.name, m.key, f"{m.linear_shrinkage * 100:.1f}%", pour, sizes)

    # Release agent advice is a sentence per material, too long for a table cell.
    release = _details_table()
    for m in MATERIALS.values():
        release.add_row(m.key, m.release_agent)

    printing = Table(
        box=box.SIMPLE_HEAD, title="Print materials (--print-material)", title_justify="left"
    )
    printing.add_column("Material")
    printing.add_column("Key", no_wrap=True)
    printing.add_column("Max. service temp.", justify="right", no_wrap=True)
    for pm in PRINT_MATERIALS.values():
        printing.add_row(pm.name, pm.key, f"{pm.max_service_temp_c:.0f} °C")

    console.print(casting)
    console.print(Text("Release agents", style="italic"))
    console.print(release)
    console.print()
    console.print(printing)
    console.print("Pass a key to --material (-m) or --print-material (-p).")


@app.command()
def gui(
    model: Annotated[
        Path | None,
        typer.Argument(metavar="MODEL", help="Model to open on start.", show_default=False),
    ] = None,
    host: Annotated[str, typer.Option(help="Address to serve the interface on.")] = "127.0.0.1",
    port: Annotated[
        int, typer.Option(min=1, max=65535, help="Port to serve the interface on.")
    ] = 8080,
    no_browser: Annotated[
        bool, typer.Option("--no-browser", help="Do not open a browser window.")
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Open the graphical interface in a web browser."""
    ui = Ui(verbose=verbose)
    # The GUI reports its address and how to stop it at INFO level.
    with _logging_to(ui.err, verbose, info_from=["moldgen.gui"]):
        try:
            if model is not None:
                _check_model(model)
            try:
                # Imported lazily so a problem in the GUI never breaks the other commands.
                from moldgen.gui.app import run
            except Exception as exc:
                raise CliError(
                    f"The graphical interface could not be loaded ({type(exc).__name__}: {exc})",
                    "The make, analyze and materials commands still work. Run with -v for details.",
                ) from exc
            run(model=model, host=host, port=port, open_browser=not no_browser)
        except KeyboardInterrupt:
            ui.err.print("Stopped.")
        except Exception as exc:  # report any failure as a single line, see _explain
            _report_error(ui, exc)
            raise typer.Exit(1) from None


def main() -> None:
    """Entry point of the ``moldgen`` console script."""
    app(prog_name="moldgen")
