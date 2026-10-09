"""Local web GUI for moldgen, built on viser.

Start it with ``moldgen gui`` or :func:`run`. The browser tab shows the part
coloured by the parting analysis, the parting plane, the mold settings and,
after generating, an exploded view of the mold pieces with a zip download.
A curved parting surface, when the mold uses one, replaces the plane until
the plane or the settings change.
When the mold has a planned piece layout the part is coloured by piece instead.
This module holds the controls and jobs; the 3D scene lives in
:class:`moldgen.gui.viewer.Viewer`.

Threading
---------
viser runs GUI callbacks on its own threads. Callbacks only read control
values and queue work; everything heavy (loading, analysis, generation,
zipping) runs on a single worker thread, so jobs never overlap. Shared state
is guarded by ``self._lock``. Revision counters let a finished job notice that
the model or the parting plane changed while it ran, in which case its result
is dropped. Parting updates are debounced and coalesced: at most one parting
job is queued and it always works on the latest request.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import threading
import time
import webbrowser
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import viser

from moldgen.config import ConfigError, MoldConfig
from moldgen.materials import MATERIALS
from moldgen.meshio import SUPPORTED_SUFFIXES, MeshLoadError, load_mesh
from moldgen.parting import (
    PartingResult,
    analyze_parting,
    best_offset,
    classify_faces,
    mold_frame,
)
from moldgen.pipeline import MoldError, MoldResult, PreparedPart, generate_mold, prepare_part
from moldgen.report import summary, zip_result

from . import state as st
from .viewer import Viewer

log = logging.getLogger(__name__)

PARTING_DEBOUNCE_S = 0.15
RELOAD_DEBOUNCE_S = 0.6
EXPECTED_ERRORS = (MoldError, MeshLoadError, ConfigError)
READY_TEXT = "Ready to generate."


class _Debouncer:
    """Calls ``func`` once, ``delay`` seconds after the most recent :meth:`trigger`."""

    def __init__(self, delay: float, func: Callable[[], None]) -> None:
        self._delay = delay
        self._func = func
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def trigger(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._delay, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def flush(self) -> None:
        """Run a pending call now instead of waiting for the delay."""
        with self._lock:
            pending = self._timer is not None
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        if pending:
            self._func()

    def cancel(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def _fire(self) -> None:
        with self._lock:
            self._timer = None
        self._func()


@dataclass
class _Model:
    """A loaded file before unit conversion and repair."""

    raw: trimesh.Trimesh
    name: str


@dataclass
class _PartingRequest:
    direction: np.ndarray
    offset: float | None
    parting_rev: int
    model_rev: int


class MoldGui:
    """One GUI session on a viser server. All connected browser tabs share it."""

    def __init__(self, server: viser.ViserServer) -> None:
        self.server = server
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moldgen-worker")
        self._lock = threading.RLock()
        self._quiet_flag = threading.local()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="moldgen-gui-")

        self._model: _Model | None = None
        self._part: PreparedPart | None = None
        self._areas: np.ndarray | None = None
        self._analysis: PartingResult | None = None
        self._choices: list[st.DirectionChoice] = []
        self._parting: PartingResult | None = None
        self._result: MoldResult | None = None
        self._result_outdated = False
        self._job: str | None = None
        self._model_rev = 0
        self._parting_rev = 0
        self._pending_parting: _PartingRequest | None = None
        self._parting_job_queued = False
        self._pending_notices: list[tuple[str, str, bool]] = []

        self._offset_debouncer = _Debouncer(PARTING_DEBOUNCE_S, self._request_offset_from_controls)
        self._reload_debouncer = _Debouncer(RELOAD_DEBOUNCE_S, self._request_reprepare)

        self._defaults = st.MoldSettings.from_config(MoldConfig())
        self._material_options = st.material_options()
        self._print_options = st.print_material_options()
        self.viewer = Viewer(server)
        self._build_gui()
        self._sync_controls()
        server.on_client_connect(self._on_client_connect)

    # ------------------------------------------------------------------
    # Public API

    def open_path(self, path: str | Path) -> None:
        """Load a model file (in the worker thread)."""
        path = Path(path)
        with self._lock:
            self._model_rev += 1
            rev = self._model_rev
        self._submit(self._load_job, path, path.stem, rev)

    def close(self) -> None:
        """Stop timers and the worker and remove temporary files."""
        self._offset_debouncer.cancel()
        self._reload_debouncer.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._tmpdir.cleanup()

    # ------------------------------------------------------------------
    # Layout

    def _build_gui(self) -> None:
        gui = self.server.gui
        gui.configure_theme(
            control_width="large",
            dark_mode=False,
            show_logo=False,
            show_share_button=False,
            brand_color=st.PLANE_COLOR,
        )
        gui.set_panel_label("moldgen")
        d = self._defaults

        with gui.add_folder("Model"):
            self.upload = gui.add_upload_button(
                "Open model",
                icon=viser.Icon.UPLOAD,
                mime_type=",".join(SUPPORTED_SUFFIXES),
                hint="STL, OBJ, PLY, OFF, 3MF, GLB or glTF",
            )
            self.part_info = gui.add_html(st.empty_part_html())

        self.parting_folder = gui.add_folder("Parting", visible=False)
        with self.parting_folder:
            self.direction = gui.add_dropdown(
                "Direction",
                (st.AUTO_DIRECTION,),
                hint="Direction in which the top half of the mold is pulled off.",
            )
            self.offset = gui.add_slider(
                "Position (mm)",
                min=0.0,
                max=1.0,
                step=0.01,
                initial_value=0.5,
                hint="Where the parting plane cuts the part, along the direction.",
            )
            self.readout = gui.add_html(st.readout_html(None, None))
            self.show_plane = gui.add_checkbox("Show plane", True)
            self.parting_note = gui.add_html(
                st.info_html(
                    intro="Drag the arrow in the view or use the slider to move the plane."
                )
            )

        with gui.add_folder("Mold settings"):
            self.material = gui.add_dropdown(
                "Cast material",
                tuple(self._material_options),
                initial_value=st.label_for(self._material_options, d.material),
            )
            self.print_material = gui.add_dropdown(
                "Print material",
                tuple(self._print_options),
                initial_value=st.label_for(self._print_options, d.print_material),
                hint="Material the mold is printed in, used for temperature checks.",
            )
            self.material_info = gui.add_html(st.material_html(d.material, d.print_material))
            self.pieces = gui.add_dropdown(
                "Pieces",
                tuple(st.PIECE_OPTIONS),
                initial_value=st.label_for(st.PIECE_OPTIONS, d.pieces),
                hint="Automatic adds side pieces where the two halves cannot release the part. "
                "Four pieces split each half once more, which helps with deep or wide parts.",
            )
            self.max_pieces = gui.add_slider(
                "Max pieces",
                min=st.MAX_PIECES_RANGE[0],
                max=st.MAX_PIECES_RANGE[1],
                step=1,
                initial_value=d.max_pieces,
                hint="Most pieces Automatic may use. Areas still locked are filled in the cavity.",
            )
            self.side_piece_cuts = gui.add_dropdown(
                "Side piece cuts",
                tuple(st.SIDE_PIECE_CUT_OPTIONS),
                initial_value=st.label_for(st.SIDE_PIECE_CUT_OPTIONS, d.side_piece_cuts),
                hint="Curved cuts follow the part and can save pieces; generating takes longer "
                "because both kinds are compared.",
            )
            self.parting_surface = gui.add_dropdown(
                "Parting surface",
                tuple(st.PARTING_SURFACE_OPTIONS),
                initial_value=st.label_for(st.PARTING_SURFACE_OPTIONS, d.parting_surface),
                hint="Lets the line between the two halves follow the part where a flat cut "
                "would trap it.",
            )
            self.wall_auto = gui.add_checkbox(
                "Auto wall", d.wall_auto, hint="Derive the wall thickness from the part size."
            )
            self.wall = gui.add_number(
                "Wall (mm)",
                round(d.wall_thickness, 1),
                min=0.5,
                max=200.0,
                step=0.5,
                hint="Minimum mold wall around the part.",
            )

        with gui.add_folder("Advanced settings", expand_by_default=False):
            self.units = gui.add_dropdown(
                "File units",
                tuple(st.UNIT_LABELS),
                initial_value=d.units,
                hint="Units the file was modelled in; everything is converted to millimetres.",
            )
            self.scale = gui.add_number(
                "Scale", d.scale, min=0.001, max=1000.0, step=0.01, hint="Extra uniform scale."
            )
            self.keys = gui.add_slider(
                "Keys",
                min=0,
                max=8,
                step=1,
                initial_value=d.keys,
                hint="Registration keys that align the pieces.",
            )
            self.clearance = gui.add_number(
                "Clearance (mm)",
                d.clearance,
                min=0.0,
                max=2.0,
                step=0.05,
                hint="Gap per side between key and socket. Increase if pieces fit too tightly.",
            )
            self.sprue_auto = gui.add_checkbox(
                "Auto sprue", d.sprue_auto, hint="Use the material's sprue diameter."
            )
            self.sprue = gui.add_number(
                "Sprue (mm)",
                d.sprue_diameter,
                min=0.5,
                max=50.0,
                step=0.5,
                hint="Diameter of the pour channel.",
            )
            self.funnel = gui.add_checkbox("Pour funnel", d.funnel)
            self.vents = gui.add_checkbox("Air vents", d.vents)
            self.shrink_auto = gui.add_checkbox(
                "Auto shrinkage",
                not d.shrinkage_override,
                hint="Use the material's shrinkage.",
            )
            self.shrink = gui.add_number(
                "Shrinkage (%)",
                d.shrinkage_percent,
                min=-5.0,
                max=19.9,
                step=0.05,
                hint="Linear shrinkage of the cast; the cavity is enlarged to compensate.",
            )
            self.draft = gui.add_number(
                "Draft limit (°)",
                d.draft_threshold_deg,
                min=0.0,
                max=44.5,
                step=0.5,
                hint="Faces with less draft than this angle are flagged as low draft.",
            )
            self.repair = gui.add_checkbox(
                "Repair mesh", d.repair, hint="Fix holes and flipped faces when loading."
            )
            self.orient = gui.add_checkbox(
                "Orient pieces",
                d.orient_for_print,
                hint="Exported pieces lie the way they print with the least support.",
            )

        self.generate = gui.add_button("Generate mold", disabled=True)
        self.progress = gui.add_progress_bar(0.0, visible=False)
        self.status = gui.add_html("")

        self.result_folder = gui.add_folder("Mold", visible=False)
        with self.result_folder:
            self.explode = gui.add_slider(
                "Explode (mm)", min=0.0, max=100.0, step=0.5, initial_value=20.0
            )
            self.show_part = gui.add_checkbox("Show part", True)
            self.show_pieces = gui.add_checkbox("Show pieces", True)
            self.result_note = gui.add_html(
                st.text_html("Settings changed. Generate again to update the mold."),
                visible=False,
            )
            self.legend = gui.add_html("")
            self.summary = gui.add_html("")
            self.warnings = gui.add_html("", visible=False)
            self.download = gui.add_button("Download mold (.zip)", icon=viser.Icon.DOWNLOAD)

        self.upload.on_upload(self._on_upload)
        for handle in (self.units, self.scale, self.repair):
            handle.on_update(self._on_load_setting)
        self.direction.on_update(self._on_direction)
        self.offset.on_update(self._on_offset)
        self.draft.on_update(self._on_draft)
        self.show_plane.on_update(self._on_show_plane)
        for handle in (
            self.material,
            self.print_material,
            self.pieces,
            self.max_pieces,
            self.side_piece_cuts,
            self.parting_surface,
            self.wall_auto,
            self.wall,
            self.keys,
            self.clearance,
            self.sprue_auto,
            self.sprue,
            self.funnel,
            self.vents,
            self.shrink_auto,
            self.shrink,
            self.orient,
        ):
            handle.on_update(self._on_mold_setting)
        self.generate.on_click(self._on_generate)
        self.explode.on_update(lambda _: self.viewer.set_explode(float(self.explode.value)))
        self.show_part.on_update(self._on_visibility)
        self.show_pieces.on_update(self._on_visibility)
        self.download.on_click(self._on_download)
        self._refresh_auto_values()

    # ------------------------------------------------------------------
    # Helpers

    @contextlib.contextmanager
    def _quiet(self) -> Iterator[None]:
        """Programmatic control updates inside this block do not trigger callbacks."""
        self._quiet_flag.active = True
        try:
            yield
        finally:
            self._quiet_flag.active = False

    def _is_quiet(self) -> bool:
        return getattr(self._quiet_flag, "active", False)

    def _submit(self, func: Callable[..., None], *args: Any) -> None:
        self._executor.submit(self._run_job, func, *args)

    def _run_job(self, func: Callable[..., None], *args: Any) -> None:
        try:
            func(*args)
        except EXPECTED_ERRORS as exc:
            self._notify("Cannot continue", str(exc), error=True)
        except Exception as exc:
            log.exception("GUI job %s failed", getattr(func, "__name__", func))
            self._notify("Unexpected error", f"{type(exc).__name__}: {exc}", error=True)

    def _notify(self, title: str, body: str, *, error: bool = False) -> None:
        """Show a notification in every open tab, or keep it for the next tab that connects."""
        if error:
            log.warning("%s: %s", title, body)
        clients = list(self.server.get_clients().values())
        if not clients:
            with self._lock:
                self._pending_notices.append((title, body, error))
            return
        for client in clients:
            self._show_notice(client, title, body, error)

    @staticmethod
    def _show_notice(client: Any, title: str, body: str, error: bool) -> None:
        client.add_notification(
            title,
            body,
            color="red" if error else None,
            auto_close_seconds=None if error else 6.0,
        )

    def _on_client_connect(self, client: Any) -> None:
        with self._lock:
            notices, self._pending_notices = self._pending_notices, []
        for title, body, error in notices:
            self._show_notice(client, title, body, error)

    def _set_status(self, text: str) -> None:
        self.status.content = st.text_html(text) if text else ""

    def _show_busy(self, busy: bool) -> None:
        """Show the progress bar as an indeterminate activity indicator."""
        with self.server.atomic():
            self.progress.animated = busy
            self.progress.value = 100.0 if busy else 0.0
            self.progress.visible = busy

    def _read_settings(self) -> st.MoldSettings:
        return st.MoldSettings(
            units=self.units.value,
            scale=float(self.scale.value),
            repair=self.repair.value,
            material=self._material_options[self.material.value],
            print_material=self._print_options[self.print_material.value],
            pieces=st.PIECE_OPTIONS[self.pieces.value],
            max_pieces=int(self.max_pieces.value),
            parting_surface=st.PARTING_SURFACE_OPTIONS[self.parting_surface.value],
            side_piece_cuts=st.SIDE_PIECE_CUT_OPTIONS[self.side_piece_cuts.value],
            wall_auto=self.wall_auto.value,
            wall_thickness=float(self.wall.value),
            keys=int(self.keys.value),
            clearance=float(self.clearance.value),
            sprue_auto=self.sprue_auto.value,
            sprue_diameter=float(self.sprue.value),
            funnel=self.funnel.value,
            vents=self.vents.value,
            shrinkage_override=not self.shrink_auto.value,
            shrinkage_percent=float(self.shrink.value),
            draft_threshold_deg=float(self.draft.value),
            orient_for_print=self.orient.value,
        )

    def _sync_controls(self) -> None:
        """Enable, disable, show and hide controls to match the current state."""
        with self._lock:
            job = self._job
            has_model = self._model is not None
            has_part = self._part is not None
            has_parting = self._parting is not None
            has_result = self._result is not None
        generating = job == "generate"
        busy = job is not None
        self.upload.disabled = busy
        # Once a part is loaded, generating becomes the primary action.
        self.upload.color = "gray" if has_part else None
        for handle in (self.units, self.scale, self.repair):
            handle.disabled = busy or not has_model
        self.parting_folder.visible = has_part
        self.direction.disabled = busy or not has_parting
        self.offset.disabled = busy or not has_parting
        self.draft.disabled = busy
        for handle in (
            self.material,
            self.print_material,
            self.pieces,
            self.max_pieces,
            self.side_piece_cuts,
            self.parting_surface,
            self.wall_auto,
            self.keys,
            self.clearance,
            self.sprue_auto,
            self.funnel,
            self.vents,
            self.shrink_auto,
            self.orient,
        ):
            handle.disabled = generating
        self.max_pieces.visible = self.pieces.value == st.AUTO_PIECES
        self.side_piece_cuts.visible = self.pieces.value == st.AUTO_PIECES
        # Four pieces always split along a flat plane.
        self.parting_surface.visible = st.PIECE_OPTIONS[self.pieces.value] != 4
        self.wall.disabled = generating or self.wall_auto.value
        self.sprue.disabled = generating or self.sprue_auto.value
        self.shrink.disabled = generating or self.shrink_auto.value
        self.generate.disabled = busy or not has_parting
        self.result_folder.visible = has_result
        self.download.disabled = generating or not has_result
        # The plane would cut through the displayed mold, so it gives way to the pieces.
        mold_shown = has_result and self.show_pieces.value
        self.viewer.set_plane_visible(self.show_plane.value and has_parting and not mold_shown)

    def _refresh_auto_values(self) -> None:
        """Show what "automatic" resolves to in the disabled number fields."""
        material_key = self._material_options[self.material.value]
        material = MATERIALS[material_key]
        with self._lock:
            part = self._part
        with self._quiet():
            if self.wall_auto.value and part is not None:
                self.wall.value = round(st.auto_wall_estimate(part.mesh.extents, material_key), 1)
            elif self.wall_auto.value:
                self.wall.value = round(material.min_wall_mm, 1)
            if self.sprue_auto.value:
                self.sprue.value = round(material.sprue_diameter_mm, 2)
            if self.shrink_auto.value:
                self.shrink.value = round(100.0 * material.linear_shrinkage, 3)
        self.material_info.content = st.material_html(
            material_key, self._print_options[self.print_material.value]
        )

    # ------------------------------------------------------------------
    # Callbacks (viser event threads)

    def _on_upload(self, event: Any) -> None:
        uploaded = event.target.value
        name = Path(uploaded.name).name or "model"
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            self._notify(
                "Unsupported file",
                f"{name} is not a supported mesh file. Use one of: "
                + ", ".join(SUPPORTED_SUFFIXES),
                error=True,
            )
            return
        if not uploaded.content:
            self._notify("Empty file", f"{name} contains no data.", error=True)
            return
        # A fresh folder per upload keeps the original name for error messages.
        folder = Path(tempfile.mkdtemp(dir=self._tmpdir.name))
        target = folder / name
        target.write_bytes(uploaded.content)
        with self._lock:
            self._model_rev += 1
            rev = self._model_rev
        self._submit(self._load_job, target, Path(name).stem, rev)

    def _on_load_setting(self, _event: Any) -> None:
        if self._is_quiet():
            return
        with self._lock:
            if self._model is None:
                return
            self._model_rev += 1
        self._reload_debouncer.trigger()

    def _on_direction(self, _event: Any) -> None:
        if self._is_quiet():
            return
        choice = next((c for c in self._choices if c.label == self.direction.value), None)
        if choice is None:
            return
        self._offset_debouncer.cancel()
        self._request_parting(choice.direction, choice.offset)

    def _on_offset(self, _event: Any) -> None:
        if self._is_quiet():
            return
        self.viewer.move_plane(float(self.offset.value))
        self._offset_debouncer.trigger()

    def _on_draft(self, _event: Any) -> None:
        if self._is_quiet():
            return
        with self._lock:
            parting = self._parting
        if parting is not None:
            # Reuse the exact offset: the slider holds a rounded copy.
            self._request_parting(parting.direction, parting.offset)

    def _on_plane_drag(self, event: Any) -> None:
        with self._lock:
            parting = self._parting
            busy = self._job is not None
        if parting is None or event.phase == "start":
            return
        if busy:
            # The plane is locked while a job runs; put the handle back.
            self.viewer.move_plane(float(self.offset.value))
            return
        position = st.offset_from_position(event.target.position, parting.direction)
        offset = float(np.clip(position, self.offset.min, self.offset.max))
        with self._quiet():
            self.offset.value = offset
        if event.phase == "end":
            self._offset_debouncer.cancel()
            self._request_offset_from_controls()
        else:
            self._offset_debouncer.trigger()

    def _on_show_plane(self, _event: Any) -> None:
        self._sync_controls()

    def _on_mold_setting(self, _event: Any) -> None:
        if self._is_quiet():
            return
        self._refresh_auto_values()
        self._sync_controls()
        with self._lock:
            if self._result is None or self._result_outdated:
                return
            self._result_outdated = True
            parting = self._parting
        self.result_note.visible = True
        # The piece colours and the curved surface describe the old mold; go back to the analysis.
        self.viewer.remove_surface()
        if parting is not None:
            self.viewer.show_classes(np.asarray(parting.face_class))

    def _on_visibility(self, _event: Any) -> None:
        self.viewer.set_visibility(part=self.show_part.value, pieces=self.show_pieces.value)
        self._sync_controls()

    def _on_generate(self, event: Any) -> None:
        self._offset_debouncer.flush()
        try:
            settings = self._read_settings()
            st.to_config(settings)
        except ConfigError as exc:
            self._notify("Check the settings", str(exc), error=True)
            return
        with self._lock:
            if self._job is not None or self._part is None:
                return
            self._job = "generate"
            rev = (self._model_rev, self._parting_rev)
        with self.server.atomic():
            self.progress.animated = False
            self.progress.value = 0.0
            self.progress.visible = True
        self._set_status("Starting...")
        self._sync_controls()
        self._submit(self._generate_job, settings, rev)

    def _on_download(self, event: Any) -> None:
        client = event.client
        if client is None:
            return
        self.download.disabled = True
        self._submit(self._download_job, client)

    # ------------------------------------------------------------------
    # Parting requests

    def _request_offset_from_controls(self) -> None:
        with self._lock:
            parting = self._parting
        if parting is not None:
            self._request_parting(parting.direction, float(self.offset.value))

    def _request_parting(self, direction: np.ndarray, offset: float | None) -> None:
        with self._lock:
            if self._part is None:
                return
            self._parting_rev += 1
            self._pending_parting = _PartingRequest(
                np.asarray(direction, dtype=float), offset, self._parting_rev, self._model_rev
            )
            if self._parting_job_queued:
                return
            self._parting_job_queued = True
        self._submit(self._parting_job)

    def _request_reprepare(self) -> None:
        with self._lock:
            rev = self._model_rev
        self._submit(self._prepare_job, rev)

    # ------------------------------------------------------------------
    # Jobs (worker thread)

    def _load_job(self, path: Path, name: str, rev: int) -> None:
        with self._lock:
            if rev != self._model_rev:
                return
            self._job = "load"
        self._sync_controls()
        try:
            self._set_status(f"Loading {name}...")
            try:
                raw = load_mesh(path)
            except (MeshLoadError, OSError) as exc:
                self._notify(f"Could not open {path.name}", str(exc), error=True)
                return
            self._prepare_and_analyse(_Model(raw=raw, name=name), rev)
        finally:
            self._finish_job(rev)

    def _prepare_job(self, rev: int) -> None:
        with self._lock:
            model = self._model
            if model is None or rev != self._model_rev:
                return
            self._job = "load"
        self._sync_controls()
        try:
            self._prepare_and_analyse(model, rev)
        finally:
            self._finish_job(rev)

    def _prepare_and_analyse(self, model: _Model, rev: int) -> None:
        settings = self._read_settings()
        config = MoldConfig(
            units=settings.units,  # type: ignore[arg-type]
            scale=settings.scale,
            repair=settings.repair,
        )
        self._set_status("Checking and repairing the mesh...")
        try:
            part = prepare_part(model.raw, config, name=model.name)
        except MoldError as exc:
            self._notify(f"{model.name} cannot be used", str(exc), error=True)
            return
        areas = np.asarray(part.mesh.area_faces, dtype=float)
        with self._lock:
            if rev != self._model_rev:
                return
            self._model = model
            self._part = part
            self._areas = areas
            self._analysis = None
            self._parting = None
            self._choices = []
        self._clear_result()
        self.viewer.set_part(part.mesh)
        self.part_info.content = st.part_html(part, settings.units)
        self.readout.content = st.readout_html(None, None)
        self._refresh_auto_values()
        self._sync_controls()

        self._set_status("Analysing parting directions. Detailed models can take a minute.")
        self._show_busy(True)
        try:
            analysis = analyze_parting(
                part.mesh, None, None, draft_threshold_deg=settings.draft_threshold_deg
            )
        finally:
            self._show_busy(False)
        choices = st.direction_choices(analysis)
        with self._lock:
            if rev != self._model_rev:
                return
            self._analysis = analysis
            self._choices = choices
            self._parting = analysis
            self._parting_rev += 1
        with self._quiet():
            self.direction.options = tuple(c.label for c in choices)
            self.direction.value = st.AUTO_DIRECTION
        self._show_parting(analysis)

    def _parting_job(self) -> None:
        with self._lock:
            request = self._pending_parting
            self._pending_parting = None
            self._parting_job_queued = False
            part = self._part
            areas = self._areas
            analysis = self._analysis
            previous = self._parting
        if request is None or part is None or areas is None:
            return
        direction = request.direction / np.linalg.norm(request.direction)
        offset = request.offset
        if offset is None:
            offset = float(best_offset(part.mesh, direction))
        face_class = classify_faces(
            part.mesh, direction, offset, draft_threshold_deg=float(self.draft.value)
        )
        undercut, low_draft = st.surface_fractions(areas, face_class)
        parting = PartingResult(
            direction=direction,
            offset=float(offset),
            to_mold=mold_frame(part.mesh, direction, offset),
            face_class=np.asarray(face_class),
            undercut_fraction=undercut,
            low_draft_fraction=low_draft,
            candidates=list(analysis.candidates) if analysis is not None else [],
        )
        with self._lock:
            if request.parting_rev != self._parting_rev or request.model_rev != self._model_rev:
                return
            self._parting = parting
        # A new draft limit only recolours the part; the mold stays valid unless the plane moved.
        moved = previous is None or not (
            np.allclose(previous.direction, direction) and np.isclose(previous.offset, offset)
        )
        if moved:
            self._clear_result()
        self._show_parting(parting)

    def _generate_job(self, settings: st.MoldSettings, rev: tuple[int, int]) -> None:
        try:
            with self._lock:
                part, parting, analysis = self._part, self._parting, self._analysis
                current = (self._model_rev, self._parting_rev)
            if part is None or parting is None:
                raise MoldError("Load a model and wait for the parting analysis first.")
            # The automatic plane, untouched, leaves the choice of pull to the pipeline,
            # which may pick another candidate when side pieces work better along it.
            automatic = (
                analysis is not None
                and np.allclose(parting.direction, analysis.direction)
                and np.isclose(parting.offset, analysis.offset)
            )
            config = st.to_config(settings, None if automatic else parting)

            def progress(stage: str, fraction: float) -> None:
                self.progress.value = float(np.clip(100.0 * fraction, 0.0, 100.0))
                self._set_status(f"{stage}...")

            started = time.perf_counter()
            result = generate_mold(part, config, parting=parting, progress=progress)
            elapsed = time.perf_counter() - started
            with self._lock:
                if current != (self._model_rev, self._parting_rev) or rev[0] != self._model_rev:
                    log.info("Discarding a mold generated for inputs that have since changed")
                    self._set_status(READY_TEXT)
                    return
                self._result = result
                self._result_outdated = False
            self._show_result(result)
            self._set_status(f"Mold generated in {elapsed:.1f} s.")
        except EXPECTED_ERRORS as exc:
            self._notify("Mold generation failed", str(exc), error=True)
            self._set_status("Generation failed. Adjust the settings and try again.")
        except Exception as exc:
            log.exception("Mold generation failed")
            self._notify("Mold generation failed", f"{type(exc).__name__}: {exc}", error=True)
            self._set_status("Generation failed because of an internal error.")
        finally:
            self.progress.visible = False
            with self._lock:
                self._job = None
            self._sync_controls()

    def _download_job(self, client: Any) -> None:
        try:
            with self._lock:
                result = self._result
            if result is None:
                return
            data = zip_result(result)
            client.send_file_download(
                st.download_name(result.part.name), data, save_immediately=True
            )
        finally:
            self._sync_controls()

    def _finish_job(self, rev: int) -> None:
        """End a load job; a newer queued job, if any, will take over the status line."""
        with self._lock:
            self._job = None
            current = rev == self._model_rev
            has_part = self._part is not None
            has_parting = self._parting is not None
        if current:
            if has_parting:
                self._set_status(READY_TEXT)
            elif has_part:
                self._set_status("The parting analysis did not finish; see the notification.")
            else:
                self._set_status("")
        self._sync_controls()

    # ------------------------------------------------------------------
    # Display

    def _clear_result(self) -> None:
        """Forget the generated mold and show the bare part again."""
        with self._lock:
            if self._result is None:
                return
            self._result = None
            self._result_outdated = False
        self.viewer.clear_mold()
        with self._quiet():
            self.show_part.value = True
        self.viewer.set_visibility(part=True, pieces=self.show_pieces.value)
        self.result_folder.visible = False

    def _show_parting(self, parting: PartingResult) -> None:
        lo, hi, step = st.slider_range(*self.viewer.projection_range(parting.direction))
        with self._quiet(), self.server.atomic():
            if self.offset.min != lo or self.offset.max != hi:
                self.offset.min = lo
                self.offset.max = hi
                self.offset.step = step
                self.offset.precision = st.step_precision(step)
            # Rounded to the displayed precision; "+ 0.0" turns -0.0 into 0.0.
            shown = round(float(np.clip(parting.offset, lo, hi)), self.offset.precision)
            self.offset.value = shown + 0.0
        self.readout.content = st.readout_html(
            parting.undercut_fraction, parting.low_draft_fraction
        )
        self.viewer.show_classes(np.asarray(parting.face_class))
        self.viewer.show_plane(parting.direction, parting.offset, self._on_plane_drag)
        with self._lock:
            idle = self._job is None
            has_result = self._result is not None
        if idle and not has_result:
            self._set_status(READY_TEXT)
        self._sync_controls()

    def _show_result(self, result: MoldResult) -> None:
        part_extent = float(np.max(result.part.mesh.extents))
        block_extent = float(np.max(result.block_bounds[1] - result.block_bounds[0]))
        _, explode_max, explode_step = st.slider_range(0.0, 1.5 * block_extent, 300)
        with self._quiet(), self.server.atomic():
            self.explode.max = explode_max
            self.explode.step = explode_step
            self.explode.precision = st.step_precision(explode_step)
            self.explode.value = round(0.3 * part_extent, 1)
        self.viewer.show_mold(result, float(self.explode.value))
        self.viewer.frame_mold()
        layout = result.layout
        # Two halves on a curved surface report only the locked share, without face regions.
        if layout is not None and len(layout.face_region):
            self.viewer.show_piece_regions(
                st.face_pieces(
                    layout.face_region,
                    len(layout.caps),
                    layout.top_needed,
                    [piece.name for piece in result.pieces],
                )
            )
        info = summary(result)
        self.legend.content = st.swatch_html(st.legend_rows(info))
        self.summary.content = st.summary_html(info)
        self.warnings.content = st.warnings_html(result.warnings)
        self.warnings.visible = bool(result.warnings)
        self.result_note.visible = False
        self._sync_controls()
        if result.warnings:
            self._notify(
                "Mold generated with warnings",
                f"{len(result.warnings)} warning(s); see the Mold panel.",
            )


def _browser_host(host: str) -> str:
    return "localhost" if host in ("0.0.0.0", "::", "") else host


def run(
    model: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = True,
) -> None:
    """Start the GUI server, optionally preload ``model`` and block until Ctrl+C."""
    server = viser.ViserServer(host=host, port=port, label="moldgen", verbose=False)
    app = MoldGui(server)
    try:
        if model is not None:
            app.open_path(model)
        url = f"http://{_browser_host(host)}:{server.get_port()}"
        log.info("moldgen GUI running at %s (press Ctrl+C to stop)", url)
        if open_browser:
            webbrowser.open(url)
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Stopping the moldgen GUI")
    finally:
        app.close()
        server.stop()
