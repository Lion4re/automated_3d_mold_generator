"""Headless run of the GUI: a real viser server, no browser."""

import io
import socket
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
import shapely
import trimesh

from moldgen.booleans import difference

viser = pytest.importorskip("viser")

from moldgen.gui.app import MoldGui  # noqa: E402


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def app():
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    gui = MoldGui(server)
    yield gui
    gui.close()
    server.stop()


def _wait(app: MoldGui) -> None:
    """Block until every queued job has run (the worker is a single thread)."""
    app._executor.submit(lambda: None).result(timeout=300)


class _FakeClient:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def send_file_download(self, name: str, data: bytes, save_immediately: bool = False) -> None:
        self.files[name] = data


def test_open_generate_and_download(app, pawn, tmp_path):
    path = tmp_path / "pawn.stl"
    pawn.export(path)

    app.open_path(path)
    _wait(app)
    assert not app.generate.disabled
    assert "pawn" in app.part_info.content
    assert " %</span>" in app.readout.content

    app.pieces.value = "4 pieces"
    app._on_generate(SimpleNamespace(client=None))
    _wait(app)
    assert app.result_folder.visible
    assert len(app.viewer._pieces) == 4
    assert "Mold generated" in app.status.content

    client = _FakeClient()
    app._on_download(SimpleNamespace(client=client))
    _wait(app)
    names = zipfile.ZipFile(io.BytesIO(client.files["pawn_mold.zip"])).namelist()
    assert sum(name.endswith(".stl") for name in names) == 4
    assert app._pending_notices == []


def _box_with_crossed_holes() -> trimesh.Trimesh:
    """40 x 30 x 30 mm box with a hole along X at z = 9 and one along Y at z = -9."""
    holes = []
    for axis, z in (([0.0, 1.0, 0.0], 9.0), ([1.0, 0.0, 0.0], -9.0)):
        hole = trimesh.creation.cylinder(radius=3.0, height=60.0, sections=32)
        hole.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, axis))
        hole.apply_translation([0.0, 0.0, z])
        holes.append(hole)
    return difference(trimesh.creation.box(extents=(40.0, 30.0, 30.0)), holes)


def test_automatic_pieces_colour_the_part_by_piece(app, tmp_path):
    path = tmp_path / "crossed.stl"
    _box_with_crossed_holes().export(path)
    app.open_path(path)
    _wait(app)
    assert app.pieces.value == "Automatic" and app.max_pieces.visible

    app._on_generate(SimpleNamespace(client=None))
    _wait(app)
    pieces = app._result.pieces
    assert len(pieces) > 2
    assert set(app.viewer._part_nodes) <= {f"piece{i}" for i in range(len(pieces))} | {"filled"}
    assert len(app.viewer._part_nodes) > 2
    assert "side_1" in app.legend.content and "Removal order" in app.summary.content
    # Every piece, side pieces included, slides out along its own pull.
    to_part = app._result.parting.from_mold[:3, :3]
    for node, piece in zip(app.viewer._pieces, pieces, strict=True):
        moved = np.asarray(node.position)
        assert moved / np.linalg.norm(moved) == pytest.approx(to_part @ piece.pull, abs=1e-6)

    app.pieces.value = "2 pieces"
    app._on_mold_setting(None)
    assert not app.max_pieces.visible
    assert set(app.viewer._part_nodes) <= {"ok", "low_draft", "undercut"}


def test_curved_parting_surface_replaces_the_plane(app, tmp_path):
    t = np.linspace(-1, 1, 60)
    path = np.column_stack([40 * t, 12 * np.sin(2 * t), 10 * t**2 - 3])
    tube = trimesh.creation.sweep_polygon(shapely.Point(0, 0).buffer(4, quad_segs=16), path)
    stl = tmp_path / "tube.stl"
    tube.export(stl)
    app.open_path(stl)
    _wait(app)
    app.pieces.value = "2 pieces"
    app._on_mold_setting(None)
    assert app.parting_surface.visible and app.parting_surface.value == "Curved where needed"

    app._on_generate(SimpleNamespace(client=None))
    _wait(app)
    if app._result.surface is None:
        app.direction.value = "Z axis"
        app._on_direction(None)
        _wait(app)
        app._on_generate(SimpleNamespace(client=None))
        _wait(app)
    result = app._result
    assert result.surface is not None and not result.surface.flat
    assert "curved (up to" in app.summary.content
    assert app.viewer._part_nodes, "the part stays visible"
    surface = app.viewer._surface
    assert surface is not None
    # In the mold frame the surface spans the block and stays at the fitted heights.
    to_mold = np.linalg.inv(result.parting.from_mold)
    points = trimesh.transform_points(np.asarray(surface.vertices, float), to_mold)
    block = result.block_bounds
    assert points[:, :2].min(axis=0) == pytest.approx(block[0, :2], abs=1e-3)
    assert points[:, :2].max(axis=0) == pytest.approx(block[1, :2], abs=1e-3)
    assert points[:, 2] == pytest.approx(result.surface.height(points[:, :2]), abs=1e-3)

    # The surface takes the plane's place and follows the "Show plane" toggle.
    assert not surface.visible and not app.viewer._plane.visible
    app.show_pieces.value = False
    app._on_visibility(None)
    assert surface.visible and not app.viewer._plane.visible
    app.show_plane.value = False
    app._on_show_plane(None)
    assert not surface.visible
    app.show_plane.value = True
    app._on_show_plane(None)

    # Changing a setting brings the analysis and the flat plane back.
    app.keys.value = 2
    app._on_mold_setting(None)
    assert app.viewer._surface is None and app.viewer._plane.visible

    app.pieces.value = "4 pieces"
    app._on_mold_setting(None)
    assert not app.parting_surface.visible


def test_unreadable_upload_is_reported(app):
    upload = SimpleNamespace(name="broken.stl", content=b"solid nothing here")
    app._on_upload(SimpleNamespace(target=SimpleNamespace(value=upload)))
    _wait(app)

    [(title, body, error)] = app._pending_notices
    assert error and "broken.stl" in title
    assert "Traceback" not in body
    assert app.generate.disabled
