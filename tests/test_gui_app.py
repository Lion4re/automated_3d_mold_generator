"""Headless run of the GUI: a real viser server, no browser."""

import io
import socket
import zipfile
from types import SimpleNamespace

import pytest

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


def test_unreadable_upload_is_reported(app):
    upload = SimpleNamespace(name="broken.stl", content=b"solid nothing here")
    app._on_upload(SimpleNamespace(target=SimpleNamespace(value=upload)))
    _wait(app)

    [(title, body, error)] = app._pending_notices
    assert error and "broken.stl" in title
    assert "Traceback" not in body
    assert app.generate.disabled
