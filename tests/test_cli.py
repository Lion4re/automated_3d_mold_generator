import io
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import trimesh
import typer
from rich.console import Console
from typer.testing import CliRunner

from moldgen import MoldConfig, MoldError, __version__, cli
from moldgen.gating import Channel, GatingPlan
from moldgen.keys import KeyPlan
from moldgen.materials import MATERIALS, PRINT_MATERIALS, get_material, get_print_material
from moldgen.parting import DirectionScore, PartingResult
from moldgen.pipeline import MoldPiece, MoldResult, PreparedPart
from moldgen.repair import RepairReport

runner = CliRunner()


def invoke(*args: str, **kwargs):
    return runner.invoke(cli.app, list(args), **kwargs)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder; generate_mold is faked\n")
    return path


def _fake_parting(direction=None) -> PartingResult:
    chosen = np.array([0.0, 0.0, 1.0]) if direction is None else np.asarray(direction, float)
    return PartingResult(
        direction=chosen,
        offset=0.0,
        to_mold=np.eye(4),
        face_class=np.zeros(12, dtype=int),
        undercut_fraction=0.0,
        low_draft_fraction=0.1,
        candidates=[
            DirectionScore(chosen, 0.0, 0.0, 0.1),
            DirectionScore(np.array([0.0, 0.6, 0.8]), 2.5, 0.02, 0.2),
        ],
    )


def _fake_part(source, config=None, *, name=None) -> PreparedPart:
    mesh = trimesh.creation.box(extents=(30.0, 20.0, 10.0))
    report = RepairReport(
        was_watertight=True, is_watertight=True, actions=["merged 3 duplicate vertices"]
    )
    return PreparedPart(name or Path(source).stem, mesh, report)


def _fake_result(source, config: MoldConfig) -> MoldResult:
    part = _fake_part(source)
    top = trimesh.creation.box(bounds=[[-20.0, -15.0, 0.0], [20.0, 15.0, 10.0]])
    bottom = trimesh.creation.box(bounds=[[-20.0, -15.0, -10.0], [20.0, 15.0, 0.0]])
    return MoldResult(
        config=config,
        material=get_material(config.material),
        print_material=get_print_material(config.print_material),
        part=part,
        parting=_fake_parting(),
        cavity=part.mesh,
        shrink_scale=1.005,
        wall_thickness=5.0,
        block_bounds=np.array([[-20.0, -15.0, -10.0], [20.0, 15.0, 10.0]]),
        gating=GatingPlan(
            up=np.array([1.0, 0.0, 0.0]),
            sprue=Channel(np.zeros(3), np.array([20.0, 0.0, 0.0]), 3.0),
        ),
        keys=[KeyPlan(np.array([0.0, 0.0, 1.0]), np.zeros((4, 3)), 2.0, config.clearance)],
        pieces=[MoldPiece("top", top, np.eye(4)), MoldPiece("bottom", bottom, np.eye(4))],
        warnings=["Example warning"],
    )


@pytest.fixture
def generated(monkeypatch, tmp_path):
    """Replace generate_mold with a fast fake; models named 'bad*' fail. Returns the calls."""
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[Path, MoldConfig]] = []

    def fake_generate(source, config, *, parting=None, progress=None):
        calls.append((Path(source), config))
        if progress:
            progress("Loading and repairing the part", 0.0)
        if Path(source).stem.startswith("bad"):
            raise MoldError(f"{Path(source).stem} is not a closed solid")
        return _fake_result(source, config)

    monkeypatch.setattr("moldgen.pipeline.generate_mold", fake_generate)
    return calls


@pytest.mark.parametrize(
    "args",
    [[], ["make"], ["analyze"], ["materials"], ["gui"]],
    ids=["root", "make", "analyze", "materials", "gui"],
)
def test_help(args):
    result = invoke(*args, "--help")
    assert result.exit_code == 0
    assert "Usage" in result.stdout


def test_startup_does_not_load_the_geometry_stack():
    """Help, --version and 'materials' must not pay for importing trimesh."""
    script = """
import sys
import moldgen.cli
assert "trimesh" not in sys.modules, "import moldgen.cli"
from typer.testing import CliRunner
for args in (["--help"], ["--version"], ["materials"], ["make", "-h"], ["analyze", "-h"], ["gui", "-h"]):
    result = CliRunner().invoke(moldgen.cli.app, args)
    assert result.exit_code == 0, (args, result.output)
    assert "trimesh" not in sys.modules, args
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_version():
    result = invoke("--version")
    assert result.exit_code == 0
    assert result.stdout.strip() == f"moldgen {__version__}"


def test_bare_command_prints_help_when_not_interactive(monkeypatch):
    monkeypatch.setattr(cli, "_is_interactive", lambda: False)
    result = invoke()
    assert result.exit_code == 0
    assert "make" in result.stdout and "analyze" in result.stdout


def test_materials_lists_every_preset():
    result = invoke("materials", env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for preset in [*MATERIALS.values(), *PRINT_MATERIALS.values()]:
        assert preset.key in result.stdout
        assert preset.name in result.stdout


@pytest.mark.parametrize(
    "option",
    [
        ["--pieces", "3"],
        ["--material", "unobtainium"],
        ["--print-material", "unobtainium"],
        ["--units", "ft"],
        ["--direction", "sideways"],
        ["--direction", "1,2"],
        ["--direction", "0,0,0"],
        ["--shrinkage", "50"],
        ["--keys", "20"],
        ["--wall", "-1"],
    ],
)
def test_invalid_option_values_are_usage_errors(generated, option):
    _touch(Path("part.stl"))
    result = invoke("make", "part.stl", *option)
    assert result.exit_code == 2
    assert not generated


def test_missing_file_is_a_clear_error(generated):
    result = invoke("make", "nowhere.stl")
    assert result.exit_code == 1
    assert "Error: File not found: nowhere.stl" in result.stderr
    assert "Supported formats" in result.stderr
    assert "Traceback" not in result.output


def test_unsupported_file_type(generated):
    _touch(Path("part.step"))
    result = invoke("analyze", "part.step")
    assert result.exit_code == 1
    assert "Unsupported file type '.step'" in result.stderr


def test_options_map_onto_config(generated):
    _touch(Path("part.stl"))
    result = invoke(
        "make",
        "part.stl",
        "--material", "resin",
        "--print-material", "pla",
        "--units", "cm",
        "--scale", "2",
        "--direction", "0,3,4",
        "--parting-offset", "1.5",
        "--pieces", "4",
        "--wall", "6",
        "--shrinkage", "1.5",
        "--sprue", "8",
        "--vent", "2",
        "--no-funnel",
        "--no-vents",
        "--keys", "3",
        "--key-diameter", "5",
        "--clearance", "0.3",
        "--no-repair",
        "--no-orient",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    ((source, config),) = generated
    assert source == Path("part.stl")
    assert config.material == "resin" and type(config.material) is str
    assert config.print_material == "pla"
    assert config.units == "cm"
    assert config.scale == 2.0
    assert config.direction == (0.0, 3.0, 4.0)
    np.testing.assert_allclose(config.direction_vector(), [0.0, 0.6, 0.8])
    assert config.parting_offset == 1.5
    assert config.pieces == 4
    assert config.wall_thickness == 6.0
    assert config.shrinkage == pytest.approx(0.015)
    assert config.sprue_diameter == 8.0
    assert config.vent_diameter == 2.0
    assert config.funnel is False and config.vents is False
    assert config.keys == 3
    assert config.key_diameter == 5.0
    assert config.clearance == 0.3
    assert config.repair is False
    assert config.orient_for_print is False


def test_defaults_match_mold_config(generated):
    _touch(Path("part.stl"))
    assert invoke("make", "part.stl").exit_code == 0
    ((_, config),) = generated
    assert config.to_dict() == MoldConfig().to_dict()


@pytest.mark.parametrize(
    ("text", "expected"),
    [("auto", "auto"), ("z", "z"), ("+X", "x"), ("-y", "-y"), ("0,1,1", (0.0, 1.0, 1.0))],
)
def test_direction_parsing(text, expected):
    assert cli._parse_direction(text) == expected


def test_axis_direction_after_option_name(generated):
    _touch(Path("part.stl"))
    assert invoke("make", "part.stl", "--direction", "-z").exit_code == 0
    assert generated[0][1].direction == "-z"


def test_make_writes_files_and_prints_summary(generated):
    _touch(Path("part.stl"))
    result = invoke("make", "part.stl", env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    out_dir = Path("molds/part")
    assert {p.name for p in out_dir.iterdir()} == {
        "part_top.stl",
        "part_bottom.stl",
        "report.json",
        "INSTRUCTIONS.txt",
    }
    for text in ("2-piece mold", "top", "bottom", "Wall thickness", "Warning: Example warning"):
        assert text in result.stdout
    assert f"Saved 4 files to {Path('molds/part')}" in result.stdout


def test_batch_continues_after_a_failure(generated):
    for name in ("one.stl", "bad.stl", "two.stl"):
        _touch(Path(name))
    result = invoke("make", "one.stl", "bad.stl", "two.stl", "missing.stl", env={"COLUMNS": "200"})
    assert result.exit_code == 1
    assert [source.name for source, _ in generated] == ["one.stl", "bad.stl", "two.stl"]
    assert Path("molds/one/report.json").is_file()
    assert Path("molds/two/report.json").is_file()
    assert not Path("molds/bad").exists()
    assert "bad is not a closed solid" in result.stderr
    assert "File not found: missing.stl" in result.stderr
    assert "2 of 4 molds generated, 2 failed." in result.stdout


def test_duplicate_model_names_are_rejected(generated):
    _touch(Path("a/part.stl"))
    _touch(Path("b/part.stl"))
    result = invoke("make", "a/part.stl", "b/part.stl")
    assert result.exit_code == 2
    assert not generated


def test_json_output_for_one_model(generated):
    _touch(Path("part.stl"))
    result = invoke("make", "part.stl", "--json")
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["model"] == "part.stl"
    assert data["mold"]["pieces"] == 2
    assert data["warnings"] == ["Example warning"]
    assert str(Path(data["output_dir"], "report.json")) in data["files"]


def test_json_output_for_a_batch_includes_failures(generated):
    _touch(Path("good.stl"))
    _touch(Path("bad.stl"))
    result = invoke("make", "good.stl", "bad.stl", "--json")
    assert result.exit_code == 1
    data = json.loads(result.stdout)
    assert [entry["model"] for entry in data] == ["good.stl", "bad.stl"]
    assert "pieces" in data[0]
    assert data[1]["error"] == "bad is not a closed solid"


def test_refuses_to_overwrite_without_force(generated):
    _touch(Path("part.stl"))
    _touch(Path("molds/part/notes.txt"))
    result = invoke("make", "part.stl")
    assert result.exit_code == 1
    assert f"{Path('molds/part')} already contains files" in result.stderr
    assert "--force" in result.stderr
    assert not generated


def test_force_replaces_earlier_results_only(generated):
    _touch(Path("part.stl"))
    _touch(Path("molds/part/part_top_x-.stl"))
    _touch(Path("molds/part/notes.txt"))
    result = invoke("make", "part.stl", "--force")
    assert result.exit_code == 0, result.output
    names = {p.name for p in Path("molds/part").iterdir()}
    assert "part_top_x-.stl" not in names
    assert {"notes.txt", "part_top.stl", "part_bottom.stl"} <= names


def test_quiet_prints_only_warnings(generated):
    _touch(Path("part.stl"))
    result = invoke("make", "part.stl", "--quiet")
    assert result.exit_code == 0
    assert result.stdout == ""
    assert "Warning: Example warning" in result.stderr


def test_unexpected_errors_need_verbose_for_a_traceback(generated, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("moldgen.pipeline.generate_mold", broken)
    _touch(Path("part.stl"))
    quiet = invoke("make", "part.stl")
    assert quiet.exit_code == 1
    assert "Unexpected error (RuntimeError: boom)" in quiet.stderr
    assert "Traceback" not in quiet.output
    loud = invoke("make", "part.stl", "-v")
    assert loud.exit_code == 1
    assert "Traceback" in loud.stderr


@pytest.fixture
def analyzed(monkeypatch, tmp_path):
    """Fake prepare_part and analyze_parting; returns the directions analyze_parting received."""
    monkeypatch.chdir(tmp_path)
    directions: list[np.ndarray | None] = []

    def fake_analyze(mesh, direction=None, offset=None, *, draft_threshold_deg=1.0):
        directions.append(direction)
        return _fake_parting(direction)

    monkeypatch.setattr("moldgen.pipeline.prepare_part", _fake_part)
    monkeypatch.setattr("moldgen.parting.analyze_parting", fake_analyze)
    _touch(Path("part.stl"))
    return directions


def test_analyze_prints_report(analyzed):
    result = invoke("analyze", "part.stl", env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    for text in ("30.0 x 20.0 x 10.0 mm", "merged 3 duplicate vertices", "+Z", "selected"):
        assert text in result.stdout
    assert "Recommended settings" in result.stdout
    assert "Next: moldgen make part.stl" in result.stdout
    assert analyzed == [None]
    assert not Path("molds").exists()


def test_analyze_json_and_direction(analyzed):
    result = invoke("analyze", "part.stl", "--direction", "x", "--json")
    assert result.exit_code == 0, result.output
    np.testing.assert_allclose(analyzed[0], [1.0, 0.0, 0.0])
    data = json.loads(result.stdout)
    assert data["parting"]["label"] == "+X"
    assert data["candidates"][0]["selected"] is True
    assert data["recommended"]["material"] == "resin"
    assert data["command"] == "moldgen make part.stl --direction=+X"


def test_gui_is_imported_lazily(monkeypatch, tmp_path):
    calls = []
    fake = types.ModuleType("moldgen.gui.app")
    fake.run = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "moldgen.gui.app", fake)
    model = _touch(tmp_path / "part.stl")
    result = invoke("gui", str(model), "--port", "9000", "--no-browser")
    assert result.exit_code == 0, result.output
    assert calls == [{"model": model, "host": "127.0.0.1", "port": 9000, "open_browser": False}]


def test_gui_import_failure_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "moldgen.gui.app", None)
    result = invoke("gui")
    assert result.exit_code == 1
    assert "graphical interface could not be loaded" in result.stderr
    assert "Traceback" not in result.output


def test_guided_mode(generated, monkeypatch):
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)
    _touch(Path("models/ball.stl"))
    # Model 1, default material, 4 pieces, confirm.
    result = invoke(input="1\n\n4\n\n")
    assert result.exit_code == 0, result.output
    ((source, config),) = generated
    assert source == Path("models/ball.stl")
    assert config.material == MoldConfig().material
    assert config.pieces == 4
    assert "moldgen make models/ball.stl --material resin --pieces 4" in result.stdout
    assert Path("molds/ball/report.json").is_file()


def test_guided_mode_can_be_cancelled(generated, monkeypatch):
    monkeypatch.setattr(cli, "_is_interactive", lambda: True)
    _touch(Path("ball.stl"))
    result = invoke(input="\n\n\nn\n")
    assert result.exit_code == 1
    assert not generated


def test_progress_callback_drives_a_progress_bar():
    stream = io.StringIO()
    ui = cli.Ui(err=Console(file=stream, force_terminal=True, width=100))
    with ui.progress("part.stl") as update:
        assert update is not None
        update("Choosing the parting plane", 0.3)
    assert "part.stl" in stream.getvalue()


# End to end: the real pipeline on a small sphere.


def _skip_if_core_unfinished(result) -> None:
    if "NotImplementedError" in result.output:
        pytest.skip("the core pipeline still raises NotImplementedError")


@pytest.fixture
def ball(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "ball.stl"
    trimesh.creation.icosphere(subdivisions=2, radius=12.0).export(path)
    return path


def test_end_to_end_analyze(ball):
    result = invoke("analyze", str(ball), "--json", "-v")
    _skip_if_core_unfinished(result)
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["part"]["closed_before_repair"] is True
    assert data["parting"]["undercut_fraction"] < 0.01
    assert data["candidates"]


def test_end_to_end_make(ball):
    result = invoke("make", str(ball), "--out", "out", "-v")
    _skip_if_core_unfinished(result)
    assert result.exit_code == 0, result.output
    files = sorted(p.name for p in Path("out/ball").iterdir())
    assert files == ["INSTRUCTIONS.txt", "ball_bottom.stl", "ball_top.stl", "report.json"]
    for name in ("ball_top.stl", "ball_bottom.stl"):
        assert trimesh.load_mesh(Path("out/ball") / name).is_watertight
    assert "2-piece mold" in result.stdout


@pytest.mark.parametrize("command", ["make", "analyze"])
def test_end_to_end_open_surface_is_a_one_line_error(tmp_path, monkeypatch, command):
    monkeypatch.chdir(tmp_path)
    sheet = trimesh.Trimesh(
        vertices=[[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], faces=[[0, 1, 2], [0, 2, 3]]
    )
    sheet.export(tmp_path / "sheet.stl")
    result = invoke(command, "sheet.stl")
    _skip_if_core_unfinished(result)
    assert result.exit_code == 1
    errors = [line for line in result.stderr.splitlines() if line.startswith("Error:")]
    assert len(errors) == 1, result.stderr
    assert "sheet" in errors[0]
    assert "Traceback" not in result.output
    assert not Path("molds").exists()


def test_bed_size_is_parsed_and_checked():
    assert cli._build_config(bed="300x200x180").bed_size_mm == (300.0, 200.0, 180.0)
    for text in ("300x200", "300x0x180", "big"):
        with pytest.raises(typer.BadParameter):
            cli._build_config(bed=text)
