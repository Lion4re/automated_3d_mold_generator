import numpy as np
import pytest
import trimesh

from moldgen.config import ConfigError, MoldConfig
from moldgen.gui import state as st
from moldgen.parting import FACE_LOW_DRAFT, FACE_OK, FACE_UNDERCUT, analyze_parting
from moldgen.surface import PartingSurface


def test_default_settings_map_back_to_the_default_config():
    config = st.to_config(st.MoldSettings.from_config())
    assert config.to_dict() == MoldConfig().to_dict()


def test_manual_values_override_the_automatic_ones(sphere):
    settings = st.MoldSettings.from_config()
    settings.wall_auto = False
    settings.wall_thickness = 7.5
    settings.sprue_auto = False
    settings.sprue_diameter = 4.0
    settings.shrinkage_override = True
    settings.shrinkage_percent = 1.5
    settings.pieces = 4
    parting = analyze_parting(sphere, np.array([0.0, 0.0, 1.0]), 2.0)

    config = st.to_config(settings, parting)

    assert config.wall_thickness == 7.5
    assert config.sprue_diameter == 4.0
    assert config.shrinkage == pytest.approx(0.015)
    assert config.pieces == 4
    assert config.direction == (0.0, 0.0, 1.0)
    assert config.parting_offset == pytest.approx(2.0)


def test_automatic_pieces_are_the_default_and_carry_the_maximum():
    settings = st.MoldSettings.from_config()
    assert next(iter(st.PIECE_OPTIONS)) == st.AUTO_PIECES
    assert settings.pieces == st.PIECE_OPTIONS[st.AUTO_PIECES] == "auto"
    settings.max_pieces = 8
    config = st.to_config(settings)
    assert (config.pieces, config.max_pieces) == ("auto", 8)


def test_invalid_settings_raise_config_error():
    settings = st.MoldSettings.from_config()
    settings.shrinkage_override = True
    settings.shrinkage_percent = 25.0
    with pytest.raises(ConfigError):
        st.to_config(settings)


def test_direction_choices_list_auto_then_candidates_then_axes(spool):
    analysis = analyze_parting(spool)
    choices = st.direction_choices(analysis)
    labels = [c.label for c in choices]

    assert labels[0] == st.AUTO_DIRECTION
    assert labels[-3:] == list(st.AXIS_DIRECTIONS)
    assert len(choices) == 1 + len(analysis.candidates) + 3
    assert choices[0].offset == analysis.offset
    assert all(c.offset is None for c in choices[-3:])
    assert st.direction_label([0.0, 0.0, -1.0]) == "-Z"


def test_surface_fractions_are_area_weighted():
    areas = np.array([1.0, 2.0, 3.0, 4.0])
    classes = np.array([FACE_OK, FACE_LOW_DRAFT, FACE_UNDERCUT, FACE_UNDERCUT])
    assert st.surface_fractions(areas, classes) == pytest.approx((0.7, 0.2))


@pytest.mark.parametrize(("lo", "hi"), [(-16.23, 16.21), (0.0, 1234.5), (3.0, 3.004)])
def test_slider_range_covers_the_span_with_a_round_step(lo, hi):
    low, high, step = st.slider_range(lo, hi)
    assert low <= lo and high >= hi
    mantissa = step / 10 ** np.floor(np.log10(step))
    assert round(mantissa, 6) in (1.0, 2.0, 5.0)
    assert round(step, st.step_precision(step)) == step


def test_plane_frame_lies_on_the_plane_and_faces_the_direction(rng):
    vertices = rng.normal(size=(200, 3)) * 10
    direction = np.array([1.0, 2.0, 2.0]) / 3.0
    frame = st.plane_frame(vertices, direction, 4.0)

    assert np.dot(frame.position, direction) == pytest.approx(4.0)
    rotation = trimesh.transformations.quaternion_matrix(frame.wxyz)[:3, :3]
    assert rotation @ [0.0, 0.0, 1.0] == pytest.approx(direction)


def test_shading_split_keeps_face_order_and_geometry(cylinder):
    vertices, faces = st.shading_split(cylinder)
    assert len(faces) == len(cylinder.faces)
    assert np.allclose(vertices[faces], cylinder.triangles, atol=1e-4)
    # The sharp rims are split, so there are more vertices than in the input.
    assert len(vertices) > len(cylinder.vertices)


def test_halves_explode_along_the_parting_direction():
    block = np.array([[-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]])
    top = np.array([[-10.0, -10.0, 0.0], [10.0, 10.0, 10.0]])
    bottom = np.array([[-10.0, -10.0, -10.0], [10.0, 10.0, 0.0]])
    dirs = st.explode_directions([top, bottom], block)
    assert dirs == pytest.approx(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))


def test_face_regions_map_to_pieces_in_removal_order():
    # Regions: side_1, side_2, top, bottom; -1 is filled.
    regions = np.array([0, 1, 2, 3, -1])
    names = ["side_1", "side_2", "top", "bottom"]
    assert st.face_pieces(regions, 2, True, names).tolist() == [0, 1, 2, 3, st.FILLED]
    # Without a top piece its region belongs to the bottom, which is now third.
    merged = st.face_pieces(regions, 2, False, ["side_1", "side_2", "bottom"])
    assert merged.tolist() == [0, 1, 2, 2, st.FILLED]


def test_legend_lists_each_piece_with_its_pull_and_colour_then_the_filling():
    pieces = [{"name": "side_1", "pull_label": "+Y"}, {"name": "top", "pull_label": "+Z"}]
    layout = {"locked_fraction": 0.021, "filled_volume_cm3": 0.4}
    rows = st.legend_rows({"pieces": pieces, "layout": layout})
    assert rows == [
        ("side_1", "+Y", st.PIECE_COLORS[0]),
        ("top", "+Z", st.PIECE_COLORS[1]),
        ("filled", "2.1 %", st.FILLED_COLOR),
    ]
    assert len(set(st.PIECE_COLORS)) == 10
    assert st.FILLED_COLOR not in st.PIECE_COLORS
    assert len(st.legend_rows({"pieces": pieces, "layout": None})) == 2


def test_text_is_escaped_and_incompatible_materials_are_flagged():
    assert "<b>" not in st.info_html([("Name", "<b>x</b>")])
    assert "softens" in st.material_html("pewter", "pla")
    assert "softens" not in st.material_html("resin", "pla")
    assert st.download_name("../my part!") == "my_part_mold.zip"


def test_parting_surface_setting_maps_to_the_config():
    settings = st.MoldSettings.from_config()
    assert st.label_for(st.PARTING_SURFACE_OPTIONS, settings.parting_surface) == (
        "Curved where needed"
    )
    settings.parting_surface = st.PARTING_SURFACE_OPTIONS["Flat"]
    assert st.to_config(settings).parting_surface == "flat"
    assert st.MoldSettings.from_config(MoldConfig(parting_surface="flat")).parting_surface == "flat"


def test_surface_mesh_follows_the_surface_inside_the_block(rng):
    heights = rng.normal(size=(9, 7))
    surface = PartingSurface(origin=np.array([-1.0, -2.0]), cell=1.0, heights=heights)
    block = np.array([[-0.5, -1.25, -5.0], [5.6, 2.3, 5.0]])

    vertices, faces = st.surface_mesh(surface, block)

    assert vertices[:, :2].min(axis=0) == pytest.approx(block[0, :2], abs=1e-5)
    assert vertices[:, :2].max(axis=0) == pytest.approx(block[1, :2], abs=1e-5)
    # Every point of every triangle lies on the surface, so the grid is split on the same
    # diagonal as PartingSurface.height and the clipping adds no new bends.
    corners = vertices[faces]
    weights = rng.dirichlet(np.ones(3), size=len(faces))
    points = np.einsum("fk,fkd->fd", weights, corners)
    assert points[:, 2] == pytest.approx(surface.height(points[:, :2]), abs=1e-5)
    area = trimesh.triangles.area(corners).sum()
    assert area >= np.prod(block[1, :2] - block[0, :2]) - 1e-3


def test_summary_names_a_curved_parting_and_its_rise():
    info = {
        "mold": {
            "outer_size_mm": [10, 10, 10],
            "wall_thickness_mm": 5.0,
            "sprue_diameter_mm": 6.0,
            "funnel": False,
            "vents": 0,
            "keys": 4,
            "key_clearance_mm": 0.25,
        },
        "parting": {
            "direction_label": "+Z",
            "undercut_fraction": 0.1,
            "surface": "curved",
            "surface_rise_mm": 4.24,
        },
        "material": {"cast_volume_cm3": 1.0},
        "pieces": [],
        "layout": {
            "side_pieces": 0,
            "locked_fraction": 0.001,
            "filled_volume_cm3": 0.0,
            "remaining_locked_fraction": 0.001,
        },
    }
    html = st.summary_html(info)
    # The undercut is the curved mold's own, not the flat-plane figure.
    assert "+Z, curved surface, up to 4.2 mm from flat" in html
    assert "0.1 % of the surface" in html
    assert "Removal order" not in html and "Filled" not in html
    info["parting"].update(surface="flat", surface_rise_mm=0.0)
    info["layout"] = None
    html = st.summary_html(info)
    assert "+Z, flat plane" in html and "10.0 % of the surface" in html
