"""3D scene of the GUI: the coloured part, the parting plane or surface and the mold.

Everything lives under the ``/scene`` frame, which is shifted so the loaded
part sits centred on the ground grid. Inside ``/scene`` the coordinates are
the part's input frame in millimetres. Mold pieces are mapped back from the
mold frame, so the mold appears around the part where the user saw it.
A curved parting surface of a generated mold takes the place of the plane
until the mold is cleared.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh
import viser

from . import state as st

if TYPE_CHECKING:
    from moldgen.pipeline import MoldResult

FOV_RAD = float(np.radians(50.0))
PLANE_OPACITY = 0.16


class Viewer:
    """Owns the scene nodes. Safe to call from the worker and from viser callbacks."""

    def __init__(self, server: viser.ViserServer) -> None:
        self.server = server
        self._lock = threading.RLock()
        scene = server.scene
        scene.set_up_direction("+z")
        scene.configure_default_lights(enabled=True, cast_shadow=True)
        self._root = scene.add_frame("/scene", show_axes=False)
        self._grid = scene.add_grid(
            "/grid",
            plane="xy",
            infinite_grid=True,
            cell_size=10.0,
            section_size=50.0,
            cell_color=(222, 225, 230),
            section_color=(190, 195, 203),
            fade_distance=3000.0,
            shadow_opacity=0.12,
        )
        self._part_frame = scene.add_frame("/scene/part", show_axes=False)
        self._mold_frame = scene.add_frame("/scene/mold", show_axes=False, visible=False)

        self._vertices: np.ndarray | None = None
        self._faces: np.ndarray | None = None
        self._part_bottom = 0.0
        self._offset = np.zeros(3)
        self._part_nodes: dict[str, tuple[Any, np.ndarray, st.RGB]] = {}
        self._plane: Any = None
        self._plane_direction: np.ndarray | None = None
        self._plane_visible = True
        self._surface: Any = None
        self._pieces: list[Any] = []
        self._piece_dirs: np.ndarray | None = None
        self._piece_corners: list[np.ndarray] = []
        self._mold_axis: np.ndarray | None = None
        self._explode = 0.0
        self._pieces_visible = True

        camera = server.initial_camera
        camera.position = (180.0, -260.0, 170.0)
        camera.look_at = (0.0, 0.0, 30.0)
        camera.fov = FOV_RAD
        camera.near = 0.5
        camera.far = 20000.0

    # ------------------------------------------------------------------
    # Part

    def set_part(self, mesh: trimesh.Trimesh) -> None:
        """Show a new part in a neutral colour and frame the camera on it.

        Removes the parting plane and any mold from the previous part.
        """
        vertices, faces = st.shading_split(mesh)
        bounds = np.asarray(mesh.bounds, dtype=float)
        self.clear_mold()
        self.remove_plane()
        with self._lock:
            self._vertices, self._faces = vertices, faces
            self._part_bottom = float(bounds[0][2])
            for node, _, _ in self._part_nodes.values():
                node.remove()
            self._part_nodes = {}
            centre = bounds.mean(axis=0)
            self._offset = np.array([-centre[0], -centre[1], -bounds[0][2]])
        self._root.position = self._offset
        extent = float((bounds[1] - bounds[0]).max())
        cell, section = st.grid_spacing(extent)
        self._grid.cell_size = cell
        self._grid.section_size = section
        self._grid.fade_distance = max(30.0 * extent, 500.0)
        self.show_classes(None)
        self._update_floor()
        self.frame(bounds)

    def show_classes(self, face_class: np.ndarray | None) -> None:
        """Colour the part by face class (``None``: all neutral)."""
        with self._lock:
            if self._faces is None:
                return
            n_faces = len(self._faces)
        labels = np.zeros(n_faces, dtype=int) if face_class is None else np.asarray(face_class)
        self._paint({st.FACE_KEYS[c]: (labels == c, color) for c, color in st.FACE_COLORS.items()})

    def show_piece_regions(self, face_piece: np.ndarray) -> None:
        """Colour the part by the mold piece that releases each face (see :func:`face_pieces`)."""
        labels = np.asarray(face_piece)
        groups = {"filled": (labels == st.FILLED, st.FILLED_COLOR)}
        for i in range(int(labels.max(initial=-1)) + 1):
            groups[f"piece{i}"] = (labels == i, st.piece_color(i))
        self._paint(groups)

    def _paint(self, groups: dict[str, tuple[np.ndarray, st.RGB]]) -> None:
        """Show the part as one mesh per ``name: (face mask, colour)``, resending only changes."""
        with self._lock:
            if self._vertices is None or self._faces is None:
                return
            for name in list(self._part_nodes):
                node, mask, color = self._part_nodes[name]
                new = groups.get(name)
                if new is None or new[1] != color or not np.array_equal(new[0], mask):
                    node.remove()
                    del self._part_nodes[name]
            for name, (mask, color) in groups.items():
                if name in self._part_nodes or not mask.any():
                    continue
                v, f = st.submesh(self._vertices, self._faces, mask)
                node = self.server.scene.add_mesh_simple(f"/scene/part/{name}", v, f, color=color)
                self._part_nodes[name] = (node, mask.copy(), color)

    def projection_range(self, direction: np.ndarray) -> tuple[float, float]:
        """Extent of the part along ``direction``."""
        with self._lock:
            if self._vertices is None:
                return 0.0, 1.0
            return st.projection_range(self._vertices, direction)

    # ------------------------------------------------------------------
    # Parting plane

    def show_plane(
        self,
        direction: np.ndarray,
        offset: float,
        on_drag: Callable[[Any], None],
    ) -> None:
        """Place the parting plane; the quad and its handle are rebuilt for a new direction."""
        with self._lock:
            if self._vertices is None:
                return
            frame = st.plane_frame(self._vertices, direction, offset)
            same = (
                self._plane is not None
                and self._plane_direction is not None
                and np.allclose(self._plane_direction, direction)
            )
            if same:
                self._plane.position = frame.position
                return
            self.remove_plane()
            scene = self.server.scene
            handle = scene.add_transform_controls(
                "/scene/plane",
                scale=0.45 * max(frame.half_size),
                line_width=2.0,
                active_axes=(False, False, True),
                disable_sliders=True,
                disable_rotations=True,
                depth_test=False,
                translation_limits=((-1e6, 1e6),) * 3,
                wxyz=frame.wxyz,
                position=frame.position,
                visible=self._plane_visible and self._surface is None,
            )
            handle.on_update(on_drag)
            quad_vertices, quad_faces = st.quad_mesh(frame.half_size)
            scene.add_mesh_simple(
                "/scene/plane/quad",
                quad_vertices,
                quad_faces,
                color=st.PLANE_COLOR,
                opacity=PLANE_OPACITY,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
            )
            scene.add_line_segments(
                "/scene/plane/edge",
                st.quad_outline(frame.half_size),
                colors=st.PLANE_COLOR,
                thickness=1.5,
                thickness_units="screen",
            )
            self._plane = handle
            self._plane_direction = np.asarray(direction, dtype=float).copy()

    def move_plane(self, offset: float) -> None:
        """Move the existing plane along its direction (cheap; no geometry is resent)."""
        with self._lock:
            if self._plane is None or self._plane_direction is None or self._vertices is None:
                return
            frame = st.plane_frame(self._vertices, self._plane_direction, offset)
            self._plane.position = frame.position

    def set_plane_visible(self, visible: bool) -> None:
        """Show or hide the parting: the curved surface if there is one, else the plane."""
        with self._lock:
            self._plane_visible = visible
            if self._surface is not None:
                self._surface.visible = visible
            if self._plane is not None:
                self._plane.visible = visible and self._surface is None

    def remove_plane(self) -> None:
        with self._lock:
            if self._plane is not None:
                self._plane.remove()
            self._plane = None
            self._plane_direction = None

    def show_surface(self, result: MoldResult) -> None:
        """Show the result's curved parting surface in place of the plane."""
        if result.surface is None:
            return
        vertices, faces = st.surface_mesh(result.surface, result.block_bounds)
        vertices = trimesh.transform_points(vertices, result.parting.from_mold)
        with self._lock:
            self.remove_surface()
            self._surface = self.server.scene.add_mesh_simple(
                "/scene/surface",
                vertices,
                faces,
                color=st.PLANE_COLOR,
                opacity=PLANE_OPACITY,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
                visible=self._plane_visible,
            )
            if self._plane is not None:
                self._plane.visible = False

    def remove_surface(self) -> None:
        """Remove the curved parting surface and bring the plane back."""
        with self._lock:
            if self._surface is None:
                return
            self._surface.remove()
            self._surface = None
            if self._plane is not None:
                self._plane.visible = self._plane_visible

    # ------------------------------------------------------------------
    # Mold

    def show_mold(self, result: MoldResult, explode: float) -> None:
        """Show the mold pieces around the part, pulled apart by ``explode`` mm.

        Pieces of a planned layout move along their own pull. Quarter pieces
        share a pull, so without a layout they spread out from the block centre.
        """
        from_mold = result.parting.from_mold
        if result.layout is not None:
            dirs = np.array([p.pull for p in result.pieces], dtype=float)
        else:
            dirs = st.explode_directions(
                [p.mesh.bounds for p in result.pieces], result.block_bounds
            )
        meshes = []
        for piece in result.pieces:
            mesh = piece.mesh.copy()
            mesh.apply_transform(from_mold)
            meshes.append(mesh)
        self.clear_mold()
        with self._lock:
            for i, mesh in enumerate(meshes):
                vertices, faces = st.shading_split(mesh)
                self._pieces.append(
                    self.server.scene.add_mesh_simple(
                        f"/scene/mold/piece{i}",
                        vertices,
                        faces,
                        color=st.piece_color(i),
                    )
                )
            self._piece_dirs = dirs @ from_mold[:3, :3].T
            self._piece_corners = [trimesh.bounds.corners(mesh.bounds) for mesh in meshes]
            self._mold_axis = np.asarray(result.parting.direction, dtype=float)
            self._mold_frame.visible = self._pieces_visible
        self.show_surface(result)
        self.set_explode(explode)

    def clear_mold(self) -> None:
        self.remove_surface()
        with self._lock:
            for node in self._pieces:
                node.remove()
            self._pieces = []
            self._piece_dirs = None
            self._piece_corners = []
            self._mold_axis = None
            self._mold_frame.visible = False
        self._update_floor()

    def set_explode(self, distance: float) -> None:
        with self._lock:
            self._explode = float(distance)
            if self._piece_dirs is None:
                return
            for node, direction in zip(self._pieces, self._piece_dirs, strict=True):
                node.position = direction * self._explode
        self._update_floor()

    def set_visibility(self, *, part: bool, pieces: bool) -> None:
        with self._lock:
            self._pieces_visible = pieces
            self._part_frame.visible = part
            self._mold_frame.visible = pieces and bool(self._pieces)
        self._update_floor()

    # ------------------------------------------------------------------
    # Camera and floor

    def frame(self, bounds: np.ndarray, view_direction: np.ndarray | None = None) -> None:
        """Point every open tab (and tabs opened later) at ``bounds`` (part frame)."""
        with self._lock:
            offset = self._offset.copy()
        pose = st.camera_pose(np.asarray(bounds) + offset, FOV_RAD, view_direction)
        initial = self.server.initial_camera
        initial.look_at = pose.look_at
        initial.position = pose.position
        initial.near = pose.near
        initial.far = pose.far
        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.near = pose.near
                client.camera.far = pose.far
                # Position first: setting it moves look_at along with it.
                client.camera.position = pose.position
                client.camera.look_at = pose.look_at

    def frame_mold(self) -> None:
        """Frame the exploded mold, looking across the direction the halves separate in."""
        with self._lock:
            if self._piece_dirs is None or self._mold_axis is None:
                return
            points = np.vstack(
                [
                    corners + direction * self._explode
                    for corners, direction in zip(
                        self._piece_corners, self._piece_dirs, strict=True
                    )
                ]
            )
            axis = self._mold_axis
        bounds = np.array([points.min(axis=0), points.max(axis=0)])
        self.frame(bounds, st.side_view_direction(axis))

    def _update_floor(self) -> None:
        """Keep the grid just under the lowest visible geometry."""
        with self._lock:
            if self._vertices is None:
                self._grid.position = (0.0, 0.0, 0.0)
                return
            floor = self._part_bottom
            if self._piece_dirs is not None and self._pieces_visible:
                lows = [float(c[:, 2].min()) for c in self._piece_corners]
                floor = min(floor, st.scene_floor(lows, self._piece_dirs * self._explode))
            self._grid.position = (0.0, 0.0, floor + self._offset[2])
