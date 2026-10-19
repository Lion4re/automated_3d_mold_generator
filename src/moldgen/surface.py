"""Curved parting surfaces: a height field ``z = h(x, y)`` between the two halves.

The top half is everything above the surface and comes off along +Z; the
bottom half is everything below it and comes off along -Z. Any height field
keeps the two halves apart while they move, so the surface may follow the
part's outline instead of a flat plane.

On every vertical line the surface must run through the cast, between its
lowest and highest point. Mold material above the cast then belongs to the
top and material below it to the bottom, so the only material left in the
wrong half is what lies between two layers of the cast on one vertical line:
a true undercut for this pull, which side pieces or filling have to handle.

When the plane z == 0 already runs through the cast on every line, it is
kept. Otherwise the surface runs through the middle of the cast's span on
each line over the part, which at the part's outline meets the outline
itself (the silhouette, where the cast's span shrinks to a point); beyond the
part it is as smooth as possible (a harmonic surface). It is held at the
height of the sprue and vents along their paths so the channels stay split
down the middle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from manifold3d import Manifold, Mesh64
from scipy.ndimage import zoom

from moldgen.gating import GatingPlan
from moldgen.parting import column_spans

CELL_FRACTION = 0.004
"""Grid spacing as a share of the block's diagonal."""

MIN_CELL_MM = 0.3
MAX_CELL_MM = 2.0

INSET_MM = 0.1
"""How far inside the cast the surface keeps from the cast's top and bottom."""

FLAT_SNAP_MM = 0.1
"""The plane z == 0 is kept if it misses the cast's span on no line by more than this."""

CHANNEL_MARGIN_CELLS = 2
"""Extra grid cells either side of a channel where the surface is held at its height."""

RELAX_ITERATIONS = 300
"""Relaxation sweeps per grid level."""

OVER_RELAXATION = 1.8

COARSEST_NODES = 12
"""Grid levels are halved until one side has about this many nodes."""

KEY_MAX_SLOPE = 0.08
"""Steepest surface (rise over run) on which registration keys are placed."""


@dataclass
class PartingSurface:
    """Heights on a regular grid; the surface is piecewise linear on the grid's triangles."""

    origin: np.ndarray
    """XY position of node (0, 0)."""
    cell: float
    heights: np.ndarray
    """(nx, ny) node heights."""
    covered: np.ndarray | None = None
    """(nx, ny) nodes whose vertical line meets the cast."""

    @property
    def flat(self) -> bool:
        return bool(np.all(self.heights == 0.0))

    @property
    def rise(self) -> float:
        """Largest distance of the surface from the plane z == 0 (mm)."""
        return float(np.max(np.abs(self.heights)))

    def height(self, xy: np.ndarray) -> np.ndarray:
        """Surface height above each XY point, on the same triangles as :meth:`below`."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        nx, ny = self.heights.shape
        u = (xy - self.origin) / self.cell
        i = np.clip(np.floor(u[:, 0]).astype(np.int64), 0, nx - 2)
        j = np.clip(np.floor(u[:, 1]).astype(np.int64), 0, ny - 2)
        fx = np.clip(u[:, 0] - i, 0.0, 1.0)
        fy = np.clip(u[:, 1] - j, 0.0, 1.0)
        h = self.heights
        h00, h10, h11, h01 = h[i, j], h[i + 1, j], h[i + 1, j + 1], h[i, j + 1]
        # Each cell is split along its (i, j)-(i+1, j+1) diagonal.
        lower = h00 + fx * (h10 - h00) + fy * (h11 - h10)
        upper = h00 + fy * (h01 - h00) + fx * (h11 - h01)
        return np.where(fx >= fy, lower, upper)

    def slope(self) -> np.ndarray:
        """Steepest rise over run at each node."""
        gx, gy = np.gradient(self.heights, self.cell)
        return np.hypot(gx, gy)

    def nodes(self) -> np.ndarray:
        """(nx * ny, 2) XY positions of the grid nodes, row-major in (i, j)."""
        nx, ny = self.heights.shape
        x = self.origin[0] + self.cell * np.arange(nx)
        y = self.origin[1] + self.cell * np.arange(ny)
        return np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)

    def below(self, floor: float) -> Manifold:
        """The solid between the surface and the plane ``z == floor``."""
        nx, ny = self.heights.shape
        xy = self.nodes()
        top = np.column_stack([xy, self.heights.ravel()])
        bottom = np.column_stack([xy, np.full(len(xy), floor)])
        vertices = np.vstack([top, bottom])
        n = len(top)
        index = np.arange(n).reshape(nx, ny)
        a, b = index[:-1, :-1].ravel(), index[1:, :-1].ravel()
        c, d = index[1:, 1:].ravel(), index[:-1, 1:].ravel()
        faces = [
            np.column_stack([a, b, c]),
            np.column_stack([a, c, d]),
            np.column_stack([a + n, c + n, b + n]),
            np.column_stack([a + n, d + n, c + n]),
        ]
        rim = np.concatenate(
            [index[:, 0], index[-1, 1:], index[-2::-1, -1], index[0, -2:0:-1], index[:1, 0]]
        )
        # The rim runs counter-clockwise seen from above, so these quads face outwards.
        p, q = rim[:-1], rim[1:]
        faces += [np.column_stack([p, q + n, q]), np.column_stack([p, p + n, q + n])]
        mesh = Mesh64(
            vert_properties=np.ascontiguousarray(vertices, dtype=np.float64),
            tri_verts=np.ascontiguousarray(np.vstack(faces), dtype=np.uint64),
        )
        return Manifold(mesh)


def fit_surface(
    cast: trimesh.Trimesh, block_bounds: np.ndarray, gating: GatingPlan | None = None
) -> PartingSurface:
    """The smoothest parting surface that keeps every vertical line's cast between the halves.

    ``cast`` is the cast in the mold frame (cavity and gating solids may simply
    be concatenated). The grid covers the block with one cell to spare on
    every side, so the surface's solid never shares a face with the block.
    """
    lo, hi = np.asarray(block_bounds, dtype=float)
    cell = float(np.clip(CELL_FRACTION * np.linalg.norm(hi - lo), MIN_CELL_MM, MAX_CELL_MM))
    origin = lo[:2] - cell
    shape = tuple(np.ceil((hi[:2] - lo[:2]) / cell).astype(np.int64) + 3)
    surface = PartingSurface(origin=origin, cell=cell, heights=np.zeros(shape))

    low, high = column_spans(cast, surface.nodes())
    inside = np.isfinite(low)
    surface.covered = inside.reshape(shape)
    span = np.where(inside, high - low, 0.0)  # inf - inf where a line misses the cast
    inset = np.minimum(INSET_MM, 0.25 * span)
    lower = np.where(inside, low + inset, -np.inf)
    upper = np.where(inside, high - inset, np.inf)
    if np.all(lower <= FLAT_SNAP_MM) and np.all(upper >= -FLAT_SNAP_MM):
        return surface  # the plane z == 0 runs through the cast everywhere
    middle = 0.5 * (np.where(inside, low, 0.0) + np.where(inside, high, 0.0))
    lower = np.where(inside, middle, -np.inf)
    upper = np.where(inside, middle, np.inf)
    if gating is not None:
        _hold_channels(surface, gating, lower, upper)
    surface.heights = _relax(lower.reshape(shape), upper.reshape(shape))
    return surface


def _hold_channels(
    surface: PartingSurface, gating: GatingPlan, lower: np.ndarray, upper: np.ndarray
) -> None:
    """Pin the surface to each channel's axis height along the channel, outside the part."""
    xy = surface.nodes()
    channels = [gating.sprue, *gating.vents]
    paths = [(c.start, c.end, c.radius) for c in channels]
    if gating.funnel is not None:
        f = gating.funnel
        paths.append((f.center - f.depth * f.axis, f.center, f.mouth_radius))
    margin = CHANNEL_MARGIN_CELLS * surface.cell
    for start, end, radius in paths:
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        along = end[:2] - start[:2]
        length2 = float(along @ along)
        t = np.zeros(len(xy)) if length2 == 0 else np.clip((xy - start[:2]) @ along / length2, 0, 1)
        nearest = start[:2] + t[:, None] * along
        near = np.linalg.norm(xy - nearest, axis=1) <= radius + margin
        value = start[2] + t * (end[2] - start[2])
        hold = near & (lower <= value) & (value <= upper)
        lower[hold] = value[hold]
        upper[hold] = value[hold]


def _relax(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Smoothest heights within ``[lower, upper]``: projected over-relaxation, coarse to fine."""
    levels = [(lower, upper)]
    while min(levels[-1][0].shape) > 2 * COARSEST_NODES:
        lo, hi = levels[-1]
        levels.append((lo[::2, ::2], hi[::2, ::2]))
    heights = np.clip(np.zeros(levels[-1][0].shape), *levels[-1])
    for lo, hi in reversed(levels):
        if heights.shape != lo.shape:
            factors = np.array(lo.shape) / np.array(heights.shape)
            heights = zoom(heights, factors, order=1, mode="nearest", grid_mode=True)
            heights = heights[: lo.shape[0], : lo.shape[1]]
        heights = _sweeps(np.clip(heights, lo, hi), lo, hi, RELAX_ITERATIONS)
    return heights


def _sweeps(h: np.ndarray, lower: np.ndarray, upper: np.ndarray, count: int) -> np.ndarray:
    """Red-black over-relaxation of the Laplace equation, clipped to the bounds every pass."""
    i, j = np.indices(h.shape)
    colours = [(i + j) % 2 == 0, (i + j) % 2 == 1]
    free = lower < upper
    if not free.any():
        return h
    for _ in range(count):
        for colour in colours:
            padded = np.pad(h, 1, mode="edge")  # zero slope at the grid's border
            mean = 0.25 * (
                padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
            )
            update = colour & free
            h = np.where(update, h + OVER_RELAXATION * (mean - h), h)
            h = np.clip(h, lower, upper)
    return h
