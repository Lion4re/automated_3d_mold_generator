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

CUT_RELAX_ITERATIONS = 80
"""Relaxation sweeps per level for a side piece's cut, which is fixed on most of its grid."""

CUT_RAISE_ROUNDS = 3
"""Times a side piece's cut is raised over faces it may not take, then refitted."""


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
        """The solid between the surface and the plane ``z == floor`` under it."""
        return self.between(floor)

    def between(self, level: float) -> Manifold:
        """The solid between the surface and the plane ``z == level``, above or below it all."""
        nx, ny = self.heights.shape
        xy = self.nodes()
        flat = np.full(len(xy), level)
        heights = self.heights.ravel()
        upper, lower = (heights, flat) if level <= heights.min() else (flat, heights)
        top = np.column_stack([xy, upper])
        bottom = np.column_stack([xy, lower])
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


def _relax(lower: np.ndarray, upper: np.ndarray, iterations: int = RELAX_ITERATIONS) -> np.ndarray:
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
        heights = _sweeps(np.clip(heights, lo, hi), lo, hi, iterations)
    return heights


def _sweeps(h: np.ndarray, lower: np.ndarray, upper: np.ndarray, count: int) -> np.ndarray:
    """Red-black over-relaxation of the Laplace equation on the nodes the bounds leave free.

    Every bound is either a fixed value (``lower == upper``) or none at all, so
    free nodes need no clipping and fixed ones are never touched.
    """
    free = lower < upper
    if not free.any():
        return h
    h = np.where(free, h, lower)
    i, j = np.indices(h.shape)
    colours = [free & ((i + j) % 2 == 0), free & ((i + j) % 2 == 1)]
    padded = np.empty((h.shape[0] + 2, h.shape[1] + 2))
    for _ in range(count):
        for colour in colours:
            # Edge rows and columns repeat the border: zero slope at the grid's edge.
            padded[1:-1, 1:-1] = h
            padded[0, 1:-1], padded[-1, 1:-1] = h[0], h[-1]
            padded[1:-1, 0], padded[1:-1, -1] = h[:, 0], h[:, -1]
            mean = 0.25 * (
                padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
            )
            h[colour] += OVER_RELAXATION * (mean[colour] - h[colour])
    return h


@dataclass
class CutSurface:
    """A curved cut for a side piece: the piece lies where ``dot(p, n) >= g(u, v)``.

    ``rows`` are ``(e1, e2, n)``: two unit vectors across the pull and the
    pull ``n`` itself, with ``u = dot(p, e1)`` and ``v = dot(p, e2)``. ``field``
    holds ``g`` on a grid over ``(u, v)``.
    """

    rows: np.ndarray
    field: PartingSurface

    def depth(self, points: np.ndarray) -> np.ndarray:
        """The cut's position along the pull on the line through each point."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        return self.field.height(points @ self.rows[:2].T)

    def beyond(self, points: np.ndarray) -> np.ndarray:
        """How far each point lies past the cut along the pull (negative: behind it)."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        return points @ self.rows[2] - self.depth(points)

    def solid(self, reach: float) -> Manifold:
        """Everything past the cut, up to ``reach`` along the pull, inside the grid."""
        local = self.field.between(float(self.field.heights.max()) + reach)
        return local.transform(np.column_stack([self.rows.T, np.zeros(3)]))


def fit_cut(
    cast: trimesh.Trimesh,
    rows: np.ndarray,
    uv_bounds: np.ndarray,
    cell: float,
    keep_behind: np.ndarray | None = None,
    keep_beyond: np.ndarray | None = None,
    iterations: int = CUT_RELAX_ITERATIONS,
) -> CutSurface:
    """The cut that gives a side piece every bit of mold beyond the cast along its pull.

    On each line along the pull (``rows[2]``) that meets the cast, the cut runs
    just inside the cast's far end, so the piece takes all the mold past it; on
    lines that miss the cast it is as smooth as possible. ``keep_beyond`` are
    points the cut must pass behind (vertices of faces the piece is meant to
    take) and ``keep_behind`` points it must pass beyond (vertices of faces it
    cannot release); both hold on the grid nodes around each point, and
    ``keep_behind`` wins where they meet.
    """
    lo, hi = np.asarray(uv_bounds, dtype=float)
    origin = lo - cell
    shape = tuple(np.ceil((hi - lo) / cell).astype(np.int64) + 3)
    field = PartingSurface(origin=origin, cell=cell, heights=np.zeros(shape))
    local = trimesh.Trimesh(cast.vertices @ rows.T, cast.faces, process=False)
    nodes = field.nodes()
    low, high = column_spans(local, nodes)
    hit = np.isfinite(high)
    field.covered = hit.reshape(shape)
    target = np.full(len(nodes), -np.inf)
    span = np.where(hit, high - low, 0.0)
    target[hit] = high[hit] - np.minimum(INSET_MM, 0.25 * span[hit])
    if keep_beyond is not None and len(keep_beyond):
        _hold_near(field, keep_beyond @ rows.T, target, -INSET_MM, lowest=True)
    if keep_behind is not None and len(keep_behind):
        _hold_near(field, keep_behind @ rows.T, target, INSET_MM, lowest=False)
    fixed = np.isfinite(target)
    lower = np.where(fixed, target, -np.inf)
    upper = np.where(fixed, target, np.inf)
    field.heights = _relax(lower.reshape(shape), upper.reshape(shape), iterations)
    return CutSurface(rows=np.asarray(rows, dtype=float), field=field)


def _hold_near(
    field: PartingSurface, points: np.ndarray, target: np.ndarray, shift: float, *, lowest: bool
) -> None:
    """Hold the cut at each local point's height plus ``shift`` on the grid nodes around it.

    With ``lowest`` the cut may go no further along the pull than that (the
    smallest such height wins); otherwise it must go at least that far (the
    largest wins). ``target`` is -inf where a node is still free.
    """
    nx, ny = field.heights.shape
    index = (points[:, :2] - field.origin) / field.cell
    # The cut's height at a point depends only on the corners of the cell it lies in.
    i0 = np.clip(np.floor(index[:, 0]).astype(np.int64), 0, nx - 1)
    j0 = np.clip(np.floor(index[:, 1]).astype(np.int64), 0, ny - 1)
    di, dj = np.meshgrid(np.arange(2), np.arange(2), indexing="ij")
    i = np.clip(i0[:, None] + di.ravel(), 0, nx - 1)
    j = np.clip(j0[:, None] + dj.ravel(), 0, ny - 1)
    nodes = (i * ny + j).ravel()
    values = np.repeat(points[:, 2] + shift, di.size)
    if not lowest:
        np.maximum.at(target, nodes, values)
        return
    bound = np.full(len(target), np.inf)
    np.minimum.at(bound, nodes, values)
    held = np.isfinite(bound)
    current = target[held]
    target[held] = np.where(np.isfinite(current), np.minimum(current, bound[held]), bound[held])
