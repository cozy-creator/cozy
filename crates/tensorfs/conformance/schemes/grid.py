"""Scale grids (spec §6, building blocks): for every element (i, j) of the [R, C] view, the index of its scale
in the stored part, and the stored shape the grid needs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LAYOUTS = ("row_major", "transposed", "cublas_128x4")


@dataclass(frozen=True)
class Grid:
    kind: str  # "tensor", "row", "col" or "blocks"
    size: int = 0  # blocks: elements per block along a row
    layout: str = "row_major"  # blocks: one of LAYOUTS
    index: np.ndarray | None = None  # blocks: each column's block (g_idx), else j // size


TENSOR, ROW, COL = Grid("tensor"), Grid("row"), Grid("col")


def blocks(size: int, layout: str, index=None) -> Grid:
    return Grid("blocks", size, layout, index)


def shape(g: Grid, c):
    if g.kind == "tensor":
        return "count", 1
    if g.kind in ("row", "col"):
        return "count", c.rows if g.kind == "row" else c.cols
    w = -(-c.cols // g.size)
    if g.layout == "row_major":
        return "exact", [*c.lead, w]
    if g.layout == "transposed":
        return "exact", [w, c.rows]
    return "exact", [128 * -(-c.rows // 128), 4 * -(-w // 4)]


def at(g: Grid, rows: int, cols: int) -> np.ndarray:
    """[R, C] indices into the stored part."""
    i, j = np.arange(rows)[:, None], np.arange(cols)[None, :]
    if g.kind == "tensor":
        return np.zeros((rows, cols), np.int64)
    if g.kind == "row":
        return np.broadcast_to(i, (rows, cols))
    if g.kind == "col":
        return np.broadcast_to(j, (rows, cols))
    w = -(-cols // g.size)
    b = (j // g.size) if g.index is None else g.index.astype(np.int64)[None, :]
    if g.layout == "row_major":
        return i * w + b
    if g.layout == "transposed":
        return b * rows + i
    return ((i // 128) * -(-w // 4) + b // 4) * 512 + (i % 32) * 16 + (i % 128 // 32) * 4 + b % 4


def check(g: Grid, c):
    """A g_idx gives each column a block below W."""
    if g.kind == "blocks" and g.index is not None:
        w = -(-c.cols // g.size)
        if len(g.index) != c.cols or (len(g.index) and (g.index.min() < 0 or g.index.max() >= w)):
            raise c.fail(f"a block index is not one of {w} per column")
