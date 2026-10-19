"""mxfp4 (spec §6 `mxfp4`): OCP MXFP4, an E2M1 element per weight and an F8_E8M0 scale per 32 elements of a
row."""

import numpy as np

from . import elem, grid, table
from .base import Part, Scheme, logical

BLOCKS = grid.blocks(32, "row_major")


def decode(c):
    raw = np.frombuffer(c.raw("weight"), np.uint8)
    codes = np.stack([raw & 15, raw >> 4], -1).reshape(-1).astype(np.int64)  # §5: element 2k low
    return table.decode(codes, elem.E2M1, c.values("scale"), BLOCKS, c.rows, c.cols)


SCHEMES = (
    Scheme("mxfp4", {"block_size": (32,)},
           (Part("weight", ("F4",), logical), Part("scale", ("F8_E8M0",), lambda c: grid.shape(BLOCKS, c))),
           decode),
)
