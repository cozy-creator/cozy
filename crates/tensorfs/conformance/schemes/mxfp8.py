"""mxfp8 (spec §6 `mxfp8`): OCP MXFP8, an F8_E8M0 scale per 32 elements of a row."""

import numpy as np

from . import elem, grid, table
from .base import Part, Scheme, logical

LAYOUTS = ("row_major", "cublas_128x4")


def blocks(c):
    return grid.blocks(c.p("block_size"), c.p("scale_layout"))


def decode(c):
    dtype, _, data = c.parts["weight"]
    n, mats = table.matrices(c, table.codes(data), c.values("scale"))
    return np.concatenate([table.decode(q, elem.TABLES[dtype], s, blocks(c), n, c.cols).reshape(-1)
                           for q, s in mats] or [np.zeros(0, np.float32)])


SCHEMES = (
    Scheme("mxfp8", {"block_size": (32,), "scale_layout": LAYOUTS},
           (Part("weight", ("F8_E4M3", "F8_E5M2"), logical),
            Part("scale", ("F8_E8M0",), lambda c: grid.shape(blocks(c), c))),
           decode),
)
