"""fp8.blockwise (spec §6 `fp8.blockwise`): FP8 elements times one scale per tile of block_size = [rows, cols] of
each matrix (DeepSeek's weight_scale_inv, compressed-tensors' block strategy)."""

from . import elem, grid, table
from .base import Part, Scheme, logical


def block_size(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and all(type(n) is int and n > 0 for n in v)


def tiles(c):
    rows, cols = c.p("block_size")
    return grid.tiles(rows, cols, c.shape[-2])


def decode(c):
    dtype, _, data = c.parts["weight"]
    return table.decode(table.codes(data), elem.TABLES[dtype], c.values("scale"), tiles(c), c.rows, c.cols)


SCHEMES = (
    Scheme("fp8.blockwise", {"block_size": block_size},
           (Part("weight", ("F8_E4M3",), logical),
            Part("scale", ("F32", "BF16", "F16", "F8_E8M0"), lambda c: grid.shape(tiles(c), c))),
           decode),
)
