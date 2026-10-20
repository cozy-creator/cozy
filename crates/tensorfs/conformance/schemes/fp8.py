"""fp8.tensorwise, fp8.rowwise and fp8.columnwise (spec §6): FP8 elements times one scale for the tensor, or one per
row, or one per column."""

from . import elem, grid, table
from .base import Part, Scheme, logical, one

FP8 = ("F8_E4M3", "F8_E5M2")
SCALE = ("F32", "BF16", "F16")  # as the producer stored it


def scaled(c, g):
    dtype, _, data = c.parts["weight"]
    return table.decode(table.codes(data), elem.TABLES[dtype], c.values("scale"), g, c.rows, c.cols)


SCHEMES = (
    Scheme("fp8.tensorwise", {},
           (Part("weight", FP8, logical), Part("scale", SCALE, one),
            Part("input_scale", ("F32",), one, required=False)),
           lambda c: scaled(c, grid.TENSOR), flat=True),
    Scheme("fp8.rowwise", {},
           (Part("weight", FP8, logical), Part("scale", SCALE, lambda c: grid.shape(grid.ROW, c))),
           lambda c: scaled(c, grid.ROW)),
    Scheme("fp8.columnwise", {},
           (Part("weight", FP8, logical), Part("scale", SCALE, lambda c: grid.shape(grid.COL, c))),
           lambda c: scaled(c, grid.COL)),
)
