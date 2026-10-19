"""fp8.tensorwise and fp8.rowwise (spec §6 `fp8.tensorwise`, `fp8.rowwise`): FP8 elements times one scale for the tensor,
or one per row or column."""

from . import elem, grid, table
from .base import Part, Scheme, logical, one

FP8 = ("F8_E4M3", "F8_E5M2")


def axis_grid(c):
    return grid.ROW if c.p("axis") == 0 else grid.COL


def per_axis(c):
    return grid.shape(axis_grid(c), c)


def scaled(c, g):
    dtype, _, data = c.parts["weight"]
    return table.decode(table.codes(data), elem.TABLES[dtype], c.values("scale"), g, c.rows, c.cols)


SCHEMES = (
    Scheme("fp8.tensorwise", {"scale_dtype": ("F32", "BF16", "F16")},
           (Part("weight", FP8, logical), Part("scale", "scale_dtype", one),
            Part("input_scale", ("F32",), one, required=False)),
           lambda c: scaled(c, grid.TENSOR), flat=True),
    Scheme("fp8.rowwise", {"axis": (0, 1), "scale_dtype": ("F32", "BF16", "F16")},
           (Part("weight", FP8, logical), Part("scale", "scale_dtype", per_axis)),
           lambda c: scaled(c, axis_grid(c))),
)
