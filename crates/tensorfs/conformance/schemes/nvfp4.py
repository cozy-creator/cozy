"""nvfp4 (spec §6 `nvfp4`): NVIDIA's NVFP4, an E2M1 element per weight, an E4M3 scale per 16 elements
of a row and a per-tensor scale_2."""

from . import affine, elem, grid, table
from .base import Part, Scheme, logical, one
from .mxfp8 import LAYOUTS, blocks


def scale_shape(c):
    if c.cols % 16:
        raise c.fail(f"{c.cols} columns in blocks of 16")
    return grid.shape(blocks(c), c)


def decode(c):
    nibbles = affine.Packing(4, 8, order=c.p("nibble_order"))
    q = nibbles.unpack(c.raw("weight"), c.rows, c.cols)
    s, g = c.values("scale"), c.values("scale_2")[0]
    if c.p("global_scale") == "multiply":
        return table.decode(q, elem.E2M1, s, blocks(c), c.rows, c.cols) * g
    return table.decode(q, elem.E2M1, s / g, blocks(c), c.rows, c.cols)


SCHEMES = (
    Scheme("nvfp4", {"block_size": (16,), "global_scale": ("multiply", "divide"),
                     "nibble_order": ("low_first", "high_first"), "scale_layout": LAYOUTS},
           (Part("weight", ("F4",), logical), Part("scale", ("F8_E4M3",), scale_shape),
            Part("scale_2", ("F32",), one), Part("input_scale", ("F32",), one, required=False)),
           decode),
)
