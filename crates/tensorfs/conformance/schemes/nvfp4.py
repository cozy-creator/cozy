"""nvfp4 (spec §6 `nvfp4`): NVIDIA's NVFP4, an E2M1 element per weight, an E4M3 scale per 16 elements
of a row, a global scale per matrix and optionally AWQ's per-column factor."""

import numpy as np

from . import affine, elem, grid, table
from .base import Part, PartDtype, Scheme, logical, one
from .mxfp8 import LAYOUTS, blocks


def scale_shape(c):
    if c.cols % 16:
        raise c.fail(f"{c.cols} columns in blocks of 16")
    return grid.shape(blocks(c), c)


def per_matrix(role):
    """One element, or M, one per matrix of [..., N, C]: any shape."""
    def shape(c):
        stored = int(np.prod(c.parts[role][1], dtype=np.int64))
        return one(c) if stored == 1 else ("count", int(np.prod(c.shape[:-2], dtype=np.int64)))
    return shape


def decode(c):
    nibbles = affine.Packing(4, 8, order=c.p("nibble_order"))
    q = nibbles.unpack(c.raw("weight"), c.rows, c.cols)
    g = c.values("scale_2")
    n, mats = table.matrices(c, q, c.values("scale"))
    out = []
    for m, (qm, s) in enumerate(mats):
        gm = g[0 if g.size == 1 else m]
        if c.p("global_scale") == "multiply":
            out.append(table.decode(qm, elem.E2M1, s, blocks(c), n, c.cols) * gm)
        else:
            out.append(table.decode(qm, elem.E2M1, s / gm, blocks(c), n, c.cols))
    out = np.concatenate(out).reshape(c.rows, c.cols) if out else np.zeros((0, c.cols), np.float32)
    return out * c.values("pre_quant_scale") if c.has("pre_quant_scale") else out


SCHEMES = (
    Scheme("nvfp4", {"block_size": (16,), "global_scale": ("multiply", "divide"),
                     "nibble_order": ("low_first", "high_first"), "scale_layout": LAYOUTS,
                     "pre_quant_scale_dtype": PartDtype("pre_quant_scale", ("F32", "BF16", "F16"))},
           (Part("weight", ("F4",), logical), Part("scale", ("F8_E4M3",), scale_shape),
            Part("scale_2", ("F32",), per_matrix("scale_2")),
            Part("input_scale", ("F32",), per_matrix("input_scale"), required=False),
            Part("pre_quant_scale", "pre_quant_scale_dtype", lambda c: grid.shape(grid.COL, c), required=False)),
           decode),
)
