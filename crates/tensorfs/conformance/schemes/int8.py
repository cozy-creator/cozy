"""int8.tensorwise, int8.rowwise and int8.convrot (spec §6)."""

import numpy as np

from . import affine, grid
from .base import Part, Scheme, logical, one
from .fp8 import SCALE, axis_grid, per_axis


def weight(c) -> np.ndarray:
    return np.frombuffer(c.raw("weight"), np.int8).astype(np.int64).reshape(c.rows, c.cols)


def kron_regular_h4(x: np.ndarray, k: int) -> np.ndarray:
    """Rows of x (int64, last axis 4**k) times H4 (x) ... (x) H4, H4 the spec's A.1, unnormalized."""
    shape = x.shape
    x = x.reshape(-1, *([4] * k))
    for axis in range(1, k + 1):
        a, b, c, d = (np.take(x, i, axis=axis) for i in range(4))
        x = np.stack([a + b + c - d, a + b - c + d, a - b + c + d, b + c + d - a], axis=axis)
    return x.reshape(shape)


def convrot(c):
    b = c.p("rotation_size")
    if c.cols % b:
        raise c.fail(f"{c.cols} columns in blocks of {b}")
    k = b.bit_length() // 2
    h = kron_regular_h4(weight(c).reshape(c.rows, c.cols // b, b), k).astype(np.float32).reshape(c.rows, c.cols)
    return (h * c.values("scale").reshape(c.rows, 1)) * np.float32(2.0**-k)


SCHEMES = (
    Scheme("int8.tensorwise", {},
           (Part("weight", ("I8",), logical), Part("scale", SCALE, one),
            Part("input_scale", ("F32",), one, required=False)),
           lambda c: affine.dequant(weight(c), c.values("scale"), grid.TENSOR), flat=True),
    Scheme("int8.rowwise", {"axis": (0, 1)},
           (Part("weight", ("I8",), logical), Part("scale", SCALE, per_axis)),
           lambda c: affine.dequant(weight(c), c.values("scale"), axis_grid(c))),
    Scheme("int8.convrot", {"rotation": ("kron_regular_h4",), "rotation_size": tuple(4**k for k in range(1, 9))},
           (Part("weight", ("I8",), logical), Part("scale", SCALE, lambda c: grid.shape(grid.ROW, c))),
           convrot),
)
