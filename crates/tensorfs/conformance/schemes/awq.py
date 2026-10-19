"""awq.int4 (spec §6 `awq.int4`): AutoAWQ's GEMM layout, 4-bit codes packed along N in the order
[0, 2, 4, 6, 1, 3, 5, 7], a zero and a scale per group of inputs."""

import numpy as np

from . import affine, grid
from .base import Part, Scheme
from .gptq import group, group_size, groups

ORDER = (0, 2, 4, 6, 1, 3, 5, 7)  # slot -> the code it holds
CODES = affine.Packing(4, 32, transposed=True, axis=1, order=ORDER)  # the [K, N] transpose, along N
ZEROS = affine.Packing(4, 32, order=ORDER)  # [W, N] as is, along N


def blocks(c):
    return grid.blocks(group(c), "transposed")


def decode(c):
    q = CODES.unpack(c.raw("qweight"), c.rows, c.cols)
    z = ZEROS.unpack(c.raw("qzeros"), groups(c), c.rows).reshape(-1).astype(np.float32)
    return affine.dequant(q, c.values("scales"), blocks(c), zero=z)


SCHEMES = (
    Scheme("awq.int4", {"group_size": group_size},
           (Part("qweight", ("I32",), lambda c: ("exact", CODES.shape([], c.rows, c.cols))),
            Part("qzeros", ("I32",), lambda c: ("exact", ZEROS.shape([groups(c)], groups(c), c.rows))),
            Part("scales", ("F16", "BF16"), lambda c: grid.shape(blocks(c), c))),
           decode),
)
