"""gptq.int2, gptq.int3, gptq.int4, gptq.int8 (spec §6 `gptq.*`): GPTQ's b-bit codes packed along K in I32
words, a zero and a scale per group of inputs, each input's group in g_idx."""

import numpy as np

from . import affine, grid
from .base import Part, Scheme


def group_size(v) -> bool:
    return type(v) is int and (v == -1 or 1 <= v <= 2**31)


def bits(c) -> int:
    return int(c.scheme.name.rsplit(".int", 1)[1])


def group(c) -> int:
    return max(c.cols, 1) if c.p("group_size") == -1 else c.p("group_size")


def groups(c) -> int:
    return -(-c.cols // group(c))


def codes(c):  # the [K, N] transpose, one dense stream along K
    return affine.Packing(bits(c), 32, transposed=True, axis=0, fill="dense")


def zeros(c):  # [W, N] as is, one dense stream along N
    return affine.Packing(bits(c), 32, fill="dense")


def decode(c):
    b, g, w = bits(c), group(c), groups(c)
    q = codes(c).unpack(c.raw("qweight"), c.rows, c.cols)
    z = ((zeros(c).unpack(c.raw("qzeros"), w, c.rows) + c.p("zero_offset")) % (1 << b)).reshape(-1)
    if c.p("symmetric") and np.any(z != 1 << (b - 1)):
        raise c.fail(f"symmetric, and a zero is not {1 << (b - 1)}")
    g_idx = np.frombuffer(c.raw("g_idx"), "<i4").astype(np.int64)
    if not c.p("act_order") and np.any(g_idx != np.arange(c.cols) // g):
        raise c.fail("act_order is false, and g_idx is not each column's j // g")
    blocks = grid.blocks(g, "transposed", g_idx)
    grid.check(blocks, c)
    return affine.dequant(q, c.values("scales"), blocks, zero=z.astype(np.float32))


PARTS = (
    Part("qweight", ("I32",), lambda c: ("exact", codes(c).shape([], c.rows, c.cols))),
    Part("qzeros", ("I32",), lambda c: ("exact", zeros(c).shape([groups(c)], groups(c), c.rows))),
    Part("scales", ("F16", "BF16"), lambda c: grid.shape(grid.blocks(group(c), "transposed"), c)),
    Part("g_idx", ("I32",), lambda c: ("exact", [c.cols])),
)
PARAMS = {"group_size": group_size, "symmetric": (True, False), "act_order": (True, False), "zero_offset": (0, 1)}
SCHEMES = tuple(Scheme(f"gptq.int{b}", PARAMS, PARTS, decode) for b in (2, 3, 4, 8))
