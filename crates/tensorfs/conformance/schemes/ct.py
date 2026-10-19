"""ct.int1 ... ct.int8 (spec §6 `ct.*`): compressed-tensors' pack-quantized, signed b-bit codes offset by
2^(b-1) and packed along K in I32 words, a scale (and, asymmetric, a zero point packed along N) per group."""

import numpy as np

from . import affine, grid
from .base import Part, Scheme
from .gptq import bits


def height(c) -> int:  # N, the rows of each matrix
    return c.shape[-2]


def groups(c) -> int:
    return -(-c.cols // c.p("group_size"))


def packing(c, axis):
    b = bits(c)
    return affine.Packing(b, 32, axis=axis, fill=c.p("packing"), offset=1 << (b - 1))


def zp_shape(c):
    return "exact", [*c.shape[:-2], *packing(c, 0).shape([], height(c), groups(c))]


def act_order(c, g, w):
    """weight_g_idx as each column's group, or None for column order (a -1 anywhere: unset)."""
    idx = np.frombuffer(c.raw("weight_g_idx"), "<i4").astype(np.int64)
    if np.any(idx == -1):
        return None
    if idx.size and (idx.min() < 0 or idx.max() >= w):
        raise c.fail(f"weight_g_idx outside {w} groups")
    want = np.full(w, g)
    if w:
        want[-1] = c.cols - (w - 1) * g
    if np.any(np.bincount(idx, minlength=w) != want):
        raise c.fail("a group does not hold its g columns")
    return idx


def decode(c):
    sd, _, raw = c.parts["weight_shape"]
    if list(np.frombuffer(raw, "<i4" if sd == "I32" else "<i8")) != list(c.shape):
        raise c.fail(f"weight_shape is not {c.shape}")
    g, w = c.p("group_size"), groups(c)
    if c.p("strategy") == "channel" and g != c.cols:
        raise c.fail(f"channel, and group_size {g} is not {c.cols}")
    if c.p("symmetric") == c.has("weight_zero_point"):
        raise c.fail("weight_zero_point is present exactly when not symmetric")
    q = packing(c, 1).unpack(c.raw("weight_packed"), c.rows, c.cols)
    index = act_order(c, g, w) if c.has("weight_g_idx") else None
    blocks = grid.blocks(g, "row_major", index)
    grid.check(blocks, c)
    s = c.values("weight_scale")
    if not c.has("weight_zero_point"):
        return affine.dequant(q, s, blocks)
    n, p = height(c), packing(c, 0)
    per = p.words(n) * w * 4
    data = c.raw("weight_zero_point")
    z = [p.unpack(data[k:k + per], n, w) for k in range(0, len(data), max(per, 1))]
    z = np.concatenate(z).reshape(-1) if z else np.zeros(0)
    return affine.dequant(q, s, blocks, zero=z.astype(np.float32))


PARTS = (
    Part("weight_packed", ("I32",), lambda c: ("exact", packing(c, 1).shape(c.lead, c.rows, c.cols))),
    Part("weight_scale", ("F32", "BF16", "F16"),
         lambda c: grid.shape(grid.blocks(c.p("group_size"), "row_major"), c)),
    Part("weight_zero_point", ("I32",), zp_shape, required=False),
    Part("weight_shape", ("I32", "I64"), lambda c: ("exact", [len(c.shape)])),
    Part("weight_g_idx", ("I32",), lambda c: ("exact", [c.cols]), required=False),
)
PARAMS = {"group_size": range(1, 2**31 + 1), "strategy": ("group", "channel"), "symmetric": (True, False),
          "packing": ("word", "dense")}
SCHEMES = tuple(Scheme(f"ct.int{b}", PARAMS, PARTS, decode) for b in range(1, 9))
