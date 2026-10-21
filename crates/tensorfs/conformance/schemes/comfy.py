"""int4.convrot, comfy.asym_w4a8_int8 and comfy.w6a8_int8 (spec §6): ConvRot-basis integer codes, rotated back per
block of rotation_size columns."""

import numpy as np

from . import grid
from .base import Part, Scheme
from .int8 import kron_regular_h4

ROTATION = ("kron_regular_h4",)
ROTATION_SIZE = tuple(4**k for k in range(1, 9))


def packed(bits, group=True):
    def shape(c):
        b, g = c.p("rotation_size"), c.p("group_size") if group else 1
        if c.cols % b or c.cols % g:
            raise c.fail(f"{c.cols} columns in blocks of {b} and groups of {g}")
        return "exact", [*c.lead, c.cols * bits // 8]
    return shape


def groups(layout):
    return lambda c: grid.blocks(c.p("group_size"), layout)


def codes(c, bits) -> np.ndarray:
    """Unsigned codes [R, C]: a row is C/2 bytes of nibbles (even column low), then at 6 bits C/4 bytes of
    each code's high 2 bits (column t at bits 2(t mod 4) of byte t // 4)."""
    w = np.frombuffer(c.raw("weight"), np.uint8).reshape(c.rows, c.cols * bits // 8).astype(np.int64)
    half = c.cols // 2
    q = np.stack([w[:, :half] & 15, w[:, :half] >> 4], -1).reshape(c.rows, c.cols)
    if bits == 6:
        q |= ((w[:, half:, None] >> np.array([0, 2, 4, 6])) & 3).reshape(c.rows, c.cols) << 4
    return q


def rotate(c, q8, s, corr=None) -> np.ndarray:
    """((y x s[i]) + z) x 2^-k per block, y = q8 . H_B exact; z = sum_b corr(i, b) x M_b, left to right."""
    b = c.p("rotation_size")
    k = b.bit_length() // 2
    y = kron_regular_h4(q8.reshape(c.rows, c.cols // b, b), k).astype(np.float32)
    v = y * s.reshape(c.rows, 1, 1)
    if corr is not None:
        g = c.p("group_size")
        cg = corr.reshape(-1, c.rows).T  # [R, C/g]
        z = np.zeros_like(v)
        for blk in range(c.cols // b):
            first = True
            for grp in range(blk * b // g, -(-(blk + 1) * b // g)):
                ind = ((np.arange(blk * b, (blk + 1) * b) // g) == grp).astype(np.int64)
                m = kron_regular_h4(ind[None, :], k)[0].astype(np.float32)
                t = cg[:, grp:grp + 1] * m[None, :]
                z[:, blk] = t if first else z[:, blk] + t
                first = False
        v = v + z
    return (v * np.float32(2.0**-k)).reshape(c.rows, c.cols)


def grouped(bits):
    def decode(c):
        q = codes(c, bits)
        if c.has("codebook"):
            v = c.values("codebook")[q]
        else:
            v = (q - (1 << (bits - 1))).astype(np.float32)
        x = np.round(v * c.values("group_scale")[grid.at(groups("row_major")(c), c.rows, c.cols)])
        if np.isnan(x).any():
            raise c.fail("a level scales to NaN")
        q8 = np.clip(x, -127, 127).astype(np.int64)
        corr = c.values("correction") if c.has("correction") else None
        return rotate(c, q8, c.values("scale"), corr)
    return decode


def int4_convrot(c):
    w = np.frombuffer(c.raw("weight"), np.uint8).reshape(c.rows, c.cols // 2).astype(np.int64)
    q = np.stack([w & 15, w >> 4], -1).reshape(c.rows, c.cols)
    return rotate(c, np.where(q >= 8, q - 16, q), c.values("scale"))


def row(c):
    return grid.shape(grid.ROW, c)


W4_GROUP = lambda g: type(g) is int and g >= 4 and (16 % g == 0 or g % 16 == 0)  # noqa: E731
W6_GROUP = lambda g: type(g) is int and g >= 16 and g % 16 == 0  # noqa: E731
S_REL = ("F8_E4M3", "F32")

SCHEMES = (
    Scheme("int4.convrot", {"rotation": ROTATION, "rotation_size": ROTATION_SIZE},
           (Part("weight", ("I8",), packed(4, group=False)), Part("scale", ("F32",), row)),
           int4_convrot),
    Scheme("comfy.asym_w4a8_int8", {"group_size": W4_GROUP, "rotation": ROTATION, "rotation_size": ROTATION_SIZE},
           (Part("weight", ("I8",), packed(4)),
            Part("group_scale", S_REL, lambda c: grid.shape(groups("row_major")(c), c)),
            Part("scale", ("F32",), row),
            Part("codebook", ("F32",), lambda c: ("count", 16), required=False),
            Part("correction", ("F32", "BF16", "F16"),
                 lambda c: grid.shape(groups("transposed")(c), c), required=False)),
           grouped(4)),
    Scheme("comfy.w6a8_int8", {"group_size": W6_GROUP, "rotation": ROTATION, "rotation_size": ROTATION_SIZE},
           (Part("weight", ("I8",), packed(6)),
            Part("group_scale", S_REL, lambda c: grid.shape(groups("row_major")(c), c)),
            Part("scale", ("F32",), row)),
           grouped(6)),
)
