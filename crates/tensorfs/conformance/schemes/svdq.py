"""svdq.nvfp4 (spec §6 `svdq.nvfp4`): SVDQuant's weight, Nunchaku's svdq-fp4 in natural layout. An nvfp4 residual
of the smoothed weight, optionally a scale per row, and a 16-bit low-rank branch."""

from . import grid
from .base import Part, Scheme, logical, one
from .nvfp4 import PARAMS, core, scale_shape

FACTORS = ("BF16", "F16", "F32")


def matrix(c):
    if len(c.shape) != 2:
        raise c.fail(f"shape {c.shape} is not [N, K]")
    return logical(c)


def rank(c):
    s = c.parts["lowrank_down"][1]
    if len(s) != 2 or s[0] < 1:
        raise c.fail(f"part lowrank_down is {s}")
    return s[0]


def decode(c):
    """nvfp4's ((T[q] x s) x g), x c[i] with a channel_scale, x p[j], then + up[i, r] x down[r, j] in r order."""
    v = core(c)
    if c.has("channel_scale"):
        v = v * c.values("channel_scale").reshape(-1, 1)
    v = v * c.values("pre_quant_scale").reshape(1, -1)
    r = rank(c)
    down, up = c.values("lowrank_down").reshape(r, c.cols), c.values("lowrank_up").reshape(c.rows, r)
    for t in range(r):
        v = v + up[:, t:t + 1] * down[t:t + 1, :]
    return v


SCHEMES = (
    Scheme("svdq.nvfp4", PARAMS,
           (Part("weight", ("F4",), matrix), Part("scale", ("F8_E4M3",), scale_shape), Part("scale_2", ("F32",), one),
            Part("input_scale", ("F32",), one, required=False),
            Part("pre_quant_scale", ("F32", "BF16", "F16"), lambda c: grid.shape(grid.COL, c)),
            Part("channel_scale", ("F32", "F8_E4M3"), lambda c: grid.shape(grid.ROW, c), required=False),
            Part("lowrank_down", FACTORS, lambda c: ("exact", [rank(c), c.cols])),
            Part("lowrank_up", FACTORS, lambda c: ("exact", [c.rows, rank(c)]))),
           decode),
)
