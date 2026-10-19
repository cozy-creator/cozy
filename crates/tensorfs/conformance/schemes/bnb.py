"""bnb.nf4, bnb.fp4 and bnb.int8 (spec §6): bitsandbytes' 4-bit blockwise tables over the flattened tensor, with
optional double quantization of the absmax, and LLM.int8() rows."""

import numpy as np

from . import affine, grid, table
from .base import Part, Scheme, logical, one

# Appendix A.3 and A.4: the tables bnb's kernels decode with (FP4 code 8 is -0; bnb stores it as +0).
NF4 = np.array([-1.0, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343, -0.091050036, 0.0, 0.0795803,
                0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1.0], np.float32)
FP4 = np.array([0.0, 0.0052083335, 0.6666667, 1.0, 0.33333334, 0.5, 0.16666667, 0.25, -0.0, -0.0052083335,
                -0.6666667, -1.0, -0.33333334, -0.5, -0.16666667, -0.25], np.float32)
BLOCKS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
R127 = np.float32(1 / 127)


def nblocks(c) -> int:
    return -(-c.cols // c.p("block_size"))


def four_bit(name, codebook):
    def decode(c):
        if c.parts["absmax"][0] != ("U8" if c.p("double_quant") else "F32"):
            raise c.fail("absmax is U8 codes exactly when double_quant")
        if not np.array_equal(c.values("quant_map"), codebook):  # as numbers: -0 == +0
            raise c.fail("quant_map is not the table")
        nested = {"nested_absmax", "nested_quant_map", "nested_offset"}
        if {r for r in nested if c.has(r)} != (nested if c.p("double_quant") else set()):
            raise c.fail("the nested parts are present exactly when double_quant")
        n, nb = c.cols, nblocks(c)
        s = c.values("absmax")
        if c.p("double_quant"):
            codes = table.codes(c.raw("absmax"))
            s = table.decode(codes, c.values("nested_quant_map"), c.values("nested_absmax"),
                             grid.blocks(256, "row_major"), 1, nb) + c.values("nested_offset")[0]
        q = affine.Packing(4, 8, order="high_first").unpack(c.raw("weight"), 1, n)
        return table.decode(q, codebook, s.reshape(-1), grid.blocks(c.p("block_size"), "row_major"), 1, n)

    return Scheme(name, {"block_size": BLOCKS, "double_quant": (False, True)},
                  (Part("weight", ("U8",), lambda c: ("exact", [-(-c.cols // 2), 1])),
                   Part("absmax", ("F32", "U8"), lambda c: ("exact", [nblocks(c)])),
                   Part("quant_map", ("F32",), lambda c: ("exact", [16])),
                   Part("nested_absmax", ("F32",), lambda c: ("exact", [-(-nblocks(c) // 256)]), required=False),
                   Part("nested_quant_map", ("F32",), lambda c: ("exact", [256]), required=False),
                   Part("nested_offset", ("F32",), one, required=False)),
                  decode, flat=True)


def int8(c):
    if len(c.shape) != 2:
        raise c.fail(f"shape {c.shape} is not [N, K]")
    if c.has("weight_format") and c.raw("weight_format") != b"\0":
        raise c.fail("weight_format is not 0 (row)")
    q = np.frombuffer(c.raw("weight"), np.int8).astype(np.float32).reshape(c.rows, c.cols)
    return (q * c.values("SCB").reshape(c.rows, 1)) * R127


SCHEMES = (
    four_bit("bnb.nf4", NF4),
    four_bit("bnb.fp4", FP4),
    Scheme("bnb.int8", {},
           (Part("weight", ("I8",), logical), Part("SCB", ("F32",), lambda c: grid.shape(grid.ROW, c)),
            Part("weight_format", ("U8",), one, required=False)),
           int8),
)
