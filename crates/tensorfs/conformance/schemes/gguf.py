"""gguf.<type> (spec §6 `gguf.*`): ggml's block structs as llama.cpp writes them in a GGUF, every struct of a
part at once, each element in ggml's float32 order."""

from __future__ import annotations

import numpy as np

from . import elem
from .base import Part, Scheme, one
from .blocks import f16, structs

F = np.float32
# ggml's kvalues_fp4, E2M1 doubled (spec Appendix A.2).
K = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int8).astype(F)
TYPES = {  # name: (elements, bytes) per struct
    "q4_0": (32, 18), "q4_1": (32, 20), "q5_0": (32, 22), "q5_1": (32, 24), "q8_0": (32, 34),
    "q2_k": (256, 84), "q3_k": (256, 110), "q4_k": (256, 144), "q5_k": (256, 176), "q6_k": (256, 210),
    "mxfp4": (32, 17), "nvfp4": (64, 36), "q1_0": (128, 18), "q2_0": (64, 18),
}


def blocks_shape(c):
    n, size = TYPES[c.scheme.name[5:]]
    numel = int(np.prod(c.shape, dtype=np.int64))
    if numel % n:
        raise c.fail(f"{c.shape} in blocks of {n}")
    if c.shape and c.shape[-1] % n == 0:
        return "exact", [*c.lead, c.shape[-1] // n, size]
    return "exact", [numel // n, size]


def nibbles(qs):
    """q4_0's order: element j in qs[j]'s low nibble, j + len(qs) in its high one."""
    return np.concatenate([qs & 15, qs >> 4], 1).astype(np.int32)


def fifth(qs, qh):
    q = nibbles(qs)
    return q + 16 * ((qh[:, None] >> np.arange(32, dtype=np.uint32)) & 1).astype(np.int32)


def u32(s, at):
    return s[:, at:at + 4].copy().view("<u4")[:, 0]


def q4_0(s):
    return (nibbles(s[:, 2:18]) - 8).astype(F) * f16(s, 0)


def q4_1(s):
    return nibbles(s[:, 4:20]).astype(F) * f16(s, 0) + f16(s, 2)


def q5_0(s):
    return (fifth(s[:, 6:22], u32(s, 2)) - 16).astype(F) * f16(s, 0)


def q5_1(s):
    return fifth(s[:, 8:24], u32(s, 4)).astype(F) * f16(s, 0) + f16(s, 2)


def q8_0(s):
    return s[:, 2:34].view(np.int8).astype(F) * f16(s, 0)


def low2(qs):
    """q2_k/q3_k: element t's 2-bit code and its sub-block b = t // 16."""
    t = np.arange(256)
    h, j, half, l = t // 128, (t % 128) // 32, (t % 32) // 16, t % 16
    return ((qs[:, 32 * h + 16 * half + l] >> (2 * j)) & 3).astype(np.int32), 16 * half + l


def q2_k(s):
    q, _ = low2(s[:, 16:80])
    sc = np.repeat(s[:, 0:16], 16, 1)
    dl = f16(s, 80) * (sc & 15).astype(F)
    ml = f16(s, 82) * (sc >> 4).astype(F)
    return dl * q.astype(F) - ml


def q3_k(s):
    q, at = low2(s[:, 32:96])
    t = np.arange(256)
    high = (s[:, at] >> (t // 32)) & 1
    b = np.arange(16)
    raw = s[:, 96:108]
    lo = np.where(b < 8, raw[:, b % 8] & 15, raw[:, b % 8] >> 4)
    sc = (lo | (((raw[:, 8 + b % 4] >> (2 * (b // 4))) & 3) << 4)).astype(np.int32) - 32
    dl = f16(s, 108) * np.repeat(sc, 16, 1).astype(F)
    return dl * (q - 4 + 4 * high.astype(np.int32)).astype(F)


def scale_min(raw):
    """get_scale_min_k4: the 8 six-bit (scale, min) pairs packed in 12 bytes."""
    a = np.arange(8) % 4
    first = np.arange(8) < 4
    c = np.where(first, raw[:, a] & 63, (raw[:, a + 8] & 15) | ((raw[:, a] >> 6) << 4))
    e = np.where(first, raw[:, a + 4] & 63, (raw[:, a + 8] >> 4) | ((raw[:, a + 4] >> 6) << 4))
    return c.astype(F), e.astype(F)


def k_quant(s, qs, qh):
    c, e = scale_min(s[:, 4:16])
    t = np.arange(256)
    a = t // 32
    byte = qs[:, 32 * (a // 2) + t % 32]
    q = np.where(a % 2 == 0, byte & 15, byte >> 4).astype(np.int32)
    if qh is not None:
        q = q + 16 * ((qh[:, t % 32] >> a) & 1).astype(np.int32)
    dl = f16(s, 0) * np.repeat(c, 32, 1)
    ml = f16(s, 2) * np.repeat(e, 32, 1)
    return dl * q.astype(F) - ml


def q4_k(s):
    return k_quant(s, s[:, 16:144], None)


def q5_k(s):
    return k_quant(s, s[:, 48:176], s[:, 16:48])


def q6_k(s):
    t = np.arange(256)
    h, j, l = t // 128, (t % 128) // 32, t % 32
    byte = s[:, 64 * h + 32 * (j % 2) + l]
    low = np.where(j < 2, byte & 15, byte >> 4)
    q = (low | (((s[:, 128 + 32 * h + l] >> (2 * j)) & 3) << 4)).astype(np.int32) - 32
    sc = s[:, 192:208].view(np.int8)[:, 8 * h + l // 16 + 2 * j].astype(F)
    return f16(s, 208) * sc * q.astype(F)


def mxfp4(s):
    e = s[:, 0].astype(np.int64)
    d = np.ldexp(np.float32(1), e - 128).astype(F)[:, None]
    return K[nibbles(s[:, 1:17])] * d


def nvfp4(s):
    d = elem.UE4M3[s[:, 0:4]] * F(0.5)
    q = np.concatenate([nibbles(s[:, 4 + 8 * k:12 + 8 * k]) for k in range(4)], 1)
    return K[q] * np.repeat(d, 16, 1)


def q1_0(s):
    t = np.arange(128)
    bit = (s[:, 2 + t // 8] >> (t % 8)) & 1
    d = f16(s, 0)
    return np.where(bit == 1, d, -d)


def q2_0(s):
    t = np.arange(64)
    q = ((s[:, 2 + t // 4] >> (2 * (t % 4))) & 3).astype(np.int32)
    return (q - 1).astype(F) * f16(s, 0)


def decode(c):
    name = c.scheme.name[5:]
    n, size = TYPES[name]
    out = DECODE[name](structs(c.raw("blocks"), size)).astype(F).reshape(-1)
    if c.has("scale"):
        out = out * c.values("scale")[0]
    return out


DECODE = {"q4_0": q4_0, "q4_1": q4_1, "q5_0": q5_0, "q5_1": q5_1, "q8_0": q8_0, "q2_k": q2_k, "q3_k": q3_k,
          "q4_k": q4_k, "q5_k": q5_k, "q6_k": q6_k, "mxfp4": mxfp4, "nvfp4": nvfp4, "q1_0": q1_0, "q2_0": q2_0}


def scheme(name):
    parts = (Part("blocks", ("U8",), blocks_shape),)
    if name == "nvfp4":
        parts += (Part("scale", ("F32",), one, required=False), Part("input_scale", ("F32",), one, required=False))
    return Scheme(f"gguf.{name}", {"block_size": (TYPES[name][0],)}, parts, decode, flat=True)


SCHEMES = tuple(scheme(name) for name in TYPES)
