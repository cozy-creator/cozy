"""Element formats (spec §5): each code's exact value, as float32 tables indexed by code."""

from __future__ import annotations

import numpy as np


def _minifloat(e_bits: int, m_bits: int, bias: int, special) -> np.ndarray:
    out = np.empty(256, np.float32)
    for b in range(256):
        e, m = (b >> m_bits) & ((1 << e_bits) - 1), b & ((1 << m_bits) - 1)
        v = special(b & 0x7F, e, m)
        if v is None:
            v = m * 2.0 ** (1 - bias - m_bits) if e == 0 else (1 + m / 2**m_bits) * 2.0 ** (e - bias)
        out[b] = -v if b & 0x80 else v
    return out


E4M3 = _minifloat(4, 3, 7, lambda b, e, m: np.nan if b == 0x7F else None)
E5M2 = _minifloat(5, 2, 15, lambda b, e, m: (np.inf if m == 0 else np.nan) if e == 31 else None)
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
E8M0 = np.array([2.0 ** (b - 127) if b < 255 else np.nan for b in range(256)], np.float32)
# ggml's UE4M3: E4M3's exponent and mantissa, bit 7 ignored, 0x7F zero, no NaN.
UE4M3 = np.abs(_minifloat(4, 3, 7, lambda b, e, m: None))
UE4M3[0x7F] = 0
I8 = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.float32)
U8 = np.arange(256, dtype=np.float32)
TABLES = {"F4": E2M1, "F8_E4M3": E4M3, "F8_E5M2": E5M2, "F8_E8M0": E8M0, "I8": I8, "U8": U8}


def values(dtype: str, data: bytes) -> np.ndarray:
    """A part's elements as float32 (F4 per §5: element 2k in the low nibble)."""
    raw = np.frombuffer(data, np.uint8)
    if dtype == "F32":
        return np.frombuffer(data, "<f4").astype(np.float32)
    if dtype == "BF16":
        return (np.frombuffer(data, "<u2").astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return np.frombuffer(data, "<f2").astype(np.float32)
    if dtype == "F4":
        return E2M1[np.stack([raw & 15, raw >> 4], -1).reshape(-1)]
    return TABLES[dtype][raw]
