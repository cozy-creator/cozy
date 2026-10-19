"""Block structs (spec §6, building blocks): a U8 [lead..., C/n, size] part of fixed-size structs, each holding n
consecutive elements of a row; the scheme decodes each struct."""

from __future__ import annotations

import numpy as np


def shape(c, n: int, size: int):
    if c.cols % n:
        raise c.fail(f"{c.cols} columns are not whole blocks of {n}")
    return "exact", [*c.lead, c.cols // n, size]


def structs(data: bytes, size: int) -> np.ndarray:
    """The structs as rows of bytes, [R*C/n, size]: a scheme decodes them all at once."""
    return np.frombuffer(data, np.uint8).reshape(-1, size)


def f16(structs: np.ndarray, at: int) -> np.ndarray:
    """The F16 field at byte `at` of every struct, as float32 [structs, 1]."""
    return structs[:, at:at + 2].copy().view("<f2").astype(np.float32)
