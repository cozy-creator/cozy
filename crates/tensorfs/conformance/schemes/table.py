"""The table decode (spec §6, building blocks): table[q] x s, then a scale on a grid."""

from __future__ import annotations

import numpy as np

from . import grid
from .base import Refused


def decode(q: np.ndarray, table: np.ndarray, s: np.ndarray, g: grid.Grid, rows: int, cols: int) -> np.ndarray:
    q = q.reshape(rows, cols)
    if q.size and (q.min() < 0 or q.max() >= len(table)):
        raise Refused("decode", f"a code is outside a table of {len(table)}")
    return table[q] * s[grid.at(g, rows, cols)]


def codes(data: bytes) -> np.ndarray:
    """A part's bytes as codes."""
    return np.frombuffer(data, np.uint8).astype(np.int64)
