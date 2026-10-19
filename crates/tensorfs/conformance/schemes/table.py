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


def matrices(c, q: np.ndarray, s: np.ndarray):
    """N, and each matrix of a stack [..., N, C] as (its codes, its share of row-block scales stored
    row_major or cublas_128x4, which tiles each matrix on its own)."""
    n = c.shape[-2]
    m = c.rows // n if c.rows else 0
    return n, [(qm, sm) for qm, sm in zip(np.split(q.reshape(-1), m), np.split(s.reshape(-1), m))] if m else []


def codes(data: bytes) -> np.ndarray:
    """A part's bytes as codes."""
    return np.frombuffer(data, np.uint8).astype(np.int64)
