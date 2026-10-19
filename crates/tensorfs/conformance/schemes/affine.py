"""Bit packing and the affine decode (spec §6, building blocks): b-bit codes packed in words, then (q - z) x s
(or q x s + b) on a scale grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import grid
from .base import Refused


@dataclass(frozen=True)
class Packing:
    bits: int  # 1..8
    word: int  # 8, 16 or 32
    transposed: bool = False  # the part stores the [C, R] transpose of the view
    axis: int = 1  # the stored array's axis the codes are packed along
    fill: str = "word"  # "word", "dense" or "planar"
    order: str | tuple = "low_first"  # "low_first", "high_first", or slot -> code
    signed: bool = False  # two's complement; else q = raw - offset
    offset: int = 0

    @property
    def per(self) -> int:
        return self.word // self.bits

    def words(self, n: int) -> int:
        if self.fill == "word":
            return -(-n // self.per)
        if self.fill == "dense":
            return -(-n * self.bits // self.word)
        if n % self.per:
            raise Refused("decode", f"{n} codes do not fill planes of {self.per}")
        return n // self.per

    def shape(self, lead, rows: int, cols: int) -> list:
        a, b = (cols, rows) if self.transposed else (rows, cols)
        if self.axis == 1:
            return [*lead, self.words(b)] if not self.transposed else [a, self.words(b)]
        return [self.words(a), b]

    def slots(self) -> np.ndarray:
        """The slot of each code index within a word."""
        per = self.per
        if self.order == "low_first":
            return np.arange(per)
        if self.fill == "dense":
            raise Refused("decode", "a dense bit stream is low first")
        if self.order == "high_first":
            return np.arange(per)[::-1].copy()
        order = list(self.order)
        if sorted(order) != list(range(per)):
            raise Refused("decode", f"slots {order} are not a permutation of {per}")
        return np.array([order.index(e) for e in range(per)])

    def lanes(self, words: np.ndarray, n: int) -> np.ndarray:
        """Codes [lanes, n] from words [lanes, nw]."""
        b, w, nw = self.bits, self.word, words.shape[1]
        u = np.uint64
        words = words.astype(u)
        t = np.arange(n)
        if self.fill == "dense":
            k, off = (t * b) // w, (t * b) % w
            hi = np.minimum(k + 1, nw - 1)
            raw = words[:, k] >> off.astype(u)
            spill = (off + b > w)[None, :]
            raw = raw | np.where(spill, words[:, hi] << (w - off).astype(u), u(0))
        else:
            k, e = (t // self.per, t % self.per) if self.fill == "word" else (t % nw, t // nw)
            raw = words[:, k] >> (self.slots()[e] * b).astype(u)
        raw = (raw & u((1 << b) - 1)).astype(np.int64)
        if self.signed:
            return np.where(raw >= 1 << (b - 1), raw - (1 << b), raw)
        return raw - self.offset

    def unpack(self, data: bytes, rows: int, cols: int) -> np.ndarray:
        """The view's codes q[i, j], int64 [R, C]."""
        a, b = (cols, rows) if self.transposed else (rows, cols)
        n, along = (a, b) if self.axis == 0 else (b, a)
        nw = self.words(n)
        words = np.frombuffer(data, f"<u{self.word // 8}")
        if words.size != nw * along:
            raise Refused("decode", f"{len(data)} bytes are not {nw}x{along} words")
        if self.axis == 0:
            stored = self.lanes(words.reshape(nw, along).T, n).T  # [a, b]
        else:
            stored = self.lanes(words.reshape(along, nw), n)
        return stored.T if self.transposed else stored

    def lane(self, data: bytes, n: int) -> np.ndarray:
        words = np.frombuffer(data, f"<u{self.word // 8}")
        if words.size != self.words(n):
            raise Refused("decode", f"{len(data)} bytes are not {self.words(n)} words")
        return self.lanes(words[None, :], n)[0]


def dequant(q: np.ndarray, s: np.ndarray, g: grid.Grid, zero=None, bias=None) -> np.ndarray:
    """(q - z) x s, q x s + b or q x s per element, each operation in float32."""
    rows, cols = q.shape
    at = grid.at(g, rows, cols)
    v = q.astype(np.float32)
    if zero is not None:
        return (v - zero[at]) * s[at]
    if bias is not None:
        return v * s[at] + bias[at]
    return v * s[at]
