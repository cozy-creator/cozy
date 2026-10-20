"""The row type every scheme module fills in, and what its decode sees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import elem


class Refused(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class Part:
    role: str
    dtypes: tuple  # the part's allowed dtype names: its own dtype is authoritative
    shape: Callable  # ctx -> ("exact", [extents]) or ("count", n)
    required: bool = True


@dataclass(frozen=True)
class Scheme:
    name: str
    params: dict  # key -> the allowed values (a tuple, a range) or a predicate
    parts: tuple
    decode: Callable  # ctx -> float32 array of R*C elements, row-major
    flat: bool = False  # any shape as one row; else rank >= 2 as [R, C]


class Ctx:
    def __init__(self, scheme, dtype, shape, rows, cols, params, parts):
        self.scheme, self.dtype, self.shape = scheme, dtype, shape
        self.rows, self.cols, self.params, self.parts = rows, cols, params, parts

    def fail(self, detail) -> Refused:
        return Refused("decode", f"{self.scheme.name}: {detail}")

    def p(self, key):
        return self.params[key]

    def has(self, role) -> bool:
        return role in self.parts

    def raw(self, role) -> bytes:
        return self.parts[role][2]

    def values(self, role) -> np.ndarray:
        dtype, _, data = self.parts[role]
        return elem.values(dtype, data)

    @property
    def lead(self) -> list:
        return list(self.shape[:-1])


def logical(c: Ctx):
    return "exact", list(c.shape)


def one(_: Ctx):
    return "count", 1
