"""Typed refusals.

Every refusal code in the compiled core has exactly one exception class here, and the set
is BUILT IN RUST from `err::Code::ALL` — this module names none of them, so the two
languages cannot drift. Catch one (`except errors.ShortRead`) or the family::

    try:
        lease.read_into(oid, length, off, n, buf)
    except errors.Refusal as e:
        log(e.code, e.detail)      # "SHORT_READ", "...is 512 B on disk, the reference..."

`CODES` maps the frozen code string to its class for callers that dispatch on the wire
name rather than on the type.
"""

from __future__ import annotations

from . import _ext

Refusal = _ext.Refusal
CODES: dict[str, type[BaseException]] = _ext.CODES

__all__ = ["CODES", "Refusal"]


def __getattr__(name: str) -> type[BaseException]:
    """A refusal class by either spelling: `ShortRead` or `SHORT_READ`."""
    cls = CODES.get(name) or getattr(_ext, name, None)
    if isinstance(cls, type) and issubclass(cls, BaseException):
        return cls
    raise AttributeError(f"tensorfs.errors has no refusal named {name!r}")
