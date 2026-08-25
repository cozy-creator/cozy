"""tensorfs — the typed facade over the compiled core (tensorfs.md §10).

Marshal only. Every byte-level fact comes from the extension; nothing here computes
one. A CI fence (scripts/fence.py) enforces that. tfs-007 owns the full facade.
"""

from __future__ import annotations

from typing import Any

from . import _ext

__all__ = ["object_id", "parse_header", "__version__"]

__version__: str = _ext.__version__


def object_id(data: bytes) -> str:
    """Identity of exactly these bytes, as `sha256:<hex>`."""
    return _ext.object_id(data)


def parse_header(data: bytes) -> dict[str, Any]:
    """Canonical CozyTensorsHeader bytes -> plain dict. A refusal raises ValueError."""
    return _ext.parse_header(data)
