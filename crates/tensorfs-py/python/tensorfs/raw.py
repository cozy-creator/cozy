"""Physical format inspection for storage integrations."""

from ._ext import raw as _raw

Part = _raw.Part
parts = _raw.parts

__all__ = ["Part", "parts"]
