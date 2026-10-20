"""Physical format inspection for storage integrations."""

from ._ext import raw as _raw

Part = _raw.Part
parts = _raw.parts
decode_into = _raw.decode_into

__all__ = ["Part", "decode_into", "parts"]
