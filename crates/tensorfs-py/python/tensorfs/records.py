"""Typed projections of a CBOR header (`parse_header`) and its read plan; ordinary dicts."""

from typing import Literal, NotRequired, TypedDict


class ObjectRef(TypedDict):
    """An immutable object's bare hexadecimal SHA-256 and byte length."""

    sha256: str
    length: int


class Asset(TypedDict):
    """One logical non-tensor file carried by a CozyTensors artifact."""

    logical_sha256: str
    logical_length: int
    media_type: str
    segments: list[ObjectRef]


class EncodingRow(TypedDict):
    """One encoding definition and the digest tensor rows cite."""

    id: str
    definition: object


class TensorLogical(TypedDict):
    """The execution-visible tensor schema, independent of stored packing."""

    logical_dtype: str
    shape: list[int]


class InlinePart(TypedDict):
    dtype: str
    shape: list[int]
    inline: bytes


class SegmentedPart(TypedDict):
    dtype: str
    shape: list[int]
    segments: list[ObjectRef]


type TensorPart = InlinePart | SegmentedPart


class Tensor(TypedDict):
    logical: TensorLogical
    encoding: str
    parts: dict[str, TensorPart]


class Header(TypedDict):
    """The exact Python projection returned by :func:`parse_header`."""

    format: Literal["cozytensors/1"]
    configs: dict[str, bytes]
    assets: dict[str, Asset]
    encodings: list[EncodingRow]
    components: dict[str, dict[str, Tensor]]


class ReadItem(TypedDict):
    what: str
    dest_off: int
    len: int
    source: Literal["object", "inline"]
    object: NotRequired[str]
    object_length: NotRequired[int]
    off: NotRequired[int]


__all__ = [
    "Asset",
    "EncodingRow",
    "Header",
    "InlinePart",
    "ObjectRef",
    "ReadItem",
    "SegmentedPart",
    "Tensor",
    "TensorLogical",
    "TensorPart",
]
