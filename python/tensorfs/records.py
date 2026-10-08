"""Typed projections of native records; these remain ordinary dictionaries.

Rust owns their validation and canonical bytes. These annotations neither grant
storage access nor reject additional fields from a newer compatible producer.
"""

from typing import TypedDict


class ObjectRef(TypedDict):
    """An immutable object's bare hexadecimal SHA-256 and byte length."""

    sha256: str
    length: int


class DerivedSource(TypedDict):
    """The source facts retained by a native derived commit."""

    alias: str
    components: list[str]
    header: ObjectRef
    manifest: ObjectRef


class InheritObservation(TypedDict):
    bytes: int
    hashes: int
    objects: int
    reads: int


class DerivedReceipt(TypedDict):
    """Native commit facts, identical through a Store writer and a scoped channel.

    A receipt describes the committed artifact; the owner separately adopts and
    retains it. The transaction identity survives replacement writer attempts.
    """

    transaction_id: str
    declaration_digest: str
    declaration_length: int
    sources: list[DerivedSource]
    header: ObjectRef
    manifest: ObjectRef
    added_objects: list[ObjectRef]
    inherit_observation: InheritObservation


class CheckpointReceipt(TypedDict):
    """Resumable progress, not a commit or a retention receipt.

    ``head`` is a prefixed object ID, or None before a checkpoint has a head.
    Its existing wire shape stays intact.
    """

    plan_digest: str
    head: str | None
    head_length: int
    index: int
    bytes: int


class PartReceipt(TypedDict):
    """The result of adding one declared tensor part."""

    bytes: int
    deduped_bytes: int
    whole_sha256: str
    objects: list[ObjectRef]


__all__ = [
    "ObjectRef", "DerivedSource", "InheritObservation", "DerivedReceipt",
    "CheckpointReceipt", "PartReceipt",
]
