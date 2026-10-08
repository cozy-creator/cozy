"""Typed projections of native records; these remain ordinary dictionaries.

Rust owns their validation and canonical bytes. These annotations neither grant
storage access nor reject additional fields from a newer compatible producer.
"""

from collections.abc import Sequence
from os import PathLike
from typing import Literal, NotRequired, Protocol, TypedDict

# Existing tuple ABI, named here so callers can discover each field's meaning.
# Object IDs accept the native API's prefixed sha256 form (or bare hex where
# documented); ObjectRef below is the canonical document's bare-hex mapping.
type ObjectKey = tuple[str, int]  # (object ID, byte length)
type TensorKey = tuple[str, str]  # (component, tensor name)
type PartKey = tuple[str, str, str]  # (component, tensor name, physical role)
type SourceMember = tuple[str, str, int, str]  # (member, object ID, length, URL)
type SourceProfile = tuple[str, str]  # (slot, profile)
type SourceHead = tuple[str, int, bytes]  # (member, total length, header bytes)
type SourceCheckpointKey = tuple[str, str, int]  # (slot, checkpoint ID, length)
type SourceCarrier = tuple[str, str, int, str | PathLike[str], bytes]
# SourceCarrier = (member, object ID, length, local path, header bytes).
type SourceBytes = tuple[str, str, int, bytes]  # (member, object ID, length, header)
type ReleaseLane = tuple[str, str, int]  # (lane, manifest ID, length)
type TensorRequirementRow = tuple[str, str, Sequence[int], str | None]
# TensorRequirementRow = (component, key, shape, optional logical dtype).


class Reader(Protocol):
    """Bounded native input stream; an exception from read propagates to the caller."""

    def read(self, size: int = -1, /) -> bytes: ...


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



class Asset(TypedDict):
    """One logical non-tensor file carried by a CozyTensors artifact."""

    logical_sha256: str
    logical_length: int
    media_type: str
    segments: list[ObjectRef]


class Encoding(TypedDict):
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
    encodings: list[Encoding]
    components: dict[str, dict[str, Tensor]]



class EnsureEvent(TypedDict):
    """One :func:`ensure` progress sample."""

    model: str
    step: str
    phase: Literal["resolving", "waiting", "collecting", "fetching"]
    bytes_done: int
    bytes_total: int
    #: Bytes per second over the last sample.
    rate: float
    attempt: int


class EnsureResult(TypedDict):
    """What :func:`ensure` did. ``bytes_fetched``, ``bytes_cached`` and
    ``cache_written_bytes`` over ``bytes_total`` are the cache observation."""

    model: str
    release: str
    lane: str
    scope: str
    manifest: str
    manifest_length: int
    bytes_total: int
    bytes_held: int
    bytes_fetched: int
    bytes_cached: int
    cache_written_bytes: int
    collected_bytes: int
    attempts: int
    joined: bool
    seconds: float



class Relief(TypedDict):
    """One :func:`relieve` pass."""

    pressure: bool
    collected_bytes: int
    capacity_bytes: int
    available_bytes: int
    #: Why a GC tier could not run (e.g. a live reader), when one could not.
    unable: str | None



class TreeRoot(TypedDict):
    owner: str
    producer: str
    receipt_digest: NotRequired[str]
    receipt: NotRequired[bytes]
    manifest_digest: str
    manifest_length: int
    complete: bool
    released: bool

class CheckpointRoot(TypedDict):
    owner: str
    repository: str
    manifest_digest: str
    manifest_length: int
    released: bool

class KeyedRoot(TypedDict):
    space: str
    key: str
    manifest_digest: str
    manifest_length: int
    bytes: int
    created_unix_ms: int

class SelectedSource(TypedDict):
    canonical: str
    selection_sha256: str
    content_manifest_digest: str
    content_manifest_length: int
    members: list[SourceMember]
    allowed_hosts: list[str]
    credential_hosts: list[str]
    skipped: list[str]

class MaterializedSource(TypedDict):
    artifact: TreeRoot
    transferred_bytes: int

class PreparedSource(TypedDict):
    """One prepared foreign source, as `Store.prepare_model_source` returns it."""

    slot: str
    profile: str
    manifest_digest: str
    manifest_length: int
class SourceCheckpoint(TypedDict):
    slot: str
    plan_digest: str
    head: str
    head_length: int
    index: int
    bytes: int

class CheckpointObject(TypedDict):
    kind: str
    object_id: str
    length: int

class SourceCheckpointPage(TypedDict):
    operation: str
    plan_digest: str
    index: int
    bytes: int
    previous: tuple[str, int] | None
    progress: tuple[str, int] | None
    next_offset: int | None
    objects: list[CheckpointObject]

class PreparedModelSource(TypedDict):
    replayed: bool
    complete: bool
    sources: list[PreparedSource]
    checkpoints: list[SourceCheckpoint]
    spent: list[str]
    converted_roles: int
    resumed_roles: int
    deferred_ops: int
    converted_bytes: int
    largest_op_bytes: int
    required_write_bytes: int
    write_interval_bytes: int

class DerivedWriteEstimate(TypedDict):
    payload_bytes: int
    largest_part_bytes: int
    parts: int

class ManifestRow(TypedDict):
    manifest_id: str
    manifest: bytes
    header: bytes | None

class ReleaseRow(TypedDict):
    org: str
    name: str
    version: str
    revision: int
    lane: str
    manifest_digest: str
    manifest_length: int

class LocalRow(TypedDict):
    name: str
    source_selection: str
    manifest_digest: str
    manifest_length: int


class Stats(TypedDict):
    catalog_opens: int
    presence_passes: int
    repo_cache_verified_reads: int
    admission_batches: int
    admitted_objects: int
    admission_nanos: int
    syncs: int
    sync_nanos: int
    recovery_rehashed: int
    recovery_removed: int
    recovery_nanos: int

# Store content, acquisition and transfer projections.
class PutResult(TypedDict):
    id: str
    length: int
    admitted: bool


class VerifyResult(TypedDict):
    verdict: Literal["verified", "invalidated", "corrupt_removed"]
    rehashed: NotRequired[bool]
    why: NotRequired[str]


class WalkObject(TypedDict):
    id: str
    length: int
    kind: str


class CheckoutResult(TypedDict):
    dirs: int
    links: int
    copies: int
    bytes: int


class AssetUpdate(TypedDict):
    header_id: str
    header_length: int
    manifest_id: str
    manifest_length: int


class RemovedRepository(TypedDict):
    """Successful removal currently returns an empty mapping; extra future fields are harmless."""


class ReadItem(TypedDict):
    what: str
    dest_off: int
    len: int
    source: Literal["object", "inline"]
    object: NotRequired[str]
    object_length: NotRequired[int]
    off: NotRequired[int]


class BatchItem(TypedDict):
    what: str
    off: int
    len: int
    io: bool


class ReadReceipt(TypedDict):
    op: str
    degraded: str
    manifest: str
    build: str
    device: str
    detail: str


class StreamResult(TypedDict):
    items: int
    bytes: int
    io_bytes: int
    readers: int
    slots: int
    elapsed_ms: int


class FetchPlan(TypedDict):
    session: str
    manifest: str
    held: list[ObjectKey]
    wanted: list[ObjectKey]
    wanted_bytes: int
    held_bytes: int
    presence: dict[str, str]
    document: bytes


class FetchAdmission(PutResult):
    key: str


class CheckpointTransfer(TypedDict):
    transferred: int
    held: bool
    http_status: NotRequired[int]  # push only


class SourceObject(TypedDict):
    object_id: str
    length: int
    resident: bool
    custody: str | None


class PreparedSelection(TypedDict):
    replayed: bool
    complete: bool
    spent: list[str]
    landed: list[str]
    converted_roles: int
    resumed_roles: int
    deferred_ops: int
    converted_bytes: int
    largest_op_bytes: int
    required_write_bytes: int
    sources: list[PreparedSource]


# 'class' is an existing wire key, so its declaration uses TypedDict's functional form.
Durability = TypedDict("Durability", {
    "class": str, "filesystem": str, "syncs": int, "sync_seconds": float,
})


class PullResult(TypedDict):
    ref: str
    model: str
    release: str
    lane: str
    manifest: str
    session: str
    store: str
    streams: int
    declared: int
    held: int
    fetched: int
    bytes_moved: int
    bytes_held: int
    bytes_total: int
    durability: Durability
    presence: dict[str, str]


class GCResult(TypedDict):
    dry_run: bool
    reclaimed_bytes: int
    reclaimed_blobs: int
    reclaimed_manifests: int
    kept_bytes: int
    kept_objects: int
    sessions: list[str]
    scratch_reaped: int


# Native declaration input maps, produced by derived.Derivation.native_arguments.
# These describe the native parser, not an independent validation implementation.
class PartSelection(TypedDict):
    source: str
    component: str
    tensor: str
    role: str


class PartDeclaration(TypedDict):
    dtype: str
    shape: Sequence[int]
    source: NotRequired[PartSelection]


class TensorDeclaration(TypedDict):
    logical_dtype: str
    shape: Sequence[int]
    encoding: str
    parts: dict[str, PartDeclaration]


class TargetDeclaration(TypedDict):
    source: NotRequired[str]
    source_component: NotRequired[str]
    drop: Sequence[str]
    add: dict[str, TensorDeclaration]


class ConfigDeclaration(TypedDict):
    kind: Literal["copy", "derive", "add"]
    source: NotRequired[str]
    source_config: NotRequired[str]


type DeclarationSources = dict[str, ObjectKey]
type DeclarationTargets = dict[str, TargetDeclaration]
type DeclarationConfigs = dict[str, ConfigDeclaration]
type DeclarationFiles = dict[str, str | ObjectKey]  # base64 bytes or retained object reference
type DerivationArguments = tuple[
    DeclarationSources, DeclarationTargets, DeclarationConfigs, list[TensorKey], int,
] | tuple[
    DeclarationSources, DeclarationTargets, DeclarationConfigs, list[TensorKey], int,
    DeclarationFiles,
]


class SourceFacts(TypedDict):
    alias: str
    manifest_id: str
    manifest_length: int
    header_id: str
    header_length: int
    components: list[str]


class InspectedPart(TypedDict):
    role: str
    dtype: str
    shape: list[int]


class InspectedTensor(TypedDict):
    key: str
    logical_dtype: str
    shape: list[int]
    encoding: str
    identity: str
    parts: list[InspectedPart]


class InspectedComponent(TypedDict):
    name: str
    tensors: list[InspectedTensor]


class InspectedConfig(TypedDict):
    name: str
    data: bytes


class InspectedSource(TypedDict):
    source: SourceFacts
    components: list[InspectedComponent]
    configs: list[InspectedConfig]


class Disposition(TypedDict):
    kind: Literal["pending", "adopted", "released"]
    scratch_root_id: NotRequired[str]


class DerivedLookup(TypedDict):
    state: Literal["absent", "open", "committed", "abandoned"]
    writer_session_id: NotRequired[int]
    receipt: NotRequired[DerivedReceipt]
    disposition: NotRequired[Disposition]


class AbandonResult(TypedDict):
    state: Literal["abandoned", "committed"]
    receipt: NotRequired[DerivedReceipt]


class DispositionResult(TypedDict):
    receipt: DerivedReceipt
    disposition: Disposition


class RetainedResult(TypedDict):
    retention_id: str
    transaction_id: str
    receipt_digest: str
    released: bool
    manifest: NotRequired[ObjectRef]
    tensorfs: NotRequired[str]


# Format inspection and compatibility results.
class ManifestEntry(TypedDict):
    path: str
    kind: str  # unknown future kinds retain their identity; materialization may refuse
    blob: ObjectRef


class ManifestDocument(TypedDict):
    entries: list[ManifestEntry]


class DocumentResult(TypedDict):
    id: str
    tensor_schema: str | None
    manifest: str | None


class CapabilityResult(TypedDict):
    admitted: bool
    code: str
    device: str
    encoding: str
    implementation: str
    note: str
    qualified_devices: list[str]
    reason: str
    vectors: str


class FitRoutes(TypedDict):
    decoded_float: int
    encoded_gemm: int
    verbatim: int


class FitResult(TypedDict):
    ok: bool
    code: str
    component: str
    custody: str
    detail: str
    device: str
    encoding: str
    ignored: list[str]
    ignored_bytes: int
    key: str
    routes: FitRoutes
    warning: str


__all__ = [
    "AbandonResult",
    "Asset",
    "AssetUpdate",
    "BatchItem",
    "CapabilityResult",
    "CheckoutResult",
    "CheckpointObject",
    "CheckpointReceipt",
    "CheckpointRoot",
    "CheckpointTransfer",
    "ConfigDeclaration",
    "DeclarationConfigs",
    "DeclarationFiles",
    "DeclarationSources",
    "DeclarationTargets",
    "DerivationArguments",
    "DerivedLookup",
    "DerivedReceipt",
    "DerivedSource",
    "DerivedWriteEstimate",
    "Disposition",
    "DispositionResult",
    "DocumentResult",
    "Durability",
    "Encoding",
    "EnsureEvent",
    "EnsureResult",
    "FetchAdmission",
    "FetchPlan",
    "FitResult",
    "FitRoutes",
    "GCResult",
    "Header",
    "InheritObservation",
    "InlinePart",
    "InspectedComponent",
    "InspectedConfig",
    "InspectedPart",
    "InspectedSource",
    "InspectedTensor",
    "KeyedRoot",
    "LocalRow",
    "ManifestDocument",
    "ManifestEntry",
    "ManifestRow",
    "MaterializedSource",
    "ObjectKey",
    "ObjectRef",
    "PartDeclaration",
    "PartKey",
    "PartReceipt",
    "PartSelection",
    "PreparedModelSource",
    "PreparedSelection",
    "PreparedSource",
    "PullResult",
    "PutResult",
    "ReadItem",
    "ReadReceipt",
    "Reader",
    "ReleaseLane",
    "ReleaseRow",
    "Relief",
    "RemovedRepository",
    "RetainedResult",
    "SegmentedPart",
    "SelectedSource",
    "SourceBytes",
    "SourceCarrier",
    "SourceCheckpoint",
    "SourceCheckpointKey",
    "SourceCheckpointPage",
    "SourceFacts",
    "SourceHead",
    "SourceMember",
    "SourceObject",
    "SourceProfile",
    "Stats",
    "StreamResult",
    "TargetDeclaration",
    "Tensor",
    "TensorDeclaration",
    "TensorKey",
    "TensorLogical",
    "TensorPart",
    "TensorRequirementRow",
    "TreeRoot",
    "VerifyResult",
    "WalkObject",
]
