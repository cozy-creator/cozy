"""tensorfs — the typed facade over the compiled core (tensorfs.md §10, tfs-007).

Marshal only. Every byte-level fact — hashing, canonical bytes, geometry, verification,
leases, the reader pool — comes from the extension, and `scripts/fence.py` enforces that
this package computes none of them. What lives here is types and names.

The default inference reader returns owned Torch tensors (install `tensorfs[torch]`).
Torch loads only when tensor access is requested. Storage and metadata remain lightweight.
The lower-level read interface still fills caller-owned buffers::

    import torch, tensorfs

    store = tensorfs.Store.open("/var/lib/tensorfs")
    manifest = store.manifest(manifest_id)
    plan  = tensorfs.plan(manifest["header"], traversal, window=WINDOW)

    slots = [torch.empty(SLOT, dtype=torch.uint8, pin_memory=True) for _ in range(4)]
    views = [memoryview(s.numpy()) for s in slots]

    with store.acquire_cozytensors(manifest_id) as lease:
        def on_batch(b: tensorfs.Batch) -> None:
            copy_h2d(slots[b.slot][: b.nbytes])   # asynchronous
            b.release()                            # after the completion event
        lease.stream(plan, views, on_batch, readers=8)

The ring is YOURS: slot count, slot size and queue depth are the caller's decisions, and
`release()` is a positive act so a slot cannot be recycled under a DMA that is still
running. `stream` never stops the reader pool at a batch boundary — that is the whole
reason it exists (tfs-005 measured the drain at ~10%).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Literal, TypedDict, cast

from . import errors
from .derived import (
    Derivation,
    OutputCapability,
    SourceCapability,
    SourceInspection,
    derive,
)
from ._tensors import (
    TensorAccessError,
    TensorAllocation,
    TensorReader,
    from_parts,
    physical_tensors,
    tensors,
)
from ._ext import (
    CAPABILITIES as _CAPABILITIES,
    ensure as _ensure,
    relieve as _relieve,
    _fit,
    Batch,
    Closure,
    DerivedWriter,
    Operation,
    PullCancellation,
    ReadLease,
    ReadPlan,
    RepoObjectCache,
    Store,
    __version__,
    capability,
    dtypes,
    doc_kinds,
    fetch_admit,
    fetch_complete,
    fetch_complete_cozytensors,
    fetch_plan,
    gc,
    home,
    manifest_max_bytes,
    model_source_objects,
    object_id,
    parse_header as _parse_header,
    parse_manifest,
    plan,
    platform_digests,
    prepare_selected_source,
    pull,
    platform_gate,
    read_source_heads,
    select_source_profile,
    source_profile_converters,
    STREAM_MEMORY,
    AS_IS_PROFILE,
    transfer_streams,
    receipt_binds,
    record_model_source_custody,
    recanonicalize,
    render,
    run_doc,
    seed_digests,
    stats,
    transfer_object,
    transfer_manifest,
)


class ObjectRef(TypedDict):
    """One content-addressed body segment."""

    sha256: str
    length: int


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


def parse_header(data: bytes, closure: Closure | None = None) -> Header:
    """Parse a verified CozyTensors header into TensorFS's typed Python projection."""

    return cast(Header, _parse_header(data, closure))


@dataclass(frozen=True)
class TensorRequirement:
    """One tensor constructed for this request."""

    component: str
    key: str
    shape: Sequence[int]
    logical_dtype: str | None = None


def fit(
    requirements: Sequence[TensorRequirement],
    header: bytes,
    *,
    custody: str,
    encoded_leaves: bool,
    device: str | None = None,
    observations: bytes | None = None,
) -> dict[str, object]:
    """Compare a transient factory census with one CozyTensors header."""
    rows = [
        (
            tensor.component,
            tensor.key,
            list(tensor.shape),
            tensor.logical_dtype,
        )
        for tensor in requirements
    ]
    return cast(
        dict[str, object],
        _fit(
            rows,
            header,
            custody=custody,
            encoded_leaves=encoded_leaves,
            device=device,
            observations=observations,
        ),
    )


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


def ensure(
    store: Store,
    ref: str,
    *,
    hub: str,
    lane: str = "",
    step: str = "",
    credential: str = "",
    ca_file: str | PathLike[str] | None = None,
    allowed_hosts: Sequence[str] = (),
    allow_local: bool = False,
    keep: Sequence[str] = (),
    streams: int = 512,
    streams_start: int = 16,
    sample_seconds: float | None = None,
    cancellation: PullCancellation | None = None,
    progress: Callable[[EnsureEvent], object] | None = None,
) -> EnsureResult:
    """Make one model resident in ``store``: resolve ``ref`` (``org/name[@release][@sha256:<hex>]``)
    at ``hub``, share one flight per manifest with every process on this Store, admit it
    against the disk (running the GC policy when short; ``keep`` names manifests it must not
    evict), and pull, retrying only after measured progress. ``ca_file`` adds the hub's CA to
    this process's trust. Refusals raise typed ``errors.*`` (``TlsUntrusted``, ``RefNotFound``,
    ``CredentialRequired``, ``CapacityExhausted``, ``Stalled``, ``HubRefused``, ...); a raising
    ``progress`` cancels and its exception is re-raised."""

    return cast(
        EnsureResult,
        _ensure(
            store,
            ref,
            hub=hub,
            lane=lane,
            step=step,
            credential=credential,
            ca_file=ca_file,
            allowed_hosts=allowed_hosts,
            allow_local=allow_local,
            keep=keep,
            streams=streams,
            streams_start=streams_start,
            sample_seconds=sample_seconds,
            cancellation=cancellation,
            progress=None if progress is None else lambda event: progress(cast(EnsureEvent, event)),
        ),
    )


class Relief(TypedDict):
    """One :func:`relieve` pass."""

    pressure: bool
    collected_bytes: int
    capacity_bytes: int
    available_bytes: int
    #: Why a GC tier could not run (e.g. a live reader), when one could not.
    unable: str | None


def relieve(store: Store, *, keep: Sequence[str] = ()) -> Relief:
    """One pressure pass: when at most 1/10 of the Store's filesystem is free (or it is under
    its reserve), run the GC policy (unreferenced, delivered products, cached lanes) until
    more than 1/5 is free. Manifests in ``keep`` and in live downloads are never evicted."""

    return cast(Relief, _relieve(store, keep=keep))


#: Feature names a caller detects instead of versions: ``"ensure/1"``, ``"get-range/1"``,
#: ``"deliver/1"``, ``"pressure/1"``, ``"keyed-roots/1"``.
CAPABILITIES: frozenset[str] = frozenset(_CAPABILITIES)

#: The closed carrier/logical dtype enum and its element sizes. The ONE export of dtype
#: knowledge: a consumer sizing a `torch.frombuffer` reads it from here, never from a
#: table of its own.
DTYPES: dict[str, int] = dtypes()

__all__ = [
    "CAPABILITIES",
    "EnsureEvent",
    "EnsureResult",
    "Relief",
    "ensure",
    "relieve",
    "TensorAccessError",
    "TensorAllocation",
    "TensorReader",
    "from_parts",
    "physical_tensors",
    "tensors",
    "Derivation",
    "OutputCapability",
    "SourceCapability",
    "SourceInspection",
    "derive",
    "Asset",
    "DTYPES",
    "Batch",
    "Closure",
    "Encoding",
    "Header",
    "InlinePart",
    "ObjectRef",
    "SegmentedPart",
    "Tensor",
    "TensorLogical",
    "TensorPart",
    "TensorRequirement",
    "DerivedWriter",
    "Operation",
    "PullCancellation",
    "ReadLease",
    "ReadPlan",
    "RepoObjectCache",
    "Store",
    "__version__",
    "capability",
    "dtypes",
    "doc_kinds",
    "fit",
    "errors",
    "fetch_admit",
    "fetch_complete",
    "fetch_complete_cozytensors",
    "fetch_plan",
    "gc",
    "home",
    "manifest_max_bytes",
    "model_source_objects",
    "object_id",
    "parse_header",
    "parse_manifest",
    "plan",
    "platform_digests",
    "platform_gate",
    "prepare_selected_source",
    "pull",
    "read_source_heads",
    "select_source_profile",
    "source_profile_converters",
    "STREAM_MEMORY",
    "AS_IS_PROFILE",
    "transfer_streams",
    "receipt_binds",
    "record_model_source_custody",
    "recanonicalize",
    "render",
    "run_doc",
    "seed_digests",
    "stats",
    "transfer_object",
    "transfer_manifest",
]
