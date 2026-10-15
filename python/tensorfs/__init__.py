"""tensorfs — the typed facade over the compiled core (tensorfs.md §10, tfs-007).

Marshal only. Every byte-level fact — hashing, canonical bytes, geometry, verification,
leases, the reader pool — comes from the extension, and `scripts/fence.py` enforces that
this package computes none of them. What lives here is types and names.

TensorFS is a library used in TensorD and Runtime's executor processes. TensorD makes
weight sources and shared CPU buffers available and coordinates machine-wide memory
policy. Runtime constructs models and performs GPU operations through the local plane
and device APIs within those grants; TensorD may retain exported GPU allocation handles.
Serialized Store content, prepared CPU buffers and GPU tensors have distinct lifetimes.

TensorFS never imports Torch. The following standalone Store example fills caller-owned
buffers. Managed Runtime uses owner-supplied sources and buffers instead of opening the
managed Store::

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
from typing import TYPE_CHECKING

from . import errors
from .records import (
    CheckpointReceipt, DerivedReceipt, ObjectRef, PartReceipt, FitResult, TensorPart,
    Asset, Encoding, TensorLogical, InlinePart, SegmentedPart, Tensor, Header, EnsureEvent, EnsureResult, Relief,
)
from .derived import (
    Derivation,
    OutputCapability,
    SourceCapability,
    SourceInspection,
    derive,
)
from ._ext import (
    CAPABILITIES as _CAPABILITIES,
    STORE,
    _fit,
    Closure,
    ReadPlan,
    __version__,
    capability,
    dtypes,
    manifest_max_bytes,
    object_id,
    parse_header as _parse_header,
    parse_manifest,
    plan,
    platform_digests,
    platform_gate,
    recanonicalize,
    seed_digests,
)

# The ordinary Python wheel always includes the full extension and CLI. Rust can
# still build without store features; installed code is never a managed executor grant.
# The channel protocols above (derived outputs, tensor reads) import in both: what serves
# them, a store, exists only in the full build.
if TYPE_CHECKING or STORE:
    from ._ext import (
        ensure as _ensure,
        relieve as _relieve,
        Batch,
        DerivedWriter,
        Operation,
        PullCancellation,
        ReadLease,
        RepoObjectCache,
        Store,
        doc_kinds,
        fetch_admit,
        fetch_complete,
        fetch_complete_cozytensors,
        fetch_plan,
        gc,
        home,
        model_source_objects,
        prepare_selected_source,
        pull,
        read_source_heads,
        select_source_profile,
        source_profile_converters,
        STREAM_MEMORY,
        AS_IS_PROFILE,
        transfer_streams,
        receipt_binds,
        record_model_source_custody,
        render,
        run_doc,
        stats,
        transfer_object,
        transfer_manifest,
    )


def parse_header(data: bytes, closure: Closure | None = None) -> Header:
    """Parse a verified CozyTensors header into TensorFS's typed Python projection."""

    return _parse_header(data, closure)


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
) -> FitResult:
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
    return _fit(
        rows,
        header,
        custody=custody,
        encoded_leaves=encoded_leaves,
        device=device,
        observations=observations,
    )


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
    sample_seconds: float | None = None,
    cancellation: PullCancellation | None = None,
    progress: Callable[[EnsureEvent], object] | None = None,
) -> EnsureResult:
    """Make one model resident in ``store``: resolve ``ref`` (``org/name[@release][@sha256:<hex>]``)
    at ``hub``, share one flight per manifest with every process on this Store, admit it
    against the disk (running the GC policy when short; ``keep`` names manifests it must not
    evict), and fetch what it lacks through the Store's fetcher. ``ca_file`` adds the hub's CA to
    this process's trust. Refusals raise typed ``errors.*`` (``TlsUntrusted``, ``RefNotFound``,
    ``CredentialRequired``, ``CapacityExhausted``, ``Stalled``, ``HubRefused``, ...); a raising
    ``progress`` cancels and its exception is re-raised."""

    return _ensure(
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
        sample_seconds=sample_seconds,
        cancellation=cancellation,
        progress=progress,
    )


def relieve(store: Store, *, keep: Sequence[str] = ()) -> Relief:
    """One pressure pass: when at most 1/10 of the Store's filesystem is free (or it is under
    its reserve), run the GC policy (unreferenced, delivered products, cached lanes) until
    more than 1/5 is free. Manifests in ``keep`` and in live downloads are never evicted."""

    return _relieve(store, keep=keep)


#: Feature names a caller detects instead of versions: ``"ensure/1"``, ``"get-range/1"``,
#: ``"deliver/1"``, ``"pressure/1"``, ``"keyed-roots/1"``.
CAPABILITIES: frozenset[str] = frozenset(_CAPABILITIES)

#: The closed carrier/logical dtype enum and its element sizes. The ONE export of dtype
#: knowledge: a consumer sizing a `torch.frombuffer` reads it from here, never from a
#: table of its own.
DTYPES: dict[str, int] = dtypes()

__all__ = [
    "STORE",
    "Derivation",
    "OutputCapability",
    "SourceCapability",
    "SourceInspection",
    "derive",
    "CAPABILITIES",
    "Asset",
    "DTYPES",
    "Closure",
    "Encoding",
    "Header",
    "InlinePart",
    "ObjectRef",
    "DerivedReceipt",
    "CheckpointReceipt",
    "PartReceipt",
    "SegmentedPart",
    "Tensor",
    "TensorLogical",
    "TensorPart",
    "TensorRequirement",
    "ReadPlan",
    "__version__",
    "capability",
    "dtypes",
    "fit",
    "errors",
    "manifest_max_bytes",
    "object_id",
    "parse_header",
    "parse_manifest",
    "plan",
    "platform_digests",
    "platform_gate",
    "recanonicalize",
    "seed_digests",
]
if STORE:
    __all__ += [
        "EnsureEvent",
        "EnsureResult",
        "Relief",
        "ensure",
        "relieve",
        "Batch",
        "DerivedWriter",
        "Operation",
        "PullCancellation",
        "ReadLease",
        "RepoObjectCache",
        "Store",
        "doc_kinds",
        "fetch_admit",
        "fetch_complete",
        "fetch_complete_cozytensors",
        "fetch_plan",
        "gc",
        "home",
        "model_source_objects",
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
        "render",
        "run_doc",
        "stats",
        "transfer_object",
        "transfer_manifest",
    ]
