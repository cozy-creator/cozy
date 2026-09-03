"""tensorfs — the typed facade over the compiled core (tensorfs.md §10, tfs-007).

Marshal only. Every byte-level fact — hashing, canonical bytes, geometry, verification,
leases, the reader pool — comes from the extension, and `scripts/fence.py` enforces that
this package computes none of them. What lives here is types and names.

Torch is OUTSIDE the boundary. A read fills a buffer YOU own and hands back plain data;
element sizes come from `DTYPES`, which is the compiled dtype enum, exported once::

    import torch, tensorfs

    store = tensorfs.Store.open("/var/lib/cozy/store")
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

from collections.abc import Sequence
from dataclasses import dataclass
from . import errors
from ._ext import (
    _fit,
    Batch,
    Closure,
    DerivedWriter,
    Operation,
    ReadLease,
    ReadPlan,
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
    manifest_max_bytes,
    object_id,
    parse_header,
    parse_manifest,
    plan,
    platform_digests,
    platform_gate,
    receipt_binds,
    recanonicalize,
    render,
    run_doc,
    seed_digests,
    transfer_object,
    transfer_manifest,
)


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
    return _fit(
        rows,
        header,
        custody=custody,
        encoded_leaves=encoded_leaves,
        device=device,
        observations=observations,
    )

#: The closed carrier/logical dtype enum and its element sizes. The ONE export of dtype
#: knowledge: a consumer sizing a `torch.frombuffer` reads it from here, never from a
#: table of its own.
DTYPES: dict[str, int] = dtypes()

__all__ = [
    "DTYPES",
    "Batch",
    "Closure",
    "TensorRequirement",
    "DerivedWriter",
    "Operation",
    "ReadLease",
    "ReadPlan",
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
    "manifest_max_bytes",
    "object_id",
    "parse_header",
    "parse_manifest",
    "plan",
    "platform_digests",
    "platform_gate",
    "receipt_binds",
    "recanonicalize",
    "render",
    "run_doc",
    "seed_digests",
    "transfer_object",
    "transfer_manifest",
]
