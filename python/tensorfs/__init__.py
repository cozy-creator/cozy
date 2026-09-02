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
from typing import Literal

from . import errors
from ._ext import (
    _code_topology,
    Batch,
    Closure,
    DerivedWriter,
    Operation,
    ReadLease,
    ReadPlan,
    Store,
    __version__,
    capability,
    config_digest,
    construction_document,
    digests,
    dtypes,
    doc_kinds,
    fit,
    fetch_admit,
    fetch_complete,
    fetch_complete_cozytensors,
    fetch_plan,
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
    shape_digest,
    topology_digest,
    transfer_object,
    transfer_manifest,
)


@dataclass(frozen=True)
class CodeTensor:
    """One destination the factory constructs: its shape, constructed dtype, which stored
    encodings it accepts (`"any"` under ``encoded_leaves="accept"``, a digest list
    otherwise), and whether the dtype is PINNED (an integer buffer, an fp32 island)."""

    component: str
    key: str
    dtype: str
    shape: Sequence[int]
    accepts: Literal["any"] | Sequence[str]
    pinned: bool = False


def code_topology(
    contract_digest: str,
    tensors: Sequence[CodeTensor],
    constructed_components: Sequence[str] | None = None,
) -> bytes:
    """Author canonical CodeTopology bytes from the meta-device census rows.

    `constructed_components` is the factory's construction order (D9); it defaults to the
    order the components first appear in `tensors`.
    """
    rows = [
        (
            tensor.component,
            tensor.key,
            tensor.dtype,
            list(tensor.shape),
            None if tensor.accepts == "any" else list(tensor.accepts),
            bool(tensor.pinned),
        )
        for tensor in tensors
    ]
    if constructed_components is None:
        constructed_components = list(dict.fromkeys(tensor.component for tensor in tensors))
    return _code_topology(contract_digest, rows, list(constructed_components))

#: The closed carrier/logical dtype enum and its element sizes. The ONE export of dtype
#: knowledge: a consumer sizing a `torch.frombuffer` reads it from here, never from a
#: table of its own.
DTYPES: dict[str, int] = dtypes()

__all__ = [
    "DTYPES",
    "Batch",
    "Closure",
    "CodeTensor",
    "DerivedWriter",
    "Operation",
    "ReadLease",
    "ReadPlan",
    "Store",
    "__version__",
    "capability",
    "code_topology",
    "config_digest",
    "construction_document",
    "digests",
    "dtypes",
    "doc_kinds",
    "fit",
    "errors",
    "fetch_admit",
    "fetch_complete",
    "fetch_complete_cozytensors",
    "fetch_plan",
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
    "shape_digest",
    "topology_digest",
    "transfer_object",
    "transfer_manifest",
]
