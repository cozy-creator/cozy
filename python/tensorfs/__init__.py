"""tensorfs — the typed facade over the compiled core (tensorfs.md §10, tfs-007).

Marshal only. Every byte-level fact — hashing, canonical bytes, geometry, verification,
leases, the reader pool — comes from the extension, and `scripts/fence.py` enforces that
this package computes none of them. What lives here is types and names.

Torch is OUTSIDE the boundary. A read fills a buffer YOU own and hands back plain data;
element sizes come from `DTYPES`, which is the compiled dtype enum, exported once::

    import torch, tensorfs

    store = tensorfs.Store.open("/var/lib/cozy/store")
    snap  = store.snapshot(snapshot_id)
    plan  = tensorfs.plan(snap["header"], traversal, window=WINDOW)

    slots = [torch.empty(SLOT, dtype=torch.uint8, pin_memory=True) for _ in range(4)]
    views = [memoryview(s.numpy()) for s in slots]

    with store.acquire_snapshot(snapshot_id) as lease:
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

from . import errors
from ._ext import (
    ERA,
    Batch,
    Closure,
    ReadLease,
    ReadPlan,
    Store,
    __version__,
    doc_kinds,
    dtypes,
    fetch_admit,
    fetch_complete,
    fetch_plan,
    ingest_installed,
    ingest_open_root,
    ingest_read_root,
    ingest_reap,
    object_id,
    parse_header,
    parse_manifest,
    plan,
    plan_in_key_order,
    platform_digests,
    platform_gate,
    receipt_binds,
    recanonicalize,
    render,
    run_doc,
    seed_digests,
    topology_digest,
    transfer_object,
)

#: The closed carrier/logical dtype enum and its element sizes. The ONE export of dtype
#: knowledge: a consumer sizing a `torch.frombuffer` reads it from here, never from a
#: table of its own.
DTYPES: dict[str, int] = dtypes()

__all__ = [
    "DTYPES",
    "ERA",
    "Batch",
    "Closure",
    "ReadLease",
    "ReadPlan",
    "Store",
    "__version__",
    "doc_kinds",
    "dtypes",
    "errors",
    "fetch_admit",
    "fetch_complete",
    "fetch_plan",
    "ingest_installed",
    "ingest_open_root",
    "ingest_read_root",
    "ingest_reap",
    "object_id",
    "parse_header",
    "parse_manifest",
    "plan",
    "plan_in_key_order",
    "platform_digests",
    "platform_gate",
    "receipt_binds",
    "recanonicalize",
    "render",
    "run_doc",
    "seed_digests",
    "topology_digest",
    "transfer_object",
]
