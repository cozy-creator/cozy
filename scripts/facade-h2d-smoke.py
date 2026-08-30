#!/usr/bin/env python3
"""tfs-007 — the consumer seam: cr-005's fill loop with the facade as its reader.

cr-005's `internal/fill.CasStore` is the interim Python CAS reader this issue replaces.
This is that loop's SHAPE — pinned staging ring, asynchronous H2D on a runtime stream, a
completion event per slot, a fence before anything is readable — with `CasStore.read_into`
swapped for a real `ReadLease` + `stream`. cozy-runtime is not touched; this driver is a
copy, and the numbers are directly comparable to cr-005's banked table (same store, same
card, same window/slot/reader shape).

The property being demonstrated is not throughput alone: **a slot goes back to the ring
only when its H2D completion event has fired**, which is the lease discipline reaching
across the FFI. `Batch.release()` is called from the drain loop, not from the callback.

    PYTHONPATH=<torch site-packages> python scripts/facade-h2d-smoke.py <store> <manifest>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import pathlib
import time
from typing import Any

import tensorfs
import torch

_spec = importlib.util.spec_from_file_location(
    "facade_runs", pathlib.Path(__file__).resolve().parent / "facade-runs.py"
)
assert _spec and _spec.loader
_fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("store")
    p.add_argument("manifest")
    p.add_argument("--slot-bytes", type=int, default=16 << 20)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--readers", type=int, default=8)
    p.add_argument("--inflight", type=int, default=0)
    p.add_argument("--window", type=int, default=4 << 20)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--expect", default="")
    a = p.parse_args()

    store, header, header_bytes = _fr.load(a.store, a.manifest)
    plan = tensorfs.plan(header_bytes, _fr.construction_order(header), window=a.window)
    dev = torch.device("cuda:0")
    print(f"device:     {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print(f"plan:       {len(plan)} items, {plan.bytes} B, order = {plan.order}")

    dest = torch.empty(plan.bytes, dtype=torch.uint8, device=dev)
    ring = [
        torch.empty(a.slot_bytes, dtype=torch.uint8, pin_memory=True)
        for _ in range(a.slots)
    ]
    views = [memoryview(t.numpy()) for t in ring]
    stream = torch.cuda.Stream(device=dev)
    inflight: collections.deque[tuple[Any, Any]] = collections.deque()
    at = [0]

    if a.cold:
        n, b = _fr.evict(a.store)
        print(f"evicted:    {n} objects, {b} B (fadvise DONTNEED)")

    keep = a.inflight or max(1, a.slots // 2)
    lease = store.acquire_manifest(a.manifest)

    def drain(keep: int) -> None:
        """Release slots whose H2D has actually completed. NEVER on the return of a
        statement — on the event.

        Query first, block last: the callback runs on the pump's delivery thread, so a
        blocking wait here stops the ring from handing out the next filled slot even when
        readers have work ready. Blocking only when the caller is genuinely out of slots is
        worth ~1.7x (measured: 0.66 -> the row below).
        """
        while inflight and inflight[0][0].query():
            _, batch = inflight.popleft()
            batch.release()
        while len(inflight) > keep:
            event, batch = inflight.popleft()
            event.synchronize()
            batch.release()

    def on_batch(b: Any) -> None:
        off = at[0]
        with torch.cuda.stream(stream):
            dest[off : off + b.nbytes].copy_(ring[b.slot][: b.nbytes], non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        at[0] = off + b.nbytes
        inflight.append((event, b))
        # A slot whose H2D has not completed is NOT free — so the copies in flight and the
        # readers share the ring, and the caller sizes both. Holding slots-1 copies leaves
        # the pool nothing to fill and reads at queue depth 1; found by measuring.
        drain(keep)

    t0 = time.perf_counter()
    st = lease.stream(plan, views, on_batch, readers=a.readers)
    drain(0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(
        f"filled:     {st['items']} items, {at[0]} B to VRAM in {elapsed:.3f} s "
        f"= {plan.io_bytes / 1e9 / elapsed:.3f} GB/s (disk -> VRAM, fence included)"
    )
    print(
        f"ring:       {a.slots} slots, {keep} H2D copies in flight, {a.readers} readers\n"
        f"peak:       host RSS {_fr.rss_mib() / 1024:.2f} GiB, "
        f"VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
        f"ring {a.slots * a.slot_bytes / 2**20:.0f} MiB pinned (the CALLER's)"
    )

    if a.verify:
        t1 = time.perf_counter()
        digest = hashlib.sha256()
        back = torch.empty(1 << 26, dtype=torch.uint8, pin_memory=True)
        for off in range(0, plan.bytes, 1 << 26):
            n = min(1 << 26, plan.bytes - off)
            back[:n].copy_(dest[off : off + n])
            digest.update(memoryview(back.numpy())[:n])
        got = digest.hexdigest()
        print(f"on-device:  sha256 {got} ({time.perf_counter() - t1:.1f} s D2H + hash)")
        if a.expect:
            want = a.expect.removeprefix("sha256:")
            print(f"BANKED:     {'MATCH' if got == want else 'MISMATCH'} vs {want[:12]}…")

    lease.release()
    print(f"lease:      released after the fence (live={lease.live})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
