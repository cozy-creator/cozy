#!/usr/bin/env python3
"""tfs-007 — the Python facade, run live against real stores.

    scripts/facade-runs.py arms    <store> <manifest>   typed refusals across the boundary
    scripts/facade-runs.py real    <store> <manifest>   lease + verified full read + digest
    scripts/facade-runs.py bench   <store> <manifest>   the facade / interim-reader sweep

No test framework, no asserts-as-tests: every line below drives a real store and prints
what actually happened (tracker README #160).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

import tensorfs
from tensorfs import errors

ROOT = pathlib.Path(__file__).resolve().parent.parent
TFS = ROOT / "target/release/tfs"


# --------------------------------------------------------------------------- helpers


def rss_mib() -> float:
    for line in pathlib.Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return 0.0


def construction_order(header: dict[str, Any]) -> list[tuple[str, str]]:
    comps = list(header["components"])
    return [
        (c, k) for c in comps for k in header["components"][c]
    ]


def evict(store: str) -> tuple[int, int]:
    """posix_fadvise(DONTNEED) — no passwordless sudo on this box for drop_caches."""
    n = b = 0
    for p in (pathlib.Path(store) / "blobs").rglob("*"):
        if not p.is_file():
            continue
        fd = os.open(p, os.O_RDONLY)
        try:
            st = os.fstat(fd)
            os.posix_fadvise(fd, 0, st.st_size, os.POSIX_FADV_DONTNEED)
            n += 1
            b += st.st_size
        finally:
            os.close(fd)
    return n, b


def load(store_path: str, manifest: str) -> tuple[Any, dict[str, Any], bytes]:
    store = tensorfs.Store.open(store_path)
    observed = store.manifest(manifest)
    header = tensorfs.parse_header(observed["header"])
    return store, header, observed["header"]


# --------------------------------------------------------------------------- streaming


def stream_facade(
    lease: Any,
    plan: Any,
    slot_bytes: int,
    nslots: int,
    readers: int,
    consume: Callable[[memoryview], None] | None,
) -> Any:
    """The facade's streaming shape: a ring of CALLER-owned slots, released explicitly."""
    slots = [bytearray(slot_bytes) for _ in range(nslots)]
    views = [memoryview(s) for s in slots]

    def on_batch(b: Any) -> None:
        if consume is not None:
            consume(views[b.slot][: b.nbytes])
        b.release()

    return lease.stream(plan, views, on_batch, readers=readers)


def stream_interim(
    cas: Any,
    plan: Any,
    slot_bytes: int,
    nslots: int,
    readers: int,
    consume: Callable[[memoryview], None] | None,
) -> dict[str, int]:
    """cr-005's INTERIM pure-Python CAS reader (`internal/fill.CasStore`), driven over the
    same plan, the same tiling, the same slot ring and the same queue depth. This is the
    thing tfs-007 replaces, so it is measured rather than remembered.

    The slot is claimed by the SUBMITTER, in batch order, exactly as `read::pump` claims
    (next batch, free slot) as one act. Letting the workers race for slots instead deadlocks
    whenever readers > slots — a later batch can win the permit an earlier one is waiting
    for while the consumer waits for the earlier one. Found by hanging, not by reading.
    """
    items = plan.items()
    batches: list[list[dict[str, Any]]] = [[]]
    used = 0
    for it in items:
        if batches[-1] and used + it["len"] > slot_bytes:
            batches.append([])
            used = 0
        batches[-1].append(it)
        used += it["len"]

    slots = [bytearray(slot_bytes) for _ in range(nslots)]
    views = [memoryview(s) for s in slots]
    free: list[int] = list(range(nslots))
    read_bytes = 0

    def fill(batch: list[dict[str, Any]], slot: int) -> int:
        at = 0
        for it in batch:
            if it["source"] == "object":
                cas.read_into(
                    it["object"].removeprefix("sha256:"),
                    it["off"],
                    it["len"],
                    views[slot][at : at + it["len"]],
                )
            at += it["len"]
        return at

    inflight: collections.deque[tuple[Any, int]] = collections.deque()
    with cf.ThreadPoolExecutor(readers) as ex:

        def drain(keep: int) -> None:
            nonlocal read_bytes
            while len(inflight) > keep:
                fut, slot = inflight.popleft()
                n = fut.result()
                read_bytes += n
                if consume is not None:
                    consume(views[slot][:n])
                free.append(slot)

        for batch in batches:
            while not free:
                drain(len(inflight) - 1)
            slot = free.pop()
            inflight.append((ex.submit(fill, batch, slot), slot))
        drain(0)
    return {"bytes": read_bytes, "batches": len(batches)}


def open_interim(store_path: str) -> Any:
    sys.path.insert(0, str(pathlib.Path.home() / "cozy_v2/cozy-runtime/src"))
    from cozy_runtime.internal.fill import CasStore  # type: ignore[import-not-found]

    return CasStore(store_path)


# --------------------------------------------------------------------------- real


def cmd_real(a: argparse.Namespace) -> int:
    store, header, header_bytes = load(a.store, a.manifest)
    print(f"store:      {store.root}")
    order = construction_order(header)
    print(f"traversal:  {len(order)} tensors in module order, first {order[0][1]}")

    t0 = time.perf_counter()
    lease = store.acquire_manifest(a.manifest)
    ms = (time.perf_counter() - t0) * 1000
    print(
        f"lease:      {len(lease.objects)} objects, {lease.bytes} B verified in {ms:.0f} ms, "
        f"{len(lease.receipts)} receipts, hold {lease.hold_id[:16]}"
    )

    plan = tensorfs.plan(header_bytes, order, window=a.window)
    print(
        f"plan:       {len(plan)} items, {plan.bytes} B ({plan.io_bytes} B disk leg), "
        f"{plan.inline_parts} inline, order = {plan.order}"
    )
    print(f"ring:       {a.slots} slots x {a.slot_bytes} B, {a.readers} readers, {plan.batches(a.slot_bytes)} batches")

    digest = hashlib.sha256()
    hashed = [0.0]

    def consume(view: memoryview) -> None:
        t = time.perf_counter()
        digest.update(view)
        hashed[0] += time.perf_counter() - t

    t0 = time.perf_counter()
    st = stream_facade(lease, plan, a.slot_bytes, a.slots, a.readers, consume)
    wall = time.perf_counter() - t0
    print(
        f"filled:     {st['items']} items, {st['bytes']} B in {wall:.2f} s wall "
        f"({st['io_bytes'] / 1e9 / wall:.3f} GB/s end to end, digest INCLUDED)"
    )
    print(
        f"digest:     {hashed[0]:.2f} s of that wall was hashlib inside the callbacks — the "
        f"readers ran through it, which is what the ring is for. Read-only throughput is "
        f"`bench`."
    )
    print(f"fill sha256:{digest.hexdigest()}")
    print(f"peak RSS:   {rss_mib():.1f} MiB (caller ring is {a.slots * a.slot_bytes} B)")
    if a.expect:
        got = digest.hexdigest()
        want = a.expect.removeprefix("sha256:")
        print(f"BANKED:     {'MATCH' if got == want else 'MISMATCH'} vs the Rust path's {want[:12]}…")
    lease.release()
    print(f"lease:      released explicitly (live={lease.live})")
    return 0


# --------------------------------------------------------------------------- bench


def cmd_bench(a: argparse.Namespace) -> int:
    store, header, header_bytes = load(a.store, a.manifest)
    order = construction_order(header)
    plan = tensorfs.plan(header_bytes, order, window=a.window)
    print(f"# {plan.io_bytes} B disk leg, window {a.window} B, slot {a.slot_bytes} B, {a.slots} slots")
    print(f"{'arm':<10} {'depth':<6} {'cold GB/s':<10} {'warm GB/s':<10} {'peak RSS':<10}")

    for depth in a.depths:
        for arm in a.arms:
            row = []
            for phase in ("cold", "warm"):
                if phase == "cold":
                    evict(a.store)
                t0 = time.perf_counter()
                if arm == "facade":
                    lease = store.acquire_manifest(a.manifest)
                    acq = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    stream_facade(lease, plan, a.slot_bytes, a.slots, depth, None)
                    d = time.perf_counter() - t0
                    lease.release()
                elif arm == "rust":
                    # The SAME mechanism (read::pump) driven by the Rust CLI: the delta
                    # against `facade` is the marshal boundary and nothing else.
                    acq = 0.0
                    out = subprocess.run(
                        [str(TFS), "read", "stream", a.store, a.manifest,
                         "--workers", str(depth), "--slots", str(a.slots),
                         "--slot-bytes", str(a.slot_bytes), "--window", str(a.window),
                         "--no-digest"],
                        capture_output=True, text=True, check=True,
                    ).stdout
                    ms = [l for l in out.splitlines() if l.startswith("filled:")][0]
                    d = int(ms.split(" in ")[1].split(" ")[0]) / 1000
                else:
                    cas = open_interim(a.store)
                    acq = 0.0
                    t0 = time.perf_counter()
                    stream_interim(cas, plan, a.slot_bytes, a.slots, depth, None)
                    d = time.perf_counter() - t0
                    cas.close()
                row.append(plan.io_bytes / 1e9 / d)
                if phase == "cold" and arm == "facade":
                    row.append(acq)
            rss = rss_mib()
            extra = f"  (lease {row[1]*1000:.0f} ms)" if arm == "facade" else ""
            print(f"{arm:<10} qd{depth:<5} {row[0]:<10.3f} {row[-1]:<10.3f} {rss:<10.1f}{extra}")
    return 0


# --------------------------------------------------------------------------- arms


class Arms:
    def __init__(self) -> None:
        self.fired = 0
        self.missed = 0

    def arm(self, name: str, want: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except errors.Refusal as e:
            if e.code == want and isinstance(e, errors.CODES[want]):
                self.fired += 1
                print(f"  RED   {name:<44} {e.code:<24} {e.detail[:78]}")
            else:
                self.missed += 1
                print(f"  WRONG {name:<44} wanted {want}, got {e.code}")
            return
        except Exception as e:  # noqa: BLE001
            self.missed += 1
            print(f"  UNTYPED {name:<42} {type(e).__name__}: {e}")
            return
        self.missed += 1
        print(f"  MISS  {name:<44} wanted {want}, nothing refused")


def cmd_arms(a: argparse.Namespace) -> int:
    work = pathlib.Path(a.work)
    arms = Arms()
    store, header, header_bytes = load(a.store, a.manifest)
    order = construction_order(header)
    plan = tensorfs.plan(header_bytes, order, window=a.window)
    lease = store.acquire_manifest(a.manifest)
    first = lease.objects[0]

    # ---- the in-place claim, proved rather than asserted -------------------------
    # TensorFS never hands Python a mapping it owns: it fills memory the CALLER already
    # holds. So the proof is not "a view stayed valid under mutation" — it is that the
    # bytes land inside the caller's own object, at the caller's own offset, with the
    # bytes on either side untouched and no allocation to show for it.
    obj = bytearray(b"\xAA") * (1 << 20)
    pad = 4096
    n = min(first[1], (1 << 20) - 2 * pad)
    lease.read_into(first[0], first[1], 0, n, memoryview(obj)[pad : pad + n])
    before = obj[:pad] == b"\xAA" * pad
    after = obj[pad + n :] == b"\xAA" * ((1 << 20) - pad - n)
    landed = bytes(obj[pad : pad + n]) == store.document(first[0])[:n]
    print(
        f"  PROOF in-place fill: wrote {n} B into the caller's bytearray at offset {pad}; "
        f"leading pad intact={before} trailing pad intact={after} bytes correct={landed}"
    )
    rss0 = rss_mib()
    stream_facade(lease, plan, a.slot_bytes, a.slots, a.readers, None)
    print(
        f"  PROOF bounded ring: streamed {plan.bytes} B through a "
        f"{a.slots * a.slot_bytes} B ring; peak RSS {rss0:.0f} -> {rss_mib():.0f} MiB"
    )

    print("typed refusals across the FFI boundary")
    arms.arm(
        "a caller buffer that is not the range",
        "BUFFER_SIZE",
        lambda: lease.read_into(first[0], first[1], 0, first[1], bytearray(first[1] - 1)),
    )
    arms.arm(
        "a read-only destination (`bytes`)",
        "BUFFER_SIZE",
        lambda: lease.read_into(first[0], first[1], 0, first[1], b"\0" * first[1]),
    )
    arms.arm(
        "a range past the object's declared end",
        "RANGE_BOUNDS",
        lambda: lease.read_into(first[0], first[1], first[1] - 4, 64, bytearray(64)),
    )
    arms.arm(
        "an object outside the lease",
        "LEASE_NOT_COVERED",
        lambda: lease.read_into("00" * 32, 64, 0, 64, bytearray(64)),
    )
    arms.arm(
        "the same digest at a lying length",
        "LEASE_NOT_COVERED",
        lambda: lease.read_into(first[0], first[1] + 1, 0, 64, bytearray(64)),
    )
    arms.arm(
        "a malformed digest",
        "MALFORMED_DIGEST",
        lambda: lease.read_into("not-a-digest", 64, 0, 64, bytearray(64)),
    )
    arms.arm(
        "a traversal that misses a tensor",
        "TRAVERSAL_INCOMPLETE",
        lambda: tensorfs.plan(header_bytes, order[:-1], window=a.window),
    )
    arms.arm(
        "a slot smaller than one plan item",
        "BUFFER_SIZE",
        lambda: stream_facade(lease, plan, 1024, 2, 2, None),
    )
    arms.arm(
        "slots of different lengths",
        "BUFFER_SIZE",
        lambda: lease.stream(
            plan, [memoryview(bytearray(1 << 20)), memoryview(bytearray(1 << 21))],
            lambda b: b.release(), readers=2,
        ),
    )

    def consume_without_releasing() -> None:
        slots = [memoryview(bytearray(a.slot_bytes)) for _ in range(2)]
        lease.stream(plan, slots, lambda b: None, readers=4)

    arms.arm(
        "every slot held, none released",
        "SLOT_STARVED",
        consume_without_releasing,
    )

    # A corrupt object, planted for real in a WRITABLE copy of the store.
    corrupt = work / "corrupt"
    shutil.rmtree(corrupt, ignore_errors=True)
    shutil.copytree(a.store, corrupt)
    victim = tensorfs.Store.open(str(corrupt))
    oid = lease.objects[1][0].removeprefix("sha256:")
    path = pathlib.Path(str(corrupt)) / "blobs" / oid[:2] / oid[2:4] / oid
    path.chmod(0o644)
    with open(path, "r+b") as f:
        f.seek(0)
        b = f.read(1)
        f.seek(0)
        f.write(bytes([b[0] ^ 0xFF]))
    arms.arm(
        "an object whose bytes disagree with its id",
        "OBJECT_CORRUPT",
        lambda: victim.acquire_manifest(a.manifest),
    )

    # Truncation UNDER a live lease: admission passed, the bytes left afterwards.
    torn = work / "torn"
    shutil.rmtree(torn, ignore_errors=True)
    shutil.copytree(a.store, torn)
    tstore = tensorfs.Store.open(str(torn))
    tlease = tstore.acquire_manifest(a.manifest)
    toid = tlease.objects[1][0].removeprefix("sha256:")
    tlen = tlease.objects[1][1]
    tpath = pathlib.Path(str(torn)) / "blobs" / toid[:2] / toid[2:4] / toid
    tpath.chmod(0o644)
    os.truncate(tpath, max(tlen - 512, 0))
    arms.arm(
        "an object truncated under a live lease",
        "SHORT_READ",
        lambda: tlease.read_into(toid, tlen, 0, min(tlen, 4096), bytearray(min(tlen, 4096))),
    )
    tlease.release()

    # The hold, stolen from the authority behind the reader's back.
    stolen = tensorfs.Store.open(a.store)
    slease = stolen.acquire_manifest(a.manifest)
    subprocess.run([str(TFS), "hold", "steal", a.store, slease.hold_id], check=True,
                   capture_output=True)
    arms.arm("a lease the authority no longer knows", "LEASE_REVOKED", slease.recheck)

    # Release is explicit, and a released lease is dead.
    lease.release()
    arms.arm(
        "read after release",
        "LEASE_REVOKED",
        lambda: lease.read_into(first[0], first[1], 0, 64, bytearray(64)),
    )
    arms.arm("release twice", "LEASE_REVOKED", lease.release)

    # Documents: the codec's own refusals, through the same typed door.
    arms.arm(
        "a semantic twin of the header (one space)",
        "NONCANONICAL_ENCODING",
        lambda: tensorfs.parse_header(header_bytes.replace(b'{"', b'{ "', 1)),
    )
    arms.arm(
        "a header with a flipped byte",
        "MALFORMED_JSON",
        lambda: tensorfs.parse_header(header_bytes[:-2] + b" }"),
    )
    arms.arm(
        "a manifest read as a header",
        "UNKNOWN_FORMAT",
        lambda: tensorfs.parse_header(store.manifest(a.manifest)["manifest"]),
    )
    arms.arm(
        "an unknown document kind",
        "UNKNOWN_FORMAT",
        lambda: tensorfs.run_doc("safetensors", header_bytes),
    )
    arms.arm(
        "a platform that has no lock semantics",
        "PLATFORM_UNSUPPORTED",
        lambda: tensorfs.platform_gate("win32"),
    )

    print(f"\n{arms.fired} fired, {arms.missed} missed")
    return 1 if arms.missed else 0


# --------------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("real", "bench", "arms"):
        s = sub.add_parser(name)
        s.add_argument("store")
        s.add_argument("manifest")
        s.add_argument("--window", type=int, default=4 << 20)
        s.add_argument("--slot-bytes", type=int, default=16 << 20)
        s.add_argument("--slots", type=int, default=4)
        s.add_argument("--readers", type=int, default=8)
        if name == "real":
            s.add_argument("--expect", default="")
        if name == "bench":
            s.add_argument("--depths", type=int, nargs="+", default=[1, 4, 8, 16])
            s.add_argument("--arms", nargs="+", default=["facade", "interim"])
        if name == "arms":
            s.add_argument("--work", default="/home/fidika/cozy_v2/tfs-007-work")
    a = p.parse_args()
    return {"real": cmd_real, "bench": cmd_bench, "arms": cmd_arms}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
