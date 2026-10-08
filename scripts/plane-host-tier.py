#!/usr/bin/env python3
"""The weight plane's pinned host tier through `tensorfs.plane`, on a real store, CPU only.

Register and layout alignment, a pinned fill checked byte for byte against the source,
eviction order and memfd RAM release under a lowered budget, Shortfall, pinning, drop,
adoption of the memfd by a second PROCESS with zero new fill bytes, claims (a closed set's
RAM is punched unless a Worker hold keeps it; releasing the hold punches it), and one TensorFS lease
per manifest that ends only when its last weight set closes and its Source is dropped. Every
Plane is `devices=[]`: libcuda is never loaded.

    PYTHONPATH=python scripts/plane-host-tier.py [workdir]
"""

import gc
import hashlib
import io
import json
import math
import mmap
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import tensorfs
from tensorfs import plane

MIB = 1 << 20
REGION_ALIGN, PART_ALIGN = 2 * MIB, 4096
# component -> key -> f32 shape. <= 256 B is inline; > 64 MiB spans two CAS objects.
MODEL = {
    "unet": {
        "norm": [64],
        "scale": [],
        "embed": [MIB // 4],
        "block0.q": [3 * MIB // 4],
        "block0.bias": [48],
        "block0.k": [700 * 1024 // 4],
        "block1.w": [65 * MIB // 4],
        "block1.b": [64 * 1024],
    },
    "vae": {"dec": [3 * MIB // 8], "gain": [8]},
}
# unet names tensors, vae names exact parts.
REGIONS = {
    "unet": [["unet/norm", "unet/scale", "unet/embed"],
             ["unet/block0.q", "unet/block0.bias", "unet/block0.k"],
             ["unet/block1.w", "unet/block1.b"]],
    "vae": [["vae/dec#value", "vae/gain#value"]],
}
UNET_INLINE = 256 + 4 + 192
HOST_KEYS = {"budget", "used", "ready_bytes", "registered_bytes", "memfd_bytes", "sealed_bytes", "cached_bytes",
             "direct_bytes", "buffered_bytes", "inline_bytes", "direct_refused", "map_refused",
             "disk_read_bytes", "fills",
             "fill_bytes", "fill_ns", "evictions", "evicted_bytes", "register_failed", "trims"}
STATS_KEYS = {"host", "devices", "sets", "stages", "regions", "events_dropped", "poisoned"}
SET_KEYS = {"id", "name", "nbytes", "regions", "host_ready_bytes", "host_memfd_bytes",
            "evictions", "evicted_bytes", "devices"}
EVENT_KEYS = {"kind", "ws", "region", "device", "bytes", "ns", "detail"}
TRIM_KEYS = {"freed", "evicted", "over_budget_unreleasable", "blockers"}

results: list[bool] = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


def raises(name, exc, call):
    try:
        call()
    except exc as e:
        check(name, True)
        return e
    except Exception as e:  # noqa: BLE001 - reported as the failure it is
        check(name, False, f"raised {type(e).__name__}: {e}")
        return None
    check(name, False, f"did not raise {exc.__name__}")
    return None


def payloads():
    return {f"{c}/{k}#value": random.Random(f"{c}/{k}").randbytes(4 * math.prod(s))
            for c, keys in MODEL.items() for k, s in keys.items()}


def build(root: Path, data):
    store = tensorfs.Store.init(root)
    plain = next(v for n, v in tensorfs.seed_digests() if n == "plain/1")
    targets = {c: {"drop": [], "add": {k: {"logical_dtype": "f32", "shape": s, "encoding": plain,
                                           "parts": {"value": {"dtype": "f32", "shape": s}}}
                                       for k, s in keys.items()}}
               for c, keys in MODEL.items()}
    order = [(c, k) for c, keys in MODEL.items() for k in keys]
    total = sum(len(b) for b in data.values())
    writer = store.begin_derived("sha256:" + "a1" * 32, 1, {}, targets, {}, order, 2 * total + MIB,
                                 work_fingerprint="sha256:" + "5a" * 32)
    for c, k in order:
        writer.add_part(c, k, "value", io.BytesIO(data[f"{c}/{k}#value"]))
    return store, writer.commit()["manifest"]["sha256"]


def plan(store, manifest, component):
    header = store.manifest(manifest)["header"]
    return tensorfs.plan(header, [(component, k) for k in MODEL[component]], [component])


def register(p, src, store, manifest, component, host_fd=None):
    return p.register(f"fixture/{component}", src, plan(store, manifest, component),
                      REGIONS[component], host_fd=host_fd)


def part_hashes(ws):
    with mmap.mmap(ws.host_fd, ws.nbytes, mmap.MAP_SHARED, mmap.PROT_READ) as mm:
        return {q["what"]: hashlib.sha256(mm[q["offset"]:q["offset"] + q["nbytes"]]).hexdigest()
                for q in ws.parts}


def held(root: Path, hold_id):
    return any(hold_id in m.name for m in (root / "tmp/leases").glob("*.lease"))


def evict_page_cache(root: Path):
    """Drop the store's clean pages so the fill must take the O_DIRECT path."""
    for f in root.rglob("*"):
        if f.is_file():
            fd = os.open(f, os.O_RDONLY)
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)


def set_of(p, ws):
    return next(s for s in p.stats()["sets"] if s["name"] == ws.name)


def child(root, manifest, fd):
    """The second process: adopt the parent's memfd for the same layout."""
    store = tensorfs.Store.open(root)
    p = plane.Plane(devices=[])
    lease = store.acquire_cozytensors(manifest)
    src = p.source(store, lease)
    ws = register(p, src, store, manifest, "unet", host_fd=int(fd))
    p.set_pinned_budget(ws.nbytes)
    p.want(ws, "pinned").wait()
    st = p.stats()
    out = {"fills": st["host"]["fills"], "fill_bytes": st["host"]["fill_bytes"],
           "details": sorted({e["detail"] for e in p.events() if e["kind"] == "fill_done"}),
           "hashes": part_hashes(ws), "lease_live": lease.live}
    ws.close()
    del src
    p.close()
    print(json.dumps(out))


def main(work: Path):
    data = payloads()
    sha = {w: hashlib.sha256(b).hexdigest() for w, b in data.items()}
    root = work / "store"
    store, manifest = build(root, data)

    p = plane.Plane(devices=[])
    check("host-only plane serves no device", p.devices == [])
    lease = store.acquire_cozytensors(manifest)
    hold = lease.hold_id
    src = p.source(store, lease)
    check("source() consumes the ReadLease", lease.live is False)
    check("the source holds the store's lease file", held(root, hold))
    ws = register(p, src, store, manifest, "unet")

    # ---- layout
    regions, parts = ws.regions, ws.parts
    check("regions start on 2 MiB and span whole 2 MiB units",
          all(r["offset"] % REGION_ALIGN == 0 and r["span"] % REGION_ALIGN == 0
              and r["nbytes"] <= r["span"] < r["nbytes"] + REGION_ALIGN for r in regions), regions)
    check("regions tile the layout in index order",
          [r["index"] for r in regions] == [0, 1, 2]
          and all(a["offset"] + a["span"] == b["offset"] for a, b in zip(regions, regions[1:]))
          and ws.nbytes == regions[-1]["offset"] + regions[-1]["span"] and ws.nbytes % REGION_ALIGN == 0)
    check("parts start on 4 KiB, inside their region",
          all(q["offset"] % PART_ALIGN == 0
              and regions[q["region"]]["offset"] <= q["offset"]
              and q["offset"] + q["nbytes"] <= regions[q["region"]]["offset"] + regions[q["region"]]["nbytes"]
              for q in parts), parts)
    by_region = [[q["what"] for q in parts if q["region"] == r] for r in range(3)]
    check("each region holds its declared tensors in plan order",
          by_region == [[f"{t}#value" for t in rs] for rs in REGIONS["unet"]], by_region)
    check("part sizes are the source sizes", all(q["nbytes"] == len(data[q["what"]]) for q in parts))
    spans = [r["span"] for r in regions]

    # ---- pinned fill
    trim = p.set_pinned_budget(ws.nbytes)
    check("trim rows carry the documented keys", TRIM_KEYS <= trim.keys() and trim["freed"] == 0, trim)
    evict_page_cache(root)
    t0 = p.want(ws, "pinned", regions=[0], priority=1)
    t1 = p.want(ws, "pinned", regions=[1, 2], priority=0)
    t0.wait()
    t1.wait()
    check("tickets report done after wait", t0.done() and t1.done())
    st = p.stats()
    host = st["host"]
    unet_bytes = sum(len(data[q["what"]]) for q in parts)
    # Require the fields the consumer uses, while allowing newer producers to add observations.
    check("stats rows carry the documented keys",
          STATS_KEYS <= st.keys() and HOST_KEYS <= host.keys() and SET_KEYS <= st["sets"][0].keys()
          and st["devices"] == [] and st["stages"] == [] and st["regions"] == []
          and st["poisoned"] is None, st)
    check("pinned tier holds every region",
          host["used"] == ws.nbytes and host["ready_bytes"] == sum(r["nbytes"] for r in regions)
          and host["fills"] == 3 and host["fill_bytes"] == sum(r["nbytes"] for r in regions), host)
    check("every source byte is counted by exactly one read mode",
          host["cached_bytes"] + host["direct_bytes"] + host["buffered_bytes"] + host["inline_bytes"] == unet_bytes
          and host["inline_bytes"] == UNET_INLINE, host)
    check("bytes the page cache lacks are read by O_DIRECT (where the filesystem allows it)",
          host["direct_bytes"] > 0 or host["direct_refused"] > 0, host)
    print(f"  fill: {host['fill_bytes']} B in {host['fill_ns'] / 1e6:.1f} ms; cached {host['cached_bytes']}, "
          f"direct {host['direct_bytes']}, buffered {host['buffered_bytes']}, inline {host['inline_bytes']}, "
          f"direct_refused {host['direct_refused']}")
    events = p.events()
    check("fills report fill_done from disk",
          all(set(e) == EVENT_KEYS for e in events)
          and sorted(e["region"] for e in events if e["kind"] == "fill_done") == [0, 1, 2]
          and {e["detail"] for e in events if e["kind"] == "fill_done"} == {"pinned<-disk"}, events)
    check("memfd bytes equal the source at every reported offset",
          part_hashes(ws) == {q["what"]: sha[q["what"]] for q in parts})
    memfd = set_of(p, ws)["host_memfd_bytes"]
    check("memfd RAM covers the filled bytes", memfd >= unet_bytes, memfd)

    # ---- adoption by a second process
    fill_bytes = host["fill_bytes"]
    run = subprocess.run([sys.executable, __file__, "--child", str(root), manifest, str(ws.host_fd)],
                         pass_fds=[ws.host_fd], capture_output=True, text=True)
    got = json.loads(run.stdout) if run.returncode == 0 else {}
    check("a second process adopts the memfd", run.returncode == 0, run.stderr[-2000:])
    if got:
        check("adoption fills nothing", got["fills"] == 0 and got["fill_bytes"] == 0
              and got["details"] == ["pinned<-adopted"], got)
        check("the second process sees the same bytes", got["hashes"] == {q["what"]: sha[q["what"]] for q in parts})
        check("the child's source consumed its lease", got["lease_live"] is False)
    after = p.stats()["host"]
    check("adoption leaves this process's fills unchanged",
          after["fill_bytes"] == fill_bytes and set_of(p, ws)["host_memfd_bytes"] == memfd, after)

    # ---- a second weight set from the same source, regions named by exact parts
    vae = register(p, src, store, manifest, "vae")
    vspan = vae.regions[0]["span"]
    check("exact part names group the regions",
          [q["what"] for q in vae.parts] == REGIONS["vae"][0] and [q["region"] for q in vae.parts] == [0, 0])
    p.set_pinned_budget(ws.nbytes + vspan)
    p.want(vae, "pinned", priority=0).wait()
    check("both weight sets of one source fill",
          part_hashes(vae) == {q["what"]: sha[q["what"]] for q in vae.parts}
          and p.stats()["host"]["used"] == ws.nbytes + vspan)
    p.events()

    # ---- eviction order: ascending priority, later weight set first, higher region first
    order = []
    for _ in range(3):
        order += p.set_pinned_budget(p.stats()["host"]["used"] - 1)["evicted"]
    check("lowering the budget evicts in the documented order",
          order == [("fixture/vae", 0), ("fixture/unet", 2), ("fixture/unet", 1)], order)
    unet = set_of(p, ws)
    check("evicted regions leave the books",
          p.stats()["host"]["used"] == spans[0] and unet["host_ready_bytes"] == regions[0]["nbytes"]
          and unet["evictions"] == 2 and unet["evicted_bytes"] == spans[1] + spans[2], unet)
    freed_ram = memfd - unet["host_memfd_bytes"]
    check("eviction frees the memfd's RAM",
          freed_ram >= sum(len(data[q["what"]]) for q in parts if q["region"] != 0)
          and set_of(p, vae)["host_memfd_bytes"] < REGION_ALIGN, freed_ram)
    check("eviction events name each region",
          [(e["ws"], e["region"]) for e in p.events() if e["kind"] == "evicted"] == order)

    # ---- want evicts only strictly lower priority; pins hold; drop
    p.set_pinned_budget(spans[1])
    e = raises("an equal-priority want raises Shortfall", plane.Shortfall,
               lambda: p.want(ws, "pinned", regions=[1], priority=1))
    if e is not None:
        check("Shortfall carries code, need, available, blockers",
              isinstance(e, plane.PlaneError) and e.code == "SHORTFALL" and e.need == spans[1]
              and e.available == spans[1] - spans[0] and e.blockers == [], vars(e))
    p.want(ws, "pinned", regions=[1], priority=2).wait()
    check("a higher-priority want evicts the lower region",
          set_of(p, ws)["host_ready_bytes"] == regions[1]["nbytes"])
    fills = p.stats()["host"]["fills"]
    p.prioritise(ws, "pinned", [1], priority=2, pin=True)
    trim = p.set_pinned_budget(0)
    check("prioritise pins without moving a byte; a pin is never evicted implicitly",
          trim["evicted"] == [] and trim["over_budget_unreleasable"] == spans[1]
          and p.stats()["host"]["fills"] == fills, trim)
    raises("a want above the whole pinned budget raises BelowFloor", plane.BelowFloor,
           lambda: p.want(ws, "pinned", regions=[0], priority=100))
    check("drop frees a pinned region", p.drop(ws, "pinned", [1]) == spans[1]
          and p.stats()["host"]["used"] == 0 and set_of(p, ws)["host_memfd_bytes"] < REGION_ALIGN)
    check("capacity refusals never poison", p.stats()["poisoned"] is None)

    # ---- refusals that must never reach a device
    raises("an unknown tier is Invalid", plane.Invalid, lambda: p.want(ws, "gpu"))
    raises("an unserved device is Invalid", plane.Invalid, lambda: p.want(ws, 0))
    raises("a view on an unserved device is Invalid", plane.Invalid, lambda: ws.view(0))
    other = plane.Plane(devices=[])
    raises("a weight set is bound to its plane", plane.Invalid, lambda: other.want(ws, "pinned"))
    other.close()
    raises("a consumed lease cannot make a second source", tensorfs.errors.LeaseRevoked,
           lambda: p.source(store, lease))
    raises("adopting a memfd of another layout is Invalid", plane.Invalid,
           lambda: register(p, src, store, manifest, "vae", host_fd=ws.host_fd))
    raises("adopting a descriptor that is not open is IoError", plane.IoError,
           lambda: register(p, src, store, manifest, "vae", host_fd=1 << 20))

    # ---- claims: RAM a closed set leaves goes unless someone still claims it
    def resident(fd):
        return os.fstat(fd).st_blocks * 512

    tier = register(p, src, store, manifest, "unet")
    p.set_pinned_budget(tier.nbytes)
    p.want(tier, "pinned").wait()
    kept = os.dup(tier.host_fd)  # what the Worker's HostTiers keeps
    full = resident(kept)
    hold_fd = plane.hold_tier(kept)
    tier.close()
    check("a Worker hold keeps a closed set's pinned RAM", resident(kept) == full, resident(kept))
    check("releasing the hold punches what no plane claims",
          plane.release_tier(hold_fd) == sum(spans) and resident(kept) < REGION_ALIGN, resident(kept))
    os.close(kept)
    tier = register(p, src, store, manifest, "unet")
    p.want(tier, "pinned").wait()
    kept = os.dup(tier.host_fd)
    tier.close()
    check("closing a set punches its pinned RAM while a handle to the memfd survives",
          resident(kept) < REGION_ALIGN, resident(kept))
    os.close(kept)

    # ---- a plane collected unclosed leaves nothing behind
    def left():
        fds = [os.readlink(f"/proc/self/fd/{n}") for n in os.listdir("/proc/self/fd")
               if os.path.exists(f"/proc/self/fd/{n}")]
        names = [Path(f"/proc/self/task/{t}/comm").read_text() for t in os.listdir("/proc/self/task")]
        return (sum("memfd:tfs-plane" in f for f in fds), sum(n.startswith("tfs-plane") for n in names))

    def unclosed():
        q = plane.Plane(devices=[])
        qsrc = q.source(store, store.acquire_cozytensors(manifest))
        qws = register(q, qsrc, store, manifest, "unet")
        q.set_pinned_budget(qws.nbytes)
        ticket = q.want(qws, "pinned")
        ticket.wait()
        del q, qws, qsrc  # the ticket is the plane's last handle
        return ticket

    before = left()
    for _ in range(3):
        unclosed()
    gc.collect()
    for _ in range(2000):
        if left() == before:
            break
        time.sleep(0.001)
    check("planes collected unclosed leave no memfds or threads", left() == before, (before, left()))

    # ---- one lease per manifest: it ends with the last weight set and the source handle
    ws.close()
    check("closing one weight set keeps the source's lease", held(root, hold))
    e = raises("a closed weight set is Closed", plane.Closed, lambda: ws.close())
    check("Closed carries its code", e is not None and e.code == "CLOSED")
    del src
    check("dropping the source handle keeps the lease while a weight set uses it", held(root, hold))
    p.close()
    check("plane.close() closes the last weight set and ends the lease",
          p.stats()["sets"] == [] and not held(root, hold))
    check("libcuda was never loaded", "libcuda" not in Path("/proc/self/maps").read_text())

    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(*sys.argv[2:5])
        sys.exit(0)
    parent = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent, prefix="plane-host-tier-") as d:
        sys.exit(main(Path(d)))
