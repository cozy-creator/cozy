#!/usr/bin/env python3
"""The weight plane under a real consumer, on one GPU: torch kernels read plane bytes through
DLPack views on the consumer's own stream.

A 1 + L region model (common + L linear layers, bf16) lives in a real store. The pinned tier
holds it; a spread resident set is wanted on the device; the rest streams through the ring.
Every step's output must equal the all-resident reference bit for bit. Prints step times
(streamed vs resident), stalls and per-region GPU compute from `stats()`.

    flock <locks>/gpu.lock python scripts/plane-gpu-torch.py [workdir] [--layers N] [--dim D]
        [--batch B] [--resident R] [--window W] [--steps S]
Runs with any python that has torch (CUDA) and `PYTHONPATH=<worktree>/python`.
"""

import argparse
import io
import random
import sys
import tempfile
import time
from pathlib import Path

import torch

import tensorfs
from tensorfs import plane

MIB = 1 << 20
results: list[bool] = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


def build(root: Path, layers: int, dim: int):
    """Store + manifest holding `embed` and `layerN.{w,b}` in bf16; returns the torch tensors."""
    g = torch.Generator().manual_seed(0)
    tensors = {"embed": torch.randn(64, dim, generator=g).bfloat16()}
    for i in range(layers):
        tensors[f"layer{i}.w"] = (torch.randn(dim, dim, generator=g) / dim**0.5).bfloat16()
        tensors[f"layer{i}.b"] = (torch.randn(dim, generator=g) / 10).bfloat16()
    store = tensorfs.Store.init(root)
    plain = next(v for n, v in tensorfs.seed_digests() if n == "plain/1")
    add = {k: {"logical_dtype": "bf16", "shape": list(t.shape), "encoding": plain,
               "parts": {"value": {"dtype": "bf16", "shape": list(t.shape)}}} for k, t in tensors.items()}
    order = [("model", k) for k in tensors]
    total = sum(t.numel() * 2 for t in tensors.values())
    w = store.begin_derived("sha256:" + "b2" * 32, 1, {}, {"model": {"drop": [], "add": add}}, {}, order,
                            2 * total + MIB, work_fingerprint="sha256:" + "6b" * 32)
    for _, k in order:
        w.add_part("model", k, "value", io.BytesIO(tensors[k].view(torch.uint8).numpy().tobytes()))
    return store, w.commit()["manifest"]["sha256"], tensors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir", nargs="?")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--resident", type=int, default=4)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--no-pinned", action="store_true", help="empty pinned tier: every streamed copy is disk-staged")
    a = ap.parse_args()
    root = Path(a.workdir or tempfile.mkdtemp(prefix="plane-gpu-")) / "store"
    store, manifest, tensors = build(root, a.layers, a.dim)
    header = store.manifest(manifest)["header"]
    rplan = tensorfs.plan(header, [("model", k) for k in tensors], ["model"])
    regions = [["model/embed"]] + [[f"model/layer{i}.w", f"model/layer{i}.b"] for i in range(a.layers)]

    dev = torch.device("cuda", 0)
    torch.cuda.init()
    s = torch.cuda.Stream(dev)
    p = plane.Plane(devices=[0], readers=4)
    src = p.source(store, store.acquire_cozytensors(manifest))
    ws = p.register("smoke/model", src, rplan, regions)
    del src
    reg = ws.regions
    parts = {x["what"]: x for x in ws.parts}
    total = sum(r["span"] for r in reg)
    if not a.no_pinned:
        p.set_pinned_budget(total)
        p.want(ws, "pinned").wait()
    span = reg[1]["span"]
    budget = reg[0]["span"] + a.resident * span + a.window * span
    p.set_vram_budget(0, budget)
    # A spread resident set (every k-th layer), the rest streamed: the placement the
    # proving model favours over a resident prefix.
    step_k = max(1, a.layers // max(a.resident, 1))
    resident = [1 + i for i in range(0, a.layers, step_k)][: a.resident]
    p.want(ws, 0, [0] + resident, priority=10).wait()
    check("pinned tier registered", p.stats()["host"]["registered_bytes"] == p.stats()["host"]["ready_bytes"])
    check("streamed copies come from the intended source",
          (p.stats()["host"]["ready_bytes"] == 0) == a.no_pinned)

    views: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    ring: list[torch.Tensor] = []

    def typed(u8, r):
        base = reg[r]["offset"]
        wp, bp = parts[f"model/layer{r - 1}.w#value"], parts[f"model/layer{r - 1}.b#value"]
        w = u8[wp["offset"] - base: wp["offset"] - base + wp["nbytes"]].view(torch.bfloat16).view(a.dim, a.dim)
        return w, u8[bp["offset"] - base: bp["offset"] - base + bp["nbytes"]].view(torch.bfloat16)

    def weights(lease, r):
        if lease.ring_offset is None:  # resident: its address never moves, build once
            if r not in views:
                views[r] = typed(torch.from_dlpack(lease.view()), r)
            return views[r]
        o = lease.ring_offset  # streamed: slice the one ring view
        return typed(ring[0][o: o + lease.nbytes], r)

    x0 = torch.randn(a.batch, a.dim, generator=torch.Generator().manual_seed(1)).bfloat16().to(dev)
    # Reference: every layer resident as ordinary tensors, same kernels, same stream.
    ref_w = [(tensors[f"layer{i}.w"].to(dev), tensors[f"layer{i}.b"].to(dev)) for i in range(a.layers)]
    with torch.cuda.stream(s):
        y = x0
        for w, b in ref_w:
            y = torch.nn.functional.linear(y, w, b)
        ref = y.clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            y = x0
            for w, b in ref_w:
                y = torch.nn.functional.linear(y, w, b)
        s.synchronize()
        t_ref = (time.perf_counter() - t0) / a.steps
    del ref_w

    order = list(range(1, 1 + a.layers))
    cur = p.stream(ws, 0, order, a.window, repeat=0, priority=10, tag="smoke")
    check("the resident set is served in place", cur.home == [1 + i in resident for i in range(a.layers)], str(cur.home))
    ring.append(torch.from_dlpack(cur.ring_view()))
    exact = True
    times = []
    with torch.cuda.stream(s):
        for step in range(a.steps + 1):
            t0 = time.perf_counter()
            y = x0
            for r in order:
                lease = cur.acquire(r, s.cuda_stream)
                w, b = weights(lease, r)
                y = torch.nn.functional.linear(y, w, b)
                lease.release()
            s.synchronize()
            if step:
                times.append(time.perf_counter() - t0)
            exact &= torch.equal(y, ref)
    check("every streamed step equals the resident reference bit for bit", exact)
    # Views keep the ring and the VA alive: drop every tensor over plane bytes before closing.
    del w, b
    ring.clear()
    views.clear()
    cur.close()
    st = p.stats()
    stage = next(x for x in st["stages"] if x["tag"] == "smoke")
    d0 = st["devices"][0]
    t_stream = sum(times) / len(times)
    gbps = (d0["host_copy_bytes"] + d0["disk_copy_bytes"]) / max(d0["host_copy_ns"] + d0["disk_copy_ns"], 1)
    print(f"step: resident {t_ref * 1e3:.2f} ms, streamed {t_stream * 1e3:.2f} ms "
          f"({t_stream / t_ref:.3f}x), {a.layers - len(resident)} of {a.layers} layers streamed, window {a.window}, "
          f"ring {cur.ring_nbytes >> 20} MiB")
    print(f"copies {d0['copies']} (pinned {d0['host_copy_bytes'] >> 20} MiB, disk-staged {d0['disk_copy_bytes'] >> 20} MiB), {gbps:.2f} GB/s; late {stage['late']} of {stage['acquires']}, "
          f"stall {stage['stall_ns'] / 1e6:.2f} ms, skipped {stage['skipped']}")
    comp = sorted((x["region"], x["compute_ns"] / max(x["uses"], 1) / 1e6) for x in st["regions"] if x["tag"] == "smoke")
    print("per-region GPU compute ms:", " ".join(f"{r}:{ms:.2f}" for r, ms in comp))
    check("per-region compute measured", len(comp) == a.layers and all(ms > 0 for _, ms in comp))
    check("not poisoned", st["poisoned"] is None)
    ws.close()
    p.close()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
