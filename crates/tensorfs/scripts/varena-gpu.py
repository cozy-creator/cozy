#!/usr/bin/env python3
"""varena's GPU core on a real card, as a standalone consumer: torch and the tensorfs wheel only.

    python varena-gpu.py [workdir]

A layer-structured v2 checkpoint (written here by the v2 writer) is registered from its chunk
files and from a sealed host layout. The script checks, and prints one JSON line of facts:
  a. map, fill and bind: every part is bit-exact with the checkpoint, at home and after refault;
  b. a view a caller keeps blocks its page's eviction; once it dies, eviction frees the bytes;
  c. layers streamed through slots with prefetch: bit-exact with all-home; stall < copy time;
  d. a torch OOM beside home pages: evict + trim, then the retry succeeds in-process;
  e. bind/unbind per region of ~20 params (M2), and page-in at 2 MiB vs 64 MiB chunks;
  f. adopt: the sealed layout registered from the regions it carries, prefetched whole before
     any module exists, is bit-exact.
"""

import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch

import tensorfs
from tensorfs import varena

LAYERS, DIM, TOKENS = 12, 4096, 8192
#: blocks of 20 one-MiB params: each block fills its own pack, so it is a region of its own
BLOCKS, PARAMS = 4, 20
MiB = 1 << 20


def write(root: Path) -> tensorfs.Checkpoint:
    store = tensorfs.Store(root)
    w = store.writer()
    g = torch.Generator().manual_seed(7)
    for i in range(LAYERS):
        w.put("model", f"layers.{i}.weight", "F16", [DIM, DIM], (torch.randn(DIM, DIM, generator=g) / DIM**0.5).half().numpy())
    for b in range(BLOCKS):
        for p in range(PARAMS):
            w.put("blocks", f"{b}.p{p}", "BF16", [512, 1024], torch.randn(512, 1024, generator=g).bfloat16().view(torch.int16).numpy())
    return store.checkpoint(w.commit())


def regions(ck: tensorfs.Checkpoint) -> list[list[str]]:
    cuts = [f"model/layers.{i}" for i in range(LAYERS)] + [f"blocks/{b}" for b in range(BLOCKS)]
    return tensorfs.regions(ck, 64 * MiB, cuts)


def views(pin: varena.Pinned, parts: list[varena.Part]) -> dict[str, torch.Tensor]:
    t = torch.from_dlpack(pin)
    dt = {"F16": torch.float16, "BF16": torch.bfloat16}
    return {p.key: t[p.offset : p.offset + p.nbytes].view(dt[p.dtype]).view(p.shape) for p in parts}


def by_region(ws: varena.WeightSet) -> dict[int, list[varena.Part]]:
    out: dict[int, list[varena.Part]] = {}
    for p in ws.parts:
        out.setdefault(p.region, []).append(p)
    return out


def exact(ws: varena.WeightSet, ck: tensorfs.Checkpoint, stream: int) -> int:
    """Every part of every region, bound, equals the checkpoint's bytes."""
    n = 0
    for r, parts in by_region(ws).items():
        pin = ws.bind(r, stream)
        t = torch.from_dlpack(pin)
        for p in parts:
            want = torch.frombuffer(bytearray(ck.read(p.component, p.key).tobytes()), dtype=torch.uint8)
            assert torch.equal(t[p.offset : p.offset + p.nbytes].cpu(), want), f"{p.component}/{p.key} differs"
            n += p.nbytes
    return n


def layer_regions(ws: varena.WeightSet) -> list[tuple[int, list[varena.Part]]]:
    groups = by_region(ws)
    order = sorted((r for r, ps in groups.items() if ps[0].component == "model"), key=lambda r: int(groups[r][0].key.split(".")[1]))
    return [(r, groups[r]) for r in order]


def forward(ws: varena.WeightSet, x: torch.Tensor, stream: int, ahead: int) -> torch.Tensor:
    layers = layer_regions(ws)
    for r, _ in layers[:ahead]:
        ws.prefetch(r)
    for i, (r, parts) in enumerate(layers):
        (w,) = views(ws.bind(r, stream), parts).values()
        if i + ahead < len(layers):
            ws.prefetch(layers[i + ahead][0])
        for _ in range(2):
            x = torch.relu(torch.nn.functional.linear(x, w))
            x = x / (x.float().pow(2).mean().sqrt().half() + 1)
        del w
    return x


def counters(gpu: varena.Gpu, name: str) -> varena.Counters:
    return next(s.total for s in gpu.stats().sets if s.name == name)


def test_exact(gpu, ck, rs, host_fd, stream, facts):
    for name, fd in (("files", None), ("sealed", host_fd)):
        ws = gpu.register(f"exact-{name}", ck, rs, fd)
        n = exact(ws, ck, stream)
        torch.cuda.synchronize()
        assert ws.evict() == sum(ws.spans), "every unpinned region is evicted"
        assert exact(ws, ck, stream) == n, "refault"
        facts[f"a_{name}_bytes"] = n
        ws.close()
    gpu.trim()


def test_kept_view(gpu, ck, rs, stream, facts):
    ws = gpu.register("kept", ck, rs)
    r = layer_regions(ws)[0][0]
    keep = torch.from_dlpack(ws.bind(r, stream))[:4096]
    torch.cuda.synchronize()
    s = next(x for x in gpu.stats().sets if x.name == "kept")
    assert ws.evict([r]) == 0 and s.pins == 1 and s.pinned == ws.spans[r], "a kept view pins its page"
    gpu.trim()
    free = gpu.stats().driver_free
    del keep
    torch.cuda.synchronize()
    freed = ws.evict([r])
    trimmed = gpu.trim()
    after = gpu.stats()
    gained = after.driver_free - free
    assert freed == ws.spans[r] and next(x for x in after.sets if x.name == "kept").resident == 0
    assert gained >= ws.spans[r] - 2 * MiB, f"driver free grew {gained} for {ws.spans[r]} evicted"
    facts.update(b_evicted=freed, b_trimmed=trimmed, b_driver_free_gained=gained)
    ws.close()


def test_stream(gpu, ck, rs, host_fd, stream, facts):
    x0 = torch.randn(TOKENS, DIM, generator=torch.Generator().manual_seed(11)).half().cuda()
    for name, fd in (("files", None), ("sealed", host_fd)):
        ws = gpu.register(f"stream-{name}", ck, rs, fd)
        home = forward(ws, x0, stream, 2)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        home = forward(ws, x0, stream, 2)
        torch.cuda.synchronize()
        home_s = time.perf_counter() - t0
        ws.home([])
        ws.evict()
        gpu.trim()
        before = counters(gpu, ws.name)
        t0 = time.perf_counter()
        out = forward(ws, x0, stream, 2)
        torch.cuda.synchronize()
        streamed_s = time.perf_counter() - t0
        c = counters(gpu, ws.name)
        copy, stall = c.copy_ns - before.copy_ns, c.stall_ns - before.stall_ns
        slot = c.streamed_bytes - before.streamed_bytes
        assert torch.equal(out, home), "streamed output differs from all-home"
        assert slot >= sum(ws.spans[r] for r, _ in layer_regions(ws)) // 2, "the layers streamed"
        assert stall < copy, f"stall {stall} ns >= copy {copy} ns: copies did not overlap compute"
        facts[f"c_{name}"] = dict(home_ms=round(home_s * 1e3, 1), streamed_ms=round(streamed_s * 1e3, 1), copy_ms=round(copy / 1e6, 1),
                                  stall_ms=round(stall / 1e6, 1), slot_mb=round(slot / MiB), slots=gpu.stats().slots)
        ws.close()
    gpu.trim()


def test_oom(gpu, ck, rs, stream, facts):
    ws = gpu.register("oom", ck, rs)
    for r in range(len(ws.spans)):
        ws.bind(r, stream)
    torch.cuda.synchronize()
    gpu.trim()
    resident = next(s for s in gpu.stats().sets if s.name == "oom").resident
    torch.cuda.empty_cache()  # torch's cached blocks would otherwise satisfy the retry
    free = gpu.stats().driver_free
    ballast = torch.empty(free - 256 * MiB, dtype=torch.uint8, device="cuda")
    need = 256 * MiB + resident // 2
    try:
        torch.empty(need, dtype=torch.uint8, device="cuda")
        raise AssertionError("expected an out-of-memory error")
    except torch.cuda.OutOfMemoryError:
        pass
    evicted = ws.evict()
    released = gpu.trim()
    t = torch.empty(need, dtype=torch.uint8, device="cuda")
    t.fill_(1)
    torch.cuda.synchronize()
    del t, ballast
    torch.cuda.empty_cache()
    assert exact(ws, ck, stream) > 0, "refault after the OOM"
    facts.update(d_resident=resident, d_evicted=evicted, d_released=released, d_retry_bytes=need)
    ws.close()
    gpu.trim()


def pagein(gpu, ws, stream) -> dict:
    """Every layer bound from absent: host wall, host map time, GPU copy time."""
    before = counters(gpu, ws.name)
    t0 = time.perf_counter()
    pins = [ws.bind(r, stream) for r, _ in layer_regions(ws)]
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    c = counters(gpu, ws.name)
    del pins
    torch.cuda.synchronize()
    ws.evict()
    gpu.trim()
    gb = (c.paged_in - before.paged_in) / 1e9
    return dict(gb=round(gb, 3), wall_ms=round(wall * 1e3, 1), map_ms=round((c.map_ns - before.map_ns) / 1e6, 2), copy_ms=round((c.copy_ns - before.copy_ns) / 1e6, 1))


def test_costs(ck, rs, host_fd, stream, facts):
    for chunk in (2 * MiB, 64 * MiB):
        gpu = varena.Gpu(0, chunk)
        for src, fd in (("files", None), ("sealed", host_fd)):
            ws = gpu.register(f"cost-{src}", ck, rs, fd)
            facts[f"e_pagein_{chunk // MiB}MiB_{src}"] = [pagein(gpu, ws, stream) for _ in range(2)]
            ws.close()
        del gpu
    gpu = varena.Gpu(0)
    ws = gpu.register("bind", ck, rs)
    block = next(r for r, ps in by_region(ws).items() if ps[0].component == "blocks")
    parts = by_region(ws)[block]
    views(ws.bind(block, stream), parts)
    torch.cuda.synchronize()
    n = 1000
    t0 = time.perf_counter()
    for _ in range(n):
        pin = ws.bind(block, stream)
    bind_us = (time.perf_counter() - t0) / n * 1e6
    del pin
    t0 = time.perf_counter()
    for _ in range(n):
        v = views(ws.bind(block, stream), parts)
        del v
    torch.cuda.synchronize()
    facts["e_bind"] = dict(params=len(parts), bind_unbind_us=round(bind_us, 1), with_views_us=round((time.perf_counter() - t0) / n * 1e6, 1))
    ws.close()


def test_adopt(gpu, ck, rs, host_fd, stream, facts):
    ws = gpu.adopt("adopt", ck, host_fd)
    assert ws.regions == rs, "a sealed layout carries the regions it was built from"
    t0 = time.perf_counter()
    for r in range(len(ws.spans)):
        ws.prefetch(r)
    torch.cuda.synchronize()
    facts["f_prefetch_ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    facts["f_adopt_bytes"] = exact(ws, ck, stream)
    ws.close()
    gpu.trim()


def main(work: Path) -> int:
    torch.use_deterministic_algorithms(True)
    ck = write(work / "store")
    rs = regions(ck)
    assert len(rs) == LAYERS + BLOCKS, f"one region per layer and per block, got {len(rs)}"
    fd = varena.seal(ck, rs, "varena-gpu")
    facts: dict = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    stream = torch.cuda.current_stream().cuda_stream
    gpu = varena.Gpu(0)
    failed = []
    for name, run in (
        ("a", lambda: test_exact(gpu, ck, rs, fd, stream, facts)),
        ("b", lambda: test_kept_view(gpu, ck, rs, stream, facts)),
        ("c", lambda: test_stream(gpu, ck, rs, fd, stream, facts)),
        ("d", lambda: test_oom(gpu, ck, rs, stream, facts)),
        ("e", lambda: test_costs(ck, rs, fd, stream, facts)),
        ("f", lambda: test_adopt(gpu, ck, rs, fd, stream, facts)),
    ):
        try:
            run()
            print(f"{name}: ok", flush=True)
        except Exception:
            failed.append(name)
            facts[f"{name}_error"] = traceback.format_exc()
            print(f"{name}: FAILED\n{facts[f'{name}_error']}", flush=True)
    facts["failed"] = failed
    print("FACTS " + json.dumps(facts), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with tempfile.TemporaryDirectory(prefix="varena-gpu-") as tmp:
        sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tmp)))
