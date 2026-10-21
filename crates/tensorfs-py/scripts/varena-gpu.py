#!/usr/bin/env python3
"""varena's GPU core on a real card, as a standalone consumer: torch and the tensorfs wheel only.

    python varena-gpu.py [workdir]

A layer-structured checkpoint (written here) is registered from its chunks (read by TensorFS)
and from a sealed host layout. The script checks, and prints one JSON line of facts:
  a. map, fill and bind: every part is bit-exact with the checkpoint, at home and after refault;
  b. a view a caller keeps blocks its page's eviction; once it dies, eviction frees the bytes;
  c. layers streamed through slots with prefetch: bit-exact with all-home; stall < copy time;
  d. a torch OOM beside home pages: evict + trim, then the retry succeeds in-process;
  e. bind/unbind per region of ~20 params (M2), and page-in at 2 MiB vs 64 MiB chunks;
  g. a refill from the sealed layout is logged against the measured H2D ceiling and link, and a
     set in use gives a slot no bound block holds.
  f. adopt: the sealed layout registered from the regions it carries, prefetched whole before
     any module exists, is bit-exact.
  h. kept pages: a process fills a shareable set and hands its pages over, then is killed (-9);
     this process imports and maps them with no copy, bit-exact; closing them frees them.
"""

import ctypes
import json
import os
import signal
import socket
import subprocess
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
    w = tensorfs.Writer(cache_dir=root)
    model, blocks = (w.component(name, cls=("test", "1.0", "Model"), config={}) for name in ("model", "blocks"))
    g = torch.Generator().manual_seed(7)
    for i in range(LAYERS):
        model.put(f"layers.{i}.weight", (torch.randn(DIM, DIM, generator=g) / DIM**0.5).half().numpy(), dtype="F16", shape=[DIM, DIM])
    for b in range(BLOCKS):
        for p in range(PARAMS):
            blocks.put(f"{b}.p{p}", torch.randn(512, 1024, generator=g).bfloat16().view(torch.int16).numpy().tobytes(), dtype="BF16", shape=[512, 1024])
    return w.commit()


def regions(ck: tensorfs.Checkpoint) -> list[list[str]]:
    cuts = [f"model/layers.{i}" for i in range(LAYERS)] + [f"blocks/{b}" for b in range(BLOCKS)]
    return varena.regions(ck, 64 * MiB, cuts)


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
            want = torch.frombuffer(bytearray(ck.components[p.component].tensors.read(p.key).tobytes()), dtype=torch.uint8)
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


def test_ceiling(ck, rs, host_fd, stream, facts):
    varena.ceiling(device=0)  # reads never measure: the H2D ceiling is measured first
    gpu = varena.Gpu(0)
    ws = gpu.register("ceiling", ck, rs, host_fd)
    varena.copies()
    ws.home(list(range(len(ws.spans))))
    for r in range(len(ws.spans)):
        ws.prefetch(r)
    pins = [ws.bind(r, stream) for r in range(len(ws.spans))]
    torch.cuda.synchronize()
    runs: list = []
    while not runs:  # the copy thread logs a run once its last copy landed
        runs = [c for c in varena.copies() if c.kind == "h2d"]
        time.sleep(0.01)
    nbytes = sum(p.nbytes for p in ws.parts)
    got, ns = sum(c.bytes for c in runs), sum(c.ns for c in runs)
    pct = 100 * got / ns / runs[0].ceiling
    assert got >= nbytes and runs[0].ceiling > 0, f"{got} B logged for {nbytes}, ceiling {runs[0].ceiling}"
    assert pct >= 50, f"a refill at {pct:.0f}% of the H2D ceiling"
    facts["g_refill"] = dict(gb=round(got / 1e9, 3), gbps=round(got / ns, 2), ceiling=round(runs[0].ceiling, 2), pct=round(pct, 1),
                             link=runs[0].link, staged_gb=round(sum(c.staged for c in runs) / 1e9, 3), runs=len(runs))
    del pins
    torch.cuda.synchronize()
    ws.home([])
    ws.evict()
    ws.busy(True)
    pin = ws.bind(0, stream)
    torch.cuda.synchronize()
    gpu.give(1 << 62)
    assert gpu.stats().slots == 1, "a bound slot is never given"
    del pin
    torch.cuda.synchronize()
    gpu.give(1 << 62)
    assert gpu.stats().slots == 0, "a slot no bound block holds is given"
    ws.bind(1, stream)
    torch.cuda.synchronize()
    assert gpu.stats().slots == 1, "the next streamed block maps a slot again"
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


class Location(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int), ("id", ctypes.c_int)]


def access(pin: varena.Pinned) -> int:
    flags = ctypes.c_ulonglong()
    location = Location(1, 0)
    get_access = ctypes.CDLL("libcuda.so.1").cuMemGetAccess
    get_access.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(Location), ctypes.c_ulonglong]
    assert get_access(ctypes.byref(flags), ctypes.byref(location), pin.ptr) == 0
    return flags.value


def keeper(store: str, identity: str, fd: int) -> None:
    """h's first process: every region filled at home, its pages handed over, then it waits for
    its kill."""
    ck = tensorfs.open(identity, cache_dir=store, local_files_only=True)
    ws = varena.Gpu(0).register("kept-a", ck, regions(ck))
    assert ws.shareable(), "this card shares VMM pages by fd"
    stream = torch.cuda.current_stream().cuda_stream
    held = [torch.from_dlpack(ws.bind(r, stream)) for r in range(len(ws.spans))]
    torch.cuda.synchronize()
    expected = [view[:4096].clone() for view in held]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = [view[:4096].clone() for view in held]
    graph.replay()
    torch.cuda.synchronize()
    assert not ws.kept(), "ordinary export still refuses lifetime-pinned mappings"
    kept = ws.kept_quiescent()
    assert len(kept) == len(held), "drained graph-held pages can be retained without unpinning"
    for r, _, _ in kept:
        pin = ws.bind(r, stream)
        assert not pin.writable and access(pin) == 1, "exported mappings are CUDA read-only"
        del pin
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(a, b) for a, b in zip(expected, replayed, strict=True)), "READ protection preserves graph replay"
    meta = json.dumps({"regions": [(r, span) for r, span, _ in kept], "graph_held": len(held)}).encode()
    socket.send_fds(socket.socket(fileno=fd), [meta], [f for _, _, f in kept])
    signal.pause()


def test_survive(gpu, ck, rs, store, stream, facts):
    ours, theirs = socket.socketpair()
    argv = [sys.executable, __file__, "--keep", str(store), ck.id, str(theirs.fileno())]
    child = subprocess.Popen(argv, pass_fds=[theirs.fileno()])
    theirs.close()
    try:
        meta, containers, _, _ = socket.recv_fds(ours, 1 << 16, 256)
        document = json.loads(meta)
        meta = document["regions"]
        assert document["graph_held"] == len(rs), "the exporter kept its actual graph pins"
        facts["h_graph_held"] = document["graph_held"]
        assert len(meta) == len(rs), f"every region kept, got {len(meta)} of {len(rs)}"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        ours.close()
    pages = []
    for (r, _), c in zip(meta, containers, strict=True):
        with socket.socket(fileno=c) as s:
            pages.append((r, socket.recv_fds(s, 1, 1)[1][0]))
    ws = gpu.register("kept-b", ck, rs)
    assert ws.shareable()
    t0 = time.perf_counter()
    adopted = ws.adopt_pages(pages)
    facts["h_adopt_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
    for _, f in pages:
        os.close(f)
    assert sorted(adopted) == sorted(r for r, _ in pages), f"adopted {adopted} of {len(pages)}"
    facts["h_exact_bytes"] = exact(ws, ck, stream)
    for r, _ in meta:
        pin = ws.bind(r, stream)
        assert not pin.writable and access(pin) == 1, "imported mappings are CUDA read-only"
        del pin
    facts["h_access"] = "read-only"
    assert counters(gpu, "kept-b").paged_in == 0, "kept pages are mapped, never copied"
    torch.cuda.synchronize()
    again = ws.kept()
    for _, _, f in again:
        os.close(f)
    assert len(again) == len(meta), "adopted pages are kept again for the next process"
    torch.cuda.synchronize()
    free = gpu.stats().driver_free
    ws.close()
    gpu.trim()
    gained = gpu.stats().driver_free - free
    kept_bytes = sum(span for _, span in meta)
    assert gained >= kept_bytes - 2 * MiB, f"closing freed {gained} of {kept_bytes} kept"
    facts.update(h_kept_bytes=kept_bytes, h_adopted=adopted, h_freed=gained)


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
        ("g", lambda: test_ceiling(ck, rs, fd, stream, facts)),
        ("f", lambda: test_adopt(gpu, ck, rs, fd, stream, facts)),
        ("h", lambda: test_survive(gpu, ck, rs, work / "store", stream, facts)),
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
    if sys.argv[1:2] == ["--keep"]:
        keeper(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    with tempfile.TemporaryDirectory(prefix="varena-gpu-") as tmp:
        sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tmp)))
