#!/usr/bin/env python3
"""varena on its own, with no TensorD, under a Diffusers model and plain torch.

A `UNet2DConditionModel` (305 MiB) is written into a TensorFS store; the same model built on
meta is loaded through `tensorfs.varena.torch.Memory` (blocks from `_no_split_modules`) and run
under `mem.request`:
  - with room for it all, against the plain model; held bound (a CUDA graph's hold); a direct
    child's forward is its own pass, keyed by its entry; a fill-identity change sheds a hold and
    bakes anew; two threads' passes measured apart;
  - in two processes on one table, the idle one making room for the other;
  - with a capacity below its size, so it streams, its outputs the uncapped run's and its fills
    reported against `tensorfs.ceiling`; and so with a declared fill on fresh pages.
First, the compiled `tensorfs._ext.varena` matches its stub name for name.

    python scripts/varena-standalone.py
Runs on the CPU over host pages (each run names its own tables), or on GPU 0 when one is visible:
its capacity is lowered by the standalone limit, and the uncapped GPU run is the reference.
"""

from __future__ import annotations

import ast
import copy
import faulthandler
import gc
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from collections.abc import Callable
from pathlib import Path

import tensorfs
import torch
from diffusers import UNet2DConditionModel
from tensorfs._ext import varena as native
from tensorfs.varena import GpuId
from tensorfs.varena import torch as adapter
from tensorfs.varena.torch import Declared, Memory, Placeholder, Weights

CONFIG = dict(
    sample_size=8,
    in_channels=4,
    out_channels=4,
    block_out_channels=(320, 640),
    layers_per_block=1,
    cross_attention_dim=768,
    attention_head_dim=8,
    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
)
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda", 0) if CUDA else torch.device("cpu")
# Every process of the test computes in F32 (TF32 convolutions differ by about 1e-3).
torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
#: The same device's kernels agree to F32 rounding; the CPU and a GPU differ by more.
SAME, ACROSS = 1e-5, 1e-3
results: list[bool] = []


def check(name: str, ok: object, detail: object = "") -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


def watch() -> None:
    """Diagnostics only, never a failure: every thread's Python stack after a minute with no new
    check (a minute: far above any one check's time here, and few enough dumps to read)."""
    seen = -1
    while True:
        time.sleep(60)
        if len(results) == seen:
            print(f"STALL no check in 60 s after {seen}; every thread's stack:", flush=True)
            faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
        seen = len(results)


def build(store: Path) -> tuple[UNet2DConditionModel, tensorfs.Checkpoint]:
    """The plain model and its checkpoint: the same bytes in every process (seeded)."""
    torch.manual_seed(0)
    plain = UNet2DConditionModel(**CONFIG).eval()
    w = tensorfs.Writer(cache_dir=store)
    unet = w.component("unet", cls=("diffusers", "0.35.1", "UNet2DConditionModel"), config=CONFIG)
    for key, t in plain.state_dict().items():
        t = t.detach().contiguous()
        unet.put(key, t.view(torch.uint8).numpy().tobytes(), dtype="F32", shape=list(t.shape))
    return plain, w.commit()


def inputs(device: torch.device, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(n)
    return torch.randn(1, 4, 8, 8, generator=g).to(device), torch.randn(1, 5, 768, generator=g).to(device)


def forward(unet: UNet2DConditionModel, device: torch.device, n: int) -> torch.Tensor:
    x, text = inputs(device, n)
    with torch.no_grad():
        return unet(x, 3, encoder_hidden_states=text).sample.cpu()


class Scale:
    """A declared fill, as a LoRA bake is: scales each attention query weight by `k` as its page
    arrives; `k` is its identity, read at every pass."""

    def __init__(self, k: int) -> None:
        self.k = k

    @property
    def identity(self) -> str:
        return f"to_q x{self.k}"

    def __call__(self, block: str, parts: dict[str, dict[str, torch.Tensor]]) -> None:
        for key, roles in parts.items():
            if key.endswith("to_q.weight"):
                roles["value"].mul_(self.k)


def scaled(plain: UNet2DConditionModel, k: int) -> list[torch.Tensor]:
    """The plain model's outputs with its attention query weights scaled by `k`."""
    model = UNet2DConditionModel(**CONFIG).eval()
    model.load_state_dict({key: v * k if key.endswith("to_q.weight") else v for key, v in plain.state_dict().items()})
    return [forward(model, torch.device("cpu"), n) for n in range(2)]


def loaded(ck: tensorfs.Checkpoint, mem: Memory, fill: Scale | None = None, release: str = "") -> UNet2DConditionModel:
    """The model on meta, loaded. A `release` names a set of its own (a filled model's is
    "scaled"), whose measured rooms are its own: a probe's big buffers stay out of the others'."""
    with torch.device("meta"):
        unet = UNet2DConditionModel(**CONFIG).eval()
    vars(unet)["_varena"] = Declared(Weights(ck, "unet", release=release or ("scaled" if fill else "")), fill=fill)
    mem.load(unet, "gpu")
    return unet


def concurrent(mem: Memory, ck: tensorfs.Checkpoint) -> None:
    """Two threads run two models' passes interleaved, each request on its own thread's pool:
    the first allocates and frees a buffer, then waits while the second runs its pass."""
    a, b = loaded(ck, mem, release="concurrent"), loaded(ck, mem, release="concurrent")
    big = 1 << 29  # far above the tiny model's own activations, so the peaks tell the passes apart
    opened, done = threading.Event(), threading.Event()

    def allocate(*_: object) -> None:
        torch.empty(big, dtype=torch.uint8, device=mem.device)  # freed at once: only the peak keeps it
        opened.set()
        done.wait()

    hook = a.conv_in.register_forward_pre_hook(allocate)
    errors: list[BaseException] = []

    def pass_a() -> None:
        try:
            with mem.request("concurrent a", plan=[a]):
                forward(a, mem.device, 0)
        except BaseException as e:
            errors.append(e)

    def pass_b() -> None:
        try:
            opened.wait()
            with mem.request("concurrent b", plan=[b]):
                forward(b, mem.device, 0)
        except BaseException as e:
            errors.append(e)
        finally:
            done.set()

    threads = [threading.Thread(target=pass_a), threading.Thread(target=pass_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    hook.remove()
    check("concurrent: two threads' passes both complete", not errors, errors)


def run(mem: Memory, unet: UNet2DConditionModel, want: list[torch.Tensor], id: str, atol: float) -> tuple[native.Stats, list[torch.Tensor]]:
    """A forward of `unet` per expected output, in one request."""
    got = []
    try:
        with mem.request(id, plan=[unet] * len(want)) as r:
            for n, expected in enumerate(want):
                got.append(forward(unet, mem.device, n))
                diff = (got[n] - expected).abs().max().item()
                check(f"{id}: pass {n} equals its reference (max diff {diff:.2e})", diff <= atol, diff)
    except native.Exhausted as short:
        print(f"EXHAUSTED {id}: {short}; STATS {short.stats.json()}", flush=True)
        raise
    stats = r.stats()
    print(f"STATS {id}: {stats.json()}; pool free at make_room {r.pool_free}; flush {r.flush_ms:.1f} ms", flush=True)
    check(f"{id}: leaves nothing behind", stats.leftover == 0, stats.leftover)
    check(f"{id}: its record crosses as JSON", json.loads(stats.json())["request"] == id, stats.json())
    return stats, got


def fills(stats: native.Stats, ceiling: tensorfs.Ceiling) -> None:
    """Each source's fills as GB/s and as % of the ceiling they run against: H2D for a GPU's
    fills (and the disk too for a store read); on the host only a store read has one."""
    for source in ("kept", "store"):
        f = getattr(stats.moved, source)
        if not f.bytes or not f.ns:
            continue
        bounds = {"h2d": ceiling.h2d_gbps} if ceiling.h2d_gbps else {}
        if source == "store":
            bounds["disk"] = ceiling.disk_gbps
        gbps = f.bytes / f.ns
        against = "".join(f" = {100 * gbps / g:.0f}% of the {n} ceiling {g:.2f} GB/s" for n, g in bounds.items())
        print(f"FILLS {source}: {f.bytes} B in {f.ns / 1e6:.1f} ms = {gbps:.2f} GB/s{against}", flush=True)


def surface(classes: dict[str, set[str]], name: str, members: set[str]) -> None:
    classes[name] = {m for m in members if not m.startswith("_") or m in ("__dlpack__", "__dlpack_device__")}


def stub_matches() -> None:
    """The compiled module against `varena/__init__.pyi`: every class and its own members."""
    stub = Path(tensorfs.__file__).parent / "varena" / "__init__.pyi"
    want: dict[str, set[str]] = {}

    def walk(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                members = {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                members |= {n.target.id for n in node.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
                exception = any(isinstance(b, ast.Name) and b.id in ("Exception", "VarenaError") for b in node.bases)
                surface(want, prefix + node.name, set() if exception else members)
                walk(node.body, prefix + node.name + ".")

    walk(ast.parse(stub.read_text()).body, "")
    got: dict[str, set[str]] = {}

    def own(cls: type, key: str) -> None:
        surface(got, key, set() if issubclass(cls, BaseException) else set(vars(cls)))
        for name, value in vars(cls).items():  # a complex enum's variant classes
            if isinstance(value, type) and issubclass(value, cls) and value is not cls:
                own(value, f"{key}.{name}")

    for name in dir(native):
        if isinstance(getattr(native, name), type) and not name.startswith("_"):
            own(getattr(native, name), name)
    check("the compiled module matches its stub", got == want, {k: (got.get(k), want.get(k)) for k in got.keys() | want.keys() if got.get(k) != want.get(k)})


def opened(table: str, capacity: int) -> Memory:
    """A standalone memory of `capacity` bytes: host pages on table `table`, or GPU 0's table
    lowered to it."""
    if CUDA:
        return Memory.open(0, limit=capacity)
    return Memory.open(GpuId(table, "cpu", capacity))


def child(store: str, table: str, capacity: int, reference: str) -> int:
    """The second process: two passes, so a block it ran has a declared later use (D), which
    varena keeps over another process's idle pages (N): the idle first gives room."""
    _, ck = build(Path(store))
    mem = opened(table, capacity)
    # A set of its own: the first's kept pages of the same set would be imported, shared, not made room for.
    stats, _ = run(mem, loaded(ck, mem, release="child"), torch.load(reference), "child", SAME)
    print(json.dumps({"child origins": [r.origin for r in stats.trace], "evicted": stats.evicted}), flush=True)
    return 0 if all(results) else 1


def gone(memory: weakref.ref[Memory], phase: str) -> None:
    """One Memory per device in a process: a phase's goes before the next opens (on a GPU they
    share the card's table, and each would count the other's pages as its own context). On a
    GPU its dropped pools' segments, torch's until it releases them, hold its row: released
    (empty_cache) and retired first."""
    gc.collect()
    if CUDA:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        while native.allocator_pending(0):
            time.sleep(1e-3)
    check(f"{phase}: its memory is gone before the next phase opens one", memory() is None, gc.get_referrers(memory()) if memory() else "")


def costs(mem: Memory, unet: UNet2DConditionModel, plain: UNet2DConditionModel) -> None:
    """Microseconds per warm bind and per pass, varena's and the adapter's (a warm bind attaches
    the views it kept; a pass is varena's pass_ and end, and the adapter's open and close). The
    adapter's own work in a forward (a bind per block and its pass) is held to the plain model's
    forward on the same machine: a ratio, as host CPUs differ."""
    model, n = mem.models[unet], 200
    block = max(model.blocks.values(), key=lambda b: len(b.tensors))
    stream = torch.cuda.current_stream(mem.device).cuda_stream if CUDA else 0
    mem._enter(block)
    mem._exit(block)

    def each(f: object, n: int = n) -> float:
        began = time.perf_counter_ns()
        for _ in range(n):
            f()  # type: ignore[operator]
        return (time.perf_counter_ns() - began) / n / 1e3

    native_bind = each(lambda: mem.native.bind(model.id, block.region, stream))
    adapter_bind = each(lambda: (mem._enter(block), mem._exit(block)))
    with mem.request("costs", plan=[unet]) as r:
        def native_pass() -> None:
            r.native.pass_(model.id, "costs|1").end(0)

        def adapter_pass() -> None:
            mem._open(model, lambda: "costs")
            mem._close(model)

        varena_pass, open_close = each(native_pass), each(adapter_pass)
    on = plain if mem.device.type == "cpu" else copy.deepcopy(plain).to(mem.device)
    x, text = inputs(mem.device, 0)

    def plain_forward() -> None:
        with torch.no_grad():
            on(x, 3, encoder_hidden_states=text)
        if CUDA:
            torch.cuda.synchronize(mem.device)

    plain_forward()
    forward = min(each(plain_forward, 1) for _ in range(10))  # the best of ten: the shared machine's load is not it
    # Its own work a forward: the largest block's warm bind for every block, and its pass.
    own = len(model.blocks) * (adapter_bind - native_bind) + (open_close - varena_pass)
    print(f"COST warm bind {native_bind:.0f} us varena, {adapter_bind:.0f} us through the adapter ({len(block.tensors)} tensors); "
          f"pass {varena_pass:.0f} us varena, {open_close:.0f} us through the adapter (its common's bind included); "
          f"plain forward {forward:.0f} us", flush=True)
    check(f"cost: the adapter's own work a forward ({len(model.blocks)} binds and its pass) is {own:.0f} us, "
          f"{100 * own / forward:.1f}% of the plain model's forward (at most 10%)", own <= 0.10 * forward, (native_bind, adapter_bind, varena_pass, open_close, forward))


def threads() -> dict[int, tuple[str, float]]:
    """Each thread's name and CPU seconds (Linux), to say where a run's CPU went."""
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            stat = Path(f"/proc/self/task/{tid}/stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2:].split()
        out[int(tid)] = (stat[stat.index("(") + 1:stat.rindex(")")], (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"))
    return out


def cpu_by_thread(before: dict[int, tuple[str, float]], after: dict[int, tuple[str, float]]) -> dict[str, float]:
    """CPU ms between two `threads()` readings: the main thread, then the others by name."""
    main, spent = threading.get_native_id(), {}
    for tid, (name, cpu) in after.items():
        key = "main" if tid == main else name
        spent[key] = round(spent.get(key, 0.0) + 1e3 * (cpu - before.get(tid, (name, 0.0))[1]), 1)
    return {k: v for k, v in sorted(spent.items(), key=lambda kv: -kv[1]) if v}


def overhead(ck: tensorfs.Checkpoint, plain: UNet2DConditionModel, tag: str) -> weakref.ref[Memory]:
    """On host pages, a warm forward through the adapter costs at most 10% more CPU than the
    plain model: what a user pays. Beside it, the split: the plain model on the same pages (the
    same bytes and kernels) is placement's part, the rest the adapter's. Its own table and model,
    every block a hit; the best of several tries each, interleaved, so the shared machine's load
    is not the measurement. On one torch thread: torch's idle workers spin while the main thread
    runs Python, so with N workers the adapter's own time counts up to N+1 times, and the ratio
    would follow the machine's core count, not the adapter."""
    threads_before = torch.get_num_threads()
    torch.set_num_threads(1)
    mem = opened(f"{tag}-overhead", 1 << 31)
    unet = loaded(ck, mem)
    x, text = inputs(torch.device("cpu"), 0)
    model = mem.models[unet]
    regions = [b for b in (model.common, *model.blocks.values()) if b is not None]
    pins = [mem.native.bind(model.id, b.region, 0) for b in regions]  # held: the pages stay put
    with torch.device("meta"):
        paged = UNet2DConditionModel(**CONFIG).eval()
    paged.load_state_dict({k: v["value"] for b, pin in zip(regions, pins) for k, v in adapter._views(b, torch.from_dlpack(pin.window())).items()}, assign=True)

    def bare(m: UNet2DConditionModel) -> Callable[[], None]:
        def run_() -> None:
            with torch.no_grad():
                for _ in range(3):
                    m(x, 3, encoder_hidden_states=text)
        return run_

    def served() -> None:
        with mem.request("overhead", plan=[unet] * 3) as r, torch.no_grad():
            for _ in range(3):
                unet(x, 3, encoder_hidden_states=text)
        origins.update(ref.origin for ref in r.stats().trace)

    runs = {"plain": bare(plain), "paged": bare(paged), "served": served}
    origins: set[str] = set()
    for run_ in runs.values():
        run_()
    origins.clear()
    best = dict.fromkeys(runs, float("inf"))
    # All runs' CPU by thread (clock ticks: too coarse for one run).
    where: dict[str, dict[int, tuple[str, float]]] = {name: {} for name in runs}
    for _ in range(11):
        for name, run_ in runs.items():
            before, began = threads(), time.process_time()
            run_()
            best[name] = min(best[name], time.process_time() - began)
            for tid, (thread, cpu) in threads().items():
                where[name][tid] = (thread, where[name].get(tid, (thread, 0.0))[1] + cpu - before.get(tid, (thread, 0.0))[1])
    del runs, paged, pins
    costs(mem, unet, plain)
    ratio, adapter_part, placement = best["served"] / best["plain"], best["served"] / best["paged"], best["paged"] / best["plain"]
    torch.set_num_threads(threads_before)
    print("OVERHEAD best CPU ms: " + ", ".join(f"{name} {1e3 * t:.1f}" for name, t in best.items()) + "; all 11 runs' CPU ms by thread: "
          + "; ".join(f"{name} {cpu_by_thread({}, w)}" for name, w in where.items()), flush=True)
    check(f"overhead: a warm forward through the adapter costs {100 * (ratio - 1):+.1f}% CPU over plain (at most +10%; "
          f"adapter {100 * (adapter_part - 1):+.1f}%, placement {100 * (placement - 1):+.1f}%)", ratio <= 1.10 and origins == {"hit"}, (best, origins))
    return weakref.ref(mem)


def ample(ck: tensorfs.Checkpoint, plain: UNet2DConditionModel, tag: str) -> tuple[list[torch.Tensor], list[int], int, int, weakref.ref[Memory]]:
    """Room for it all: against the plain model, held, entries, identity, concurrency, attempts.
    Returns the reference outputs, the region spans, the measured room, the base (contexts and
    foreign bytes) and the memory."""
    expected = [forward(plain, torch.device("cpu"), n) for n in range(2)]
    mem = Memory.open(0) if CUDA else opened(f"{tag}-ample", 1 << 31)
    again = Memory.open(0) if CUDA else opened(f"{tag}-ample", 1 << 31)
    check("one Memory per device: opening it again returns it", again is mem, again)
    del again
    unet = loaded(ck, mem)
    stats, reference = run(mem, unet, expected, "ample", ACROSS if CUDA else SAME)
    model = mem.models[unet]
    with model.hold():
        weight = unet.down_blocks[0].attentions[0].proj_in.weight
        check("hold: every block stays bound", not isinstance(weight, Placeholder), type(weight))
        del weight  # kept past the hold it would be an escaped weight (and reported as one)
        run(mem, unet, reference[:1], "held", SAME)
    check("hold: released, placeholders are back", isinstance(unet.down_blocks[0].attentions[0].proj_in.weight, Placeholder))
    binding, pins = model.retain(), native.pins_held()
    check("retained: a model just run is resident (a read: no pin, no fill)", binding.resident() and native.pins_held() == pins, (pins, native.pins_held()))
    binding.close()
    serving = mem.cache("ample graph pool", lambda want: 0, sets=unet)  # a graph pool names its component
    check("cache: one serving a loaded model registers with its set", isinstance(serving, adapter.Cache), serving)
    serving.close()
    free = model.hold(free_only=True)
    check("hold: free only, a model that fits beside what is resident is held", free is not None and not isinstance(unet.conv_in.weight, Placeholder), free)
    if free is not None:
        free.release()
    keys: list[str] = []
    with mem.request("entries", plan=[unet, unet]) as r:
        def seen(*_: object) -> None:
            keys.append(adapter._pass.get().key.split("|")[0])

        probe = unet.conv_in.register_forward_pre_hook(seen)
        x, _ = inputs(mem.device, 0)
        with torch.no_grad():
            got = unet.conv_in(x).cpu()
        forward(unet, mem.device, 0)
    probe.remove()
    diff = (got - plain.conv_in(x.cpu()).detach()).abs().max().item()
    check(f"entries: a direct child's forward runs bound (max diff {diff:.2e})", diff <= ACROSS, diff)
    check("entries: each entry is its own pass key", keys == ["conv_in", ""], keys)
    steps = r.stats().steps
    # The capacity varena ran a step against is the card less contexts (grown by now: cuDNN,
    # cuBLAS) and foreign bytes; the limits later are that much plus what they mean to hold.
    base = mem.gpu.memory - min(step.capacity for step in steps)

    shed: list[bool] = []
    scale = Scale(2)
    filled = loaded(ck, mem, scale)
    hold = mem.models[filled].hold(shed=lambda: shed.append(True))
    run(mem, filled, scaled(plain, 2)[:1], "scaled x2, held", ACROSS if CUDA else SAME)
    scale.k = 3
    run(mem, filled, scaled(plain, 3)[:1], "scaled x3", ACROSS if CUDA else SAME)
    check("identity: a pass under a new identity shed the hold of the old bake", shed == [True] and not mem.models[filled].holds, (shed, hold.blocks))
    concurrent(mem, ck)
    if CUDA:
        costs(mem, unet, plain)
    spans = model.set.spans
    return reference, spans, stats.compute, base, weakref.ref(mem)


def two(ck: tensorfs.Checkpoint, store: Path, table: str, capacity: int, reference: list[torch.Tensor]) -> tuple[int, weakref.ref[Memory]]:
    """Two processes on one table, the idle first making room for the second. Returns the most
    the request held at a pass end (pages and segments) and the memory."""
    saved = store.parent / "reference.pt"
    torch.save(reference, saved)  # the child runs two passes: its run blocks have a declared later use
    mem = opened(table, capacity)
    unet = loaded(ck, mem)
    run(mem, unet, reference[:1], "first", SAME)
    stats, _ = run(mem, unet, reference[:1], "first warm", SAME)
    check("two processes: alone, the first reads nothing again", all(r.origin == "hit" for r in stats.trace), [r.origin for r in stats.trace])
    done = subprocess.run([sys.executable, __file__, "child", str(store), table, str(capacity), str(saved)], capture_output=True, text=True)
    print(done.stdout, done.stderr, sep="", end="", flush=True)
    check("two processes: the second completes beside the idle first", done.returncode == 0, done.returncode)
    stats, _ = run(mem, unet, reference[:1], "first again", SAME)
    check("two processes: the first gave room and reads again", any(r.origin != "hit" for r in stats.trace), [r.origin for r in stats.trace])
    return stats.peak, weakref.ref(mem)


def capped(ck: tensorfs.Checkpoint, plain: UNet2DConditionModel, table: str, capacity: int, base: int, spans: list[int], reference: list[torch.Tensor], ceiling: tensorfs.Ceiling) -> weakref.ref[Memory]:
    """Below its size it streams, its outputs the uncapped run's; its fills against the ceiling.
    Then weights that escape their block's run while it streams."""
    mem = opened(table, capacity)
    unet = loaded(ck, mem)
    stats, _ = run(mem, unet, reference, "short", SAME)
    check(f"short: below its {sum(spans)} B it streams in {capacity - base} B", sum(stats.evicted) > 0, stats.evicted)
    pins = native.pins_held()
    binding = mem.models[unet].retain()
    check("short: retained, a model bigger than its room is not all resident", not binding.resident(), "resident")
    binding.close()
    refused = mem.models[unet].hold(free_only=True)
    check("short: free only, a model bigger than its room is not held, and nothing stays bound", refused is None and native.pins_held() == pins and isinstance(unet.conv_in.weight, Placeholder), (refused, pins, native.pins_held()))
    fills(stats, ceiling)
    print(f"STATS_JSON {stats.json()}", flush=True)
    for derived in (False, True):
        escapes(mem, unet, plain, reference, derived)
    off_forward(mem, unet, plain)
    return weakref.ref(mem)


def pools(tag: str) -> weakref.ref[Memory]:
    """torch's handling of the request pool's segments (CUDA), which the allocator's NULL-once
    rests on: a live pool keeps its cached segment through torch's OOM retry chain; a dropped
    pool (a request's end) gives its unused segments back through varena's free at once; one
    still in use comes back at the next retry chain after its tensor dies."""
    mem, mib = opened(f"{tag}-pools", 1 << 31), 1 << 20
    device = mem.device.index or 0

    def segments(pool: tuple[int, int]) -> tuple[int, int]:
        """The pool's live segment bytes and segments freed, once retirements in flight land."""
        torch.cuda.synchronize(mem.device)
        while native.allocator_pending(device):
            time.sleep(1e-3)
        live, *_, freed = native.allocator_stats(device, pool)
        return live, freed

    def retry_chain() -> None:
        """An allocation over torch's fraction cap: torch runs its retry chain, then raises."""
        torch.cuda.set_per_process_memory_fraction(1e-6, mem.device)
        try:
            torch.empty(64 * mib, dtype=torch.uint8, device=mem.device)
        except torch.OutOfMemoryError:
            pass
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0, mem.device)

    with mem.request("pools") as r:
        pool = r.pool.id
        cached = torch.empty(32 * mib, dtype=torch.uint8, device=mem.device)
        del cached  # a cached, unused segment in the live pool
        before = segments(pool)
        retry_chain()
        check("pools: a live pool keeps its cached segment through torch's retry chain", segments(pool) == before and before[0] >= 32 * mib, (before, segments(pool)))
    check("pools: a request's end gives its pool's unused segments back through varena's free", segments(pool) == (0, before[1] + 1), (before, segments(pool), r.flush_ms))
    with mem.request("pools kept") as r:
        pool = r.pool.id
        kept = torch.empty(32 * mib, dtype=torch.uint8, device=mem.device)
    held = segments(pool)
    del kept
    retry_chain()
    check("pools: a dropped pool's segment in use comes back at the retry chain after its tensor dies", held[0] >= 32 * mib and segments(pool)[0] == 0, (held, segments(pool)))
    return weakref.ref(mem)


def held_caches(tag: str) -> weakref.ref[Memory]:
    """A request's end asks the caches it owns for everything, unless one is held (a graph pool
    a replay still reads)."""
    mem = opened(f"{tag}-held", 1 << 31)
    asked: list[tuple[str, int]] = []
    errors: list[object] = []
    hook, sys.unraisablehook = sys.unraisablehook, errors.append

    def answer(name: str, freed: int) -> Callable[[int], int]:
        return lambda want: (asked.append((name, want)), freed)[1]

    try:
        held = mem.cache("held", answer("held", 1 << 20))
        held.cached(4 << 20)
        with held.holding():
            with mem.request("held"):
                pass
        mem.cache("whole", answer("whole", 4 << 20)).cached(4 << 20)
        with mem.request("whole"):
            pass
    finally:
        sys.unraisablehook = hook
    check("held: a held cache is not shrunk at its request's end", "held" not in dict(asked), asked)
    check("held: an unheld one is asked for everything", asked == [("whole", (1 << 64) - 1)] and not errors, (asked, errors))
    return weakref.ref(mem)


def off_forward(mem: Memory, unet: UNet2DConditionModel, plain: UNet2DConditionModel) -> None:
    """A weight read outside any forward (a package's table, an overlay's factor) binds on read:
    it has its bytes, the tensor made of it holds the page's pin while it lives, the pin goes
    when it dies, and the read is counted."""
    held = native.pins_held()
    weight = unet.down_blocks[1].resnets[0].conv1.weight  # a placeholder: no forward runs
    want = plain.down_blocks[1].resnets[0].conv1.weight.detach()
    row = weight[0]
    check("off-forward: a weight read outside any forward has its bytes", torch.equal(row.cpu(), want[0]), (row.cpu() - want[0]).abs().max())
    check("off-forward: the tensor read holds its page's pin while it lives", native.pins_held() == held + 1, (native.pins_held(), held))
    total = weight.float().sum()  # a converted placeholder reads too
    check("off-forward: a converted read has its bytes", torch.allclose(total.cpu(), want.float().sum(), rtol=1e-5), (total, want.float().sum()))
    del row
    gc.collect()
    check("off-forward: the pin goes when the tensor dies", native.pins_held() == held, (native.pins_held(), held))
    # The runtime's gateway for a read no op sees (a launch by pointer): Placeholder._varena_read.
    bound = weight._varena_read()
    check("off-forward: _varena_read is the bound tensor, its bytes", torch.equal(bound.detach().cpu(), want), "differs")
    check("off-forward: it holds its page's pin while it lives", native.pins_held() == held + 1 and bound.data_ptr() != 0, native.pins_held())
    del bound
    gc.collect()
    check("off-forward: the pin goes when it dies", native.pins_held() == held, native.pins_held())
    try:
        adapter.Placeholder((2,), torch.float32, torch.device("cpu"))._varena_read()
        check("off-forward: a placeholder of no loaded model refuses", False, "read")
    except adapter.Unbound:
        check("off-forward: a placeholder of no loaded model refuses", True)
    reads = {path: n for (_, path), n in mem.off_forward.items()}.get("down_blocks.1.resnets.0", (0, 0))
    check("off-forward: the reads are counted with their bytes", reads[0] == 3 and reads[1] > 0, mem.off_forward)


def escapes(mem: Memory, unet: UNet2DConditionModel, plain: UNet2DConditionModel, reference: list[torch.Tensor], derived: bool) -> None:
    """A module keeps a weight (or a view of one) past its forward and reads it in the next, while
    the model streams: it reads the right bytes, because its page stays bound while the kept
    tensor lives, and the escape is reported."""
    what = "a view of a weight" if derived else "a weight"
    block = unet.down_blocks[1].resnets[0]
    want = plain.down_blocks[1].resnets[0].conv1.weight.detach()
    want = want[0] if derived else want
    kept: dict[str, torch.Tensor] = {}
    read: list[bool] = []

    def keep(module: torch.nn.Module, *_: object) -> None:
        if "w" in kept:
            read.append(torch.equal(kept["w"].cpu(), want))
        weight = module.conv1.weight
        kept["w"] = weight[0] if derived else weight

    hook = block.register_forward_pre_hook(keep)
    with mem.request(f"escape {what}", plan=[unet] * 2) as r:
        for n in range(2):
            got = forward(unet, mem.device, n)
            check(f"escape ({what}): pass {n} equals its reference", torch.allclose(got, reference[n], atol=SAME), (got - reference[n]).abs().max())
    hook.remove()
    late = torch.equal(kept["w"].cpu(), want)  # after both passes streamed the rest of the model
    check(f"escape ({what}): read in the next forward and after, it holds its bytes", read == [True] and late, (read, late))
    check(f"escape ({what}): reported with its block's bytes", sum(r.escaped.values()) > 0, r.escaped)
    del kept["w"]


def baked(ck: tensorfs.Checkpoint, plain: UNet2DConditionModel, table: str, capacity: int) -> weakref.ref[Memory]:
    """A declared fill while it streams: each fresh page baked once."""
    mem = opened(table, capacity)
    stats, _ = run(mem, loaded(ck, mem, Scale(2)), scaled(plain, 2), "fill", ACROSS if CUDA else SAME)
    check("fill: each fresh page is filled once", sum(r.origin != "hit" for r in stats.trace) > 0, stats.trace)
    return weakref.ref(mem)


def main() -> int:
    stub_matches()
    store = Path(tempfile.mkdtemp(prefix="varena-standalone-")) / "store"
    plain, ck = build(store)
    tag = f"varena-standalone-{os.getpid()}"
    reference, spans, room, base, memory = ample(ck, plain, tag)
    gone(memory, "ample")
    if not CUDA:
        gone(overhead(ck, plain, tag), "overhead")
    gone(held_caches(tag), "held caches")
    if CUDA:
        gone(pools(tag), "pools")
    ceiling = tensorfs.ceiling(device=0 if CUDA else None, path=store)
    blocks = sorted(spans[1:])
    # Capacities only fall on one GPU's table: the largest first. A GPU child's context is about ours.
    capacity = base * (2 if CUDA else 1) + sum(spans) * 3 // 2 + room
    w, memory = two(ck, store, f"{tag}-two", capacity, reference)
    gone(memory, "two processes")
    # One block below the working set varena measured: it must stream, with room to spare for
    # what streaming needs (the pass's room, common and its slots).
    capacity = base + w - max(blocks)
    gone(capped(ck, plain, f"{tag}-short", capacity, base, spans, reference, ceiling), "short")
    gone(baked(ck, plain, f"{tag}-fill", capacity), "fill")
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    threading.Thread(target=watch, daemon=True).start()
    sys.exit(child(*sys.argv[2:4], int(sys.argv[4]), sys.argv[5]) if sys.argv[1:2] == ["child"] else main())
