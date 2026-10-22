#!/usr/bin/env python3
"""SPIKE: torch's OOM retried inside its allocator. An SDXL-sized bf16 UNet at 4 MP under a
12 GiB standalone limit, beside a filler set big enough that its passes need evictions. Each
pass allocates from a `torch.cuda.MemPool` whose segments come from varena's allocator: a
shortfall asks varena, and torch sees a segment or, when varena cannot make room, its
OutOfMemoryError. Measured against torch's default allocator under the same cap, interleaved:
  1. it completes with no OOM raised to Python, with the shortfalls resolved in the allocator
     (count, bytes) and the evictions they caused;
  2. warm ms per step, pool against default, best of 3 each;
  3. the allocator's cost per new segment, and segments per warm forward;
  4. a forced exhaustion is torch's OutOfMemoryError at its call site, and the next pass runs;
  5. the bytes torch holds outside the pool in a forward, by source;
  6. no deadlock while a second thread allocates on the device as varena evicts;
  7. torch holding a large cached slack when varena is short: varena asks for no trim, the
     allocator refuses, torch releases its own cache and asks again (no callback runs inside);
  8. no deadlock when a stream host callback that takes the GIL is queued ahead of varena's
     unmaps.

    python scripts/varena-mempool.py main
    python scripts/varena-mempool.py  # SPIKE DIAG: the ILLEGAL_ADDRESS diagnostic (debugging only)
GPU 0 only.
"""

from __future__ import annotations

import ctypes
import faulthandler
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import tensorfs
import torch
from diffusers import UNet2DConditionModel
from tensorfs import _ext
from tensorfs.varena import VarenaError, allocator_reentries, allocator_stats
from tensorfs.varena.torch import Declared, Memory, Weights

CONFIG = dict(
    sample_size=128,
    in_channels=4,
    out_channels=4,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    block_out_channels=(320, 640, 1280),
    layers_per_block=2,
    transformer_layers_per_block=(1, 2, 10),
    attention_head_dim=(5, 10, 20),
    cross_attention_dim=2048,
    use_linear_projection=True,
    addition_embed_type="text_time",
    addition_time_embed_dim=256,
    projection_class_embeddings_input_dim=2816,
)
LATENT = 256  # 2048x2048 pixels: 4 MP
STEPS = 4  # the first runs at a new shape or after the filler; the rest are warm
CAP = 12 << 30  # the vertical slice's card size
results: list[bool] = []


def check(name: str, ok: object, detail: object = "") -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


def watch() -> None:
    """Diagnostics only: every thread's stack after a minute with no new check."""
    seen = -1
    while True:
        time.sleep(60)
        if len(results) == seen:
            print(f"STALL no check in 60 s after {seen}; every thread's stack:", flush=True)
            faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
        seen = len(results)


def build(store: Path, dest: Path | None = None) -> tensorfs.Checkpoint:
    """SDXL's UNet shapes in bf16, random (std 0.02): its bytes, not its pictures, matter here."""
    with torch.device("meta"):
        shapes = UNet2DConditionModel(**CONFIG).state_dict()
    w = tensorfs.Writer(dest, cache_dir=store)
    unet = w.component("unet", cls=("diffusers", "0.35.1", "UNet2DConditionModel"), config=CONFIG)
    g = torch.Generator().manual_seed(0)
    for key, meta in shapes.items():
        t = (torch.randn(meta.shape, generator=g) * 0.02).to(torch.bfloat16)
        unet.put(key, t.view(torch.uint8).numpy().tobytes(), dtype="BF16", shape=list(t.shape))
    return w.commit()


def loaded(mem: Memory, ck: tensorfs.Checkpoint, release: str) -> UNet2DConditionModel:
    with torch.device("meta"):
        unet = UNet2DConditionModel(**CONFIG).to(torch.bfloat16).eval()
    vars(unet)["_varena"] = Declared(Weights(ck, "unet", release=release))
    mem.load(unet, "gpu")
    return unet


class Run:
    def __init__(self, mem: Memory, ck: tensorfs.Checkpoint) -> None:
        self.mem = mem
        self.filler, self.unet = loaded(mem, ck, "filler"), loaded(mem, ck, "")
        g = torch.Generator().manual_seed(1)
        self.x = torch.randn(1, 4, LATENT, LATENT, generator=g).to(mem.device, torch.bfloat16)
        self.text = torch.randn(1, 77, 2048, generator=g).to(mem.device, torch.bfloat16)
        self.cond = {"text_embeds": torch.randn(1, 1280, generator=g).to(mem.device, torch.bfloat16),
                     "time_ids": torch.tensor([[2048, 2048, 0, 0, 2048, 2048]], dtype=torch.bfloat16, device=mem.device)}

    def fill(self) -> None:
        """The filler runs once: its pages are resident and idle, so the model's room needs them."""
        with self.mem.request("filler", plan=[self.filler]), torch.no_grad():
            self.filler(self.x[:, :, :64, :64], 999, encoder_hidden_states=self.text, added_cond_kwargs=self.cond)

    def steps(self, label: str, pool: bool) -> tuple[list[float], list[int], object, BaseException | None]:
        """STEPS passes of the model, its ms and new pool segments per step, its stats."""
        self.mem.pooling = pool
        torch.cuda.empty_cache()
        times, segments, raised = [], [], None
        with self.mem.request(label, plan=[self.unet] * STEPS) as r, torch.no_grad():
            for n in range(STEPS):
                made = allocator_stats()[1]
                torch.cuda.synchronize()
                began = time.perf_counter()
                try:
                    self.unet(self.x, 999 - n, encoder_hidden_states=self.text, added_cond_kwargs=self.cond)
                except (torch.OutOfMemoryError, VarenaError) as e:
                    raised = e
                    break
                torch.cuda.synchronize()
                times.append(1e3 * (time.perf_counter() - began))
                segments.append(allocator_stats()[1] - made)
        return times, segments, r, raised


def main() -> int:
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        print("needs a GPU")
        return 1
    ck = build(Path(tempfile.mkdtemp(prefix="varena-mempool-")) / "store")
    mem = Memory.open(0, limit=CAP, pool=True)
    run = Run(mem, ck)
    shortfalls(mem, run, ck)

    # 1. beside the filler, through the pool: shortfalls resolved in the allocator.
    run.fill()
    before = allocator_stats()
    times, segments, r, raised = run.steps("pool beside the filler", True)
    after = allocator_stats()
    stats = r.stats()
    resolved, nbytes = after[2] - before[2], after[3] - before[3]
    made, spent = after[1] - before[1], after[5] - before[5]
    print(f"POOL beside the filler: ms/step {[round(t) for t in times]}; shortfalls resolved in the allocator {resolved} ({nbytes} B); "
          f"{made} new segments at {spent / max(made, 1) / 1e3:.0f} us each; "
          f"evicted by class {stats.evicted}; varena retries {stats.retries} cuts {stats.cuts} parks {stats.parks}; segments per step {segments}", flush=True)
    check("1: beside the filler, the pool's passes complete with no OOM raised to Python", raised is None and len(times) == STEPS, raised)
    check("1: the filler made the room (evictions)", sum(stats.evicted) > 0, stats.evicted)

    # 2. and 3. interleaved warm runs, the filler refilled before each.
    warm: dict[bool, list[float]] = {True: [], False: []}
    for _ in range(3):
        for pool in (False, True):
            run.fill()
            made, ns = allocator_stats()[1], allocator_stats()[5]
            times, segments, r, raised = run.steps("pool" if pool else "default", pool)
            check(f"2: {'pool' if pool else 'default'} run completes", raised is None and len(times) == STEPS, raised)
            warm[pool].append(sum(times[1:]) / max(len(times[1:]), 1))
            if pool:
                new, spent = allocator_stats()[1] - made, allocator_stats()[5] - ns
                print(f"SEGMENTS per step {segments}; {spent / max(new, 1) / 1e3:.0f} us per new segment ({new} new)", flush=True)
    best = {pool: min(v) for pool, v in warm.items()}
    print(f"WARM best of 3: default {best[False]:.0f} ms, pool {best[True]:.0f} ms ({100 * (best[True] / best[False] - 1):+.1f}%)", flush=True)

    # 5. what torch holds outside the pool in a forward, by source.
    torch.cuda.memory._record_memory_history(max_entries=100000)
    with mem.request("sources", plan=[run.unet]), torch.no_grad():
        run.unet(run.x, 2, encoder_hidden_states=run.text, added_cond_kwargs=run.cond)
    snapshot = torch.cuda.memory._snapshot()
    torch.cuda.memory._record_memory_history(enabled=None)
    pool_id = mem.pool.id if mem.pool is not None else None
    outside: dict[str, int] = {}
    for segment in snapshot["segments"]:
        if tuple(segment.get("segment_pool_id", ())) == tuple(pool_id or ()):
            continue
        for block in segment["blocks"]:
            if block["state"] != "active_allocated":
                continue
            frames = block.get("frames") or []
            where = next((f"{f['filename'].rsplit('/', 1)[-1]}:{f['line']}" for f in frames if "torch/" not in f["filename"]), "unknown")
            outside[where] = outside.get(where, 0) + block["size"]
    print(f"OUTSIDE the pool: Memory.outside {mem.outside} B; allocated outside by source {sorted(outside.items(), key=lambda kv: -kv[1])[:8]}", flush=True)
    here = Path(__file__).name
    check("5: the forward's own allocations are all in the pool (outside: this script's inputs, made before it)", all(k.startswith(here) for k in outside), outside)

    # 6. a second thread allocates on the device while varena evicts for the pool's passes.
    stop, count, errors = threading.Event(), [0], []

    def churn() -> None:
        try:
            while not stop.is_set():
                torch.empty(256 << 20, dtype=torch.uint8, device=mem.device)
                count[0] += 1
                if count[0] % 8 == 0:
                    torch.cuda.empty_cache()  # its segments go back and come again: the driver and the allocator lock
        except BaseException as e:
            errors.append(e)

    other = threading.Thread(target=churn, daemon=True)
    run.fill()
    other.start()
    times, _, r, raised = run.steps("beside a second thread", True)
    stop.set()
    other.join()
    print(f"SECOND thread: {count[0]} allocations of 256 MiB; evicted by class {r.stats().evicted}", flush=True)
    check("6: no deadlock: the pool's passes complete beside a thread allocating as varena evicts", raised is None and len(times) == STEPS and count[0] > 0, (raised, count, errors))
    host_callback(mem, run)
    slack(mem, run)
    # 4. a forced exhaustion.
    refused = allocator_stats()[4]
    caught: list[BaseException] = []

    def huge(*_: object) -> None:
        try:
            torch.empty(2 * CAP, dtype=torch.uint8, device=mem.device)
        except torch.OutOfMemoryError as e:
            caught.append(e)

    mem.pooling = True
    hook = run.unet.conv_in.register_forward_pre_hook(huge)
    after_hook: list[BaseException] = []
    with mem.request("exhaustion", plan=[run.unet]), torch.no_grad():
        try:
            run.unet(run.x, 1, encoder_hidden_states=run.text, added_cond_kwargs=run.cond)
        except (torch.OutOfMemoryError, VarenaError) as e:
            after_hook.append(e)  # the pass's own later binds, after varena answered the impossible need
    print(f"EXHAUSTION at the call site {caught[:1]}; the pass after it {after_hook[:1]}; refusals {allocator_stats()[4] - refused}", flush=True)
    hook.remove()
    check("4: a forced exhaustion is torch's OutOfMemoryError at its call site, varena having refused", len(caught) == 1 and allocator_stats()[4] > refused, (caught, allocator_stats()))
    check("4: the pass that met it goes on", not after_hook, after_hook)
    times, _, _, raised = run.steps("after exhaustion", True)
    check("4: the next passes run", raised is None and len(times) == STEPS, raised)

    print(f"{sum(results)}/{len(results)} checks passed", flush=True)
    return 0 if all(results) else 1


def shortfalls(mem: Memory, run: Run, ck: tensorfs.Checkpoint, fillers: int = 2) -> None:
    """0: the allocator's own path. Two fillers occupy the card; the model's first pass at a
    shape no pass measured (so varena makes no room for its activations ahead) runs through the
    pool; then, in a pass, a buffer bigger than torch's headroom: each shortfall is resolved in
    the allocator by evictions, with no OOM raised to Python."""
    mem.pooling = True
    second = loaded(mem, ck, "filler2") if fillers == 2 else None
    run.fill()
    if second is not None:
        with mem.request("filler2", plan=[second]), torch.no_grad():
            second(run.x[:, :, :64, :64], 999, encoder_hidden_states=run.text, added_cond_kwargs=run.cond)
    before = allocator_stats()
    x = run.x[:, :, :224, :224].contiguous()  # a shape no pass ran
    raised = None
    with mem.request("unmeasured", plan=[run.unet]) as r, torch.no_grad():
        try:
            run.unet(x, 999, encoder_hidden_states=run.text, added_cond_kwargs=run.cond)
        except (torch.OutOfMemoryError, VarenaError) as e:
            raised = e
    after = allocator_stats()
    made, spent = after[1] - before[1], after[5] - before[5]
    print(f"UNMEASURED beside two fillers: shortfalls resolved in the allocator {after[2] - before[2]} ({after[3] - before[3]} B); "
          f"evicted by class {r.stats().evicted}; {made} new segments at {spent / max(made, 1) / 1e3:.0f} us each", flush=True)
    check("0: an unmeasured pass beside two fillers completes with no OOM raised to Python", raised is None, raised)

    run.fill()
    seen: dict[str, object] = {}

    def big(*_: object) -> None:
        r = mem._current()
        headroom = mem.native.torch_limit() - torch.cuda.memory_reserved(mem.device)
        size = (headroom + (1 << 30)) // (2 << 20) * (2 << 20)
        evicted, stats0 = sum(r.stats().evicted), allocator_stats()
        try:
            buffer = torch.empty(size, dtype=torch.uint8, device=mem.device)
            del buffer
            seen["ok"] = True
        except torch.OutOfMemoryError as e:
            seen["ok"] = e
        stats1 = allocator_stats()
        seen.update(size=size, headroom=headroom, resolved=stats1[2] - stats0[2], evicted=sum(r.stats().evicted) - evicted,
                    us=(stats1[5] - stats0[5]) / max(stats1[1] - stats0[1], 1) / 1e3)

    hook = run.unet.conv_in.register_forward_pre_hook(big)
    with mem.request("beyond headroom", plan=[run.unet]), torch.no_grad():
        run.unet(run.x, 998, encoder_hidden_states=run.text, added_cond_kwargs=run.cond)
    hook.remove()
    print(f"BEYOND headroom: {seen}", flush=True)
    check("0: a buffer past torch's headroom is resolved in the allocator, with no OOM raised to Python",
          seen.get("ok") is True and isinstance(seen.get("resolved"), int) and seen["resolved"] > 0, seen)
    check("0: the evictions made it room", isinstance(seen.get("evicted"), int) and seen["evicted"] >= (1 << 30), seen)
    if second is not None:
        mem.models[second].close()


def slack(mem: Memory, run: Run) -> None:
    """7: the filler gone and every block pinned (a hold), so varena has nothing more to free;
    torch's pool caching a
    large freed buffer: a bigger one fits only once torch lets its cache go. varena gets no slack
    and asks for no trim, the allocator refuses once, torch empties its cache and asks again."""
    mem.pooling = True
    mem.models[run.filler].close()  # its pages go: varena has nothing left to free but by a cut
    hold = mem.models[run.unet].hold()
    seen: dict[str, object] = {}

    def probe(*_: object) -> None:
        free = mem.native.torch_limit() - (torch.cuda.memory_reserved(mem.device))
        first, second = int(free * 0.6) // (2 << 20) * (2 << 20), int(free * 0.7) // (2 << 20) * (2 << 20)
        before, reentries = allocator_stats(), allocator_reentries()
        a = torch.empty(first, dtype=torch.uint8, device=mem.device)
        del a  # cached in the pool: torch's slack
        try:
            b = torch.empty(second, dtype=torch.uint8, device=mem.device)
            del b
            seen["ok"] = True
        except torch.OutOfMemoryError as e:
            seen["ok"] = e
        after = allocator_stats()
        seen.update(refused=after[4] - before[4], segments=after[1] - before[1], reentries=(allocator_reentries()[0] - reentries[0], allocator_reentries()[1] - reentries[1]), sizes=(first, second))

    hook = run.unet.conv_in.register_forward_pre_hook(probe)
    with mem.request("slack", plan=[run.unet]), torch.no_grad():
        run.unet(run.x, 3, encoder_hidden_states=run.text, added_cond_kwargs=run.cond)
    hook.remove()
    hold.release()
    print(f"SLACK {seen}", flush=True)
    check("7: torch's cached slack is used by torch: the allocator refused, torch released its cache and the buffer came", seen.get("ok") is True and seen.get("refused", 0) >= 1, seen)
    check("7: varena asked for no trim and ran no shrink inside the allocator", seen.get("reentries") == (0, 0), seen)


def host_callback(mem: Memory, run: Run) -> None:
    """8: a host function that takes the GIL is queued on the stream the pass binds on, ahead of
    the unmaps varena makes for a shortfall in the allocator; the pass completes and it ran."""
    mem.pooling = True
    run.fill()
    driver = ctypes.CDLL("libcuda.so.1")
    ran: list[float] = []
    callback = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(lambda _: ran.append(time.perf_counter()))
    resolved = allocator_stats()[2]

    def queue(*_: object) -> None:
        stream = torch.cuda.current_stream(mem.device).cuda_stream
        rc = driver.cuLaunchHostFunc(ctypes.c_void_p(stream), callback, None)
        ran.append(float(rc))

    hook = run.unet.conv_in.register_forward_pre_hook(queue)
    times, _, _, raised = run.steps("host callback", True)
    hook.remove()
    torch.cuda.synchronize()
    print(f"HOST callback: launches and runs {ran[:8]}; shortfalls resolved {allocator_stats()[2] - resolved}", flush=True)
    check("8: no deadlock with a GIL-taking host callback queued ahead of varena's unmaps", raised is None and len(times) == STEPS and len(ran) >= 2, (raised, ran))


def diag() -> int:
    """The ILLEGAL_ADDRESS diagnostic (debugging only, never product behaviour): gpu11's
    configuration up to its third interleaved iteration, in a child process per variant with
    CUDA_LAUNCH_BLOCKING=1 (the faulting call returns the error) and varena's mechanism ops
    logged (RUST_LOG=debug): both fillers, then one. Then the pool free's ordering, in a child
    of its own (its comparison with a bare cuMemFree may fault the context)."""
    work = Path(tempfile.mkdtemp(prefix="varena-mempool-"))
    dest = work / "unet.cozytensors"
    build(work / "store", dest)
    out = Path("out")  # the rental script's output tree
    for label, fillers in (("both-fillers", 2), ("one-filler", 1)):
        log = work / f"{label}.stderr.log"
        env = {**os.environ, "CUDA_LAUNCH_BLOCKING": "1", "RUST_LOG": "debug"}
        began = time.perf_counter()
        with log.open("w") as err, subprocess.Popen([sys.executable, "-u", __file__, "variant", str(fillers), str(dest), str(work / "store")],
                                                    env=env, stdout=subprocess.PIPE, stderr=err, text=True) as child:
            for line in child.stdout or ():
                print(f"[{label}] {line}", end="", flush=True)
        print(f"DIAG {label}: exit {child.returncode} after {time.perf_counter() - began:.0f} s", flush=True)
        summarize(label, log)
        with log.open("rb") as raw, gzip.open(out / f"{label}.stderr.log.gz", "wb") as packed:
            shutil.copyfileobj(raw, packed)
    done = subprocess.run([sys.executable, "-u", __file__, "free"], capture_output=True, text=True)
    print(done.stdout + done.stderr[-4000:], flush=True)
    return 0


def summarize(label: str, log: Path) -> None:
    """The first fault in a variant's stderr, the mechanism ops before it, and its page's history."""
    lines = log.read_text(errors="replace").splitlines()
    ops = [n for n, line in enumerate(lines) if "varena_diag" in line]
    first = next((n for n, line in enumerate(lines) if "ILLEGAL_ADDRESS" in line or "illegal memory access" in line), None)
    print(f"DIAG {label}: {len(lines)} stderr lines, {len(ops)} mechanism ops logged, first fault at line {first}", flush=True)
    python = [line for line in lines if not line[:4].isdigit()]
    if first is None:
        print(f"DIAG {label}: no fault; stderr's last Python lines:", *python[-10:], sep="\n  ", flush=True)
        return
    print(f"DIAG {label}: the 80 lines before the first fault and 20 after:", *(line[:400] for line in lines[max(0, first - 80):first + 20]), sep="\n  ", flush=True)
    key = re.search(r"\((\d+), (\d+)\)", lines[first])
    if key is not None:
        page = f"({key[1]}, {key[2]})"
        history = [line[:400] for line in lines[:first + 1] if page in line]
        print(f"DIAG {label}: page {page}'s last 40 ops:", *history[-40:], sep="\n  ", flush=True)
    print(f"DIAG {label}: stderr's Python lines (the faulting call's traceback):", *python[-60:], sep="\n  ", flush=True)


def variant(fillers: int, ck: tensorfs.Checkpoint) -> int:
    """gpu11's sequence: case 0 (with `fillers` fillers), case 1, then case 2's interleaved
    default and pool runs, three iterations; the first error ends it."""
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    mem = Memory.open(0, limit=CAP, pool=True)
    assert mem.pool is not None
    print(f"POOL use count outside a pass {mem.pool.use_count()}; torch {torch.__version__}", flush=True)
    run = Run(mem, ck)
    inside: list[int] = []
    hook = run.unet.conv_in.register_forward_pre_hook(lambda *_: inside.append(mem.pool.use_count()) if mem.pooling else None)  # type: ignore[union-attr]
    shortfalls(mem, run, ck, fillers)
    run.fill()
    times, _, _, raised = run.steps("pool beside the filler", True)
    print(f"CASE 1: ms {[round(t) for t in times]} raised {raised!r}; allocator {allocator_stats()}", flush=True)
    for i in range(3):
        for pool in (False, True):
            run.fill()
            frees = allocator_stats()[6]
            times, _, _, raised = run.steps("pool" if pool else "default", pool)
            print(f"ITERATION {i + 1} {'pool' if pool else 'default'}: ms {[round(t) for t in times]} raised {raised!r}; "
                  f"segments freed {allocator_stats()[6] - frees}; allocator {allocator_stats()}; cross-stream block runs {mem.crossed}; "
                  f"pool use counts in its passes {sorted(set(inside))}", flush=True)
            if raised is not None:
                return 1
    hook.remove()
    print(f"POOL use count outside a pass {mem.pool.use_count()}", flush=True)
    return 0


def free() -> int:
    """The pool's free waits for the work queued on its segment's stream: a 64 MiB segment
    from varena's allocator, a sleep then a memset into it queued, then varena_torch_free; then
    the same with the driver's bare cuMemFree, for comparison (last: it may fault)."""
    lib, driver = ctypes.CDLL(_ext.__file__), ctypes.CDLL("libcuda.so.1")
    lib.varena_torch_alloc.restype, lib.varena_torch_alloc.argtypes = ctypes.c_void_p, (ctypes.c_ssize_t, ctypes.c_int, ctypes.c_void_p)
    lib.varena_torch_free.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)
    driver.cuMemsetD8Async.argtypes = (ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t, ctypes.c_void_p)
    driver.cuMemAlloc_v2.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t)
    driver.cuMemFree_v2.argtypes = (ctypes.c_uint64,)
    torch.zeros(1, device="cuda")
    stream, size = torch.cuda.current_stream(), 64 << 20
    for name in ("varena_torch_free", "cuMemFree"):
        if name == "varena_torch_free":
            ptr = lib.varena_torch_alloc(size, 0, stream.cuda_stream)
        else:
            out = ctypes.c_uint64()
            driver.cuMemAlloc_v2(ctypes.byref(out), size)
            ptr = out.value
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(stream)
        torch.cuda._sleep(1 << 29)
        rc = driver.cuMemsetD8Async(ptr, 1, size, ctypes.c_void_p(stream.cuda_stream))
        end.record(stream)
        began = time.perf_counter()
        freed = lib.varena_torch_free(ptr, size, 0, stream.cuda_stream) if name == "varena_torch_free" else driver.cuMemFree_v2(ptr)
        waited, idle = 1e3 * (time.perf_counter() - began), stream.query()
        try:
            torch.cuda.synchronize()
            queue, fault = f"{start.elapsed_time(end):.0f} ms", None
        except RuntimeError as e:
            queue, fault = "unknown", str(e).splitlines()[0]
        print(f"FREE {name}: memset queued rc {rc}, free rc {freed}; the free returned after {waited:.0f} ms for a {queue} queue; "
              f"stream idle when it returned: {idle}; fault after: {fault}", flush=True)
        if fault is not None:
            return 1
    print(f"FREE allocator {allocator_stats()}", flush=True)
    return 0


if __name__ == "__main__":
    # SPIKE DIAG: with no argument (as the rental script runs it) the diagnostic; `main` for the cases.
    mode = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if mode in ("main", "variant"):
        threading.Thread(target=watch, daemon=True).start()
    if mode == "variant":
        sys.exit(variant(int(sys.argv[2]), tensorfs.open(sys.argv[3], cache_dir=sys.argv[4], local_files_only=True)))
    sys.exit({"diag": diag, "free": free, "main": main}[mode]())
