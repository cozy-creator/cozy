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
  6. no deadlock while a second thread allocates on the device as varena evicts.

    python scripts/varena-mempool.py
GPU 0 only.
"""

from __future__ import annotations

import faulthandler
import sys
import tempfile
import threading
import time
from pathlib import Path

import tensorfs
import torch
from diffusers import UNet2DConditionModel
from tensorfs.varena import VarenaError, allocator_stats
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


def build(store: Path) -> tensorfs.Checkpoint:
    """SDXL's UNet shapes in bf16, random (std 0.02): its bytes, not its pictures, matter here."""
    with torch.device("meta"):
        shapes = UNet2DConditionModel(**CONFIG).state_dict()
    w = tensorfs.Writer(cache_dir=store)
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

    # 1. beside the filler, through the pool: shortfalls resolved in the allocator.
    run.fill()
    before = allocator_stats()
    times, segments, r, raised = run.steps("pool beside the filler", True)
    after = allocator_stats()
    stats = r.stats()
    resolved, nbytes = after[2] - before[2], after[3] - before[3]
    print(f"POOL beside the filler: ms/step {[round(t) for t in times]}; shortfalls resolved in the allocator {resolved} ({nbytes} B); "
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
    with mem.request("exhaustion", plan=[run.unet]), torch.no_grad():
        run.unet(run.x, 1, encoder_hidden_states=run.text, added_cond_kwargs=run.cond)
    hook.remove()
    check("4: a forced exhaustion is torch's OutOfMemoryError at its call site, varena having refused", len(caught) == 1 and allocator_stats()[4] > refused, (caught, allocator_stats()))
    times, _, _, raised = run.steps("after exhaustion", True)
    check("4: the next passes run", raised is None and len(times) == STEPS, raised)

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
    print(f"{sum(results)}/{len(results)} checks passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    threading.Thread(target=watch, daemon=True).start()
    sys.exit(main())
