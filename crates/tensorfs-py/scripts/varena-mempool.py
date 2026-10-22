#!/usr/bin/env python3
"""SPIKE: torch's OOM retried inside its allocator. An SDXL-sized bf16 UNet at 4 MP under a
12 GiB standalone limit, beside another set big enough that its passes need evictions, with each
pass allocating from a `torch.cuda.MemPool` whose segments come from varena's allocator (a
shortfall asks varena, and torch only ever sees a segment or, when varena cannot make room, its
OutOfMemoryError). Against the same run on torch's default allocator: warm ms per step, the
allocator's cost per new segment, bytes outside the pool, and a forced exhaustion.

    python scripts/varena-mempool.py
GPU 0 only.
"""

from __future__ import annotations

import gc
import sys
import tempfile
import time
import weakref
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
STEPS = 4  # the first runs unmeasured; the rest are warm
CAP = 12 << 30  # the vertical slice's card size
results: list[bool] = []


def check(name: str, ok: object, detail: object = "") -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


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


def inputs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 4, LATENT, LATENT, generator=g).to(device, torch.bfloat16)
    text = torch.randn(1, 77, 2048, generator=g).to(device, torch.bfloat16)
    cond = {"text_embeds": torch.randn(1, 1280, generator=g).to(device, torch.bfloat16),
            "time_ids": torch.tensor([[2048, 2048, 0, 0, 2048, 2048]], dtype=torch.bfloat16, device=device)}
    return x, text, cond


def phase(ck: tensorfs.Checkpoint, label: str, pool: bool) -> tuple[float, weakref.ref[Memory]]:
    """The filler set runs once (its pages stay, idle), then the model runs STEPS passes beside
    it under the cap: its room comes from evicting the filler. Returns the warm mean ms."""
    mem = Memory.open(0, limit=CAP, pool=pool)
    filler, unet = loaded(mem, ck, "filler"), loaded(mem, ck, "")
    x, text, cond = inputs(mem.device)
    with mem.request(f"{label} filler", plan=[filler]), torch.no_grad():
        filler(x[:, :, :64, :64], 999, encoder_hidden_states=text, added_cond_kwargs=cond)
    before = allocator_stats()
    times: list[float] = []
    raised: list[BaseException] = []
    with mem.request(label, plan=[unet] * STEPS) as r, torch.no_grad():
        for n in range(STEPS):
            torch.cuda.synchronize()
            began = time.perf_counter()
            try:
                unet(x, 999 - n, encoder_hidden_states=text, added_cond_kwargs=cond)
            except (torch.OutOfMemoryError, VarenaError) as e:
                raised.append(e)
                break
            torch.cuda.synchronize()
            times.append(1e3 * (time.perf_counter() - began))
    after = allocator_stats()
    stats = r.stats()
    segments, resolved, ns = after[1] - before[1], after[2] - before[2], after[4] - before[4]
    warm = sum(times[1:]) / max(len(times[1:]), 1)
    print(f"PHASE {label}: ms/step {[round(t) for t in times]}, warm mean {warm:.0f} ms; pool segments {segments}, "
          f"shortfalls resolved in the allocator {resolved}, {ns / max(segments, 1) / 1e3:.0f} us per new segment; "
          f"outside the pool at most {mem.outside} B; evicted {stats.evicted}; retries {stats.retries} cuts {stats.cuts} parks {stats.parks}; churn {r.churn}", flush=True)
    check(f"{label}: completes with no OOM raised to Python", not raised and len(times) == STEPS, raised)
    if pool:
        check(f"{label}: its passes' segments came from the pool", segments > 0, after)
        check(f"{label}: the filler's pages made room", sum(stats.evicted) > 0 or resolved > 0, (stats.evicted, resolved))
        exhaustion(mem, unet, x, text, cond)
    return warm, weakref.ref(mem)


def exhaustion(mem: Memory, unet: UNet2DConditionModel, x: torch.Tensor, text: torch.Tensor, cond: dict[str, torch.Tensor]) -> None:
    """A buffer larger than anything varena can make room for surfaces as torch's
    OutOfMemoryError at its call site, and the next pass runs."""
    refused = allocator_stats()[3]
    hook_raised: list[BaseException] = []

    def huge(*_: object) -> None:
        try:
            torch.empty(2 * CAP, dtype=torch.uint8, device=mem.device)
        except torch.OutOfMemoryError as e:
            hook_raised.append(e)

    hook = unet.conv_in.register_forward_pre_hook(huge)
    with mem.request("exhaustion", plan=[unet]), torch.no_grad():
        unet(x, 1, encoder_hidden_states=text, added_cond_kwargs=cond)
    hook.remove()
    check("exhaustion: torch's OutOfMemoryError at the call site, after varena refused", len(hook_raised) == 1 and allocator_stats()[3] > refused, (hook_raised, allocator_stats()))
    with mem.request("after", plan=[unet]), torch.no_grad():
        unet(x, 0, encoder_hidden_states=text, added_cond_kwargs=cond)
    check("exhaustion: the next pass runs", True)


def main() -> int:
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        print("needs a GPU")
        return 1
    ck = build(Path(tempfile.mkdtemp(prefix="varena-mempool-")) / "store")
    warm: dict[str, float] = {}
    for label, pool in (("default allocator", False), ("varena pool", True)):
        warm[label], memory = phase(ck, label, pool)
        gc.collect()
        torch.cuda.empty_cache()
        check(f"{label}: its memory is gone before the next phase", memory() is None)
    base, pooled = warm["default allocator"], warm["varena pool"]
    print(f"WARM default {base:.0f} ms, pool {pooled:.0f} ms ({100 * (pooled / base - 1):+.1f}%)", flush=True)
    print(f"{sum(results)}/{len(results)} checks passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
