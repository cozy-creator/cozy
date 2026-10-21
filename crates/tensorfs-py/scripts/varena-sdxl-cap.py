#!/usr/bin/env python3
"""An SDXL-sized UNet (bf16, random weights) at 4 MP (a 256x256 latent) through varena, uncapped
and then under a 12 GiB standalone limit, with each pass's time and torch's allocator churn:
retries after freeing its cache, device allocations and frees, and their stream syncs. A pass
that got its room must not churn against the cap varena sets.

    python scripts/varena-sdxl-cap.py
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


def phase(ck: tensorfs.Checkpoint, label: str, limit: int | None) -> weakref.ref[Memory]:
    mem = Memory.open(0, limit=limit)
    with torch.device("meta"):
        unet = UNet2DConditionModel(**CONFIG).to(torch.bfloat16).eval()
    vars(unet)["_varena"] = Declared(Weights(ck, "unet"))
    mem.load(unet, "gpu")
    model = mem.models[unet]
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 4, LATENT, LATENT, generator=g).to(mem.device, torch.bfloat16)
    text = torch.randn(1, 77, 2048, generator=g).to(mem.device, torch.bfloat16)
    cond = {"text_embeds": torch.randn(1, 1280, generator=g).to(mem.device, torch.bfloat16),
            "time_ids": torch.tensor([[2048, 2048, 0, 0, 2048, 2048]], dtype=torch.bfloat16, device=mem.device)}
    times, churn = [], []
    with mem.request(label, plan=[unet] * STEPS) as r, torch.no_grad():
        for n in range(STEPS):
            torch.cuda.synchronize()
            began = time.perf_counter()
            unet(x, 999 - n, encoder_hidden_states=text, added_cond_kwargs=cond)
            torch.cuda.synchronize()
            times.append(1e3 * (time.perf_counter() - began))
            churn.append(model.last[3] if model.last else {})
    stats = r.stats()
    warm = churn[1:]
    print(f"PHASE {label}: limit {limit}, cap {mem.capped}, ms/step {[round(t) for t in times]}, warm mean {sum(times[1:]) / len(times[1:]):.0f} ms", flush=True)
    for n, c in enumerate(churn):
        print(f"  pass {n}: churn {c}; last {model.last[:3] if model.last else None}", flush=True)
    print(f"  request churn {r.churn}; steps {[(s.capacity, s.w) for s in stats.steps]}; evicted {stats.evicted}; retries {stats.retries} cuts {stats.cuts} parks {stats.parks}", flush=True)
    check(f"{label}: warm passes never retry an allocation", all(c.get("num_alloc_retries", 0) == 0 for c in warm), warm)
    check(f"{label}: warm passes never sync to free the cache", all(c.get("num_sync_all_streams", 0) == 0 for c in warm), warm)
    check(f"{label}: nothing left behind", stats.leftover == 0, stats.leftover)
    return weakref.ref(mem)


def main() -> int:
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        print("needs a GPU")
        return 1
    started = time.perf_counter()
    ck = build(Path(tempfile.mkdtemp(prefix="varena-sdxl-")) / "store")
    print(f"BUILT the checkpoint in {time.perf_counter() - started:.0f} s", flush=True)
    for label, limit in (("uncapped", None), ("capped 12 GiB", CAP)):
        memory = phase(ck, label, limit)
        gc.collect()
        torch.cuda.empty_cache()
        check(f"{label}: its memory is gone before the next phase", memory() is None)
    print(f"{sum(results)}/{len(results)} checks passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
