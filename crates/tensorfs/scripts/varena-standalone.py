#!/usr/bin/env python3
"""varena on its own, with no TensorD (memory-varena §2 "Standalone"), under a Diffusers model.

A tiny `UNet2DConditionModel` is written into a TensorFS store; the same model built on meta is
loaded through `Memory.load(model, ck=...)` (its blocks from `_no_split_modules`, its bytes from
the store's chunk files) and run under `with mem.request(...)`, then compared with the plain
model. Again with `limit` leaving room for half its weights: on a GPU it streams.

    python scripts/varena-standalone.py
Runs with any python that has torch, diffusers and tensorfs; on the CPU when no GPU is visible.
"""

import sys
import tempfile
from pathlib import Path

import tensorfs
import torch
from diffusers import UNet2DConditionModel
from tensorfs.varena.torch import Memory

CONFIG = dict(
    sample_size=8,
    in_channels=4,
    out_channels=4,
    block_out_channels=(128, 256),
    layers_per_block=2,
    cross_attention_dim=256,
    attention_head_dim=8,
    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
    norm_num_groups=8,
)
results: list[bool] = []


def check(name: str, ok: object, detail: object = "") -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


def main() -> int:
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(0)
    plain = UNet2DConditionModel(**CONFIG).eval()
    store = tensorfs.Store(str(Path(tempfile.mkdtemp(prefix="varena-standalone-")) / "store"))
    w = store.writer()
    for key, t in plain.state_dict().items():
        t = t.detach().contiguous()
        w.put("unet", key, "F32", list(t.shape), t.view(torch.uint8).numpy())
    ck = store.checkpoint(w.commit())
    size = sum(t.numel() * t.element_size() for t in plain.state_dict().values())
    plain.to(device)
    x = torch.randn(1, 4, 8, 8, device=device)
    text = torch.randn(1, 5, 256, device=device)
    with torch.no_grad():
        want = plain(x, 3, encoder_hidden_states=text).sample
    plain.cpu()
    capped = None
    if device.type == "cuda":  # what torch holds now, its working room, half the weights
        torch.cuda.empty_cache()
        capped = torch.cuda.memory_reserved(device) + (16 << 20) + size // 2
    for limit in dict.fromkeys((None, capped)):  # the cap last: its torch fraction stays
        with torch.device("meta"):
            unet = UNet2DConditionModel(**CONFIG).eval()
        mem = Memory.open(device, limit=limit)
        model = mem.load(unet, ck=ck)
        with torch.no_grad(), mem.request(f"limit {limit}") as r:
            got = unet(x, 3, encoder_hidden_states=text).sample
        name = f"loaded from the store, limit {limit}"
        check(
            f"{name}: equals the plain model", torch.allclose(got, want), (got - want).abs().max()
        )
        check(f"{name}: leaves nothing behind", all(n == 0 for _, n in r.stats.leftover))
        if limit is not None and device.type == "cuda":
            streamed = sum(s.streamed for s in r.stats.steps)
            check(f"{name}: streams", streamed > 0 and model.home_bytes() <= limit, streamed)
        model.close()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
