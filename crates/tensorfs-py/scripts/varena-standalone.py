#!/usr/bin/env python3
"""varena on its own, with no TensorD (memory-varena §2 "Standalone"), under a Diffusers model.

A tiny `UNet2DConditionModel` is written into a TensorFS store; the same model built on meta is
loaded through `Memory.load(model, ck=...)` (its blocks from `_no_split_modules`, its bytes from
its chunks, read by TensorFS) and run under `with mem.request(...)`, then compared with the plain
model. Again with `limit` leaving room for common and half its blocks: on a GPU it streams.

    python scripts/varena-standalone.py
Runs with any python that has torch, diffusers and tensorfs; on the CPU when no GPU is visible.
"""

import sys
import tempfile
from pathlib import Path

import tensorfs
import torch
from diffusers import UNet2DConditionModel
from tensorfs.varena.torch import Exhausted, Memory, blocks_of

CONFIG = dict(  # blocks of 16 MiB and more: a smaller one shares a pack, so its region is common
    sample_size=8,
    in_channels=4,
    out_channels=4,
    block_out_channels=(320, 640),
    layers_per_block=1,
    cross_attention_dim=768,
    attention_head_dim=8,
    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
    norm_num_groups=8,
)
results: list[bool] = []


def check(name: str, ok: object, detail: object = "") -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


def run(device: torch.device, ck: tensorfs.Checkpoint, want: torch.Tensor, limit: int | None) -> tuple[int, int]:
    """The model loaded from the store under `limit`, checked; its common and block bytes."""
    with torch.device("meta"):
        unet = UNet2DConditionModel(**CONFIG).eval()
    mem = Memory.open(device, limit=limit)
    model = mem.load(unet, ck=ck)
    x, text = (torch.randn(*s, generator=torch.Generator().manual_seed(1)) for s in ((1, 4, 8, 8), (1, 5, 768)))
    with torch.no_grad(), mem.request(f"limit {limit}") as r:
        got = unet(x.to(device), 3, encoder_hidden_states=text.to(device)).sample.cpu()
    name = f"loaded from the store, limit {limit}"
    if limit is not None:  # the CUDA context is counted: measured from the driver each request
        check(f"{name}: outside bytes are measured", mem.context > 0, mem.context)
    owners = {p for p in blocks_of(unet) if any(k.startswith(p + ".") for k in ck.tensors["unet"])}
    check(  # its small blocks share packs with their neighbours: each still owns its regions
        f"{name}: every block owns its regions",
        len(model.blocks) == len(owners),
        f"{len(model.blocks)} of {len(owners)} blocks, common {model.common.nbytes}",
    )
    # a capped torch picks other cuDNN algorithms: F32 rounding, not a wrong weight
    check(f"{name}: equals the plain model", torch.allclose(got, want, atol=1e-5), (got - want).abs().max())
    check(f"{name}: leaves nothing behind", all(n == 0 for _, n in r.stats.leftover), r.stats.leftover)
    if limit is not None:
        streamed = sum(s.streamed for s in r.stats.steps)
        check(f"{name}: streams", streamed > 0 and model.home_bytes() <= limit, streamed)
    sizes = model.common.nbytes, sum(b.nbytes for b in model.blocks)
    model.close()
    return sizes


def main() -> int:
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.allow_tf32 = False  # TF32 convolutions differ by cuDNN algorithm
    torch.manual_seed(0)
    plain = UNet2DConditionModel(**CONFIG).eval()
    w = tensorfs.Writer(cache_dir=Path(tempfile.mkdtemp(prefix="varena-standalone-")) / "store")
    unet = w.component("unet", cls="diffusers.UNet2DConditionModel", config=CONFIG)
    for key, t in plain.state_dict().items():
        t = t.detach().contiguous()
        unet.put(key, "F32", list(t.shape), t.view(torch.uint8).numpy())
    ck = w.commit()
    x, text = (torch.randn(*s, generator=torch.Generator().manual_seed(1)) for s in ((1, 4, 8, 8), (1, 5, 768)))
    plain.to(device)
    with torch.no_grad():
        want = plain(x.to(device), 3, encoder_hidden_states=text.to(device)).sample.cpu()
    plain.cpu()
    common, blocks = run(device, ck, want, None)
    if device.type == "cuda":  # what the process holds, working room, common and half the blocks
        torch.cuda.empty_cache()
        used = lambda: (lambda f, t: t - f)(*torch.cuda.mem_get_info(device))  # noqa: E731
        run(device, ck, want, used() + (16 << 20) + common + blocks // 2)
        torch.cuda.empty_cache()  # common never fits: the bind is short, it does not spin
        with torch.device("meta"):
            unet = UNet2DConditionModel(**CONFIG).eval()
        mem = Memory.open(device, limit=used() + common // 2)
        model = mem.load(unet, ck=ck)
        x, text = (torch.randn(*s, device=device) for s in ((1, 4, 8, 8), (1, 5, 768)))
        try:
            with torch.no_grad(), mem.request("never fits"):
                unet(x, 3, encoder_hidden_states=text)
            check("a model that never fits raises Exhausted", False, "it ran")
        except Exhausted as exc:
            check("a model that never fits raises Exhausted", exc.need > 0, exc)
        model.close()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
