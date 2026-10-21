#!/usr/bin/env python3
"""varena on its own, with no TensorD, under a Diffusers model and plain torch.

A tiny `UNet2DConditionModel` is written into a TensorFS store; the same model built on meta is
loaded through `tensorfs.varena.torch.Memory` (blocks from `_no_split_modules`) and run under
`mem.request`, then compared with the plain model:
  - with room for it all;
  - with a capacity below its size, so it streams;
  - in two processes on one table, the idle one making room for the other.
First, the compiled `tensorfs._ext.varena` matches its stub name for name.

    python scripts/varena-standalone.py
Runs on the CPU over host pages (each run names its own tables); on CUDA when a GPU is visible.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import tensorfs
import torch
from diffusers import UNet2DConditionModel
from tensorfs._ext import varena as native
from tensorfs.varena import GpuId
from tensorfs.varena.torch import Declared, Memory, Weights

CONFIG = dict(
    sample_size=8,
    in_channels=4,
    out_channels=4,
    block_out_channels=(32, 64),
    layers_per_block=1,
    cross_attention_dim=32,
    attention_head_dim=8,
    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
    norm_num_groups=8,
)
results: list[bool] = []


def check(name: str, ok: object, detail: object = "") -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


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
    return torch.randn(1, 4, 8, 8, generator=g).to(device), torch.randn(1, 5, 32, generator=g).to(device)


def forward(unet: UNet2DConditionModel, device: torch.device, n: int) -> torch.Tensor:
    x, text = inputs(device, n)
    with torch.no_grad():
        return unet(x, 3, encoder_hidden_states=text).sample.cpu()


def loaded(ck: tensorfs.Checkpoint, mem: Memory) -> UNet2DConditionModel:
    with torch.device("meta"):
        unet = UNet2DConditionModel(**CONFIG).eval()
    vars(unet)["_varena"] = Declared(Weights(ck, "unet"))
    mem.load(unet, "gpu")
    return unet


def run(mem: Memory, unet: UNet2DConditionModel, plain: UNet2DConditionModel, id: str, passes: int) -> native.Stats:
    """`passes` forwards of `unet` in one request, each against the plain model."""
    with mem.request(id, plan=[unet] * passes) as r:
        for n in range(passes):
            got, want = forward(unet, mem.device, n), forward(plain, torch.device("cpu"), n)
            check(f"{id}: pass {n} equals the plain model", torch.allclose(got, want, atol=1e-5), (got - want).abs().max())
    stats = r.stats()
    check(f"{id}: leaves nothing behind", stats.leftover == 0, stats.leftover)
    return stats


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


def child(store: str, table: str, capacity: int) -> int:
    """The second process: its request needs the room the idle first one holds."""
    plain, ck = build(Path(store))
    mem = Memory.open(GpuId(table, "cpu", capacity))
    stats = run(mem, loaded(ck, mem), plain, "child", 1)
    print(json.dumps({"evicted": stats.evicted}), flush=True)
    return 0 if all(results) else 1


def main() -> int:
    stub_matches()
    store = Path(tempfile.mkdtemp(prefix="varena-standalone-")) / "store"
    plain, ck = build(store)
    tag = f"varena-standalone-{os.getpid()}"
    cuda = torch.cuda.is_available()

    mem = Memory.open(0) if cuda else Memory.open(GpuId(f"{tag}-ample", "cpu", 1 << 30))
    unet = loaded(ck, mem)
    run(mem, unet, plain, "ample", 2)
    spans = mem.models[unet].set.spans
    common, blocks = spans[0], sorted(spans[1:])
    if cuda:  # a standalone CUDA table has the card's capacity: the rest needs host pages
        print(f"{sum(results)}/{len(results)} checks passed")
        return 0 if all(results) else 1

    capacity = common + sum(blocks) // 2
    mem = Memory.open(GpuId(f"{tag}-short", "cpu", capacity))
    stats = run(mem, loaded(ck, mem), plain, "short", 2)
    check("short: it streams", sum(stats.evicted) > 0, stats.evicted)

    capacity = sum(spans) * 3 // 2
    table = f"{tag}-two"
    mem = Memory.open(GpuId(table, "cpu", capacity))
    unet = loaded(ck, mem)
    run(mem, unet, plain, "first", 1)
    done = subprocess.run([sys.executable, __file__, "child", str(store), table, str(capacity)], capture_output=True, text=True)
    print(done.stdout, done.stderr, sep="", end="", flush=True)
    check("two processes: the second completes beside the idle first", done.returncode == 0, done.returncode)
    stats = run(mem, unet, plain, "first again", 1)
    check("two processes: the first gave room and reads again", any(r.origin != "hit" for r in stats.trace), [r.origin for r in stats.trace])

    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(child(*sys.argv[2:4], int(sys.argv[4])) if sys.argv[1:2] == ["child"] else main())
