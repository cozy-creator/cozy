#!/usr/bin/env python3
"""varena on its own, with no TensorD, under a Diffusers model and plain torch.

A `UNet2DConditionModel` (305 MiB) is written into a TensorFS store; the same model built on
meta is loaded through `tensorfs.varena.torch.Memory` (blocks from `_no_split_modules`) and run
under `mem.request`:
  - with room for it all, against the plain model; held bound (a CUDA graph's hold); a direct
    child's forward is its own pass, keyed by its entry;
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
import gc
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
#: The same device's kernels agree to F32 rounding; the CPU and a GPU differ by more.
SAME, ACROSS = 1e-5, 1e-3
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
    return torch.randn(1, 4, 8, 8, generator=g).to(device), torch.randn(1, 5, 768, generator=g).to(device)


def forward(unet: UNet2DConditionModel, device: torch.device, n: int) -> torch.Tensor:
    x, text = inputs(device, n)
    with torch.no_grad():
        return unet(x, 3, encoder_hidden_states=text).sample.cpu()


class Double:
    """A declared fill, as a LoRA bake is: doubles each attention query weight as its page arrives."""

    identity = "double to_q"

    def __call__(self, block: str, parts: dict[str, dict[str, torch.Tensor]]) -> None:
        for key, roles in parts.items():
            if key.endswith("to_q.weight"):
                roles["value"].mul_(2)


def loaded(ck: tensorfs.Checkpoint, mem: Memory, fill: Double | None = None) -> UNet2DConditionModel:
    with torch.device("meta"):
        unet = UNet2DConditionModel(**CONFIG).eval()
    vars(unet)["_varena"] = Declared(Weights(ck, "unet"), fill=fill)
    mem.load(unet, "gpu")
    return unet


def run(mem: Memory, unet: UNet2DConditionModel, want: list[torch.Tensor], id: str, atol: float) -> tuple[native.Stats, list[torch.Tensor]]:
    """A forward of `unet` per expected output, in one request."""
    got = []
    with mem.request(id, plan=[unet] * len(want)) as r:
        for n, expected in enumerate(want):
            got.append(forward(unet, mem.device, n))
            diff = (got[n] - expected).abs().max().item()
            check(f"{id}: pass {n} equals its reference (max diff {diff:.2e})", diff <= atol, diff)
    stats = r.stats()
    check(f"{id}: leaves nothing behind", stats.leftover == 0, stats.leftover)
    check(f"{id}: its record crosses as JSON", json.loads(stats.json())["request"] == id, stats.json())
    return stats, got


def fills(stats: native.Stats, ceiling: tensorfs.Ceiling) -> None:
    """Each source's fills as GB/s and as % of the ceiling they run against: H2D for a GPU's
    fills (and the disk too for a store read); on the host only a store read has one."""
    for source in ("kept", "tier", "store"):
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
    """The second process: its request needs the room the idle first one holds."""
    _, ck = build(Path(store))
    mem = opened(table, capacity)
    stats, _ = run(mem, loaded(ck, mem), torch.load(reference), "child", SAME)
    print(json.dumps({"child origins": [r.origin for r in stats.trace], "evicted": stats.evicted}), flush=True)
    return 0 if all(results) else 1


def main() -> int:
    stub_matches()
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    store = Path(tempfile.mkdtemp(prefix="varena-standalone-")) / "store"
    plain, ck = build(store)
    tag = f"varena-standalone-{os.getpid()}"
    expected = [forward(plain, torch.device("cpu"), n) for n in range(2)]

    mem = Memory.open(0) if CUDA else opened(f"{tag}-ample", 1 << 31)
    # A GPU's capacity counts the whole card: this process's context and anything foreign.
    base = (lambda free, total: total - free)(*torch.cuda.mem_get_info(0)) if CUDA else 0
    unet = loaded(ck, mem)
    stats, reference = run(mem, unet, expected, "ample", ACROSS if CUDA else SAME)
    model = mem.models[unet]
    with model.hold():
        weight = unet.down_blocks[0].attentions[0].proj_in.weight
        check("hold: every block stays bound", not isinstance(weight, Placeholder), type(weight))
        run(mem, unet, reference[:1], "held", SAME)
    check("hold: released, placeholders are back", isinstance(unet.down_blocks[0].attentions[0].proj_in.weight, Placeholder))
    keys: list[str] = []
    probe = unet.conv_in.register_forward_pre_hook(lambda *_: keys.append(adapter._pass.get().shape))
    with mem.request("entries", plan=[unet, unet]):
        x, _ = inputs(mem.device, 0)
        with torch.no_grad():
            got = unet.conv_in(x).cpu()
        forward(unet, mem.device, 0)
    probe.remove()
    diff = (got - plain.conv_in(x.cpu()).detach()).abs().max().item()
    check(f"entries: a direct child's forward runs bound (max diff {diff:.2e})", diff <= ACROSS, diff)
    check("entries: each entry is its own pass key", len(keys) == 2 and keys[0].startswith("conv_in|") and keys[1].startswith("|"), keys)
    spans = model.set.spans
    room = stats.peak - sum(spans)  # the passes' measured room beside the weights
    common, blocks = spans[0], sorted(spans[1:])
    del mem, unet, model
    gc.collect()
    ceiling = tensorfs.ceiling(device=0 if CUDA else None, path=store)

    # Capacities only fall on one GPU's table: the largest first.
    table, capacity = f"{tag}-two", base * (2 if CUDA else 1) + sum(spans) * 3 // 2 + room  # a GPU child's context is about ours
    saved = store.parent / "reference.pt"
    torch.save(reference[:1], saved)
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
    del mem, unet
    gc.collect()

    capacity = base + common + sum(blocks) // 2 + room
    mem = opened(f"{tag}-short", capacity)
    stats, _ = run(mem, loaded(ck, mem), reference, "short", SAME)
    check(f"short: below its {sum(spans)} B it streams in {capacity - base} B", sum(stats.evicted) > 0, stats.evicted)
    fills(stats, ceiling)
    print(f"STATS_JSON {stats.json()}", flush=True)

    mem = opened(f"{tag}-fill", capacity)
    doubled = UNet2DConditionModel(**CONFIG).eval()
    doubled.load_state_dict({k: v * 2 if k.endswith("to_q.weight") else v for k, v in plain.state_dict().items()})
    want = [forward(doubled, torch.device("cpu"), n) for n in range(2)]
    stats, _ = run(mem, loaded(ck, mem, Double()), want, "fill", ACROSS if CUDA else SAME)
    check("fill: each fresh page is filled once", sum(r.origin != "hit" for r in stats.trace) > 0, stats.trace)

    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(child(*sys.argv[2:4], int(sys.argv[4]), sys.argv[5]) if sys.argv[1:2] == ["child"] else main())
