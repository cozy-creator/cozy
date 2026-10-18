#!/usr/bin/env python3
"""`tensorfs.varena.torch` under a real torch model on one GPU (xs-046 B2).

A small block model (embed, N blocks, head; `_no_split_modules`) lives in a real TensorFS store
and runs through `Memory`. Every check compares against the same model built from ordinary
tensors, bit for bit:

- cold, then warm: the warm pass copies nothing (sticky);
- a block's params are placeholders outside its forward, and an op on one binds it for that op;
- the watermark falls inside a request (`room`), the streamed passes stay exact, and it is
  restored after;
- a view a hook keeps is its page's pin: eviction leaves it;
- a failed request leaves no bytes; two threads run one model at once;
- a hold keeps every page for a CUDA graph; `fill` runs once per page-in; `bind` makes the
  module's tensors; `Exhausted` leaves the memory usable;
- the ledger row carries the facts.

    flock <locks>/gpu.lock python scripts/varena-torch.py [workdir] [--blocks N] [--dim D]
Runs with any python that has torch (CUDA) and `PYTHONPATH=<worktree>/python`.
"""

import argparse
import sys
import tempfile
import threading
from pathlib import Path

import torch
from torch import nn

import tensorfs
from tensorfs.varena import Ledger
from tensorfs.varena.torch import (
    Exhausted,
    LedgerGrant,
    Memory,
    Placeholder,
    WeightGrant,
)

results: list[bool] = []


def check(name: str, ok: object, detail: str = "") -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


class Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, dtype=torch.bfloat16)
        self.lin = nn.Linear(dim, dim, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin(self.norm(x))


class Net(nn.Module):
    _no_split_modules = ["Block"]

    def __init__(self, dim: int, blocks: int) -> None:
        super().__init__()
        self.embed = nn.Linear(dim, dim, dtype=torch.bfloat16)
        self.blocks = nn.ModuleList(Block(dim) for _ in range(blocks))
        self.head = nn.Linear(dim, dim, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def build(root: Path, dim: int, blocks: int) -> tuple[tensorfs.Checkpoint, dict[str, torch.Tensor]]:
    """The model's weights in a v2 store, in construction order; and the tensors."""
    torch.manual_seed(0)
    reference = Net(dim, blocks)
    tensors = {k: v.detach().contiguous() for k, v in reference.state_dict().items()}
    store = tensorfs.Store(root)
    w = store.writer()
    for key, t in tensors.items():
        w.put("net", key, "BF16", list(t.shape), t.view(torch.uint8).numpy().tobytes())
    return store.checkpoint(w.commit()), tensors


def meta(dim: int, blocks: int) -> Net:
    with torch.device("meta"):
        return Net(dim, blocks)


def reference(tensors: dict[str, torch.Tensor], dim: int, blocks: int, dev: torch.device) -> Net:
    net = Net(dim, blocks).to(dev)
    net.load_state_dict({k: v.to(dev) for k, v in tensors.items()})
    return net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir", nargs="?")
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    root = Path(a.workdir or tempfile.mkdtemp(prefix="varena-torch-")) / "store"
    ck, tensors = build(root, a.dim, a.blocks)
    dev = torch.device("cuda", 0)
    torch.cuda.init()
    table = Ledger.create(
        str(torch.cuda.get_device_properties(0).uuid), torch.cuda.mem_get_info(0)[1]
    )
    mem = Memory.open([0], [LedgerGrant(0, table.fd)])
    net = meta(a.dim, a.blocks)
    model = mem.register("net", net, WeightGrant(ck, "net"), 0)
    ref = reference(tensors, a.dim, a.blocks, dev)
    x = torch.randn(a.batch, a.dim, generator=torch.Generator().manual_seed(1)).bfloat16().to(dev)
    with torch.no_grad():
        want = ref(x)
    block = model.blocks[0].nbytes

    # cold, then warm: one request, one repeating use, two passes
    with torch.no_grad(), mem.request("cold-warm", [0]) as r, mem.use([model], repeats=True):
        cold, warm = net(x), net(x)
    steps = r.stats.steps
    check(
        "cold and warm passes equal the reference",
        torch.equal(cold, want) and torch.equal(warm, want),
    )
    check(
        "the first pass copies the model home",
        steps[0].paged_in >= model.nbytes - model.common.nbytes,
        str(steps),
    )
    check(
        "the warm pass copies nothing",
        steps[-1].paged_in == 0 and steps[-1].streamed == 0,
        str(steps[-1]),
    )
    check("a request leaves nothing behind", r.stats.leftover == ((0, 0),), str(r.stats.leftover))

    # placeholders outside the forward; an op on one binds its block for that op
    w = net.blocks[3].lin.weight
    check(
        "a param outside its forward is a placeholder on the GPU",
        isinstance(w, Placeholder) and w.device == dev,
    )
    with torch.no_grad():
        read = net.blocks[3].lin.weight.float().sum()
    check(
        "an op on a placeholder binds its block",
        torch.equal(read, tensors["blocks.3.lin.weight"].to(dev).float().sum()),
    )
    check(
        "the block is a placeholder again after the op",
        isinstance(net.blocks[3].lin.weight, Placeholder),
    )

    # the watermark falls inside a request, the streamed passes stay exact, and is restored
    total = sum(b.nbytes for b in model.blocks)
    with torch.no_grad(), mem.request("lowered", [0]) as r, mem.use([model], repeats=True):
        mem.room(0, torch.cuda.mem_get_info(0)[0] + total // 2)
        lowered = model.home_bytes() - model.common.nbytes
        a1, a2 = net(x), net(x)
    check("room lowers the hot model's watermark", lowered < total, f"{lowered} of {total}")
    check("streamed passes equal the reference", torch.equal(a1, want) and torch.equal(a2, want))
    check(
        "the lowered blocks stream", sum(s.streamed for s in r.stats.steps) > 0, str(r.stats.steps)
    )
    check(
        "the eviction is a rung-4 row",
        any(e.rung == 4 for e in r.stats.evictions),
        str(r.stats.evictions),
    )
    torch.cuda.synchronize()
    check(
        "the watermark rises again after the request",
        model.home_bytes() - model.common.nbytes == total,
    )

    # a view a hook keeps is its page's pin
    kept: list[torch.Tensor] = []
    hook = net.blocks[5].lin.register_forward_hook(lambda m, i, o: kept.append(m.weight))
    with torch.no_grad(), mem.request("kept", [0]), mem.use([model]):
        net(x)
    hook.remove()
    torch.cuda.synchronize()
    model.evict()
    stats = next(s for s in mem.stats().models if s.name == "net")
    check("a kept view's page survives eviction", stats.resident >= block, str(stats))
    check(
        "the kept view still reads its weights",
        torch.equal(kept[0], tensors["blocks.5.lin.weight"].to(dev)),
    )
    kept.clear()
    model.evict()
    stats = next(s for s in mem.stats().models if s.name == "net")
    check("the page goes once the view does", stats.resident < block, str(stats))

    # a failed request frees what it allocated
    try:
        with mem.request("fails", [0]) as r:
            ballast = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
            raise RuntimeError("a package fault")
    except RuntimeError:
        pass
    del ballast
    check("a failed request leaves no bytes", r.stats.leftover == ((0, 0),), str(r.stats.leftover))

    # two threads, one model, at once
    outs: list[bool] = []

    def run() -> None:
        with (
            torch.no_grad(),
            mem.request(f"t{threading.get_ident()}", [0]),
            mem.use([model], repeats=True),
        ):
            outs.extend(torch.equal(net(x), want) for _ in range(3))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("two threads run one model at once, exactly", outs == [True] * 6, str(outs))

    # a hold keeps every page at its address for a CUDA graph
    hold = model.hold()
    graph, static = torch.cuda.CUDAGraph(), x.clone()
    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            net(static)
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph):
            out = net(static)
    model.evict()
    graph.replay()
    check("a held model's graph replays exactly after an evict", torch.equal(out, want))
    hold.close()
    del graph, out

    # fill once per page-in; bind makes the module's tensors
    fills = [0]

    def fill(blk: object, parts: dict[str, dict[str, torch.Tensor]]) -> None:
        if "blocks.1.lin.bias" in parts:
            parts["blocks.1.lin.bias"]["value"].add_(1)
            fills[0] += 1

    def bind(key: str, parts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if key == "head.weight":
            return {key: parts["value"].float().bfloat16()}  # made here, lives for the bind
        return {key: parts["value"]}

    net2 = meta(a.dim, a.blocks)
    model2 = mem.register("net-filled", net2, WeightGrant(ck, "net"), 0, fill=fill, bind=bind)
    changed = dict(tensors, **{"blocks.1.lin.bias": tensors["blocks.1.lin.bias"] + 1})
    with torch.no_grad():
        want2 = reference(changed, a.dim, a.blocks, dev)(x)
        with mem.request("filled", [0]), mem.use([model2], repeats=True):
            f1, f2 = net2(x), net2(x)
        model2.evict()
        with mem.request("refilled", [0]), mem.use([model2]):
            f3 = net2(x)
    check(
        "fill runs on page-in and bind makes the tensors",
        torch.equal(f1, want2) and torch.equal(f2, want2),
    )
    check(
        "fill runs once per page-in", fills[0] == 2 and torch.equal(f3, want2), f"{fills[0]} fills"
    )

    # Exhausted leaves the memory usable
    try:
        mem.room(0, torch.cuda.mem_get_info(0)[1] * 2)
        check("an impossible room raises Exhausted", False)
    except Exhausted as exc:
        check("an impossible room raises Exhausted", exc.gpu == 0 and exc.need > 0)
    with torch.no_grad(), mem.request("after", [0]), mem.use([model]):
        check("the memory serves after Exhausted", torch.equal(net(x), want))

    # the ledger row carries the facts
    row = next(f for f in table.snapshot() if f.state == "live")
    check(
        "the ledger row publishes resident bytes",
        row.resident > 0 and row.last_use > 0,
        str(row.resident),
    )

    model2.close()
    model.close()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
