#!/usr/bin/env python3
"""`tensorfs.varena.torch` under a real torch model on one GPU (xs-046 B2).

A small block model (embed, N blocks, head; `_no_split_modules`) lives in a real TensorFS store
and runs through `Memory`. Every check compares against the same model built from ordinary
tensors, bit for bit:

- cold, then warm: the warm pass copies nothing (sticky), and two passes make it repeat;
- a block's params are placeholders outside its forward, and an op on one binds it for that op;
- out of memory inside its forward, the watermark falls (rung 4), the streamed passes stay
  exact, and it rises after the request;
- a view a hook keeps is its page's pin: eviction leaves it;
- a failed request leaves no bytes; two threads run one model at once;
- a cache's hold keeps every page for a CUDA graph; `fill` runs once per page-in and again
  when its `prepare()` changes; `bind` makes the module's tensors; `Exhausted` leaves the
  memory usable;
- an idle (RAM-level) model's cold pages go before a warm one's; a request short of room waits
  for another's bytes; `load` seals the host tier for deposit; the row carries the facts.

    flock <locks>/gpu.lock python scripts/varena-torch.py [workdir] [--blocks N] [--dim D]
Runs with any python that has torch (CUDA) and `PYTHONPATH=<worktree>/python`.
"""

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import torch
from torch import nn

import tensorfs
from tensorfs.varena import Ledger
from tensorfs.varena.torch import (
    Declared,
    Exhausted,
    LedgerGrant,
    Level,
    Memory,
    Placeholder,
    WeightGrant,
)

MANIFEST = "sha256:" + "0" * 64

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
    # a block under 16 MiB shares the writer's packs with the next, so its region is common
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    root = Path(a.workdir or tempfile.mkdtemp(prefix="varena-torch-")) / "store"
    ck, tensors = build(root, a.dim, a.blocks)
    dev = torch.device("cuda", 0)
    torch.cuda.init()
    table = Ledger.create(
        str(torch.cuda.get_device_properties(0).uuid), torch.cuda.mem_get_info(0)[1]
    )
    mem = Memory.open(dev, LedgerGrant(0, table.fd, table.events_fd))
    net = meta(a.dim, a.blocks)
    net._varena = Declared(WeightGrant(ck, "net", MANIFEST))
    model = mem.load(net)
    sealed = mem.sealed()
    check("load seals the host tier for deposit", [m for m, _ in sealed] == [model], str(sealed))
    for _, fd in sealed:
        os.close(fd)
    ref = reference(tensors, a.dim, a.blocks, dev)
    x = torch.randn(a.batch, a.dim, generator=torch.Generator().manual_seed(1)).bfloat16().to(dev)
    with torch.no_grad():
        want = ref(x)
    block = model.blocks[0].nbytes

    def resident(name: str) -> int:
        return mem.resident()[name]

    # cold, then warm: one request, two passes
    with torch.no_grad(), mem.request("cold-warm") as r:
        cold, warm = net(x), net(x)
        same = torch.equal(cold, want) and torch.equal(warm, want)
        del cold, warm  # what outlives a request is its leftover
    steps = r.stats.steps
    check("cold and warm passes equal the reference", same)
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
    check("two passes in a request make it repeat", model.repeats == 2, str(model.repeats))
    check("a load is warm", model.level.value == "gpu", str(model.level))
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

    # out of memory inside its forward: the watermark falls, then rises after the request
    total = sum(b.nbytes for b in model.blocks)
    ballast: list[torch.Tensor] = []

    def crowd(*_: object) -> None:
        if not ballast:
            need = torch.cuda.mem_get_info(0)[0] + total // 2
            ballast.append(torch.empty(need, dtype=torch.uint8, device=dev))

    hook = net.register_forward_pre_hook(crowd)
    with torch.no_grad(), mem.request("lowered") as r:
        a1 = net(x)
        lowered = model.home_bytes() - model.common.nbytes
        hook.remove()
        a2 = net(x)
        ballast.clear()
    check("out of memory lowers the hot model's watermark", lowered < total, f"{lowered}/{total}")
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
    with torch.no_grad(), mem.request("kept"):
        net(x)
    hook.remove()
    torch.cuda.synchronize()
    model.evict()
    check("a kept view's page survives eviction", resident(model.name) >= block)
    check(
        "the kept view still reads its weights",
        torch.equal(kept[0], tensors["blocks.5.lin.weight"].to(dev)),
    )
    kept.clear()
    model.evict()
    check("the page goes once the view does", resident(model.name) < block)

    # a failed request frees what it allocated
    def fault() -> None:  # a package's frame: the request clears its locals
        junk = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
        raise RuntimeError(f"a package fault holding {junk.numel()} bytes")

    try:
        with mem.request("fails") as r:
            fault()
    except RuntimeError:
        pass
    check("a failed request leaves no bytes", r.stats.leftover == ((0, 0),), str(r.stats.leftover))

    # two threads, one model, at once
    outs: list[bool] = []

    def run() -> None:
        with torch.no_grad(), mem.request(f"t{threading.get_ident()}"):
            outs.extend(torch.equal(net(x), want) for _ in range(3))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("two threads run one model at once, exactly", outs == [True] * 6, str(outs))
    check("no forward is in flight after them", model.flight == 0)

    # a cache's hold keeps every page at its address for a CUDA graph
    pool = mem.cache("graphs:net", lambda n: 0)
    hold = pool.hold(model)
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
    pool.close()
    del graph, out

    # fill once per page-in, again when what it adds changes; bind makes the module's tensors
    class Bias:
        def __init__(self) -> None:
            self.add, self.fills = 1.0, 0

        def __call__(self, blk: object, parts: dict[str, dict[str, torch.Tensor]]) -> None:
            if "blocks.1.lin.bias" in parts:
                parts["blocks.1.lin.bias"]["value"].add_(self.add)
                self.fills += 1

        def prepare(self) -> float:
            return self.add

    def bind(key: str, parts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if key == "head.weight":
            return {key: parts["value"].float().bfloat16()}  # made here, lives for the bind
        return {key: parts["value"]}

    fill = Bias()
    net2 = meta(a.dim, a.blocks)
    net2._varena = Declared(WeightGrant(ck, "net", "sha256:" + "1" * 64), fill=fill, bind=bind)
    model2 = mem.load(net2)
    for _, fd in mem.sealed():
        os.close(fd)

    def biased(add: float) -> torch.Tensor:
        changed = dict(tensors, **{"blocks.1.lin.bias": tensors["blocks.1.lin.bias"] + add})
        return reference(changed, a.dim, a.blocks, dev)(x)

    with torch.no_grad():
        want1, want2 = biased(1.0), biased(2.0)
        with mem.request("filled"):
            f1, f2 = net2(x), net2(x)
        fills = fill.fills
        with mem.request("warm"):
            f3, f4 = net2(x), net2(x)
        warm = fill.fills
        fill.add = 2.0
        with mem.request("prepared"):
            f5 = net2(x)
    check(
        "fill runs on page-in and bind makes the tensors",
        all(torch.equal(f, want1) for f in (f1, f2, f3, f4)),
    )
    check("a resident page is not filled again", warm == fills, f"{fills} -> {warm}")
    check("a new prepare() refills the pages", torch.equal(f5, want2))

    # rung 2 takes an idle model's cold pages before a warm one's
    torch.cuda.synchronize()
    mem.load(net2, Level.RAM)
    with torch.no_grad(), mem.request("cold"):
        net(x), net2(x)
    before = resident(model.name), resident(model2.name)
    with mem.request("rung2"):
        fill_up = torch.cuda.mem_get_info(0)[0] + block
        junk = torch.empty(fill_up, dtype=torch.uint8, device=dev)
        after = resident(model.name), resident(model2.name)
        del junk
    check(
        "the idle model's cold pages go first",
        after[1] < before[1] and after[0] == before[0],
        f"{before} -> {after}",
    )

    # a request short of room waits for another's bytes, without a timer
    holding, done = threading.Event(), threading.Event()
    waited: list[int] = []

    def holder() -> None:
        with mem.request("holds"):
            junk = torch.empty(torch.cuda.mem_get_info(0)[0] - block, dtype=torch.uint8, device=dev)
            holding.set()
            time.sleep(2)  # the waiter runs out of room meanwhile
            del junk

    def waiter() -> None:
        holding.wait()
        with mem.request("waits") as r:
            junk = torch.empty(4 * block, dtype=torch.uint8, device=dev)
            del junk
        waited.append(r.stats.retries)
        done.set()

    model.evict(), model2.evict()  # nothing cold is left to free: only the holder's bytes
    pair = [threading.Thread(target=holder), threading.Thread(target=waiter)]
    for t in pair:
        t.start()
    for t in pair:
        t.join()
    check("a request short of room waits for another's, then runs", waited and waited[0] >= 1)

    # Exhausted leaves the memory usable
    try:
        with mem.request("impossible"):
            torch.empty(torch.cuda.mem_get_info(0)[1] * 2, dtype=torch.uint8, device=dev)
        check("an impossible allocation raises Exhausted", False)
    except Exhausted as exc:
        check("an impossible allocation raises Exhausted", exc.gpu == 0 and exc.need > 0)
    with torch.no_grad(), mem.request("after"):
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
