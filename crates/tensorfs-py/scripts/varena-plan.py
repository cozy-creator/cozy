#!/usr/bin/env python3
"""varena's placement under a declared plan, on the CPU: the current step's working set wins.

The real `tensorfs.varena.torch.Memory` (eviction, restore, advance, the catch's rungs) and the
real per-GPU ledger run against a simulated card. The fake core below only maps and unmaps
region bytes as `varena.Gpu` does (home pages, three scratch slots, copies that land later,
`give`'s order); torch's allocator is a counted budget that runs out as a GPU's does, and a
kernel's first launch takes device bytes outside torch (lazy module loading, run 11952).

H3 ref2va_turbo 5 s runs with its declared plan, cold then warm, at the sizes of runs
11951-11955 and 12122 on RTX 4090s: 24 GiB on one GPU; an 8 GiB limit on the 24 GiB card with a gone
executor's kept pages over the limit (11955). 8 GiB at 2 GPUs is one such memory per GPU.

Each request must complete, and: no restore evicts a page with a declared use; no prefetch
refills a step after demand made room in it; no region comes back twice in a step; the bytes
moved stay within one read per pass plus one prefetch per model; on 24 GiB the warm DiT reads
nothing after its first pass. Kept pages no process uses go back on demand. A failed request
frees everything, a core error is typed, and the next request completes.

    python scripts/varena-plan.py
Runs with any python that has torch and the tensorfs wheel; no GPU.
"""

from __future__ import annotations

import atexit
import gc
import subprocess
import sys
import threading
import weakref
from collections import Counter
from types import SimpleNamespace

import torch
from tensorfs import varena
from tensorfs.varena import torch as vt
from torch import nn

MIB, GIB = 1 << 20, 1 << 30
PLAN = [["text_encoder"], ["audio_vae", "vae"], ["ref2va_turbo", "transformer_ref"], ["audio_vae"], ["vae"]]
#: component: (common bytes, block bytes, blocks, activation bytes per block)
SIZES = {
    "text_encoder": (98 * MIB, 254 * MIB, 30, 281_000_000),
    "audio_vae": (610_271_232, 0, 0, 0),
    "vae": (128 * MIB, 98 * MIB, 54, 0),
    "transformer_ref": (700 * MIB, 406_847_488, 50, 1_700_000_000),
    "ref2va_turbo": (700 * MIB, 0, 0, 0),
}
VAE_ENCODER, VAE_ACT = 18, (int(0.6 * GIB), int(1.3 * GIB))
#: torch bytes a request holds across its steps (12122: 0.93 GB as the DiT's passes start)
STATE = 930_000_000
#: the DiT's FFN as H3's (bf16, as float32 elements): projection 2 x 14336, activation 14336,
#: output 5376 wide; with the rest of the block, a 5.14 GB peak (12122)
DIT_FFN = (14336, 7168, 2688)
PASSES = {"transformer_ref": 8}
#: a LoRA bake's temporaries as each DiT block's fresh pages arrive (12139: 20 MiB on GPU 1)
BAKE = 20 * MIB
LAUNCH = 48 * MIB  # a kernel's first launch: its module's device bytes
results: list[bool] = []


def check(name: str, ok: object, detail: object = "") -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'} {name}" + ("" if ok else f": {detail}"), flush=True)


# ------------------------------------------------------------------------- the card


class Card:
    """One GPU: the driver's free count and torch's caching allocator."""

    def __init__(self, total: int, context: int) -> None:
        self.total, self.context, self.foreign, self.mapped = total, context, 0, 0
        self.reserved = self.allocated = self.peak = 0
        self.cap, self.loaded = total, set()
        self.core: Core | None = None

    def free(self) -> int:
        return max(self.total - self.context - self.foreign - self.mapped - self.reserved, 0)

    def malloc(self, n: int) -> None:
        grow = max(self.allocated + n - self.reserved, 0)
        if grow and (grow > self.free() or self.reserved + grow > self.cap):
            raise torch.OutOfMemoryError(f"CUDA out of memory. Tried to allocate {n / MIB:.2f} MiB")
        self.reserved += grow
        self.allocated += n
        self.peak = max(self.peak, self.allocated)

    def launch(self, kernel: str) -> None:
        """A kernel's first launch maps its module: device bytes no allocator sees."""
        if kernel not in self.loaded:
            if self.free() < LAUNCH:
                raise torch.OutOfMemoryError("8-bit linear: the quantization: cudaErrorMemoryAllocation")
            self.context += LAUNCH
            self.loaded.add(kernel)


CARD: Card = Card(0, 0)


class Stream:
    cuda_stream = 1

    def synchronize(self) -> None:
        CARD.core.land()

    def wait_stream(self, other: object) -> None:
        pass


STREAM = Stream()


def fake_cuda() -> None:
    """torch.cuda as varena's glue calls it, answered by `CARD`."""
    c = torch.cuda
    c.mem_get_info = lambda *_: (CARD.free(), CARD.total)
    c.memory_reserved = lambda *_: CARD.reserved
    c.memory_allocated = lambda *_: CARD.allocated
    c.max_memory_allocated = lambda *_: CARD.peak
    c.reset_peak_memory_stats = lambda *_: setattr(CARD, "peak", CARD.allocated)
    c.empty_cache = lambda: setattr(CARD, "reserved", CARD.allocated)
    c.set_per_process_memory_fraction = lambda f, *_: setattr(CARD, "cap", int(f * CARD.total))
    c.current_stream = lambda *_: STREAM
    c.synchronize = lambda *_: CARD.core.land()
    c.get_device_properties = lambda i: SimpleNamespace(uuid=f"GPU-varena-plan-{i}-{id(CARD)}")
    c.is_current_stream_capturing = lambda: False
    c.is_initialized = lambda: True
    c.is_available = lambda: True
    c.current_device = lambda: 0
    torch._C._cuda_clearCublasWorkspaces = lambda: None
    torch.from_dlpack = lambda pinned: pinned.tensor()


# ------------------------------------------------------------------------ the core


def full(need: int) -> varena.DeviceFull:
    short = need - CARD.free()
    e = varena.DeviceFull(f"device full: mapping {need} bytes, {short} short")
    e.need, e.deficit = need, short
    return e


class Pin:
    def __init__(self, release: object, nbytes: int, fresh: bool) -> None:
        self.nbytes, self.fresh, self.release = nbytes, fresh, release

    def tensor(self) -> torch.Tensor:
        t = torch.empty(self.nbytes, dtype=torch.uint8, device="meta")
        weakref.finalize(t, self.release)
        self.release = lambda: None
        return t

    def __del__(self) -> None:
        self.release()


class Slot:
    def __init__(self, span: int) -> None:
        self.span, self.holds, self.pins, self.pending, self.used = span, None, 0, False, 0


class Reg:
    def __init__(self, span: int) -> None:
        self.span, self.home, self.mapped, self.landing, self.pins = span, True, False, False, 0


class Set:
    """`varena.WeightSet` on the card: one region per block (and common)."""

    def __init__(self, core: Core, name: str, regions: list[list[str]], spans: list[int]) -> None:
        self.core, self.name, self.regions, self.spans = core, name, regions, spans
        self.nbytes = sum(spans)
        self.regs = [Reg(s) for s in spans]
        self.parts = [
            SimpleNamespace(region=r, offset=0, nbytes=s, dtype="U8", shape=[s], key=n[0].split("/", 1)[1],
                            role="", component=n[0].split("/", 1)[0])
            for r, (n, s) in enumerate(zip(regions, spans))
        ]  # fmt: skip
        self.paged_in = self.streamed = self.used = 0
        self.warm_, self.busy_ = True, False

    def home(self, regions: list[int]) -> None:
        for r, reg in enumerate(self.regs):
            reg.home = r in regions

    def resident(self, r: int) -> bool:
        return self.regs[r].mapped

    def _map(self, r: int, landing: bool) -> None:
        reg = self.regs[r]
        if CARD.free() < reg.span:
            raise full(reg.span)
        CARD.mapped += reg.span
        reg.mapped, reg.landing = True, landing
        self.paged_in += reg.span
        self.core.note("prefetch" if landing else "page", self, r)

    def bind(self, r: int, stream: int = 0) -> Pin:
        self.core.check()
        self.core.clock += 1
        self.used, reg = self.core.clock, self.regs[r]
        if reg.mapped or reg.home:
            fresh = not reg.mapped
            if fresh:
                self._map(r, landing=False)
            reg.landing = False
            reg.pins += 1
            return Pin(lambda: setattr(reg, "pins", reg.pins - 1), reg.span, fresh)
        slot = self.core.slot_for(self, r, urgent=True)
        slot.pending = False
        slot.pins += 1
        return Pin(lambda: setattr(slot, "pins", slot.pins - 1), reg.span, True)

    def prefetch(self, r: int) -> None:
        reg = self.regs[r]
        if not reg.mapped:
            self._map(r, landing=True) if reg.home else self.core.slot_for(self, r, urgent=False)

    def evict(self, regions: list[int] | None = None, budget: int = 1 << 62) -> int:
        freed = 0
        for r in range(len(self.regs)) if regions is None else regions:
            reg = self.regs[r]
            if freed < budget and reg.mapped and not reg.pins and not reg.landing:
                reg.mapped = False
                CARD.mapped -= reg.span
                freed += reg.span
        return freed

    def warm(self, warm: bool) -> None:
        self.warm_ = warm

    def busy(self, busy: bool) -> None:
        self.busy_ = busy

    def shareable(self) -> bool:
        return False

    def kept(self) -> list:
        return []

    def close(self) -> None:
        self.evict()


class Core:
    """`varena.Gpu` on the card: sets, three scratch slots, copies landing at a sync."""

    def __init__(self, gpu: int) -> None:
        self.sets, self.slots, self.clock = [], [], 0
        self.events: list[tuple] = []
        self.memory: vt.Memory | None = None
        self.fail_bind = False
        CARD.core = self

    def note(self, kind: str, ws: Set, r: int) -> None:
        step = self.memory._here_step() if self.memory else -1
        self.events.append((kind, ws.name.rsplit("/", 1)[-1], r, step, ws.regs[r].span))

    def check(self) -> None:
        if self.fail_bind:
            self.fail_bind = False
            e = varena.VarenaError("varena panicked: an injected bug")
            e.code = "PANIC"
            raise e

    def serve(self, row: object) -> None:
        pass

    def register(self, name: str, ck: object, regions: list[list[str]], host_fd: object) -> Set:
        ws = Set(self, name, regions, [ck.span[n[0]] for n in regions])
        self.sets.append(ws)
        return ws

    def land(self) -> None:
        for ws in self.sets:
            for reg in ws.regs:
                reg.landing = False

    def span(self) -> int:
        return max((rg.span for ws in self.sets for rg in ws.regs if not rg.home), default=0)

    def slot_for(self, ws: Set, r: int, urgent: bool) -> Slot | None:
        held = next((s for s in self.slots if s.holds == (ws.name, r)), None)
        if held is not None:
            return held
        need = ws.regs[r].span
        free = [s for s in self.slots if not s.pins and s.span >= need and (urgent or not s.pending)]
        if free:
            slot = min(free, key=lambda s: (s.holds is not None, s.pending, s.used))
        elif urgent or len(self.slots) < vt._SLOTS:
            span = max(need, self.span())
            if CARD.free() < span:
                raise full(span)
            CARD.mapped += span
            slot = Slot(span)
            self.slots.append(slot)
        else:
            return None
        slot.holds, slot.pending, slot.used = (ws.name, r), not urgent, self.clock
        ws.streamed += need
        self.note("stream", ws, r)
        return slot

    def shed(self, keep: int) -> int:
        span, freed = self.span(), 0
        for slot in list(self.slots):
            if (span == 0 or slot.span < span or len(self.slots) > keep) and not slot.pins:
                self.slots.remove(slot)
                CARD.mapped -= slot.span
                freed += slot.span
        return freed

    def trim(self) -> int:
        return self.shed(vt._SLOTS)

    def give(self, want: int) -> list[tuple[str, int]]:
        freed = self.trim()
        gave = [("", freed)] if freed else []
        for ws in sorted((w for w in self.sets if not w.busy_), key=lambda w: (w.warm_, w.used)):
            if freed >= want:
                break
            if n := ws.evict(None, want - freed):
                gave.append((ws.name, n))
                freed += n
        if freed < want and (n := self.shed(0)):
            gave.append(("", n))
        return gave

    def stats(self) -> SimpleNamespace:
        sets = [
            SimpleNamespace(name=ws.name, resident=sum(g.span for g in ws.regs if g.mapped),
                            total=SimpleNamespace(paged_in=ws.paged_in, streamed_bytes=ws.streamed, stall_ns=0))
            for ws in self.sets
        ]  # fmt: skip
        return SimpleNamespace(mapped=CARD.mapped, idle=0, slot_bytes=sum(s.span for s in self.slots), sets=sets)

    def moved(self) -> int:
        return sum(ws.paged_in + ws.streamed for ws in self.sets)


# ---------------------------------------------------------------------- H3's models


#: an op allocating device bytes on the card, as aten ops do: the catch sees it at the op
LIB = torch.library.Library("varenaplan", "DEF")
LIB.define("alloc(int n, *, Device device) -> Tensor")
LIB.impl("alloc", lambda n, device: CARD.malloc(n) or torch.empty(0), "CompositeExplicitAutograd")


class FeedForward(nn.Module):
    """Stands for diffusers' FeedForward, which `cuts.install` cuts by tokens."""


FeedForward.__module__ = "diffusers.models.attention"


class FFN(FeedForward):
    """An FFN that holds its projection's output and its activation at once, per token."""

    def __init__(self, proj: int, inner: int, out: int) -> None:
        super().__init__()
        self.proj, self.out = nn.Module(), nn.Module()
        self.proj.out_features, self.out.in_features, self.out.out_features = proj, inner, out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = self.proj.out_features + self.out.in_features
        n = width * x.element_size() * (x.numel() // x.shape[-1])
        torch.ops.varenaplan.alloc(n, device=torch.device("cuda", 0))
        CARD.allocated -= n
        return x


class Bake:
    """The turbo LoRA baked into the DiT's fresh pages (`Declared.fill`): its temporaries are
    allocations inside a bind."""

    def __call__(self, block: object, parts: object) -> None:
        torch.ops.varenaplan.alloc(BAKE, device=torch.device("cuda", 0))
        CARD.allocated -= BAKE

    def prepare(self) -> str:
        return "turbo"


class Block(nn.Module):
    def __init__(self, act: int, kernel: str) -> None:
        super().__init__()
        self.act, self.kernel = act, kernel
        self.ff = FFN(*DIT_FFN) if kernel == "transformer_ref" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        CARD.launch(self.kernel)
        torch.ops.varenaplan.alloc(self.act, device=torch.device("cuda", 0))
        if self.ff is not None:
            x = self.ff(x)
        CARD.allocated -= self.act
        return x


class Component(nn.Module):
    def __init__(self, name: str, blocks: int, act: int) -> None:
        super().__init__()
        self.name, self.blocks = name, nn.ModuleList(Block(act, name) for _ in range(blocks))
        self.fail_at: int | None = None
        self.passes = 0

    def forward(self, x: torch.Tensor, part: slice = slice(None)) -> torch.Tensor:
        self.passes += 1
        if self.name == "transformer_ref":  # the turbo LoRA, read by tensor in every pass
            with MEM._reading([LORA.common]):
                pass
        for i, block in enumerate(self.blocks[part]):
            if self.fail_at == (self.passes, i):
                raise RuntimeError("a package bug in the DiT")
            x = block(x)
        return x


class Checkpoint:
    """What `regions_of` reads of a checkpoint: each component's tensor keys in header order."""

    def __init__(self) -> None:
        self.components, self.span = {}, {}
        self.id = "sha256:" + "0" * 64
        for name, (common, block, n, _) in SIZES.items():
            keys = ["common.weight", *(f"blocks.{i}.weight" for i in range(n))]
            self.components[name] = SimpleNamespace(tensors=dict.fromkeys(keys))
            for key in keys:
                self.span[f"{name}/{key}"] = common if key == "common.weight" else block


MEM: vt.Memory
LORA: vt.Model


def build(
    card: Card, limit: int | None, orphans: int, weights: bool
) -> tuple[vt.Memory, Core, dict[str, vt.Model]]:
    global CARD, MEM, LORA
    CARD = card
    vt.Gpu = Core  # the seam: the card's core in place of the driver's
    mem = MEM = vt.Memory.open(torch.device("cuda", 0), limit=limit)
    core = mem.core
    core.memory = mem
    if orphans:
        kept(mem, card, orphans, weights)
    mem._here_step = lambda: next((p.cursor for p in mem.plans.values() if p.started), -1)
    ck, models = Checkpoint(), {}
    for name, (_, _, n, act) in SIZES.items():
        module = Component(name, n, act)
        if name == "vae":  # its encoder's blocks, then its decoder's
            for i, block in enumerate(module.blocks):
                block.act = VAE_ACT[i >= VAE_ENCODER]
        blocks = [f"blocks.{i}" for i in range(n)]
        grant = vt.WeightGrant(ck, name, "sha256:" + "0" * 64)
        fill = Bake() if name == "transformer_ref" else None
        vars(module)["_varena"] = vt.Declared(grant, blocks, fill=fill)
        models[name] = mem.load(module)
    LORA = models["ref2va_turbo"]
    record = mem._evicted

    def evicted(model: vt.Model, rung: int, why: str, nbytes: int) -> None:
        core.events.append(("evicted", model.component, why, mem._here_step(), mem._distance(model)))
        record(model, rung, why, nbytes)

    mem._evicted = evicted  # type: ignore[method-assign]
    return mem, core, models


def h3(
    mem: vt.Memory, models: dict[str, vt.Model], rid: str, tokens: int = 4096, dit: int = 40_000
) -> vt.RequestStats:
    """One ref2va_turbo 5 s request: condition, encode the reference, 8 turbo steps, decode."""
    x = lambda n: torch.zeros(n, 8)  # noqa: E731
    m = {k: v.module for k, v in models.items()}
    request = None
    try:
        with mem.request(rid, plan=PLAN) as request:
            # its inputs, embeddings and latents, held until it ends: room from the first step's
            # weights still landing
            torch.ops.varenaplan.alloc(STATE, device=torch.device("cuda", 0))
            try:
                m["text_encoder"](x(tokens))
                m["audio_vae"](x(16))
                m["vae"](x(1024), slice(0, VAE_ENCODER))
                for _ in range(8):
                    m["transformer_ref"](x(dit))
                m["audio_vae"](x(32))
                m["vae"](x(2048), slice(VAE_ENCODER, None))
            finally:
                CARD.allocated -= STATE
    finally:
        gc.collect()
    return request.stats


# --------------------------------------------------------------------------- checks


def steps(core: Core, since: int) -> dict[int, list[tuple]]:
    by: dict[int, list[tuple]] = {}
    for e in core.events[since:]:
        by.setdefault(e[3], []).append(e)
    return by


def plan_checks(case: str, mem: vt.Memory, core: Core, since: int, moved: int, stats: vt.RequestStats) -> None:
    current = {k: {n for n in PLAN[k]} for k in range(len(PLAN))}
    restore_now, refill, twice = [], [], []
    for step, events in steps(core, since).items():
        if step < 0:
            continue
        demanded, restored = False, Counter()
        for e in events:
            if e[0] == "evicted":
                if e[2] == "restore" and e[4] < float("inf"):
                    restore_now.append(e)
                demanded |= e[2] != "restore"
            elif e[0] in ("prefetch", "stream") and e[1] not in current[step]:
                restored[(e[1], e[2])] += 1
                if demanded:
                    refill.append(e)
        twice += [k for k, n in restored.items() if n > 1]
    check(f"{case}: no restore evicts a page with a declared use", not restore_now, restore_now[:3])
    check(f"{case}: no prefetch refills a step after demand", not refill, refill[:3])
    check(f"{case}: no region restored twice in a step", not twice, twice[:3])
    sizes = {ws.name.rsplit("/", 1)[-1]: ws.nbytes for ws in core.sets}
    over = []
    for step, events in steps(core, since).items():
        took = sum(e[4] for e in events if e[0] in ("page", "prefetch", "stream"))
        bound = sum(sizes[n] * PASSES.get(n, 1) for n in PLAN[step]) + sum(sizes.values()) if step >= 0 else 0
        if took > bound:
            over.append((step, round(took / GIB, 1), round(bound / GIB, 1)))
    check(f"{case}: each step moves at most one read per pass and one prefetch per model "
          f"({moved / GIB:.1f} GiB in all)", not over, over)  # fmt: skip


def settled(case: str, mem: vt.Memory, core: Core, start: int) -> None:
    pins = [ws.name for ws in core.sets for g in ws.regs if g.pins] + [s.holds for s in core.slots if s.pins]
    counts = [b.path for m in mem.models.values() for b in (m.common, *m.blocks) if b.count]
    check(f"{case}: nothing pinned or bound after it", not pins and not counts, (pins[:3], counts[:3]))
    check(f"{case}: no torch bytes left", CARD.allocated == start, CARD.allocated - start)
    check(f"{case}: no plan or request left", not mem.plans and not mem.active)


def tensord(fd: int, held: int, weights: bool) -> None:
    """Another row in its own process: TensorD's, a gone executor's kept pages (spare), or an
    idle executor's cold weights; it gives what it is asked."""
    ledger = varena.Ledger.adopt(fd)
    row = ledger.claim()
    facts = lambda n: {"resident": n, "cold": n, "kept": n, "last_use": 1} if weights else {"idle": n}  # noqa: E731
    row.publish(**facts(held))
    print("ready", flush=True)
    while True:
        want = row.wait_want()
        freed = min(want, held)
        print(freed, flush=True)
        sys.stdin.readline()  # the card has them back
        held -= freed
        row.publish(**facts(held))
        row.ack(freed)
        ledger.freed()


def kept(mem: vt.Memory, card: Card, orphans: int, weights: bool) -> None:
    fd = mem.row[0].fd
    args = [sys.executable, __file__, "--tensord", str(fd), str(orphans), str(int(weights))]
    child = subprocess.Popen(args, pass_fds=[fd], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    atexit.register(child.kill)
    assert child.stdout.readline().strip() == "ready"
    card.foreign = orphans

    def relay() -> None:
        for line in child.stdout:
            card.foreign -= int(line)
            child.stdin.write("ok\n")
            child.stdin.flush()

    threading.Thread(target=relay, daemon=True).start()


def case(name: str, card: Card, limit: int | None, orphans: int = 0, weights: bool = False) -> None:
    mem, core, models = build(card, limit, orphans, weights)
    dit, asks = models["transformer_ref"].ws.name, []
    for warm in (False, True):
        label, since, moved, retries = f"{name} {'warm' if warm else 'cold'}", len(core.events), core.moved(), mem.memo.retries
        start = CARD.allocated
        try:
            # warm: a prompt of another length, a DiT shape key never run (12163)
            stats = h3(mem, models, label, dit=40_000 + 8 * warm)
        except Exception as exc:  # noqa: BLE001
            check(f"{label}: completes", False, f"{type(exc).__name__}: {exc}")
            return
        check(f"{label}: completes", True)
        asks += [e for e in stats.evictions if e.reason.startswith("asked") and e.nbytes > 0]
        plan_checks(label, mem, core, since, core.moved() - moved, stats)
        settled(label, mem, core, start)
        passes = [s for s in stats.steps if s.model == dit]
        most = max((s.paged_in + s.streamed for s in passes), default=0)
        whole = models["transformer_ref"].nbytes
        check(f"{label}: no DiT pass reads a block twice ({most / 1e9:.1f} GB)", most <= whole, most)
        if warm and limit is None:
            moved = core.moved() - moved
            check(f"{label}: reads {moved / 1e9:.1f} GB, at most 40 GB (B2)", moved <= 40e9)
            check(f"{label}: no out-of-memory retry", mem.memo.retries == retries, mem.memo.retries - retries)
            later = [s for s in stats.steps if s.model == dit][1:]
            read = sum(s.paged_in + s.streamed for s in later)
            check(f"{label}: the DiT reads 0 B after its first pass", len(later) == 7 and read == 0, read)
    if orphans:
        check(f"{name}: the other row gave bytes back on demand ({card.foreign / GIB:.2f} GiB left)", card.foreign < orphans)
        check(f"{name}: the request's stats name each ask and what it got", asks, asks)
    failure(name, mem, core, models)


def failure(name: str, mem: vt.Memory, core: Core, models: dict[str, vt.Model]) -> None:
    """A failed request frees everything; a core error is typed; the next request completes."""
    dit = models["transformer_ref"].module
    for label, arm in (("a package bug", "bug"), ("a core error", "core")):
        start = CARD.allocated
        dit.passes, dit.fail_at = 0, (3, 7) if arm == "bug" else None
        if arm == "core":
            core.fail_bind = True
        try:
            h3(mem, models, f"{name} {label}")
            check(f"{name} {label}: fails", False, "it completed")
        except BaseException as exc:  # noqa: BLE001
            typed = isinstance(exc, RuntimeError if arm == "bug" else varena.VarenaError)
            check(f"{name} {label}: fails typed ({type(exc).__name__})", typed, exc)
        dit.fail_at = None
        settled(f"{name} after {label}", mem, core, start)
        try:
            h3(mem, models, f"{name} after {label}")
            check(f"{name} after {label}: the next request completes", True)
        except Exception as exc:  # noqa: BLE001
            check(f"{name} after {label}: the next request completes", False, exc)


def main() -> int:
    fake_cuda()
    card = 25_757_220_864  # an RTX 4090 (11951-11955)
    case("24 GiB", Card(card, GIB // 2), None)
    case("8 GiB", Card(card, GIB // 2), 8 * GIB)
    case("8 GiB beside kept pages", Card(card, GIB // 2), 8 * GIB, orphans=int(8.74 * GIB))
    case("24 GiB beside an idle executor", Card(card, GIB // 2), None, orphans=18 * GIB, weights=True)
    failed = results.count(False)
    print(f"{len(results) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--tensord"]:
        tensord(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1")
    sys.exit(main())
