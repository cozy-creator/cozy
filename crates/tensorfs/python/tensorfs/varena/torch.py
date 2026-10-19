"""varena's memory for torch modules (memory-varena.md §2, §5).

    mem = Memory.open([0])
    unet = mem.register("unet", module, WeightGrant(ck, "unet", host_fd), 0)
    with mem.request("r1", [0]), mem.use([unet], repeats=True):
        module(x)

One `Memory` per process decides where every weight byte of its GPUs lives. A block's params
are views of its region's pages only while the block runs: its forward binds them (the view is
the pin) and puts zero-byte placeholders back after, so no tensor can point at an unmapped page.
An op on a placeholder outside a forward binds its block for that op. Room is made in one order
(§5.2): idle bytes, cold weights, then the hot model's watermark, which falls only within a
request and is restored into free room after it. On a CPU or Apple device every weight binds
once, at registration.
"""

from __future__ import annotations

import gc
import threading
import time
import traceback
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import tensorfs
import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import _disable_current_modes
from torch.utils._pytree import tree_map_only

from . import DeviceFull, Gpu, Host, Ledger, Pinned, Row, WeightSet, cuts

#: Regions are cut only at block boundaries: one per block, plus each run of common tensors.
_WHOLE = 1 << 62
#: Blocks copied ahead of the one entered, in the recorded order.
_AHEAD = 2
#: A streamed model's scratch slots, in its largest blocks (the core keeps 2-3).
_SLOTS = 3
#: TensorFS dtype names; an `F4` part is u8 pairs.
DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F4": torch.uint8,
}


class Mode(Enum):
    SHARED = "shared"
    DEDICATED = "dedicated"


@dataclass(frozen=True, slots=True)
class WeightGrant:
    """TensorD's read grant for one component: its header, the sealed host-cache memfd or the
    chunk files (`(sha256, length, fd)`), and the stored tensors the module builds (None: all
    of the component's)."""

    ck: tensorfs.Checkpoint
    component: str
    host_fd: int | None = None
    keys: Collection[str] | None = None
    files: Sequence[tuple[str, int, int]] = ()


@dataclass(frozen=True, slots=True)
class LedgerGrant:
    """This executor's row in one GPU's ledger (§5.6)."""

    gpu: int
    fd: int


class Exhausted(Exception):
    """It still does not fit with every other evictable byte gone: §2's one memory error."""

    def __init__(self, gpu: int, need: int, op: str, stats: Stats) -> None:
        super().__init__(f"GPU {gpu}: {op} needs {need} B more and nothing else can be freed")
        self.gpu, self.need, self.op, self.stats = gpu, need, op, stats


class Unbound(Exception):
    """An op on a placeholder that binds to nothing."""


#: `bind(key, parts)`: a stored tensor's typed part views -> the module tensors they become, by
#: dotted name. What it makes lives for that bind only.
type Bind = Callable[[str, Mapping[str, Tensor]], Mapping[str, Tensor]]
#: `fill(block, parts)`: on page-in, before any use, `parts[key][role]` over the fresh bytes.
type Fill = Callable[["Block", Mapping[str, Mapping[str, Tensor]]], None]


class Shed(Protocol):
    def __call__(self, nbytes: int, /) -> int: ...


# ------------------------------------------------------------------------------- stats


@dataclass(frozen=True, slots=True)
class Step:
    """One pass of a model: its home set, bytes resident, copied home and streamed through
    slots, the GPU-timed prefetch stall, and torch's allocated peak above the pass's start."""

    model: str
    gpu: int
    step: int
    home: int
    resident: int
    paged_in: int
    streamed: int
    stall_ns: int
    activations: int


@dataclass(frozen=True, slots=True)
class Eviction:
    model: str
    gpu: int
    rung: int
    reason: str
    nbytes: int


@dataclass(frozen=True, slots=True)
class ModelStats:
    """A model now (bytes at home, resident) and since it registered (copied home, streamed,
    prefetch stall)."""

    name: str
    gpu: int
    nbytes: int
    home: int
    resident: int
    paged_in: int
    streamed: int
    stall_ns: int
    uses: int
    kept: bool
    reserve: int


@dataclass(frozen=True, slots=True)
class Peak:
    """A model's passes at one shape key: the activation peak last measured (-1: none yet) and
    the cuts in force, `"<cut> <size per piece>"` (§5.8)."""

    model: str
    shape: str
    peak: int
    cuts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RequestStats:
    id: str
    steps: tuple[Step, ...]
    evictions: tuple[Eviction, ...]
    #: torch bytes still allocated at its end beyond its start, by GPU (§5.7)
    leftover: tuple[tuple[int, int], ...]
    #: weight bytes copied to its GPUs while it ran, home or through slots
    copied: int
    #: out-of-memory retries during it, and the shape keys its models ran at
    retries: int = 0
    peaks: tuple[Peak, ...] = ()


@dataclass(frozen=True, slots=True)
class Stats:
    models: tuple[ModelStats, ...]
    #: requests ended since the last draining `stats()`
    requests: tuple[RequestStats, ...]
    #: evictions since the memory opened
    evictions: int
    #: the memo: every shape key's peak and cuts, and every retry since the memory opened
    peaks: tuple[Peak, ...] = ()
    retries: int = 0


# ------------------------------------------------------------------------ placeholders


@dataclass(frozen=True, slots=True)
class _Site:
    table: dict[str, Tensor | None]
    attr: str
    parameter: bool


class Placeholder(Tensor):
    """A weight while no pages are bound to it: shape, dtype and execution device, no bytes."""

    block: Block | None
    site: _Site | None

    @staticmethod
    def __new__(
        cls,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        block: Block | None,
        site: _Site | None,
    ) -> Placeholder:
        held: Placeholder = Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, tuple(shape), dtype=dtype, device=device, storage_size=0
        )
        held.block, held.site = block, site
        return held

    def __repr__(self) -> str:  # type: ignore[override]
        return f"Placeholder({tuple(self.shape)}, {self.dtype}, {self.device})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        aten = torch.ops.aten
        kwargs = kwargs or {}
        if func in (aten.detach.default, aten.alias.default):
            return args[0]
        if func is aten._to_copy.default:
            held = args[0]
            device = kwargs.get("device")
            return cls(
                held.shape,
                kwargs.get("dtype") or held.dtype,
                held.device if device is None else torch.device(device),
                held.block,
                held.site,
            )
        for i, argument in enumerate(func._schema.arguments):
            if argument.alias_info is not None and argument.alias_info.is_write:
                written = args[i] if i < len(args) else kwargs.get(argument.name)
                if isinstance(written, Placeholder):
                    raise Unbound(f"{func}: an op writes into a weight no pages are bound to")
        blocks: dict[int, Block] = {}

        def origin(p: Placeholder) -> Placeholder:
            if p.block is None:
                raise Unbound(f"{func}: a placeholder of no registered block")
            blocks[id(p.block)] = p.block
            return p

        tree_map_only(Placeholder, origin, (args, kwargs))
        found = list(blocks.values())
        with found[0].model.memory._reading(found):
            real_args, real_kwargs = tree_map_only(Placeholder, _real, (args, kwargs))
            return func(*real_args, **real_kwargs)


def _real(p: Placeholder) -> Tensor:
    """What is bound at `p`'s site, in `p`'s dtype and device (a converted copy keeps its own)."""
    value = p.site.table[p.site.attr] if p.site is not None else None
    if value is None or isinstance(value, Placeholder):
        raise Unbound(f"a weight of {p.block.model.name if p.block else '?'} binds to nothing")
    if value.dtype != p.dtype or value.device != p.device:
        return value.to(dtype=p.dtype, device=p.device)
    return value


# ---------------------------------------------------------------------------- the model


@dataclass(frozen=True, slots=True)
class _PartAt:
    region: int
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass(eq=False, slots=True)
class Block:
    """One block, or a model's `common` (index -1): its regions and stored tensors, and while
    anyone runs it, the pins and writes that bind it."""

    model: Model
    index: int
    path: str
    module: nn.Module | None
    regions: list[int] = field(default_factory=list)
    tensors: dict[str, dict[str, _PartAt]] = field(default_factory=dict)
    count: int = 0
    stream: torch.cuda.Stream | None = None
    pins: list[Tensor] = field(default_factory=list)
    written: list[tuple[_Site, Tensor | None]] = field(default_factory=list)
    forward: Callable[..., object] | None = None
    own_forward: bool = False
    site: cuts.Site | None = None

    @property
    def nbytes(self) -> int:
        return sum(self.model.sizes[r] for r in self.regions)

    def parts(self, region: int, base: Tensor) -> dict[str, dict[str, Tensor]]:
        """Typed views of the stored tensors in `region`, over its bytes `base`."""
        return {
            key: {role: _view(base, at) for role, at in roles.items()}
            for key, roles in self.tensors.items()
            if next(iter(roles.values())).region == region
        }


def _view(base: Tensor, at: _PartAt) -> Tensor:
    return base[at.offset : at.offset + at.nbytes].view(at.dtype).view(at.shape)


class Model:
    """One registered component: its weight set, blocks, watermark and per-step stats."""

    def __init__(
        self,
        memory: Memory,
        name: str,
        module: nn.Module,
        gpu: int,
        device: torch.device,
        ws: WeightSet,
        fill: Fill | None,
        bind: Bind | None,
    ) -> None:
        self.memory, self.name, self.module, self.gpu, self.device = (
            memory,
            name,
            module,
            gpu,
            device,
        )
        self.ws, self.fill, self.bind = ws, fill, bind
        self.sites: dict[str, list[_Site]] = {}
        self.sizes: dict[int, int] = {}
        self.blocks: list[Block] = []
        self.common = Block(self, -1, "", module)
        #: block indexes in the order its first pass ran them (header order until then)
        self.order: list[int] = []
        self.position: dict[int, int] = {}
        self.seen: dict[int, None] = {}
        #: blocks at their home VA (the rest stream through slots); common is always home
        self.home: set[int] = set()
        #: home blocks whose pages are resident now
        self.loaded: set[int] = set()
        self.lowered = False
        #: its last use repeats (a denoise loop): restored before models used once
        self.repeating = False
        self.uses = 0
        self.held = 0
        self.kept = False
        self.last_use = 0
        #: torch bytes its passes allocated above their start in its last request; -1 unmeasured
        self.reserve = -1
        self.peak = 0
        self.steps = 0
        self.mark = (0, 0, 0)
        self.base = 0
        self.closed = False
        #: its pass now: the memo entry of the shape it runs at
        self.track = cuts.Track(name)
        #: its open use repeats, for a pass's plan
        self.repeats = False

    @property
    def host(self) -> bool:
        return self.gpu < 0

    @property
    def nbytes(self) -> int:
        return sum(self.sizes.values())

    def home_bytes(self) -> int:
        return self.common.nbytes + sum(self.blocks[b].nbytes for b in self.home)

    def hold(self) -> Hold:
        """Every block bound until the hold closes: what a CUDA graph captured stays mapped.
        Only a model wholly at home can be held (a slot's address moves)."""
        return self.memory._hold(self)

    def evict(self) -> int:
        """Give back every resident page nobody is using: they refill, and `fill` runs again."""
        with self.memory.lock:
            return self.memory._evict(self, "refill", 0)

    def close(self) -> None:
        self.memory._close(self)


class Hold:
    def __init__(self, model: Model) -> None:
        self.model, self.open = model, True

    def close(self) -> None:
        if self.open:
            self.open = False
            self.model.memory._release(self.model)


class Cache:
    """Device bytes an owner keeps for speed (KV slots, graph pools): counted, shed on ask."""

    def __init__(self, memory: Memory, owner: str, gpu: int, shed: Shed) -> None:
        self.memory, self.owner, self.gpu, self.shed = memory, owner, gpu, shed
        self.nbytes = 0

    def resize(self, nbytes: int) -> bool:
        """Whether the owner may hold `nbytes` now. Growing takes idle and cold bytes, never a
        hot model's; False pauses the owner, and nothing waits."""
        if nbytes > self.nbytes and not self.memory._grant(self, nbytes - self.nbytes):
            return False
        self.nbytes = nbytes
        return True

    def close(self) -> None:
        with self.memory.lock:
            self.memory.caches.pop(self, None)


@dataclass(eq=False, slots=True)
class Request:
    id: str
    gpus: tuple[int, ...]
    start: dict[int, int]
    #: each model's weight bytes copied to the device so far, as it began
    copies: dict[str, int]
    steps: list[Step] = field(default_factory=list)
    evictions: list[Eviction] = field(default_factory=list)
    stats: RequestStats | None = None
    #: the memo's retries when it started
    retried: int = 0
    shapes: dict[tuple[str, str], cuts.Shape] = field(default_factory=dict)


# --------------------------------------------------------------------------- the memory


class Memory:
    """This process's devices: every registered model, cache and request on them."""

    def __init__(self, gpus: Sequence[int], ledger: Sequence[LedgerGrant], mode: Mode) -> None:
        self.mode = mode
        self.gpus = {gpu: Gpu(gpu) for gpu in gpus}
        #: this process's device bytes outside torch and the weights (its CUDA context)
        self.context = 0
        self.last_use = 0
        self.host: Host | None = None
        self.models: dict[str, Model] = {}
        self.caches: dict[Cache, None] = {}
        self.lock = threading.RLock()
        self.local = threading.local()
        self.active: dict[Request, None] = {}
        self.ended: list[RequestStats] = []
        self.clock = 0
        self.evictions = 0
        self.memo = cuts.Memo()
        cuts.serve(self.memo, self)
        #: each GPU's ledger and this process's row in it, held for the process's life (§5.6)
        self.rows: dict[int, tuple[Ledger, Row]] = {}
        for grant in ledger:
            table = Ledger.adopt(grant.fd)
            row = table.claim()
            self.rows[grant.gpu] = (table, row)
            threading.Thread(
                target=self._answer, args=(grant.gpu, row), name="varena-ledger", daemon=True
            ).start()

    @classmethod
    def open(
        cls, gpus: Sequence[int], ledger: Sequence[LedgerGrant] = (), mode: Mode = Mode.SHARED
    ) -> Memory:
        return cls(gpus, ledger, mode)

    # -- registration

    def register(
        self,
        name: str,
        module: nn.Module,
        weights: WeightGrant,
        gpu: int | torch.device,
        blocks: Sequence[str] | None = None,
        fill: Fill | None = None,
        bind: Bind | None = None,
        state: Sequence[str] = (),
    ) -> Model:
        """Register `module`, built on meta, over its stored weights. On a GPU its blocks
        (default: the innermost `_no_split_modules`) bind per forward and run in the catch
        (`cuts.run`), its FFNs cut in tokens; on a CPU or Apple device every weight binds once,
        here. `state`: attribute paths under `module` a block's forward may change, put back
        before a block that ran out of memory runs again."""
        paths = list(blocks) if blocks is not None else innermost_blocks(module)
        prefix = f"{weights.component}/"
        regions = regions_of(weights, paths)
        if isinstance(gpu, torch.device):
            if self.host is None:
                self.host = Host()
            device, index, core = gpu, -1, self.host
        else:
            device, index, core = torch.device("cuda", gpu), gpu, self.gpus[gpu]
        with self.lock:
            if name in self.models:
                raise ValueError(f"a model named {name!r} is registered already")
            model = Model(
                self,
                name,
                module,
                index,
                device,
                core.register(
                    name, weights.ck, regions, weights.host_fd, list(weights.files) or None
                ),
                fill,
                bind,
            )
            _layout(model, prefix, paths, regions)
            model.sites = _sites(module)
            _install(model)
            model.order = [b.index for b in model.blocks]
            model.position = {b: i for i, b in enumerate(model.order)}
            model.home = set(model.order)
            self.models[name] = model
            if model.host:
                _bind_host(model)
            else:
                model.ws.home(_home_regions(model))
                for block in model.blocks:
                    block.site = cuts.Site(
                        (name, block.index), device, model.track, block.module, module, tuple(state)
                    )
                    _hook(block)
                cuts.install(module)
        return model

    # -- needed now

    @contextmanager
    def use(self, models: Sequence[Model], repeats: bool = False) -> Iterator[Sequence[Model]]:
        """These models are needed now: a shared count, never exclusive. Their common weights
        stay bound, and cold-weight eviction passes them by. `repeats`: called again and again
        (a denoise loop), so worth keeping at home."""
        opened: list[Model] = []
        try:
            for model in models:
                if not model.host:
                    self._open_use(model, repeats)
                    opened.append(model)
            yield models
        finally:
            for model in reversed(opened):
                self._close_use(model)

    def _open_use(self, model: Model, repeats: bool) -> None:
        with self.lock:
            if model.uses == 0:
                self._plan(model, repeats)
                model.repeats = repeats
                model.repeating = repeats
            model.uses += 1
            self.clock += 1
            model.last_use = self.clock
            self.last_use = time.monotonic_ns()
        self._enter(model.common)
        self._publish(model.gpu)
        cuts.enter()

    def _close_use(self, model: Model) -> None:
        cuts.leave()
        self._exit(model.common)
        with self.lock:
            model.uses -= 1
            if model.uses == 0 and model.seen:
                self._step(model)
        self._publish(model.gpu)

    def _plan(self, model: Model, repeats: bool) -> None:
        """The watermark as a model's use opens, from its measured reserve. A repeating model is
        wholly at home when it fits beside its reserve and the cold weights it may evict, else a
        spread set; one used once takes only free room and streams the rest. A model never
        measured is wholly at home if it repeats, else keeps only what is resident (it evicts
        nothing to become resident). Within a request the watermark never rises."""
        if not model.blocks or (model.reserve < 0 and repeats):
            return
        if model.reserve < 0:
            want = model.home & model.loaded
        else:
            gpu = model.gpu
            free = torch.cuda.mem_get_info(gpu)[0] + torch.cuda.memory_reserved(gpu)
            room = free - torch.cuda.memory_allocated(gpu) + _resident(model)
            room -= model.common.nbytes + model.reserve
            if repeats:
                room += sum(_resident(m) for m in self._cold(gpu) if m is not model)
            sizes = [model.blocks[b].nbytes for b in model.order]
            if sum(sizes) <= room:
                want = set(model.order)
            else:
                want = set(_spread(model.order, sizes, room - 2 * max(sizes)))
        if self._in_request(model.gpu):
            want &= model.home
        self._set_home(model, want, "plan", 4)

    # -- blocks

    def _enter(
        self,
        block: Block,
        record: bool = True,
        call: tuple[Sequence[object], Mapping[str, object]] = ((), {}),
    ) -> None:
        model = block.model
        if model.host:
            raise Unbound(f"{model.name}: {block.path or 'common'} binds to nothing")
        stream = torch.cuda.current_stream(model.device)
        with self.lock, _disable_current_modes():
            if block.count == 0:
                self._bind(block, stream)
            elif stream != block.stream and not torch.cuda.is_current_stream_capturing():
                stream.wait_stream(block.stream)
            block.count += 1
            if record and block.index >= 0:
                self._passed(block, call)
        if record and block.index >= 0 and not model.held:
            self._prefetch(block)

    def _exit(self, block: Block) -> None:
        stream = torch.cuda.current_stream(block.model.device)
        with self.lock, _disable_current_modes():
            if (
                stream != block.stream
                and block.stream is not None
                and not torch.cuda.is_current_stream_capturing()
            ):
                block.stream.wait_stream(stream)
            block.count -= 1
            if block.count == 0:
                _unbind(block)

    def _bind(self, block: Block, stream: torch.cuda.Stream) -> None:
        model = block.model
        block.stream = stream
        try:
            views: dict[str, dict[str, Tensor]] = {}
            fresh: dict[str, dict[str, Tensor]] = {}
            home = True
            for region in block.regions:
                pinned = self._pin(model, region, stream.cuda_stream)
                home = home and pinned.home
                base = torch.from_dlpack(pinned)
                block.pins.append(base)
                parts = block.parts(region, base)
                views.update(parts)
                if pinned.fresh:
                    fresh.update(parts)
            if fresh and model.fill is not None:
                model.fill(block, fresh)
            _write(block, views)
        except BaseException:
            _unbind(block)
            raise
        if home and block.index >= 0:
            model.loaded.add(block.index)

    def _pin(self, model: Model, region: int, stream: int) -> Pinned:
        """The region's pages, room made in §5.2's order until they map."""
        while True:
            try:
                return model.ws.bind(region, stream)
            except DeviceFull as full:
                target = torch.cuda.mem_get_info(model.gpu)[0] + full.deficit
                if not self._make_room(model.gpu, target, f"bind {model.name}", model):
                    raise Exhausted(
                        model.gpu, full.deficit, f"bind {model.name}", self.stats(False)
                    ) from None

    def _passed(self, block: Block, call: tuple[Sequence[object], Mapping[str, object]]) -> None:
        """A block run again starts the model's next pass (step), at the shape of its first
        block's arguments."""
        model = block.model
        if block.index in model.seen:
            self._step(model)
        if not model.seen:
            model.base = torch.cuda.memory_allocated(model.device)
            torch.cuda.reset_peak_memory_stats(model.device)
            self._shape(model, cuts.shape_of(*call))
        model.seen[block.index] = None

    def _shape(self, model: Model, key: str) -> None:
        """A pass starts at shape `key`: its cuts are noted there, and the peak the memo measured
        for it is the reserve the watermark plans by (inside a request, only lowering it)."""
        shape = model.track.shape = self.memo.shape(model.name, key)
        for request in self._requests(model.gpu):
            request.shapes[(model.name, key)] = shape
        if shape.peak >= 0 and shape.peak != model.reserve:
            model.reserve = shape.peak
            self._plan(model, model.repeats)

    def _step(self, model: Model) -> None:
        if not model.steps:  # the first pass fixes the order prefetch follows
            model.order = [*model.seen, *(b for b in model.order if b not in model.seen)]
            model.position = {b: i for i, b in enumerate(model.order)}
        model.seen.clear()
        resident, paged_in, streamed, stall = _set_stats(self.gpus[model.gpu], model.name)
        last_in, last_streamed, last_stall = model.mark
        model.mark = (paged_in, streamed, stall)
        activations = max(torch.cuda.max_memory_allocated(model.device) - model.base, 0)
        model.peak = max(model.peak, activations)
        cuts.peaked(model.track, activations)
        step = Step(
            model.name,
            model.gpu,
            model.steps,
            model.home_bytes(),
            resident,
            paged_in - last_in,
            streamed - last_streamed,
            stall - last_stall,
            activations,
        )
        model.steps += 1
        for request in self._requests(model.gpu):
            request.steps.append(step)

    def _prefetch(self, block: Block) -> None:
        """Copy the next blocks of the recorded order now, unless they are resident."""
        model = block.model
        at = model.position.get(block.index)
        if at is None:
            return
        for ahead in range(1, _AHEAD + 1):
            nxt = model.order[(at + ahead) % len(model.order)]
            if nxt in model.loaded and nxt in model.home or model.blocks[nxt].count:
                continue
            for region in model.blocks[nxt].regions:
                try:
                    model.ws.prefetch(region)
                except DeviceFull:
                    return

    def _reading(self, blocks: Sequence[Block]) -> _Reading:
        return _Reading(self, blocks)

    # -- room (§5.2)

    def room(self, gpu: int, nbytes: int) -> int:
        """Make `nbytes` free on `gpu` for torch, in §5.2's order; returns what was freed. Raises
        `Exhausted` when everything evictable is gone and it still does not fit."""
        with self.lock:
            before = torch.cuda.mem_get_info(gpu)[0]
            if before < nbytes and not self._make_room(gpu, nbytes, "room", None):
                raise Exhausted(gpu, nbytes - before, "room", self.stats(False))
            after = torch.cuda.mem_get_info(gpu)[0]
            if after < nbytes:
                raise Exhausted(gpu, nbytes - after, "room", self.stats(False))
            return max(after - before, 0)

    def _make_room(self, gpu: int, target: int, why: str, asking: Model | None) -> bool:
        """Free bytes on `gpu` until `target` are free: rungs 1-2, then rung 4 (the catch cuts
        the op between them, rung 3). False when nothing could be freed."""
        freed = self._free_upto(gpu, target, why)
        return self._lower_until(gpu, target, why, asking) or freed

    def _free_upto(self, gpu: int, target: int, why: str) -> bool:
        """Idle bytes (rung 1), then cold weights least recently used first, warm ones last, in
        this process and others (rung 2), until `target` bytes are free."""
        freed = self._idle(gpu, target, None)
        if _free(gpu) >= target:
            return True
        for model in self._cold(gpu):
            freed += self._evict(model, why, 2, target - _free(gpu))
            if _free(gpu) >= target:
                return True
        freed += self._ask(gpu, target)
        return freed > 0 or _free(gpu) >= target

    def _lower_until(self, gpu: int, target: int, why: str, asking: Model | None) -> bool:
        """Rung 4: the hot models' watermarks fall from the top of their spread sets, the
        asker's last, until `target` bytes are free."""
        if _free(gpu) >= target:
            return False
        freed = 0
        hot = [m for m in self.models.values() if m.gpu == gpu and m.uses and not m.held]
        for model in sorted(hot, key=lambda m: (m is asking, m.kept, m.last_use)):
            while model.home and _free(gpu) < target:
                freed += self._lower(model, why)
            if _free(gpu) >= target:
                break
        return freed > 0

    # -- the catch's rungs (`cuts.Rungs`)

    def free(self, device: torch.device, need: int) -> bool:
        if device.type != "cuda" or device.index not in self.gpus:
            return False
        with self.lock:
            return self._free_upto(device.index, need, "out of memory")

    def lower(self, device: torch.device, need: int) -> bool:
        """At least one block lower when `need` is free already (an unsaid need)."""
        if device.type != "cuda" or device.index not in self.gpus:
            return False
        with self.lock:
            target = max(need, _free(device.index) + 1)
            return self._lower_until(device.index, target, "out of memory", None)

    def available(self, device: torch.device) -> int:
        if device.type != "cuda" or device.index is None:
            return 0
        idle = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        return _free(device.index) + idle

    def exhausted(self, device: torch.device, need: int, op: str) -> Exception:
        return Exhausted(device.index or 0, need, op, self.stats(False))

    def _idle(self, gpu: int, target: int, asking: Cache | None) -> int:
        """Rung 1: torch's slack, the core's idle chunks, then caches whose owners give them."""
        reserved = torch.cuda.memory_reserved(gpu)
        torch.cuda.empty_cache()
        freed = reserved - torch.cuda.memory_reserved(gpu) + int(self.gpus[gpu].trim())
        for cache in list(self.caches):
            short = target - _free(gpu)
            if short <= 0:
                break
            if cache.gpu == gpu and cache is not asking and cache.nbytes:
                shed = min(int(cache.shed(min(short, cache.nbytes))), cache.nbytes)
                cache.nbytes -= shed
                freed += shed
                torch.cuda.empty_cache()
        return freed

    def _cold(self, gpu: int) -> list[Model]:
        cold = [m for m in list(self.models.values()) if m.gpu == gpu and not m.uses and not m.held]
        return sorted(cold, key=lambda m: (m.kept, m.last_use))

    def _ask(self, gpu: int, target: int) -> int:
        """Rung 2 across processes: the ledger asks other rows for their cold bytes, each
        evicting in its own process, and this waits for every answer (or death)."""
        held = self.rows.get(gpu)
        if held is None:
            return 0
        table, row = held
        before = _free(gpu)
        for victim in table.ask(target - before, row.id):
            table.wait(victim)
        return max(_free(gpu) - before, 0)

    def _answer(self, gpu: int, row: Row) -> None:
        """Another process's ask of this row: idle bytes and cold weights go, the facts are
        published, then the ask is acked. It takes no lock: pages in use are pinned, and a
        model a use opens meanwhile only refaults."""
        with torch.cuda.device(gpu):
            while True:
                want = row.wait_want()
                freed = self._idle(gpu, _free(gpu) + want, None)
                for model in self._cold(gpu):
                    if freed >= want:
                        break
                    freed += self._evict(model, "asked", 2, want - freed)
                self._publish(gpu)
                row.ack(freed)

    def _publish(self, gpu: int) -> None:
        """This row's facts: at use boundaries, request ends and after an ask (§5.6)."""
        held = self.rows.get(gpu)
        if held is None:
            return
        models = [m for m in list(self.models.values()) if m.gpu == gpu]
        resident = {m.name: _resident(m) for m in models}
        cold = [m for m in models if not m.uses and not m.held]
        reserved, allocated = torch.cuda.memory_reserved(gpu), torch.cuda.memory_allocated(gpu)
        held[1].publish(
            context=self.context,
            resident=sum(resident.values()),
            cold=sum(resident[m.name] for m in cold),
            kept=sum(resident[m.name] for m in cold if m.kept),
            pinned=sum(resident[m.name] for m in models if m.uses or m.held),
            torch_reserved=reserved,
            torch_allocated=allocated,
            cache=sum(c.nbytes for c in list(self.caches) if c.gpu == gpu),
            idle=reserved - allocated,
            active_peak=torch.cuda.max_memory_allocated(gpu) if self._in_request(gpu) else 0,
            last_use=self.last_use,
        )

    def _evict(self, model: Model, why: str, rung: int, need: int | None = None) -> int:
        """Pages of `model` nobody is using: all of them, or with `need` its lowest-ranked
        blocks until that many bytes are free, its common last."""
        freed = 0
        if need is not None:
            sizes = [model.blocks[b].nbytes for b in model.order]
            for b in reversed(_spread(model.order, sizes, model.nbytes)):
                if freed >= need:
                    break
                if b in model.loaded and not model.blocks[b].count:
                    freed += int(model.ws.evict(model.blocks[b].regions))
                    model.loaded.discard(b)
        if need is None or freed < need:
            freed += int(model.ws.evict(None))
            model.loaded = {b for b in model.loaded if model.blocks[b].count}
        if freed:
            self._evicted(model, rung, why, freed)
        return freed

    def _lower(self, model: Model, why: str) -> int:
        """Rung 4: the last-ranked block of the spread set leaves home for the request."""
        ranked = _spread(model.order, [model.blocks[b].nbytes for b in model.order], model.nbytes)
        last = next(b for b in reversed(ranked) if b in model.home)
        return self._set_home(model, model.home - {last}, why, 4)

    def _set_home(self, model: Model, want: set[int], why: str, rung: int) -> int:
        gone = model.home - want
        if want == model.home:
            return 0
        model.home = set(want)
        model.ws.home(_home_regions(model))
        if not gone:
            return 0
        if self._in_request(model.gpu):
            model.lowered = True
        freed = int(model.ws.evict([r for b in gone for r in model.blocks[b].regions]))
        model.loaded -= {b for b in gone if not model.blocks[b].count}
        if freed:
            self._evicted(model, rung, why, freed)
        return freed

    def _evicted(self, model: Model, rung: int, why: str, nbytes: int) -> None:
        self.evictions += 1
        eviction = Eviction(model.name, model.gpu, rung, why, nbytes)
        for request in self._requests(model.gpu):
            request.evictions.append(eviction)

    def _grant(self, cache: Cache, grow: int) -> bool:
        """A cache grows into free, idle or cold bytes (rungs 1-2), never a hot model's."""
        gpu = cache.gpu
        with self.lock:
            idle = torch.cuda.memory_reserved(gpu) - torch.cuda.memory_allocated(gpu)
            if _free(gpu) + idle >= grow:
                return True
            self._idle(gpu, grow, cache)
            for model in self._cold(gpu):
                if _free(gpu) >= grow:
                    break
                self._evict(model, cache.owner, 2, grow - _free(gpu))
            return _free(gpu) >= grow

    # -- caches, warm set, holds

    def cache(self, owner: str, gpu: int, shed: Shed) -> Cache:
        cache = Cache(self, owner, gpu, shed)
        with self.lock:
            self.caches[cache] = None
        return cache

    def keep(self, models: Sequence[Model]) -> None:
        """The controller's warm set: evicted last, restored first."""
        kept = {id(m) for m in models}
        with self.lock:
            for model in self.models.values():
                model.kept = id(model) in kept

    def _hold(self, model: Model) -> Hold:
        with self.lock:
            if model.host or len(model.home) != len(model.blocks):
                raise Unbound(f"{model.name} streams part of its weights: it cannot be held")
            model.held += 1
            entered: list[Block] = []
            try:
                for block in (model.common, *model.blocks):
                    self._enter(block, record=False)
                    entered.append(block)
            except BaseException:
                for block in entered:
                    self._exit(block)
                model.held -= 1
                raise
        return Hold(model)

    def _release(self, model: Model) -> None:
        with self.lock:
            for block in (model.common, *model.blocks):
                self._exit(block)
            model.held -= 1

    def _close(self, model: Model) -> None:
        with self.lock:
            if model.closed:
                return
            model.closed = True
            self.models.pop(model.name, None)
            for block in (model.common, *model.blocks):
                _unhook(block)
                block.count = 0
                _unbind(block)
            model.ws.close()

    # -- requests (§5.7)

    @contextmanager
    def request(self, id: str, gpus: Sequence[int]) -> Iterator[Request]:
        """Everything the request allocates is freed on exit, failed or not. A capacity failure
        never poisons; bytes package state still holds show as `leftover`."""
        mine = tuple(g for g in gpus if g in self.gpus)
        request = Request(
            id, mine, {g: torch.cuda.memory_allocated(g) for g in mine}, self._copies(mine)
        )
        request.retried = self.memo.retries
        with self.lock:
            self.active[request] = None
        stack: list[Request] = self.local.__dict__.setdefault("requests", [])
        stack.append(request)
        try:
            yield request
        except BaseException as exc:
            _clear_frames(exc)
            raise
        finally:
            stack.remove(request)
            self._end(request)

    def _end(self, request: Request) -> None:
        gc.collect()
        for gpu in request.gpus:
            torch.cuda.synchronize(gpu)
        torch.cuda.empty_cache()
        leftover = tuple(
            (g, max(torch.cuda.memory_allocated(g) - request.start[g], 0)) for g in request.gpus
        )
        copied = sum(
            n - request.copies.get(name, 0) for name, n in self._copies(request.gpus).items()
        )
        request.stats = RequestStats(
            request.id,
            tuple(request.steps),
            tuple(request.evictions),
            leftover,
            copied,
            self.memo.retries - request.retried,
            tuple(_peak(s) for s in request.shapes.values()),
        )
        with self.lock:
            self.active.pop(request, None)
            self.ended.append(request.stats)
            for gpu in request.gpus:
                if not self._in_request(gpu):
                    self._restore(gpu)
        for gpu in request.gpus:
            self._publish(gpu)

    def _restore(self, gpu: int) -> None:
        """After the last request on `gpu`: each model's reserve is what that request measured,
        and every watermark rises again. Pages copy home in priority order (the warm set, then
        models whose use repeats, then the rest, most recently used first): a model takes free
        room, then cold pages of models after it, always leaving the largest reserve and a
        streamed model's slots. The copies are prefetches: an ask evicts them again at once."""
        models = [m for m in self.models.values() if m.gpu == gpu]
        for model in models:
            if model.peak:
                model.reserve, model.peak = model.peak, 0
            model.lowered = False
        ranked = sorted(models, key=lambda m: (not m.kept, not m.repeating, -m.last_use))
        slots = max((_SLOTS * _largest(m) for m in models if not m.repeating), default=0)
        room = _free(gpu) - max((max(m.reserve, 0) for m in models), default=0) - slots
        for i, model in enumerate(ranked):
            lower = [m for m in ranked[i + 1 :] if not m.uses and not m.held]
            for b in model.order:
                if b in model.home and b in model.loaded:
                    continue
                block = model.blocks[b]
                while room < block.nbytes and lower:
                    room += self._evict(lower[-1], "restore", 2, block.nbytes - room)
                    if room < block.nbytes:  # it gave all it had
                        lower.pop()
                if room < block.nbytes:
                    return
                room -= block.nbytes
                model.home.add(b)
                model.ws.home(_home_regions(model))
                for region in block.regions:
                    try:
                        model.ws.prefetch(region)
                    except DeviceFull:
                        return
                model.loaded.add(b)

    def _copies(self, gpus: Sequence[int]) -> dict[str, int]:
        """Each model's weight bytes copied to `gpus` since it registered, home or slots."""
        return {m.name: sum(_counters(m)[1:3]) for m in list(self.models.values()) if m.gpu in gpus}

    def _requests(self, gpu: int) -> list[Request]:
        """The request a step or an eviction belongs to: this thread's, else the GPU's first."""
        mine = [r for r in self.local.__dict__.get("requests", ()) if gpu in r.gpus]
        return mine[-1:] or [r for r in self.active if gpu in r.gpus][:1]

    def _in_request(self, gpu: int) -> bool:
        return any(gpu in r.gpus for r in self.active)

    # -- stats (§5.8)

    def stats(self, drain: bool = True) -> Stats:
        with self.lock:
            models = tuple(
                ModelStats(
                    m.name,
                    m.gpu,
                    m.nbytes,
                    m.home_bytes(),
                    *_counters(m),
                    m.uses,
                    m.kept,
                    m.reserve,
                )
                for m in self.models.values()
            )
            ended = tuple(self.ended)
            if drain:
                self.ended.clear()
            peaks = tuple(_peak(s) for s in list(self.memo.shapes.values()))
            return Stats(models, ended, self.evictions, peaks, self.memo.retries)


class _Reading:
    """A placeholder's blocks bound for one op outside their forward."""

    def __init__(self, memory: Memory, blocks: Sequence[Block]) -> None:
        self.memory, self.blocks = memory, blocks
        self.entered: list[Block] = []

    def __enter__(self) -> None:
        try:
            for block in self.blocks:
                self.memory._enter(block, record=False)
                self.entered.append(block)
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *_: object) -> None:
        while self.entered:
            self.memory._exit(self.entered.pop())


# ----------------------------------------------------------------------------- binding


def regions_of(weights: WeightGrant, blocks: Sequence[str]) -> list[list[str]]:
    """The component's weight-set regions, cut at `blocks`: what `register` maps and what a
    sealed host layout of it must hold."""
    prefix = f"{weights.component}/"
    keys = None if weights.keys is None else set(weights.keys)
    return [
        kept
        for region in tensorfs.regions(weights.ck, _WHOLE, [prefix + b for b in blocks])
        if (
            kept := [
                n
                for n in region
                if n.startswith(prefix) and (keys is None or n[len(prefix) :] in keys)
            ]
        )
    ]


def innermost_blocks(module: nn.Module) -> list[str]:
    """The innermost instances of the root's `_no_split_modules` (or of a subclass of one)."""
    names = set(getattr(module, "_no_split_modules", None) or ())
    found = [
        path
        for path, sub in module.named_modules()
        if path and any(kind.__name__ in names for kind in type(sub).__mro__)
    ]
    return [a for a in found if not any(b.startswith(a + ".") for b in found)]


def _layout(
    model: Model, prefix: str, paths: Sequence[str], regions: Sequence[Sequence[str]]
) -> None:
    """Blocks (in header order) and common, from the weight set's regions and parts."""
    by_path = {p: Block(model, -1, p, model.module.get_submodule(p)) for p in paths}
    owner: list[Block] = []
    for index, names in enumerate(regions):
        homes = {_block_of(n[len(prefix) :], paths) for n in names}
        block = model.common
        if len(homes) == 1 and (path := homes.pop()) is not None:
            block = by_path[path]
        block.regions.append(index)
        owner.append(block)
    for part in model.ws.parts:
        key, _, role = part.key[len(prefix) :].partition("#")
        role = role or "value"
        shape = tuple(part.shape)
        if part.dtype == "F4":
            shape = (*shape[:-1], shape[-1] // 2)
        at = _PartAt(part.region, part.offset, part.nbytes, DTYPES[part.dtype], shape)
        owner[part.region].tensors.setdefault(key, {})[role] = at
        model.sizes[part.region] = max(model.sizes.get(part.region, 0), part.offset + part.nbytes)
    model.blocks = sorted((b for b in by_path.values() if b.regions), key=lambda b: b.regions[0])
    for index, block in enumerate(model.blocks):
        block.index = index


def _block_of(key: str, paths: Sequence[str]) -> str | None:
    found = [p for p in paths if key == p or key.startswith(p + ".")]
    return max(found, key=len) if found else None


def _sites(module: nn.Module) -> dict[str, list[_Site]]:
    """Every module attribute naming each tensor, tied names included."""
    by_tensor: dict[int, list[_Site]] = {}
    names: dict[str, int] = {}
    for path, sub in module.named_modules(remove_duplicate=False):
        for table, parameter in ((sub._parameters, True), (sub._buffers, False)):
            for attr, tensor in table.items():
                if tensor is None:
                    continue
                names[f"{path}.{attr}" if path else attr] = id(tensor)
                sites = by_tensor.setdefault(id(tensor), [])
                if not any(s.table is table and s.attr == attr for s in sites):
                    sites.append(_Site(table, attr, parameter))
    return {name: by_tensor[t] for name, t in names.items()}


def _install(model: Model) -> None:
    """Every meta weight becomes a placeholder on the execution device, owned by its block."""
    by_path = {b.path: b for b in model.blocks}
    done: set[int] = set()
    paths = list(by_path)
    for name, group in model.sites.items():
        first = group[0]
        meta = first.table[first.attr]
        if meta is None or id(meta) in done or meta.device.type != "meta":
            continue
        done.add(id(meta))
        path = _block_of(name, paths)
        block = model.common if path is None else by_path[path]
        held: Tensor = Placeholder(meta.shape, meta.dtype, model.device, block, first)
        if first.parameter:
            held = nn.Parameter(held, requires_grad=False)
        for site in group:
            site.table[site.attr] = held


def _hook(block: Block) -> None:
    module = block.module
    assert module is not None
    memory = block.model.memory
    block.own_forward = "forward" in module.__dict__
    forward = block.forward = module.forward
    site = block.site
    assert site is not None

    def bound(*args: object, **kwargs: object) -> object:
        memory._enter(block, call=(args, kwargs))
        try:
            return cuts.run(site, forward, args, kwargs)
        finally:
            memory._exit(block)

    module.forward = bound  # type: ignore[method-assign]


def _unhook(block: Block) -> None:
    module = block.module
    if module is None or block.forward is None:
        return
    if block.own_forward:
        module.forward = block.forward  # type: ignore[method-assign]
    else:
        del module.forward
    block.forward = None


def _write(block: Block, views: Mapping[str, Mapping[str, Tensor]]) -> None:
    """Point the block's destinations at its views, through the model's `bind`."""
    model = block.model
    for key, parts in views.items():
        if model.bind is not None:
            made = model.bind(key, parts)
        elif set(parts) == {"value"}:
            made = {key: parts["value"]}
        else:
            raise Unbound(f"{model.name}: {key} is encoded ({sorted(parts)}) and no bind was given")
        for name, value in made.items():
            for site in model.sites.get(name, ()):
                held = site.table[site.attr]
                block.written.append((site, held))
                if held is not None and held.dtype != value.dtype:
                    value = value.to(held.dtype)  # a converted module (`module.to(dtype)`)
                site.table[site.attr] = (
                    nn.Parameter(value, requires_grad=False) if site.parameter else value
                )


def _unbind(block: Block) -> None:
    """Placeholders back; the views go, and with them the pins (released on the bind stream)."""
    for site, held in reversed(block.written):
        site.table[site.attr] = held
    block.written.clear()
    block.pins.clear()
    block.stream = None


def _bind_host(model: Model) -> None:
    """Every weight once: CPU params view the host pages; an Apple device's are copied in."""
    for block in (model.common, *model.blocks):
        views: dict[str, dict[str, Tensor]] = {}
        for region in block.regions:
            base = torch.from_dlpack(model.ws.bind(region, 0))
            parts = block.parts(region, base)
            if model.device.type != "cpu":
                parts = {k: {r: t.to(model.device) for r, t in v.items()} for k, v in parts.items()}
            views.update(parts)
        _write(block, views)
        block.written.clear()


def _home_regions(model: Model) -> list[int]:
    return [
        *model.common.regions,
        *(r for b in sorted(model.home) for r in model.blocks[b].regions),
    ]


def _peak(shape: cuts.Shape) -> Peak:
    cut = tuple(f"{name} {size}" for name, size in shape.cuts.items())
    return Peak(shape.model, shape.shape, shape.peak, cut)


def _largest(model: Model) -> int:
    return max((b.nbytes for b in model.blocks), default=0)


def _free(gpu: int) -> int:
    return int(torch.cuda.mem_get_info(gpu)[0])


def _resident(model: Model) -> int:
    return _counters(model)[0]


def _counters(model: Model) -> tuple[int, int, int, int]:
    if model.host:
        return model.nbytes, 0, 0, 0
    return _set_stats(model.memory.gpus[model.gpu], model.name)


def _set_stats(gpu: Gpu, name: str) -> tuple[int, int, int, int]:
    """One weight set's resident bytes, and its cumulative bytes paged in, streamed and stall."""
    for s in gpu.stats().sets:
        if s.name == name:
            return int(s.resident), int(s.paged_in), int(s.slot_bytes), int(s.stall_ns)
    return 0, 0, 0, 0


def _spread(order: Sequence[int], sizes: Sequence[int], room: int) -> list[int]:
    """Blocks to keep at home within `room` bytes, ranked so that every prefix is spread evenly
    along `order` (bit-reversed positions): each streamed block's copy overlaps a resident
    block's compute."""
    n = len(order)
    if not n:
        return []
    bits = max(n - 1, 1).bit_length()
    ranked = list(
        dict.fromkeys(int(f"{i:0{bits}b}"[::-1], 2) * n >> bits for i in range(1 << bits))
    )
    kept, total = [], 0
    for at in ranked:
        if total + sizes[at] <= room:
            total += sizes[at]
            kept.append(order[at])
    return kept


def _clear_frames(exc: BaseException | None) -> None:
    """Drop a failed request's frame locals: they hold its tensors (§5.7)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
        exc = exc.__cause__ or exc.__context__
