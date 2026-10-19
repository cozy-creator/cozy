"""varena's memory for torch modules (memory-varena.md §2, §5).

    mem = Memory.open(torch.device("cuda", 0), LedgerGrant(0, fd, events))
    unet._varena = Declared(WeightGrant(ck, "unet", "sha256:…"))  # the author API's
    mem.load(unet)
    with mem.request("r1"):
        unet(x)

A block's params are views of its region's pages only while the block runs (the view is the
pin); zero-byte placeholders stand in otherwise, and an op on one binds its block for that op.
Bytes rank in use > warm (loaded at GPU level) > idle, least recently used first within each.
Room is made in §5.2's order; the hot watermark falls only within a request and refills after
it. On a CPU or Apple device every weight binds once.
"""

from __future__ import annotations

import gc
import threading
import time
import traceback
from collections.abc import Callable, Collection, Hashable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Protocol

import tensorfs
import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import _disable_current_modes
from torch.utils._pytree import tree_map_only
from torch.utils.hooks import RemovableHandle

from . import DeviceFull, Gpu, Host, Ledger, Pinned, Row, WeightSet, cuts, seal

#: Regions are cut only at block boundaries: one per block, plus each run of common tensors.
_WHOLE = 1 << 62
#: Blocks copied ahead of the one entered, in the recorded order.
_AHEAD = 2
#: A streamed model's scratch slots, in its largest blocks (the core keeps 2-3).
_SLOTS = 3
#: TensorFS dtype names; an `F4` part is u8 pairs.
DTYPES: dict[str, torch.dtype] = {
    name: getattr(torch, kind)
    for name, kind in (
        pair.split("=")
        for pair in "F64=float64 F32=float32 F16=float16 BF16=bfloat16 F8_E4M3=float8_e4m3fn "
        "F8_E5M2=float8_e5m2 F8_E8M0=float8_e8m0fnu I64=int64 I32=int32 I16=int16 I8=int8 "
        "U8=uint8 BOOL=bool F4=uint8".split()
    )
}


class Mode(Enum):
    SHARED = "shared"
    DEDICATED = "dedicated"


class Level(Enum):
    """Where `load` keeps a model: the host tier only, or mapped on the card too (warm)."""

    RAM = "ram"
    GPU = "gpu"


@dataclass(frozen=True, slots=True)
class WeightGrant:
    """TensorD's grant for one component: its header, the binding's manifest, the sealed host
    layout or the chunk files `(sha256, length, fd)`, and the stored tensors the module builds."""

    ck: tensorfs.Checkpoint
    component: str
    manifest: str
    host_fd: int | None = None
    keys: Collection[str] | None = None
    files: Sequence[tuple[str, int, int]] = ()

    @property
    def key(self) -> str:
        return f"{self.manifest}/{self.component}"


@dataclass(frozen=True, slots=True)
class LedgerGrant:
    gpu: int
    fd: int
    events: int


class Exhausted(Exception):
    """It still does not fit with every other evictable byte gone: §2's one memory error, with
    each model's resident bytes and the memo's peaks then."""

    def __init__(
        self, gpu: int, need: int, op: str, resident: Mapping[str, int], peaks: tuple[Peak, ...]
    ) -> None:
        super().__init__(f"GPU {gpu}: {op} needs {need} B more and nothing else can be freed")
        self.gpu, self.need, self.op, self.resident, self.peaks = gpu, need, op, resident, peaks


class Unbound(Exception):
    """An op on a placeholder that binds to nothing."""


class _Short(Exception):
    """A bind found no room: `_enter` waits for another request's bytes, or `Exhausted`."""


#: `bind(key, parts)`: a stored tensor's part views -> the module tensors they become, by name.
type Bind = Callable[[str, Mapping[str, Tensor]], Mapping[str, Tensor]]


class Fill(Protocol):
    """A page-in transform (a LoRA bake) on fresh bytes. `prepare()`, as each forward starts,
    readies the module and names what filling adds; pages filled under another name refill."""

    def __call__(self, block: Block, parts: Mapping[str, Mapping[str, Tensor]], /) -> None: ...

    def prepare(self) -> Hashable: ...


class Shrink(Protocol):
    def __call__(self, nbytes: int, /) -> int: ...


@dataclass(frozen=True, slots=True)
class Declared:
    """What the author API attaches as `module._varena`: its weights, its blocks (default: the
    innermost `_no_split_modules`), the attributes a retried block puts back (`state`), `fill`,
    and `bind` (default: a plain part as itself)."""

    weights: WeightGrant | None = None
    blocks: Sequence[str] | None = None
    state: tuple[str, ...] = ()
    fill: Fill | None = None
    bind: Bind | None = None


# ------------------------------------------------------------------------------- stats


@dataclass(frozen=True, slots=True)
class Step:
    """One pass of a model: bytes at home and resident, copied home and streamed, the prefetch
    stall, and torch's allocated peak above the pass's start."""

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
class Peak:
    """A model's activation peak at one shape key (-1: unmeasured) and its cuts in force."""

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
    #: weight bytes copied while it ran, home or through slots
    copied: int
    retries: int = 0
    peaks: tuple[Peak, ...] = ()


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
    def __new__(cls, shape, dtype, device, block, site):  # type: ignore[no-untyped-def]
        held: Placeholder = Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls, tuple(shape), dtype=dtype, device=device, storage_size=0
        )
        held.block, held.site = block, site
        return held

    def __repr__(self) -> str:  # type: ignore[override]
        return f"Placeholder({tuple(self.shape)}, {self.dtype}, {self.device})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        aten, kwargs = torch.ops.aten, kwargs or {}
        if func in (aten.detach.default, aten.alias.default):
            return args[0]
        if func is aten._to_copy.default:
            held, device = args[0], kwargs.get("device")
            device = held.device if device is None else torch.device(device)
            return cls(held.shape, kwargs.get("dtype") or held.dtype, device, held.block, held.site)
        for i, argument in enumerate(func._schema.arguments):
            written = args[i] if i < len(args) else kwargs.get(argument.name)
            if argument.alias_info is not None and argument.alias_info.is_write:
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
    """What is bound at `p`'s site, in `p`'s dtype and device."""
    value = p.site.table[p.site.attr] if p.site is not None else None
    if value is None or isinstance(value, Placeholder):
        raise Unbound(f"a weight of {p.block.model.name if p.block else '?'} binds to nothing")
    return value.to(dtype=p.dtype, device=p.device)


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
    """One block, or a model's `common` (index -1), and while anyone runs it, its pins."""

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
        view = lambda a: base[a.offset : a.offset + a.nbytes].view(a.dtype).view(a.shape)  # noqa: E731
        return {
            key: {role: view(a) for role, a in roles.items()}
            for key, roles in self.tensors.items()
            if next(iter(roles.values())).region == region
        }


class Model:
    """One loaded component: its weight set, blocks, watermark and per-step stats."""

    def __init__(
        self,
        memory: Memory,
        name: str,
        key: str,
        module: nn.Module,
        ws: WeightSet,
        declared: Declared,
    ) -> None:
        self.memory, self.name, self.key, self.module, self.ws = memory, name, key, module, ws
        self.gpu, self.device = memory.gpu, memory.device
        self.fill, self.bind = declared.fill, declared.bind
        self.sites: dict[str, list[_Site]] = {}
        self.sizes: dict[int, int] = {}
        self.blocks: list[Block] = []
        self.common = Block(self, -1, "", module)
        #: block indexes in the order its first pass ran them, and their positions
        self.order: list[int] = []
        self.position: dict[int, int] = {}
        self.seen: dict[int, None] = {}
        #: blocks at their home VA (the rest stream through slots)
        self.home: set[int] = set()
        self.level, self.closed = Level.GPU, False
        #: forwards running now, and whether the first bound common for them
        self.flight, self.aloft, self.held = 0, False, 0
        #: passes in its last request (this one's so far, if more)
        self.repeats = self.last_use = self.steps = self.peak = self.base = 0
        self.planned = -1
        self.filled: Hashable | None = None
        #: torch bytes its passes allocated above their start in its last request; -1 unmeasured
        self.reserve = -1
        self.mark = (0, 0, 0)
        self.handles: list[RemovableHandle] = []
        self.track = cuts.Track(name)

    @property
    def host(self) -> bool:
        return self.gpu < 0

    @property
    def nbytes(self) -> int:
        return sum(self.sizes.values())

    def home_bytes(self) -> int:
        return self.common.nbytes + sum(self.blocks[b].nbytes for b in self.home)

    def evict(self) -> int:
        """Give back every resident page nobody is using: they refill, and `fill` runs again."""
        with self.memory.lock:
            return self.memory._evict(self, "refill", 0)

    def close(self) -> None:
        self.memory._close(self)


class Hold:
    def __init__(self, model: Model, cache: Cache) -> None:
        self.model, self.cache, self.open = model, cache, True

    def close(self) -> None:
        if self.open:
            self.open = False
            self.model.memory._release(self.model, self.cache)


class Cache:
    """Long-lived bytes outside a forward (KV slots, graph pools, a held model), counted as
    `nbytes` and given back through `shrink` among warm bytes (§5.2)."""

    def __init__(self, memory: Memory, owner: str, shrink: Shrink) -> None:
        self.memory, self.owner, self.shrink = memory, owner, shrink
        self.nbytes, self.last_use = 0, memory.clock
        self.holding: dict[Model, int] = {}

    def serving(self) -> AbstractContextManager[None]:
        """The owner's own loop, outside any request: its ops are caught like a request's."""
        self.last_use = self.memory.clock
        return cuts.caught()

    def hold(self, model: Model) -> Hold:
        """Every block of `model` bound until the hold closes (a CUDA graph's addresses). Only a
        model wholly at home can be held."""
        return self.memory._hold(model, self)

    def close(self) -> None:
        with self.memory.lock:
            self.memory.caches.pop(self, None)


@dataclass(eq=False, slots=True)
class Request:
    id: str
    #: torch's allocated bytes, and each model's weight bytes copied, as it began
    start: int
    copies: dict[str, int]
    #: TensorD's `<construction>|<shape>` for it (the ledger's dispatch)
    key: str | None
    retried: int
    steps: list[Step] = field(default_factory=list)
    evictions: list[Eviction] = field(default_factory=list)
    passes: dict[str, int] = field(default_factory=dict)
    #: its peak above its start, what its key's next run's `fits` reserves
    peak: int = 0
    shapes: dict[tuple[str, str], cuts.Shape] = field(default_factory=dict)
    stats: RequestStats | None = None


# --------------------------------------------------------------------------- the memory


class Memory:
    """This process's device: every loaded model, cache and request on it."""

    def __init__(
        self, device: torch.device, ledger: LedgerGrant | None, mode: Mode, limit: int | None
    ) -> None:
        cuda = device.type == "cuda"
        self.mode, self.device, self.gpu = mode, device, (device.index or 0) if cuda else -1
        self.core: Gpu | Host = Gpu(self.gpu) if cuda else Host()
        #: this process's device bytes outside torch and the weights (its CUDA context)
        self.context = self.last_use = self.clock = self.epoch = self.evictions = 0
        self.models: dict[str, Model] = {}
        self.by_module: dict[nn.Module, Model] = {}
        self.caches: dict[Cache, None] = {}
        self.lock, self.local = threading.RLock(), threading.local()
        self.active: dict[Request, None] = {}
        #: host tiers sealed here for the executor to deposit
        self.deposits: list[tuple[Model, int]] = []
        #: its requests waiting for room (§2), counted under `freeing`
        self.waiting = 0
        self.freeing = threading.Lock()
        self.memo = cuts.Memo()
        cuts.serve(self.memo, self)
        self.row: tuple[Ledger, Row] | None = None
        #: a ledger capacity below the card (`gpu_memory_limit`, §5.6), and the card's total
        self.limit: int | None = None
        self.total = 0
        #: TensorD's ledger, which takes host-tier deposits; standalone, one of its own (§2)
        self.granted = ledger is not None
        if isinstance(self.core, Gpu):
            self.total = int(torch.cuda.mem_get_info(self.gpu)[1])
            if ledger is not None:
                table = Ledger.adopt(ledger.fd, ledger.events)
            else:
                uuid = str(torch.cuda.get_device_properties(self.gpu).uuid)
                table = Ledger.create(uuid, min(self.total, limit or self.total))
            self.row = (table, table.claim())
            self.core.serve(self.row[1])  # other processes' asks, answered in Rust
            self.limit = table.capacity if table.capacity < self.total else None

    @classmethod
    def open(
        cls,
        device: torch.device | None = None,
        ledger: LedgerGrant | None = None,
        mode: Mode = Mode.SHARED,
        limit: int | None = None,
    ) -> Memory:
        """A TensorD executor's memory (`ledger`), or a standalone one (an engine using varena on
        its own): GPU 0 by default, `limit` a cap like `gpu_memory_limit`."""
        if device is None:
            device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        return cls(device, ledger, mode, limit)

    @property
    def cuda(self) -> bool:
        return self.gpu >= 0

    # -- loading

    def load(
        self,
        module: nn.Module,
        level: Level | str = Level.GPU,
        *,
        ck: tensorfs.Checkpoint | None = None,
        component: str | None = None,
    ) -> Model:
        """Load `module`, built on meta, at `level` from `module._varena`; standalone, from a
        store's checkpoint `ck` (`component`: its only one by default). Under TensorD the first
        load seals its host tier unless one was granted (`sealed()`). At GPU level it is warm and
        fills free room now; at RAM level its resident pages are idle."""
        level = Level(level)
        declared = _declared(module)
        if ck is not None and declared.weights is None:
            names = [component] if component is not None else list(ck.tensors)
            if len(names) != 1 or ck.manifest is None:
                raise Unbound(f"name the component of {sorted(ck.tensors)} to load (`component=`)")
            grant = WeightGrant(ck, names[0], f"sha256:{ck.manifest.sha256}")
            vars(module)["_varena"] = replace(declared, weights=grant)
        with self.lock:
            model = self.by_module.get(module) or self._register(module)
            model.level = level
            if self.cuda:
                model.ws.warm(level is Level.GPU)
                if level is Level.GPU and not self.active:
                    self._restore()
        self._publish()
        return model

    def sealed(self) -> list[tuple[Model, int]]:
        """Host tiers sealed since the last call, each fd the caller's to deposit and close."""
        with self.lock:
            deposits, self.deposits = self.deposits, []
        return deposits

    def _register(self, module: nn.Module) -> Model:
        declared = _declared(module)
        grant = declared.weights
        if grant is None:
            raise Unbound(f"{type(module).__name__}: no weights attached (`Declared.weights`)")
        paths, regions = blocks_of(module), regions_of(module)
        name, n = grant.key, 1
        while name in self.models:
            n += 1
            name = f"{grant.key}#{n}"
        files, host_fd = list(grant.files) or None, grant.host_fd
        if host_fd is None and self.cuda and self.granted:
            host_fd = seal(grant.ck, regions, name, files=files)
        ws = self.core.register(name, grant.ck, regions, host_fd, files)
        model = Model(self, name, grant.key, module, ws, declared)
        if host_fd is not None and grant.host_fd is None:
            self.deposits.append((model, host_fd))
        _layout(model, f"{grant.component}/", paths, regions)
        model.sites = _sites(module)
        _install(model)
        model.order = [b.index for b in model.blocks]
        model.position = {b: i for i, b in enumerate(model.order)}
        model.home = set(model.order)
        self.models[name], self.by_module[module] = model, model
        if model.host:
            _bind_host(model)
            return model
        model.ws.home(_home_regions(model))
        for b in model.blocks:
            b.site = cuts.Site(
                (name, b.index), self.device, model.track, b.module, module, declared.state
            )
            _hook(b)
        cuts.install(module)
        _fly(model)
        return model

    # -- in use: a forward of the model runs

    def _start(self, model: Model) -> None:
        """The first forward of several readies its fill and binds common. A raise here still
        lands (`always_call`)."""
        with self.lock:
            model.flight += 1
            if model.flight > 1:
                return
            self.clock += 1
            model.last_use, self.last_use = self.clock, time.monotonic_ns()
            model.ws.busy(True)  # another process's ask never takes a model in use
            self._fresh(model)
        self._enter(model.common, record=False)
        with self.lock:
            model.aloft = True
        self._publish()

    def _land(self, model: Model) -> None:
        with self.lock:
            model.flight -= 1
            if model.flight:
                return
            model.ws.busy(False)
            if model.seen:
                self._step(model)
            if model.aloft:
                model.aloft = False
                self._exit(model.common)
        self._publish()

    def _fresh(self, model: Model) -> None:
        """Pages filled under another `prepare()` refill; the caches holding them shrink first."""
        version = model.fill.prepare() if model.fill is not None else None
        if version != model.filled:
            if model.filled is not None:
                for cache in [c for c in self.caches if model in c.holding]:
                    self._shrink(cache, cache.nbytes)
                self._evict(model, "refill", 0)
            model.filled = version

    def _plan(self, model: Model) -> None:
        """The watermark as a pass starts, from the measured reserve: a repeating model wholly at
        home if it fits beside its reserve and the cold weights worth less, else a spread set;
        one run once takes only free room. Unmeasured, a repeating model stays at home and one
        run once keeps only what is resident. Within a request it never rises."""
        repeating = model.repeats >= 2
        if not model.blocks or (model.reserve < 0 and repeating):
            return
        if model.reserve < 0:
            want = {b for b in model.home if _resident(model, b)}
        else:
            room = self.available() + _set(model)[0] - model.common.nbytes - model.reserve
            if repeating:
                value = self._value(model)
                cold = (m for m in self.models.values() if not m.flight and not m.held)
                room += sum(_set(m)[0] for m in cold if self._value(m) < value)
            sizes = [model.blocks[b].nbytes for b in model.order]
            fits = sum(sizes) <= room
            want = set(model.order if fits else _spread(model.order, sizes, room - 2 * max(sizes)))
        if self.active:
            want &= model.home
        self._set_home(model, want, "plan", 4)

    # -- blocks

    def _enter(
        self, block: Block, record: bool = True, call: tuple = ((), {}), wait: bool = True
    ) -> None:
        """Bind the block (refcounted). Short of room while another request holds it, wait for
        bytes to free (`wait`: the caller holds no lock), else `Exhausted`."""
        model = block.model
        if model.host:
            raise Unbound(f"{model.name}: {block.path or 'common'} binds to nothing")
        stream = torch.cuda.current_stream(model.device)
        while True:
            try:
                with self.lock, _disable_current_modes():
                    if block.count == 0:
                        self._bind(block, stream)
                    elif stream != block.stream and not torch.cuda.is_current_stream_capturing():
                        stream.wait_stream(block.stream)
                    block.count += 1
                    if record and block.index >= 0:
                        self._passed(block, call)
                break
            except _Short as short:
                need, op = short.args
                if not (wait and self._wait_room()):
                    raise self._exhausted(need, op) from None
        if record and block.index >= 0 and not model.held:
            self._prefetch(block)

    def _exit(self, block: Block) -> None:
        stream = torch.cuda.current_stream(block.model.device)
        with self.lock, _disable_current_modes():
            if block.stream not in (None, stream) and not torch.cuda.is_current_stream_capturing():
                block.stream.wait_stream(stream)
            block.count -= 1
            if block.count == 0:
                _unbind(block)

    def _bind(self, block: Block, stream: torch.cuda.Stream) -> None:
        model, block.stream = block.model, stream
        views: dict[str, dict[str, Tensor]] = {}
        fresh: dict[str, dict[str, Tensor]] = {}
        try:
            for region in block.regions:
                pinned = self._pin(model, region, stream.cuda_stream)
                base = torch.from_dlpack(pinned)
                block.pins.append(base)
                parts = block.parts(region, base)
                views.update(parts)
                if pinned.fresh:
                    fresh.update(parts)
            if fresh and model.fill is not None:
                model.fill(block, fresh)
            _write(block, views)
            if self.limit is not None and self._free() < 0:  # mapped past the limit
                self._make_room(0, f"limit {model.name}", model)
            self._cap()
        except BaseException:
            _unbind(block)
            raise

    def _pin(self, model: Model, region: int, stream: int) -> Pinned:
        """The region's pages, room made in §5.2's order until they map."""
        while True:
            try:
                return model.ws.bind(region, stream)
            except DeviceFull as full:
                why = f"bind {model.name}"
                if not self._make_room(self._free() + full.deficit, why, model):
                    raise _Short(full.deficit, why) from None

    def _passed(self, block: Block, call: tuple) -> None:
        """A block run again starts the model's next pass, at its first block's argument shapes.
        The watermark is planned once a request, or when the measured reserve moves."""
        model = block.model
        if block.index in model.seen:
            self._step(model)
        if not model.seen:
            model.base = torch.cuda.memory_allocated(model.device)
            torch.cuda.reset_peak_memory_stats(model.device)
            key = cuts.shape_of(*call)
            shape = model.track.shape = self.memo.shape(model.name, key)
            for request in self._requests():
                request.shapes[(model.name, key)] = shape
            moved = shape.peak >= 0 and shape.peak != model.reserve
            if moved:
                model.reserve = shape.peak
            if moved or model.planned != self.epoch:
                model.planned = self.epoch
                self._plan(model)
        model.seen[block.index] = None

    def _step(self, model: Model) -> None:
        if not model.steps:  # the first pass fixes the order prefetch follows
            model.order = [*model.seen, *(b for b in model.order if b not in model.seen)]
            model.position = {b: i for i, b in enumerate(model.order)}
        model.seen.clear()
        resident, paged, streamed, stall = _set(model)
        (last_paged, last_streamed, last_stall), model.mark = model.mark, (paged, streamed, stall)
        activations = max(torch.cuda.max_memory_allocated(model.device) - model.base, 0)
        model.peak = max(model.peak, activations)
        cuts.peaked(model.track, activations)
        home = model.home_bytes()
        moved = (paged - last_paged, streamed - last_streamed, stall - last_stall)
        step = Step(model.name, self.gpu, model.steps, home, resident, *moved, activations)
        model.steps += 1
        for request in self._requests():
            request.steps.append(step)
            passes = request.passes[model.name] = request.passes.get(model.name, 0) + 1
            model.repeats = max(model.repeats, passes)
            request.peak = max(request.peak, model.base + activations - request.start)
        self._running()

    def _prefetch(self, block: Block) -> None:
        """Copy the next blocks of the recorded order now, unless they are resident."""
        model = block.model
        at = model.position.get(block.index)
        if at is None:
            return
        for ahead in range(1, _AHEAD + 1):
            nxt = model.order[(at + ahead) % len(model.order)]
            if model.blocks[nxt].count or nxt in model.home and _resident(model, nxt):
                continue
            try:
                for region in model.blocks[nxt].regions:
                    model.ws.prefetch(region)
            except DeviceFull:
                return

    @contextmanager
    def _reading(self, blocks: Sequence[Block]) -> Iterator[None]:
        """A placeholder's blocks bound for one op outside their forward."""
        self._enter_all(blocks)
        try:
            yield
        finally:
            for block in blocks:
                self._exit(block)

    def _enter_all(self, blocks: Sequence[Block], wait: bool = True) -> None:
        """Every block bound, or none."""
        for i, block in enumerate(blocks):
            try:
                self._enter(block, record=False, wait=wait)
            except BaseException:
                for done in blocks[:i]:
                    self._exit(done)
                raise

    # -- room (§5.2)

    def _make_room(self, target: int, why: str, asking: Model | None) -> bool:
        """Rungs 1-2, then rung 4 (the catch cuts the op between them, rung 3), until `target`
        bytes are free. False when nothing could be freed."""
        freed = self._free_upto(target, why)
        return self._lower_until(target, why, asking) or freed

    def _free_upto(self, target: int, why: str) -> bool:
        """Idle bytes (rung 1); cold weights by the core's own order (`Gpu.give`: idle before
        warm, least recently bound first, never a busy set), then caches; then other processes
        (rung 2), until `target` bytes are free."""
        freed = self._idle()
        if self._free() < target:
            for name, n in self.core.give(target - self._free()):  # type: ignore[union-attr]
                freed += n
                if name in self.models:
                    self._evicted(self.models[name], 2, why, n)
        for cache in [c for c in list(self.caches) if c.nbytes]:
            if self._free() >= target:
                return True
            freed += self._shrink(cache, target - self._free())
            torch.cuda.empty_cache()
        if self._free() >= target:
            return True
        if self.row is not None:  # other rows, each evicting in its own process
            table, row, before = *self.row, self._free()
            for asked in table.ask(target - before, row.id):
                table.wait(asked)
            freed += max(self._free() - before, 0)
        return freed > 0 or self._free() >= target

    def _lower_until(self, target: int, why: str, asking: Model | None) -> bool:
        """Rung 4: the hot watermarks fall from the top of their spread sets, the asker's last."""
        freed = 0
        hot = [m for m in self.models.values() if m.flight and not m.held]
        for model in sorted(hot, key=lambda m: (m is asking, self._value(m))):
            while model.home and self._free() < target:
                ranked = _spread(model.order, _sizes(model), model.nbytes)
                last = next(b for b in reversed(ranked) if b in model.home)
                freed += self._set_home(model, model.home - {last}, why, 4)
        return freed > 0

    # the catch's rungs (`cuts.Rungs`)

    def free(self, device: torch.device, need: int) -> bool:
        if device.type != "cuda" or device.index != self.gpu:
            return False
        with self.lock:
            freed = self._free_upto(need, "out of memory")
            self._cap()
            return freed

    def lower(self, device: torch.device, need: int) -> bool:
        """At least one block lower; with nothing left to lower, a wait for another request's
        bytes (`_wait_room`)."""
        if device.type != "cuda" or device.index != self.gpu:
            return False
        with self.lock:
            lowered = self._lower_until(max(need, self._free() + 1), "out of memory", None)
            self._cap()
            if lowered:
                return True
        return self._wait_room()

    def _free(self) -> int:
        """The driver's free bytes, never beyond what a `gpu_memory_limit` leaves (§5.6)."""
        free, room = int(torch.cuda.mem_get_info(self.gpu)[0]), self._room()
        return free if room is None else min(free, room - torch.cuda.memory_reserved(self.gpu))

    def _room(self) -> int | None:
        """Under a limit, what this process's torch may reserve; None without one."""
        if self.limit is None or self.row is None:
            return None
        table, row = self.row
        others = sum(f.held for f in table.snapshot() if f.slot != row.id)
        core = self.core.stats()  # type: ignore[union-attr]
        return limited(self.limit, self.context, others, core.mapped + core.idle)

    def _cap(self) -> None:
        """Torch's allocator held to what the limit leaves it, at block boundaries."""
        room = self._room()
        if room is not None:
            torch.cuda.set_per_process_memory_fraction(min(room / self.total, 1.0), self.gpu)

    def available(self, device: torch.device | None = None) -> int:
        if not self.cuda or device is not None and device.index != self.gpu:
            return 0
        return (
            self._free()
            + torch.cuda.memory_reserved(self.gpu)
            - torch.cuda.memory_allocated(self.gpu)
        )

    def exhausted(self, device: torch.device, need: int, op: str) -> Exception:
        return self._exhausted(need, op)

    def _idle(self) -> int:
        """Rung 1: torch's slack and the core's idle chunks."""
        reserved = torch.cuda.memory_reserved(self.gpu)
        torch.cuda.empty_cache()
        return reserved - torch.cuda.memory_reserved(self.gpu) + int(self.core.trim())  # type: ignore[union-attr]

    def _value(self, model: Model) -> tuple[int, int]:
        """Its class (in use 2, warm 1, idle 0), then recency."""
        use = 2 if model.flight or model.held else 1 if model.level is Level.GPU else 0
        return (use, model.last_use)

    def _shrink(self, cache: Cache, nbytes: int) -> int:
        freed = min(int(cache.shrink(min(nbytes, cache.nbytes))), cache.nbytes)
        cache.nbytes -= freed
        return freed

    def _wait_room(self) -> bool:
        """§2's safety net, never under the lock: wait on the ledger for bytes to free while
        another request runs (here or in another row) and is not itself waiting. When every one
        waits, the lowest row gives up: its `Exhausted` frees its bytes and wakes the rest."""
        if self.row is None:
            return False
        with self.freeing:
            here = len(self.active) - 1 - self.waiting
            if here <= 0 and not self._elsewhere():
                return False
            gen = self.row[0].freed_gen()
            self.waiting += 1
        try:
            self._publish()
            if here <= 0 and not self._elsewhere():  # they began waiting meanwhile
                return False
            self.row[0].wait_freed(gen)
            return True
        finally:
            with self.freeing:
                self.waiting -= 1
            self._publish()

    def _elsewhere(self) -> bool:
        """Another row runs a request that is not waiting; or, all waiting, this row is not the
        lowest of them."""
        table, row = self.row  # type: ignore[misc]
        active = [
            f for f in table.snapshot() if f.state == "live" and f.running and f.slot != row.id
        ]
        if any(len(f.running) > f.waiting for f in active):
            return True
        return bool(active) and row.id != min(row.id, *(f.slot for f in active))

    def _freed(self) -> None:
        """Bytes went back: wake whoever waits for room, here and in TensorD."""
        if self.row is not None:
            self.row[0].freed()

    def _publish(self) -> None:
        """This row's facts (§5.6), as forwards start and land and at request ends. An ask the
        Rust thread answers from weight pages also gets torch's slack and the caches here."""
        if self.row is None:
            return
        self._cap()
        row, gpu = self.row[1], self.gpu
        if row.want() > 0:
            freed = self._idle() + sum(
                self._shrink(c, c.nbytes) for c in list(self.caches) if c.nbytes
            )
            if freed:
                self._freed()
        resident = {s.name: int(s.resident) for s in self.core.stats().sets}  # type: ignore[union-attr]
        models = list(self.models.values())
        bytes_of = lambda ms: sum(resident.get(m.name, 0) for m in ms)  # noqa: E731
        cold = [m for m in models if not m.flight and not m.held]
        reserved, allocated = torch.cuda.memory_reserved(gpu), torch.cuda.memory_allocated(gpu)
        row.publish(
            context=self.context,
            resident=bytes_of(models),
            cold=bytes_of(cold),
            kept=bytes_of(m for m in cold if m.level is Level.GPU),
            pinned=bytes_of(m for m in models if m.flight or m.held),
            torch_reserved=reserved,
            torch_allocated=allocated,
            cache=sum(c.nbytes for c in list(self.caches)),
            idle=reserved - allocated,
            active_peak=torch.cuda.max_memory_allocated(gpu) if self.active else 0,
            last_use=self.last_use,
            waiting=self.waiting,
        )

    def _running(self) -> None:
        """This row's running requests and what each may still need beyond what this process
        holds (§5.6's `fits`): its key's measured peak less what it has taken; -1 unmeasured."""
        if self.row is None:
            return
        table, row = self.row
        taken = torch.cuda.memory_allocated(self.gpu)
        peaks = {r: table.peak(r.key) if r.key is not None else None for r in list(self.active)}
        row.running(
            [(r.id, -1 if p is None else max(p - taken + r.start, 0)) for r, p in peaks.items()]
        )

    def _evict(self, model: Model, why: str, rung: int, need: int | None = None) -> int:
        """Pages nobody uses: all, or with `need` its lowest-ranked blocks until that many bytes
        are free, its common last."""
        freed = 0
        if need is not None:
            for b in reversed(_spread(model.order, _sizes(model), model.nbytes)):
                if freed >= need:
                    break
                if not model.blocks[b].count:
                    freed += int(model.ws.evict(model.blocks[b].regions))
        if need is None or freed < need:
            freed += int(model.ws.evict(None))
        if freed:
            self._evicted(model, rung, why, freed)
        return freed

    def _set_home(self, model: Model, want: set[int], why: str, rung: int) -> int:
        gone = model.home - want
        if want == model.home:
            return 0
        model.home = set(want)
        model.ws.home(_home_regions(model))
        if not gone:
            return 0
        freed = int(model.ws.evict([r for b in gone for r in model.blocks[b].regions]))
        if freed:
            self._evicted(model, rung, why, freed)
        return freed

    def _evicted(self, model: Model, rung: int, why: str, nbytes: int) -> None:
        self.evictions += 1
        for request in self._requests():
            request.evictions.append(Eviction(model.name, self.gpu, rung, why, nbytes))

    # -- caches and holds

    def cache(self, owner: str, shrink: Shrink) -> Cache:
        """Long-lived bytes outside a forward (LM KV, graph pools, the memo)."""
        cache = Cache(self, owner, shrink)
        with self.lock:
            self.caches[cache] = None
        return cache

    def _hold(self, model: Model, cache: Cache) -> Hold:
        with self.lock:
            if model.host or len(model.home) != len(model.blocks):
                raise Unbound(f"{model.name} streams part of its weights: it cannot be held")
            self._enter_all((model.common, *model.blocks), wait=False)
            model.held += 1
            cache.holding[model] = cache.holding.get(model, 0) + 1
        return Hold(model, cache)

    def _release(self, model: Model, cache: Cache) -> None:
        with self.lock:
            for block in (model.common, *model.blocks):
                self._exit(block)
            model.held -= 1
            cache.holding[model] -= 1
            if not cache.holding[model]:
                del cache.holding[model]

    def _close(self, model: Model) -> None:
        with self.lock:
            if model.closed:
                return
            model.closed = True
            self.models.pop(model.name, None)
            self.by_module.pop(model.module, None)
            for handle in model.handles:
                handle.remove()
            for block in (model.common, *model.blocks):
                _unhook(block)
                block.count = 0
                _unbind(block)
            model.ws.close()

    # -- requests (§5.7)

    @contextmanager
    def request(self, id: str) -> Iterator[Request]:
        """Everything the request allocates is freed on exit, failed or not; its ops are caught
        (`cuts.caught`). A capacity failure never poisons; what package state still holds shows
        as `leftover`."""
        start = torch.cuda.memory_allocated(self.gpu) if self.cuda else 0
        key = self.row[0].dispatched(id) if self.row is not None else None
        request = Request(id, start, self._copies(), key, self.memo.retries)
        with self.lock:
            self.active[request] = None
            self.epoch += 1
        self._running()
        stack: list[Request] = self.local.__dict__.setdefault("requests", [])
        stack.append(request)
        try:
            with cuts.caught() if self.cuda else nullcontext():
                yield request
        except BaseException as exc:
            _clear_frames(exc)
            raise
        finally:
            self._end(request, stack)

    def _end(self, request: Request, stack: list[Request]) -> None:
        with self.lock:  # passes no forward landed (blocks run directly) end with it
            for model in list(self.models.values()):
                if model.seen and not model.flight:
                    self._step(model)
        stack.remove(request)
        gc.collect()
        leftover: tuple[tuple[int, int], ...] = ()
        if self.cuda:
            torch.cuda.synchronize(self.gpu)
            torch.cuda.empty_cache()
            tail = torch.cuda.max_memory_allocated(self.gpu) - request.start
            request.peak = max(request.peak, tail)
            leftover = ((self.gpu, max(torch.cuda.memory_allocated(self.gpu) - request.start, 0)),)
        copied = sum(n - request.copies.get(name, 0) for name, n in self._copies().items())
        retries = self.memo.retries - request.retried
        peaks = tuple(_peak(s) for s in request.shapes.values())
        steps, evictions = tuple(request.steps), tuple(request.evictions)
        request.stats = RequestStats(request.id, steps, evictions, leftover, copied, retries, peaks)
        if request.key is not None and self.row is not None:
            self.row[0].set_peak(request.key, request.peak)
        with self.lock:
            self.active.pop(request, None)
            self.epoch += 1
            for name, passes in request.passes.items():
                if (ran := self.models.get(name)) is not None:
                    ran.repeats = passes
            if self.cuda and not self.active:
                self._restore()
        self._running()
        self._publish()
        self._freed()

    def _restore(self) -> None:
        """Between requests, or at a load: each model's reserve is what that request measured,
        and the watermarks rise. Warm models refill most worth keeping first, into free room and
        then cold pages of this process's models worth less, always leaving the largest reserve
        and a streamed model's slots. The copies are prefetches: an ask evicts them at once."""
        models = list(self.models.values())
        for model in models:
            if model.peak:
                model.reserve, model.peak = model.peak, 0
        slots = max(
            (_SLOTS * max(_sizes(m), default=0) for m in models if m.repeats < 2), default=0
        )
        room = self._free() - max((max(m.reserve, 0) for m in models), default=0) - slots
        ranked = sorted(models, key=self._value, reverse=True)
        for i, model in enumerate(ranked):
            if model.level is not Level.GPU:
                return
            lower = [m for m in ranked[i + 1 :] if not m.flight and not m.held]
            for b in model.order:
                block = model.blocks[b]
                if b in model.home and _resident(model, b):
                    continue
                while room < block.nbytes and lower:
                    room += self._evict(lower[-1], "restore", 2, block.nbytes - room)
                    if room < block.nbytes:  # it gave all it had
                        lower.pop()
                if room < block.nbytes:
                    return
                room -= block.nbytes
                model.home.add(b)
                model.ws.home(_home_regions(model))
                try:
                    for region in block.regions:
                        model.ws.prefetch(region)
                except DeviceFull:
                    return

    def _copies(self) -> dict[str, int]:
        """Each model's weight bytes copied since it loaded, home or through slots."""
        return {m.name: sum(_set(m)[1:3]) for m in list(self.models.values())}

    def _requests(self) -> list[Request]:
        """The request a step or an eviction belongs to: this thread's, else the first."""
        mine = self.local.__dict__.get("requests", [])
        return mine[-1:] or list(self.active)[:1]

    # -- stats (§5.8)

    def resident(self) -> dict[str, int]:
        """Each model's resident bytes."""
        with self.lock:
            return {m.name: _set(m)[0] for m in self.models.values()}

    def _exhausted(self, need: int, op: str) -> Exhausted:
        peaks = tuple(_peak(s) for s in list(self.memo.shapes.values()))
        return Exhausted(self.gpu, need, op, self.resident(), peaks)


# ----------------------------------------------------------------------------- binding


def _declared(module: nn.Module) -> Declared:
    found = vars(module).get("_varena")
    return found if isinstance(found, Declared) else Declared()


def blocks_of(module: nn.Module) -> list[str]:
    """Its declared blocks, else the innermost instances of its `_no_split_modules`."""
    declared = _declared(module).blocks
    if declared is not None:
        return list(declared)
    names = set(getattr(module, "_no_split_modules", None) or ())
    found = [
        path
        for path, sub in module.named_modules()
        if path and any(kind.__name__ in names for kind in type(sub).__mro__)
    ]
    return [a for a in found if not any(b.startswith(a + ".") for b in found)]


def regions_of(module: nn.Module) -> list[list[str]]:
    """Its weight set's regions, cut at its blocks: what `load` maps and a sealed layout holds."""
    grant = _declared(module).weights
    if grant is None:
        raise Unbound(f"{type(module).__name__}: no weights attached (`Declared.weights`)")
    prefix, keys = f"{grant.component}/", None if grant.keys is None else set(grant.keys)
    cuts_at = [prefix + b for b in blocks_of(module)]
    regions = [
        [n for n in region if n.startswith(prefix) and (keys is None or n[len(prefix) :] in keys)]
        for region in tensorfs.regions(grant.ck, _WHOLE, cuts_at)
    ]
    return [r for r in regions if r]


def _layout(
    model: Model, prefix: str, paths: Sequence[str], regions: Sequence[Sequence[str]]
) -> None:
    """Blocks (in header order) and common, from the weight set's regions and parts."""
    by_path = {p: Block(model, -1, p, model.module.get_submodule(p)) for p in paths}
    owner: list[Block] = []
    for index, names in enumerate(regions):
        homes = {_block_of(n[len(prefix) :], paths) for n in names}
        path = homes.pop() if len(homes) == 1 else None
        block = by_path[path] if path is not None else model.common
        block.regions.append(index)
        owner.append(block)
    for part in model.ws.parts:
        shape = tuple(part.shape)
        if part.dtype == "F4":
            shape = (*shape[:-1], shape[-1] // 2)
        at = _PartAt(part.region, part.offset, part.nbytes, DTYPES[part.dtype], shape)
        owner[part.region].tensors.setdefault(part.key, {})[part.role or "value"] = at
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
                if tensor is not None:
                    names[f"{path}.{attr}" if path else attr] = id(tensor)
                    sites = by_tensor.setdefault(id(tensor), [])
                    if not any(s.table is table and s.attr == attr for s in sites):
                        sites.append(_Site(table, attr, parameter))
    return {name: by_tensor[t] for name, t in names.items()}


def _install(model: Model) -> None:
    """Every meta weight becomes a placeholder on the execution device, owned by its block."""
    by_path = {b.path: b for b in model.blocks}
    done: set[int] = set()
    for name, group in model.sites.items():
        meta = group[0].table[group[0].attr]
        if meta is None or id(meta) in done or meta.device.type != "meta":
            continue
        done.add(id(meta))
        path = _block_of(name, list(by_path))
        block = model.common if path is None else by_path[path]
        held: Tensor = Placeholder(meta.shape, meta.dtype, model.device, block, group[0])
        if group[0].parameter:
            held = nn.Parameter(held, requires_grad=False)
        for site in group:
            site.table[site.attr] = held


def _fly(model: Model) -> None:
    """A forward of the root or a direct child (`vae.decode` calls `decoder`) is the model in
    use, counted so concurrent forwards share it (cr-202)."""
    memory = model.memory
    for module in (model.module, *model.module.children()):
        model.handles.append(module.register_forward_pre_hook(lambda *_: memory._start(model)))
        hook = module.register_forward_hook(lambda *_: memory._land(model), always_call=True)
        model.handles.append(hook)


def _hook(block: Block) -> None:
    module, memory, site = block.module, block.model.memory, block.site
    assert module is not None and site is not None
    block.own_forward = "forward" in module.__dict__
    forward = block.forward = module.forward

    def bound(*args: object, **kwargs: object) -> object:
        memory._enter(block, call=(args, kwargs))
        try:
            return cuts.run(site, forward, args, kwargs)
        finally:
            memory._exit(block)

    module.forward = bound  # type: ignore[method-assign]


def _unhook(block: Block) -> None:
    if block.module is None or block.forward is None:
        return
    if block.own_forward:
        block.module.forward = block.forward  # type: ignore[method-assign]
    else:
        del block.module.forward
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
                site.table[site.attr] = nn.Parameter(value, False) if site.parameter else value


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
            parts = block.parts(region, torch.from_dlpack(model.ws.bind(region, 0)))
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


def _resident(model: Model, b: int) -> bool:
    return all(model.ws.resident(r) for r in model.blocks[b].regions)


def _sizes(model: Model) -> list[int]:
    return [model.blocks[b].nbytes for b in model.order]


def _peak(shape: cuts.Shape) -> Peak:
    cut = tuple(f"{name} {size}" for name, size in shape.cuts.items())
    return Peak(shape.model, shape.shape, shape.peak, cut)


def limited(capacity: int, context: int, others: int, varena: int) -> int:
    """Under a limit, what this process's torch may reserve: the capacity less its CUDA context,
    the bytes other rows hold and its own varena pages (§5.6)."""
    return max(capacity - context - others - varena, 0)


def _set(model: Model) -> tuple[int, int, int, int]:
    """Its resident bytes, and its cumulative bytes paged in, streamed and prefetch stall."""
    if model.host:
        return model.nbytes, 0, 0, 0
    for s in model.memory.core.stats().sets:  # type: ignore[union-attr]
        if s.name == model.name:
            total = s.total
            return s.resident, total.paged_in, total.streamed_bytes, total.stall_ns
    return 0, 0, 0, 0


def _spread(order: Sequence[int], sizes: Sequence[int], room: int) -> list[int]:
    """Blocks to keep at home within `room` bytes, ranked so that every prefix is spread evenly
    along `order` (bit-reversed positions): each streamed copy overlaps a resident compute."""
    n = len(order)
    if not n:
        return []
    bits = max(n - 1, 1).bit_length()
    ranked = dict.fromkeys(int(f"{i:0{bits}b}"[::-1], 2) * n >> bits for i in range(1 << bits))
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
