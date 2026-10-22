"""varena's memory for torch modules: conversion, with no policy (design varena-slim.md §2).

    mem = Memory.open(0)  # before torch initializes CUDA
    vars(unet)["_varena"] = Declared(Weights(ck, "unet"))
    mem.load(unet, "gpu")
    with mem.request("r1", plan=[te, unet, vae]):
        image = pipe(prompt)

A block's params are views of its pin only while the block runs; placeholders (shape, dtype and
device, no bytes) stand in otherwise, so no tensor outlives its page. A model's top-level forward
is one pass: its fill identity is set, varena makes its measured room, torch's allocator is
capped to `torch_limit`, and the pass reports torch's allocated peak, the allocator's slack and
each cut site's peak, with torch's bytes at its start and end and at every `oom`. Peaks reset
device-wide, so every reset first folds them into every open pass. varena calls torch's
`empty_cache` when its decision trims. cozy-runtime's catch calls `r.oom` and `r.cut` and
measures sites with `r.site`.
"""

from __future__ import annotations

import os
import sys
import threading
import warnings
import weakref
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

import tensorfs
import torch
from torch import Tensor, nn
from torch.utils._pytree import tree_leaves

from . import Exhausted, Fill, GpuId, Grant, Level, Next, Pass, Pin, Stats, VarenaError, Window, Yielded
from . import Memory as Native
from .._ext.varena import allocated as _allocated
from .._ext.varena import allocator_stats as _allocator_stats
from .. import _ext
from . import Request as NativeRequest
from . import Set

__all__ = [
    "Bind", "Cache", "Declared", "Exhausted", "Filling", "GpuId", "Grant", "Hold", "Level", "Memory", "Model",
    "Next", "Placeholder", "Request", "Stats", "Unbound", "VarenaError", "Weights", "blocks_of",
]

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


class Unbound(Exception):
    """An op on a weight no pin is bound to, or a forward outside a request."""


@dataclass(frozen=True, slots=True)
class Weights:
    """A module's stored weights: a component of a TensorFS checkpoint. `release` names the
    package release (the fill code's identity), `shard` the rank's slice, `keys` the stored
    tensors the module builds (all by default)."""

    ck: tensorfs.Checkpoint
    component: str
    release: str = ""
    shard: str = "0/1"
    keys: Collection[str] | None = None


class Filling(Protocol):
    """A transform of a fresh block's bytes (a LoRA bake), writing in place. `identity` covers
    what it writes (LoRA digests, scales, recipe, torch/GPU arch); None: never kept."""

    @property
    def identity(self) -> str | None: ...

    def __call__(self, block: str, parts: Mapping[str, Mapping[str, Tensor]], /) -> None: ...


#: `bind(key, parts)`: a stored tensor's part views -> the module tensors they become, by name.
Bind: TypeAlias = "Callable[[str, Mapping[str, Tensor]], Mapping[str, Tensor]]"


@dataclass(frozen=True, slots=True)
class Declared:
    """What a module carries as `_varena`: its weights, its blocks (default: the innermost
    `_no_split_modules`), its fill and its bind (default: a plain part as itself)."""

    weights: Weights
    blocks: Sequence[str] | None = None
    fill: Filling | None = None
    bind: Bind | None = None


class Placeholder(Tensor):
    """A weight while no pin is bound to it: shape, dtype and device, no bytes. Only a dtype or
    device change (another placeholder) and detach work on it."""

    @staticmethod
    def __new__(cls, shape, dtype, device):  # type: ignore[no-untyped-def]
        return Tensor._make_wrapper_subclass(cls, tuple(shape), dtype=dtype, device=device, storage_size=0)  # type: ignore[attr-defined]

    def __repr__(self) -> str:  # type: ignore[override]
        return f"Placeholder({tuple(self.shape)}, {self.dtype}, {self.device})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        aten, kwargs = torch.ops.aten, kwargs or {}
        if func in (aten.detach.default, aten.alias.default):
            return args[0]
        if func is aten._to_copy.default:
            held = args[0]
            return cls(held.shape, kwargs.get("dtype") or held.dtype, kwargs.get("device") or held.device)
        raise Unbound(f"{func}: a weight with no pin bound (used outside its block's forward)")


def blocks_of(module: nn.Module) -> list[str]:
    """Its declared blocks, else the innermost instances of its `_no_split_modules`."""
    declared = vars(module).get("_varena")
    if isinstance(declared, Declared) and declared.blocks is not None:
        return list(declared.blocks)
    names = set(getattr(module, "_no_split_modules", None) or ())
    found = [
        path
        for path, sub in module.named_modules()
        if path and any(kind.__name__ in names for kind in type(sub).__mro__)
    ]
    return [a for a in found if not any(b.startswith(a + ".") for b in found)]


# ------------------------------------------------------------------------------ the model


@dataclass(frozen=True, slots=True)
class _Site:
    table: dict[str, Tensor | None]
    attr: str
    parameter: bool


@dataclass(frozen=True, slots=True)
class _At:
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass(eq=False, slots=True)
class _Block:
    """One region of the set: a block's tensors, or common's (`path` "")."""

    model: Model
    path: str
    region: int
    tensors: dict[str, dict[str, _At]] = field(default_factory=dict)
    count: int = 0
    stream: torch.cuda.Stream | None = None
    pin: Pin | None = None
    written: list[tuple[_Site, Tensor | None]] = field(default_factory=list)
    #: its mapping's address, the part views of it and the module tensors made of them: kept
    #: across binds of that mapping (they hold no pin and are installed only under one)
    at: int | None = None
    views: dict[str, dict[str, Tensor]] | None = None
    assign: list[tuple[_Site, Tensor]] | None = None
    #: the window the kept views are of, and what holds them when nothing escaped: the tensors,
    #: each one's reference count and their storage's use count
    window: Window | None = None
    track: tuple[list[Tensor], list[int], int] | None = None


@dataclass(eq=False, slots=True)
class _Pass:
    """An open pass and what it measured so far; `open` holds each running site's peak."""

    native: Pass
    request: Request
    #: its model's set
    set: int
    #: the module its forward entered: its path in the model ("" the root)
    entry: str
    #: its tensor arguments' dims, joined by `x`
    shape: str
    #: torch's allocated bytes, and its allocator's churn counters, at its start
    start: int = 0
    churned: dict[str, int] = field(default_factory=dict)
    peak: int = 0
    reserved_peak: int = 0
    #: each cut site's allocated peak, whether it is a collective's, and its finest level
    sites: dict[str, tuple[int, bool, int]] = field(default_factory=dict)
    #: cut levels read in this pass per (site, shape), until an oom answer or an agreement
    cuts: dict[tuple[str, str], int] = field(default_factory=dict)
    open: list[list[object]] = field(default_factory=list)
    #: the exception being handled where it opened: another at its close is a raise
    entered: BaseException | None = None
    token: object = None
    #: SPIKE: the pool's context while the pass allocates from it
    pooled: object = None


class Model:
    """A loaded module: its set, its blocks and hooks."""

    def __init__(self, memory: Memory, module: nn.Module, declared: Declared, name: str, set: Set, id: int) -> None:
        self.memory, self.module, self.name, self.set, self.id = memory, module, name, set, id
        self.component = declared.weights.component
        self.fill, self.bind = declared.fill, declared.bind
        self.lock = threading.RLock()
        self.blocks: dict[str, _Block] = {}
        self.common: _Block | None = None
        self.sites = _sites(module)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        #: per thread: its forwards' depth, and the pass the outermost opened
        self.local = threading.local()
        self.holds: list[Hold] = []
        #: its last measured pass: `(entry|shape, allocated peak, slack, allocator churn)`
        self.last: tuple[str, int, int, dict[str, int]] | None = None
        #: (entry, shape) -> (allocated growth, slack) of its last completed pass there
        self.rooms: dict[tuple[str, str], tuple[int, int]] = {}
        #: the fill identity its pages were last declared under
        self.identity: str | None = declared.fill.identity if declared.fill is not None else None

    def retain(self, paths: Collection[str] = ("",)) -> Binding:
        """Retain addresses for every weight/buffer in the declared module subtrees.

        An empty path declares the whole model. Runtime declares indirect dependencies too;
        observing the modules visited by one capture is not a complete declaration.
        """
        all_blocks = [b for b in (self.common, *self.blocks.values()) if b is not None]
        blocks = [b for b in all_blocks if any(not p or b.path == p or b.path.startswith(p + ".") for p in paths)]
        # Common weights can be read by any child block, so include them conservatively.
        if self.common is not None and self.common not in blocks:
            blocks.insert(0, self.common)
        return Binding(self, sorted(blocks, key=lambda b: b.region))

    def hold(self, shed: Callable[[], None] | None = None) -> Hold:
        """Every block bound, pins and param views, until the hold's release (a CUDA graph's
        fixed addresses). Register the graph as a `mem.cache` whose shrink drops it and releases
        this. A pass that changes the fill identity sheds the hold first and calls `shed` (it
        drops the graph, whose addresses hold the old bake)."""
        blocks = [b for b in (self.common, *self.blocks.values()) if b is not None]
        held: list[_Block] = []
        try:
            for b in blocks:
                self.memory._enter(b)
                held.append(b)
        except BaseException:
            for b in reversed(held):
                self.memory._exit(b)
            raise
        hold = Hold(self, held, shed)
        self.holds.append(hold)
        return hold

    def close(self) -> None:
        """Unhooks the module and unloads its set."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        for b in (self.common, *self.blocks.values()):
            if b is not None:
                b.views = b.assign = b.window = b.track = None
        self.memory.models.pop(self.module, None)
        self.memory.native.unload(self.id)


class Binding:
    """A graph's virtual addresses, leased physically only around capture or replay."""

    def __init__(self, model: Model, blocks: list[_Block]) -> None:
        self.model, self.blocks = model, blocks
        self.native = model.memory.native.retain(model.id, [b.region for b in blocks])
        self.identity = self.native.identity
        self.closed = False
        # Covers input copies, launch, and output uses: each graph has one static buffer set.
        self.lock = threading.RLock()

    @contextmanager
    def prepare(self, stream: torch.cuda.Stream | int | None = None) -> Iterator[Binding]:
        with self.lock:
            if self.closed:
                raise Unbound("graph binding is closed")
            memory = self.model.memory
            if memory.cuda:
                if torch.cuda.is_current_stream_capturing():
                    raise Unbound("graph residency must be prepared before capture")
                if stream is None:
                    stream = torch.cuda.current_stream(memory.device)
                elif isinstance(stream, int):
                    stream = torch.cuda.ExternalStream(stream, device=memory.device)
            with torch.cuda.stream(stream) if memory.cuda else nullcontext():
                memory.native.fill(self.model.id, _fill(self.model.fill))
                pins = self.native.prepare(stream.cuda_stream if stream is not None else 0)
                held: list[_Block] = []
                try:
                    with self.model.lock:
                        for block, pin in zip(self.blocks, pins, strict=True):
                            if block.count == 0:
                                memory._bind(block, stream, pin)
                            elif stream is not None and block.stream != stream:
                                stream.wait_stream(block.stream)
                            block.count += 1
                            held.append(block)
                    yield self
                finally:
                    # `_exit` joins a caller's current consumer stream back to each bind stream.
                    # More streams must be joined by the caller before leaving this scope.
                    for block in reversed(held):
                        memory._exit(block)
                    pins.clear()  # Native pin drops record completion after the queued uses.

    def close(self) -> None:
        with self.lock:
            if not self.closed:
                self.closed = True
                self.native.close()

    def __del__(self) -> None:
        if hasattr(self, "lock"):
            self.close()


class Hold:
    """A model's blocks held bound (`Model.hold`); `release`, or leaving its `with`, lets them go."""

    def __init__(self, model: Model, blocks: list[_Block], shed: Callable[[], None] | None) -> None:
        self.model, self.blocks, self.on_shed = model, blocks, shed

    def release(self) -> None:
        blocks, self.blocks = self.blocks, []
        if self in self.model.holds:
            self.model.holds.remove(self)
        for b in reversed(blocks):
            self.model.memory._exit(b)

    def shed(self) -> None:
        """Released for a fill-identity change; its owner drops what it built on the old bake."""
        self.release()
        if self.on_shed is not None:
            self.on_shed()

    def __enter__(self) -> Hold:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


# -------------------------------------------------------------------------- the memory

#: This process's Memory of each device, while something holds it.
_memories: weakref.WeakValueDictionary[str, Memory] = weakref.WeakValueDictionary()
_opening = threading.Lock()

#: The request a forward runs in, and the pass a site op runs in.
_request: ContextVar[Request | None] = ContextVar("varena_request", default=None)
_pass: ContextVar[_Pass | None] = ContextVar("varena_pass", default=None)
_pool: ContextVar[tuple[Memory, torch.cuda.MemPool] | None] = ContextVar("varena_pool", default=None)


class Memory:
    """This process's memory on one device: its models, caches and requests."""

    def __init__(self, native: Native, device: torch.device, pool: bool = True) -> None:
        self.native, self.device = native, device
        self.cuda = device.type == "cuda"
        self.models: dict[nn.Module, Model] = {}
        self.requests: dict[Request, None] = {}
        self.torch = _Allocator(device)
        #: every open pass on the device, of every thread: a reset folds into each first
        self.passes: list[_Pass] = []
        self.measuring = threading.Lock()
        #: (set, block) -> bytes kept bound for tensors that outlived their block's run
        self.escaped: dict[tuple[str, str], int] = {}
        #: the bytes torch's allocator is capped at now
        self.capped: int | None = None
        if device.type != "cpu":
            native.trim_with(self.torch.trim)
        #: SPIKE (pending the owner): passes allocate from a pool whose segments come from this
        #: library's allocator, which asks varena for room on a shortfall instead of raising
        self.pool: torch.cuda.MemPool | None = None
        #: whether passes use it now (the spike's measurement alternates with torch's default)
        self.pooling = pool
        #: the largest torch reserve a pass held outside the pool (other threads' allocations)
        self.outside = 0
        #: SPIKE DIAG: block runs on a stream other than the one its pin was bound on
        self.crossed = 0
        if pool and self.cuda:
            allocator = torch.cuda.memory.CUDAPluggableAllocator(_ext.__file__, "varena_torch_alloc", "varena_torch_free")
            self.pool = torch.cuda.MemPool(allocator._allocator)

    @classmethod
    def open(cls, gpu: int | str | torch.device | GpuId = 0, grant: Grant | None = None, *, limit: int | None = None, pool: bool = True) -> Memory:
        """A TensorD executor's memory (`grant`), or a standalone engine's: the device's table by
        name, its capacity lowered to `limit` (the first opener sets it, a later one can only
        lower it). On CUDA the backend opens first and measures this process's context, so open
        it before torch initializes CUDA, or torch's context counts as foreign bytes. CPU and MPS
        use host pages on the machine's table `host` (its RAM); a `GpuId` names a host table.
        A process has one Memory per device: while it lives, opening the device again returns it."""
        device = torch.device("cpu") if isinstance(gpu, GpuId) else torch.device("cuda", gpu) if isinstance(gpu, int) else torch.device(gpu)
        if device.type == "cuda":
            device = torch.device("cuda", device.index or 0)
        key = gpu.uuid if isinstance(gpu, GpuId) else str(device)
        with _opening:
            memory = _memories.get(key)
            if memory is None:  # one Memory per device and process: a second open is the first
                if isinstance(gpu, GpuId):
                    native = Native.host(gpu, grant, limit)
                elif device.type == "cuda":
                    native = Native.cuda(device.index or 0, grant, limit)
                else:
                    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                    native = Native.host(GpuId("host", device.type, ram), grant, limit)
                memory = _memories[key] = cls(native, device, pool)
        return memory

    @property
    def gpu(self) -> GpuId:
        return self.native.gpu

    # -- loading

    def load(self, module: nn.Module, level: Level | str = Level.GPU) -> Model:
        """`module` (built on meta) from its `_varena` weights at `level`: one region per block
        and one for common, and hooks that bind blocks while they run. A loaded module moves to
        `level` in place."""
        if isinstance(level, str):
            level = Level.GPU if level == "gpu" else Level.RAM
        if module in self.models:
            model = self.models[module]
            self.native.load(model.set, level)
            return model
        declared = vars(module).get("_varena")
        if not isinstance(declared, Declared):
            raise Unbound(f"{type(module).__name__}: no weights declared (`_varena = Declared(...)`)")
        w, paths = declared.weights, blocks_of(module)
        owned: dict[str, list[str]] = {}
        for key in w.ck.components[w.component].tensors:
            if w.keys is None or key in w.keys:
                owned.setdefault(_block_of(key, paths), []).append(f"{w.component}/{key}")
        order = [p for p in ("", *paths) if p in owned]
        name = f"{w.release or w.ck.id}/{w.component}"
        set = Set(name, w.ck, [owned[p] for p in order], fill=_fill(declared.fill), release=w.release, shard=w.shard, gpu=self.gpu.uuid)
        model = Model(self, module, declared, name, set, self.native.load(set, level))
        for region, path in enumerate(order):
            block = _Block(model, path, region)
            if path:
                model.blocks[path] = block
            else:
                model.common = block
        regions = [model.common, *model.blocks.values()] if model.common else list(model.blocks.values())
        for p in set.parts:
            shape = (*p.shape[:-1], p.shape[-1] // 2) if p.dtype == "F4" else tuple(p.shape)
            regions[p.region].tensors.setdefault(p.key, {})[p.role] = _At(p.offset, p.nbytes, DTYPES[p.dtype], shape)
        _placeholders(model)
        _hook(model)
        self.models[module] = model
        return model

    # -- requests

    @contextmanager
    def request(self, id: str, plan: Sequence[object] = ()) -> Iterator[Request]:
        """Request `id` and its plan: its steps in order, each a model (`Model`, module or set
        name) or a collection of models that run together. Leaving ends it."""
        r = Request(self, self.native.request(id, self._plan(plan)))
        self.requests[r] = None
        token = _request.set(r)
        try:
            with self.allocating():
                yield r
        except BaseException as failed:
            try:
                r.ended = r.native.end()
            except BaseException as ending:
                failed.add_note(f"ending request {id!r} also failed: {ending!r}")  # the request's own error stays the one raised
            raise
        else:
            r.ended = r.native.end()
        finally:
            _request.reset(token)
            self.requests.pop(r, None)

    def new_pool(self) -> torch.cuda.MemPool:
        """An independently owned graph scratch pool using the native allocation path."""
        allocator = torch.cuda.memory.CUDAPluggableAllocator(_ext.__file__, "varena_torch_alloc", "varena_torch_free")
        return torch.cuda.MemPool(allocator._allocator)

    @contextmanager
    def allocating(self, pool: torch.cuda.MemPool | None = None) -> Iterator[None]:
        """Use the request's allocator between passes too, or a graph's dedicated scratch pool.

        Warm and capture must name the same graph pool. Capture cannot grow it; a capacity
        shortfall aborts capture, without native pressure recovery or arbitrary package replay.
        """
        selected = self.pool if pool is None else pool
        r, p = _request.get(), _pass.get()
        if selected is None or r is None:
            yield
            return
        if self.cuda and torch.cuda.is_current_stream_capturing():
            raise Unbound("select and preseed the graph pool before capture")
        r.native.allocating(self.native, p.set if p else None, *self.torch.now(), selected.id)
        token = _pool.set((self, selected))
        try:
            with torch.cuda.use_mem_pool(selected, self.device):
                yield
        finally:
            _pool.reset(token)
            _allocated()

    def cache(self, owner: str, shrink: Callable[[int], int]) -> Cache:
        """Bytes the caller holds that `shrink(want) -> freed` gives back on varena's ask; a
        cache owned by a request id ends with it."""
        return Cache(self, self.native.cache(owner, shrink))

    def stats(self) -> Stats | None:
        """The record of the request that ended last."""
        return self.native.stats()

    def _plan(self, steps: Sequence[object]) -> list[tuple[list[str], int]]:
        out: list[tuple[list[str], int]] = []
        for step in steps:
            many = isinstance(step, Collection) and not isinstance(step, (str, nn.Module))
            names = sorted({self._name(s) for s in (step if many else [step])})  # type: ignore[union-attr]
            if out and out[-1][0] == names:
                out[-1] = (names, out[-1][1] + 1)
            else:
                out.append((names, 1))
        return out

    def _name(self, step: object) -> str:
        if isinstance(step, Model):
            return step.name
        if isinstance(step, nn.Module):
            return self.models[step].name
        for model in self.models.values():
            if step == model.component:
                return model.name
        return str(step)

    def _current(self) -> Request:
        r = _request.get()
        if r is None and len(self.requests) == 1:
            r = next(iter(self.requests))
        if r is None:
            raise Unbound("a loaded model's forward runs inside `mem.request(...)`")
        return r

    # -- the torch allocator

    def _cap(self, room: int = 0, start: bool = False) -> None:
        """torch's allocator cap. At a pass's start, the larger of `torch_limit` and the pass's
        measured `room`; otherwise, or beside another running pass, it only rises: a pass that
        got its room never has its allocator churn against a smaller cap. With the pool, its
        segments ask varena themselves: no cap."""
        if self.pool is not None and self.pooling:
            if self.capped != self.torch.total:
                self.capped = self.torch.total
                self.torch.cap(self.capped)
            return
        want = max(self.native.torch_limit(), room)
        if not start or self.passes:
            want = max(want, self.capped or 0)
        if want != self.capped:
            self.capped = want
            self.torch.cap(want)

    def _fold(self, discard: _Pass | None = None) -> None:
        """The allocator's peaks since the last reset into every open pass and its open sites,
        but `discard`'s (a failed attempt's); then the reset. The only place peaks reset."""
        with self.measuring:
            peak, reserved = self.torch.peaks()
            for p in self.passes:
                if p is not discard:
                    p.peak, p.reserved_peak = max(p.peak, peak), max(p.reserved_peak, reserved)
                    for entry in p.open:
                        entry[1] = max(entry[1], peak)  # type: ignore[call-overload]
            self.torch.reset()

    # -- passes

    def _open(self, model: Model, entry: str, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        """The outermost forward of `model` (its root or a direct child, `entry`) is a pass of its
        entry at its tensor arguments' dims: each entry point has its own measured room."""
        local = model.local
        local.depth = getattr(local, "depth", 0) + 1
        if local.depth > 1:
            return
        if self.cuda and torch.cuda.is_current_stream_capturing():
            local.depth -= 1
            raise Unbound("open the model pass and prepare graph dependencies before capture")
        p: _Pass | None = None
        try:
            r = self._current()
            shape = "x".join(str(d) for t in tree_leaves((args, kwargs)) if isinstance(t, Tensor) for d in t.shape)
            identity = model.fill.identity if model.fill is not None else None
            if identity != model.identity:
                for hold in list(model.holds):  # pins of the old bake would keep its pages
                    hold.shed()
                model.identity = identity
            self.native.fill(model.id, _fill(model.fill))
            allocated, reserved = self.torch.now()
            p = _Pass(r.native.pass_(model.id, entry, shape, allocated, reserved), r, model.id, entry, shape, allocated, self.torch.churn())
            growth, slack = model.rooms.get((entry, shape), (0, 0))
            self._cap(allocated + growth + slack if growth or slack else 0, start=True)
            if self.pool is not None and self.pooling and not (self.cuda and torch.cuda.is_current_stream_capturing()):  # capture owns its exact pool
                selected = _pool.get()
                pool = selected[1] if selected is not None and selected[0] is self else self.pool
                r.native.allocating(self.native, model.id, allocated, reserved, pool.id)
                p.pooled = torch.cuda.use_mem_pool(pool, self.device)
                p.pooled.__enter__()
            p.entered = sys.exc_info()[1]
            self._fold()  # the passes open already keep what ran before; this one starts here
            with self.measuring:
                self.passes.append(p)
            if model.common is not None:
                self._enter(model.common)
            local.passing, p.token = p, _pass.set(p)
        except BaseException:
            local.depth -= 1
            with self.measuring:
                if p in self.passes:
                    self.passes.remove(p)
            if p is not None and p.pooled is not None:
                try:
                    p.pooled.__exit__(None, None, None)
                finally:
                    _allocated()
            raise

    def _close(self, model: Model) -> None:
        local = model.local
        local.depth -= 1
        p = getattr(local, "passing", None)
        if local.depth or p is None:
            return
        local.passing = None
        _pass.reset(p.token)  # type: ignore[arg-type]
        if model.common is not None:
            self._exit(model.common)
        if p.pooled is not None:
            try:
                p.pooled.__exit__(None, None, None)
            finally:
                _allocated()
            self.outside = max(self.outside, self.torch.now()[1] - _allocator_stats(self.device.index or 0)[0])
        self._fold()
        with self.measuring:
            self.passes.remove(p)
        if sys.exc_info()[1] is not p.entered:
            return  # torch runs this hook in its handler of the forward's raise: unended, it records nothing
        sites = [(name, *site) for name, site in p.sites.items()]
        slack, now = max(p.reserved_peak - p.peak, 0), self.torch.churn()
        churn = {k: now[k] - p.churned.get(k, 0) for k in now}
        for k, n in churn.items():
            p.request.churn[k] = p.request.churn.get(k, 0) + n
        model.rooms[(p.entry, p.shape)] = (max(p.peak - p.start, 0), slack)
        model.last = (f"{p.entry}|{p.shape}", p.peak, slack, churn)
        p.native.end(p.peak, slack, sites, *self.torch.now())

    # -- blocks

    def _enter(self, block: _Block) -> None:
        stream = torch.cuda.current_stream(self.device) if self.cuda else None
        with block.model.lock:
            if block.count == 0:
                if self.cuda and torch.cuda.is_current_stream_capturing():
                    raise Unbound(f"block {block.path!r} was not prepared before capture")
                self._bind(block, stream)
            elif stream is not None and stream != block.stream and not torch.cuda.is_current_stream_capturing():
                self.crossed += 1  # SPIKE DIAG
                stream.wait_stream(block.stream)  # type: ignore[arg-type]  # a capture binds under a hold
            block.count += 1

    def _exit(self, block: _Block) -> None:
        stream = torch.cuda.current_stream(self.device) if self.cuda else None
        with block.model.lock:
            if stream is not None and block.stream not in (None, stream) and not torch.cuda.is_current_stream_capturing():
                self.crossed += 1  # SPIKE DIAG
                block.stream.wait_stream(stream)  # type: ignore[union-attr]
            block.count -= 1
            if block.count == 0:
                pin = block.pin
                _unbind(block)
                if block.track is not None and _escaped(block.track):
                    self._escape(block, pin)

    def _escape(self, block: _Block, pin: Pin | None) -> None:
        """A tensor of the block's kept views outlived its run (a package kept a weight or a view
        of one): its window's tensors take the pin, so that page stays bound while they live,
        and the block's next bind makes views anew."""
        model = block.model
        if pin is not None and block.window is not None:
            block.window.keep(pin)
            key = (model.name, block.path)
            self.escaped[key] = self.escaped.get(key, 0) + pin.len
            r = _request.get() or (next(iter(self.requests)) if len(self.requests) == 1 else None)
            if r is not None:
                r.escaped[key] = r.escaped.get(key, 0) + pin.len
            warnings.warn(f"{model.name}: a weight tensor of block {block.path or 'common'!r} outlived its forward; its page stays bound while it lives", stacklevel=2)
        block.views = block.assign = block.window = block.track = block.at = None

    def _bind(self, block: _Block, stream: torch.cuda.Stream | None, pin: Pin | None = None) -> None:
        model = block.model
        if pin is None:
            pin = self.native.bind(model.id, block.region, stream.cuda_stream if stream is not None else 0)
        if self.torch.kind in ("cpu", "cuda"):
            if block.views is None or block.at != pin.ptr:  # a new mapping: its views once
                window = pin.window()
                block.views, block.at, block.assign, block.window = _views(block, torch.from_dlpack(window)), pin.ptr, None, window
                block.track = _track(block.views)
            views = block.views
            if pin.fresh and model.fill is not None:
                model.fill(block.path, views)  # in place: the views alias the page
                pin.filled()
        else:  # MPS: the host bytes are baked, then copied in for the block's run
            base = torch.from_dlpack(pin)
            if pin.fresh and model.fill is not None:
                model.fill(block.path, _views(block, base))
                pin.filled()
            views, block.assign = _views(block, base.to(self.device)), None
        block.pin, block.stream = pin, stream
        try:
            assign = block.assign
            if assign is None:
                assign, kept = _assign(block, views)
                if kept and views is block.views:
                    block.assign, block.track = assign, None  # the old record would count itself
                    block.track = _track(block.views, assign)
            for site, value in assign:
                block.written.append((site, site.table[site.attr]))
                site.table[site.attr] = value
        except BaseException:
            _unbind(block)
            raise


class Request:
    """A request in flight (`Memory.request`). `oom` and `cut` are cozy-runtime's catch's."""

    def __init__(self, memory: Memory, native: NativeRequest) -> None:
        self.memory, self.native = memory, native
        self.ended: Stats | None = None
        #: torch's allocator churn over its passes (retries, device allocs and frees, syncs)
        self.churn: dict[str, int] = {}
        #: (set, block) -> bytes kept bound in it for weight tensors that outlived their run
        self.escaped: dict[tuple[str, str], int] = {}

    def plan(self, steps: Sequence[object]) -> None:
        """Replaces the plan (as `Memory.request` takes it); its cursor starts over."""
        self.native.plan(self.memory._plan(steps))

    def oom(self, need: int | None = None, site: str | None = None, collective: bool = False, finest: int = 0, shape: str = "") -> Next:
        """An op ran out of memory: varena's answer, with torch's cap applied after it. `site` is
        its cut site (None: it cannot be cut), `shape` the op's shape key, cut no finer than
        `finest`; a `collective` site's finer level waits for the next pass's agreement. Outside
        any pass (package code between passes) the request's memo of (site, shape) answers."""
        p = _pass.get()
        answer = self.native.oom(p.set if p else None, *self.memory.torch.now(), need, site, collective, finest, shape)
        if p is not None:
            p.cuts.clear()
        # The failed attempt's peak is not the pass's: when it ran in its site, it is dropped, and
        # the retry is measured afresh from here.
        attempt = p is not None and bool(p.open) and p.open[-1][0] == site
        self.memory._fold(discard=p if attempt else None)
        self.memory._cap()
        return answer

    def parked(self, nbytes: int) -> None:
        """The bytes a `Next.Park(budget)` answer moved to the host."""
        self.native.parked(nbytes)

    def cut(self, site: str, shape: str = "") -> int:
        """The current pass's level for cut site `site` (0: whole) at the op's shape key
        `shape`, read from varena once a pass per (site, shape) until an oom answer or an
        agreement; outside any pass (a split in package code) the request's memo of it."""
        p = _pass.get()
        if p is None:
            return self.native.cut(None, site, shape)
        level = p.cuts.get((site, shape))
        if level is None:
            level = p.cuts[(site, shape)] = self.native.cut(p.set, site, shape)
        return level

    def levels(self) -> list[tuple[str, int]]:
        """The current pass's proposed `(site, level)` for its collective cut sites; the group
        agrees them (cozy-runtime's all-reduce) and hands them to `agree` before any exchange."""
        return _open_pass().native.levels()

    def agree(self, levels: list[tuple[str, int]]) -> None:
        """The levels every rank uses for the current pass's collective sites."""
        p = _open_pass()
        p.cuts.clear()
        p.native.agree(levels)

    def measured(self) -> bool:
        """Whether the current pass's room came from a measurement (False with no pass open)."""
        p = _pass.get()
        return p is not None and p.native.measured()

    def yielding(self, family: str) -> Yielded:
        """`with r.yielding(family):` while the request waits on a call of its `family` (a
        child): its room goes to that family alone until the wait ends."""
        return self.native.yielding(family)

    @contextmanager
    def site(self, name: str, collective: bool = False, finest: int = 0) -> Iterator[None]:
        """Measures cut site `name`'s op (a `collective`'s: its levels are agreed per pass; cut no
        finer than `finest`): its allocated peak goes into the open pass's sites and its peak."""
        p = _pass.get()
        if p is None:
            yield
            return
        memory = self.memory
        memory._fold()
        entry: list[object] = [name, memory.torch.now()[0]]
        p.open.append(entry)
        try:
            yield
        finally:
            memory._fold()
            p.open.remove(entry)
            p.sites[name] = (max(p.sites.get(name, (0,))[0], entry[1]), collective, finest)  # type: ignore[call-overload]

    def stats(self) -> Stats:
        return self.ended if self.ended is not None else self.native.stats()


class Cache:
    """Bytes held outside a forward (KV, graph pools); `cached` reports what it holds now."""

    def __init__(self, memory: Memory, id: int) -> None:
        self.memory, self.id = memory, id

    def cached(self, nbytes: int) -> None:
        self.memory.native.cached(self.id, nbytes)

    def close(self) -> None:
        self.memory.native.uncache(self.id)


# ------------------------------------------------------------------------------ helpers


#: torch's allocator counters a capped pass's churn shows in.
CHURN = ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_sync_all_streams")


class _Allocator:
    """torch's allocator on the memory's device: its bytes, peaks, cap and trim. CUDA's keeps
    peaks. MPS's keeps none, so its peak is what each fold reads. The CPU's is the host's
    malloc: nothing to read or cap."""

    def __init__(self, device: torch.device) -> None:
        self.device, self.kind = device, device.type
        self.total = torch.cuda.mem_get_info(device)[1] if self.kind == "cuda" else 0
        if self.kind == "mps":
            self.total = torch.mps.recommended_max_memory()

    def now(self) -> tuple[int, int]:
        """Bytes allocated and reserved now."""
        if self.kind == "cuda":
            return torch.cuda.memory_allocated(self.device), torch.cuda.memory_reserved(self.device)
        if self.kind == "mps":
            return torch.mps.current_allocated_memory(), torch.mps.driver_allocated_memory()
        return 0, 0

    def peaks(self) -> tuple[int, int]:
        """The allocated and reserved peaks since the last reset."""
        if self.kind == "cuda":
            return torch.cuda.max_memory_allocated(self.device), torch.cuda.max_memory_reserved(self.device)
        return self.now()

    def reset(self) -> None:
        if self.kind == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def churn(self) -> dict[str, int]:
        """The counters of the allocator working against its cap: retries after freeing its
        cache, device allocations and frees, and the syncs they cost (CUDA's; zero elsewhere)."""
        stats = torch.cuda.memory_stats(self.device) if self.kind == "cuda" else {}
        return {k: int(stats.get(k, 0)) for k in CHURN}

    def cap(self, limit: int) -> None:
        if self.kind == "cuda":
            torch.cuda.set_per_process_memory_fraction(min(limit / self.total, 1.0), self.device)
        elif self.kind == "mps":
            # MPS takes a fraction of its recommended working set, at most twice it.
            torch.mps.set_per_process_memory_fraction(min(limit / self.total, 2.0))

    def trim(self) -> int:
        """`empty_cache`, returning the bytes it gave back (varena calls it when it trims)."""
        before = self.now()[1]
        if self.kind == "cuda":
            torch.cuda.empty_cache()
        elif self.kind == "mps":
            torch.mps.empty_cache()
        return max(before - self.now()[1], 0)


def _track(views: Mapping[str, Mapping[str, Tensor]], assign: Sequence[tuple[_Site, Tensor]] = ()) -> tuple[list[Tensor], list[int], int]:
    """The kept tensors with nothing but the adapter holding them: each one's reference count
    and their storage's use count, which a kept weight or a view of one raises."""
    tensors = [t for parts in views.values() for t in parts.values()] + [t for _, t in assign]
    return tensors, _references(tensors), _uses(tensors[0]) if tensors else 0


def _references(tensors: list[Tensor]) -> list[int]:
    return [sys.getrefcount(t) for t in tensors]


def _uses(t: Tensor) -> int:
    return torch._C._storage_Use_Count(t.untyped_storage()._cdata)


def _escaped(track: tuple[list[Tensor], list[int], int]) -> bool:
    """Whether anything but the adapter holds a kept tensor or a view of one."""
    tensors, references, uses = track
    return bool(tensors) and (_references(tensors) != references or _uses(tensors[0]) != uses)


def _views(block: _Block, base: Tensor) -> dict[str, dict[str, Tensor]]:
    """The block's stored parts as typed views of its bytes `base`."""
    return {
        key: {role: base[a.offset : a.offset + a.nbytes].view(a.dtype).view(a.shape) for role, a in roles.items()}
        for key, roles in block.tensors.items()
    }


def _open_pass() -> _Pass:
    """The pass the calling op runs in."""
    p = _pass.get()
    if p is None:
        raise Unbound("no pass is open here: a loaded model's forward opens one")
    return p


def _fill(fill: Filling | None) -> Fill:
    """A fill as varena names it: none, declared by its identity, or undeclared."""
    if fill is None:
        return Fill.Stored()
    return Fill.Undeclared() if fill.identity is None else Fill.Declared(fill.identity)


def _block_of(key: str, paths: Sequence[str]) -> str:
    found = [p for p in paths if key == p or key.startswith(p + ".")]
    return max(found, key=len) if found else ""


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


def _placeholders(model: Model) -> None:
    """Every meta tensor becomes a placeholder on the execution device."""
    done: set[int] = set()
    for group in model.sites.values():
        meta = group[0].table[group[0].attr]
        if meta is None or id(meta) in done or meta.device.type != "meta":
            continue
        done.add(id(meta))
        held: Tensor = Placeholder(meta.shape, meta.dtype, model.memory.device)
        if group[0].parameter:
            held = nn.Parameter(held, requires_grad=False)
        for site in group:
            site.table[site.attr] = held


def _hook(model: Model) -> None:
    """The root's or a direct child's outermost forward is a pass (its hooks run first); a block
    binds while it runs."""
    memory = model.memory
    for entry, sub in (("", model.module), *model.module.named_children()):
        opened = lambda args, kwargs, entry=entry: memory._open(model, entry, args, kwargs)  # noqa: E731
        model.handles += _paired(sub, opened, lambda: memory._close(model))
    for block in model.blocks.values():
        sub = model.module.get_submodule(block.path)
        model.handles += _paired(sub, lambda *_, b=block: memory._enter(b), lambda b=block: memory._exit(b))


_threads = threading.local()


def _owed() -> dict[object, int]:
    """Per thread, the `exit`s each hook pair owes."""
    owed = getattr(_threads, "owed", None)
    if owed is None:
        owed = _threads.owed = {}
    return owed


def _paired(sub: nn.Module, enter: Callable[..., None], exit: Callable[[], None]) -> list[torch.utils.hooks.RemovableHandle]:
    """`enter(args, kwargs)` before `sub`'s forward and `exit()` after it, even when it raises.
    torch also runs an always-call hook when a pre-hook raised: `exit` runs only after its
    `enter` completed."""
    key = object()

    def pre(_: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        enter(args, kwargs)
        owed = _owed()
        owed[key] = owed.get(key, 0) + 1

    def post(*_: object) -> None:
        owed = _owed()
        if owed.get(key):
            owed[key] -= 1
            exit()

    return [sub.register_forward_pre_hook(pre, with_kwargs=True), sub.register_forward_hook(post, always_call=True)]


def _assign(block: _Block, views: Mapping[str, Mapping[str, Tensor]]) -> tuple[list[tuple[_Site, Tensor]], bool]:
    """What the block's module tensors become, through the model's `bind`; and whether every one
    is a view itself (not a tensor made from one), so it can be kept with the views."""
    model, out, kept = block.model, [], True
    for key, parts in views.items():
        if model.bind is not None:
            made = model.bind(key, parts)
        elif set(parts) == {"value"}:
            made = {key: parts["value"]}
        else:
            raise Unbound(f"{model.name}: {key} is encoded ({sorted(parts)}) and no bind was given")
        for name, value in made.items():
            kept = kept and any(value is part for part in parts.values())
            for site in model.sites.get(name, ()):
                held = site.table[site.attr]
                if held is not None and held.dtype != value.dtype:
                    value, kept = value.to(held.dtype), False  # a converted module (`module.to(dtype)`)
                out.append((site, nn.Parameter(value, requires_grad=False) if site.parameter else value))
    return out, kept


def _unbind(block: _Block) -> None:
    """Placeholders back, and the pin goes (the kept views own none)."""
    for site, held in reversed(block.written):
        site.table[site.attr] = held
    block.written.clear()
    block.pin, block.stream = None, None
