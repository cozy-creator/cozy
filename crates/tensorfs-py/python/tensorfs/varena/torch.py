"""varena's memory for torch modules: conversion, with no policy (design varena-slim.md §2).

    mem = Memory.open(0)  # before torch initializes CUDA
    vars(unet)["_varena"] = Declared(Weights(ck, "unet"))
    mem.load(unet, "gpu")
    with mem.request("r1", plan=[te, unet, vae]):
        image = pipe(prompt)

A block's params are views of its pin only while the block runs; placeholders (shape, dtype and
device, no bytes) stand in otherwise, so no tensor outlives its page. A model's top-level forward
is one pass: varena makes its measured room, torch's allocator is capped to `torch_limit`, and
the pass reports torch's allocated peak, the allocator's slack, the context's growth and each cut
site's peak. cozy-runtime's catch calls `r.oom` and `r.cut` and measures sites with `r.site`.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

import tensorfs
import torch
from torch import Tensor, nn
from torch.utils._pytree import tree_leaves

from . import Exhausted, Fill, GpuId, Grant, Level, Next, Pass, Pin, Stats, VarenaError
from . import Memory as Native
from . import Request as NativeRequest
from . import Set

__all__ = [
    "Bind", "Cache", "Declared", "Exhausted", "Filling", "Memory", "Model", "Next", "Placeholder",
    "Request", "Stats", "Unbound", "VarenaError", "Weights", "blocks_of",
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

    identity: str | None

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


@dataclass(eq=False, slots=True)
class _Pass:
    """An open pass and what it measured so far; `open` holds each running site's peak."""

    native: Pass
    request: Request
    allocated: int
    reserved: int
    used: int
    paged: int
    peak: int = 0
    reserved_peak: int = 0
    sites: dict[str, int] = field(default_factory=dict)
    open: list[list[object]] = field(default_factory=list)
    #: the exception being handled where it opened: another at its close is a raise
    entered: BaseException | None = None
    token: object = None


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
        self.depth = 0
        self.passing: _Pass | None = None

    def close(self) -> None:
        """Unhooks the module and unloads its set."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.memory.models.pop(self.module, None)
        self.memory.native.unload(self.id)


# -------------------------------------------------------------------------- the memory

#: The request a forward runs in, and the pass a site op runs in.
_request: ContextVar[Request | None] = ContextVar("varena_request", default=None)
_pass: ContextVar[_Pass | None] = ContextVar("varena_pass", default=None)


class Memory:
    """This process's memory on one device: its models, caches and requests."""

    def __init__(self, native: Native, device: torch.device) -> None:
        self.native, self.device = native, device
        self.cuda = device.type == "cuda"
        self.models: dict[nn.Module, Model] = {}
        self.requests: dict[Request, None] = {}
        #: the allocator's slack, given back by `empty_cache` when varena's decision asks
        self.torch = native.cache("torch", self._trim) if self.cuda else None
        self.total = torch.cuda.mem_get_info(device)[1] if self.cuda else 0

    @classmethod
    def open(cls, gpu: int | str | torch.device | GpuId = 0, grant: Grant | None = None) -> Memory:
        """A TensorD executor's memory (`grant`), or a standalone engine's (the device's table by
        name). On CUDA the backend opens first and measures this process's context, so open it
        before torch initializes CUDA, or torch's context counts as foreign bytes. CPU and MPS use
        host pages on the machine's table `host` (its RAM); a `GpuId` names a host table and its
        capacity explicitly."""
        if isinstance(gpu, GpuId):
            return cls(Native.host(gpu, grant), torch.device("cpu"))
        device = torch.device("cuda", gpu) if isinstance(gpu, int) else torch.device(gpu)
        if device.type == "cuda":
            return cls(Native.cuda(device.index or 0, grant), torch.device("cuda", device.index or 0))
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return cls(Native.host(GpuId("host", device.type, ram), grant), device)

    @property
    def gpu(self) -> GpuId:
        return self.native.gpu

    # -- loading

    def load(self, module: nn.Module, level: Level | str = Level.GPU) -> Model:
        """`module` (built on meta) from its `_varena` weights at `level`: one region per block
        and one for common, its fill's identity in the fingerprint, and hooks that bind blocks
        while they run."""
        declared = vars(module).get("_varena")
        if not isinstance(declared, Declared):
            raise Unbound(f"{type(module).__name__}: no weights declared (`_varena = Declared(...)`)")
        w, paths = declared.weights, blocks_of(module)
        owned: dict[str, list[str]] = {}
        for key in w.ck.components[w.component].tensors:
            if w.keys is None or key in w.keys:
                owned.setdefault(_block_of(key, paths), []).append(f"{w.component}/{key}")
        order = [p for p in ("", *paths) if p in owned]
        fill = declared.fill
        kind = Fill.Stored() if fill is None else Fill.Undeclared() if fill.identity is None else Fill.Declared(fill.identity)
        name = f"{w.release or w.ck.id}/{w.component}"
        set = Set(name, w.ck, [owned[p] for p in order], fill=kind, release=w.release, shard=w.shard, gpu=self.gpu.uuid)
        if isinstance(level, str):
            level = Level.GPU if level == "gpu" else Level.RAM
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
            yield r
        finally:
            _request.reset(token)
            self.requests.pop(r, None)
            r.ended = r.native.end()
            self._slack()

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

    def _cap(self) -> None:
        """torch's allocator may hold `torch_limit` bytes from now on."""
        if self.cuda:
            fraction = min(self.native.torch_limit() / self.total, 1.0)
            torch.cuda.set_per_process_memory_fraction(fraction, self.device)

    def _trim(self, want: int) -> int:
        before = torch.cuda.memory_reserved(self.device)
        torch.cuda.empty_cache()
        self._slack()
        return max(before - torch.cuda.memory_reserved(self.device), 0)

    def _slack(self) -> None:
        if self.torch is not None:
            held = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
            self.native.cached(self.torch, max(held, 0))

    # -- passes

    def _open(self, model: Model, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        """The outermost forward of `model` (its root or a direct child) is a pass."""
        with model.lock:
            model.depth += 1
            if model.depth > 1:
                return
            try:
                r = self._current()
                shape = "x".join(str(d) for t in tree_leaves((args, kwargs)) if isinstance(t, Tensor) for d in t.shape)
                allocated, reserved = _torch(self)
                native = r.native.pass_(model.id, shape, allocated, reserved)
                self._cap()
                p = _Pass(native, r, allocated, reserved, _used(self), _paged(r))
                p.entered = sys.exc_info()[1]
                if self.cuda:
                    torch.cuda.reset_peak_memory_stats(self.device)
                if model.common is not None:
                    self._enter(model.common)
                model.passing, p.token = p, _pass.set(p)
            except BaseException:
                model.depth -= 1
                raise

    def _close(self, model: Model) -> None:
        with model.lock:
            model.depth -= 1
            p = model.passing
            if model.depth or p is None:
                return
            model.passing = None
            _pass.reset(p.token)  # type: ignore[arg-type]
            if model.common is not None:
                self._exit(model.common)
            if sys.exc_info()[1] is not p.entered:
                return  # torch runs this hook in its handler of the forward's raise: unended, it records nothing
            peak = slack = context = 0
            if self.cuda:
                _fold(p, self.device)
                allocated, reserved = _torch(self)
                peak, slack = p.peak, max(p.reserved_peak - p.peak, 0)
                grew = (_used(self) - p.used) - (reserved - p.reserved) - (_paged(p.request) - p.paged)
                context = max(grew, 0)
            p.native.end(peak, slack, context, list(p.sites.items()))
            self._slack()

    # -- blocks

    def _enter(self, block: _Block) -> None:
        stream = torch.cuda.current_stream(self.device) if self.cuda else None
        with block.model.lock:
            if block.count == 0:
                self._bind(block, stream)
            elif stream is not None and stream != block.stream:
                stream.wait_stream(block.stream)  # type: ignore[arg-type]
            block.count += 1

    def _exit(self, block: _Block) -> None:
        stream = torch.cuda.current_stream(self.device) if self.cuda else None
        with block.model.lock:
            if stream is not None and block.stream not in (None, stream):
                block.stream.wait_stream(stream)  # type: ignore[union-attr]
            block.count -= 1
            if block.count == 0:
                _unbind(block)

    def _bind(self, block: _Block, stream: torch.cuda.Stream | None) -> None:
        model = block.model
        pin = self.native.bind(model.id, block.region, stream.cuda_stream if stream is not None else 0)
        base = torch.from_dlpack(pin)
        if base.device != self.device and not self.cuda:
            base = base.to(self.device)  # MPS: host pages are copied in for the block's run
        views = {
            key: {role: base[a.offset : a.offset + a.nbytes].view(a.dtype).view(a.shape) for role, a in roles.items()}
            for key, roles in block.tensors.items()
        }
        if pin.fresh and model.fill is not None:
            model.fill(block.path, views)
            pin.filled()
        block.pin, block.stream = pin, stream
        try:
            _write(block, views)
        except BaseException:
            _unbind(block)
            raise


class Request:
    """A request in flight (`Memory.request`). `oom` and `cut` are cozy-runtime's catch's."""

    def __init__(self, memory: Memory, native: NativeRequest) -> None:
        self.memory, self.native = memory, native
        self.ended: Stats | None = None

    def plan(self, steps: Sequence[object]) -> None:
        """Replaces the plan (as `Memory.request` takes it); its cursor starts over."""
        self.native.plan(self.memory._plan(steps))

    def oom(self, need: int | None = None, site: str | None = None, collective: bool = False) -> Next:
        """An op ran out of memory: varena's answer, with torch's cap applied after it. A
        `collective` site's finer level waits for the next pass's agreement."""
        answer = self.native.oom(need, site, collective)
        self.memory._cap()
        return answer

    def parked(self, nbytes: int) -> None:
        """The bytes a `Next.Park(budget)` answer moved to the host."""
        self.native.parked(nbytes)

    def cut(self, site: str) -> int:
        """The current pass's level for cut site `site` (0: whole)."""
        return self.native.cut(site)

    @contextmanager
    def site(self, name: str) -> Iterator[None]:
        """Measures cut site `name`'s op: its allocated peak goes into the open pass's sites and
        its peak."""
        p = _pass.get()
        if p is None or not self.memory.cuda:
            yield
            return
        device = self.memory.device
        _fold(p, device)
        entry: list[object] = [name, torch.cuda.memory_allocated(device)]
        p.open.append(entry)
        try:
            yield
        finally:
            _fold(p, device)
            p.open.remove(entry)
            p.sites[name] = max(p.sites.get(name, 0), entry[1])  # type: ignore[call-overload]

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


def _torch(memory: Memory) -> tuple[int, int]:
    if not memory.cuda:
        return 0, 0
    return torch.cuda.memory_allocated(memory.device), torch.cuda.memory_reserved(memory.device)


def _used(memory: Memory) -> int:
    """The device's used bytes, as the driver counts them."""
    if not memory.cuda:
        return 0
    free, total = torch.cuda.mem_get_info(memory.device)
    return total - free


def _paged(r: Request) -> int:
    """Pages this request's fills created less those its evictions removed (kept pages import
    existing memory)."""
    if not r.memory.cuda:
        return 0
    s = r.native.stats()
    return s.moved.tier.bytes + s.moved.store.bytes - sum(s.evicted)


def _fold(p: _Pass, device: torch.device) -> None:
    """The allocator's peaks since the last reset into the pass and its open sites; then reset."""
    peak = torch.cuda.max_memory_allocated(device)
    p.peak = max(p.peak, peak)
    p.reserved_peak = max(p.reserved_peak, torch.cuda.max_memory_reserved(device))
    for entry in p.open:
        entry[1] = max(entry[1], peak)  # type: ignore[call-overload]
    torch.cuda.reset_peak_memory_stats(device)


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
    for sub in (model.module, *model.module.children()):
        model.handles += _paired(sub, lambda args, kwargs: memory._open(model, args, kwargs), lambda: memory._close(model))
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


def _write(block: _Block, views: Mapping[str, Mapping[str, Tensor]]) -> None:
    """The block's module tensors become its views, through the model's `bind`."""
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
                site.table[site.attr] = nn.Parameter(value, requires_grad=False) if site.parameter else value


def _unbind(block: _Block) -> None:
    """Placeholders back; the views go, and with the last of them the pin."""
    for site, held in reversed(block.written):
        site.table[site.attr] = held
    block.written.clear()
    block.pin, block.stream = None, None
