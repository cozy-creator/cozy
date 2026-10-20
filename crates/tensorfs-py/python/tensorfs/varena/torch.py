"""Torch lifetime hooks for the native page-placement manager.

TensorFS reads bytes. Native varena places pages and follows declared request plans.
This module describes blocks, binds views while they run, and delegates exact OOM cuts.
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import threading
import time
import traceback
import weakref
from collections.abc import Callable, Collection, Hashable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypeAlias

import torch
from torch import Tensor, nn
from torch.utils._python_dispatch import _disable_current_modes
from torch.utils._pytree import tree_map_only
from torch.utils.hooks import RemovableHandle

import tensorfs

from . import Copy as CopyRun
from . import Plane, VarenaError, WeightSet, copies, cuts

#: TensorFS dtype names; an `F4` part is u8 pairs.
DTYPES: dict[str, torch.dtype] = {
    name: getattr(torch, kind)
    for name, kind in (
        pair.split("=")
        for pair in [
            "F64=float64",
            "F32=float32",
            "F16=float16",
            "BF16=bfloat16",
            "F8_E4M3=float8_e4m3fn",
            "F8_E5M2=float8_e5m2",
            "F8_E8M0=float8_e8m0fnu",
            "I64=int64",
            "I32=int32",
            "I16=int16",
            "I8=int8",
            "U8=uint8",
            "BOOL=bool",
            "F4=uint8",
        ]
    )
}


class Level(Enum):
    RAM = "ram"
    GPU = "gpu"


@dataclass(frozen=True, slots=True)
class Source:
    ck: tensorfs.Checkpoint
    component: str
    identity: str
    keys: Collection[str] | None = None
    fingerprint: str = ""

    @property
    def key(self) -> str:
        return f"{self.identity}/{self.component}"


class Exhausted(Exception):
    """It still does not fit with every other evictable byte gone: §2's one memory error, with
    each model's resident bytes and the memo's peaks then."""

    def __init__(
        self,
        gpu: int,
        need: int,
        op: str,
        resident: Mapping[str, int],
        peaks: tuple[Peak, ...],
        held: str = "",
    ) -> None:
        super().__init__(
            f"GPU {gpu}: {op} needs {need} B more and nothing else can be freed"
            + (f" ({held})" if held else "")
        )
        self.gpu, self.need, self.op, self.resident, self.peaks = (
            gpu,
            need,
            op,
            resident,
            peaks,
        )


class Unbound(Exception):
    """An op on a placeholder that binds to nothing."""


#: `bind(key, parts)`: a stored tensor's part views -> the module tensors they become, by name.
type Bind = Callable[[str, Mapping[str, Tensor]], Mapping[str, Tensor]]


class Fill(Protocol):
    """A page-in transform (a LoRA bake) on fresh bytes. `prepare()`, as each forward starts,
    readies the module and names what filling adds; pages filled under another name refill."""

    def __call__(
        self, block: Block, parts: Mapping[str, Mapping[str, Tensor]], /
    ) -> None: ...

    def prepare(self) -> Hashable: ...


class Shrink(Protocol):
    def __call__(self, nbytes: int, /) -> int: ...


@dataclass(frozen=True, slots=True)
class Declared:
    """What the author API attaches as `module._varena`: its weights, its blocks (default: the
    innermost `_no_split_modules`), the attributes a retried block puts back (`state`), `fill`,
    and `bind` (default: a plain part as itself)."""

    source: Source | None = None
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
class Copy:
    """A run of weight copies since the last request (`varena.copies()`): a restore, refill or
    fill (`h2d`), or a host layout read from disk (`disk`), against the measured `ceiling` GB/s.
    `busy_s` is the GPU's copy time, `pin_s` the host's page locking."""

    kind: str
    gpu: int
    bytes: int
    seconds: float
    gbps: float
    ceiling: float
    pct: float
    link: str
    busy_s: float
    pin_s: float
    staged: int
    cached: int


def _copy(c: CopyRun) -> Copy:
    gbps = c.bytes / c.ns if c.ns else 0.0
    pct = round(100 * gbps / c.ceiling, 1) if c.ceiling else 0.0
    seconds, busy, pin = c.ns / 1e9, c.busy_ns / 1e9, c.pin_ns / 1e9
    ceiling = round(c.ceiling, 2)
    return Copy(
        c.kind,
        c.gpu,
        c.bytes,
        seconds,
        round(gbps, 2),
        ceiling,
        pct,
        c.link,
        busy,
        pin,
        c.staged,
        c.cached,
    )


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
    #: its declared plan (`Memory.plan`), each step's models by component name
    plan: tuple[tuple[str, ...], ...] = ()
    copies: tuple[Copy, ...] = ()
    trace: tuple[Mapping[str, object], ...] = ()


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
            return cls(
                held.shape,
                kwargs.get("dtype") or held.dtype,
                device,
                held.block,
                held.site,
            )
        for i, argument in enumerate(func._schema.arguments):
            written = args[i] if i < len(args) else kwargs.get(argument.name)
            if (
                argument.alias_info is not None
                and argument.alias_info.is_write
                and isinstance(written, Placeholder)
            ):
                raise Unbound(
                    f"{func}: an op writes into a weight no pages are bound to"
                )
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
        raise Unbound(
            f"a weight of {p.block.model.name if p.block else '?'} binds to nothing"
        )
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
    binding: bool = False
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
        view = lambda a: (
            base[a.offset : a.offset + a.nbytes].view(a.dtype).view(a.shape)
        )
        return {
            key: {role: view(a) for role, a in roles.items()}
            for key, roles in self.tensors.items()
            if next(iter(roles.values())).region == region
        }


class Model:
    """A registered component and its lifetime hooks; residency decisions stay native."""

    def __init__(self, memory: Memory, module: nn.Module, declared: Declared) -> None:
        if declared.source is None:
            raise Unbound("module has no TensorFS source")
        self.memory, self.module, self.source = memory, module, declared.source
        self.name, self.component = self.source.key, self.source.component
        self.device, self.gpu = memory.device, memory.gpu
        self.fill, self.bind = declared.fill, declared.bind
        self.sites = _sites(module)
        self.sizes: dict[int, int] = {}
        self.blocks: list[Block] = []
        self.common = Block(self, -1, "", module)
        self.handles: list[RemovableHandle] = []
        self.ws: WeightSet
        self.flight, self.aloft, self.held = 0, False, 0
        self.starting = False
        self.peak = 0
        self.steps, self.base = 0, 0
        self.binding_ns = 0
        self.closed = False
        self.version: Hashable | None = None
        self.track = cuts.Track(self.name)

    @property
    def host(self) -> bool:
        return not self.memory.cuda

    @property
    def nbytes(self) -> int:
        return self.ws.nbytes if self.memory.cuda else sum(self.sizes.values())

    def home_bytes(self) -> int:
        return self.memory.resident().get(self.name, 0)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        with ExitStack() as cleanup:
            cleanup.callback(self.memory._forget, self)
            weights = getattr(self, "ws", None)
            if self.memory.native is not None and weights is not None:
                cleanup.callback(self.memory.native.close_model, weights.id)
            for block in (self.common, *self.blocks):
                cleanup.callback(_unbind, block)
                cleanup.callback(_unhook, block)
            for handle in self.handles:
                cleanup.callback(handle.remove)


class Hold:
    def __init__(self, model: Model, cache: Cache, blocks: Sequence[Block]) -> None:
        self.model, self.cache, self.blocks = model, cache, list(blocks)
        self.open = True

    def close(self) -> None:
        if self.open:
            self.open = False
            for block in reversed(self.blocks):
                self.model.memory._exit(block)
            self.model.held -= 1
            self.cache.holds.discard(self)


class Cache:
    """A declared reclaim callback; only the native policy decides when to invoke it."""

    def __init__(self, memory: Memory, owner: str, shrink: Shrink) -> None:
        self.memory, self.owner, self.shrink = memory, owner, shrink
        self.holds: set[Hold] = set()
        self._nbytes = 0
        self.id = memory.native.cache(0, shrink) if memory.native is not None else 0
        self.closed = False

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @nbytes.setter
    def nbytes(self, value: int) -> None:
        self._nbytes = value
        if self.memory.native is not None:
            self.memory.native.cache_bytes(self.id, value)

    @contextmanager
    def serving(self) -> Iterator[None]:
        native, active = self.memory.native, False
        try:
            if native is not None:
                request = self.memory._current()
                native.cache_active(self.id, True)
                active = True
                for hold in self.holds:
                    native.retained_bind(request.native_id, hold.model.ws.id)
            with cuts.caught() if self.memory.cuda else nullcontext():
                yield
        finally:
            if native is not None and active and not self.closed:
                native.cache_active(self.id, False)

    def hold(self, model: Model) -> Hold:
        if model.home_bytes() < model.nbytes:
            raise Unbound("streamed weights cannot be held by a graph")
        blocks = [model.common, *model.blocks]
        entered: list[Block] = []
        try:
            for block in blocks:
                self.memory._enter(block, record=False)
                entered.append(block)
        except BaseException:
            for block in reversed(entered):
                self.memory._exit(block)
            raise
        model.held += 1
        hold = Hold(model, self, blocks)
        self.holds.add(hold)
        return hold

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for hold in list(self.holds):
            hold.close()
        if self.memory.native is not None:
            self.memory.native.close_cache(self.id)
        self.memory.caches.discard(self)


@dataclass(eq=False, slots=True)
class Request:
    id: str
    native_id: int
    memory: Memory
    start: int
    steps: list[Step] = field(default_factory=list)
    declared: tuple[tuple[str, ...], ...] = ()
    stats: RequestStats | None = None
    raw_plan: tuple[PlanStep, ...] = ()
    reads: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    passes: dict[str, int] = field(default_factory=dict)

    def plan(self, steps: Sequence[PlanStep] | None) -> None:
        self.raw_plan = tuple(
            step if isinstance(step, (str, Model, nn.Module)) else tuple(step)
            for step in (steps or ())
        )
        names, models = self.memory._plan(self.raw_plan)
        self.declared = names
        if self.memory.native is not None:
            self.memory.native.declare(self.native_id, models)


PlanStep: TypeAlias = "Model | nn.Module | str | Collection[Model | nn.Module | str]"


class Memory:
    """One native page manager per device. Python only supplies declarations and hooks."""

    def __init__(
        self, device: torch.device, grant: tuple[int, int, int] | None
    ) -> None:
        self.device = device
        self.gpu = (device.index or 0) if device.type == "cuda" else -1
        self.native = Plane(self.gpu, grant) if self.gpu >= 0 else None
        self.models: dict[str, Model] = {}
        self.by_module: dict[nn.Module, Model] = {}
        self.caches: set[Cache] = set()
        self.active: dict[Request, None] = {}
        self.local = threading.local()
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.loading: set[nn.Module] = set()
        self._ids = itertools.count(1)
        self._names: dict[int, str] = {}
        self.context = 0
        self.measure: Callable[[], int | None] = lambda: None
        self.memo = cuts.Memo()
        cuts.serve(self.memo, self)
        if self.native is not None:
            self.native.trim_with(self._trim)

    @classmethod
    def open(
        cls,
        device: torch.device | None = None,
        grant: tuple[int, int, int] | None = None,
    ) -> Memory:
        if device is None:
            device = (
                torch.device("cuda", 0)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        return cls(device, grant)

    @property
    def cuda(self) -> bool:
        return self.native is not None

    def _trim(self, _need: int) -> int:
        before = torch.cuda.memory_reserved(self.gpu)
        torch.cuda.empty_cache()
        return max(before - torch.cuda.memory_reserved(self.gpu), 0)

    def _measure(self) -> None:
        if self.native is None:
            return
        allocated = torch.cuda.memory_allocated(self.gpu)
        reserved = torch.cuda.memory_reserved(self.gpu)
        measured = self.measure()
        context = (
            self.context
            if measured is None
            else max(measured - reserved - sum(self.resident().values()), 0)
        )
        self.native.allocator(context, allocated, reserved)
        total = torch.cuda.get_device_properties(self.gpu).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(self.native.allocator_budget() / total, 1.0), self.gpu
        )

    def load(
        self,
        module: nn.Module,
        level: Level | str = Level.GPU,
        *,
        ck: tensorfs.Checkpoint | None = None,
        component: str | None = None,
    ) -> Model:
        with self.changed:
            while module in self.loading:
                self.changed.wait()
            self.loading.add(module)
        try:
            return self._load(module, level, ck=ck, component=component)
        finally:
            with self.changed:
                self.loading.remove(module)
                self.changed.notify_all()

    def _load(
        self,
        module: nn.Module,
        level: Level | str,
        *,
        ck: tensorfs.Checkpoint | None,
        component: str | None,
    ) -> Model:
        with self.lock:
            model = self.by_module.get(module)
        if model is not None:
            if self.native is not None:
                self.native.load(model.ws.id, Level(level).value)
            return model

        original_declaration = vars(module).get("_varena")
        declared = _declared(module)
        if ck is not None:
            component = component or next(iter(ck.components))
            declared = Declared(
                Source(ck, component, str(ck.id)),
                declared.blocks,
                declared.state,
                declared.fill,
                declared.bind,
            )
            vars(module)["_varena"] = declared
        model = Model(self, module, declared)
        model.name = f"{model.source.key}:{next(self._ids)}"
        originals = {
            (id(site.table), site.attr): (site, site.table[site.attr])
            for sites in model.sites.values()
            for site in sites
        }
        forwards = [
            (sub, vars(sub).get("forward"), vars(sub).get("_varena_ffn"))
            for sub in module.modules()
        ]
        try:
            if self.native is None:
                self._load_host(model)
            else:
                self._register(model)
                paths = blocks_of(module)
                _layout(model, f"{model.component}/", paths, regions_of(module))
                _install(model)
                for block in model.blocks:
                    block.site = cuts.Site(
                        (model.name, block.index),
                        self.device,
                        model.track,
                        block.module,
                        module,
                        declared.state,
                    )
                    _hook(block)
                cuts.install(module)
                _fly(model)
                self.native.load(model.ws.id, Level(level).value)
        except BaseException as error:
            try:
                model.close()
            except BaseException as cleanup_error:
                error.add_note(f"registration cleanup failed: {cleanup_error}")
                raise error from cleanup_error
            finally:
                for site, value in originals.values():
                    site.table[site.attr] = value
                for sub, forward, cut in forwards:
                    for key, value in (("forward", forward), ("_varena_ffn", cut)):
                        if value is None:
                            vars(sub).pop(key, None)
                        else:
                            vars(sub)[key] = value
                if original_declaration is None:
                    vars(module).pop("_varena", None)
                else:
                    vars(module)["_varena"] = original_declaration
            raise
        with self.lock:
            self.models[model.name], self.by_module[module] = model, model
        return model

    def _forget(self, model: Model) -> None:
        with self.lock:
            self.models.pop(model.name, None)
            self.by_module.pop(model.module, None)
            weights = getattr(model, "ws", None)
            if weights is not None:
                self._names.pop(weights.id, None)

    def _register(self, model: Model) -> None:
        assert self.native is not None
        version = model.fill.prepare() if model.fill is not None else None
        fingerprint = hashlib.sha256(
            repr((model.source.fingerprint, version)).encode()
        ).hexdigest()
        weak, device = weakref.ref(model), self.device

        def fill(region: int, raw: Any, stream: int) -> None:
            current = weak()
            if current is None or current.fill is None:
                return
            block = next(
                b for b in (current.common, *current.blocks) if region in b.regions
            )
            with torch.cuda.stream(torch.cuda.ExternalStream(stream, device=device)):
                current.fill(block, block.parts(region, torch.from_dlpack(raw)))

        model.ws = self.native.register(
            model.name,
            model.source.ck,
            regions_of(model.module),
            fingerprint,
            fill if model.fill is not None else None,
        )
        model.version = version
        with self.lock:
            self._names[model.ws.id] = model.name

    def _load_host(self, model: Model) -> None:
        tensors = model.source.ck.components[model.component].tensors
        keys = model.source.keys if model.source.keys is not None else tensors
        views: dict[str, dict[str, Tensor]] = {}
        for key in keys:
            stored = tensors.read(key, decode=False)
            parts = stored.parts if hasattr(stored, "parts") else {"value": stored}
            views[key] = {
                role: torch.from_dlpack(value).to(self.device)
                for role, value in parts.items()
            }
        if model.fill is not None:
            model.fill.prepare()
            model.fill(model.common, views)
        _write(model.common, views)
        model.common.written.clear()
        model.sizes = {
            0: sum(
                t.numel() * t.element_size()
                for parts in views.values()
                for t in parts.values()
            )
        }

    def _current(self) -> Request:
        requests = getattr(self.local, "requests", ())
        if not requests:
            raise Unbound("a model forward requires Memory.request")
        return requests[-1]

    def _plan(
        self, steps: Sequence[PlanStep]
    ) -> tuple[tuple[tuple[str, ...], ...], list[list[int]]]:
        if self.native is None:
            names = []
            for step in steps:
                entries = [step] if isinstance(step, (str, Model, nn.Module)) else step
                names.append(
                    tuple(
                        sorted(
                            entry.name if isinstance(entry, Model) else str(entry)
                            for entry in entries
                        )
                    )
                )
            return tuple(names), [[] for _ in names]
        names, ids = [], []
        for step in steps:
            entries = [step] if isinstance(step, (str, Model, nn.Module)) else step
            found: list[Model] = []
            for entry in entries:
                if isinstance(entry, Model):
                    matches = [entry]
                elif isinstance(entry, nn.Module):
                    matches = [self.by_module[entry]] if entry in self.by_module else []
                else:
                    matches = [
                        model
                        for model in self.models.values()
                        if entry in (model.name, model.component)
                    ]
                if len(matches) != 1:
                    raise Unbound(f"plan model {entry!r} is absent or ambiguous")
                found.extend(matches)
            names.append(tuple(sorted(model.component for model in found)))
            ids.append(
                [model.ws.id for model in found] if self.native is not None else []
            )
        return tuple(names), ids

    def _fresh(self, model: Model) -> None:
        version = model.fill.prepare() if model.fill is not None else None
        if version != model.version:
            for cache in tuple(self.caches):
                if any(hold.model is model for hold in cache.holds):
                    cache.shrink(cache.nbytes)
            assert self.native is not None
            old = model.ws.id
            self._register(model)
            self.native.close_model(old)
            for request in tuple(self.active):
                request.plan(request.raw_plan)

    def _start(self, model: Model) -> None:
        with self.changed:
            while model.starting:
                self.changed.wait()
            model.flight += 1
            if model.flight != 1:
                return
            model.starting = True
        try:
            self._fresh(model)
            self._measure()
            assert self.native is not None
            self.native.enter(self._current().native_id, model.ws.id)
            with self.lock:
                self._capture_peaks()
                model.base = torch.cuda.memory_allocated(self.gpu)
                model.peak = 0
                torch.cuda.reset_peak_memory_stats(self.gpu)
                model.aloft = True
            self._enter(model.common, record=False)
        except BaseException:
            with self.lock:
                model.flight -= 1
                model.aloft = False
                _unbind(model.common)
            raise
        finally:
            with self.changed:
                model.starting = False
                self.changed.notify_all()

    def _capture_peaks(self) -> None:
        """Preserve every active model's measured peak before a nested start resets CUDA."""
        peak = torch.cuda.max_memory_allocated(self.gpu)
        for model in self.models.values():
            if model.aloft:
                model.peak = max(model.peak, peak - model.base)

    def _land(self, model: Model) -> None:
        with self.changed:
            if not model.flight:
                return
            model.flight -= 1
            if model.flight:
                return
            self._capture_peaks()
            aloft, model.aloft = model.aloft, False
            model.starting = True
        try:
            if aloft:
                self._exit(model.common)
            self._step(model)
        finally:
            with self.changed:
                model.starting = False
                self.changed.notify_all()

    def _enter(
        self,
        block: Block,
        record: bool = True,
        call: tuple = ((), {}),
        wait: bool = True,
    ) -> None:
        del wait
        stream = torch.cuda.current_stream(self.device)
        with self.changed:
            while block.binding:
                self.changed.wait()
            bind = block.count == 0
            block.binding = bind
            held_stream = block.stream
            acquired = not bind
            if acquired:
                block.count += 1
        try:
            if bind:
                with _disable_current_modes():
                    self._bind(block, stream)
            elif (
                stream != held_stream and not torch.cuda.is_current_stream_capturing()
            ):
                stream.wait_stream(held_stream)
            with self.changed:
                if bind:
                    block.count += 1
                    acquired = True
                peak = None
                if record and block.index >= 0:
                    shape = block.model.track.shape = self.memo.shape(
                        block.model.name, cuts.shape_of(*call)
                    )
                    if shape.peak >= 0:
                        peak = shape.peak
            if peak is not None and self.native is not None:
                self.native.leave(self._current().native_id, peak)
        except BaseException:
            if acquired:
                with self.changed:
                    block.count -= 1
                    if not block.count:
                        _unbind(block)
            raise
        finally:
            if bind:
                with self.changed:
                    block.binding = False
                    self.changed.notify_all()

    def _exit(self, block: Block) -> None:
        stream = torch.cuda.current_stream(self.device)
        with self.changed, _disable_current_modes():
            while block.binding:
                self.changed.wait()
            if (
                block.stream not in (None, stream)
                and not torch.cuda.is_current_stream_capturing()
            ):
                block.stream.wait_stream(stream)
            block.count = max(block.count - 1, 0)
            if block.count == 0:
                _unbind(block)

    def _bind(self, block: Block, stream: torch.cuda.Stream) -> None:
        assert self.native is not None
        self._measure()
        block.stream = stream
        views: dict[str, dict[str, Tensor]] = {}
        started = time.perf_counter_ns()
        try:
            for region in block.regions:
                pinned = self._pin(block.model, region, stream.cuda_stream)
                base = torch.from_dlpack(pinned)
                block.pins.append(base)
                views.update(block.parts(region, base))
            _write(block, views)
        except BaseException:
            _unbind(block)
            raise
        finally:
            block.model.binding_ns += time.perf_counter_ns() - started

    def _pin(self, model: Model, region: int, stream: int) -> Any:
        assert self.native is not None
        try:
            return self.native.bind(
                self._current().native_id, model.ws.id, region, stream
            )
        except VarenaError as error:
            if error.code != "DEVICE_FULL":
                raise
            raise self.exhausted(
                self.device, int(getattr(error, "deficit", 0)), f"bind {model.name}"
            ) from error

    def _step(self, model: Model) -> None:
        if self.native is None:
            return
        request = self._current()
        peak = model.peak
        self.native.leave(request.native_id, peak)
        cuts.peaked(model.track, peak)
        raw = self.native.stats(request.native_id)
        copies_ = [
            row for row in raw["copies"] if self._names.get(row["model"]) == model.name
        ]
        paged = sum(row["stats"]["bytes"] for row in copies_ if not row["streamed"])
        streamed = sum(row["stats"]["bytes"] for row in copies_ if row["streamed"])
        prior = request.reads.get(model.name, (0, 0, 0))
        resident = dict(self.native.resident()).get(model.name, 0)
        step = request.passes.get(model.name, 0)
        request.steps.append(
            Step(
                model.name,
                self.gpu,
                step,
                resident,
                resident,
                paged - prior[0],
                streamed - prior[1],
                model.binding_ns - prior[2],
                peak,
            )
        )
        request.passes[model.name] = step + 1
        request.reads[model.name] = (paged, streamed, model.binding_ns)
        model.steps += 1

    @contextmanager
    def _reading(self, blocks: Sequence[Block]) -> Iterator[None]:
        entered: list[Block] = []
        try:
            for block in blocks:
                self._enter(block, record=False)
                entered.append(block)
            yield
        finally:
            for block in reversed(entered):
                self._exit(block)

    def cache(self, owner: str, shrink: Shrink) -> Cache:
        cache = Cache(self, owner, shrink)
        self.caches.add(cache)
        return cache

    @contextmanager
    def request(
        self,
        id: str,
        plan: Sequence[PlanStep] | None = None,
        *,
        package: str = "",
        shape: str = "",
    ) -> Iterator[Request]:
        request = Request(
            id,
            next(self._ids),
            self,
            torch.cuda.memory_allocated(self.gpu) if self.cuda else 0,
        )
        request.reads = {
            model.name: (0, 0, model.binding_ns) for model in self.models.values()
        }
        names, models = self._plan(plan or ())
        request.declared = names
        request.raw_plan = tuple(
            step if isinstance(step, (str, Model, nn.Module)) else tuple(step)
            for step in (plan or ())
        )
        if self.native is not None:
            self._measure()
            self.native.begin(request.native_id, models, package, shape)
        self.active[request] = None
        stack = self.local.__dict__.setdefault("requests", [])
        stack.append(request)
        try:
            with cuts.caught() if self.cuda else nullcontext():
                yield request
        except BaseException as exc:
            _clear_frames(exc)
            raise
        finally:
            stack.remove(request)
            self.active.pop(request, None)
            gc.collect()
            raw: dict[str, Any] = (
                self.native.end(request.native_id)
                if self.native is not None
                else {"events": [], "copies": [], "peak": 0}
            )
            trace = tuple(
                {
                    **event,
                    "request": id,
                    "model": self._names.get(event.get("model"), ""),
                    "gpu": self.gpu,
                }
                for event in raw["events"]
            )
            copied = sum(row["stats"]["bytes"] for row in raw["copies"])
            leftover = (
                max(torch.cuda.memory_allocated(self.gpu) - request.start, 0)
                if self.cuda
                else 0
            )
            request.stats = RequestStats(
                id,
                tuple(request.steps),
                tuple(
                    Eviction(
                        str(event["model"] or ""),
                        self.gpu,
                        {
                            "free": 0,
                            "orphan": 1,
                            "speculative": 2,
                            "unused": 3,
                            "declared": 4,
                            "working": 5,
                            "pinned": 6,
                        }[str(event["class"])],
                        str(event["reason"]),
                        int(event["bytes"]),
                    )
                    for event in trace
                    if event["kind"] == "evict"
                ),
                ((self.gpu, leftover),) if self.cuda else (),
                copied,
                self.memo.retries,
                (),
                request.declared,
                tuple(_copy(row) for row in copies()),
                trace,
            )

    def resident(self) -> dict[str, int]:
        if self.native is not None:
            return dict(self.native.resident())
        with self.lock:
            return {name: model.nbytes for name, model in self.models.items()}

    def free(self, device: torch.device, need: int) -> bool:
        if self.native is None or device != self.device:
            return False
        return (
            self.native.relief(self._current().native_id, need, False)[0] == "complete"
        )

    def lower(self, device: torch.device, need: int) -> bool:
        if self.native is None or device != self.device:
            return False
        return (
            self.native.relief(self._current().native_id, need, True)[0] == "complete"
        )

    def available(self, device: torch.device | None = None) -> int:
        return (
            self.native.available()
            if self.native is not None and device in (None, self.device)
            else 0
        )

    def exhausted(self, device: torch.device, need: int, op: str) -> Exhausted:
        peaks = tuple(
            Peak(
                s.model,
                s.shape,
                s.peak,
                tuple(f"{key} {value}" for key, value in s.cuts.items()),
            )
            for s in self.memo.shapes.values()
        )
        return Exhausted(self.gpu, need, op, self.resident(), peaks)

    def close(self) -> None:
        with ExitStack() as closing:
            if self.native is not None:
                closing.callback(self.native.close)
            for model in list(self.models.values()):
                closing.callback(model.close)
            for cache in list(self.caches):
                closing.callback(cache.close)


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
    """Its weight set's regions, what `load` maps and a sealed layout holds: its tensors in
    header order, cut wherever the block that owns them changes, so one region is one block's
    run or a run of common. A chunk two blocks share is read once per region (the layout
    reads parts), never a reason to join the blocks: packs chained a DiT's 48 blocks into one
    18 GiB region."""
    grant = _declared(module).source
    if grant is None:
        raise Unbound(
            f"{type(module).__name__}: no weights attached (`Declared.source`)"
        )
    keys = None if grant.keys is None else set(grant.keys)
    prefix, paths = f"{grant.component}/", blocks_of(module)
    regions: dict[str | None, list[str]] = {}
    for key in grant.ck.components[grant.component].tensors:
        if keys is None or key in keys:
            regions.setdefault(_block_of(key, paths), []).append(prefix + key)
    return list(regions.values())


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
        model.sizes[part.region] = max(
            model.sizes.get(part.region, 0), part.offset + part.nbytes
        )
    model.blocks = sorted(
        (b for b in by_path.values() if b.regions), key=lambda b: b.regions[0]
    )
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
    """Every meta weight becomes a placeholder on the execution device, owned by the block
    whose regions hold its bytes (a small block's tensors can sit in common), else by its
    module's block."""
    by_path = {b.path: b for b in model.blocks}
    owner = {key: b for b in (model.common, *model.blocks) for key in b.tensors}
    done: set[int] = set()
    for name, group in model.sites.items():
        meta = group[0].table[group[0].attr]
        if meta is None or id(meta) in done or meta.device.type != "meta":
            continue
        done.add(id(meta))
        path = _block_of(name, list(by_path))
        block = owner.get(name) or (model.common if path is None else by_path[path])
        held: Tensor = Placeholder(
            meta.shape, meta.dtype, model.device, block, group[0]
        )
        if group[0].parameter:
            held = nn.Parameter(held, requires_grad=False)
        for site in group:
            site.table[site.attr] = held


def _fly(model: Model) -> None:
    """A forward of the root or a direct child (`vae.decode` calls `decoder`) is the model in
    use, counted so concurrent forwards share it (cr-202)."""
    memory = model.memory
    for module in (model.module, *model.module.children()):
        model.handles.append(
            module.register_forward_pre_hook(lambda *_: memory._start(model))
        )
        hook = module.register_forward_hook(
            lambda *_: memory._land(model), always_call=True
        )
        model.handles.append(hook)


def _hook(block: Block) -> None:
    module, memory, site = block.module, block.model.memory, block.site
    assert module is not None and site is not None
    block.own_forward = "forward" in module.__dict__
    forward = block.forward = module.forward

    def bound(*args: object, **kwargs: object) -> object:
        loose = (
            not block.model.aloft
        )  # run outside its model's forward: common binds with it
        if loose:
            memory._enter(block.model.common, record=False)
        try:
            memory._enter(block, call=(args, kwargs))
            try:
                return cuts.run(site, forward, args, kwargs)
            finally:
                memory._exit(block)
        finally:
            if loose:
                memory._exit(block.model.common)

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
            raise Unbound(
                f"{model.name}: {key} is encoded ({sorted(parts)}) and no bind was given"
            )
        for name, value in made.items():
            for site in model.sites.get(name, ()):
                held = site.table[site.attr]
                block.written.append((site, held))
                if held is not None and held.dtype != value.dtype:
                    value = value.to(
                        held.dtype
                    )  # a converted module (`module.to(dtype)`)
                site.table[site.attr] = (
                    nn.Parameter(value, False) if site.parameter else value
                )


def _unbind(block: Block) -> None:
    """Placeholders back; the views go, and with them the pins (released on the bind stream)."""
    for site, held in reversed(block.written):
        site.table[site.attr] = held
    block.written.clear()
    block.pins.clear()
    block.stream = None


def _clear_frames(exc: BaseException | None) -> None:
    """Drop a failed request's frame locals: they hold its tensors (§5.7)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
        exc = exc.__cause__ or exc.__context__
