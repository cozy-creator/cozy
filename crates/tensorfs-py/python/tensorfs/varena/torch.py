"""varena's memory for torch modules: conversion, with no policy (design varena-slim.md §2).

    mem = Memory.open(0)  # before torch initializes CUDA
    vars(unet)["_varena"] = Declared(Weights(ck, "unet"))
    mem.load(unet, "gpu")
    with mem.request("r1", plan=[te, unet, vae]):
        image = pipe(prompt)

A block's params are views of its pin only while the block runs; placeholders (shape, dtype and
device, no bytes) stand in otherwise, so no tensor outlives its page. On CUDA varena is torch's
only allocator: a request's torch blocks come from its arena, made room for in Rust. A model's
top-level forward is a pass: the plan's cursor, and its arena's peak.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import threading
import warnings
import weakref
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeAlias

import tensorfs
import torch
from torch import Tensor, nn
from torch.utils._pytree import tree_leaves, tree_map

from . import Ended, Exhausted, Fill, GpuId, Level, Pass, Pin, Stats, VarenaError, Window, host_limit
from . import Memory as Native
from .. import _ext
from . import Request as NativeRequest
from . import Set

__all__ = [
    "Bind", "Declared", "Exhausted", "Filling", "GpuId", "Hold", "Level", "Memory", "Model",
    "NotRegistered", "GraphRegion", "Placeholder", "Region", "Request", "Room", "Stats", "Unbound", "VarenaError", "Weights", "blocks_of",
]

#: Blocks entered, process-wide: the work a call does in its models, which the Runtime tells the
#: machine (cl-648).
progress = 0
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


@dataclass(frozen=True, slots=True)
class Room:
    """A cut's room (design varena-leases §8): `free` is this process's float plus the manager's
    free bytes; `cold` revocable bytes outside the step's sets; `warm` this process's inside
    them; every need rounds up to `granularity`; `floor` is what the step's sets need resident."""

    free: int
    cold: int
    warm: int
    granularity: int
    floor: int


class Unbound(Exception):
    """An op on a weight no pin is bound to, or a forward outside a request."""


class NotRegistered(VarenaError):
    """TensorD's manager refused this GPU's lease: this process is no executor it registered."""


@dataclass(frozen=True, slots=True)
class Weights:
    """A module's stored weights: a component of a TensorFS checkpoint. `release` names the
    package release (the fill code's identity), `shard` the rank's slice, `keys` the stored
    tensors the module builds (all by default), `bound` the module path a stored key binds to
    where the two differ (a checkpoint stored under another naming): blocks are module paths."""

    ck: tensorfs.Checkpoint
    component: str
    release: str = ""
    shard: str = "0/1"
    keys: Collection[str] | None = None
    bound: Mapping[str, str] | None = None


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
    """A weight while no pin is bound to it: shape, dtype and device, no bytes. A dtype or
    device change is another placeholder and detach is itself; any other op reads it outside its
    block's forward, which binds the block on read (`_varena_read`): the op runs on the bound
    tensor, which (with what it views) owns the pin until it dies."""

    @staticmethod
    def __new__(cls, shape, dtype, device):  # type: ignore[no-untyped-def]
        return Tensor._make_wrapper_subclass(cls, tuple(shape), dtype=dtype, device=device, storage_size=0)  # type: ignore[attr-defined]

    def __repr__(self) -> str:  # type: ignore[override]
        return f"Placeholder({tuple(self.shape)}, {self.dtype}, {self.device})"

    def _varena_read(self) -> Tensor:
        """Its block bound now, for a read no op sees too (a compiled extension's or Triton's
        launch reads it by pointer, which is 0 here): the bound tensor, which with its views
        holds the pin until it dies. A loaded model's placeholder carries its own; this one
        belongs to none."""
        raise Unbound("a weight of no loaded model, with no pin bound")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        aten, kwargs = torch.ops.aten, kwargs or {}
        if func in (aten.detach.default, aten.alias.default):
            return args[0]
        if func is aten._to_copy.default:
            held = args[0]
            dtype, device = kwargs.get("dtype") or held.dtype, kwargs.get("device") or held.device
            made = cls(held.shape, dtype, device)
            if "_varena_read" in vars(held):  # a loaded model's: the copy reads converted
                read = held._varena_read
                made._varena_read = lambda: read().to(device=device, dtype=dtype)
            return made

        def bound(a: object) -> object:
            return a._varena_read() if isinstance(a, Placeholder) else a

        args, kwargs = tree_map(bound, (args, kwargs))
        return func(*args, **kwargs)


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
    #: the raw CUstream it was bound on (None off CUDA)
    stream: int | None = None
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
    """An open pass: the plan's cursor at its model, keyed by its entry and tensor dims."""

    native: Pass
    request: Request
    #: `entry|dims`: the module its forward entered ("" the root) and its tensor arguments' dims
    key: str
    token: object = None
    #: this thread's allocating into the request's arena, when the pass runs off the request's thread
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
        #: the fill identity its pages were last declared under
        self.identity: str | None = declared.fill.identity if declared.fill is not None else None

    @contextmanager
    def passing(self, key: str = "") -> Iterator[Model]:
        """A pass of this model keyed `key` (a component-scoped call's method), opened before
        its forwards: the root's and children's hooks inside it (graph capture included) only
        nest in it."""
        self.memory._open(self, lambda: key)
        try:
            yield self
        finally:
            self.memory._close(self)

    def retain(self, paths: Collection[str] = ("",)) -> Binding:
        """Retain addresses for every weight/buffer in the declared module subtrees.

        An empty path declares the whole model. Runtime declares indirect dependencies too;
        observing the modules visited by one capture is not a complete declaration.
        """
        all_blocks = [b for b in (self.common, *self.blocks.values()) if b is not None]
        blocks = [b for b in all_blocks if any(not p or b.path == p or b.path.startswith(p + ".") or p.startswith(b.path + ".") for p in paths)]
        # Common weights can be read by any child block, so include them conservatively.
        if self.common is not None and self.common not in blocks:
            blocks.insert(0, self.common)
        return Binding(self, sorted(blocks, key=lambda b: b.region))

    def hold(self, shed: Callable[[], None] | None = None, *, free_only: bool = False) -> Hold | None:
        """Every block bound, pins and param views, until the hold's release: its forwards then
        bind nothing. A pass that changes the fill identity sheds the hold first and calls `shed`
        (it drops what was built on the old bake). `free_only`: bound into free bytes only,
        evicting nothing, all or none (None: the model does not fit beside what is resident; its
        blocks bind as they run). Held pages are pins: nothing evicts them, inside torch's
        allocator included, so a hold ends before another component needs the room."""
        blocks = [b for b in (self.common, *self.blocks.values()) if b is not None]
        held: list[_Block] = []
        try:
            for b in blocks:
                self.memory._enter(b, evict=not free_only)
                held.append(b)
        except Exhausted:
            for b in reversed(held):
                self.memory._exit(b)
            if free_only:
                return None
            raise
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
        self.last: torch.cuda.Event | None = None
        if model.memory.cuda:
            self.last = torch.cuda.Event()
            self.last.record(torch.cuda.current_stream(model.memory.device))

    def valid(self) -> bool:
        with self.lock:
            if self.closed:
                return False
            self.model.memory._declare(self.model)
            return self.native.valid()

    def resident(self) -> bool:
        """Every block mapped and filled now (an idle capture runs only into what is resident):
        a read, no pin, no fill."""
        with self.lock:
            return not self.closed and self.native.resident()

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
            with ExitStack() as scope:
                scope.enter_context(torch.cuda.stream(stream) if memory.cuda else nullcontext())
                if self.last is not None:
                    stream.wait_event(self.last)
                memory._declare(self.model)
                request = _request.get()
                pins = self.native.prepare(stream.cuda_stream if stream is not None else 0, request.native if request is not None else None)
                held: list[_Block] = []
                try:
                    with self.model.lock:
                        for block, pin in zip(self.blocks, pins, strict=True):
                            raw = stream.cuda_stream if stream is not None else None
                            if block.count == 0:
                                memory._bind(block, raw, pin)
                            elif raw is not None and block.stream != raw:
                                memory._join(raw, block.stream)  # type: ignore[arg-type]
                            block.count += 1
                            held.append(block)
                    yield self
                finally:
                    if self.last is not None:
                        try:
                            self.last.record(stream)
                        except BaseException:
                            self.closed = True
                            self.quarantined = pins
                            raise
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
#: the request whose arena this thread's torch blocks go to (`Request.on_thread`)
_arena: ContextVar[Request | None] = ContextVar("varena_arena", default=None)
#: CUDA devices whose Memory is torch's allocator: held for the process, whose torch blocks it serves
_serving: dict[int, Memory] = {}
#: varena's torch allocator shim and its build record (shim/build.sh, shim/build.ps1)
_SHIM = Path(_ext.__file__).parent / ("_varena_torch.dll" if os.name == "nt" else "_varena_torch.so")
#: the shim, once installed
_shim: ctypes.CDLL | None = None


def _install() -> None:
    """varena as torch's only CUDA allocator, once per process and before torch initializes CUDA:
    the shim, built against torch's C++ ABI, bound to the agent's C ABI in the extension."""
    if _serving:
        return
    try:
        built = json.loads(_SHIM.with_suffix(".json").read_text())
    except OSError as e:
        raise VarenaError(f"this tensorfs wheel has no torch allocator for varena ({_SHIM.name}): {e}") from e
    running = {"torch": ".".join(torch.__version__.split(".")[:2]), "cxx11_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI)}
    if built != running:
        raise VarenaError(f"varena's torch allocator was built for {built}, and this process runs {running}: install the tensorfs wheel for this torch")
    if torch.cuda.is_initialized():
        raise VarenaError("torch initialized CUDA before Memory.open: open the Memory first, as varena becomes torch's only CUDA allocator")
    global _shim
    try:
        shim, ext = ctypes.CDLL(str(_SHIM)), ctypes.CDLL(_ext.__file__)
        shim.varena_torch_abandon_pool.argtypes = (ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64)
        agent = ("alloc", "free", "record_stream", "pool_begin", "pool_end", "pool_release")
        shim.varena_torch_bind((ctypes.c_void_p * len(agent))(*(ctypes.cast(getattr(ext, f"varena_agent_{f}"), ctypes.c_void_p) for f in agent)))
        hook = {name: ctypes.cast(getattr(shim, f"varena_torch_{name}"), ctypes.c_void_p).value for name in ("malloc", "free", "record_stream", "begin_pool", "end_pool", "release_pool")}
        allocator = torch._C._cuda_customAllocator(hook["malloc"], hook["free"])
        allocator.set_record_stream_fn(hook["record_stream"])
        allocator.set_begin_allocate_to_pool(hook["begin_pool"])
        allocator.set_end_allocate_to_pool_fn(hook["end_pool"])
        allocator.set_release_pool(hook["release_pool"])
        torch._C._cuda_changeCurrentAllocator(allocator)
    except (OSError, AttributeError, RuntimeError) as e:  # a shim that won't load, a torch without the pluggable allocator
        raise VarenaError(f"varena could not become torch's allocator ({_SHIM.name}): {e}") from e
    _shim = shim


def _lease(native: Native, ns: int) -> str | None:
    """The GPU's weight pages leased from the manager TensorD serves in namespace `ns`
    (varena-leases §11), or why not. TensorD holds a follower's dial until it registers it, so
    `NotRegistered` is a failure, never a fallback. Only no compatible manager runs standalone,
    warned."""
    refused = native.lease(ns)
    if refused is None:
        return None
    code, why = refused
    if code in ("Incompatible", "Absent"):
        warnings.warn(f"varena: GPU {native.gpu.uuid} runs standalone, with no lease from TensorD: {why}", stacklevel=3)
        return why
    raise (NotRegistered if code == "NotRegistered" else VarenaError)(f"varena: TensorD refused GPU {native.gpu.uuid}'s lease ({code}): {why}")


class Memory:
    """This process's memory on one device: its models, regions and requests."""

    def __init__(self, native: Native, device: torch.device) -> None:
        self.native, self.device = native, device
        self.cuda = device.type == "cuda"
        #: why this GPU runs standalone though TensorD asked it to lease (`Memory.open(leases=)`)
        self.standalone: str | None = None
        self._graph_pool: tuple[int, int] | None = None
        self.models: dict[nn.Module, Model] = {}
        self.requests: dict[Request, None] = {}
        #: (set, block) -> bytes kept bound for tensors that outlived their block's run
        self.escaped: dict[tuple[str, str], int] = {}
        #: weights read outside their block's forward, bound on read: (reads, bytes) per block
        self.off_forward: dict[tuple[str, str], tuple[int, int]] = {}
        #: Cross-stream consumers joined back into the binding stream.
        self.crossed = 0

    @classmethod
    def open(cls, gpu: int | str | torch.device | GpuId = 0, *, leases: int | None = None) -> Memory:
        """The device's memory. On CUDA, with `leases` (TensorD's namespace, from its Start), its
        pages and torch's are leased from TensorD's manager (`_lease`); without, or with no
        compatible manager, it runs standalone on the device's free bytes at open less the
        backend's reserve. Then varena becomes torch's only allocator (`_install`), so open it
        before any torch CUDA call. CPU and MPS use host pages of the machine's RAM; a `GpuId`
        names a host table and its size. A process has one Memory per device: opening the device
        again returns it, and a CUDA device's lives as long as the process (torch's blocks are
        its)."""
        device = torch.device("cpu") if isinstance(gpu, GpuId) else torch.device("cuda", gpu) if isinstance(gpu, int) else torch.device(gpu)
        if device.type == "cuda":
            device = torch.device("cuda", device.index or 0)
        key = gpu.uuid if isinstance(gpu, GpuId) else str(device)
        with _opening:
            memory = _memories.get(key)
            if memory is None:  # one Memory per device and process: a second open is the first
                if isinstance(gpu, GpuId):
                    native = Native.host(gpu)
                elif device.type == "cuda":
                    native = Native.cuda(device.index or 0)
                else:
                    ram = host_limit()  # past it the kernel kills: a cgroup's limit, not physical memory
                    if ram is None:
                        raise VarenaError("this process's memory limit is unreadable: the host table has no size")
                    native = Native.host(GpuId("host", device.type, ram))
                memory = cls(native, device)
                if device.type == "cuda":
                    if leases is not None:
                        memory.standalone = _lease(native, leases)  # leased: torch is served from its grants
                    elif os.name == "nt":
                        memory.standalone = "TensorD leases Linux GPUs only so far"
                        warnings.warn(f"varena: GPU {native.gpu.uuid} runs standalone on Windows, within its free bytes less the reserve: {memory.standalone}", stacklevel=2)
                    if leases is None or memory.standalone is not None:
                        native.serve_standalone()
                    _install()
                    _serving[device.index or 0] = memory
                _memories[key] = memory
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
                path = w.bound.get(key, key) if w.bound else key
                owned.setdefault(_block_of(path, paths), []).append(f"{w.component}/{key}")
        order = [p for p in ("", *paths) if p in owned]
        # A set is one checkpoint's component as one module builds it: two models of a release that share a
        # component name (two DiTs' `transformer`, two LoRAs' `model`) are two sets, and so are two modules
        # building other keys of one checkpoint (a text encoder without the head beside the whole LM).
        built = "" if w.keys is None else "#" + hashlib.sha256("\n".join(sorted(w.keys)).encode()).hexdigest()[:12]
        name = f"{w.release}/{w.ck.id}/{w.component}{built}" if w.release else f"{w.ck.id}/{w.component}{built}"
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
    def request(self, id: str, plan: Sequence[object] = (), *, idle: bool = False) -> Iterator[Request]:
        """Request `id` and its plan: its steps in order, each a model (`Model`, module or set
        name) or a collection of models that run together. torch's blocks on this thread go to
        its arena (`Request.on_thread` for others). An `idle` one (a capture or warm-up beside
        others' work) takes free bytes only."""
        r = Request(self, self.native.request(id, self._plan(plan), idle))
        self.requests[r] = None
        token = _request.set(r)
        try:
            with r.on_thread():
                yield r
        except BaseException as failed:
            try:
                self._end(r)
            except BaseException as ending:
                failed.add_note(f"ending request {id!r} also failed: {ending!r}")  # the request's own error stays the one raised
            raise
        else:
            self._end(r)
        finally:
            _request.reset(token)

    def _end(self, r: Request) -> None:
        self.requests.pop(r, None)
        r.ended = r.native.end()

    def prepare_pool(self, pool: tuple[int, int], expected: int) -> None:
        """Before a capture into torch pool `pool` (a `graph_pool_handle()`) begins: `expected`
        bytes mapped into it now, making room here, so the capture never needs to (a capturing
        thread is served from the pool's mapped bytes or its float, or refused)."""
        self.native.prepare_pool(pool, expected)

    def torch_stats(self) -> dict[str, object]:
        """varena's figures for torch's blocks here (torch's own memory statistics don't see a
        pluggable allocator): allocated, mapped and peak bytes, fragmentation (mapped less live);
        per open request (allocated, peak); per capture pool (mapped, allocated)."""
        allocated, mapped, peak, fragmentation, requests, pools = self.native.torch_stats()
        return {
            "allocated": allocated, "mapped": mapped, "peak": peak, "fragmentation": fragmentation,
            "requests": {r: (a, p) for r, a, p in requests}, "pools": {pool: (m, a) for pool, m, a in pools},
        }

    def fence_pool(self, pool: tuple[int, int], stream: torch.cuda.Stream) -> None:
        """After a replay from `pool` and its outputs' readers on `stream`: the pool's chunks
        leave, once torch released it, only after this point on `stream`."""
        self.native.fence_pool(pool, stream.cuda_stream)

    def capture_failed(self, pool: tuple[int, int]) -> None:
        """`capture_begin` into `pool` threw (torch then never ends or releases it): its route
        in torch's allocator goes and varena ends and releases it. A no-op for a pool with no
        open capture, so call it whenever capture_begin raises."""
        if self.cuda and _shim is not None:
            _shim.varena_torch_abandon_pool(self.device.index or 0, *pool)

    def region(self, owner: str, kind: str, sets: nn.Module | Model | Sequence[nn.Module | Model] = (), spill: bool = True) -> Region:
        """A long-lived intermediate's region: `kind` is kv, latent, conditioning, derived or
        workspace; `sets` the models it serves; `spill` False drops it for recompute on revoke."""
        return Region(self.native.region(owner, kind, self._set_ids(sets), spill), owner, kind)

    @property
    def graph_pool(self) -> tuple[int, int]:
        """The device's one CUDA graph capture pool token, shared by its graph regions."""
        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        return self._graph_pool

    def graph_region(self, sets: nn.Module | Model | Sequence[nn.Module | Model] = (), retained: nn.Module | Model | Sequence[nn.Module | Model] = ()) -> GraphRegion:
        """One component's graphs on this device: `sets` the models they read, `retained` those
        held bound across replays."""
        return GraphRegion(self, self.native.graph_region(self.graph_pool, self._set_ids(sets), self._set_ids(retained)))

    def _set_ids(self, models: nn.Module | Model | Sequence[nn.Module | Model]) -> list[int]:
        many = models if isinstance(models, Sequence) else [models]
        return [(m if isinstance(m, Model) else self.models[m]).id for m in many]

    def classes(self) -> dict[str, int]:
        """Mapped bytes by class: F (fixed), O (other's), S (speculative), N (no declared use),
        D (declared later use), W (working), P (pinned)."""
        return dict(zip("FOSNDWP", self.native.classes(), strict=True))

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

    def _declare(self, model: Model) -> None:
        """The model's fill, declared to varena when its identity changed (a LoRA set swapped):
        holds of the old bake go first, as their pins would keep its pages."""
        identity = model.fill.identity if model.fill is not None else None
        if identity != model.identity:
            for hold in list(model.holds):
                hold.shed()
            model.identity = identity
            self.native.fill(model.id, _fill(model.fill))

    def _current(self) -> Request:
        r = self._open_request()
        if r is None:
            raise Unbound("a loaded model's forward runs inside `mem.request(...)`")
        return r

    def _open_request(self) -> Request | None:
        """This thread's request, else the only one open."""
        r = _request.get()
        if r is None and len(self.requests) == 1:
            r = next(iter(self.requests))
        return r

    def _for(self) -> NativeRequest | None:
        """The native request a bind's fill and trace go to: a hold binds before its pass opens."""
        r = self._open_request()
        return r.native if r is not None and r.ended is None else None

    # -- passes

    def _open(self, model: Model, key: Callable[[], str]) -> None:
        """The outermost pass of `model` on this thread opens, keyed `key()`; inner ones nest."""
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
            self._declare(model)
            k = key()
            p = _Pass(r.native.pass_(model.id, k), r, k)
            # Off the request's thread its torch blocks go to the request's arena for the pass.
            if self.cuda and _arena.get() is not r:
                p.pooled = r.on_thread()
                p.pooled.__enter__()
            if model.common is not None:
                self._enter(model.common)
            local.passing, p.token = p, _pass.set(p)
        except BaseException:
            local.depth -= 1
            if p is not None and p.pooled is not None:
                p.pooled.__exit__(None, None, None)
            raise

    def _close(self, model: Model) -> None:
        """A pass ends, failed or not. On CUDA varena reads its peak from the request's arena; MPS
        has none, so its allocated bytes then stand in for it."""
        local = model.local
        local.depth -= 1
        p = getattr(local, "passing", None)
        if local.depth or p is None:
            return
        local.passing = None
        _pass.reset(p.token)  # type: ignore[arg-type]
        try:
            if model.common is not None:
                self._exit(model.common)
        finally:
            if p.pooled is not None:
                p.pooled.__exit__(None, None, None)
            p.native.end(torch.mps.current_allocated_memory() if self.device.type == "mps" else 0)


    # -- blocks

    def _stream(self) -> int | None:
        """The current raw CUstream (None off CUDA): an int, not a Stream object, per block."""
        return torch._C._cuda_getCurrentRawStream(self.device.index or 0) if self.cuda else None

    def _join(self, waiter: int, waited: int) -> None:
        """`waiter` waits for the work queued on `waited` so far (two raw CUstreams)."""
        self.crossed += 1
        torch.cuda.ExternalStream(waiter, device=self.device).wait_stream(torch.cuda.ExternalStream(waited, device=self.device))

    def _enter(self, block: _Block, evict: bool = True) -> None:
        global progress
        stream = self._stream()
        with block.model.lock:
            if block.count == 0:
                if self.cuda and torch.cuda.is_current_stream_capturing():
                    raise Unbound(f"block {block.path!r} was not prepared before capture")
                self._bind(block, stream, evict=evict)
            elif stream is not None and stream != block.stream and not torch.cuda.is_current_stream_capturing():
                self._join(stream, block.stream)  # type: ignore[arg-type]  # a capture binds under a hold
            block.count += 1
        progress += 1

    def _exit(self, block: _Block) -> None:
        stream = self._stream()
        with block.model.lock:
            if stream is not None and block.stream not in (None, stream) and not torch.cuda.is_current_stream_capturing():
                self._join(block.stream, stream)  # type: ignore[arg-type]
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

    @torch.inference_mode(False)  # what it returns is ordinary: a caller's view aliases it (se-074)
    def _read(self, block: _Block, site: _Site) -> Tensor:
        """A weight read outside its block's forward (a package's table, an overlay's factor):
        its block bound now, on the current stream; the tensor returned, and any view of it,
        owns the pin, so the page stays bound while it lives (the escaped-view rule). Counted as
        an off-forward read."""
        model = block.model
        pin = self.native.bind(model.id, block.region, self._stream() or 0, request=self._for())
        if self.device.type in ("cpu", "cuda"):
            window = pin.window()
            window.keep(pin)
            views = _views(block, torch.from_dlpack(window))
            if pin.fresh and model.fill is not None:
                model.fill(block.path, views)  # in place: the views alias the page
                pin.filled()
        else:  # MPS: the host bytes are baked, then copied in
            base = torch.from_dlpack(pin)
            if pin.fresh and model.fill is not None:
                model.fill(block.path, _views(block, base))
                pin.filled()
            views = _views(block, base.to(self.device))
        value = next((v for s, v in _assign(block, views)[0] if s.table is site.table and s.attr == site.attr), None)
        if value is None:
            raise Unbound(f"{model.name}: {site.attr} is not a tensor of block {block.path or 'common'!r}")
        key = (model.name, block.path)
        r = _request.get()
        for counts in (self.off_forward, *((r.off_forward,) if r is not None else ())):
            reads, nbytes = counts.get(key, (0, 0))
            counts[key] = (reads + 1, nbytes + pin.len)
        return value

    # Bound under a caller's inference_mode, the views (kept across binds) and the module tensors
    # made of them would be inference tensors, and a view of a placeholder outside its forward
    # could not alias one (se-074, Klein's VAE batch norm statistics): ordinary, zero-copy.
    @torch.inference_mode(False)
    def _bind(self, block: _Block, stream: int | None, pin: Pin | None = None, evict: bool = True) -> None:
        model = block.model
        if pin is None:
            pin = self.native.bind(model.id, block.region, stream or 0, evict, self._for())
        if self.device.type in ("cpu", "cuda"):
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
    """A request in flight (`Memory.request`)."""

    def __init__(self, memory: Memory, native: NativeRequest) -> None:
        self.memory, self.native = memory, native
        self.ended: Stats | None = None
        #: (set, block) -> bytes kept bound in it for weight tensors that outlived their run
        self.escaped: dict[tuple[str, str], int] = {}
        #: (set, block) -> (reads, bytes) of weights read outside their block's forward
        self.off_forward: dict[tuple[str, str], tuple[int, int]] = {}

    def plan(self, steps: Sequence[object]) -> None:
        """Replaces the plan (as `Memory.request` takes it); its cursor starts over."""
        self.native.plan(self.memory._plan(steps))

    @contextmanager
    def yielding(self) -> Iterator[None]:
        """Around a wait on a child call (a call this request made and awaits): its step's pages
        are a later use, taken before the child's own."""
        yielded = self.native.yielding()
        try:
            yield
        finally:
            yielded.end()

    def reset_peak(self) -> int:
        """Its torch arena's high-water since the last call or the pass's start, restarting
        here: a section's peak (a split's window). The pass's own peak still covers it."""
        return self.native.reset_peak()

    def room(self, step: int | None = None) -> Room:
        """What a cut sizes from at plan step `step` (the cursor's by default). Lock-free."""
        return Room(*self.memory.native.room(step))

    @contextmanager
    def on_thread(self) -> Iterator[Request]:
        """This thread's torch blocks go to this request's arena until it leaves: a Runtime
        worker doing the request's work (R9). On this thread only."""
        memory = self.memory
        if not memory.cuda or _arena.get() is self:
            yield self
            return
        guard = self.native.allocating(memory.native)
        token = _arena.set(self)
        try:
            yield self
        finally:
            _arena.reset(token)
            guard.end()

    def stats(self) -> Stats:
        return self.ended if self.ended is not None else self.native.stats()


class Region:
    """A long-lived intermediate (KV, latents, conditioning, a derived weight, a workspace), in
    an arena of its own (design varena-leases §6). Revoked, it is spilled or dropped; its next
    `pin()` then says fresh, and the owner recomputes it. There are no shrink callbacks."""

    def __init__(self, native: object, owner: str, kind: str) -> None:
        self.native, self.owner, self.kind = native, owner, kind

    @contextmanager
    def pin(self) -> Iterator[bool]:
        """Holds it mapped while inside, mapping it again first if it was reclaimed; yields True
        when it came back empty (recompute it)."""
        pin = self.native.pin()
        try:
            yield pin.fresh
        finally:
            pin.end()

    @contextmanager
    def allocating(self) -> Iterator[bool]:
        """As `pin`, and this thread's torch blocks are the region's while inside."""
        pin = self.native.allocating()
        try:
            yield pin.fresh
        finally:
            pin.end()

    def produced(self, ns: int) -> None:
        """What producing it took, in nanoseconds: the spill-versus-drop choice."""
        self.native.produced(ns)

    def next_use(self, step: int | None) -> None:
        """The plan step of its next use; None: none declared."""
        self.native.next_use(step)

    def close(self) -> Ended:
        """Its arena ends; its record (allocated and peak bytes, its blocks still live)."""
        return self.native.close()


class GraphRegion:
    """A component's CUDA graphs on one device (design varena-leases §5), captured into the
    device's one pool (`Memory.graph_pool`); each replay runs inside `replaying(stream)`."""

    def __init__(self, memory: Memory, native: object) -> None:
        self.memory, self.native = memory, native
        self.pool: tuple[int, int] = native.pool

    def prepare(self, expected: int | None) -> None:
        """Before `capture_begin`: maps `expected` bytes into the pool, making room now (a
        shape's first capture: None)."""
        self.native.prepare(expected)

    def captured(self, ns: int) -> None:
        """What capturing its graphs took, in nanoseconds."""
        self.native.captured(ns)

    @contextmanager
    def replaying(self, stream: torch.cuda.Stream) -> Iterator[bool]:
        """Around a replay and its outputs' readers on `stream`: yields False while the region is
        being dropped (run eager instead, recapture off-path); a True one is released at the
        exit, by an event on `stream`, even when the replay raises."""
        if not self.native.acquire():
            yield False
            return
        try:
            yield True
        finally:
            self.native.release(stream.cuda_stream)
            self.memory.fence_pool(self.pool, stream)

    @property
    def dropping(self) -> bool:
        return self.native.dropping

    def close(self) -> None:
        """Drops its graphs' pins and its use of the pool."""
        self.native.close()


# ------------------------------------------------------------------------------ helpers


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
    """Every meta tensor becomes a placeholder on the execution device, read on demand."""
    done: set[int] = set()
    paths = list(model.blocks)
    for name, group in model.sites.items():
        meta = group[0].table[group[0].attr]
        if meta is None or id(meta) in done or meta.device.type != "meta":
            continue
        done.add(id(meta))
        held: Tensor = Placeholder(meta.shape, meta.dtype, model.memory.device)
        if group[0].parameter:
            held = nn.Parameter(held, requires_grad=False)
        block = model.blocks.get(_block_of(name, paths)) or model.common
        held._varena_read = lambda block=block, site=group[0]: model.memory._read(block, site)  # type: ignore[attr-defined]
        for site in group:
            site.table[site.attr] = held


def _hook(model: Model) -> None:
    """The root's or a direct child's outermost forward is a pass keyed `entry|dims` (its hooks
    run first) unless one is open (`Model.passing`); a block binds while it runs."""
    memory = model.memory
    for entry, sub in (("", model.module), *model.module.named_children()):
        def opened(args: tuple[object, ...], kwargs: dict[str, object], entry: str = entry) -> None:
            memory._open(model, lambda: f"{entry}|" + "x".join(str(d) for t in tree_leaves((args, kwargs)) if isinstance(t, Tensor) for d in t.shape))
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
