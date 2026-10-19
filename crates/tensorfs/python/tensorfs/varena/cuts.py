"""The out-of-memory catch and exact cuts for torch modules in varena's memory (memory-varena.md
§5.2-§5.3).

An op that runs out of device memory runs again, each time in a strictly smaller configuration,
so the catch ends without timers: idle bytes and cold weights first (rungs 1-2), then the
innermost cut around it finer, then its own bands, then the hot watermark lower (rung 4), else
`Exhausted`. Freed bytes are never progress. Every cut is exact and keeps its result on the
device; the size that fit is remembered per shape key (the memo) and applied up front after.

- A block whose first run wrote no tensor that existed before it runs bare, without the
  dispatch layer; out of memory, it gets its generators and declared state back and runs again
  under the layer. Other blocks, and ops outside blocks within a `use`, run under the layer,
  caught at the op.
- Bands: convolution, constant padding, nearest upsampling and group norm (two-pass statistics),
  in rows with their halos.
- Calls cut themselves: FFN token chunks (`install`), attention by head groups then query rows
  (`attend`, from the Runtime's attention dispatch), and a package's `split` loop (`split`).
"""

from __future__ import annotations

import functools
import re
import threading
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import Tensor, nn
from torch._ops import OpOverload, OpOverloadPacket
from torch.utils._python_dispatch import (
    TorchDispatchMode,
    _disable_current_modes,
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)

aten = torch.ops.aten

#: Free bytes a retry asks beyond the failed allocation: the allocator's rounding, a workspace.
SLACK = 64 << 20
_ALLOCATE = re.compile(r"Tried to allocate ([\d.]+) (B|KiB|MiB|GiB|TiB)")
_UNITS = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40}
_ALLOC_FAILED = (
    "CUDA error: out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDNN_STATUS_ALLOC_FAILED",
    "CUFFT_ALLOC_FAILED",
    "CUSOLVER_STATUS_ALLOC_FAILED",
    "CUSPARSE_STATUS_ALLOC_FAILED",
)
#: diffusers' token-wise feed-forwards and the activation projections inside them
_FFN = frozenset({"FeedForward", "GELU", "GEGLU", "SwiGLU", "ApproximateGELU", "LinearActivation"})


class Room(Protocol):
    """The memory's side of the catch (`torch.Memory`)."""

    def make(self, device: torch.device, need: int) -> bool:
        """Rungs 1-2: idle bytes, then cold weights, until `need` bytes are free."""
        ...

    def lower(self, device: torch.device, need: int) -> bool:
        """Rung 4: the hot watermarks fall until `need` bytes are free; False when none can."""
        ...

    def available(self, device: torch.device) -> int:
        """Bytes one allocation may take now (0: unknown)."""
        ...

    def exhausted(self, device: torch.device, need: int, op: str) -> Exception: ...


class _NoRoom:
    """No memory serves this process: cuts still cut, nothing is freed."""

    def make(self, device: torch.device, need: int) -> bool:
        return False

    def lower(self, device: torch.device, need: int) -> bool:
        return False

    def available(self, device: torch.device) -> int:
        return 0

    def exhausted(self, device: torch.device, need: int, op: str) -> Exception:
        return torch.OutOfMemoryError(f"{op} needs {need} B more on {device}")


class Finer(Exception):
    """An op inside a cut still ran out of memory with rungs 1-2 done: the cut runs finer."""

    def __init__(self, scope: _Scope, need: int) -> None:
        super().__init__(f"cut finer: {need} B more")
        self.scope, self.need = scope, need


class _Whole(Exception):
    """A cut's pieces do not join (a call returned no tensor): it runs whole."""


# ------------------------------------------------------------------------------- the memo


@dataclass(slots=True)
class Shape:
    """A model's passes at one shape key: the latest measured activation peak and the cuts in
    force (name -> rows, tokens or query rows per piece)."""

    model: str
    shape: str
    peak: int = -1
    cuts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Facts:
    """A block as its first run showed it: bare (wrote nothing that existed before it, needed
    no band), and the generators its seeded ops drew from."""

    bare: bool
    generators: tuple[torch.Generator, ...]


class Memo:
    """Per process: what each block does, the cut size each shape key needed, and each model's
    peak per shape key."""

    def __init__(self) -> None:
        self.blocks: dict[tuple[str, int], Facts] = {}
        self.sizes: dict[tuple[object, ...], int] = {}
        #: ops with a band size in force for some shape: the layer's fast check
        self.banded: set[OpOverload] = set()
        self.whole: set[tuple[object, ...]] = set()
        self.shapes: dict[tuple[str, str], Shape] = {}
        self.retries = 0

    def shape(self, model: str, key: str) -> Shape:
        found = self.shapes.get((model, key))
        if found is None:
            found = self.shapes[(model, key)] = Shape(model, key)
        return found


@dataclass(eq=False, slots=True)
class Track:
    """A model's pass now: the memo entry its blocks' cuts are noted in."""

    model: str
    shape: Shape | None = None


@dataclass(eq=False, slots=True)
class Site:
    """One block for the catch: its memo key, device, model's track and declared state (attribute
    paths under `root` put back before a bare block runs again)."""

    key: tuple[str, int]
    device: torch.device
    track: Track
    root: nn.Module | None = None
    state: tuple[str, ...] = ()


@dataclass(slots=True)
class _Served:
    memo: Memo
    room: Room


_served = _Served(Memo(), _NoRoom())


def serve(memo: Memo, room: Room) -> None:
    """The process's memory: its memo and room serve every catch and cut from now on."""
    _served.memo, _served.room = memo, room


@dataclass(eq=False, slots=True)
class _Scope:
    finer: bool = False


class _Local(threading.local):
    def __init__(self) -> None:
        self.layer: Layer | None = None
        self.used: Layer | None = None
        self.depth = 0
        self.scopes: list[_Scope] = []
        self.site: Site | None = None


_local = _Local()


def shape_of(args: Sequence[object], kwargs: Mapping[str, object]) -> str:
    """A pass's shape key: its tensor arguments' dtypes and shapes."""
    return ", ".join(
        f"{str(t.dtype).removeprefix('torch.')}{list(t.shape)}" for t in _tensors((args, kwargs))
    )


def peaked(track: Track, nbytes: int) -> None:
    if track.shape is not None:
        track.shape.peak = nbytes


def _note(name: str, size: int) -> None:
    site = _local.site
    if site is not None and site.track.shape is not None:
        site.track.shape.cuts[name] = size


# ------------------------------------------------------------------------ out of memory


def short(exc: BaseException) -> int | None:
    """The bytes a device allocation failure asked for (0: not said); None for any other."""
    if isinstance(exc, torch.OutOfMemoryError):
        found = _ALLOCATE.search(str(exc))
        return int(float(found[1]) * _UNITS[found[2]]) if found else 0
    if isinstance(exc, torch.AcceleratorError):
        return 0 if getattr(exc, "error_code", None) == 2 or "out of memory" in str(exc) else None
    if isinstance(exc, RuntimeError) and any(t in str(exc) for t in _ALLOC_FAILED):
        return 0
    return None


def _refused(exc: BaseException) -> bool:
    """A cuBLAS or cuDNN status, which may be a library's own allocation failing: it runs again
    once with room made, and a second is the library's own."""
    text = str(exc)
    return isinstance(exc, RuntimeError) and ("CUBLAS_STATUS_" in text or "CUDNN_STATUS_" in text)


def _clear(exc: BaseException) -> None:
    """The failed attempt's frames go, and with them its tensors."""
    seen: set[int] = set()
    at: BaseException | None = exc
    while at is not None and id(at) not in seen:
        seen.add(id(at))
        if at.__traceback__ is not None:
            traceback.clear_frames(at.__traceback__)
        at = at.__cause__ or at.__context__


def _attempt[T](call: Callable[[], T]) -> tuple[bool, T | None, int]:
    """`call()`, or (False, None, need) when the device ran out of memory."""
    try:
        return True, call(), 0
    except Exception as exc:
        need = short(exc)
        if need is None or _capturing():
            raise
        _clear(exc)
    return False, None, need


def _capturing() -> bool:
    return torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()


def _finer() -> _Scope | None:
    for scope in reversed(_local.scopes):
        if scope.finer:
            return scope
    return None


# -------------------------------------------------------------------------------- blocks


def run(
    site: Site, forward: Callable[..., object], args: Sequence[object], kwargs: Mapping[str, object]
) -> object:
    """A block's forward, its weights bound (`torch._hook`): observed on its first run, bare
    when that showed it may run again, else under the layer."""
    local = _local
    outer, local.site = local.site, site
    try:
        facts = _served.memo.blocks.get(site.key)
        if facts is None:
            return _observed(site, forward, args, kwargs)
        if facts.bare:
            return _bare(site, facts, forward, args, kwargs)
        with _layered():
            return forward(*args, **kwargs)
    finally:
        local.site = outer


def _observed(
    site: Site, forward: Callable[..., object], args: Sequence[object], kwargs: Mapping[str, object]
) -> object:
    journal = _Journal()
    with _layered() as layer:
        banded = layer.banded
        layer.journals.append(journal)
        try:
            out = forward(*args, **kwargs)
        finally:
            layer.journals.remove(journal)
    bare = not journal.wrote and layer.banded == banded
    _served.memo.blocks[site.key] = Facts(bare, tuple(journal.generators.values()))
    return out


def _bare(
    site: Site,
    facts: Facts,
    forward: Callable[..., object],
    args: Sequence[object],
    kwargs: Mapping[str, object],
) -> object:
    states = [(g, g.get_state()) for g in facts.generators]
    kept = _take(site.root, site.state) if site.root is not None and site.state else []
    try:
        with _unlayered():
            return forward(*args, **kwargs)
    except Exception as exc:
        need = short(exc)
        if need is None and not _refused(exc) or _capturing():
            raise
        _clear(exc)
    memo = _served.memo
    memo.retries += 1
    for generator, state in states:
        generator.set_state(state)
    _put(kept)
    _served.room.make(site.device, (need or 0) + SLACK)
    with _layered() as layer:
        banded = layer.banded
        out = forward(*args, **kwargs)
    if layer.banded != banded:
        memo.blocks[site.key] = Facts(False, facts.generators)
    return out


def enter() -> None:
    """A `use` opened on this thread: until it closes, ops here are caught at the op."""
    local = _local
    local.depth += 1
    if local.depth == 1 and local.layer is None:
        local.used = local.layer = Layer()
        local.used.__enter__()


def leave() -> None:
    local = _local
    local.depth -= 1
    if local.depth == 0 and local.used is not None:
        used, local.used = local.used, None
        if local.layer is used:
            local.layer = None
        used.__exit__(None, None, None)


@contextmanager
def _layered() -> Iterator[Layer]:
    local = _local
    if local.layer is not None:
        yield local.layer
        return
    layer = local.layer = Layer()
    try:
        with layer:
            yield layer
    finally:
        local.layer = None


@contextmanager
def _unlayered() -> Iterator[None]:
    """Without this thread's layer, when it is the innermost mode (else under it)."""
    local = _local
    layer = local.layer
    if layer is None or _get_current_dispatch_mode() is not layer:
        yield
        return
    local.layer = None
    try:
        with _pop_mode_temporarily():
            yield
    finally:
        local.layer = layer


_MISSING = object()


def _take(root: nn.Module, paths: Sequence[str]) -> list[tuple[object, str, object, object]]:
    """Declared state as it is now: each (owner, attribute, value, a list's or dict's items)."""
    taken = []
    for path in paths:
        where, _, attr = path.rpartition(".")
        owner = functools.reduce(getattr, where.split("."), root) if where else root
        value = getattr(owner, attr, _MISSING)
        items = (
            list(value)
            if isinstance(value, list)
            else dict(value)
            if isinstance(value, dict)
            else None
        )
        taken.append((owner, attr, value, items))
    return taken


def _put(taken: Sequence[tuple[object, str, object, object]]) -> None:
    for owner, attr, value, items in taken:
        if value is _MISSING:
            vars(owner).pop(attr, None)
            continue
        if isinstance(value, list) and isinstance(items, list):
            value[:] = items
        elif isinstance(value, dict) and isinstance(items, dict):
            value.clear()
            value.update(items)
        setattr(owner, attr, value)


# --------------------------------------------------------------------------- the layer


@dataclass(frozen=True, slots=True)
class _Trait:
    writes: tuple[int, ...]
    names: tuple[str, ...]
    seeded: bool
    fresh: bool
    rule: Callable[[OpOverload, Sequence[object], Mapping[str, object]], _Plan | None] | None


_TRAITS: dict[OpOverload, _Trait] = {}


def _trait(func: OpOverload) -> _Trait:
    found = _TRAITS.get(func)
    if found is None:
        schema = func._schema
        writes = tuple(
            i for i, a in enumerate(schema.arguments) if a.alias_info and a.alias_info.is_write
        )
        found = _TRAITS[func] = _Trait(
            writes,
            tuple(schema.arguments[i].name for i in writes),
            torch.Tag.nondeterministic_seeded in func.tags,
            all(r.alias_info is None for r in schema.returns),
            _RULES.get(func.overloadpacket),
        )
    return found


@dataclass(eq=False, slots=True)
class _Journal:
    """A block's first run: the storages its ops made, whether one wrote into any other, and
    the generators its seeded ops drew from."""

    made: set[int] = field(default_factory=set)
    wrote: bool = False
    generators: dict[int, torch.Generator] = field(default_factory=dict)

    def before(
        self, trait: _Trait, func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object]
    ) -> None:
        if trait.writes and not self.wrote:
            written = [
                args[i] if i < len(args) else kwargs.get(n)
                for i, n in zip(trait.writes, trait.names, strict=True)
            ]
            self.wrote = any(p and p not in self.made for p in map(_storage, _tensors(written)))
        if trait.seeded:
            for generator in _generators(func, args, kwargs):
                self.generators[id(generator)] = generator

    def after(self, trait: _Trait, out: object) -> None:
        if trait.fresh:
            self.made.update(p for p in map(_storage, _tensors(out)) if p)


class Layer(TorchDispatchMode):
    """Every op, caught at itself: out of memory, it runs again after rungs 1-2, then asks the
    cut around it to cut finer, then runs in bands, then after rung 4 (§5.3)."""

    def __init__(self) -> None:
        super().__init__()
        self.journals: list[_Journal] = []
        self.banded = 0

    @classmethod
    def _should_skip_dynamo(cls) -> bool:
        return False  # nothing compiles here: no Dynamo wrapper around every op

    def __torch_dispatch__(  # type: ignore[override]
        self,
        func: OpOverload,
        types: object,
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        trait = _trait(func)
        for journal in self.journals:
            journal.before(trait, func, args, kwargs)
        memo = _served.memo
        if func in memo.banded and trait.rule is not None:
            key = _op_key(func, args, kwargs)
            size = memo.sizes.get(key)
            if size and (plan := trait.rule(func, args, kwargs)) is not None:
                return self._made(trait, self._banded(func, plan, key, size))
        states = (
            [(g, g.get_state()) for g in _generators(func, args, kwargs)] if trait.seeded else []
        )
        try:
            out = func(*args, **kwargs)
        except Exception as exc:
            need = short(exc)
            device = _device(args, kwargs)
            if need is None and not _refused(exc) or device is None or _capturing():
                raise
            _clear(exc)
        else:
            return self._made(trait, out)
        return self._made(trait, self._caught(func, trait, args, kwargs, device, need or 0, states))

    def _made(self, trait: _Trait, out: object) -> object:
        for journal in self.journals:
            journal.after(trait, out)
        return out

    def _caught(
        self,
        func: OpOverload,
        trait: _Trait,
        args: Sequence[object],
        kwargs: Mapping[str, object],
        device: torch.device,
        need: int,
        states: Sequence[tuple[torch.Generator, Tensor]],
    ) -> object:
        memo, room = _served.memo, _served.room

        def again() -> object:
            for generator, state in states:
                generator.set_state(state)
            return func(*args, **kwargs)

        memo.retries += 1
        room.make(device, need + SLACK)
        ok, out, need = _attempt(again)
        if ok:
            return out
        scope = _finer()
        if scope is not None:
            raise Finer(scope, need)
        if trait.rule is not None and (plan := trait.rule(func, args, kwargs)) is not None:
            return self._banded(func, plan, _op_key(func, args, kwargs), 0)
        while room.lower(device, need + SLACK):
            memo.retries += 1
            ok, out, need = _attempt(again)
            if ok:
                return out
        raise room.exhausted(device, need, str(func))

    def _banded(self, func: OpOverload, plan: _Plan, key: tuple[object, ...], size: int) -> object:
        """The op in bands of `size` rows (0: as many as the room holds, at most half), into
        its whole result on the device; rows halve while a band runs out, down to `align`."""
        memo, room, device, align = _served.memo, _served.room, plan.device, plan.align
        if not size:
            fits = room.available(device) // max(2 * plan.per_row, 1)
            size = min(plan.height // 2, fits) if fits else plan.height // 2
        size = max(align, size // align * align)
        name = f"{func.overloadpacket.__name__} rows"
        out = _shrinking(lambda _: plan.empty(), align, align, device, name)[1]
        if plan.gather is not None:
            size = _shrinking(plan.gather, size, align, device, name)[0]
        at = 0
        while at < plan.height:

            def band(rows: int, a: int = at) -> None:
                b = min(a + rows, plan.height)
                out.narrow(plan.axis, a, b - a).copy_(plan.band(a, b))

            size = _shrinking(band, size, align, device, name)[0]
            at += size
        memo.sizes[key] = size
        memo.banded.add(func)
        self.banded += 1
        _note(name, size)
        return out if plan.finish is None else plan.finish(out)


def _shrinking[T](
    step: Callable[[int], T], size: int, align: int, device: torch.device, name: str
) -> tuple[int, T]:
    """`step(size)` until it fits: room made, then half the rows down to `align`, then the
    hot watermark lower."""
    memo, room = _served.memo, _served.room
    made = False
    while True:
        ok, out, need = _attempt(functools.partial(step, size))
        if ok:
            return size, out  # type: ignore[return-value]
        memo.retries += 1
        room.make(device, need + SLACK)
        if size > align:
            size = max(align, size // 2 // align * align)
        elif not made:
            made = True
        elif not room.lower(device, need + SLACK):
            raise room.exhausted(device, need, name)


def _storage(t: Tensor) -> int:
    try:
        return int(t.untyped_storage().data_ptr())
    except RuntimeError:  # a wrapper with no storage (a weight placeholder)
        return 0


def _tensors(value: object) -> list[Tensor]:
    """The tensors in `value`, two levels of sequences and mappings deep."""
    found: list[Tensor] = []
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, Tensor):
            found.append(item)
        elif depth < 3 and isinstance(item, (list, tuple)):
            stack.extend((v, depth + 1) for v in reversed(item))
        elif depth < 3 and isinstance(item, Mapping):
            stack.extend((v, depth + 1) for v in reversed(list(item.values())))
    return found


def _device(args: Sequence[object], kwargs: Mapping[str, object]) -> torch.device | None:
    for t in _tensors((args, kwargs)):
        return t.device
    device = kwargs.get("device")
    return torch.device(device) if isinstance(device, (str, torch.device)) else None


def _generators(
    func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object]
) -> list[torch.Generator]:
    """The generators a seeded op draws from: its own, else its device's default."""
    own = [v for v in (*args, *kwargs.values()) if isinstance(v, torch.Generator)]
    if own:
        return own
    device = _device(args, kwargs) or torch.device("cpu")
    if device.type == "cuda":
        return [
            torch.cuda.default_generators[
                device.index if device.index is not None else torch.cuda.current_device()
            ]
        ]
    return [torch.default_generator]


def _op_key(
    func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object]
) -> tuple[object, ...]:
    return (func, _signature(tuple(args)), _signature(tuple(sorted(kwargs.items()))))


def _signature(value: object) -> object:
    if isinstance(value, Tensor):
        return (tuple(value.shape), value.dtype)
    if isinstance(value, (list, tuple)):
        return tuple(_signature(v) for v in value)
    return value


# --------------------------------------------------------------------------------- bands


@dataclass(frozen=True, slots=True)
class _Plan:
    """An op as rows [a, b) of its result along `axis`: `band(a, b)` computes them (starting
    on multiples of `align`, about `per_row` device bytes a row); `gather(rows)` reads what
    every band needs first; `finish` makes the op's own result from the filled tensor."""

    device: torch.device
    empty: Callable[[], Tensor]
    axis: int
    height: int
    band: Callable[[int, int], Tensor]
    per_row: int
    align: int = 1
    gather: Callable[[int], None] | None = None
    finish: Callable[[Tensor], object] | None = None


def _arguments(
    func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object]
) -> dict[str, object]:
    named: dict[str, object] = {}
    for i, a in enumerate(func._schema.arguments):
        if i < len(args):
            named[a.name] = args[i]
        elif a.name in kwargs:
            named[a.name] = kwargs[a.name]
        elif a.has_default_value():
            named[a.name] = a.default_value
    return named


def _meta(func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object]) -> list[Tensor]:
    """The op's results as meta tensors (empty: not new tensors, or no meta kernel)."""

    def meta(value: object) -> object:
        if isinstance(value, Tensor):
            return torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device="meta"
            )
        if isinstance(value, (list, tuple)):
            return type(value)(meta(v) for v in value)
        return value

    try:
        with _disable_current_modes():
            made = func(*(meta(a) for a in args), **{k: meta(v) for k, v in kwargs.items()})
    except Exception:
        return []
    return [m for m in (made if isinstance(made, tuple) else (made,)) if isinstance(m, Tensor)]


def _empty(like: Tensor, device: torch.device) -> Callable[[], Tensor]:
    return lambda: torch.empty_strided(like.shape, like.stride(), dtype=like.dtype, device=device)


def _each(value: object, n: int, default: int) -> list[int]:
    values = (
        [int(v) for v in value] if isinstance(value, (list, tuple)) else [int(value or default)]
    )
    return values * n if len(values) == 1 else values


def _conv(func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object]) -> _Plan | None:
    """Output rows [a, b) read input rows [a*s - p, (b-1)*s - p + d*(k-1) + 1), zeros past the
    edges: the conv's own padding when both sides take as many, else padded rows."""
    a = _arguments(func, args, kwargs)
    x, w, bias = a.get("input"), a.get("weight"), a.get("bias")
    if not isinstance(x, Tensor) or not isinstance(w, Tensor) or a.get("transposed"):
        return None
    n = w.dim() - 2
    if n < 1 or x.dim() != w.dim() or isinstance(a.get("padding"), str):
        return None
    stride, padding = _each(a.get("stride"), n, 1), _each(a.get("padding"), n, 0)
    dilation, groups = _each(a.get("dilation"), n, 1), int(a.get("groups") or 1)  # type: ignore[call-overload]
    made = _meta(func, args, kwargs)
    j = n - 2 if n >= 2 else 0
    axis = 2 + j
    if not made or made[0].shape[axis] < 2:
        return None
    out = made[0]
    s, d, k, p, size = stride[j], dilation[j], int(w.shape[axis]), padding[j], int(x.shape[axis])

    def band(lo: int, hi: int) -> Tensor:
        first, last = lo * s - p, (hi - 1) * s - p + d * (k - 1) + 1
        start, stop = max(first, 0), min(last, size)
        rows = x.narrow(axis, start, stop - start)
        pads = list(padding)
        if start - first == last - stop:
            pads[j] = start - first
        else:
            edge = [0, 0] * (x.dim() - 1 - axis) + [start - first, last - stop]
            rows, pads[j] = aten.constant_pad_nd.default(rows, edge, 0.0), 0
        return aten.convolution.default(
            rows, w, bias, stride, pads, dilation, False, [0] * n, groups
        )

    height = int(out.shape[axis])
    per_row = (_nbytes(out) + _nbytes(x) * s) // height
    return _Plan(x.device, _empty(out, x.device), axis, height, band, per_row)


def _pad(func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object]) -> _Plan | None:
    """A constant padding: output rows [a, b) are input rows [a - top, b - top), the value
    past its edges."""
    a = _arguments(func, args, kwargs)
    x = a.get("self", a.get("input"))
    if func.overloadpacket is aten.pad and a.get("mode", "constant") != "constant":
        return None
    pad = [int(v) for v in a.get("pad") or ()]  # type: ignore[union-attr]
    value = float(a.get("value") or 0.0)  # type: ignore[arg-type]
    if not isinstance(x, Tensor) or x.dim() < 2 or min(pad, default=0) < 0:
        return None
    axis, size = x.dim() - 2, int(x.shape[x.dim() - 2])
    rest = pad + [0] * max(4 - len(pad), 0)
    top = rest[2]
    made = _meta(aten.constant_pad_nd.default, (x, pad, value), {})
    if not made or made[0].shape[axis] < 2:
        return None

    def band(lo: int, hi: int) -> Tensor:
        first, last = lo - top, hi - top
        before = min(max(-first, 0), hi - lo)
        length = max(min(last, size) - max(first, 0), 0)
        edge = list(rest)
        edge[2], edge[3] = before, hi - lo - before - length
        rows = x.narrow(axis, min(max(first, 0), size), length)
        return aten.constant_pad_nd.default(rows, edge, value)

    height = int(made[0].shape[axis])
    per_row = (_nbytes(made[0]) + _nbytes(x)) // height
    return _Plan(x.device, _empty(made[0], x.device), axis, height, band, per_row)


def _upsample(
    func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object]
) -> _Plan | None:
    """Nearest upsampling by a whole factor `r` along the rows: output rows [a, b), on multiples
    of `r`, read input rows [a/r, b/r)."""
    x = args[0] if args else None
    made = _meta(func, args, kwargs)
    if not isinstance(x, Tensor) or x.dim() < 3 or not made:
        return None
    out = made[0]
    axis = x.dim() - 2 if x.dim() > 3 else 2
    height, size = int(out.shape[axis]), int(x.shape[axis])
    r = height // max(size, 1)
    if r < 2 or r * size != height:
        return None
    vec, spatial = func.overloadpacket.vec, [int(n) for n in out.shape[2:]]

    def band(lo: int, hi: int) -> Tensor:
        sizes = list(spatial)
        sizes[axis - 2] = hi - lo
        return vec(x.narrow(axis, lo // r, (hi - lo) // r), sizes, None)

    per_row = (_nbytes(out) + _nbytes(x)) // height
    return _Plan(x.device, _empty(out, x.device), axis, height, band, per_row, align=r)


def _group_norm(
    func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object]
) -> _Plan | None:
    """Per image and group, mean and variance over every band in float32, merged as one pass
    over all of it would be (up to rounding); then each band normalized."""
    a = _arguments(func, args, kwargs)
    x, weight, bias = a.get("input"), a.get("weight"), a.get("bias")
    groups = int(a.get("num_groups", a.get("group")) or 1)  # type: ignore[call-overload]
    eps = float(a.get("eps") or 1e-5)  # type: ignore[arg-type]
    if not isinstance(x, Tensor) or x.dim() < 3 or x.shape[1] % groups:
        return None
    made = _meta(func, args, kwargs)
    axis = x.dim() - 2 if x.dim() > 3 else 2
    batch, channels, height = int(x.shape[0]), int(x.shape[1]), int(x.shape[axis])
    if not made or height < 2:
        return None
    shape = [batch, channels] + [1] * (x.dim() - 2)
    affine: list[Tensor] = []
    moments: list[Tensor] = []

    wide = torch.promote_types(x.dtype, torch.float32)

    def gather(rows: int) -> None:
        parts = []
        for lo in range(0, height, rows):
            flat = x.narrow(axis, lo, min(rows, height - lo)).to(wide).reshape(batch, groups, -1)
            parts.append((*torch.var_mean(flat, dim=-1, correction=0), flat.shape[-1]))
            del flat
        total = sum(n for _, _, n in parts)
        mean = sum(m * (n / total) for _, m, n in parts)
        var = sum((v + (m - mean) ** 2) * (n / total) for v, m, n in parts)
        rstd = torch.rsqrt(var + eps)
        scale = rstd.repeat_interleave(channels // groups, dim=1)
        if isinstance(weight, Tensor):
            scale = scale * weight.to(wide)
        shift = -mean.repeat_interleave(channels // groups, dim=1) * scale
        if isinstance(bias, Tensor):
            shift = shift + bias.to(wide)
        affine[:] = [scale.reshape(shape), shift.reshape(shape)]
        moments[:] = [mean, rstd]

    def band(lo: int, hi: int) -> Tensor:
        return torch.addcmul(affine[1], x.narrow(axis, lo, hi - lo), affine[0])

    def finish(out: Tensor) -> object:
        if len(made) == 1:
            return out
        return out, moments[0].to(made[1].dtype), moments[1].to(made[2].dtype)

    per_row = (_nbytes(x) + 8 * x.numel()) // height
    return _Plan(
        x.device, _empty(made[0], x.device), axis, height, band, per_row, 1, gather, finish
    )


def _nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _packets(
    names: Sequence[str],
    rule: Callable[[OpOverload, Sequence[object], Mapping[str, object]], _Plan | None],
) -> dict[
    OpOverloadPacket, Callable[[OpOverload, Sequence[object], Mapping[str, object]], _Plan | None]
]:
    return {getattr(aten, n): rule for n in names}


_RULES = {
    **_packets(("convolution", "conv1d", "conv2d", "conv3d"), _conv),
    **_packets(("constant_pad_nd", "pad"), _pad),
    **_packets(
        tuple(
            f"{p}{d}d" for p in ("upsample_nearest", "_upsample_nearest_exact") for d in (1, 2, 3)
        ),
        _upsample,
    ),
    **_packets(("group_norm", "native_group_norm"), _group_norm),
}


# ---------------------------------------------------------------------------- call cuts


def _cut[T](
    name: str,
    key: tuple[object, ...],
    device: torch.device,
    total: int,
    finest: int,
    run: Callable[[int], T],
    shrink: Callable[[int], int],
    per_unit: int = 0,
) -> T:
    """`run(size)`: the call in pieces of at most `size` units of `total`. Up front at the
    memo's size; out of memory (its own, or an op's ask), room is made and it runs again once,
    then at each smaller size down to `finest`, past which the failure is its caller's."""
    memo, room = _served.memo, _served.room
    size = min(memo.sizes.get(key, total), total)
    scope = _Scope()
    scopes = _local.scopes
    scopes.append(scope)
    fresh = True
    try:
        while True:
            scope.finer = size > finest
            try:
                out = run(size)
            except Finer as asked:
                if asked.scope is not scope:
                    raise
                need, made = asked.need, True
                _clear(asked)
            except Exception as exc:
                need = short(exc)
                if need is None or (size <= finest and not fresh) or _capturing():
                    raise
                made = False
                _clear(exc)
            else:
                if size < total:
                    memo.sizes[key] = size
                    _note(name, size)
                return out
            memo.retries += 1
            if not made:
                room.make(device, need + SLACK)
                if fresh:
                    fresh = False
                    continue
            smaller = shrink(size)
            if per_unit and size == total and (fits := room.available(device) // per_unit):
                smaller = max(min(smaller, fits), finest)
            size, fresh = smaller, False
    finally:
        scopes.pop()


def _halving(finest: int) -> Callable[[int], int]:
    return lambda size: max(finest, size // 2)


def install(module: nn.Module) -> int:
    """The token cut on every FFN under `module`: diffusers' FeedForward and the activation
    projections inside one (GELU, GEGLU, SwiGLU and subclasses such as a fused SwiGLU), all
    token-wise. A block that calls its FFN's parts itself still cuts at the projection, whose
    [tokens, inner] intermediate is the widest. Returns how many it cut."""
    count = 0
    for sub in module.modules():
        kind = next(
            (
                c.__name__
                for c in type(sub).__mro__
                if c.__name__ in _FFN and c.__module__.startswith("diffusers.")
            ),
            None,
        )
        if kind is not None and "_varena_ffn" not in vars(sub):
            width = sum(m.out_features for m in sub.modules() if isinstance(m, nn.Linear))
            sub.forward = _ffn(kind, width, sub.forward)  # type: ignore[method-assign]
            vars(sub)["_varena_ffn"] = True
            count += 1
    return count


def _ffn(kind: str, width: int, forward: Callable[..., object]) -> Callable[..., object]:
    def cut(hidden_states: object, *args: object, **kwargs: object) -> object:
        x = hidden_states
        if args or kwargs or not isinstance(x, Tensor) or x.dim() < 2:
            return forward(x, *args, **kwargs)
        tokens = x.numel() // max(int(x.shape[-1]), 1)
        key = ("ffn", kind, int(x.shape[-1]), width, x.dtype)
        if tokens < 2 or key in _served.memo.whole:
            return forward(x)
        try:
            return _cut(
                "ffn tokens",
                key,
                x.device,
                tokens,
                1,
                lambda size: _tokens(forward, x, tokens, size),
                _halving(1),
                width * x.element_size(),
            )
        except _Whole:
            _served.memo.whole.add(key)
        return forward(x)

    return cut


def _tokens(forward: Callable[..., object], x: Tensor, tokens: int, size: int) -> object:
    if size >= tokens:
        return forward(x)
    flat = x.reshape(tokens, x.shape[-1])
    out: Tensor | None = None
    for at in range(0, tokens, size):
        part = forward(flat[at : at + size])
        if not isinstance(part, Tensor):
            raise _Whole
        if out is None:
            out = part.new_empty((tokens, *part.shape[1:]))
        out[at : at + part.shape[0]].copy_(part)
        del part
    assert out is not None
    return out.view(*x.shape[:-1], *out.shape[1:])


def attend(
    call: Callable[[Tensor, Tensor, Tensor, Tensor | None], Tensor],
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    *,
    rows: int = 0,
    heads: bool = True,
) -> Tensor:
    """`call(q, k, v, mask)` over [batch, sequence, heads, dim]. Out of memory it runs in pieces,
    each computed as the whole computes it: query heads with the K/V heads they read (whole K/V
    groups, then heads sharing one), when `heads` (the kernel reduces within a head); then query
    rows on multiples of `rows` (the kernel's query block; 0: never), each beside every key."""
    count, length = int(q.shape[2]), int(q.shape[1])
    group = max(count // int(k.shape[2]), 1)
    if mask is not None and mask.dim() != 4:
        rows, heads = 0, False
    units = _head_units(count, group) if heads else [count]
    steps = max(rows, 1) if rows and rows < length else length

    def piece(size: int) -> tuple[int, int]:
        """Query heads and rows per piece for `size` query rows (all heads count)."""
        hp = next((u for u in units if u * length <= size), units[-1])
        rp = min(length, size // hp)
        return hp, length if rp >= length or not rows else max(steps, rp // steps * steps)

    def shrink(size: int) -> int:
        hp, rp = piece(size)
        if rp == length and hp > units[-1]:
            return units[units.index(hp) + 1] * length
        return hp * max(steps, rp // 2 // steps * steps)

    key = (
        "attention",
        count,
        int(k.shape[2]),
        int(q.shape[3]),
        int(k.shape[1]),
        q.dtype,
        mask is None,
        rows,
        heads,
    )
    total, finest = count * length, units[-1] * steps
    return _cut(
        "attention rows",
        key,
        q.device,
        total,
        finest,
        lambda size: _attend(call, q, k, v, mask, group, *piece(size)),
        shrink,
    )


def _head_units(count: int, group: int) -> list[int]:
    """Query heads per piece that keep each piece on whole K/V groups or inside one."""
    units = {group * m for m in range(1, count // group + 1)} | {
        d for d in range(1, group + 1) if group % d == 0
    }
    return sorted(units, reverse=True)


def _attend(
    call: Callable[[Tensor, Tensor, Tensor, Tensor | None], Tensor],
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None,
    group: int,
    hp: int,
    rp: int,
) -> Tensor:
    batch, length, count = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    if hp >= count and rp >= length:
        return call(q, k, v, mask)
    out: Tensor | None = None
    for h in range(0, count, hp):
        n = min(hp, count - h)
        lo, hi = h // group, (h + n - 1) // group + 1
        kh, vh = k[:, :, lo:hi], v[:, :, lo:hi]
        for s in range(0, length, rp):
            m = min(rp, length - s)
            part = mask
            if part is not None:
                part = part if part.shape[1] == 1 else part[:, h : h + n]
                part = part if part.shape[2] == 1 else part[:, :, s : s + m]
            piece = call(q[:, s : s + m, h : h + n], kh, vh, part)
            if out is None:
                out = piece.new_empty((batch, length, count, *piece.shape[3:]))
            out[:, s : s + m, h : h + n].copy_(piece)
            del piece
    assert out is not None
    return out


def split[T](
    name: str, key: str, device: torch.device, total: int, finest: int, run: Callable[[int], T]
) -> T:
    """A package's `split` loop (`cozy_runtime.author.split`): `run(most)` runs it in windows of
    at most `most` of its `total` elements, `finest` the least one can take."""
    return _cut(f"split {name}", ("split", key), device, total, finest, run, _halving(finest))
