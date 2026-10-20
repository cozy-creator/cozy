"""The out-of-memory catch and exact cuts for torch modules in varena's memory (§5.2-§5.3).

An op out of device memory runs again, each try strictly smaller (no timers; freed bytes are not
progress): rungs 1-2, the innermost cut around it finer, its own bands, rung 4, else `exhausted`.
Cuts are exact and keep results on the device; the size that fit is remembered per shape key
(the memo) and applied up front after. A block whose first run wrote no tensor that existed before
it and changed none of its attributes runs bare; out of memory it gets its generators and declared
state back and runs again under the dispatch layer, which catches other blocks' ops at the op.
"""

from __future__ import annotations

import functools
import re
import threading
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, TypeVar

import torch
from torch import Tensor, nn
from torch._ops import OpOverload
from torch.utils._python_dispatch import (
    TorchDispatchMode,
    _disable_current_modes,
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)

aten = torch.ops.aten
T = TypeVar("T")
Args: TypeAlias = Sequence[object]
Kwargs: TypeAlias = Mapping[str, object]
Band: TypeAlias = Callable[[int, int], Tensor]
Rule: TypeAlias = Callable[[OpOverload, Args, Kwargs], "_Plan | None"]
Attention: TypeAlias = Callable[[Tensor, Tensor, Tensor, Tensor | None], Tensor]


class Output(Protocol):
    """A fresh exact result and its checked band writer; aliases retain its storage owner."""

    @property
    def tensor(self) -> Tensor: ...

    def write_band(self, source: Tensor, axis: int, offset: int) -> None: ...


@dataclass(frozen=True, slots=True)
class _DeviceOutput:
    tensor: Tensor

    def write_band(self, source: Tensor, axis: int, offset: int) -> None:
        self.tensor.narrow(axis, offset, source.shape[axis]).copy_(source)

#: Free bytes a retry asks beyond the failed allocation: the allocator's rounding, a workspace.
SLACK = 64 << 20
_ALLOCATE = re.compile(r"Tried to allocate ([\d.]+) (B|KiB|MiB|GiB|TiB)")
_UNITS = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40}
#: diffusers' token-wise feed-forwards and the activation projections inside them
_FFN = frozenset({"FeedForward", "GELU", "GEGLU", "SwiGLU", "ApproximateGELU", "LinearActivation"})
_PLAIN = (int, float, bool, str, bytes, type(None))
_MISSING = object()


class Rungs(Protocol):
    """The memory's side (§5.2): `free` runs rungs 1-2 and `lower` rung 4 until `need` bytes are
    free (False: nothing left); `available` is what one allocation may take now (0: unknown)."""

    def free(self, device: torch.device, need: int) -> bool: ...
    def lower(self, device: torch.device, need: int) -> bool: ...
    def available(self, device: torch.device) -> int: ...
    def mapped_output(self, like: Tensor, device: torch.device) -> Output | None: ...
    def exhausted(self, device: torch.device, need: int, op: str) -> Exception: ...


class _NoRungs:
    """No memory serves this process: cuts still cut, nothing is freed."""

    def free(self, device: torch.device, need: int) -> bool:
        return False

    lower = free

    def available(self, device: torch.device) -> int:
        return 0

    def mapped_output(self, like: Tensor, device: torch.device) -> Output | None:
        return None

    def exhausted(self, device: torch.device, need: int, op: str) -> Exception:
        return torch.OutOfMemoryError(f"{op} needs {need} B more on {device}")


class Finer(Exception):
    """An op inside a cut still ran out with rungs 1-2 done: that cut runs finer."""

    def __init__(self, scope: _Scope, need: int) -> None:
        super().__init__(f"cut finer: {need} B more")
        self.scope, self.need = scope, need


class _Whole(Exception):
    """A cut's pieces do not join (a call returned no tensor): it runs whole."""


@dataclass(slots=True)
class Shape:
    """A model's passes at one shape key: the latest activation peak, the cuts in force."""

    model: str
    shape: str
    peak: int = -1
    cuts: dict[str, int] = field(default_factory=dict)
    #: the cuts its passes ran: key -> (total, finest, shrink), what `finer` steps
    scopes: dict[tuple[object, ...], tuple[int, int, Callable[[int], int]]] = field(
        default_factory=dict
    )


@dataclass(frozen=True, slots=True)
class Facts:
    """A block as its first run showed it: bare, and the generators its seeded ops drew from."""

    bare: bool
    generators: tuple[torch.Generator, ...]


@dataclass(slots=True)
class Memo:
    """Per process: each block's facts, each cut's size per shape key, each model's peaks."""

    blocks: dict[tuple[str, int], Facts] = field(default_factory=dict)
    sizes: dict[tuple[object, ...], int] = field(default_factory=dict)
    banded: set[OpOverload] = field(default_factory=set)  # ops with a band size in force
    whole: set[tuple[object, ...]] = field(default_factory=set)
    shapes: dict[tuple[str, str], Shape] = field(default_factory=dict)
    retries: int = 0

    def shape(self, model: str, key: str) -> Shape:
        return self.shapes.setdefault((model, key), Shape(model, key))


@dataclass(eq=False, slots=True)
class Track:
    """A model's pass now: the memo entry its blocks' cuts are noted in."""

    model: str
    shape: Shape | None = None


@dataclass(eq=False, slots=True)
class Site:
    """A block for the catch: its memo key, device, model's track and module, and declared
    state (attribute paths under `root` put back before it runs again)."""

    key: tuple[str, int]
    device: torch.device
    track: Track
    block: nn.Module | None = None
    root: nn.Module | None = None
    state: tuple[str, ...] = ()


@dataclass(slots=True)
class _Served:
    memo: Memo
    rungs: Rungs


_served = _Served(Memo(), _NoRungs())


def serve(memo: Memo, rungs: Rungs) -> None:
    """The process's memory: its memo and rungs serve every catch and cut from now on."""
    _served.memo, _served.rungs = memo, rungs


@dataclass(eq=False, slots=True)
class _Scope:
    finer: bool = False


class _Local(threading.local):
    def __init__(self) -> None:
        self.layer: Layer | None = None
        self.scopes: list[_Scope] = []
        self.site: Site | None = None


_local = _Local()


def shape_of(args: Args, kwargs: Kwargs) -> str:
    """A pass's shape key: its tensor arguments' dtypes and shapes."""
    tensors = _tensors((args, kwargs))
    return ", ".join(f"{str(t.dtype).removeprefix('torch.')}{list(t.shape)}" for t in tensors)


def peaked(track: Track, nbytes: int) -> None:
    if track.shape is not None:
        track.shape.peak = nbytes


def _note(name: str, size: int) -> None:
    site = _local.site
    if site is not None and site.track.shape is not None:
        site.track.shape.cuts[name] = size


# ------------------------------------------------------------------------ out of memory


def short(exc: BaseException, first: bool = False) -> int | None:
    """The bytes a device allocation failure asked for (0: not said); None for any other. A
    `first` failure counts a cuBLAS or cuDNN status too (its own allocation may be behind it): it
    runs once more with room made, and a second is the library's own."""
    if type(exc).__module__ == "tensorfs.varena" and getattr(exc, "code", None) == "DEVICE_FULL":
        return int(getattr(exc, "deficit", getattr(exc, "need", 0)))
    if isinstance(exc, torch.OutOfMemoryError):
        found = _ALLOCATE.search(str(exc))
        return int(float(found[1]) * _UNITS[found[2]]) if found else 0
    text = str(exc)
    if isinstance(exc, torch.AcceleratorError):
        return 0 if getattr(exc, "error_code", None) == 2 or "out of memory" in text else None
    failed = "CUDA error: out of memory" in text or "_ALLOC_FAILED" in text
    failed |= first and ("CUBLAS_STATUS_" in text or "CUDNN_STATUS_" in text)
    return 0 if isinstance(exc, RuntimeError) and failed else None


def _clear(exc: BaseException) -> None:
    """The failed attempt's frames go, and with them its tensors."""
    seen: set[int] = set()
    at: BaseException | None = exc
    while at is not None and id(at) not in seen:
        seen.add(id(at))
        if at.__traceback__ is not None:
            traceback.clear_frames(at.__traceback__)
        at = at.__cause__ or at.__context__


def _capturing() -> bool:
    return torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing()


def _settle(
    run: Callable[[int], T],
    size: int,
    finest: int,
    shrink: Callable[[int], int],
    device: torch.device,
    name: str,
    key: tuple[object, ...] | None = None,
    again: bool | None = None,
    before: Callable[[int], T | None] | None = None,
    fallback: Callable[[], T | None] | None = None,
) -> tuple[int, T]:
    """`run(size)` until it fits, each try strictly smaller: room made (rungs 1-2), each `shrink`
    down to `finest`, the first and the finest size tried again with room made, `before` (an
    op's cut) once, then rung 4 and an optional fresh-output fallback, else `exhausted`.
    With `key` it is a call that cuts itself:
    up front at the memo's size, finer when an op inside asks (`Finer`), past `finest` its
    failure its caller's, and the size that fit remembered."""
    memo, rungs, total = _served.memo, _served.rungs, size
    scope = _Scope() if key is not None else None
    if scope is not None:
        size = min(memo.sizes.get(key, total), total)  # type: ignore[arg-type]
        _local.scopes.append(scope)
        site = _local.site
        if site is not None and site.track.shape is not None:
            site.track.shape.scopes[key] = (total, finest, shrink)  # type: ignore[index]
    again = scope is not None or size <= finest if again is None else again
    try:
        while True:
            if scope is not None:
                scope.finer = size > finest
            try:
                out = run(size)
                break
            except Finer as asked:
                if asked.scope is not scope:
                    raise
                need, again = asked.need, False
                _clear(asked)
            except Exception as exc:
                need = short(exc)
                if need is None or _capturing() or scope and size <= finest and not again:
                    raise
                _clear(exc)
                rungs.free(device, need + SLACK)
            memo.retries += 1
            if again:
                again = False
            elif size > finest:
                size = max(finest, shrink(size))
                again = scope is None and size <= finest
            elif before is not None and (out := before(need)) is not None:
                break
            elif not rungs.lower(device, need + SLACK):
                if fallback is not None and (out := fallback()) is not None:
                    break
                raise rungs.exhausted(device, need, name)
            else:
                before = None
    finally:
        if scope is not None:
            _local.scopes.pop()
    if key is not None and size < total:
        memo.sizes[key] = size
        _note(name, size)
    return size, out


def finer(shape: Shape | None) -> bool:
    """Rung 3 ahead of the planner's rung 4: every cut a shape's passes ran goes one step finer,
    and its peak waits to be measured again. False when each is at its finest already."""
    if shape is None:
        return False
    memo, moved = _served.memo, False
    for key, (total, finest, shrink) in shape.scopes.items():
        size = min(memo.sizes.get(key, total), total)
        if size > finest:
            memo.sizes[key] = max(finest, shrink(size))
            moved = True
    if moved:
        shape.peak = -1
    return moved


def _halve(align: int, size: int) -> int:
    return max(align, size // 2 // align * align)


# -------------------------------------------------------------------------------- blocks


def run(site: Site, forward: Callable[..., object], args: Args, kwargs: Kwargs) -> object:
    """A block's forward, its weights bound (`torch._hook`): observed under the layer on its
    first run, then bare when that showed it may run again, else under the layer."""
    memo, local = _served.memo, _local
    outer, local.site = local.site, site
    try:
        facts = memo.blocks.get(site.key)
        if facts is not None and facts.bare:
            return _bare(site, facts, forward, args, kwargs)
        with caught() as layer:
            if facts is not None:
                return forward(*args, **kwargs)
            journal, before, banded = _Journal(), _attributes(site.block), layer.banded
            layer.journals.append(journal)
            try:
                out = forward(*args, **kwargs)
            finally:
                layer.journals.remove(journal)
        # a step counter or a band means it must never run twice
        bare = not journal.wrote and before == _attributes(site.block) and layer.banded == banded
        memo.blocks[site.key] = Facts(bare, tuple(journal.generators.values()))
        return out
    finally:
        local.site = outer


def _bare(
    site: Site, facts: Facts, forward: Callable[..., object], args: Args, kwargs: Kwargs
) -> object:
    states, put = [(g, g.get_state()) for g in facts.generators], _kept(site.root, site.state)
    layer = _local.layer
    pop = layer is not None and _get_current_dispatch_mode() is layer  # else it runs under it
    try:
        with _pop_mode_temporarily() if pop else nullcontext():
            _local.layer = None if pop else layer
            return forward(*args, **kwargs)
    except Exception as exc:
        need = short(exc, True)
        if need is None or _capturing():
            raise
        _clear(exc)
    finally:
        _local.layer = layer
    _served.memo.retries += 1
    for generator, state in states:
        generator.set_state(state)
    put()
    _served.rungs.free(site.device, need + SLACK)
    with caught() as layer:
        banded = layer.banded
        out = forward(*args, **kwargs)
    if layer.banded != banded:
        _served.memo.blocks[site.key] = Facts(False, facts.generators)
    return out


@contextmanager
def caught() -> Iterator[Layer]:
    """Ops on this thread are caught at the op until it closes (a request, a cache's loop, a
    block): this thread's layer, pushed if there is none."""
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


def _kept(root: nn.Module | None, paths: Sequence[str]) -> Callable[[], None]:
    """Declared state as it is now, and how to put it back (lists and dicts by their items)."""
    taken = []
    for path in paths if root is not None else ():
        where, _, attr = path.rpartition(".")
        owner = functools.reduce(getattr, where.split("."), root) if where else root
        value = getattr(owner, attr, _MISSING)
        taken.append((owner, attr, value, value.copy() if isinstance(value, list | dict) else 0))

    def put() -> None:
        for owner, attr, value, items in taken:
            if value is _MISSING:
                vars(owner).pop(attr, None)
                continue
            if isinstance(value, list | dict):
                value.clear()
                (value.update if isinstance(value, dict) else value.extend)(items)
            setattr(owner, attr, value)

    return put


def _attributes(module: nn.Module | None) -> dict[tuple[int, str], object]:
    """A module tree's plain attributes, tensors held (by identity), public lists and dicts
    (their items by identity): what a forward may change that no op shows."""
    found: dict[tuple[int, str], object] = {}
    for m in module.modules() if module is not None else ():
        for name, value in vars(m).items():
            if isinstance(value, _PLAIN):
                found[(id(m), name)] = (type(value), value)
            elif isinstance(value, Tensor):
                found[(id(m), name)] = id(value)
            elif not name.startswith("_") and isinstance(value, list | dict):
                items = value.values() if isinstance(value, dict) else value
                found[(id(m), name)] = (id(value), tuple(map(id, items)))
    return found


# --------------------------------------------------------------------------- the layer


#: an op's traits: the arguments it writes, whether it draws from a generator, whether its
#: results are new tensors, its band rule
Trait: TypeAlias = tuple[tuple[tuple[int, str], ...], bool, bool, Rule | None]


@functools.cache
def _trait(func: OpOverload) -> Trait:
    schema = func._schema
    written = [a.alias_info is not None and a.alias_info.is_write for a in schema.arguments]
    writes = tuple((i, a.name) for i, a in enumerate(schema.arguments) if written[i])
    fresh = all(r.alias_info is None for r in schema.returns)
    seeded = torch.Tag.nondeterministic_seeded in func.tags
    return writes, seeded, fresh, _RULES.get(func.overloadpacket)


@dataclass(eq=False, slots=True)
class _Journal:
    """A block's first run: the storages its ops made, whether one wrote into any other, and
    the generators its seeded ops drew from."""

    made: set[int] = field(default_factory=set)
    wrote: bool = False
    generators: dict[int, torch.Generator] = field(default_factory=dict)

    def before(self, func: OpOverload, args: Args, kwargs: Kwargs) -> None:
        writes, seeded, _, _ = _trait(func)
        if writes and not self.wrote:
            written = [args[i] if i < len(args) else kwargs.get(n) for i, n in writes]
            self.wrote = any(p and p not in self.made for p in map(_storage, _tensors(written)))
        if seeded:
            self.generators.update((id(g), g) for g in _generators(func, args, kwargs))

    def after(self, func: OpOverload, out: object) -> None:
        if _trait(func)[2]:
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
        self, func: OpOverload, types: object, args: Args = (), kwargs: Kwargs | None = None
    ) -> object:
        kwargs = kwargs or {}
        for journal in self.journals:
            journal.before(func, args, kwargs)
        out = self._run(func, args, kwargs)
        for journal in self.journals:
            journal.after(func, out)
        return out

    def _run(self, func: OpOverload, args: Args, kwargs: Kwargs) -> object:
        memo, rungs, name = _served.memo, _served.rungs, str(func)
        _, draws, _, rule = _trait(func)
        key = _op_key(func, args, kwargs) if rule is not None else ()
        if func in memo.banded and (size := memo.sizes.get(key)):
            if (plan := rule(func, args, kwargs)) is not None:  # type: ignore[misc]
                return self._banded(func, plan, key, size)
        seeded = _generators(func, args, kwargs) if draws else []
        states = [(g, g.get_state()) for g in seeded]

        def again(_: int) -> object:
            for generator, state in states:
                generator.set_state(state)
            return func(*args, **kwargs)

        try:
            return func(*args, **kwargs)
        except Exception as exc:
            need, device = short(exc, True), _device(args, kwargs)
            if need is None or device is None or _capturing():
                raise
            _clear(exc)
        memo.retries += 1
        rungs.free(device, need + SLACK)

        def cut(need: int) -> object:  # before rung 4: the cut around it finer, else its bands
            if (scope := next((s for s in reversed(_local.scopes) if s.finer), None)) is not None:
                raise Finer(scope, need)
            plan = rule(func, args, kwargs) if rule is not None else None
            return None if plan is None else self._banded(func, plan, key, 0)

        return _settle(again, 1, 1, int, device, name, again=False, before=cut)[1]

    def _banded(self, func: OpOverload, plan: _Plan, key: tuple[object, ...], size: int) -> object:
        """The op in bands of `size` rows (0: what the room holds, at most half) into its whole
        result on the device; the rows halve while a band runs out, down to `align`."""
        device, align, height = plan.x.device, plan.align, int(plan.out.shape[plan.axis])
        if not size:
            fits = _served.rungs.available(device) // max(2 * plan.per_row, 1)
            size = min(height // 2, fits or height // 2)
        size, name = max(align, size // align * align), f"{func.overloadpacket.__name__} rows"
        halve = functools.partial(_halve, align)
        out = _output(plan.out, device, name)
        if plan.gather is not None:
            size = _settle(plan.gather, size, align, halve, device, name)[0]
        at = 0
        while at < height:
            band = functools.partial(_band, plan, out, at, height)
            size = _settle(band, size, align, halve, device, name)[0]
            at += size
        _served.memo.sizes[key] = size
        _served.memo.banded.add(func)
        self.banded += 1
        _note(name, size)
        return out.tensor if plan.finish is None else plan.finish(out.tensor)


def _band(plan: _Plan, out: Output, at: int, height: int, rows: int) -> None:
    stop = min(at + rows, height)
    out.write_band(plan.band(at, stop), plan.axis, at)


def _storage(t: Tensor) -> int:
    try:
        return int(t.untyped_storage().data_ptr())
    except RuntimeError:  # a wrapper with no storage (a weight placeholder)
        return 0


def _tensors(value: object, depth: int = 3) -> list[Tensor]:
    """The tensors in `value`, `depth` levels of sequences and mappings deep."""
    if isinstance(value, Tensor):
        return [value]
    if depth and isinstance(value, list | tuple | Mapping):
        items = value.values() if isinstance(value, Mapping) else value
        return [t for item in items for t in _tensors(item, depth - 1)]
    return []


def _device(args: Args, kwargs: Kwargs) -> torch.device | None:
    """Where the op allocates: its `device` argument (a copy to the GPU from a CPU tensor),
    else its first tensor's."""
    device = kwargs.get("device")
    if isinstance(device, str | torch.device):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device
    for t in _tensors((args, kwargs)):
        return t.device
    return None


def _generators(func: OpOverload, args: Args, kwargs: Kwargs) -> list[torch.Generator]:
    """The generators a seeded op draws from: its own, else its device's default."""
    own = [v for v in (*args, *kwargs.values()) if isinstance(v, torch.Generator)]
    device = _device(args, kwargs) or torch.device("cpu")
    if own or device.type != "cuda":
        return own or [torch.default_generator]
    index = device.index if device.index is not None else torch.cuda.current_device()
    return [torch.cuda.default_generators[index]]


def _op_key(func: OpOverload, args: Args, kwargs: Kwargs) -> tuple[object, ...]:
    return (func, _signature(tuple(args)), _signature(tuple(sorted(kwargs.items()))))


def _signature(value: object) -> object:
    if isinstance(value, Tensor):
        return (tuple(value.shape), value.dtype)
    return tuple(map(_signature, value)) if isinstance(value, list | tuple) else value


# --------------------------------------------------------------------------------- bands


@dataclass(frozen=True, slots=True)
class _Plan:
    """An op as rows [a, b) of its result `out` (meta) along `axis`: `band(a, b)` computes them,
    starting on multiples of `align`, about `per_row` device bytes a row; `gather(rows)` reads
    what every band needs first; `finish` makes the op's own result from the filled tensor."""

    x: Tensor
    out: Tensor
    axis: int
    band: Band
    per_row: int
    align: int = 1
    gather: Callable[[int], None] | None = None
    finish: Callable[[Tensor], object] | None = None


def _plan(
    made: list[Tensor], x: Tensor, axis: int, band: Band, reads: int = 1, **more: object
) -> _Plan | None:
    """When the op makes a tensor of two rows or more; `reads`: input bytes per output byte."""
    if not made or made[0].shape[axis] < 2:
        return None
    per_row = (_nbytes(made[0]) + reads * _nbytes(x)) // int(made[0].shape[axis])
    return _Plan(x, made[0], axis, band, per_row, **more)  # type: ignore[arg-type]


def _arguments(func: OpOverload, args: Args, kwargs: Kwargs) -> dict[str, object]:
    named: dict[str, object] = {}
    for i, a in enumerate(func._schema.arguments):
        if i < len(args) or a.name in kwargs or a.has_default_value():
            default = a.default_value if a.has_default_value() else None
            named[a.name] = args[i] if i < len(args) else kwargs.get(a.name, default)
    return named


def _meta(func: OpOverload, args: Args, kwargs: Kwargs) -> list[Tensor]:
    """The op's results as meta tensors (empty: not new tensors, or no meta kernel)."""

    def meta(v: object) -> object:
        if isinstance(v, Tensor):
            return torch.empty_strided(v.shape, v.stride(), dtype=v.dtype, device="meta")
        return type(v)(map(meta, v)) if isinstance(v, list | tuple) else v

    try:
        with _disable_current_modes():
            made = func(*map(meta, args), **{k: meta(v) for k, v in kwargs.items()})
    except Exception:
        return []
    return [m for m in (made if isinstance(made, tuple) else (made,)) if isinstance(m, Tensor)]


def _empty(like: Tensor, device: torch.device) -> Tensor:
    return torch.empty_strided(like.shape, like.stride(), dtype=like.dtype, device=device)


def _output(like: Tensor, device: torch.device, name: str) -> Output:
    """Prefer device storage through every relief rung, then an admitted host destination."""
    def allocate(_: int) -> Output:
        # This allocation owns its exhaustion/fallback. The surrounding dispatch
        # layer must not turn its OOM into Exhausted before this ladder sees it.
        layer = _local.layer
        pop = layer is not None and _get_current_dispatch_mode() is layer
        with _pop_mode_temporarily() if pop else nullcontext():
            return _DeviceOutput(_empty(like, device))

    def finer(need: int) -> None:
        scope = next((scope for scope in reversed(_local.scopes) if scope.finer), None)
        if scope is not None:
            raise Finer(scope, need)

    def mapped() -> Output | None:
        if not like.is_contiguous() or _capturing():
            return None
        return _served.rungs.mapped_output(like, device)

    return _settle(allocate, 1, 1, int, device, name, before=finer, fallback=mapped)[1]


def _rows(x: Tensor, axis: int, first: int, last: int, value: float = 0.0) -> Tensor:
    """Rows [first, last) of `x` along `axis`, `value` past its edges."""
    size = int(x.shape[axis])
    start, before = min(max(first, 0), size), min(max(-first, 0), last - first)
    length = max(min(last, size) - start, 0)
    edge = [0, 0] * (x.dim() - 1 - axis) + [before, last - first - before - length]
    rows = x.narrow(axis, start, length)
    return aten.constant_pad_nd.default(rows, edge, value) if any(edge) else rows


def _axis(x: Tensor) -> int:
    """The rows bands cut: a 1D signal's length, else the second-to-last dim."""
    return x.dim() - 2 if x.dim() > 3 else 2


def _each(value: object, n: int, default: int) -> list[int]:
    values = list(value) if isinstance(value, list | tuple) else [value or default]
    return [int(v) for v in values] * (n if len(values) == 1 else 1)  # type: ignore[call-overload]


def _nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _contiguous(func: OpOverload, args: Args, kwargs: Kwargs) -> _Plan | None:
    """Copy logical rows to fresh contiguous storage, with only one device band staged."""
    a = _arguments(func, args, kwargs)
    x = a.get("self")
    memory_format = a.get("memory_format")
    # ATen exposes contiguous's omitted default as enum integer 0. Clone's
    # omitted default is None (preserve format), which must remain distinct.
    contiguous = memory_format is torch.contiguous_format or type(memory_format) is int and memory_format == 0
    if (not isinstance(x, Tensor) or x.layout is not torch.strided
            or torch.is_grad_enabled() or _capturing() or x.is_contiguous()
            or not contiguous):
        return None
    axes = [i for i, size in enumerate(x.shape) if size > 1]
    if not axes:
        return None
    room = _served.rungs.available(x.device)
    axis = next((i for i in axes if not room or 2 * _nbytes(x) // x.shape[i] <= room),
                max(axes, key=lambda i: x.shape[i]))

    def band(lo: int, hi: int) -> Tensor:
        # Prefer contiguous destination bands; with less room a later axis still
        # coalesces each bounded source instead of copying per permuted element.
        return func(x.narrow(axis, lo, hi - lo), memory_format=torch.contiguous_format)

    return _plan(_meta(func, args, kwargs), x, axis, band)


def _silu(func: OpOverload, args: Args, kwargs: Kwargs) -> _Plan | None:
    """Native inference SiLU in exact logical bands; its full result may be mapped."""
    if func is not aten.silu.default:
        return None
    x = _arguments(func, args, kwargs).get("self")
    if (not isinstance(x, Tensor) or x.layout is not torch.strided
            or torch.is_grad_enabled() or _capturing()):
        return None
    made = _meta(func, args, kwargs)
    axes = [i for i, size in enumerate(x.shape) if size > 1]
    if not axes or not made or not made[0].is_contiguous():
        return None
    room = _served.rungs.available(x.device)
    axis = next((i for i in axes if not room or 2 * _nbytes(x) // x.shape[i] <= room),
                max(axes, key=lambda i: x.shape[i]))

    def band(lo: int, hi: int) -> Tensor:
        return func(x.narrow(axis, lo, hi - lo))

    return _plan(made, x, axis, band)



def _add(func: OpOverload, args: Args, kwargs: Kwargs) -> _Plan | None:
    """Equal-shape inference add in native bands, preserving alpha and promotion."""
    if func is not aten.add.Tensor:
        return None
    a = _arguments(func, args, kwargs)
    x, y = a.get("self"), a.get("other")
    if (not isinstance(x, Tensor) or not isinstance(y, Tensor)
            or x.layout is not torch.strided or y.layout is not torch.strided
            or x.shape != y.shape or x.device != y.device
            or torch.is_grad_enabled() or _capturing()):
        return None
    made = _meta(func, args, kwargs)
    axes = [i for i, size in enumerate(x.shape) if size > 1]
    if not axes or not made or not made[0].is_contiguous():
        return None
    row_bytes = _nbytes(made[0]) + _nbytes(x) + _nbytes(y)
    room = _served.rungs.available(x.device)
    axis = next((i for i in axes if not room or 2 * row_bytes // x.shape[i] <= room),
                max(axes, key=lambda i: x.shape[i]))
    alpha = a.get("alpha", 1)

    def band(lo: int, hi: int) -> Tensor:
        return func(x.narrow(axis, lo, hi - lo), y.narrow(axis, lo, hi - lo), alpha=alpha)

    return _Plan(x, made[0], axis, band, row_bytes // int(x.shape[axis]))


def _conv(func: OpOverload, args: Args, kwargs: Kwargs) -> _Plan | None:
    """Output rows [a, b) read input rows [a*s - p, (b-1)*s - p + d*(k-1) + 1), zeros past the
    edges."""
    a = _arguments(func, args, kwargs)
    x, w, bias, groups = a.get("input"), a.get("weight"), a.get("bias"), a.get("groups") or 1
    if not isinstance(x, Tensor) or not isinstance(w, Tensor) or not x.dim() == w.dim() > 2:
        return None
    if a.get("transposed") or isinstance(a.get("padding"), str):
        return None
    n, axis = w.dim() - 2, _axis(x)
    stride, padding = _each(a.get("stride"), n, 1), _each(a.get("padding"), n, 0)
    dilation, j = _each(a.get("dilation"), n, 1), axis - 2
    s, d, k = stride[j], dilation[j], int(w.shape[axis])

    def band(lo: int, hi: int) -> Tensor:
        rows = _rows(x, axis, lo * s - padding[j], (hi - 1) * s - padding[j] + d * (k - 1) + 1)
        pads = [0 if i == j else p for i, p in enumerate(padding)]
        conv = aten.convolution.default
        return conv(rows, w, bias, stride, pads, dilation, False, [0] * n, groups)

    return _plan(_meta(func, args, kwargs), x, axis, band, reads=s)


def _pad(func: OpOverload, args: Args, kwargs: Kwargs) -> _Plan | None:
    """A constant padding: output rows [a, b) are input rows [a - top, b - top), the value past
    its edges."""
    a = _arguments(func, args, kwargs)
    x, pad, value = a.get("self"), list(a.get("pad") or ()), float(a.get("value") or 0)  # type: ignore[call-overload, arg-type]
    if not isinstance(x, Tensor) or x.dim() < 2 or min(pad, default=0) < 0:
        return None
    if a.get("mode", "constant") != "constant":
        return None
    axis, rest = x.dim() - 2, [*pad, 0, 0, 0, 0][: max(len(pad), 4)]
    others = [*rest[:2], 0, 0, *rest[4:]]

    def band(lo: int, hi: int) -> Tensor:
        rows = _rows(x, axis, lo - rest[2], hi - rest[2], value)
        return aten.constant_pad_nd.default(rows, others, value)

    return _plan(_meta(aten.constant_pad_nd.default, (x, pad, value), {}), x, axis, band)


def _upsample(func: OpOverload, args: Args, kwargs: Kwargs) -> _Plan | None:
    """Nearest upsampling by a whole factor `r`: output rows [a, b), on multiples of `r`, read
    input rows [a/r, b/r)."""
    x, made = args[0] if args else None, _meta(func, args, kwargs)
    if not isinstance(x, Tensor) or x.dim() < 3 or not made:
        return None
    axis, sizes = _axis(x), [int(n) for n in made[0].shape[2:]]
    r = sizes[axis - 2] // max(int(x.shape[axis]), 1)
    if r < 2 or r * x.shape[axis] != sizes[axis - 2]:
        return None

    def band(lo: int, hi: int) -> Tensor:
        cut = [hi - lo if i == axis - 2 else n for i, n in enumerate(sizes)]
        return func.overloadpacket.vec(x.narrow(axis, lo // r, (hi - lo) // r), cut, None)

    return _plan(made, x, axis, band, align=r)


def _group_norm(func: OpOverload, args: Args, kwargs: Kwargs) -> _Plan | None:
    """Per image and group, mean and variance over every band in float32 or wider, merged as one
    pass over all of it would be (up to rounding); then each band normalized."""
    a = _arguments(func, args, kwargs)
    x, weight, bias = a.get("input"), a.get("weight"), a.get("bias")
    groups, eps = int(a.get("num_groups", a.get("group")) or 1), float(a.get("eps") or 1e-5)  # type: ignore[call-overload, arg-type]
    if not isinstance(x, Tensor) or x.dim() < 3 or x.shape[1] % groups:
        return None
    axis, made, batch = _axis(x), _meta(func, args, kwargs), int(x.shape[0])
    per, height, total = int(x.shape[1]) // groups, int(x.shape[axis]), x.numel() // batch // groups
    wide, shape = torch.promote_types(x.dtype, torch.float32), [batch, -1] + [1] * (x.dim() - 2)
    held: list[Tensor] = []  # scale, shift, mean, rstd

    def gather(rows: int) -> None:
        parts = []
        for lo in range(0, height, rows):
            flat = x.narrow(axis, lo, min(rows, height - lo)).to(wide).reshape(batch, groups, -1)
            parts.append((*torch.var_mean(flat, dim=-1, correction=0), flat.shape[-1] / total))
        mean = sum(m * n for _, m, n in parts)
        rstd = torch.rsqrt(sum((v + (m - mean) ** 2) * n for v, m, n in parts) + eps)
        scale = rstd.repeat_interleave(per, dim=1)
        scale = scale * weight.to(wide) if isinstance(weight, Tensor) else scale
        shift = -mean.repeat_interleave(per, dim=1) * scale
        shift = shift + bias.to(wide) if isinstance(bias, Tensor) else shift
        held[:] = [scale.reshape(shape), shift.reshape(shape), mean, rstd]

    def band(lo: int, hi: int) -> Tensor:
        return torch.addcmul(held[1], x.narrow(axis, lo, hi - lo), held[0])

    def finish(out: Tensor) -> object:
        stats = (held[2].to(made[1].dtype), held[3].to(made[2].dtype)) if len(made) > 2 else ()
        return (out, *stats) if stats else out

    return _plan(made, x, axis, band, reads=3, gather=gather, finish=finish)


_UPSAMPLES = [f"{p}{d}d" for p in ("upsample_nearest", "_upsample_nearest_exact") for d in "123"]
_SPEC = {
    # Tensor.contiguous also dispatches clone(memory_format=contiguous_format).
    _contiguous: "contiguous clone",
    _silu: "silu",
    _add: "add",
    _conv: "convolution conv1d conv2d conv3d",
    _pad: "constant_pad_nd pad",
    _upsample: " ".join(_UPSAMPLES),
    _group_norm: "group_norm native_group_norm",
}
_RULES: dict[object, Rule] = {getattr(aten, n): r for r, ns in _SPEC.items() for n in ns.split()}


# ---------------------------------------------------------------------------- call cuts


def install(module: nn.Module) -> int:
    """Independent-axis cuts under `module`: spatial bands for channel RMS normalization,
    and token bands for Diffusers' FFNs and their activation projections (GELU, GEGLU,
    SwiGLU and subclasses). A block that calls its FFN's parts still cuts at the projection,
    whose [tokens, inner] intermediate is the widest. Returns how many it cut."""
    count = 0
    for sub in module.modules():
        kinds = [c.__name__ for c in type(sub).__mro__ if c.__module__.startswith("diffusers.")]
        rms = _rms_contract(sub, kinds)
        if rms is not None and "_varena_rms" not in vars(sub):
            sub.forward = _rms(*rms, sub.forward)  # type: ignore[method-assign]
            vars(sub)["_varena_rms"] = True
            count += 1
        kind = next((k for k in kinds if k in _FFN), None)
        if kind is not None and "_varena_ffn" not in vars(sub):
            # a quantized lane's linears are encoded leaves: any module naming out_features counts
            width = sum(int(getattr(m, "out_features", 0) or 0) for m in sub.modules())
            sub.forward = _ffn(kind, width, sub.forward)  # type: ignore[method-assign]
            vars(sub)["_varena_ffn"] = True
            count += 1
    return count


def _rms_contract(module: nn.Module, kinds: Sequence[str]) -> tuple[str, bool, int, int] | None:
    """Diffusers' channel RMS norm: channel-only affine terms, no spatial reduction."""
    kind = next((name for name in kinds if name.endswith("RMS_norm")), None)
    gamma, bias = getattr(module, "gamma", None), getattr(module, "bias", None)
    first, scale = getattr(module, "channel_first", None), getattr(module, "scale", None)
    if kind is None or type(first) is not bool or not isinstance(gamma, Tensor):
        return None
    shape = tuple(gamma.shape)
    if not shape or (any(n != 1 for n in shape[1:]) if first else len(shape) != 1):
        return None
    channels = int(shape[0])
    if channels < 1 or not isinstance(scale, float) or scale != channels**0.5:
        return None
    if isinstance(bias, Tensor):
        if bias.shape != gamma.shape:
            return None
    elif bias != 0.0:
        return None
    return kind, first, channels, gamma.dim()


def _rms(
    kind: str, first: bool, channels: int, affine_dims: int, forward: Callable[..., object]
) -> Callable[..., object]:
    def cut(x: object, *args: object, **kwargs: object) -> object:
        if (args or kwargs or not isinstance(x, Tensor) or x.layout is not torch.strided
                or not x.is_contiguous() or torch.is_grad_enabled() or x.dim() < 3):
            return forward(x, *args, **kwargs)
        normalized = 1 if first else x.dim() - 1
        if x.shape[normalized] != channels or (first and x.dim() != affine_dims + 1):
            return forward(x)
        axis = max((i for i in range(1, x.dim()) if i != normalized), key=lambda i: x.shape[i])
        height, device = int(x.shape[axis]), x.device
        if height < 2:
            return forward(x)
        key = ("rms_norm", kind, tuple(x.shape), x.dtype, axis)
        per_row = x.numel() // height * max(x.element_size(), 4) * 4

        def shrink(size: int) -> int:
            return min(size // 2, _served.rungs.available(device) // per_row or size)

        try:
            run = functools.partial(_rms_bands, forward, x, axis)
            return _settle(run, height, 1, shrink, device, "rms spatial rows", key)[1]
        except _Whole:
            return forward(x)

    return cut


def _rms_bands(forward: Callable[..., object], x: Tensor, axis: int, rows: int) -> object:
    if rows >= x.shape[axis]:
        return forward(x)
    out: Output | None = None
    for at in range(0, int(x.shape[axis]), rows):
        length = min(rows, int(x.shape[axis]) - at)
        part = forward(x.narrow(axis, at, length))
        expected = (*x.shape[:axis], length, *x.shape[axis + 1 :])
        if not isinstance(part, Tensor) or part.shape != expected:
            raise _Whole()
        if out is None:
            like = torch.empty(tuple(x.shape), dtype=part.dtype, device="meta")
            out = _output(like, part.device, "RMS norm result")
        if part.dtype != out.tensor.dtype or part.device != out.tensor.device:
            raise _Whole()
        out.write_band(part, axis, at)
    assert out is not None
    return out.tensor


def _ffn(kind: str, width: int, forward: Callable[..., object]) -> Callable[..., object]:
    def cut(x: object, *args: object, **kwargs: object) -> object:
        if args or kwargs or not isinstance(x, Tensor) or x.dim() < 2:
            return forward(x, *args, **kwargs)
        tokens, key = x.numel() // max(x.shape[-1], 1), ("ffn", kind, x.shape[-1], width, x.dtype)
        if tokens < 2 or key in _served.memo.whole:
            return forward(x)
        per_token = max(width, x.shape[-1]) * x.element_size()
        device = x.device

        def shrink(size: int) -> int:  # half, or the tokens the room holds
            return min(size // 2, _served.rungs.available(device) // per_token or size)

        try:
            run = functools.partial(_tokens, forward, x, tokens)
            return _settle(run, tokens, 1, shrink, device, "ffn tokens", key)[1]
        except _Whole:
            _served.memo.whole.add(key)
        return forward(x)

    return cut


def _tokens(forward: Callable[..., object], x: Tensor, tokens: int, size: int) -> object:
    if size >= tokens:
        return forward(x)
    flat, out = x.reshape(tokens, x.shape[-1]), None
    for at in range(0, tokens, size):
        part = forward(flat[at : at + size])
        if not isinstance(part, Tensor):
            raise _Whole
        out = part.new_empty((tokens, *part.shape[1:])) if out is None else out
        out[at : at + part.shape[0]].copy_(part)
        del part
    assert out is not None
    return out.view(*x.shape[:-1], *out.shape[1:])


def attend(
    call: Attention,
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
    levels = [(h, length) for h in (_head_units(count, group) if heads else [count])]
    while 0 < rows < levels[-1][1]:
        levels.append((levels[-1][0], _halve(rows, levels[-1][1])))
    key = ("attention", *q.shape[1:], *k.shape[1:3], q.dtype, mask is None, rows, heads)
    run = functools.partial(_attend, call, q, k, v, mask, group)

    def piece(size: int) -> Tensor:  # a size is how many levels are left, the finest one last
        return run(*levels[len(levels) - size])

    return _settle(piece, len(levels), 1, lambda s: s - 1, q.device, "attention level", key)[1]


@functools.cache
def _head_units(count: int, group: int) -> list[int]:
    """Query heads per piece that keep each piece on whole K/V groups or inside one."""
    wholes = {group * m for m in range(1, count // group + 1)}
    return sorted(wholes | {d for d in range(1, group + 1) if group % d == 0}, reverse=True)


def _attend(
    call: Attention, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, *cut: int
) -> Tensor:
    """Query heads [h, h + hp) with the K/V heads they read, by query rows [s, s + rp)."""
    group, hp, rp = cut
    batch, length, count = int(q.shape[0]), int(q.shape[1]), int(q.shape[2])
    if hp >= count and rp >= length:
        return call(q, k, v, mask)
    whole = mask.expand(-1, count, length, -1) if mask is not None else None
    out: Tensor | None = None
    for h in range(0, count, hp):
        n, lo, hi = min(hp, count - h), h // group, (min(h + hp, count) - 1) // group + 1
        for s in range(0, length, rp):
            part = whole[:, h : h + n, s : s + rp] if whole is not None else None
            piece = call(q[:, s : s + rp, h : h + n], k[:, :, lo:hi], v[:, :, lo:hi], part)
            out = piece.new_empty((batch, length, count, *piece.shape[3:])) if out is None else out
            out[:, s : s + rp, h : h + n].copy_(piece)
            del piece
    assert out is not None
    return out


def split(
    name: str, key: str, device: torch.device, total: int, finest: int, run: Callable[[int], T]
) -> T:
    """A package's `split` loop (`cozy_runtime.author.split`): `run(most)` runs it in windows of
    at most `most` of its `total` elements, `finest` the least one can take."""
    halve = functools.partial(_halve, 1)
    return _settle(run, total, finest, halve, device, f"split {name}", ("split", key))[1]
