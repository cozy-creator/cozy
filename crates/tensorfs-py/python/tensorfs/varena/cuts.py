"""The out-of-memory catch and exact cuts for torch modules in varena's memory (§5.2-§5.3).

An op out of device memory runs again, each try strictly smaller (no timers; freed bytes are not
progress): rungs 1-2, the innermost cut around it finer, its own bands, rung 4, else `exhausted`.
Cuts are exact and keep results on the device; the size that fit is remembered per shape key
(the memo) and applied up front after. A block whose first run wrote no tensor that existed before
it, joined no collective and changed none of its attributes runs bare; out of memory it gets its
generators and declared state back and runs again under the dispatch layer, which catches other
blocks' ops at the op.
"""

from __future__ import annotations

import functools
import gc
import re
import threading
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Protocol

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
type Args = Sequence[object]
type Kwargs = Mapping[str, object]
type Band = Callable[[int, int], Tensor]
type Rule = Callable[[OpOverload, Args, Kwargs], _Plan | None]
type Attention = Callable[[Tensor, Tensor, Tensor, Tensor | None], Tensor]

#: Free bytes a retry asks beyond the failed allocation: the allocator's rounding, a workspace.
SLACK = 64 << 20
_ALLOCATE = re.compile(r"Tried to allocate ([\d.]+) (B|KiB|MiB|GiB|TiB)")
_UNITS = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40}
#: diffusers' token-wise feed-forwards and the activation projections inside them
_FFN = frozenset({"FeedForward", "GELU", "GEGLU", "SwiGLU", "ApproximateGELU", "LinearActivation"})
_PLAIN = (int, float, bool, str, bytes, type(None))
#: ops another GPU waits on: a block that runs one never runs again on this GPU alone
_COLLECTIVES = frozenset({"c10d", "_c10d_functional", "_c10d_functional_autograd"})
_MISSING = object()


class Rungs(Protocol):
    """The memory's side (§5.2): `free` runs rungs 1-2 and `lower` rung 4 until `need` bytes are
    free (False: nothing left); `available` is what one allocation may take now (0: unknown)."""

    def free(self, device: torch.device, need: int) -> bool: ...
    def lower(self, device: torch.device, need: int) -> bool: ...
    def available(self, device: torch.device) -> int: ...
    def exhausted(self, device: torch.device, need: int, op: str) -> Exception: ...
    def noted(self, rung: int, why: str, nbytes: int) -> None: ...


class _NoRungs:
    """No memory serves this process: cuts still cut, nothing is freed."""

    def free(self, device: torch.device, need: int) -> bool:
        return False

    lower = free

    def available(self, device: torch.device) -> int:
        return 0

    def exhausted(self, device: torch.device, need: int, op: str) -> Exception:
        return torch.OutOfMemoryError(f"{op} needs {need} B more on {device}")

    def noted(self, rung: int, why: str, nbytes: int) -> None:
        pass


class Finer(Exception):
    """An op inside a cut still ran out with rungs 1-2 done: that cut runs finer."""

    def __init__(self, scope: _Scope, need: int) -> None:
        super().__init__(f"cut finer: {need} B more")
        self.scope, self.need = scope, need


class _Whole(Exception):
    """A cut's pieces do not join (a call returned no tensor): it runs whole."""


@dataclass(slots=True)
class Shape:
    """A model's passes at one shape key: the latest activation peak, the cuts in force, and
    the peak of its latest pass that ran uncut (-1: none did)."""

    model: str
    shape: str
    peak: int = -1
    whole: int = -1
    #: a cut that measured no lower than whole: planning never cuts this shape again
    whole_only: bool = False
    cuts: dict[str, int] = field(default_factory=dict)
    #: the cuts its passes ran: key -> (total, finest, shrink), what `coarser` relaxes
    scopes: dict[tuple[object, ...], tuple[int, int, Callable[[int], int]]] = field(
        default_factory=dict
    )
    #: an FFN cut's bytes per token no longer live at once when it runs in pieces (`planned`)
    units: dict[tuple[object, ...], int] = field(default_factory=dict)


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
    #: each model's latest shape: a key it never ran (a prompt of another length) starts from
    #: its cut scopes and FFN units, so planning can cut before its first pass (12163)
    latest: dict[str, Shape] = field(default_factory=dict)
    retries: int = 0

    def shape(self, model: str, key: str) -> Shape:
        shape = self.shapes.get((model, key))
        if shape is None:
            shape = self.shapes[(model, key)] = Shape(model, key)
            if (last := self.latest.get(model)) is not None:
                shape.scopes, shape.units = dict(last.scopes), dict(last.units)
        self.latest[model] = shape
        return shape


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
        if not _cut(track.shape):
            track.shape.whole = nbytes


def _cut(shape: Shape) -> bool:
    memo = _served.memo
    return any(memo.sizes.get(k, t) < t for k, (t, _, _) in shape.scopes.items())


def ffn_only(shape: Shape | None) -> bool:
    """Every cut of `shape` in force is an FFN's (planning's own kind)."""
    memo = _served.memo
    scopes = shape.scopes.items() if shape is not None else ()
    cut = [k for k, (t, _, _) in scopes if memo.sizes.get(k, t) < t]
    return bool(cut) and all(k[0] == "ffn" for k in cut)


def _note(name: str, size: int) -> None:
    site = _local.site
    if site is not None and site.track.shape is not None:
        site.track.shape.cuts[name] = size


# ------------------------------------------------------------------------ out of memory


def short(exc: BaseException, first: bool = False) -> int | None:
    """The bytes a device allocation failure asked for (0: not said); None for any other. A
    `first` failure counts a cuBLAS or cuDNN status too (its own allocation may be behind it): it
    runs once more with room made, and a second is the library's own."""
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


def _settle[T](
    run: Callable[[int], T],
    size: int,
    finest: int,
    shrink: Callable[[int], int],
    device: torch.device,
    name: str,
    key: tuple[object, ...] | None = None,
    again: bool | None = None,
    before: Callable[[int], T | None] | None = None,
    last: Callable[[], T | None] | None = None,
) -> tuple[int, T]:
    """`run(size)` until it fits, each try strictly smaller: room made (rungs 1-2), each `shrink`
    down to `finest`, the first and the finest size tried again with room made, `before` (an
    op's cut) once, then rung 4 while it makes the room, then `last` (rung 5), else
    `exhausted`. With `key` it is a call that cuts itself:
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
                if not rungs.free(device, need + SLACK) and again:
                    again = False  # the same size again, with no room made, fails the same way
                    if scope and size <= finest:
                        raise
            memo.retries += 1
            if again:
                again = False
            elif size > finest:
                size = max(finest, shrink(size))
                again = scope is None and size <= finest
            elif before is not None and (out := before(need)) is not None:
                break
            elif not rungs.lower(device, need + SLACK):
                if last is not None and (out := last()) is not None:
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


def planned(shape: Shape | None, short: int) -> bool:
    """Planning's one cut: FFN tokens, never attention. The widest FFN scope of `shape` (the
    outermost: its bytes per token over its tokens) that can cover `short` bytes goes to the
    fewest pieces whose saving (those bytes over the tokens no longer live at once) does;
    False, cutting nothing, when none can or the shape measured a cut no lower than whole."""
    if shape is None or shape.whole_only:
        return False
    memo = _served.memo
    for key in sorted(shape.units, key=lambda k: -shape.units[k] * shape.scopes[k][0]):
        unit, (total, finest, _) = shape.units[key], shape.scopes[key]
        size, pieces = min(memo.sizes.get(key, total), total), 2
        while (piece := -(-total // pieces)) > finest and unit * (size - piece) < short:
            pieces *= 2
        if piece < size and unit * (size - piece) >= short:
            memo.sizes[key] = piece
            return True
    return False


def relaxing(shape: Shape | None, peak: int) -> int | None:
    """What a pass of `shape` needs with its cuts relaxed (`coarser`): its peak measured uncut,
    when one was, else twice `peak` (one step at most doubles it); None when none is cut."""
    if shape is None or not _cut(shape):
        return None
    return shape.whole if shape.whole >= 0 else 2 * peak


def coarser(shape: Shape) -> None:
    """Relaxation (the planner's): a shape's cuts go back to whole when it ran uncut before,
    else one step toward it, half as many pieces (a floor-halved size doubles back to whole
    exactly); its peak waits to be measured again."""
    memo = _served.memo
    for key, (total, _, _) in shape.scopes.items():
        size = min(memo.sizes.get(key, total), total)
        if size < total:
            pieces = 1 if shape.whole >= 0 else (-(-total // size) + 1) // 2
            memo.sizes[key] = -(-total // max(pieces, 1))
    shape.peak, shape.cuts = -1, {}


def _halve(align: int, size: int) -> int:
    return max(align, size // 2 // align * align)


# -------------------------------------------------------------------------------- blocks


def run(site: Site, forward: Callable[..., object], args: Args, kwargs: Kwargs) -> object:
    """A block's forward, its weights bound (`torch._hook`): observed under the layer on its
    first run, then bare when that showed it may run again, else under the layer. A first run
    out of memory outside any op (a kernel's launch) that changed nothing yet runs once more
    with room made and its default generators put back."""
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
            states = [(g, g.get_state()) for g in _defaults()]
            layer.journals.append(journal)
            try:
                out = forward(*args, **kwargs)
            except Exception as exc:  # out of memory where no op shows it (a kernel's launch)
                need, drawn = short(exc, True), {id(g) for g, _ in states}
                changed = journal.wrote or set(journal.generators) - drawn
                if need is None or _capturing() or changed or before != _attributes(site.block):
                    raise
                _clear(exc)
                memo.retries += 1
                for generator, state in states:
                    generator.set_state(state)
                _served.rungs.free(site.device, need + SLACK)
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


@contextmanager
def own_layer() -> Iterator[None]:
    """Ops caught at the op where this thread's modes are off (a bind's page-in transform):
    a layer of its own meanwhile."""
    outer, _local.layer = _local.layer, None
    try:
        with caught():
            yield
    finally:
        _local.layer = outer


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
type Trait = tuple[tuple[tuple[int, str], ...], bool, bool, Rule | None]


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
    """A block's first run: the storages its ops made, whether one wrote into any other or
    joined a collective, and the generators its seeded ops drew from."""

    made: set[int] = field(default_factory=set)
    wrote: bool = False
    generators: dict[int, torch.Generator] = field(default_factory=dict)

    def before(self, func: OpOverload, args: Args, kwargs: Kwargs) -> None:
        writes, seeded, _, _ = _trait(func)
        self.wrote |= func.namespace in _COLLECTIVES
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

        def hosted() -> object:  # rung 5: idle tensors to the host, else the result itself
            if _spill(device, need + SLACK, (args, kwargs)) >= need + SLACK:
                try:
                    return again(0)
                except Exception as exc:
                    if short(exc, True) is None:
                        raise
                    _clear(exc)
            for generator, state in states:
                generator.set_state(state)
            return _hosted(func, args, kwargs, device)

        return _settle(again, 1, 1, int, device, name, again=False, before=cut, last=hosted)[1]

    def _banded(self, func: OpOverload, plan: _Plan, key: tuple[object, ...], size: int) -> object:
        """The op in bands of `size` rows (0: what the room holds, at most half) into its whole
        result on the device; the rows halve while a band runs out, down to `align`."""
        device, align, height = plan.x.device, plan.align, int(plan.out.shape[plan.axis])
        if not size:
            fits = _served.rungs.available(device) // max(2 * plan.per_row, 1)
            size = min(height // 2, fits or height // 2)
        size, name = max(align, size // align * align), f"{func.overloadpacket.__name__} rows"
        halve = functools.partial(_halve, align)
        def spill() -> Tensor:  # rung 5: idle tensors to the host, else the result itself
            nbytes = plan.out.numel() * plan.out.element_size() + SLACK
            if _spill(device, nbytes, (plan.x,)) >= nbytes:
                try:
                    return _empty(plan.out, device)
                except torch.OutOfMemoryError:
                    pass
            _served.rungs.noted(5, f"hosted {name}", nbytes - SLACK)
            return _host_like(plan.out, device)

        empty = functools.partial(_empty, plan.out, device)
        out = _settle(lambda _: empty(), align, align, halve, device, name, last=spill)[1]
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
        return out if plan.finish is None else plan.finish(out)


def _band(plan: _Plan, out: Tensor, at: int, height: int, rows: int) -> None:
    stop = min(at + rows, height)
    out.narrow(plan.axis, at, stop - at).copy_(plan.band(at, stop))


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


def _defaults() -> list[torch.Generator]:
    """The default generators: the CPU's, and each GPU's once CUDA is up."""
    cuda = torch.cuda.default_generators if torch.cuda.is_initialized() else ()
    return [torch.default_generator, *cuda]


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


#: page-locked host blocks whose tensors are gone, until the device passes their readers
_DEAD: list[Tensor] = []
_made = 0
_COPIES = (aten.contiguous, aten.clone, aten._to_copy)


def _hosted(func: OpOverload, args: Args, kwargs: Kwargs, device: torch.device) -> object:
    """Rung 5: the result no cut splits goes to page-locked host memory the GPU addresses over
    the bus, as 0.27.4 spilled exactly it; None for an op with no `out` form or no one result."""
    made, packet = _meta(func, args, kwargs), func.overloadpacket
    out = getattr(packet, "out", None) if packet not in _COPIES else None
    if device.type != "cuda" or len(made) != 1 or out is None and packet not in _COPIES:
        return None
    result = _host_like(made[0], device)
    with _disable_current_modes():
        if out is None:
            result.copy_(args[0])
        else:
            out(*args, **kwargs, out=result)
    _served.rungs.noted(5, f"hosted {func}", result.untyped_storage().nbytes())
    return result


def _spill(device: torch.device, need: int, operands: object) -> int:
    """Rung 5's first part, as 0.27.4 spilled: tensors this process holds in torch's allocator
    on `device` that the failing op does not read move to page-locked host memory in place,
    largest first, until `need` bytes went back. Each stays a CUDA tensor, read over the bus
    from then on, so whatever holds it is unchanged. Returns the bytes moved."""
    if device.type != "cuda":
        return 0
    torch.cuda.synchronize(device)
    segments = [
        (s["address"], s["address"] + s["total_size"])
        for s in torch.cuda.memory_snapshot()
        if s["device"] == device.index
    ]
    read = {t.untyped_storage().data_ptr() for t in _tensors(operands) if t.is_cuda}
    found: dict[int, torch.UntypedStorage] = {}
    for value in gc.get_objects():
        if type(value) not in (Tensor, nn.Parameter) or value.device != device:
            continue
        storage = value.untyped_storage()
        at = storage.data_ptr()
        if at and at not in read and any(lo <= at < hi for lo, hi in segments):
            found.setdefault(at, storage)
    moved = 0
    for storage in sorted(found.values(), key=lambda st: -st.nbytes()):
        if moved >= need:
            break
        alias = _host(storage.nbytes(), device)
        alias.copy_(torch.empty(0, dtype=torch.uint8, device=device).set_(storage))
        storage._swap_data_ptr_(alias.untyped_storage())  # its device bytes go with `alias`
        moved += storage.nbytes()
    if moved:
        _served.rungs.noted(5, "spilled idle tensors to the host", moved)
    return moved


class _Block:
    """A rung-5 block's CUDA face: its tensor keeps it. When that tensor goes, the block waits in
    `_DEAD` until the device has passed every kernel that may still read it (`_reclaim`)."""

    def __init__(self, nbytes: int) -> None:
        global _made
        self.host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        _made += nbytes
        data = (self.host.data_ptr(), False)
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": data, "version": 3}

    def __del__(self) -> None:
        _DEAD.append(self.host)


def _reclaim(device: torch.device) -> int:
    """Dead blocks go back to the host allocator once the device passed their readers; the bytes."""
    n = len(_DEAD)
    if not n:
        return 0
    torch.cuda.synchronize(device)
    nbytes = sum(h.numel() for h in _DEAD[:n])
    del _DEAD[:n]
    return nbytes


def _host(nbytes: int, device: torch.device) -> Tensor:
    """A CUDA uint8 tensor of `nbytes` whose bytes are page-locked host memory, which the GPU
    addresses over the bus. They go back once the tensor goes (`_Block`), not at the request's
    end: a pass hosting a result per block would otherwise hold every one of them."""
    _reclaim(device)
    return torch.as_tensor(_Block(nbytes), device=device)


def _host_like(like: Tensor, device: torch.device) -> Tensor:
    """`like`'s shape, strides and dtype over page-locked host memory (`_host`)."""
    span = 1 + sum((n - 1) * s for n, s in zip(like.shape, like.stride(), strict=True))
    nbytes = span * like.element_size() if like.numel() else 0
    return _host(nbytes, device).view(like.dtype).as_strided(like.shape, like.stride())


def hosted_end(device: torch.device) -> int:
    """At a request's end, dead blocks go back (live results keep theirs); the bytes rung 5
    hosted since the last end, so the caller empties the host allocator's cache when any."""
    global _made
    _reclaim(device)
    made, _made = _made, 0
    return made


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
    _conv: "convolution conv1d conv2d conv3d",
    _pad: "constant_pad_nd pad",
    _upsample: " ".join(_UPSAMPLES),
    _group_norm: "group_norm native_group_norm",
}
_RULES: dict[object, Rule] = {getattr(aten, n): r for r, ns in _SPEC.items() for n in ns.split()}


# ---------------------------------------------------------------------------- call cuts


def install(module: nn.Module) -> int:
    """The token cut on every FFN under `module`: diffusers' FeedForward and the activation
    projections inside one (GELU, GEGLU, SwiGLU and subclasses such as a fused SwiGLU), all
    token-wise. A block that calls its FFN's parts itself still cuts at the projection, whose
    [tokens, inner] intermediate is the widest. Returns how many it cut."""
    count = 0
    for sub in module.modules():
        kinds = [c.__name__ for c in type(sub).__mro__ if c.__module__.startswith("diffusers.")]
        kind = next((k for k in kinds if k in _FFN), None)
        if kind is not None and "_varena_ffn" not in vars(sub):
            # a quantized lane's linears are encoded leaves: any module naming out_features counts
            width = _width([m for m in sub.modules() if getattr(m, "out_features", 0)])
            own = vars(sub).get("forward")
            sub.forward = _ffn(kind, width, own or current(sub))  # type: ignore[method-assign]
            vars(sub)["_varena_ffn"] = True
            count += 1
    return count


def _width(linears: list[nn.Module]) -> int:
    """Elements per token an FFN holds at once beyond its result, which pieces stop holding at
    once: its first projection's output and its last linear's input (the activation), less the
    last linear's output, which a cut assembles whole anyway."""
    if not linears:
        return 0
    first, last = linears[0], linears[-1]
    if first is last:
        return int(first.out_features)
    return int(first.out_features) + int(last.in_features) - int(last.out_features)


def current(module: nn.Module) -> Callable[..., object]:
    """`module`'s class forward as it is at each call."""
    return lambda *args, **kwargs: type(module).forward(module, *args, **kwargs)


def _ffn(kind: str, width: int, forward: Callable[..., object]) -> Callable[..., object]:
    def cut(x: object, *args: object, **kwargs: object) -> object:
        if args or kwargs or not isinstance(x, Tensor) or x.dim() < 2:
            return forward(x, *args, **kwargs)
        tokens, key = x.numel() // max(x.shape[-1], 1), ("ffn", kind, x.shape[-1], width, x.dtype)
        if tokens < 2 or key in _served.memo.whole:
            return forward(x)
        per_token, device = max(width, x.shape[-1]) * x.element_size(), x.device
        site = _local.site
        if site is not None and site.track.shape is not None:
            site.track.shape.units[key] = per_token

        def shrink(size: int) -> int:  # half, or the tokens the room holds; the memo keeps it,
            return min(size // 2, _served.rungs.available(device) // per_token or size)  # not x

        try:
            run = functools.partial(_tokens, forward, x, tokens)
            return _settle(run, tokens, 1, shrink, x.device, "ffn tokens", key)[1]
        except _Whole:
            _served.memo.whole.add(key)
        return forward(x)

    cut._varena = True  # type: ignore[attr-defined]  # what a package's shape check allows
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


def split[T](
    name: str, key: str, device: torch.device, total: int, finest: int, run: Callable[[int], T]
) -> T:
    """A package's `split` loop (`cozy_runtime.author.split`): `run(most)` runs it in windows of
    at most `most` of its `total` elements, `finest` the least one can take."""
    halve = functools.partial(_halve, 1)
    return _settle(run, total, finest, halve, device, f"split {name}", ("split", key))[1]
