"""varena: one process's memory for model weights on one device (design varena-slim.md).

`Memory` places weight sets in pages its device's shared table shows free. On CUDA it is
torch's only allocator: a request's torch blocks come from its arena, which makes room (idle
pages, caches, then the request's own pages by furthest declared use) when free bytes fall short.
A `Pin` is a block mapped and filled: a uint8 DLPack tensor that, with every tensor made from it,
holds the block until the last one dies. Every call releases the GIL; a panic raises
`VarenaError` with code `INTERNAL`.
"""

from collections.abc import Sequence
from typing import Final, Literal

from .._ext import Checkpoint

class VarenaError(Exception):
    """`code`: EXHAUSTED, INVALID, DEVICE, IO, CLOSED or INTERNAL; `cuda_code`: a device error's
    CUresult (2: out of memory); `sticky`: the device context is gone and the process must end."""

    code: str
    cuda_code: int | None
    sticky: bool

class Exhausted(VarenaError):
    """The one memory error: `need` bytes short at op `site` on `gpu`, with the request's record."""

    gpu: str
    need: int
    site: str | None
    stats: Stats

class GpuId:
    """A device as the shared table names it: the driver's UUID, its model, its bytes."""

    def __init__(self, uuid: str, model: str, memory: int) -> None: ...
    @property
    def uuid(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def memory(self) -> int: ...

class Fill:
    """What turns a page's stored bytes into its weights."""

    class Stored(Fill):
        def __init__(self) -> None: ...

    class Declared(Fill):
        """A caller's transform (a LoRA bake) whose identity covers what it writes: kept."""

        def __init__(self, identity: str) -> None: ...
        @property
        def identity(self) -> str: ...

    class Undeclared(Fill):
        """A transform of unknown identity: never kept, never fetched speculatively."""

        def __init__(self) -> None: ...

class Level:
    RAM: Final[Level]
    GPU: Final[Level]

class Part:
    """One stored part's place: `offset` is within its region's pin."""

    @property
    def component(self) -> str: ...
    @property
    def key(self) -> str: ...
    @property
    def role(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    @property
    def shape(self) -> list[int]: ...
    @property
    def region(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def nbytes(self) -> int: ...

class Set:
    """A weight set: `regions[r]` names region (block) `r`'s tensors as `component/key`. Its
    fingerprint covers the layout, parts, `fill`, `release`, `shard` (`0/1` whole) and `gpu`."""

    def __init__(
        self,
        name: str,
        ck: Checkpoint,
        regions: list[list[str]],
        *,
        fill: Fill,
        release: str,
        shard: str,
        gpu: str,
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def fingerprint(self) -> str: ...
    @property
    def regions(self) -> list[list[str]]: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def spans(self) -> list[int]:
        """Each region's mapped bytes."""

    @property
    def parts(self) -> list[Part]: ...

class Memory:
    @staticmethod
    def cuda(ordinal: int) -> Memory:
        """CUDA device `ordinal`'s memory. Open it before torch initializes CUDA."""

    @staticmethod
    def host(gpu: GpuId) -> Memory:
        """Host pages of `gpu.memory` bytes (CPU, MPS) on the table `gpu.uuid` names."""

    @property
    def gpu(self) -> GpuId: ...
    def load(self, set: Set, level: Level) -> int:
        """Registers `set` and brings it to `level`; its id."""

    def unload(self, set: int) -> None: ...
    def fill(self, set: int, fill: Fill) -> None:
        """`set`'s current fill (a no-op when unchanged): pages filled under another come back
        fresh."""
    def request(self, id: str, plan: list[tuple[list[str], int]], idle: bool = False) -> Request:
        """Opens request `id`; `plan` is its steps, each `(set names, passes)`. An `idle` one (a
        capture or warm-up beside others' work) takes free bytes only: it evicts nothing."""

    def retain(self, set: int, blocks: list[int]) -> Retained: ...
    def context_measurable(self) -> bool: ...
    def bind(self, set: int, block: int, stream: int = 0, evict: bool = True) -> Pin:
        """Block (region) `block` of `set`, mapped and filled; `stream` (a CUstream, 0 the default)
        waits on the fill, and tensors made from the pin are consumed on it only. With `evict`
        false it maps into free bytes only, `Exhausted` at once when they are short."""

    def classes(self) -> list[int]:
        """Mapped bytes by class (F, O, S, N, D, W, P)."""

    def stats(self) -> Stats | None:
        """The record of the request that ended last."""

    def torch_stats(self) -> tuple[int, int, int, int, list[tuple[str, int, int]], list[tuple[tuple[int, int], int, int]]]:
        """varena's figures for torch's blocks: (allocated, mapped, peak, fragmentation (mapped
        less live), per open request (id, allocated, peak), per pool (pool, mapped, allocated))."""

    def serve_standalone(self) -> None:
        """Serves torch's allocations standalone, before any: from the device's free bytes less
        the backend's reserve. A leased Memory's are served by its lease."""

    def region(self, owner: str, kind: str, sets: Sequence[int] = (), spill: bool = True) -> Region:
        """A long-lived intermediate's region: kind kv, latent, conditioning, derived or workspace."""

    def graph_region(self, pool: tuple[int, int], sets: Sequence[int] = (), retained: Sequence[int] = ()) -> GraphRegion:
        """A component's graphs on this device, captured into `pool`."""

    def room(self, step: int | None = None) -> tuple[int, int, int, int, int]:
        """(free, cold, warm, granularity, floor) at plan step `step` (the cursor's when None)."""

    def fence_pool(self, pool: tuple[int, int], stream: int) -> None:
        """An event on `stream` after torch pool `pool`'s use there: its chunks leave, once
        released, only after every fence completed."""


    def prepare_pool(self, pool: tuple[int, int], expected: int) -> None:
        """Maps `expected` bytes into torch pool `pool` before a capture into it begins."""

    def lease(self, ns: int) -> tuple[str, str] | None:
        """Leases this GPU's weight pages from the manager TensorD serves in namespace `ns` (its
        uid), right after open: None when leased, else (code, why): `NotRegistered`,
        `Incompatible` or `Absent`."""

class Request:
    def pass_(self, set: int, key: str) -> Pass:
        """A pass of `set` keyed `key` (`entry|dims`): the plan's cursor moves to it."""

    def plan(self, plan: list[tuple[list[str], int]]) -> None:
        """Replaces the plan; its cursor starts over."""

    def yielding(self) -> Yielded:
        """The request waits on a child call until the guard ends."""
    def reset_peak(self) -> int:
        """Its torch arena's high-water since the last call or pass start, restarting there."""
    def allocating(self, memory: Memory) -> Allocating:
        """This thread's torch blocks go to this request's arena in `memory` until the guard
        ends (on this thread)."""

    def stats(self) -> Stats: ...
    def end(self) -> Stats:
        """Ends it: its pins and caches go."""

class Yielded:
    def end(self) -> None:
        """The wait on the child call is over."""

class Region:
    """A long-lived intermediate in an arena of its own; `close` (or its drop) releases it."""

    @property
    def id(self) -> int: ...
    def pin(self) -> RegionPin:
        """Pins it while in use, mapping it again first if it was reclaimed."""
    def allocating(self) -> RegionPin:
        """As `pin`, and this thread's torch blocks go to its arena until the pin ends."""
    def produced(self, ns: int) -> None: ...
    def next_use(self, step: int | None = None) -> None: ...
    def close(self) -> tuple[int, int, list[tuple[int, int, int]]]:
        """(allocated, peak, its blocks still live (ptr, bytes, stream))."""

class RegionPin:
    @property
    def fresh(self) -> bool:
        """It was remapped empty since its last pin: recompute it."""
    def end(self) -> None: ...

class GraphRegion:
    """A component's graphs on one device."""

    @property
    def id(self) -> int: ...
    @property
    def pool(self) -> tuple[int, int]: ...
    def captured(self, ns: int) -> None: ...
    def prepare(self, expected: int | None = None) -> None: ...
    def acquire(self) -> bool:
        """Before a replay: False while it is being dropped."""
    def release(self, stream: int) -> None:
        """After the replay's launch on `stream`."""
    @property
    def dropping(self) -> bool: ...
    def close(self) -> None: ...

class Allocating:
    """A thread allocating into a request's arena (`Request.allocating`)."""

    def end(self) -> None:
        """Ends it, on its thread."""

class Pass:
    def end(self, peak: int) -> None:
        """Records torch's allocated peak over the pass (absolute)."""

class Pin:
    @property
    def ptr(self) -> int: ...
    @property
    def len(self) -> int: ...
    @property
    def fresh(self) -> bool:
        """Its bytes were just copied: a set with a fill runs it, then `filled()`."""

    def filled(self) -> None: ...
    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def window(self) -> Window:
        """The same bytes holding no pin: valid only while the block is bound."""

class Window:
    """A block's bytes without a pin (`Pin.window`): its tensors are valid only while the block is
    bound, unless `keep` hands them a pin; the pin that binds it orders their consumers."""

    def keep(self, pin: Pin) -> None:
        """Its tensors hold `pin` from now on: the page stays bound while any of them lives."""

    @property
    def ptr(self) -> int: ...
    @property
    def len(self) -> int: ...
    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

class Fills:
    @property
    def bytes(self) -> int: ...
    @property
    def ns(self) -> int: ...

class Moved:
    @property
    def kept(self) -> Fills: ...
    @property
    def store(self) -> Fills:
        """The store, read through the page cache."""
    @property
    def speculative(self) -> int: ...

class Ref:
    """One bind: the page-reference trace."""

    @property
    def step(self) -> int: ...
    @property
    def set(self) -> str: ...
    @property
    def block(self) -> int: ...
    @property
    def bytes(self) -> int: ...
    @property
    def origin(self) -> Literal["hit", "kept", "store"]: ...
    @property
    def speculative(self) -> bool: ...

class StepStats:
    @property
    def capacity(self) -> int: ...
    @property
    def evicted(self) -> list[int]: ...

class Stats:
    """One request's record; `evicted` is what make_room took, by class (F, O, S, N, D, W, P)."""

    @property
    def request(self) -> str: ...
    @property
    def moved(self) -> Moved: ...
    @property
    def trace(self) -> list[Ref]: ...
    @property
    def steps(self) -> list[StepStats]: ...
    @property
    def stall_ns(self) -> int: ...
    @property
    def evicted(self) -> list[int]: ...
    @property
    def peak(self) -> int: ...
    @property
    def compute(self) -> int: ...
    @property
    def leftover(self) -> int: ...
    @property
    def make_room(self) -> list[tuple[int, str, int, int, int]]:
        """Each make_room that took bytes: (step, trigger, need, freed, release events waited on)."""
    @property
    def outlived(self) -> list[tuple[int, int, int]]:
        """Its torch blocks still live at its end (ptr, bytes, stream): the request-end audit."""

    def json(self) -> str:
        """The record as JSON, field for field (what a Runtime relays)."""

class Retained:
    """Stable virtual references. `prepare` leases their physical pages through stream completion."""
    identity: str
    def prepare(self, stream: int = 0, request: Request | None = None) -> list[Pin]: ...
    def valid(self) -> bool: ...
    def resident(self) -> bool:
        """Every block mapped and filled now: a read (no pin, no fill, no room made)."""
    def close(self) -> None: ...

class StaleBinding(VarenaError): ...

def pins_held() -> int:
    """Pins the binding holds now: Python's pin objects and the tensors that keep one."""
