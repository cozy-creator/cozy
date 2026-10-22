"""varena: one process's memory for model weights on one device (design varena-slim.md).

`Memory` places weight sets in pages its device's shared table shows free; a request's passes
get their measured room before they run, and `torch_limit` is what torch's allocator may hold.
A `Pin` is a block mapped and filled: a uint8 DLPack tensor that, with every tensor made from it,
holds the block until the last one dies. Every call releases the GIL; a panic raises
`VarenaError` with code `INTERNAL`.
"""

from collections.abc import Callable
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

class Grant:
    """TensorD's descriptors (handles on Windows) for this process's memory: the table and its
    row lock's own description; ownership passes."""

    def __init__(self, table: int, row: int) -> None: ...
    @staticmethod
    def open(gpu: GpuId, limit: int | None = None) -> Grant:
        """A standalone engine's grant: `gpu`'s table by name; `limit` lowers its capacity (the
        first opener sets it, a later one can only lower it)."""

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
    def cuda(ordinal: int, grant: Grant | None = None, limit: int | None = None) -> Memory:
        """CUDA device `ordinal`'s table through TensorD's `grant`, or by name (standalone, its
        capacity lowered to `limit`). Open it before torch initializes CUDA."""

    @staticmethod
    def host(gpu: GpuId, grant: Grant | None = None, limit: int | None = None) -> Memory:
        """Host pages of `gpu.memory` bytes (CPU, MPS) on the table `gpu.uuid` names."""

    @property
    def gpu(self) -> GpuId: ...
    def load(self, set: Set, level: Level) -> int:
        """Registers `set` and brings it to `level`; its id."""

    def unload(self, set: int) -> None: ...
    def trim_with(self, trim: Callable[[], int]) -> None:
        """The allocator's trim (torch's `empty_cache`), returning the bytes it freed (called from
        varena's thread): a demand trims before it evicts anything."""

    def fill(self, set: int, fill: Fill) -> None:
        """`set`'s current fill (a no-op when unchanged): pages filled under another come back
        fresh."""
    def cache(self, owner: str, shrink: Callable[[int], int]) -> int:
        """Bytes the caller holds that `shrink(want) -> freed` gives back (from varena's thread);
        a cache owned by a request id ends with it."""

    def cached(self, cache: int, nbytes: int) -> None: ...
    def uncache(self, cache: int) -> None: ...
    def request(self, id: str, plan: list[tuple[list[str], int]]) -> Request:
        """Opens request `id`; `plan` is its steps, each `(set names, passes)`."""

    def retain(self, set: int, blocks: list[int]) -> Retained: ...
    def workspace(self, bytes: int, allocated: int, reserved: int) -> Workspace: ...
    def observe(self, allocated: int, reserved: int) -> None: ...
    def bind(self, set: int, block: int, stream: int = 0) -> Pin:
        """Block (region) `block` of `set`, mapped and filled; `stream` (a CUstream, 0 the default)
        waits on the fill, and tensors made from the pin are consumed on it only."""

    def torch_limit(self) -> int:
        """Bytes torch's allocator may hold now."""

    def stats(self) -> Stats | None:
        """The record of the request that ended last."""

class Request:
    def pass_(self, set: int, entry: str, shape: str, allocated: int, reserved: int) -> Pass:
        """A pass of `set` entered at `entry` (the module's path in the model, "" the root) at
        `shape` (its tensors' dims joined by `x`), with torch's bytes now; returns once its
        measured room is free. Rooms match on (set, entry) and interpolate over shapes."""

    def plan(self, plan: list[tuple[list[str], int]]) -> None:
        """Replaces the plan; its cursor starts over."""

    def oom(
        self,
        set: int | None,
        allocated: int,
        reserved: int,
        need: int | None = None,
        site: str | None = None,
        collective: bool = False,
        finest: int = 0,
        shape: str = "",
    ) -> Next:
        """An op of `set`'s pass (None: outside any pass) ran out of memory (`need` more bytes
        when known) at cut site `site` (None: it cannot be cut) whose op has shape key `shape`,
        cut no finer than `finest`, with torch's bytes now; a `collective` site's finer level
        waits for the next pass's agreement."""

    def parked(self, nbytes: int) -> None:
        """The bytes a `Park` answer moved to the host (0: none, and the next `oom` is Exhausted)."""

    def cut(self, set: int | None, site: str, shape: str = "") -> int:
        """The cut level of `set`'s current pass for `site` (0: whole); with no set (outside any
        pass) the request's own memo of that site at the op's shape key `shape`."""

    def allocating(self, memory: Memory, set: int | None, allocated: int, reserved: int) -> None:
        """SPIKE: this thread's segments for torch's pool are `set`'s pass's (None: outside any
        pass) from now on: a shortfall asks this request."""

    def yielding(self, family: str) -> Yielded:
        """The request waits on a call of its `family` (a child): until the guard ends it gives
        back its reserved room to that family alone; ended, it waits for its room back."""

    def stats(self) -> Stats: ...
    def end(self) -> Stats:
        """Ends it: its pins, rooms and caches go."""

def allocated() -> None:
    """SPIKE: this thread's pass is over for torch's pool (`Request.allocating`)."""

def allocator_stats() -> tuple[int, int, int, int, int, int, int]:
    """SPIKE: the pool's segments: (live bytes, segments made, shortfalls varena resolved, their
    bytes, refusals, nanoseconds making segments, segments freed)."""

def allocator_reentries() -> tuple[int, int]:
    """SPIKE: the trims and cache shrinks varena reached on a thread inside the allocator
    (answered 0, never run)."""

class Pass:
    def end(self, allocated_peak: int, slack: int, sites: list[tuple[str, int, bool, int]], allocated: int, reserved: int) -> None:
        """Records what the pass measured (torch's absolute allocated peak, the allocator's slack
        at it, each cut site's `(name, peak, collective, finest)`) and torch's bytes now;
        dropping it unended records nothing."""

    def measured(self) -> bool:
        """Whether its room is the one measured for its set, shape and cut levels."""

    def levels(self) -> list[tuple[str, int]]:
        """Its collective sites' proposed `(site, level)`, for the ranks to agree before any
        exchange."""

    def agree(self, levels: list[tuple[str, int]]) -> None:
        """The levels the ranks agreed (the maximum of their proposals): in effect for it."""

class Yielded:
    """A request waiting on a child call (`Request.yielding`): `end`, or leaving its `with`, ends
    the wait."""

    def end(self) -> None: ...
    def __enter__(self) -> Yielded: ...
    def __exit__(self, *exc: object) -> None: ...

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

class Next:
    """`Request.oom`'s answer."""

    class Retry(Next):
        def __init__(self) -> None: ...

    class Cut(Next):
        """Run the op one level finer: at `level`."""

        def __init__(self, level: int) -> None: ...
        @property
        def level(self) -> int: ...

    class Park(Next):
        """Move idle activations, up to `budget` bytes, to the host; report them with `parked`."""

        def __init__(self, budget: int) -> None: ...
        @property
        def budget(self) -> int: ...

    class Exhausted(Next):
        """`error` is the `Exhausted` to raise."""

        def __init__(self, error: object) -> None: ...
        @property
        def error(self) -> Exhausted: ...

class Fills:
    @property
    def bytes(self) -> int: ...
    @property
    def ns(self) -> int: ...

class Moved:
    @property
    def kept(self) -> Fills: ...
    @property
    def tier(self) -> Fills: ...
    @property
    def store(self) -> Fills: ...
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
    def origin(self) -> Literal["hit", "kept", "tier", "store"]: ...
    @property
    def speculative(self) -> bool: ...

class StepStats:
    @property
    def capacity(self) -> int: ...
    @property
    def w(self) -> int: ...
    @property
    def evicted(self) -> list[int]: ...

class Stats:
    """One request's record; `evicted` is by class (F, O, S, N, D, W, P)."""

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
    def evicted_lowered(self) -> int: ...
    @property
    def speculative_evicted_d(self) -> int: ...
    @property
    def retries(self) -> int: ...
    @property
    def cuts(self) -> int: ...
    @property
    def parks(self) -> int: ...
    @property
    def parked(self) -> int: ...
    @property
    def lowered(self) -> bool: ...
    @property
    def peak(self) -> int: ...
    @property
    def leftover(self) -> int: ...
    def json(self) -> str:
        """The record as JSON, field for field (what a Runtime relays)."""

class Retained:
    """Stable virtual references. `prepare` leases their physical pages through stream completion."""
    identity: str
    def prepare(self, stream: int = 0) -> list[Pin]: ...
    def valid(self) -> bool: ...
    def close(self) -> None: ...

class StaleBinding(VarenaError): ...

class Workspace:
    def __enter__(self) -> None: ...
    def __exit__(self, ty: object, value: object, trace: object) -> None: ...
