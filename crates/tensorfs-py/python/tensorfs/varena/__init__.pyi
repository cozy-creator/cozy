"""varena: one process's memory for model weights on one device (design varena-slim.md).

`Memory` places weight sets in pages its device's shared table shows free; torch's segments
for a request come from its allocator, which makes room (`make_room`: idle pages, caches, then
the request's own pages by furthest declared use) when the row's free bytes fall short.
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
    def fill(self, set: int, fill: Fill) -> None:
        """`set`'s current fill (a no-op when unchanged): pages filled under another come back
        fresh."""
    def cache(self, owner: str, shrink: Callable[[int], int]) -> int:
        """Bytes the caller holds that `shrink(want) -> freed` gives back (from varena's thread,
        never inside torch's allocator), after idle pages, least recently used first. A cache
        owned by a request id ends with it."""

    def cached(self, cache: int, nbytes: int) -> None: ...
    def hold_cache(self, cache: int) -> int:
        """Holds `cache` (never shrunk) until `release_cache`; holds nest."""

    def release_cache(self, cache: int, hold: int, stream: int = 0) -> None:
        """Ends `hold` once the work queued on `stream` so far completes."""

    def uncache(self, cache: int) -> None: ...
    def request(self, id: str, plan: list[tuple[list[str], int]], idle: bool = False) -> Request:
        """Opens request `id`; `plan` is its steps, each `(set names, passes)`. An `idle` one (a
        capture or warm-up beside others' work) takes free bytes only: it evicts nothing."""

    def retain(self, set: int, blocks: list[int]) -> Retained: ...
    def context_measurable(self) -> bool: ...
    def bind(self, set: int, block: int, stream: int = 0, evict: bool = True) -> Pin:
        """Block (region) `block` of `set`, mapped and filled; `stream` (a CUstream, 0 the default)
        waits on the fill, and tensors made from the pin are consumed on it only. With `evict`
        false it maps into free bytes only, `Exhausted` at once when they are short."""

    def free(self) -> int:
        """The row's free bytes now: its budget less its context, pages, mappings and segments."""

    def stats(self) -> Stats | None:
        """The record of the request that ended last."""

class Request:
    def pass_(self, set: int, key: str) -> Pass:
        """A pass of `set` keyed `key` (`entry|dims`): the plan's cursor moves to it."""

    def plan(self, plan: list[tuple[list[str], int]]) -> None:
        """Replaces the plan; its cursor starts over."""

    def allocating(self, memory: Memory, pool: tuple[int, int] = (0, 0)) -> None:
        """This thread's segments for torch's pool `pool` are this request's from now on: a
        shortfall makes room in `memory`."""

    def stats(self) -> Stats: ...
    def end(self) -> Stats:
        """Ends it: its pins and caches go."""

def allocated() -> None:
    """SPIKE: this thread's pass is over for torch's pool (`Request.allocating`)."""

def allocator_stats(device: int | None = None, pool: tuple[int, int] | None = None) -> tuple[int, int, int, int, int, int, int]:
    """The pool's segments: (live bytes, segments made, make_room calls, NULLs for torch's
    release, refusals, nanoseconds making segments, segments freed)."""

def allocator_retire(device: int, pool: tuple[int, int]) -> None:
    """Torch dropped `pool` (a request's end): its segments come back as torch releases them,
    and an allocation short of room first lets torch's retry release them."""

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

def allocator_pending(device: int) -> int: ...

def allocator_check(device: int) -> None: ...

def allocator_refusals(device: int | None = None) -> list[tuple[int, tuple[int, int], str, int, int]]:
    """Each device and pool's last refusal: (device, pool, reason, size, the row's free bytes)."""

def pins_held() -> int:
    """Pins the binding holds now: Python's pin objects and the tensors that keep one."""

def allocator_quarantined(device: int | None = None) -> list[tuple[int, tuple[int, int], int, str]]:
    """Quarantined segments by device and pool: (device, pool, bytes, last reason). They stay
    charged and are never reused; the device goes on."""
