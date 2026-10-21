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
    """`code`: EXHAUSTED, INVALID, DEVICE, IO, CLOSED or INTERNAL; `sticky`: the device context
    is gone and the process must end."""

    code: str
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
    """TensorD's descriptors (handles on Windows) for this process's memory; ownership passes."""

    def __init__(self, table: int, events: int) -> None: ...

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
    def cuda(ordinal: int, grant: Grant | None = None) -> Memory:
        """CUDA device `ordinal`'s table through TensorD's `grant`, or by name (standalone). Open
        it before torch initializes CUDA: the backend measures this process's context then."""

    @staticmethod
    def host(gpu: GpuId, grant: Grant | None = None) -> Memory:
        """Host pages of `gpu.memory` bytes (CPU, MPS) on the table `gpu.uuid` names."""

    @property
    def gpu(self) -> GpuId: ...
    def load(self, set: Set, level: Level) -> int:
        """Registers `set` and brings it to `level`; its id."""

    def unload(self, set: int) -> None: ...
    def cache(self, owner: str, shrink: Callable[[int], int]) -> int:
        """Bytes the caller holds that `shrink(want) -> freed` gives back (from varena's thread);
        a cache owned by a request id ends with it."""

    def cached(self, cache: int, nbytes: int) -> None: ...
    def uncache(self, cache: int) -> None: ...
    def request(self, id: str, plan: list[tuple[list[str], int]]) -> Request:
        """Opens request `id`; `plan` is its steps, each `(set names, passes)`."""

    def bind(self, set: int, block: int, stream: int = 0) -> Pin:
        """Block (region) `block` of `set`, mapped and filled; `stream` (a CUstream, 0 the default)
        waits on the fill, and tensors made from the pin are consumed on it only."""

    def torch_limit(self) -> int:
        """Bytes torch's allocator may hold now."""

    def stats(self) -> Stats | None:
        """The record of the request that ended last."""

class Request:
    def pass_(self, set: int, shape: str, allocated: int, reserved: int) -> Pass:
        """A pass of `set` at `shape` (dims joined by `x`) with torch's bytes now; returns once its
        measured room is free."""

    def oom(self, need: int | None = None, site: str | None = None) -> Next:
        """An op ran out of memory (`need` more bytes when known) at cut site `site` (None: it
        cannot be cut)."""

    def cut(self, site: str) -> int:
        """The current pass's cut level for `site` (0: whole)."""

    def stats(self) -> Stats: ...
    def end(self) -> Stats:
        """Ends it: its pins, rooms and caches go."""

class Pass:
    def end(self, allocated_peak: int, slack: int, context: int, sites: list[tuple[str, int]]) -> None:
        """Records what the pass measured; dropping it unended records nothing."""

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
        def __init__(self) -> None: ...

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
    def lowered(self) -> bool: ...
    @property
    def peak(self) -> int: ...
    @property
    def leftover(self) -> int: ...
