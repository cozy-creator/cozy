"""The weight plane: pinned host tier, per-GPU VMM arenas, copy streams, streaming cursors.

The contract is `crates/tensorfs-plane/API.md`. `scripts/stub_conform.py` holds this stub
against the compiled `tensorfs._ext.plane`, name for name and parameter for parameter. The
TypedDicts describe returned rows only; they do not exist at runtime.
"""

from collections.abc import Sequence
from typing import Literal, Self, TypedDict

from ._ext import ReadLease, ReadPlan, Store

#: `"pinned"` or a CUDA ordinal.
type Tier = Literal["pinned"] | int

class Region(TypedDict):
    index: int
    offset: int
    nbytes: int
    span: int

class Part(TypedDict):
    what: str
    region: int
    offset: int
    nbytes: int

class Trim(TypedDict):
    freed: int
    evicted: list[tuple[str, int]]
    over_budget_unreleasable: int
    blockers: list[str]

class HostStats(TypedDict):
    budget: int
    used: int
    ready_bytes: int
    registered_bytes: int
    memfd_bytes: int
    cached_bytes: int
    direct_bytes: int
    buffered_bytes: int
    inline_bytes: int
    direct_refused: int
    map_refused: int  # always 0 since 0.3.93
    disk_read_bytes: int  # read from storage by this process, from /proc/self/io
    fills: int
    fill_bytes: int
    fill_ns: int
    evictions: int
    evicted_bytes: int
    register_failed: int
    trims: int

class DeviceStats(TypedDict):
    ordinal: int
    uuid: str
    granularity: int
    exportable: bool
    budget: int
    committed: int
    mapped: int
    idle: int
    leased_bytes: int
    held_bytes: int
    over_budget_unreleasable: int
    reserved_va: int
    copies: int
    host_copy_bytes: int
    host_copy_ns: int
    disk_copy_bytes: int
    disk_copy_ns: int
    mapped_copy_bytes: int  # always 0 since 0.3.93 (0.3.92 copied from page-cache mappings)
    mapped_copy_ns: int
    misses: int
    evictions: int
    evicted_bytes: int
    trims: int

class SetDevice(TypedDict):
    ordinal: int
    ready_bytes: int

class SetStats(TypedDict):
    id: int
    name: str
    nbytes: int
    regions: int
    host_ready_bytes: int
    host_memfd_bytes: int
    evictions: int
    evicted_bytes: int
    devices: list[SetDevice]

class StageStats(TypedDict):
    ws: str
    device: int
    tag: str
    acquires: int
    skipped: int
    misses: int
    late: int
    late_bytes: int
    stall_ns: int
    streamed_bytes: int

class RegionStats(TypedDict):
    ws: str
    device: int
    tag: str
    region: int
    uses: int
    compute_ns: int
    stall_ns: int
    late: int

class Stats(TypedDict):
    host: HostStats
    devices: list[DeviceStats]
    sets: list[SetStats]
    stages: list[StageStats]
    regions: list[RegionStats]
    events_dropped: int
    poisoned: str | None

class Event(TypedDict):
    #: fill_done, fill_failed, evicted, budget, late, register_failed
    kind: str
    ws: str | None
    region: int | None
    device: int | None
    bytes: int
    ns: int
    detail: str

class PlaneError(Exception):
    code: str

class BudgetExceeded(PlaneError):
    pool: str
    requested: int
    available: int

class Shortfall(PlaneError):
    pool: str
    need: int
    available: int
    blockers: list[str]

class BelowFloor(PlaneError):
    pool: str
    need: int
    budget: int

class LeaseViolation(PlaneError): ...
class Poisoned(PlaneError): ...
class CudaError(PlaneError): ...
class CudaUnavailable(PlaneError): ...
class Invalid(PlaneError): ...
class IoError(PlaneError): ...
class Closed(PlaneError): ...

class Ticket:
    def done(self) -> bool: ...
    def wait(self) -> None: ...

class Lease:
    @property
    def ptr(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def region(self) -> int: ...
    @property
    def ring_offset(self) -> int | None: ...
    def view(self) -> DeviceView: ...
    def verify_bytes(self) -> list[str]: ...
    def release(self, stream: int | None = None) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *_args: object) -> bool: ...

class DeviceView:
    @property
    def stale(self) -> bool: ...
    def __dlpack__(
        self,
        *,
        stream: object = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

class Cursor:
    @property
    def position(self) -> int: ...
    @property
    def passes(self) -> int: ...
    @property
    def home(self) -> list[bool]: ...
    @property
    def window(self) -> int: ...
    @property
    def ring_ptr(self) -> int: ...
    @property
    def ring_nbytes(self) -> int: ...
    def acquire(self, region: int, stream: int) -> Lease: ...
    def ring_view(self) -> DeviceView: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *_args: object) -> bool: ...

class Source:
    """A manifest's byte source: one TensorFS lease shared by its weight sets."""

class WeightSet:
    @property
    def name(self) -> str: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def regions(self) -> list[Region]: ...
    @property
    def parts(self) -> list[Part]: ...
    @property
    def host_fd(self) -> int: ...
    def view(self, device: int) -> DeviceView: ...
    def verify(self, device: int) -> int: ...
    def close(self) -> None: ...

class Plane:
    def __init__(
        self,
        devices: Sequence[int],
        readers: int = 8,
        copy_streams: int = 1,
        slab_bytes: int = 64 << 20,
        staging_buffers: int = 4,
        staging_bytes: int = 64 << 20,
        direct_io: bool = True,
    ) -> None: ...
    @property
    def devices(self) -> list[tuple[int, str]]: ...
    def source(self, store: Store, lease: ReadLease) -> Source: ...
    def register(
        self,
        name: str,
        source: Source,
        plan: ReadPlan,
        regions: Sequence[Sequence[str]],
        host_fd: int | None = None,
    ) -> WeightSet: ...
    def set_vram_budget(self, device: int, nbytes: int) -> Trim: ...
    def set_pinned_budget(self, nbytes: int) -> Trim: ...
    def want(
        self,
        ws: WeightSet,
        tier: Tier,
        regions: Sequence[int] | None = None,
        priority: int = 0,
        pin: bool = False,
    ) -> Ticket: ...
    def drop(self, ws: WeightSet, tier: Tier, regions: Sequence[int] | None = None) -> int: ...
    def prioritise(
        self,
        ws: WeightSet,
        tier: Tier,
        regions: Sequence[int] | None = None,
        priority: int = 0,
        pin: bool | None = None,
    ) -> None: ...
    def replicate(
        self,
        ws: WeightSet,
        devices: Sequence[int],
        regions: Sequence[int] | None = None,
        priority: int = 0,
        pin: bool = False,
    ) -> list[Ticket]: ...
    def acquire(
        self, ws: WeightSet, device: int, region: int, stream: int, priority: int | None = None
    ) -> Lease: ...
    def stream(
        self,
        ws: WeightSet,
        device: int,
        order: Sequence[int],
        window: int,
        repeat: int = 1,
        priority: int = 0,
        tag: str = "denoise",
        ring_bytes: int | None = None,
    ) -> Cursor: ...
    def stats(self) -> Stats: ...
    def events(self) -> list[Event]: ...
    def close(self) -> None: ...

def hold_tier(fd: int) -> int:
    """Claim every region of the pinned-tier memfd `fd` for a process without a plane (the
    Worker): returns a new descriptor. While it is held no plane punches the tier's RAM. End it
    with `release_tier`."""

def release_tier(hold: int) -> int:
    """End a `hold_tier` claim (closes `hold`): punches every Ready region no plane claims and
    returns the bytes punched."""
