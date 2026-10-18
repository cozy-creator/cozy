"""The weight plane: pinned host tier, per-GPU VMM arenas, copy streams, streaming cursors.

TensorD coordinates machine-wide budgets and shared-buffer/handle retention. Runtime
constructs models and performs GPU allocation, copies and mappings through this local
library within those grants. A prepared CPU layout is distinct from a GPU tensor.

The contract is `crates/varena/API.md`. `scripts/stub_conform.py` holds this stub
against the compiled `tensorfs._ext.varena`, name for name and parameter for parameter. The
TypedDicts describe returned rows only; they do not exist at runtime.
"""

from collections.abc import Sequence
from typing import Literal, Self, TypedDict

from ._ext import ReadLease, ReadPlan, Store
from .v2 import Checkpoint

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
    sealed_bytes: int  # sealed layouts this plane maps: their owner's RAM
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
    shared_bytes: int  # mapped shared regions, outside committed and the budget
    copies: int
    host_copy_bytes: int  # one DMA from page-locked host memory
    host_copy_ns: int
    bounce_copy_bytes: int  # host memory that could not be page-locked, via the staging buffers
    bounce_copy_ns: int
    disk_copy_bytes: int  # the object files, via the staging buffers
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
    shared_bytes: int

class SharedRegion(TypedDict):
    region: int
    chunks: list[tuple[int, int]]  # (bytes, fd) in VA order; the caller owns each fd

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
class Unsupported(PlaneError):
    """No such mechanism on this OS: the host tier off Linux, the whole plane on Windows."""

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
    @property
    def digest(self) -> str: ...
    def view(self, device: int) -> DeviceView: ...
    def verify(self, device: int) -> int: ...
    def close(self) -> None: ...

class Plane:
    def __init__(
        self,
        devices: Sequence[int],
        readers: int = 16,
        copy_streams: int = 1,
        slab_bytes: int = 64 << 20,
        staging_buffers: int = 32,
        staging_bytes: int = 1 << 20,
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
    def register_checkpoint(
        self,
        name: str,
        ck: Checkpoint,
        regions: Sequence[Sequence[str]] | None = None,
        host_fd: int | None = None,
    ) -> WeightSet:
        """Register `regions` (`tensorfs.v2.regions(ck)` when None) of a v2 checkpoint: its
        header, in order, is the read plan and its chunk files the source."""
    def register_sealed(
        self,
        name: str,
        plan: ReadPlan,
        regions: Sequence[Sequence[str]],
        fd: int,
        files: Sequence[tuple[str, int, int]] | None = None,
    ) -> WeightSet:
        """Adopt the owner's shared CPU layout (`fill_sealed`, possibly still filling),
        using each region once Ready. This opens no Store and neither fills nor reclaims
        that CPU layout. Regions not staged there copy from the granted `files`
        (`(sha256, length, fd)` per verified object file). GPU operations run in this plane."""
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
    def device_sharing(self, device: int) -> bool: ...
    def export_device(
        self, ws: WeightSet, device: int, regions: Sequence[int] | None = None
    ) -> list[SharedRegion]: ...
    def attach_device(
        self,
        ws: WeightSet,
        device: int,
        region: int,
        chunks: Sequence[tuple[int, int]],
        digest: str,
    ) -> None: ...
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

class SealedFill(TypedDict):
    cached_bytes: int
    direct_bytes: int
    buffered_bytes: int
    inline_bytes: int

#: `Plane.register_sealed` adopts a layout still filling (absent in older releases).
SEALED_WHILE_FILLING: bool
#: `Plane.register_sealed` takes a layout its machine staged in part, the rest from `files`.
SEALED_HOLES: bool

def fill_sealed(
    store: Store,
    lease: ReadLease,
    plan: ReadPlan,
    regions: Sequence[Sequence[str]],
    name: str,
    threads: int = 8,
    reserved: int | None = None,
    staged: Sequence[int] | None = None,
) -> tuple[int, SealedFill]:
    """Prepare a shared CPU layout through verified reads and seal it; CONSUMES `lease`.
    This is an owner-side byte operation, not Python model construction or a GPU transfer.
    Returns the sealed memfd (the caller's) and each read mode's bytes. `staged`: only those
    regions, the rest holes an adopter reads from object files."""

def reserve(name: str, nbytes: int, threads: int = 8) -> int:
    """A memfd of `nbytes` whose pages are allocated now; `fill_sealed(reserved=)` consumes it."""

# ------------------------------------------------------------ the per-GPU ledger (§5.6)
# One memfd table per GPU, one row per process holding bytes on it. A row's lock dies with its
# process; dead rows count as held until reaped and returned. Bytes, and `last_use` in
# CLOCK_MONOTONIC ns (`time.monotonic_ns()`).

class LedgerError(Exception):
    code: str

class Victim:
    """A row asked (or, from `admit`, to ask) for `bytes`."""

    @property
    def slot(self) -> int: ...
    @property
    def pid(self) -> int: ...
    @property
    def bytes(self) -> int: ...

class Admit:
    """`now`; or `victims` whose reclaimable bytes cover the gap, and `short`: what no ask frees."""

    @property
    def now(self) -> bool: ...
    @property
    def victims(self) -> list[Victim]: ...
    @property
    def short(self) -> int: ...

class RowFacts:
    @property
    def slot(self) -> int: ...
    @property
    def gen(self) -> int: ...
    @property
    def pid(self) -> int: ...
    @property
    def state(self) -> Literal["live", "dead", "reaping"]: ...
    @property
    def context(self) -> int: ...
    @property
    def resident(self) -> int: ...
    @property
    def cold(self) -> int: ...
    @property
    def kept(self) -> int: ...
    @property
    def pinned(self) -> int: ...
    @property
    def torch_reserved(self) -> int: ...
    @property
    def torch_allocated(self) -> int: ...
    @property
    def cache(self) -> int: ...
    @property
    def idle(self) -> int: ...
    @property
    def active_peak(self) -> int: ...
    @property
    def last_use(self) -> int: ...
    @property
    def want(self) -> int: ...
    @property
    def acked(self) -> int: ...
    @property
    def progress(self) -> int: ...
    @property
    def held(self) -> int: ...
    @property
    def unevictable(self) -> int: ...
    @property
    def reclaimable(self) -> int: ...

class Row:
    """This process's row: it alone publishes into it and answers asks of it."""

    @property
    def id(self) -> int: ...
    @property
    def pid(self) -> int: ...
    def publish(
        self,
        *,
        context: int = 0,
        resident: int = 0,
        cold: int = 0,
        kept: int = 0,
        pinned: int = 0,
        torch_reserved: int = 0,
        torch_allocated: int = 0,
        cache: int = 0,
        idle: int = 0,
        active_peak: int = 0,
        last_use: int = 0,
    ) -> None: ...
    def want(self) -> int: ...
    def wait_want(self) -> int: ...
    def ack(self, freed: int) -> None: ...

class Ledger:
    @staticmethod
    def create(gpu_uuid: str, capacity: int) -> Ledger: ...
    @staticmethod
    def adopt(fd: int) -> Ledger: ...
    @property
    def fd(self) -> int: ...
    @property
    def gpu_uuid(self) -> str: ...
    @property
    def capacity(self) -> int: ...
    def claim(self) -> Row: ...
    def ask(self, need: int, asker: int | None = None) -> list[Victim]: ...
    def answered(self, victim: Victim) -> bool: ...
    def wait(self, victim: Victim) -> None: ...
    def admit(self, need: int) -> Admit: ...
    def reap(self, pid: int) -> int: ...
    def returned(self, pid: int) -> None: ...
    def stalled(self) -> list[int]: ...
    def snapshot(self) -> list[RowFacts]: ...
