"""The weight plane: a checkpoint's chunks as regions on per-GPU VMM ranges, filled from chunk
files or a sealed host layout, streamed through rings, pinned only by leases. No budgets: it
maps until the driver refuses or the floor would be crossed (`DeviceFull`), and unmaps on `drop`.

The contract is `crates/varena/API.md`. `scripts/stub_conform.py` holds this stub against the
compiled `tensorfs._ext.varena`, name for name and parameter for parameter. The TypedDicts
describe returned rows only; they do not exist at runtime.
"""

from collections.abc import Sequence
from typing import Self, TypedDict

from .v2 import Checkpoint

class Region(TypedDict):
    index: int
    offset: int
    nbytes: int
    span: int
    chunks: list[int]

class Part(TypedDict):
    component: str
    key: str
    role: str  # "" for a plain tensor's data
    region: int
    offset: int
    nbytes: int

class DeviceStats(TypedDict):
    ordinal: int
    uuid: str
    granularity: int
    exportable: bool
    mapped: int
    leased: int
    shared_bytes: int
    copies: int
    misses: int
    host_copy_bytes: int
    host_copy_ns: int
    bounce_copy_bytes: int
    bounce_copy_ns: int
    disk_copy_bytes: int
    disk_copy_ns: int

class RegionStats(TypedDict):
    ws: int
    device: int
    region: int
    bytes: int
    state: str  # "filling", "ready" or "invalid"
    live: int
    last_use: int  # plane-clock ns; `Stats.now` is the clock now

class IoStats(TypedDict):
    direct_bytes: int
    cached_bytes: int
    buffered_bytes: int
    direct_refused: int
    register_failed: int

class Stats(TypedDict):
    devices: list[DeviceStats]
    regions: list[RegionStats]  # mapped regions only
    io: IoStats
    now: int
    events_dropped: int
    poisoned: str | None

class Event(TypedDict):
    kind: str
    ws: int | None
    region: int | None
    device: int | None
    bytes: int
    detail: str

class PlaneError(Exception):
    code: str

class DeviceFull(PlaneError):
    deficit: int

class Floor(PlaneError):
    need: int
    capacity: int

class CudaError(PlaneError): ...
class Closed(PlaneError): ...
class Invalid(PlaneError): ...
class Poisoned(PlaneError): ...
class IoError(PlaneError): ...

class PageCache:
    def __init__(self, direct: bool = True) -> None: ...
    @property
    def direct(self) -> bool: ...

class SealedLayout:
    def __init__(self, fd: int) -> None: ...
    @property
    def fd(self) -> int: ...

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
    def ring_ptr(self) -> int: ...
    @property
    def ring_nbytes(self) -> int: ...
    def acquire(self, region: int, stream: int) -> Lease: ...
    def close(self) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *_args: object) -> bool: ...

class WeightSet:
    @property
    def nbytes(self) -> int: ...
    @property
    def regions(self) -> list[Region]: ...
    @property
    def parts(self) -> list[Part]: ...
    @property
    def digest(self) -> str: ...
    def close(self) -> None: ...

class Plane:
    def __init__(
        self,
        devices: Sequence[int],
        readers: int = 16,
        copy_streams: int = 1,
        staging_buffers: int = 16,
        staging_bytes: int = 16 << 20,
        floor_bytes: int = 256 << 20,
    ) -> None: ...
    @property
    def devices(self) -> list[tuple[int, str]]: ...
    def register(self, ck: Checkpoint, regions: Sequence[Sequence[int]], host: PageCache | SealedLayout | None = None) -> WeightSet: ...
    def want(self, ws: WeightSet, device: int, regions: Sequence[int] | None = None) -> Ticket: ...
    def drop(self, ws: WeightSet, device: int, regions: Sequence[int] | None = None) -> int: ...
    def device_sharing(self, device: int) -> bool: ...
    def export_device(self, ws: WeightSet, device: int, regions: Sequence[int] | None = None) -> list[dict[str, object]]: ...
    def attach_device(self, ws: WeightSet, device: int, region: int, chunks: Sequence[tuple[int, int]], digest: str) -> None: ...
    def acquire(self, ws: WeightSet, device: int, region: int, stream: int) -> Lease: ...
    def stream(self, ws: WeightSet, device: int, regions: Sequence[int], ring_bytes: int | None = None, repeat: int = 1) -> Cursor: ...
    def stats(self) -> Stats: ...
    def events(self) -> list[Event]: ...
    def close(self) -> None: ...
