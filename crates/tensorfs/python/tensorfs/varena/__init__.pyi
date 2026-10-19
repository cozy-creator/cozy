"""varena's core: one process's model-weight memory (design memory-varena §2, §5.1).

`Gpu` maps each weight set's regions in VMM chunks at a home VA. Regions outside the home set
(the watermark) stream through 2-3 scratch slots. A bound region is a `Pinned`, a uint8 DLPack
tensor whose deleter releases the pin with an event on the bind stream, so a pin lasts exactly as
long as any tensor made from it. Eviction unmaps only unpinned pages whose fill and release
events completed. Copies run on varena's own stream; the bind stream waits on an event and the
host never syncs. `Host` is the CPU core: the same sets, as host memory, and it never loads
libcuda.

`scripts/stub_conform.py` holds this stub against the compiled `tensorfs._ext.varena`, name for
name and parameter for parameter.
"""

from collections.abc import Sequence

from .._ext import ReadLease, ReadPlan, Store
from ..v2 import Checkpoint

class VarenaError(Exception):
    code: str

class DeviceFull(VarenaError):
    """The driver refused to map. `deficit` of the `need` bytes found no memory; nothing was
    mapped, and nothing is poisoned."""

    need: int
    deficit: int

class CudaError(VarenaError): ...
class CudaUnavailable(VarenaError): ...
class Invalid(VarenaError): ...
class IoError(VarenaError): ...

class Closed(VarenaError):
    """The weight set was closed."""

class Unsupported(VarenaError):
    """There is no such mechanism on this OS: the sealed host tier off Linux, all of varena on
    Windows."""

class Part:
    """One stored part: a plain tensor's data (`role` is `""`) or an encoded tensor's part."""

    @property
    def component(self) -> str: ...
    @property
    def key(self) -> str: ...
    @property
    def role(self) -> str: ...
    @property
    def dtype(self) -> str:
        """The v2 name, such as `"BF16"` or `"U8"`."""
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def region(self) -> int: ...
    @property
    def offset(self) -> int:
        """Bytes from the start of its region (its `Pinned`)."""
    @property
    def nbytes(self) -> int: ...

class Counters:
    """Per weight set, for one step or in total."""

    @property
    def binds(self) -> int: ...
    @property
    def faults(self) -> int:
        """Binds that found their region absent and waited for its fill."""
    @property
    def paged_in(self) -> int:
        """Bytes filled at home."""
    @property
    def streamed_bytes(self) -> int:
        """Bytes copied into scratch slots."""
    @property
    def evicted(self) -> int: ...
    @property
    def copy_ns(self) -> int:
        """Copy-stream time of the fills, GPU-timed."""
    @property
    def stall_ns(self) -> int:
        """Bind streams' waits for fills that had not landed yet, GPU-timed."""
    @property
    def map_ns(self) -> int:
        """Host time spent creating and mapping chunks."""

class SetStats:
    @property
    def name(self) -> str: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def regions(self) -> int: ...
    @property
    def home(self) -> int:
        """Bytes of the home set."""
    @property
    def resident(self) -> int:
        """Bytes mapped at home."""
    @property
    def pinned(self) -> int:
        """Bytes mapped at home that are pinned."""
    @property
    def pins(self) -> int:
        """Live pins, at home and in slots."""
    @property
    def slot_bytes(self) -> int:
        """Bytes of the slots holding this set's regions now."""
    @property
    def step(self) -> Counters:
        """The step under way."""
    @property
    def last(self) -> Counters:
        """The last finished step (`Gpu.step()`)."""
    @property
    def total(self) -> Counters: ...

class GpuStats:
    @property
    def device(self) -> int: ...
    @property
    def chunk(self) -> int: ...
    @property
    def steps(self) -> int: ...
    @property
    def mapped(self) -> int:
        """Bytes mapped, homes and slots."""
    @property
    def idle(self) -> int:
        """Chunks kept unmapped for reuse; `trim()` releases them."""
    @property
    def slots(self) -> int: ...
    @property
    def slot_bytes(self) -> int:
        """Bytes the slots map now."""
    @property
    def slot_span(self) -> int:
        """The largest streamed region's span: what a slot maps."""
    @property
    def driver_free(self) -> int: ...
    @property
    def driver_total(self) -> int: ...
    @property
    def sets(self) -> list[SetStats]: ...

class Pinned:
    """A bound region: `nbytes` uint8 at `ptr`. Through `torch.from_dlpack(pinned)`, every view
    of the tensor holds the pin. When the last one dies, the pin is released with an event on
    the bind stream."""

    @property
    def region(self) -> int: ...
    @property
    def ptr(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def home(self) -> bool:
        """At the region's home VA; False in a scratch slot."""
    @property
    def fresh(self) -> bool:
        """This bind is the first since a copy put the bytes there. Run a fill transform (the
        LoRA bake) on it, once per page-in. A later bind of the same bytes is not fresh."""
    @property
    def writable(self) -> bool:
        """False for a `Host` view of a sealed host layout, which is mapped read-only."""
    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

class WeightSet:
    """A checkpoint's regions on one GPU (or the host). The regions are
    `tensorfs.v2.regions(ck, max_bytes, boundaries)`."""

    @property
    def name(self) -> str: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def spans(self) -> list[int]:
        """Bytes each region maps."""
    @property
    def device(self) -> int | None:
        """The CUDA ordinal; None on the host."""
    @property
    def parts(self) -> list[Part]: ...
    def home(self, regions: Sequence[int]) -> None:
        """The watermark: these regions live at home; the rest stream through slots. A region
        left out stays mapped, and serves binds, until evicted. Default: every region. A no-op
        on the host."""
    def bind(self, region: int, stream: int = 0) -> Pinned:
        """GPU: at home, mapped and filled if absent, or in the slot that holds it. `stream`
        waits for the bytes. The host waits only for reads that feed a copy. Host: a view of
        the sealed layout, or private memory read once from the chunk files."""
    def prefetch(self, region: int) -> None:
        """Start the fill ahead of its bind, at home or into a free slot. It never waits. A
        no-op on the host."""
    def evict(self, regions: Sequence[int] | None = None) -> int:
        """Unmap these regions at home (all when None): only those unpinned whose fill and
        release events completed. Returns the bytes unmapped. The chunks go idle; past a small
        cache, they go back to the driver. Always 0 on the host."""
    def close(self) -> None:
        """Let the set go. Its VA and pages are released now, or when its last pin dies. Every
        later call raises `Closed`."""

class Gpu:
    """One per (process, device): VA, the chunk cache, a copy stream and the scratch slots."""

    def __init__(self, device: int, chunk: int = 2 << 20) -> None:
        """`chunk`: the physical page size, a multiple of 2 MiB."""
    @property
    def device(self) -> int: ...
    @property
    def chunk(self) -> int: ...
    def register(
        self,
        name: str,
        ck: Checkpoint,
        regions: Sequence[Sequence[str]],
        host_fd: int | None = None,
        files: Sequence[tuple[str, int, int]] | None = None,
    ) -> WeightSet:
        """`regions`: `tensorfs.v2.regions(ck, max_bytes, boundaries)`, every part of the
        components they name. `host_fd`: a sealed host layout of exactly these regions, filled
        by the owner (TensorD) with the same layout function (`seal()` here, Rust
        `varena::memory::open_checkpoint` + `host::open_sealed`). Its digest is checked; a
        layout of other regions is refused. Copies come from it, and a region its owner left a
        hole comes from the chunk files. Without `host_fd`, every copy comes from the chunk files
        through the page cache. `files`: the chunk files a grant hands over, `(sha256, length,
        fd)` each, the only ones read; without it, the checkpoint's own paths. The core
        duplicates every descriptor it keeps and closes nothing the caller passed."""
    def trim(self) -> int:
        """Idle chunks and unneeded slots go back to the driver. Returns the bytes released."""
    def step(self) -> None:
        """Ends a step: each set's `step` counters become its `last`."""
    def stats(self) -> GpuStats: ...

class Host:
    """The CPU core. It never loads libcuda."""

    def __init__(self) -> None: ...
    def register(
        self,
        name: str,
        ck: Checkpoint,
        regions: Sequence[Sequence[str]],
        host_fd: int | None = None,
        files: Sequence[tuple[str, int, int]] | None = None,
    ) -> WeightSet:
        """As `Gpu.register`. Binds are views of the sealed layout (read-only), or private
        memory read once from the chunk files."""

def hold_tier(fd: int) -> int:
    """A claim on every region of the host-tier memfd `fd`, for an owner that maps none of it
    (TensorD). Returns a new descriptor. End it with `release_tier`."""

def release_tier(hold: int) -> int:
    """End a `hold_tier` claim (closes `hold`). Punches every Ready region nobody claims and
    returns the bytes punched."""

def fill_sealed(
    store: Store,
    lease: ReadLease,
    plan: ReadPlan,
    regions: Sequence[Sequence[str]],
    name: str,
    threads: int = 8,
    reserved: int | None = None,
    staged: Sequence[int] | None = None,
) -> tuple[int, dict[str, int]]:
    """A v1 read plan's sealed host layout, filled through verified reads; CONSUMES `lease`.
    Returns the memfd and each read mode's bytes."""

def seal(
    ck: Checkpoint,
    regions: Sequence[Sequence[str]],
    name: str,
    threads: int = 8,
    files: Sequence[tuple[str, int, int]] | None = None,
) -> int:
    """`regions` of a v2 checkpoint in a sealed host layout, filled in this process from its
    chunk files (`files`, as in `register`, else the checkpoint's own paths). The result is what
    `register(host_fd=)` adopts, the same layout by construction; deposit it with TensorD as an
    opaque fd. Returns the memfd (the caller's)."""

def reserve(name: str, nbytes: int, threads: int = 8) -> int:
    """A memfd of `nbytes` whose pages are allocated now. `fill_sealed(reserved=)` consumes it."""

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
