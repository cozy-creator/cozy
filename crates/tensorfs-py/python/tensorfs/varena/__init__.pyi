"""varena's core: one process's model-weight memory (design memory-varena §2, §5.1).

`Gpu` maps each weight set's regions in VMM chunks at a home VA. Regions outside the home set
(the watermark) stream through 2-3 scratch slots. A bound region is a `Pinned`, a uint8 DLPack
tensor whose deleter releases the pin with an event on the bind stream, so a pin lasts exactly as
long as any tensor made from it. Eviction unmaps only unpinned pages whose fill and release
events completed. Copies run on varena's own stream; the bind stream waits on an event and the
host never syncs. `Host` is the CPU core: the same sets, as host memory, and it never loads
libcuda.
"""

from collections.abc import Sequence
from typing import Literal

from .._ext import Checkpoint

def regions(
    ck: Checkpoint,
    max_bytes: int = ...,
    boundaries: Sequence[str] | None = None,
    only: Sequence[str] | None = None,
) -> list[list[str]]:
    """`<component>/<key>` regions: chunk-sharing tensors together, up to `max_bytes` each, never
    across a `<component>/<key prefix>` boundary (a block) unless a shared chunk does. `only`
    (blocks or tensors, in the same form) limits the result to them and tensors sharing their chunks."""

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
    `tensorfs.varena.regions(ck, max_bytes, boundaries)`."""

    @property
    def name(self) -> str: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def spans(self) -> list[int]:
        """Bytes each region maps."""
    @property
    def regions(self) -> list[list[str]]:
        """The regions it was registered with (`Gpu.adopt`: those its sealed layout carries)."""
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
    def resident(self, region: int) -> bool:
        """Mapped at home, its bytes there or on their way (a region only in a slot is not).
        Always True on the host."""
    def warm(self, warm: bool) -> None:
        """Whether its model is warm (loaded at GPU level, the default): another process's ask
        takes idle sets' pages before warm ones'."""
    def busy(self, busy: bool) -> None:
        """Whether a forward of its model is in flight (from its first forward's start to its
        last one's end): another process's ask never takes a busy set's pages."""
    def shareable(self) -> bool:
        """Home regions map as one shareable allocation each from now on, so their pages can be
        kept past this process (`kept`); False where the device cannot share them."""
    def kept(self) -> list[tuple[int, int, int]]:
        """Landed home regions not handed out yet: `(region, span, fd)`, each fd a container of
        the region's pages for TensorD to hold, the caller's to close. Unmapping a region here
        frees it at once; only this process's death leaves its pages to TensorD."""
    def kept_quiescent(self) -> list[tuple[int, int, int]]:
        """Export landed read-only weights after admission stops and all active uses drain.
        Cached graph address pins stay intact; the caller owns the returned container fds."""

    def adopt_pages(self, pages: Sequence[tuple[int, int]]) -> int:
        """Home regions `(region, fd)` mapped from pages a dead process kept: imported, no copy.
        Each fd is duplicated; the caller closes its own. A page that will not map fills as
        usual. Returns the bytes adopted."""
    def evict(self, regions: Sequence[int] | None = None) -> int:
        """Unmap these regions at home (all when None): only those unpinned whose fill and
        release events completed. Returns the bytes unmapped. The chunks go idle; past a small
        cache, they go back to the driver. Always 0 on the host."""
    def close(self) -> None:
        """Let the set go. Its VA and pages are released now, or when its last pin dies. Every
        later call raises `Closed`."""

class Gpu:
    """One per (process, device): VA, the chunk cache, a copy stream and the scratch slots."""

    def __init__(self, device: int, chunk: int = 64 << 20) -> None:
        """`chunk`: the physical allocation size, a multiple of 2 MiB; a full device maps its last
        pages 2 MiB at a time. 2 MiB chunks map at about 30 GB/s, below an H2D ceiling."""
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
    ) -> WeightSet:
        """`regions`: `tensorfs.varena.regions(ck, max_bytes, boundaries, only)`; the set is the parts
        they name and nothing else of the checkpoint. `host_fd`: a sealed host layout of exactly
        these regions (`seal()`). Its digest is checked; a layout of other regions is refused.
        Copies come from it, and a region left a hole comes from the checkpoint's chunks, read
        by TensorFS (fetched when absent). The core duplicates every descriptor it keeps and
        closes nothing the caller passed."""
    def adopt(
        self,
        name: str,
        ck: Checkpoint,
        host_fd: int,
    ) -> WeightSet:
        """`register` of the sealed host layout `host_fd` with the regions it carries, before
        any module exists: `prefetch` starts its copies while the module is built. A module
        whose `regions` differ cannot use it."""
    def trim(self) -> int:
        """Idle chunks and unneeded slots go back to the driver. Returns the bytes released."""
    def give(self, nbytes: int) -> list[tuple[str, int]]:
        """Frees up to `nbytes` in the order `serve` answers asks, and returns what each gave:
        idle chunks and slots under `""`, then each weight set by name. Sets that gave nothing
        are left out."""
    def serve(self, row: Row) -> None:
        """Answers other processes' asks of `row` from now on with `give`, on a Rust thread with
        no Python: idle chunks and slots, then unpinned home pages of sets not busy, idle before
        warm, least recently bound first. Each answer moves the row's progress, so a long kernel
        is never a stall. Caches and torch's slack stay the control thread's to shed."""
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
    ) -> WeightSet:
        """As `Gpu.register`. Binds are views of the sealed layout (read-only), or private
        memory read once from the chunks."""
    def adopt(
        self,
        name: str,
        ck: Checkpoint,
        host_fd: int,
    ) -> WeightSet:
        """As `Gpu.adopt`."""

class Copy:
    """A run of weight copies: `h2d`, a GPU's copy stream from its first job to its last
    landing; `disk`, a host layout filled from chunk files. `ceiling` is the measured GB/s it
    is held against (pinned H2D on that GPU, the volume's O_DIRECT read), 0 when unmeasured."""

    @property
    def kind(self) -> str: ...
    @property
    def gpu(self) -> int: ...
    @property
    def bytes(self) -> int: ...
    @property
    def ns(self) -> int: ...
    @property
    def ceiling(self) -> float: ...
    @property
    def link(self) -> str: ...
    @property
    def busy_ns(self) -> int: ...
    @property
    def pin_ns(self) -> int: ...
    @property
    def staged(self) -> int: ...
    @property
    def cached(self) -> int: ...

def copies() -> list[Copy]:
    """Every copy run since the last call."""

def hold_tier(fd: int) -> int:
    """A claim on every region of the host-tier memfd `fd`, for an owner that maps none of it
    (TensorD). Returns a new descriptor. End it with `release_tier`."""

def release_tier(hold: int) -> int:
    """End a `hold_tier` claim (closes `hold`). Punches every Ready region nobody claims and
    returns the bytes punched."""

def seal(
    ck: Checkpoint,
    regions: Sequence[Sequence[str]],
    name: str,
    threads: int = 8,
) -> int:
    """`regions` of a checkpoint in a sealed host layout, filled in this process from its
    chunks. The result is what
    `register(host_fd=)` adopts, the same layout by construction; deposit it with TensorD as an
    opaque fd. Returns the memfd (the caller's)."""

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
    def waiting(self) -> int: ...
    @property
    def want(self) -> int: ...
    @property
    def acked(self) -> int: ...
    @property
    def progress(self) -> int: ...
    @property
    def running(self) -> list[tuple[str, int]]:
        """Its requests running now: (request id, remaining measured peak; -1 never measured)."""
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
        waiting: int = 0,
    ) -> None: ...
    def want(self) -> int: ...
    def wait_want(self) -> int: ...
    def ack(self, freed: int) -> None: ...
    def running(self, requests: list[tuple[str, int]]) -> None:
        """Its requests running now, at most 8: (request id of at most 64 bytes, the most it
        still needs beyond what this process holds, from its measured profile; -1 when never
        measured). Replaces the last list; written at pass ends and request ends."""

class Ledger:
    @staticmethod
    def create(gpu_uuid: str, capacity: int) -> Ledger: ...
    @staticmethod
    def adopt(fd: int, events_fd: int | None = None) -> Ledger: ...
    @property
    def fd(self) -> int: ...
    @property
    def events_fd(self) -> int | None: ...
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
    def freed(self) -> None:
        """Bytes freed on this GPU: the free generation moves and TensorD's eventfd is signalled."""
    def freed_gen(self) -> int: ...
    def wait_freed(self, seen: int) -> int:
        """Blocks until the free generation is other than `seen`; returns it."""
    def peak(self, key: str) -> int | None:
        """The measured peak recorded under `key` (`<package>|<shape>`, at most 120 bytes)."""
    def set_peak(self, key: str, nbytes: int) -> None: ...
    def dispatched(self, request_id: str) -> str | None:
        """The key TensorD dispatched `request_id` under; None: it runs unrecorded."""
    def dispatch(self, request_id: str, key: str) -> None: ...
    def undispatch(self, request_id: str) -> None: ...
    def fits(self, package: str, shape: str | None = None, spawn: bool = False) -> bool: ...
