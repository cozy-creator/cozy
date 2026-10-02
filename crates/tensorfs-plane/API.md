# tensorfs.plane — Python API (v3.2, 2026-10-02)

The weight plane moves bytes; cozy-runtime decides. The plane never measures free memory,
never picks victims among equals, never reorders a schedule. It ships inside the `tensorfs`
wheel (`from tensorfs import plane`), so it takes TensorFS `Store`, `ReadLease` and
`ReadPlan` objects directly. Changes to this file are announced in its changelog below.

## Units

- **Weight set**: one component of one checkpoint (a `ReadPlan` scope), registered once.
- **Region**: a policy-declared group of tensors (a block from `paging.partition()`, plus
  `common`). The plane never infers regions.
- **Layout**: every byte has ONE offset, the same in the pinned tier (memfd), the device VA
  and inside the streaming ring. Regions start on 2 MiB; parts start on 4 KiB. A region moves in
  one copy.
- **Tier**: `"pinned"` or a CUDA ordinal (`int`).
- **Stream**: a raw `CUstream` as `int` (`torch.cuda.current_stream().cuda_stream`; 0 = legacy
  default stream).

## Open

```python
p = plane.Plane(
    devices=[0],             # required: CUDA ordinals; [] = host tier only (libcuda never loaded)
    readers=8,               # persistent reader threads
    copy_streams=1,          # copy streams (mover threads) per device
    slab_bytes=64 << 20,     # physical VRAM handle size (multiple of 2 MiB); 0 = one per region
    staging_buffers=4,       # pinned bounce buffers per copy stream (disk -> VRAM fills)
    staging_bytes=64 << 20,
    direct_io=True,          # pinned-tier fills: page-cache bytes by memcpy, the rest O_DIRECT
)
p.devices                    # [(ordinal, "GPU-uuid"), ...]
p.close()                    # closes every weight set; refuses while leases/cursors/views live
```

Budgets start at 0: set them before `want`.

## Register

```python
src = p.source(store, lease) # tensorfs.Store + tensorfs.ReadLease; the lease is CONSUMED
                             #   (lease.live -> False) and ends when the last weight set using
                             #   the source closes and `src` is dropped: one lease per manifest
ws = p.register(
    name,                    # content id; every stats/events row names the weight set by it
    src,                     # a plane.Source (shared by the manifest's components)
    plan,                    # tensorfs.ReadPlan (tensorfs.plan(header, traversal, components))
    regions,                 # list[list[str]]: region r = tensor ids "component/key" (all roles)
                             #   or exact parts "component/key#role"; every plan part in exactly
                             #   one region; a region's parts keep plan order; no empty region
    host_fd=None,            # adopt a memfd holding this layout (another process's pinned tier);
                             #   every region already Ready in it joins this plane's books at
                             #   once (claimed, counted in `used`, lowest priority)
)
ws.name, ws.nbytes           # nbytes = layout span (multiple of 2 MiB)
ws.regions                   # [{"index", "offset", "nbytes", "span"}]
ws.parts                     # [{"what": "comp/key#role", "region", "offset", "nbytes"}]
ws.host_fd                   # the memfd (plane-owned, an unlocked description); pass it to another
                             #   process to share the tier
ws.close()                   # refuses while leases, holds, cursors or views remain; punches the
                             #   pinned RAM no other process claims (see Host tier across processes)
```

Binding a part: `base[part.offset - region.offset : ... + nbytes]` of the region's bytes,
`.view(dtype).view(shape)`; offsets are 4 KiB aligned so any dtype view is legal.

## Budgets (hard ceilings)

```python
trim = p.set_vram_budget(device, nbytes)
trim = p.set_pinned_budget(nbytes)
# trim = {"freed": int, "evicted": [(ws_name, region)], "over_budget_unreleasable": int,
#         "blockers": ["ws#region@dev0", ...]}
```

Lowering a budget releases idle handles, then evicts unpinned, unleased, unheld regions in
eviction order until under the ceiling. What cannot be released is reported, never forced.

**Eviction order** (a mechanical rule over the policy's inputs): ascending `priority`; ties by
later-registered weight set first; then higher region index first (the tail of the address
order). Pinned (`pin=True`) regions are never evicted implicitly — only by `drop`.

## Residency intent

```python
t = p.want(ws, tier, regions=None, priority=0, pin=False)   # regions None = all
t.done() -> bool
t.wait()                     # host wait: condvar for host steps, then each copy's own event
p.drop(ws, tier, regions=None) -> int   # bytes freed; refuses on live leases / holds
ts = p.replicate(ws, devices, regions=None, priority=0, pin=False)  # [Ticket]
```

- `want` may evict only strictly lower-priority, unpinned, unleased regions; otherwise it
  raises `Shortfall` (need, available, blockers: dropping or re-ranking them would help).
  A request larger than the whole budget raises `BelowFloor` (need, budget: nothing helps).
- `"pinned"`: fill the memfd from the store (page-cache bytes copied, the rest O_DIRECT),
  page-lock it (`cuMemHostRegister`, portable) when the plane has devices. **Registration
  serializes this process's CUDA driver** (G1: ~0.1 ms per MiB; other threads' kernel launches
  wait meanwhile; unregister on eviction ~0.03 ms/MiB): want the pinned tier at admission or
  between stages, never while this process computes. A host-only plane registers nothing.
- device: map the regions (slab-sized handles), then copy from the pinned tier if it holds the
  region; else, for a `want` of bytes the page cache holds, straight from read-only mappings
  of their objects (no host copy, no wait for a pinned fill, which then runs beside it;
  counted in `host_copy_bytes` and, as its part, `mapped_copy_bytes`); else after a pinned
  fill that is running (one disk read
  serves both); else disk → staging buffers → device. A demand miss (`acquire`) and a
  streaming ring never map. An object that cannot be mapped (not ext4, xfs, btrfs, tmpfs or
  overlayfs; not at its declared length; pages gone from the cache) sends its region through
  staging (`map_refused`). A mapping's fault is a signal, not an error: a disk read error on a
  page evicted between the check and the copy kills the process with SIGBUS. The cache query
  runs on the caller before the fill is queued and opens each object of the wanted regions
  (a record lookup, a rehash when the record is missing): a device `want` can block on a
  first open.
- `replicate`: one pinned fill (if the pinned budget allows), then one copy per device.
- A `want` of regions already resident only updates their priority/pin: no bytes move and
  nothing is evicted. To re-prioritise without any chance of a fill, use:

```python
p.prioritise(ws, tier, regions=None, priority=0, pin=None)   # never moves a byte
```

## Resident access

```python
lease = p.acquire(ws, device, region, stream, priority=None)
lease.ptr, lease.nbytes, lease.region, lease.ring_offset   # ring_offset is None for resident leases
lease.view()                 # ONE uint8 DLPack view of the region's bytes (see Views)
lease.verify_bytes()         # diagnostic: parts whose device bytes differ from the source
                             #   (waits for the copy, reads the store; debugging runs only);
                             #   Invalid after release
lease.release()              # records the release event on the acquire stream (or pass stream=)
with p.acquire(ws, device, region, stream) as lease: ...
```

`acquire` enqueues `cuStreamWaitEvent(stream, fill_event)` when the fill is still in flight —
no host block. A region not resident is filled on demand (counted as a miss) at `priority`
(when given, it becomes the region's priority first) and may evict only regions strictly
below it; when nothing can go it raises `Shortfall` (need, available, blockers). Encode LRU
among a stage's blocks as increasing priorities (e.g. a clock). Release exactly once; a lease
garbage-collected unreleased POISONS the plane.

## Streaming (the hot path)

```python
cur = p.stream(ws, device, order, window, repeat=1, priority=0, tag="denoise",
               ring_bytes=None)  # repeat=0: cycle until close
cur.home                     # per order index: True = served in place (resident at stream())
cur.window                   # most streamed positions queued ahead of the consumer
cur.ring_ptr, cur.ring_nbytes
cur.ring_view()              # uint8 DLPack view of the whole ring (keeps the cursor from closing)
cur.position, cur.passes     # positions consumed; completed cycles over `order`
for step in range(passes):
    for r in order:
        lease = cur.acquire(r, stream)
        ...block forward on `stream`...  # bytes at lease.ptr (ring_offset in the ring, or resident)
        lease.release()
cur.close()                  # refuses while a lease or ring view is live; waits for copies
```

- Regions resident (Ready, filling or queued) on the device at `stream()` are served in place
  and held (not evictable) until `close()`. The resident set can be any subset (a spread set,
  not only a prefix); the rest streams in `order`.
- Streamed regions go through a **byte ring**: one device range of `ring_bytes` (default
  `window` × the largest streamed region; at least the largest), backed by slab-sized
  handles and charged to the device budget (may evict below `priority`). Each position is
  placed right after the previous one, wrapping to 0 when it does not fit before the end.
  Regions of any size share it; the floor is the largest streamed region, not
  `window × largest`.
- Copies run up to `window` streamed positions ahead on the copy stream. Each waits on the
  release events of exactly the earlier positions whose bytes it overwrites; `acquire` never
  blocks the host once the copy is queued.
- **Departures never raise.** `acquire(r)` for a region that is not the next one: if `r` is
  scheduled later within the next `len(order)` positions, the positions before it are
  skipped (counted as `skipped`; queued bytes are handed on unread, unqueued ones take no
  ring space) and `r` is served at its position; otherwise (not in `order`, or past the end)
  it is a demand `p.acquire` (a miss when not resident) and the position does not move.
- Holding leases over the whole ring and asking for a position that does not fit raises
  (ring exhausted).
- A cursor garbage-collected while its leases or ring views are live stops prefetching and
  closes itself when the last of them is released.
- To change the resident set, ring or window mid-stage (e.g. after an OOM at a block
  boundary): release, `close()`, change budgets, `stream()` again with the remaining order.

## Views

```python
ws.verify(device) -> int     # driver truth: every chunk the books call mapped is the driver's
                             #   mapping there (a request-boundary check); a mismatch poisons
v = ws.view(device)          # uint8 DLPack, shape [ws.nbytes]: the stable VA (reserved once)
t = torch.from_dlpack(v)     # bytes valid only for resident regions under a lease
v.stale                      # True if any region was unmapped since the view was made
lv = lease.view()            # uint8 DLPack, shape [lease.nbytes], at lease.ptr (lv.stale as above)
ring = torch.from_dlpack(cur.ring_view())   # once per cursor; a streamed region's bytes are
                                            # ring[lease.ring_offset : + lease.nbytes]
```

Views keep the VA alive (a weight set's: blocks `ws.close()`; a ring's: blocks `cur.close()`),
never the pages. Part `x` of region `r` is at `ptr + x.offset - r.offset`. A resident region's
address never moves, so its typed views are built once. A streamed region's ring offset
depends on the placement: slice the one ring view per acquire (offsets repeat when regions
are equal-sized).

## Telemetry (exact facts)

```python
p.stats() -> {
  "host":    {"budget", "used", "ready_bytes", "registered_bytes", "memfd_bytes",
              "cached_bytes", "direct_bytes", "buffered_bytes", "inline_bytes", "direct_refused",
              "map_refused", "disk_read_bytes",
              "fills", "fill_bytes", "fill_ns", "evictions", "evicted_bytes", "register_failed",
              "trims"},
  "devices": [{"ordinal", "uuid", "granularity", "exportable", "budget", "committed", "mapped",
               "idle", "leased_bytes", "held_bytes", "over_budget_unreleasable", "reserved_va",
               "copies", "host_copy_bytes", "host_copy_ns", "disk_copy_bytes", "disk_copy_ns",
               "mapped_copy_bytes", "mapped_copy_ns", "misses", "evictions", "evicted_bytes",
               "trims"}],
  # host_copy_* counts every copy from host memory with no staging: the pinned tier's and the
  # mappings' (mapped_copy_* is that part of it; subtract it for the tier alone, whose rate a
  # reader of the host pair otherwise sees blended). host + disk copies are every byte that
  # landed on the device. A pinned fill served by the page cache counts in fill_bytes and
  # cached_bytes, never in disk_read_bytes.
  "sets":    [{"id", "name", "nbytes", "regions", "host_ready_bytes", "host_memfd_bytes",
               "evictions", "evicted_bytes", "devices": [{"ordinal", "ready_bytes"}]}],
  "stages":  [{"ws", "device", "tag", "acquires", "skipped", "misses", "late", "late_bytes",
               "stall_ns", "streamed_bytes"}],
  "regions": [{"ws", "device", "tag", "region", "uses", "compute_ns", "stall_ns", "late"}],
  "events_dropped": int,
  "poisoned": str | None,
}
p.events() -> [{"kind", "ws", "region", "device", "bytes", "ns", "detail"}]   # drains
```

- Copy GB/s = `*_copy_bytes / *_copy_ns` (GPU-timed by events around each copy).
- `late`: acquires whose copy had not completed when acquired; `stall_ns`: GPU-measured wait
  of the consumer stream on late copies plus any host wait for an unqueued copy.
- `regions[].compute_ns`: GPU time from the end of the acquire's wait to the release event,
  per region and stage tag (the cost model's cᵢ). Timing pairs resolve lazily: `stats()` never
  syncs the device; unfinished pairs count at a later call.
- `ws` in every row and event is the `name` given to `register`.
- Event kinds: `fill_done`, `fill_failed`, `evicted`, `budget`, `late`, `register_failed`.

## Errors

`plane.PlaneError` (`.code`), subclasses: `BudgetExceeded`, `Shortfall`, `BelowFloor`, `LeaseViolation`,
`Poisoned`, `CudaError`, `CudaUnavailable`, `Invalid`, `IoError`, `Closed`. Source refusals
surface as the usual `tensorfs.errors.*` classes. Capacity errors carry their numbers as
attributes: `BudgetExceeded` (`pool`, `requested`, `available`), `Shortfall` (`pool`, `need`,
`available`, `blockers`), `BelowFloor` (`pool`, `need`, `budget`). `BudgetExceeded` with
`pool == "device"` means the driver itself had no memory left (the budget exceeds what is
free): lower it.

**A failed copy leaves nothing in flight.** A fill or ring copy that fails after enqueuing part
of its work is reported (its ticket, `acquire`, `fill_failed`) only after the copy stream has
drained, so unmapping, dropping or closing what it touched is safe at once. A failure the
stream cannot drain is `Poisoned`.

**Poison is latched plane-wide**: an ambiguous driver outcome (a failed unmap or rollback, an
undrainable copy failure, a driver mapping that disagrees with the books) or a lease dropped
unreleased makes every later call raise `Poisoned` with the first cause; the runtime replaces
the executor. A poisoned plane never unmaps, unregisters or frees a range while it lives
(copies may still touch it): `ws.close()`, `cur.close()` and `p.close()` raise `Poisoned`.
Capacity never poisons (`BudgetExceeded`, `Shortfall` leave the plane usable).

**A plane collected without `close()`** (its handles dropped in any order, or poisoned) is
torn down when its last handle goes: its threads stop, and once every device has run all
queued work, its device ranges are unmapped, its handles released and its pinned RAM let go as
`close()` would. Only a device that cannot be synchronized keeps its ranges and the pinned
tier until the process exits. Closing is still the contract: collection order decides when
an unclosed plane ends.

## Host tier across processes

- One memfd per weight set, laid out exactly like the weight set, with a state page at its
  end: a layout digest, one Ready word and the extent of each region. That page IS the shared
  index.
- **Claims.** A process holds a shared OFD lock on a region's word for as long as the region is
  filling or ready in its books. Evicting, dropping or closing ends this process's claim; the
  last claimant punches the region's RAM. So pinned RAM is always inside some live budget, a
  closed set does not stay resident because another process kept the fd, and no process loses
  pages it still has registered. The kernel drops a dead process's claims.
- `register(..., host_fd=fd)` in another process (or a replacement executor) reopens the fd
  (its own open file description), refuses a different layout digest, and claims every region
  whose Ready word is set — registering it, reading nothing (`stats()["host"]["fill_bytes"]`
  unchanged) — at the lowest priority, counted in `used`. The digest covers the plan's parts,
  their object ranges and the region grouping: register with the same plan and regions.
- `ws.host_fd` is a separate, never-locked description: a process that keeps it (or a dup) keeps
  the memfd alive but no claim.
- **Keeping a tier without a plane** (the Worker between executors): `h = plane.hold_tier(fd)`
  claims every region (one shared lock over the whole file, on a new description); while it is
  held, no plane eviction or close punches the tier. `plane.release_tier(h)` closes it and punches
  every Ready region no plane claims, returning the bytes punched. Bound held tiers with the
  holder's own budget (`os.fstat(fd).st_blocks * 512` is the memfd's RAM).
- A host-only `Plane(devices=[])` (never loads libcuda) can fill a tier before any executor
  holds a GPU; executors adopt and register it.
- A memfd from a plane before v3.1 (state page `TFSPLN01`) is a different layout: adopting it
  raises `Invalid`; register a fresh one.

## Changelog

- v3.3 2026-10-02: `stats()["host"]["disk_read_bytes"]` exists. The plane has counted it
  since v3.1 and this file listed it, but the Python facade never carried it: a caller read 0
  whatever the disk did, so a "nothing read from disk" check on it could not fail.

- v3.3 2026-10-02 (warm loads bound by the plane's own host copy): a device `want` of bytes
  the page cache holds copies straight from mappings of their objects, beside any pinned
  fill. `host_copy_*` counts them with the pinned tier's copies (host + disk copies stay every
  byte that landed on the device); `mapped_copy_*` is their part of it. The cache query now also answers where `cachestat` cannot:
  overlayfs and a seccomp filter that refuses it. Pod-visible: there the pinned fill copies
  freshly downloaded weights from the page cache where it re-read them by O_DIRECT, so the
  tier and the cache copy coexist until the kernel reclaims the cache.

- v3.2 2026-10-02 (W3: fd and thread growth per weight set): a plane collected unclosed or
  poisoned is torn down at its last handle (threads, device memory, pinned memfds) once its
  devices are synchronized; before, its reader threads stayed and a poisoned plane kept its
  memfds for the life of the process.

- v3.1 2026-10-02 (Xid 31; W3 run 2675): a failed copy is reported only after its stream drained;
  an undrainable one poisons; a poisoned plane never unmaps (close raises `Poisoned`).
  `verify_bytes()` after release raises `Invalid`. Pinned-tier claims: close and eviction punch
  RAM no other process claims, `register(host_fd=)` claims what the memfd already holds,
  `plane.hold_tier` / `plane.release_tier` for the Worker. The state page gains region extents
  (`TFSPLN02`).

- v3 2026-09-30 (proving-cpu.md: fine regions, spread sets): streaming uses a byte ring
  instead of fixed slots. `stream(..., ring_bytes=None)`; `cur.ring_ptr`, `cur.ring_nbytes`,
  `cur.ring_view()` replace `slot_ptrs`, `slot_nbytes`, `slot_view(i)`; `lease.ring_offset`
  replaces `lease.slot`.
- v2.1 2026-09-30 (from proving-cpu.md): `BelowFloor` (a request above the whole budget);
  a dropped cursor with live leases closes itself when they are released.
- v2 2026-09-30 (W3 requests R2–R19):
  - `p.source(store, lease)` + `register(name, source, plan, regions, host_fd=None)`: one lease
    per manifest.
  - `regions` accept exact part names.
  - `acquire(..., priority=None)`; `p.prioritise(...)`.
  - `stream(repeat=0)` cycles until close; `cur.passes`.
  - Cursor departures skip ahead or demand-fill; `ScheduleMismatch` is gone.
  - `lease.view()`; per-region GPU compute/stall ns in `stats()["regions"]`; `skipped` in
    stages; per-set evictions; plane-wide poison latch.
- v1 2026-09-30: first surface.
