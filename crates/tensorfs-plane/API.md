# tensorfs.plane — Python API (v1, 2026-09-30)

The weight plane moves bytes; cozy-runtime decides. The plane never measures free memory,
never picks victims among equals, never reorders a schedule. It ships inside the `tensorfs`
wheel (`from tensorfs import plane`), so it takes TensorFS `Store`, `ReadLease` and
`ReadPlan` objects directly. Changes to this file are announced in its changelog below.

## Units

- **Weight set**: one component of one checkpoint (a `ReadPlan` scope), registered once.
- **Region**: a policy-declared group of tensors (a block from `paging.partition()`, plus
  `common`). The plane never infers regions.
- **Layout**: every byte has ONE offset, the same in the pinned tier (memfd), the device VA
  and inside a streaming slot. Regions start on 2 MiB; parts start on 4 KiB. A region moves in
  one copy.
- **Tier**: `"pinned"` or a CUDA ordinal (`int`).
- **Stream**: a raw `CUstream` as `int` (`torch.cuda.current_stream().cuda_stream`; 0 = legacy
  default stream).

## Open

```python
p = plane.Plane(
    devices=[0],             # CUDA ordinals; [] = host tier only (libcuda never loaded)
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
ws = p.register(
    name,                    # content id for telemetry, e.g. f"{header_sha}/{component}/{route}"
    store,                   # tensorfs.Store
    lease,                   # tensorfs.ReadLease covering the plan; CONSUMED (lease.live -> False),
                             #   the weight set releases it at ws.close()
    plan,                    # tensorfs.ReadPlan (tensorfs.plan(header, traversal, components))
    regions,                 # list[list[str]]: region r = tensor ids "component/key"; every plan
                             #   tensor in exactly one region; a region's parts keep plan order
    host_fd=None,            # adopt a memfd holding this layout (another process's pinned tier)
)
ws.name, ws.nbytes           # nbytes = layout span (multiple of 2 MiB)
ws.regions                   # [{"index", "offset", "nbytes", "span"}]
ws.parts                     # [{"what": "comp/key#role", "region", "offset", "nbytes"}]
ws.host_fd                   # the memfd (plane-owned); pass to another process to share the tier
ws.close()                   # refuses while leases, holds, cursors or views remain
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
  raises `Shortfall` (need, available, blockers).
- `"pinned"`: fill the memfd from the store (page-cache bytes copied, the rest O_DIRECT),
  page-lock it (`cuMemHostRegister`, portable) when the plane has devices.
- device: map the regions (slab-sized handles), then copy from the pinned tier if it holds the
  region, after it if it is filling, else disk → staging buffers → device.
- `replicate`: one pinned fill (if the pinned budget allows), then one copy per device.

## Resident access

```python
lease = p.acquire(ws, device, region, stream)
lease.ptr, lease.nbytes, lease.region, lease.slot   # slot is None for resident leases
lease.release()              # records the release event on the acquire stream (or pass stream=)
with p.acquire(ws, device, region, stream) as lease: ...
```

`acquire` enqueues `cuStreamWaitEvent(stream, fill_event)` when the fill is still in flight —
no host block. A region not resident is filled on demand (counted as a miss). Release exactly
once; a lease garbage-collected unreleased POISONS the weight set on that device.

## Streaming (the hot path)

```python
cur = p.stream(ws, device, order, window, repeat=1, priority=0, tag="denoise")
cur.home                     # per order index: True = served in place (resident at stream())
cur.window, cur.slot_nbytes, cur.slot_ptrs
cur.slot_view(i)             # uint8 DLPack view of slot i (keeps the cursor from closing)
for step in range(repeat):
    for r in order:
        lease = cur.acquire(r, stream)   # must be the next scheduled region, else ScheduleMismatch
        ...block forward on `stream`...  # bytes at lease.ptr (slot lease.slot, or resident)
        lease.release()
cur.close()                  # refuses while a lease is live; waits for copies in flight
```

- Regions resident on the device at `stream()` are held (not evictable) until `close()`; the
  rest stream through `window` slots of `max(streamed span)` bytes (charged to the device
  budget; may evict below `priority`).
- Copies run `window` positions ahead on the copy stream. Each waits on the release event of
  the slot's previous user; `acquire` never blocks the host once the copy is queued.
- Holding `window` streamed leases at once and asking for one more raises (window exhausted).
- To change the resident set or window mid-stage (e.g. after an OOM at a block boundary):
  release, `close()`, change budgets, `stream()` again with the remaining order.

## Views

```python
v = ws.view(device)          # uint8 DLPack, shape [ws.nbytes]: the stable VA (reserved once)
t = torch.from_dlpack(v)     # bytes valid only for resident regions under a lease
v.stale                      # True if any region was unmapped since the view was made
```

Views keep the VA alive (block `ws.close()`), never the pages.

## Telemetry (exact facts)

```python
p.stats() -> {
  "host":    {"budget", "used", "ready_bytes", "registered_bytes", "memfd_bytes",
              "cached_bytes", "direct_bytes", "buffered_bytes", "inline_bytes", "direct_refused",
              "fills", "fill_bytes", "fill_ns", "evictions", "evicted_bytes", "register_failed",
              "trims"},
  "devices": [{"ordinal", "uuid", "granularity", "exportable", "budget", "committed", "mapped",
               "idle", "leased_bytes", "held_bytes", "over_budget_unreleasable", "reserved_va",
               "copies", "host_copy_bytes", "host_copy_ns", "disk_copy_bytes", "disk_copy_ns",
               "misses", "evictions", "evicted_bytes", "trims"}],
  "sets":    [{"id", "name", "nbytes", "regions", "host_ready_bytes", "host_memfd_bytes",
               "devices": [{"ordinal", "ready_bytes", "poisoned"}]}],
  "stages":  [{"ws", "device", "tag", "acquires", "misses", "late", "late_bytes", "stall_ns",
               "streamed_bytes"}],
  "events_dropped": int,
}
p.events() -> [{"kind", "ws", "region", "device", "bytes", "ns", "detail"}]   # drains
```

- Copy GB/s = `*_copy_bytes / *_copy_ns` (GPU-timed by events around each copy).
- `late`: acquires whose copy had not completed when acquired; `stall_ns`: GPU-measured wait
  of the consumer stream on late copies plus any host wait for an unqueued copy.
- Event kinds: `fill_done`, `fill_failed`, `evicted`, `budget`, `late`, `register_failed`.

## Errors

`plane.PlaneError` (`.code`), subclasses: `BudgetExceeded`, `Shortfall`, `LeaseViolation`,
`Poisoned`, `ScheduleMismatch`, `CudaError`, `CudaUnavailable`, `Invalid`, `IoError`, `Closed`.
Source refusals surface as the usual `tensorfs.errors.*` classes.

## Changelog

- v1 2026-09-30: first surface.
