# tensorfs.varena — the weight plane (v4, 2026-10-18)

Mechanism with a public API: TensorD, the Runtime and anyone with a `Store` are its consumers.
There is no budget and no eviction policy. The plane maps when asked, until the driver refuses or
the floor would be crossed (`DeviceFull`), and it unmaps only on `drop`. Only a live lease pins.

```python
store = tensorfs.v2.Store("~/.tensorfs"); ck = store.checkpoint("acme/sdxl@1")
p = tensorfs.varena.Plane(devices=[0])
ws = p.register(ck, tensorfs.v2.regions(ck, max_bytes=64 << 20))   # host=PageCache() by default
p.want(ws, 0).wait()
with p.acquire(ws, 0, region=0, stream=torch.cuda.current_stream().cuda_stream) as l:
    x = torch.from_dlpack(l.view())          # uint8 bytes of region 0
```

## Units

- **Region**: a union of whole chunks of one component, in the header's order
  (`tensorfs.v2.regions(ck, max_bytes, boundaries=None)`). Chunks sharing a tensor stay together;
  neighbours merge while they total at most `max_bytes`, and a tensor larger than that is a region
  alone. `boundaries` are `<component>/<key prefix>` module paths a region never spans unless one
  chunk does (the Runtime passes its blocks).
- **Layout**: one byte layout per weight set, the same in a sealed host layout and in device VA.
  Regions start on 2 MiB; each chunk sits at its own 4 KiB offset, so parts keep their in-chunk
  offsets (cozytensors/2 aligns parts to 4 KiB). `ws.regions`, `ws.parts` give the offsets; a
  part's bytes are `lease.view()[part.offset - region.offset : ... + nbytes]`.
- **Stream**: a raw `CUstream` as `int` (`torch.cuda.current_stream().cuda_stream`).

## Open and register

```python
p = Plane(devices, readers=16, copy_streams=1, staging_buffers=16, staging_bytes=16 << 20,
          floor_bytes=256 << 20)
ws = p.register(ck, regions, host=PageCache(direct=True))   # or SealedLayout(fd)
ws.nbytes, ws.digest, ws.regions, ws.parts; ws.close()
```

- `PageCache(direct=True)`: chunk files through the plane's pinned staging slots, one read and one
  device copy per slot, many in flight. A chunk the page cache holds whole is copied from it; any
  other is read with O_DIRECT (`direct=False`: buffered, so it stays in the page cache).
- `SealedLayout(fd)`: an owner's sealed memfd of this exact layout (`varena::host::open_sealed`,
  checked by `ws.digest`). A Ready region copies from it in one DMA (page-locked on first copy,
  bounced through the staging slots where the driver refuses); a hole reads the chunk files,
  buffered. The plane owns no pinned weight RAM beyond its staging slots.
- `floor_bytes`: device memory the plane never maps into (the driver's own room, a display's).

## Residency

```python
t = p.want(ws, device, regions=None)       # map + fill -> Ticket; t.done(), t.wait()
p.drop(ws, device, regions=None) -> freed  # unmap and release to the driver; leased regions stay
l = p.acquire(ws, device, region, stream)  # an unmapped region is filled first (a miss)
l.view() -> DLPack; l.release(stream=None) # release fails only if repeated
c = p.stream(ws, device, regions, ring_bytes=None, repeat=1)   # repeat=0: cycle until close
c.acquire(region, stream); c.home; c.ring_ptr; c.ring_nbytes; c.passes; c.close()
```

- `want` maps all or nothing: `DeviceFull(deficit)` (free `deficit` bytes and ask again) or
  `Floor(need, capacity)` (it can never fit) leave nothing mapped.
- `drop` gives each region's physical memory back to the driver at once (no idle pool). A region
  still filling is unmapped once its copy lands. A cursor serving a region in place does not keep
  it: its next acquire refills it.
- A cursor serves regions mapped at `stream()` in place; the rest stream through one ring of
  `ring_bytes` (default twice the largest streamed region). Copies run ahead as far as the ring
  has room; each waits on the release events of exactly the bytes it overwrites. A consumer that
  departs from the schedule skips ahead within a cycle, or gets a demand acquire.
- `release` records an event on the stream. Without one it synchronizes the stream
  (`release_synced`); when that fails too the bytes are never served again as they are
  (`release_unconfirmed`). A lease collected unreleased poisons the plane.

## Facts

```python
p.stats() -> {"devices": [{ordinal, uuid, granularity, mapped, leased, copies, misses,
                           host_copy_bytes/ns, bounce_copy_bytes/ns, disk_copy_bytes/ns}],
              "regions": [{ws, device, region, bytes, state, live, last_use}],   # mapped only
              "io": {direct_bytes, cached_bytes, buffered_bytes, direct_refused, register_failed},
              "now", "events_dropped", "poisoned"}
p.events() -> [{kind, ws, region, device, bytes, detail}]   # drains
```

`last_use` and `now` are plane-clock nanoseconds: a fact for the caller choosing what to drop,
not a priority. Event kinds: `fill_failed`, `release_synced`, `release_unconfirmed`,
`prefetch_failed`.

## Errors

`PlaneError` (`.code`) and its subclasses: `DeviceFull` (`.deficit`), `Floor` (`.need`,
`.capacity`), `CudaError`, `Closed`, `Invalid`, `Poisoned`, `IoError`. Capacity never poisons. A
failed copy is reported only after its stream drained; one that cannot drain is `Poisoned`, and a
poisoned plane never unmaps (its copies may still touch its ranges) until it is collected after
its devices ran all queued work.
