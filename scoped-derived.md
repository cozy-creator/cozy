# Scoped derived writers (tfs-116 / tfs-115)

Implementation owner: Codex `/root/tensorfs_scoped_finish`, continuing the
preserved work of `/root/offline_owner`.
Worktree: `/home/fidika/cozy/.worktrees/tensorfs/tfs116-scoped-derived`.
Branch: `feat/tfs116-scoped-derived`; fresh upstream base
`49705e8294a152757ddc11ecfee4dca2810eb666`.
Design: tracker-v2 proto-054 / PR101, merged `3b72d987`.

This increment provides TensorFS-owned typed declarations and a local byte channel
for one already-open native `DerivedWriter`. The trusted execution owner binds
input/output authority, assigns its writer epoch, admits capacity and records
checkpoint/commit receipts. Package code receives a connected handle, not a Store
path or administrative Store API. TensorFS contains no Runtime imports, package
interpreter, request scheduler, credential registry or additional durable journal.

The transport must use bounded pull-driven buffers, including source reads from a
lazy output producer; it must not spool an entire output role through Runtime.
The native implementation still owns exact geometry, verification, writer fences,
partial state, leases and commit. Trusted owner hooks complete before successful
checkpoint/commit replies. A lost reply cannot remove native pending custody.

This document records implementation ownership, not completed product proof.
Runtime adapter/consumer migration is cr-174; ordinary CLI qualification is cl-265.
No existing worker, release, profile, SDK or rental is changed by the isolated work.
The source uses the unpublished `0.3.40.dev0` development identity; private test
wheels additionally identify the exact source archive and build recipe. This is
not a public release and does not mutate the published `0.3.39` identity.

## Integration contract

`tensorfs.derived.Derivation` contains typed `Source`, `Target`, `Tensor`, `Part`,
`PartSource` and `Config` values. `native_arguments(max_new_bytes)` marshals into
the unchanged native declaration/estimate/begin APIs. The owner injects
`OutputCapability(open_output)`; `tensorfs.derive(output, declaration)` invokes
that single opener. The opener freezes the exact native declaration, checks
admitted inputs/slot/limits, binds the epoch, restores custody and admits capacity.

Before constructing a declaration, algorithms can call
`SourceCapability(manifest, length, inspect_bound_source).inspect(components=...,
configs=...)`. The trusted owner binds that callback to the accepted source and
its input/component scope; the callback accepts no path, Store or different
manifest. It returns `SourceInspection.from_native(store.inspect_derived_source(...))`
through the existing execution metadata channel. The projection reuses `Tensor`
and `Part` for native logical/stored geometry and exposes canonical config bytes;
it does not parse headers, validate geometry, move payloads or open a Store. A
source capability is also a `Source` in a derivation. The output opener still
checks the exact admitted roots even if a caller constructs its own Source value.

The opener returns `DerivedTransaction(fd)` with ownership of a preconnected
AF_UNIX stream FD, or `DerivedTransaction.replay(receipt)` after validating native
committed and retained custody. Its peer serves an already-open native writer:

```python
serve_derived(
    writer, fd, operation_id=request_id, slot=output_slot, checkpoint=restore_head,
    check_current=check_attempt_authority,
    admit_io=admit_operation,
    record_checkpoint=record_native_head,
    record_receipt=record_native_receipt,
)
```

`check_current()` revalidates the owner/attempt and is called before each native
operation and every streamed pull. The Runtime owner still fences the native
writer during cancellation/replacement. The owner keeps a shutdown-capable socket
duplicate and calls `shutdown(SHUT_RDWR)` to wake blocked I/O; merely closing a
duplicate FD does not interrupt another thread's blocked syscall.
`admit_io(operation, native_bound)` returns a context manager. `part`/`config`
admission includes one bounded source-read window (at most 64 MiB); a nested read
borrows that admission instead of recursively acquiring its global lock.
Metadata operations have bound zero but still need the existing native document
allowance. Native `part_write_bound` and `config_write_bound` own payload geometry;
the existing whole-writer estimate supplies preflight before the opener returns.

`record_checkpoint(facts)` and `record_receipt(facts)` are mandatory trusted-owner
hooks, completed before success replies. A receipt hook must reconcile a real
native commit even if cancellation raced with it. The native pending root survives
a hook failure or owner-process exit; the owner reconciles it through native
lookup on restart. The channel has no durable state or automatic reconnection.

The client exposes completed roles/configs, source reads, buffer/reader additions,
checkpoint, commit and fence. The peer cannot select a transaction/epoch, open a
path, inspect arbitrary objects, release custody or invoke GC. The only nested
command during a pull is a bounded read from the same native writer's sources.
Binary pulls are capped at 1 MiB; source destinations at 64 MiB. Transform input
buffers are retained until the synchronous add call finishes. Copying a bounded
chunk does not make a complete duplicate tensor/role. No async DMA buffer ownership
is transferred by this channel; native asynchronous readers still require their
existing lease/batch-release discipline.

Native checkpoints may retain completed roles of an incomplete encoded tensor.
Such a checkpoint is recovery state, not a usable artifact. Commit refuses until
all required roles (for example FP8 data and scale) exist. Algorithms choose to
checkpoint after their semantic unit; this API does not change that policy.
