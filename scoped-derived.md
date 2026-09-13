# Scoped derived writers (tfs-116 / tfs-115)

Implementation owner: Codex `/root/offline_owner`.
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
