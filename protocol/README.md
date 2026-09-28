# Vendored worker protocol

This directory contains the Go files Cozy needs from
`cozy-creator/worker-protocol-v2`. `cozy/worker/v1/SOURCE` names the exact
commit and the sha256 of every vendored file; that file, not this prose, is the
provenance record.

The four files under `cozy/worker/v1/` are byte-identical to that commit's
`gen/go/cozy/worker/v1/` output, generated with the worker-protocol repository's pinned
`scripts/regen.sh`. They are copied rather than imported as another Go module. Refresh the
complete set from worker-protocol; do not hand-edit generated files here.

`cozy.worker.v1` is the wire major because it is part of the protobuf package and gRPC
service path. Components release independently: the minimum is the oldest deployed minor,
never the current one, and an additive minor is used only after checking the peer's minor or
capability (a peer without a new RPC answers `UNIMPLEMENTED` for that call only). A peer
outside the range fails preparation and execution with `capability_unavailable` naming its
update; Claim, status, collection, keepalive and release still work.

`SOURCE` pins the upstream commit and per-file digests, matching what tensorhub and
cozy-runtime carry, so a hand edit or a stale re-vendor is detectable from this repository
alone. Upstream's `scripts/vendored-diff.sh` regenerates from the `.proto` and byte-compares
this set when CI can read the worker-protocol repository.
