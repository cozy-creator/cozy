# Vendored worker protocol

This directory contains the Go files Cozy needs from
`cozy-creator/worker-protocol-v2`. `cozy/worker/v1/SOURCE` names the exact
commit and the sha256 of every vendored file; that file, not this prose, is the
provenance record.

The four files under `cozy/worker/v1/` are byte-identical to that commit's
`gen/go/cozy/worker/v1/` output. They were generated with official protoc 35.1,
protoc-gen-go 1.36.11, and the worker-protocol repository's pinned regeneration command.
They are copied rather than imported as another Go module. Refresh the complete set from
worker-protocol; do not hand-edit generated files here.

`cozy.worker.v1` is the wire major because it is part of the protobuf package and gRPC
service path. `WireMinor` is its additive compatibility level. Additive changes normally bump the
minor; independently negotiated optional capabilities are documented by the canonical
protocol and default to unsupported, without inferring support from a version range. A breaking change creates `cozy.worker.v2` instead of revising v1 in place.

This cohort pins wire 59 with minimum 59. Machine submissions first discover the
execution workspace through an authenticated RPC and persist that identity with
the exact submission before transmission. Every replay retains the same workspace,
even after a lost acceptance reply or client restart. A replacement journal refuses
the submission; an older unresolved submission without a workspace identity cannot
be safely upgraded by discovering a new one. Creator, Runtime, and Host must use
the coordinated workspace-fenced protocol cohort.

`SOURCE` pins the upstream commit and per-file digests, matching what tensorhub and
cozy-runtime already carry, so a hand edit or a stale re-vendor is detectable from this
repository alone. When CI has permission to read the private worker-protocol repository, its
independent `vendored-diff.sh` comparison additionally regenerates from the `.proto` and
byte-compares this complete generated set against upstream. That job is gated on the
`WORKER_PROTOCOL_TOKEN` secret; while the secret is unset it is a no-op and Cozy makes no
independent upstream-provenance or drift claim beyond `SOURCE`.

This integration retains the qualified generic LoRA adapter protocol and the complete
unpublished dependency requirement fields in one generated snapshot. Both families use
wire 59 with minimum 59; no dependency compatibility path is negotiated.
