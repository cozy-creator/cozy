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

This consumer cohort pins the official `4ae21c2` generated snapshot at wire 52.
It includes machine execution and publication authority references, together with
mixed-input and model-materialization recovery capabilities. Runtime-owned execution
without publication requires actual worker minor 51; publication requires actual
Runtime minor 52 and the exact accepted authority ID.

`SOURCE` pins the upstream commit and per-file digests, matching what tensorhub and
cozy-runtime already carry, so a hand edit or a stale re-vendor is detectable from this
repository alone. When CI has permission to read the private worker-protocol repository, its
independent `vendored-diff.sh` comparison additionally regenerates from the `.proto` and
byte-compares this complete generated set against upstream. That job is gated on the
`WORKER_PROTOCOL_TOKEN` secret; while the secret is unset it is a no-op and Cozy makes no
independent upstream-provenance or drift claim beyond `SOURCE`.
