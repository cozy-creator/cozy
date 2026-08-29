# Vendored worker protocol

This directory contains the Go files Cozy needs from
`cozy-creator/worker-protocol-v2` commit
`6fdaa950e2cf1d003460f84b8a110bf5f389a065`.

The four files under `cozy/worker/v1/` are byte-identical to that commit's
`gen/go/cozy/worker/v1/` output. They were generated with official protoc 35.1,
protoc-gen-go 1.36.11, and the worker-protocol repository's pinned regeneration command.
They are copied rather than imported as another Go module. Refresh the complete set from
worker-protocol; do not hand-edit generated files here.

`cozy.worker.v1` is the wire major because it is part of the protobuf package and gRPC
service path. `WireMinor` is its additive compatibility level. Additive changes bump the
minor; a breaking change creates `cozy.worker.v2` instead of revising v1 in place.

Git records the exact vendored bytes and every local change to them. A checksum file committed
beside the files would merely hash one part of the same commit and would add no provenance.
When CI has permission to read the private worker-protocol repository, its independent
`vendored-diff.sh` comparison checks this complete generated set against upstream. Without that
credential, Cozy makes no independent upstream-provenance or drift claim.
