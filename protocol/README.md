# protocol — copied `cozy.worker.v1` bindings

COPIED, never imported as a module (boundaries.md): these four files are byte-identical
to `worker-protocol@9e754cec` `gen/go/cozy/worker/v1/` — schema rev 9. Rev 9 hard-cuts
ClaimAck/BootFailure provenance to `control_runtime_digest`: the measured image-owned control
Runtime wheel, never the image's recursively unknowable OCI digest. cozy-creator DIALS this contract as the record
owner; the schema is th-024/th-049's and is never edited here.

Refresh = re-copy from the worker-protocol repo. Editing a file in this directory is the
one thing that turns a shared contract into two — and `ci.yaml`'s `vendored-protocol` job
is what makes that unstateable in silence: worker-protocol regenerates from its own
`.proto` with its own pinned protoc and byte-compares against exactly these files.
`SHA256SUMS` always proves the checked-in set is the reviewed `9e754cec` set; the live
master comparison additionally arms when CI has a repository-scoped
`WORKER_PROTOCOL_TOKEN`. GitHub's ordinary per-repo token cannot read a private sibling,
so absence of that explicit token is a named unarmed drift check, not a fake code failure.

`wire_identity.go` carries the three answers to three different questions. `WireMinor` is
the additive-train position and does NOT move for a breaking in-place revision, so it
fences nothing; `WireSchemaRev` is the human ordinal; `SchemaDigest` is the FENCE — the
digest of the schema's own descriptor, which a stale copy cannot spell without being the
schema it names. It rides `Claim.wire_schema_digest` and is checked before any other body
field. The retired `wire_minor.go` is deleted: a constant still claiming currency beside a
fresh binding is the whole of #530-A1.

Schema rev 9 adds the four public endpoint/base compatibility coordinates while preserving
worker-measured control Runtime provenance separately from provider-verified OCI
identity; a RecordOwner requires both facts and neither substitutes for the other. Schema rev 7 makes acquisition overlap and reuse observable without moving identity,
readiness, convergence, or admission. Schema rev 6 makes the model-object-set subject part of immutable desired state while its
expiring locations remain on the independently refreshable grant lane. Schema rev 5 moves wire
minor to 1 for the paired finalize frame slots and changes the canonical
outcome document to `AttemptOutcomeBody/3`; the schema digest remains the live-session shape
fence. Rev 4 changed no live wire number or document version: it only tombstoned the two retired
field numbers (`OutputManifest` 2, `TriageBundleRef` 2), moving the descriptor and schema digest.
Rev 3 before it also changed no wire number or document version; it hardcut numeric origin 4 from
`SUPERVISOR` to `WORKER` and numeric origin 7 from `COORDINATOR` to `RECORD_OWNER`, with no
aliases. The descriptor and schema digest move, so a mixed rev-2/rev-3 live session refuses
at the handshake even though already-journaled outcome bytes remain replayable.
