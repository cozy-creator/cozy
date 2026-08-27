# protocol — copied `cozy.worker.v1` bindings

COPIED, never imported as a module (boundaries.md): these three files are byte-identical
to `worker-protocol@b0243af` `gen/go/cozy/worker/v1/` — schema rev 3, the actor-vocabulary
hardcut. cozy-creator DIALS this contract as the record owner; the schema is th-024's and is never
edited here.

Refresh = re-copy from the worker-protocol repo. Editing a file in this directory is the
one thing that turns a shared contract into two — and `ci.yaml`'s `vendored-protocol` job
is what makes that unstateable in silence: worker-protocol regenerates from its own
`.proto` with its own pinned protoc and byte-compares against exactly these files.
`SHA256SUMS` always proves the checked-in set is the reviewed `b0243af` set; the live
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

Schema rev 3 changes no wire number or document version. It hardcuts numeric origin 4 from
`SUPERVISOR` to `WORKER` and numeric origin 7 from `COORDINATOR` to `RECORD_OWNER`, with no
aliases. The descriptor and schema digest move, so a mixed rev-2/rev-3 live session refuses
at the handshake even though already-journaled outcome bytes remain replayable.
