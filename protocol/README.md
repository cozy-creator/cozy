# protocol — copied `cozy.worker.v1` bindings

COPIED, never imported as a module (boundaries.md): these four files are byte-identical
to `worker-protocol@c9da3c4` `gen/go/cozy/worker/v1/` — schema rev 5. Rev 5 hard-cuts
`AttemptOutcomeBody/2` to `/3`, adding bounded exact job artifact-receipt references and the
durable artifact-finalize exchange. cozy-creator DIALS this contract as the record owner; the
schema is th-024/th-049's and is never edited here.

Refresh = re-copy from the worker-protocol repo. Editing a file in this directory is the
one thing that turns a shared contract into two — and `ci.yaml`'s `vendored-protocol` job
is what makes that unstateable in silence: worker-protocol regenerates from its own
`.proto` with its own pinned protoc and byte-compares against exactly these files.
`SHA256SUMS` always proves the checked-in set is the reviewed `c9da3c4` set; the live
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

Schema rev 5 keeps wire minor 0 but adds the paired finalize frame slots and changes the
canonical outcome document to `AttemptOutcomeBody/3`; the schema digest is the live-session
fence. Rev 4 changed no live wire number or document version: it only tombstoned the two retired
field numbers (`OutputManifest` 2, `TriageBundleRef` 2), moving the descriptor and schema digest.
Rev 3 before it also changed no wire number or document version; it hardcut numeric origin 4 from
`SUPERVISOR` to `WORKER` and numeric origin 7 from `COORDINATOR` to `RECORD_OWNER`, with no
aliases. The descriptor and schema digest move, so a mixed rev-2/rev-3 live session refuses
at the handshake even though already-journaled outcome bytes remain replayable.
