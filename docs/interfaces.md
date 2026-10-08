# TensorFS consumer interfaces

Implementation is tracked in [tracker #339](https://github.com/cozy-creator/tracker/issues/339).
TensorFS remains a library embedded by TensorD (`cozy-machine`). It has no separate
daemon. A Python installation does not grant authority over a managed execution's store.

## Consumers

| Consumer | Interface |
|---|---|
| TensorD | Rust `Store::ensure_owned`, acquisition, native transactions, roots and collection |
| Runtime | Granted model sources and buffers, header parsing, read plans and `tensorfs.plane` |
| Package code | `SourceCapability`, `OutputCapability`, `Derivation`, `DerivedTransaction` |
| Standalone library users | Explicit Store construction and normal coordinated reads/writes |
| Tensorhub | CLI metadata operations over explicitly staged files and census directories |

Constructing a managed Runtime model requires the machine's existing `model_sources` offer.
Missing access refuses that operation; it cannot select `Store.open` by omission. Reusing
an existing construction keeps its held grants and does not acquire new authority. Explicit
standalone checkpoint construction is a separate Runtime API. Job source/output channels
require the existing `weights` grant, even when the installed library contains a Store.
Work without model or weights requirements remains usable without these offers.

These are operation contracts, not whole-peer version checks. They do not require equal
Runtime, TensorD, Hub or TensorFS versions, and do not sandbox arbitrary same-UID code.

## One Python distribution

The supported Python wheel includes the full native extension, typed Python facade and
`tfs` executable. Build it with `scripts/build-wheel.sh`; `--plane` is no longer an
alternate distribution under the same name. Normal builds enable the store feature and
the installed-wheel contract checks both `Store` and the CLI. Rust feature isolation
remains available to native consumers.

The wheel's contents do not decide a managed caller's authority. Installing the full
wheel must not activate a missing Runtime source callback or output channel. TensorD
links the Rust libraries directly and does not need this Python wheel to own a Store.
Previously published reduced wheels remain immutable; consolidation applies to a new
release, with Runtime's grant boundary adopted first. Source changes and local wheel
proofs do not imply release or installed/deployed adoption.

## References, capabilities and receipts

An `ObjectRef` is the existing `{sha256, length}` dictionary. A `derived.Source` names a
manifest; neither grants access. A live `SourceCapability` is bound to an accepted input
by its owner. It provides inspection and bounded reads, and may use an owner channel.
Granted sealed buffers can instead be consumed locally by the weight plane. Neither
mechanism opens an arbitrary store path when access is missing.

An `OutputCapability` opens one declared output. The executor receives a
`DerivedTransaction`, not a root path or an unrestricted native Store writer. TensorD
retains transaction authority and checks the attempt behind that channel.

`tensorfs.records` supplies typed dictionary projections of native outputs:

- `ObjectRef`: bare hexadecimal SHA-256 and length.
- `PartReceipt`: bytes written/deduplicated, whole-part digest and object references.
- `CheckpointReceipt`: resumable plan/head identity, head length, index and byte count.
- `DerivedReceipt`: transaction and declaration identity, source facts, header, manifest,
  added objects and inheritance observations.

The four types are also exported from `tensorfs` and `tensorfs.derived`, so package
authors can annotate their existing scoped interface. `DerivedSource` and `InheritObservation`
name nested records in `tensorfs.records`. A checkpoint's existing nullable prefixed
`head` field stays unchanged; this typing change introduces no new document format.
Native Store writers and scoped channels return the same ordinary dictionaries. Rust
continues to validate and canonicalize them. Python annotations neither revalidate
native documents nor discard harmless extra fields from a newer producer.

## Transaction lifetime

| Operation | Meaning |
|---|---|
| `close()` or context exit | Disconnect this handle; no implicit commit, abandonment or retention release |
| `checkpoint()` | Record resumable progress in the existing transaction |
| `fence()` | End the current writer attempt according to native fencing rules |
| `commit()` | Finalize the artifact and return native facts; another local call returns the held receipt |
| Owner retry/resume | Bind a new writer attempt to the same transaction |
| Owner abandonment | Explicitly abandon the transaction through the owner's existing operation |
| Owner retain/release | Manage independent custody of the committed result |

A receipt is not a grant or an independent retention claim. The owner validates retained
custody before giving a replay receipt to another attempt. Socket loss is not cancellation.

## Store ownership

The [owned-store contract](owned-store.md) remains unchanged. An owner claim prevents a
second owner and destructive collection that cannot see its private pins. It deliberately
permits ordinary cooperating external readers/writers. Immutable objects still install
atomically; mutable references use conditional commits. Independent transfers can run in
parallel. This interface revision introduces no global writer mutex or store migration.

## Tensorhub metadata CLI

Tensorhub owns its `tfs` installation. Its Dockerfile installs the CLI from its evaluator
environment's lock; local configuration names `verifier.tfs_path`. Runtime's tool
environment is not the Hub's dependency provider. The Hub can use a compatible CLI
release independently of the TensorFS crates embedded in TensorD.

The Hub uses `key`, `manifest build/inspect`, `repo inspect/apply`, `cbor decode`, staged
`store rebuild`, and `gc headers/plan`. Inputs and outputs are explicitly named files
or existing structured stdout fields. Typed failures use `REFUSED <CODE>: <detail>`
on stderr and a nonzero exit. These operations run on the Hub's own scratch/census;
they do not acquire a worker's managed Store or connect to TensorD. The Hub supplies
object-storage operations, database coordination and actual cloud deletion.

Keep each operation's format/result compatibility independent of deployment identity.
Administrative commands that do mutate a local Store still obey its ownership and
collector rules; metadata planning on a separate census is not such a mutation.
