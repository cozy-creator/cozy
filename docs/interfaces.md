# TensorFS consumer interfaces

Implementation is tracked in [tracker #339](https://github.com/cozy-creator/tracker/issues/339).
TensorFS is a library used by TensorD (`cozy-machine`) and Runtime's executor processes.
TensorD embeds its Rust storage and shared-buffer mechanisms; Runtime calls its Python
facade and native weight-plane/device APIs. TensorFS has no separate daemon. A Python
installation does not grant authority over a managed execution's store.

## Consumers

| Consumer | Interface |
|---|---|
| TensorD | Makes model weights available; coordinates machine-wide memory policy, owns the managed Store, fills/retains shared CPU buffers and can retain exported GPU allocation handles through TensorFS |
| Runtime | Constructs Python/PyTorch models; parses granted sources and performs executor-side GPU allocation, copies, mapping and unmapping through `tensorfs.plane` and device APIs within TensorD's grants |
| Package code | `SourceCapability`, `OutputCapability`, `Derivation`, `DerivedTransaction` |
| Standalone library users | Explicit Store construction and normal coordinated reads/writes |
| Tensorhub | CLI metadata operations over explicitly staged files and census directories |

The word *weights* spans three representations with different lifetimes:

| Representation | Meaning and responsibility |
|---|---|
| Serialized weight bytes and metadata | Immutable Store objects plus headers/manifests describing tensor parts, encodings, configs and assets; TensorD acquires and retains managed content through TensorFS |
| Prepared CPU weight buffers | Memory layouts filled from verified bytes, including shared sealed buffers; TensorD fills/retains them through TensorFS, and Runtime maps/reads the supplied buffers and verified object files |
| GPU-resident tensors | Allocations and tensor views used by a constructed model; Runtime performs device operations through TensorFS and its framework, while TensorD supplies budgets and can retain exported allocation handles for reuse |

TensorD's source and buffer delivery makes model data available. Runtime still constructs
the model, binds its parameters and executes it. TensorD can request executor preparation
or prefetch that includes GPU work; those device operations still run in Runtime.
TensorFS implements storage, layout, read and memory operations in whichever process
calls the library; it is not another
machine-wide policy authority. See the [weight-plane contract](../crates/tensorfs-plane/API.md).

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

An `ObjectRef` is the existing `{sha256, length}` dictionary. A `derived.Source` names an
exact content manifest; neither grants access by itself. A live `SourceCapability` binds
that content to the owner's inspection and bounded-read operations. It provides actual
tensor metadata, canonical configs and part bytes, and may use an owner channel.
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

## Typed consumer contracts

[Tracker #340](https://github.com/cozy-creator/tracker/issues/340) completes the native
Python input/result descriptions in `tensorfs.records`. These are ordinary dictionary
projections and named aliases of existing tuple inputs; parameter lists, on-disk
formats, canonical bytes and Python runtime representations are unchanged.

| Consumer operations | Input/result records | Native producer |
|---|---|---|
| Standalone or owner content admission, copying, walking | `PutResult`, `VerifyResult`, `WalkObject`, `CheckoutResult` | `store.rs`, `project.rs`, `checkpoint.rs` |
| Verified acquisition and streamed reads | `ObjectKey`, `ReadItem`, `BatchItem`, `ReadReceipt`, `StreamResult` | `read.rs`, `receipt.rs` |
| Repository pointers and retained roots | `ReleaseLane`, `ReleaseRow`, `LocalRow`, `TreeRoot`, `CheckpointRoot`, `KeyedRoot` | `repository.rs`, root modules |
| Owner source preparation and acquisition | `SourceMember`, `SourceProfile`, `SourceCarrier`, `SelectedSource`, `PreparedModelSource`, `PreparedSelection`, `FetchPlan`, `PullResult`, `EnsureEvent`, `EnsureResult` | `providers`, `source_model.rs`, `ingest`, `fetch.rs`, `ensure.rs` |
| Native derived inputs | `DeclarationSources`, `DeclarationTargets`, `DeclarationConfigs`, `DeclarationFiles`, `DerivationArguments` | `derived.rs`; native Python declaration parser |
| Scoped inspection and writer recovery | `InspectedSource`, `SourceFacts`, `DerivedLookup`, `DispositionResult`, `RetainedResult` | `derived.rs`, `derived/retention.rs` |
| Store pressure and maintenance | `Relief`, `GCResult` | `reclaim.rs`, `gc.rs` |
| Document and compatibility inspection | `Header`, `ManifestDocument`, `DocumentResult`, `CapabilityResult`, `FitResult` | `header.rs`, `manifest.rs`, `corpus.rs`, `machine.rs` |

Package authors keep using the typed `Derivation` inputs in `tensorfs.derived`.
`native_arguments()` now describes the same native map/tuple shapes to the type checker.
Direct Store callers may use the declaration records explicitly. The native parser still
checks all cross-field, dtype, geometry and custody constraints; annotations do not replace it.
`Reader` describes the existing bounded `.read(size)` source accepted by native writes.

Trusted-owner `serve_derived` callbacks receive `CheckpointReceipt` and `DerivedReceipt`;
`serve_derived_replay` also passes `DerivedReceipt`. A callback that consumes only some
fields can accept `Mapping[str, object]`. Callers storing receipts without mutating them
should use that mapping type rather than require a mutable `dict[str, Any]`.

`ObjectKey` names the existing `(object_id, length)` tuple, while `ObjectRef` names the
canonical `{sha256, length}` mapping. The aliases document existing native arguments;
they are not additional storage objects or grants. Source/roster aliases document each
field order beside their definitions. `SourceInspection.from_native` takes the actual
`InspectedSource` producer shape; callers should pass native inspection results directly.

Rust consumers already use `ObjectRef`, `ReceiptFacts`, typed Store/read/derived methods
and explicit ownership. This Python contract cleanup does not add parallel Rust wrappers.
Open-ended encoding definitions and channel envelopes stay at their existing dynamic
boundary; fixed native outputs no longer erase their contract into `Any`.

## Owner-assigned writer attempts

An embedding owner that does not already possess a durable writer session uses Rust
`derived::begin_next(store, meta, transaction, declaration, checkpoint)` or
`derived::channel::Writer::begin_next(...)`. The first returns `NextBegin` containing
`writer_session_id` and the existing `Begin` holds/views. The channel constructor owns
those same guards and uses the assigned session throughout its lifetime.

TensorFS chooses the next session above the durable transaction's highest session and
any retained restore fence, under the existing transaction lock. This works after an old
owner used a large timestamp, after restart, and when two authorized attempts open
concurrently. A stale channel's cleanup cannot fence the newer session. The owner still
checks that an execution is authorized/current before opening a replacement; this API
does not authorize the replacement itself. A newer attempt fences the older writer.

The existing explicit `begin(..., writer_session, ...)` API is unchanged. No clock,
process-global counter, daemon-specific journal, Python wrapper or new on-disk format is
introduced. Only an observed row changing during native validation is retried; different
declarations, terminal states, invalid sources and exhausted integer space refuse.
Native proofs are in `derived/next_attempt_tests.rs`; TensorD adoption belongs to
[tracker #341](https://github.com/cozy-creator/tracker/issues/341).
