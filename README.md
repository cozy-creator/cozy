# TensorFS

The current release is `0.0.7`.

TensorFS stores every domain repository as the same graph:

```text
repos/<org>/<name>.json
manifests/<aa>/<bb>/<sha256>.json
blobs/<aa>/<bb>/<sha256>
tensorfs.sqlite                    # seven small local index/coordination/cache tables
checkouts/<manifest-sha256>/...    # optional disposable local view
tmp/                               # all transient local state; delete when stopped
```

Repository files are authoritative checkpoint roots plus revisioned model release pointers.
Manifests are immutable virtual file trees. Blobs are immutable content bytes. TensorFS backs model
and dataset CAS domains; package source and user media are ordinary object storage. SQLite locally
and PostgreSQL in Tensorhub are
local indexes and coordination. Repository files rebuild released-manifest lookup; open operation
holds protect in-flight writes from GC; derived transactions are resumable local work; verified
blob rows avoid rehashing unchanged files. SQL never makes repository content durable.

A writable local TensorFS root has exactly one host owner. SQLite/advisory locks coordinate
processes on that host; they are not a cross-host lock service and must not sit on NFS as shared
mutation authority. Multi-host publication and GC use Tensorhub's R2/PostgreSQL control plane.

TensorFS has exactly three stored schemas:

1. Repository JSON, typed by `repos/`.
2. Manifest JSON, typed by `manifests/`.
3. `cozytensors/1`, the portable CBOR tensor header stored as a blob.

Encoding definitions are nested CBOR values inside the CozyTensors header. Qualification vectors,
operation requests, receipts, verification observations, and provider grants are fixtures or SQL
rows—not additional stored document formats.

## Build and test

The Python package supports normal GIL-enabled CPython 3.12 only. Its native wheel is
`cp312-abi3`; other Python implementations and CPython lines are refused at the tooling and import
boundaries.

`tensorfs.CodeTensor(component, key, dtype, shape, accepts)` and
`tensorfs.code_topology(contract_digest, tensors)` are the one Python author seam for canonical
CodeTopology bytes. Callers project exact component/key fields from their own code contract;
TensorFS validates and writes them but never parses the foreign contract or guesses key prefixes.

```bash
export PYO3_PYTHON="$(uv python find 3.12)"
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check

cargo build --release --bin tfs
uv run --python 3.12 --project conformance/cbor/python \
  python scripts/cbor-conformance.py --tfs target/release/tfs conform
uv run --no-project --python 3.12 python scripts/fence.py
uv run --no-project --python 3.12 python scripts/version_contract.py source
```

The CBOR gate runs independent Rust, Go, and Python codecs over the same green and red vectors.

## Store commands

```text
tfs store init <root>
tfs store info <root>
tfs store rebuild <census> --rows <rows.jsonl> [--sqlite <tensorfs.sqlite>]

tfs put <root> <file>
tfs get <root> <sha256>
tfs contains <root> <sha256>
tfs verify <root> <sha256>
tfs scrub <root>
tfs reap <root>
```

Blob admission hashes while streaming, installs with no-clobber semantics, and records its
rebuildable verification observation in SQLite. Manifests use the same digest/length identity but
live under their typed namespace. A blob or Manifest that fails an exact rehash is removed from its
authoritative path immediately; TensorFS retains no forensic copy.

## Repository publication

```text
tfs manifest build <entries.jsonl> --out <manifest.json>
tfs manifest inspect <manifest.json> --refs <refs.jsonl>
tfs manifest get <store> <manifest_id> --out <manifest.json>
tfs manifest admit <store> <manifest.json> --expect <sha256> [--expect-length <bytes>]
tfs repo inspect <repo.json> --rows <rows.jsonl>
tfs repo list <store> --rows <rows.jsonl>
tfs repo get <store> <org> <name> --out <current.json>
tfs repo apply <current.json|-> <mutation.json> \
  --manifest <manifest.json> --header <header.cbor> --inventory <blobs.jsonl> \
  --out <replacement.json> --rows <rows.jsonl>
tfs repo commit <store> <current.json|-> <mutation.json>

tfs local resolve <store> <name>
tfs local replace <store> <name> <source-sha256> <manifest> <length> \
  --observed <repository-sha256|absent> --evidence <evidence.json>
tfs local remove <store> <name> --observed <repository-sha256>

tfs key blob <sha256>
tfs key manifest <sha256>
tfs key repo <org> <name>
```

`manifest build` is the one ordinary-file producer. Its strictly path-sorted canonical JSONL rows
are `{blob:{length,sha256},path}`; it validates the common Manifest grammar, writes exact canonical
bytes, and prints their ObjectRef. Local and cloud callers use this seam instead of defining a
second source-tree codec.

Publication order is blobs, manifest, checkpoint-root compare-and-swap, then an optional release
compare-and-swap and public-checkpoint index. `put_checkpoint` retains one exact Manifest ObjectRef.
`update_release` adds/repoints supplied lanes, removes explicit lanes, preserves omissions, and
increments one release revision. First publication permanently marks the checkpoint public;
`remove_checkpoint` therefore accepts only never-public, unreferenced checkpoints. A crash
afterward leaves a durable unreleased checkpoint; `store rebuild` catches SQL projections up. The
database never leads the repository file.

Each model row emitted by `repo list` carries the repository's exact canonical `evidence` object.
Clients may preserve those bytes opaquely and base64-wrap them for a later `put_checkpoint`; TensorFS
remains the only parser and canonical writer. Ordinary release rows omit evidence.

Evidence is present only for a Manifest containing one typed CozyTensors header. Ordinary files do
not fabricate model classification. Rebuild always recovers the repository, Manifest, and complete
reached blob set;
model headers additionally recover the diagnostic projection emitted by `store rebuild --rows`.

## Reviewed source planning and ingest

Foreign carrier paths never name components. A review act may bank one whole-source profile beside
the existing fingerprint entries; the profile supplies the exact required component set, target
encodings, and canonical construction traversal. An optional canonical provider member/slot binds a
component only when bytes cannot distinguish it, such as byte-identical H3 DiT headers:

```text
tfs ingest bank <dialect> <family> <variant> <converter> \
  --source <component>=<review-carrier>... --registry <registry.json> \
  --source-profile <name> --target <component>=<encoding>... \
  [--member <component>=<canonical-provider-member>]... --order <order.json>

tfs ingest source-plan --carrier [<reviewed-member>=]<path>... \
  [--source-profile <reviewed-name>] --out <source-plan.json>
tfs ingest plan --source-plan <source-plan.json>
tfs ingest run <store> --source-plan <source-plan.json>
```

`source-plan.json` is a bounded operational DTO, not a stored document kind. Its exact canonical
JSON fields are:

```text
carriers             [{path, member?}]
construction_order   [[component, canonical-output-key], ...]
converter            reviewed converter name
profile              reviewed source-profile name
registry             builtin:fingerprint-registry/1, or an operator-test path
registry_sha256      digest of the exact reviewed registry bytes
session              deterministic ingest session
sources              [{component, path, projected, source_member?}]
target               component=encoding,...
```

The reviewed product registry is compiled into `tfs`; users do not configure or download it. The
explicit `--registry` flag remains only for platform review and tests. A provider may present a
superset of candidate headers. TensorFS selects one exact reviewed subset and refuses if zero or
multiple profile assignments survive, so adapters never choose a variant by filename. A reviewed
production may name its exact required source profile; this narrows only the registry profile, while
the same header/member proof still has to match it exactly.

The CLI and Python facade call the same `tensorfs-core::ingest::source` planner/executor. They do not
trust or reinterpret asserted fields: the core re-reads carrier headers and the exact registry,
reproduces the unique source plan byte-for-byte, then executes its sources, target, order, and
session through the ordinary streaming ingest/CAS pass.
Editing a plan, changing carrier bytes or registry bytes, omitting an input, or supplying an
unreviewed member refuses. Unique keysets need no member. Equal headers with no reviewed member
admit multiple assignments and refuse; a provider adapter may supply a member only when its own
pinned metadata binds that exact member to the downloaded path.

Pod Runtime uses the compiled facade directly—never the CLI or a subprocess:

```python
prepared = store.prepare_model_source(
    operation_id,
    source_selection_digest,
    [("dits", "hf/minimax-h3/native-dual-bf16/1"),
     ("shared", "hf/minimax-h3/shared-bf16/1")],
    [(member, object_id, length, verified_path), ...],
    "hf://MiniMaxAI/MiniMax-H3@<commit>",
    declared_license,
)
# prepared = {replayed, sources:[{slot, profile, manifest_digest, manifest_length,
#                                 checkpoint_evidence_canonical_bytes}]}
store.release_model_source(operation_id)
```

Inputs are sorted exact profile and verified-file rows. TensorFS binds the supervisor-verified
ObjectRefs into operation identity and re-fences each regular non-symlink path by device, inode,
length, and change token before and after conversion. It deliberately does not perform a second
whole-file SHA pass: the existing streaming ingest pass hashes every admitted tensor byte into the
CAS. All profiles prepare under one operation; only a complete sorted result set becomes replayable.
Each row carries the canonical `CheckpointEvidence/1` bytes bound to its header topology,
converter recipe, credential-free source URI/license, and approved producer. Exact replay returns
the same Manifest and evidence bytes; changed profile/file/provenance facts conflict, and explicit
release removes only operation holds. Access URLs, credentials, repository names, and catalog
release identity have no argument or durable field.

`repo apply` validates the exact manifest, header, and complete blob inventory before the caller may
perform its repository CAS, then records only `(org,name,manifest_sha256)` in SQLite. Missing or
length-mismatched bytes refuse publication. Full repo/release/object rows remain diagnostic rebuild
output, not duplicated database tables.

`local` is the narrow Creator alias API over the same Repository bytes. Resolve emits one compact
canonical JSON row. Replace atomically swaps the complete `repos/local/<name>.json` document to
exactly one hidden `(source-selection, local) -> Manifest` release after verifying its closure;
stale observed repository digests refuse. Remove deletes the complete alias. Ordinary repository
mutations cannot append a second row under org `local`.

## Read views

```text
tfs checkout <store> <manifest> <destination>
tfs materialize <store> <manifest> <path> <destination>
```

A checkout is a faithful disposable relative-symlink view of every manifest entry, including the
validated typed CozyTensors header. `--no-symlink` writes small read-only copies instead. Tensor
payload segments remain CAS-only and are never duplicated into the checkout. Materialization
writes one independent ordinary file atomically and continues to refuse CozyTensors entries.

Inspect a checked-out header with `tfs cbor check <checkout>/model.cozytensors` (or decode its
diagnostic projection with `tfs cbor decode <path>`). Python readers use
`tensorfs.parse_header(path.read_bytes())`. The file is canonical `cozytensors/1` CBOR containing
CAS ObjectRefs; it is not a self-contained weights file, so tensor payload reads still require the
original TensorFS store.

## GC and recovery

```text
tfs gc headers <census> --out <header-requests.jsonl>
tfs gc holds <store> --out <holds.jsonl>
tfs gc plan <census> --holds <holds.jsonl> --out <plan.jsonl>
```

GC marks `repositories -> manifests -> blobs`, adds exact operation/read holds, and plans deletion
of everything else. Inventory and hold JSONL rows contain only `{key,length}` and
`{kind,key,length}`; the digest is derived from the canonical domain-local key. Any malformed
metadata, missing referenced byte, length mismatch, incomplete/out-of-order inventory, or failed
fence makes the pass delete nothing.

All temporary admission files, ingest candidates, writer markers, and read-lease markers live under
`tmp/`. The whole directory is disposable when no TensorFS process is running. Local writers retain
advisory-locked marker files there, so database rebuild refuses while a writer is
alive and reaps only markers whose lock the kernel releases. Open source preparations and ordinary
operations hold exact keys.
Seven days without accepted object/byte progress only makes an operation eligible for an explicit
abandonment after its writer is gone and its grants are revoked; time never releases holds itself.

Cloud storage uses the same Repository, Manifest, ObjectRef, projection, and reachability code.
Tensorhub supplies R2 conditional writes/listing and PostgreSQL coordination; TensorFS does not add
cloud-specific catalog-root, grant, or verification documents.
