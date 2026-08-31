# TensorFS

The current release is `0.0.3`.

TensorFS stores every domain repository as the same graph:

```text
repos/<org>/<name>.json
manifests/<aa>/<bb>/<sha256>.json
blobs/<aa>/<bb>/<sha256>
tensorfs.sqlite                    # five small local index/coordination/cache tables
checkouts/<manifest-sha256>/...    # optional disposable local view
tmp/                               # all transient local state; delete when stopped
```

Repository files are authoritative mutable release indexes. Manifests are immutable virtual file
trees. Blobs are immutable content bytes. The same engine is used independently for model,
package-source, dataset, and user-file stores; the physical domain/bucket supplies access and
erasure isolation, not a different catalog format. SQLite locally and PostgreSQL in Tensorhub are
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

Publication order is blobs, manifest, repository compare-and-swap, then released-manifest index. A crash
before repository replacement leaves harmless unreachable content. A crash afterward leaves
storage ahead of SQL; `store rebuild` catches SQL up. The database never leads the repository file.

Each model row emitted by `repo list` carries the repository's exact canonical `evidence` object.
Clients may preserve those bytes opaquely and base64-wrap them for a later `put_release`; TensorFS
remains the only parser and canonical writer. Ordinary release rows omit evidence.

Evidence is present only for a Manifest containing one typed CozyTensors header. Ordinary releases
do not fabricate model classification. A release may carry one optional `metadata` blob ref;
package repositories use it for their exact PackageRelease while datasets and user files may omit
it. Rebuild always recovers the repository, metadata, Manifest, and complete reached blob set;
model headers additionally recover the diagnostic projection emitted by `store rebuild --rows`.

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
alive and reaps only markers whose lock the kernel releases. Open operations hold exact keys.
Seven days without accepted object/byte progress only makes an operation eligible for an explicit
abandonment after its writer is gone and its grants are revoked; time never releases holds itself.

Cloud storage uses the same Repository, Manifest, ObjectRef, projection, and reachability code.
Tensorhub supplies R2 conditional writes/listing and PostgreSQL coordination; TensorFS does not add
cloud-specific catalog-root, grant, or verification documents.
