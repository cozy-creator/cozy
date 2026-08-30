# TensorFS

TensorFS stores every domain repository as the same graph:

```text
repos/<org>/<name>.json
manifests/<aa>/<bb>/<sha256>.json
blobs/<aa>/<bb>/<sha256>
tensorfs.sqlite                    # local rebuildable projection
checkouts/<manifest-sha256>/...    # optional disposable local view
```

Repository files are authoritative mutable release indexes. Manifests are immutable virtual file
trees. Blobs are immutable content bytes. The same engine is used independently for model,
package-source, dataset, and user-file stores; the physical domain/bucket supplies access and
erasure isolation, not a different catalog format. SQLite locally and PostgreSQL in Tensorhub are
query and coordination projections that rebuild from repository files; SQL never decides GC
liveness.

TensorFS has exactly three stored schemas:

1. Repository JSON, typed by `repos/`.
2. Manifest JSON, typed by `manifests/`.
3. `cozytensors/1`, the portable CBOR tensor header stored as a blob.

Encoding definitions are nested CBOR values inside the CozyTensors header. Qualification vectors,
operation requests, receipts, verification observations, and provider grants are fixtures or SQL
rows—not additional stored document formats.

## Build and test

```bash
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check

cargo build --release --bin tfs
uv run --project conformance/cbor/python \
  python scripts/cbor-conformance.py --tfs target/release/tfs conform
uv run python scripts/fence.py
uv run python scripts/version_contract.py
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
```

Blob admission hashes while streaming, installs with no-clobber semantics, and records its
rebuildable verification observation in SQLite. Manifests use the same digest/length identity but
live under their typed namespace.

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

tfs key blob <sha256>
tfs key manifest <sha256>
tfs key repo <org> <name>
```

`manifest build` is the one ordinary-file producer. Its strictly path-sorted canonical JSONL rows
are `{blob:{length,sha256},path}`; it validates the common Manifest grammar, writes exact canonical
bytes, and prints their ObjectRef. Local and cloud callers use this seam instead of defining a
second source-tree codec.

Publication order is blobs, manifest, repository compare-and-swap, then SQL projection. A crash
before repository replacement leaves harmless unreachable content. A crash afterward leaves
storage ahead of SQL; `store rebuild` catches SQL up. The database never leads the repository file.

Each model row emitted by `repo list` carries the repository's exact canonical `evidence` object.
Clients may preserve those bytes opaquely and base64-wrap them for a later `put_release`; TensorFS
remains the only parser and canonical writer. Ordinary release rows omit evidence.

Evidence is present only for a Manifest containing one typed CozyTensors header. Ordinary releases
do not fabricate model classification. A release may carry one optional `metadata` blob ref;
package repositories use it for their exact PackageRelease while datasets and user files may omit
it. Rebuild always recovers the repository, metadata, Manifest, and complete reached blob set;
model headers additionally recover topology, component, encoding, and config projections.

`repo apply` derives full release projection rows from the exact manifest, header, and complete
blob inventory before the caller may perform its repository CAS. Missing or length-mismatched bytes
refuse publication.

## Read views

```text
tfs checkout <store> <manifest> <destination>
tfs materialize <store> <manifest> <path> <destination>
```

A checkout is a disposable relative-symlink view of ordinary manifest files. CozyTensors entries
are omitted because a header is not a conventional weight file. Materialization writes one
independent ordinary file atomically.

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

Local writers retain advisory-locked marker files, so database rebuild refuses while a writer is
alive and reaps only markers whose lock the kernel releases. Open operations hold exact keys.
Seven days without accepted object/byte progress only makes an operation eligible for an explicit
abandonment after its writer is gone and its grants are revoked; time never releases holds itself.

Cloud storage uses the same Repository, Manifest, ObjectRef, projection, and reachability code.
Tensorhub supplies R2 conditional writes/listing and PostgreSQL coordination; TensorFS does not add
cloud-specific catalog-root, grant, or verification documents.
