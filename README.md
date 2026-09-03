# TensorFS

The current release is `0.3.0`.

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

Workers may additionally use a caller-selected root (for example,
`/workspace/cozy/repo-cache`) as an optional datacenter cache. The example path is deployment
policy, not part of the TensorFS format. The cache is not a TensorFS Store and contains only
immutable repository objects:

```text
blobs/<first-2-hex>/<next-2-hex>/<64-lowercase-hex>
manifests/<first-2-hex>/<next-2-hex>/<64-lowercase-hex>.json
```

There is no SQLite file, repository projection, lease, operation, verification authority, or GC
in that cache. Every hit is checked against an exact SHA-256 and length while TensorFS copies it
directly into the ordinary local Store. Missing, corrupt,
read-only, full, and unavailable cache states are soft misses; the authenticated cloud source
remains authoritative. Cloud-success backfill uses a hidden unique sibling, fsync, and an atomic
no-clobber link, so concurrent writers converge and an interrupted write never has a final name.
An existing corrupt final entry is never overwritten: it remains a soft miss until an operator
deletes that exact cache object, after which the next successful backfill repairs it. This explicit
drop is intentional; the cache has no independent repair authority or garbage collector.

Cache calls perform synchronous filesystem I/O and must run on a worker data path or subprocess,
never its control loop. A mounted filesystem which stops responding cannot provide a timely soft
miss from inside the blocked syscall, so subprocess progress/termination and live hung-mount proof
remain the supervisor's responsibility.

The supervisor subprocess seam emits exactly one JSON record and exits successfully for cache
weather:

```bash
tfs store admit-file <store> blob|manifest <sha256> <length> <source>
tfs repo-cache admit <cache> <store> blob|manifest <sha256> <length>
tfs repo-cache backfill <cache> <store> blob|manifest <sha256> <length>
```

Three schemas are STORED — written to a path, digested, and identified by their exact bytes:

1. Repository JSON, typed by `repos/`.
2. Manifest JSON, typed by `manifests/`.
3. `cozytensors/1`, the portable CBOR tensor header, stored as a blob.

A model Manifest selects one typed CozyTensors entry and may also carry ordinary files
such as a README, license, or sample image. `Store.walk_cozytensors` and
`Store.acquire_cozytensors` cover only the selected header and its transitive model assets and
tensor parts; configs are canonical JSON bytes inside the header. The existing
`Store.walk` and `Store.acquire_manifest` retain their
full-snapshot meaning, including every ordinary sibling.

While that lease is live, `lease.read_asset(header_bytes, name, max_bytes=64<<20)` reads a
header-declared tokenizer or other model asset through the same pinned verified descriptors,
then checks its declared logical length and SHA-256. The supplied header must itself belong to
the lease.

Those three are the durable format contract. The frozen corpus also verifies the bounded
ingest, encoding-vector, and device-capability grammars that cross process boundaries:

| grammar | where it lives |
| --- | --- |
| `EncodingSpec` | nested CBOR inside the CozyTensors header |
| `EncodingVectors` | compiled-in registry fixture |
| `IngestProfile`, `IngestSubject`, `IngestSession` | the ingest border's own documents |
| `IngestVerificationReceipt`, `Stamp` | transient candidate-byte admission proof, never release metadata |

Tensor requirements are deliberately absent from that list. They are transient rows passed
directly from Runtime to TensorFS for one cold request; they are never serialized, published,
cached, hashed, or graded.

Plus one bounded operational DTO, `source-plan.json`, specified below. Naming these does not
make them stored formats; refusing to name them only made the count look smaller than the
system.

## Build and test

The native wheel is `cp312-abi3`, so CPython 3.12 is the floor and every later CPython loads
the same wheel.

`tensorfs.TensorRequirement(component, key, shape, logical_dtype=None)` describes one row of
Runtime's request-time `factory(config)` census.

## Fit

`tensorfs.fit(requirements, header, *, custody, encoded_leaves, device=None,
observations=None)` compares those in-memory rows with one `cozytensors/1` header. There is
no CLI or document form.

| code | rule |
| --- | --- |
| `ok` | routes counted: `verbatim` (plain), `decoded_float`, `encoded_gemm` |
| `component_missing` | (1) every required component exists; extra components are ignored |
| `tensor_missing`, `shape_mismatch` | (2) every constructed key exists at the same logical shape |
| `dtype_mismatch` | (3) an optional logical dtype constraint disagrees |
| `extra_tensor` | (4) a stored key inside a constructed component absent from the requirements — refused on `canonical` custody, listed and ignored on `local` |
| `encoding_unsupported` | (5) the package refuses encoded leaves |
| `encoding_unqualified` | (6) with `--device`, every encoded leaf has a qualification record for that class (plain storage has nothing to qualify) |

`tensorfs.capability(encoding, device, observations=None)` answers the device axis alone.
TensorFS has no compatibility grade or search plane.

Each named construction config is one bounded RFC 8785 JSON-object byte string stored directly in
the CozyTensors header. Header parsing refuses duplicate keys, trailing bytes, non-finite numbers,
non-object JSON, oversized values, and any noncanonical spelling. TensorFS stores no separate
config objects or aggregate config digest.

TensorFS 0.3.0 intentionally has no pre-0.3.0 header reader. Existing 0.1.2 roots migrate
directly, and intermediate 0.2.x roots use the same standalone tool:

```bash
uv run tools/migrate-to-0.3.0.py /var/lib/cozy/store \
  --tfs /path/to/tensorfs-0.3.0/tfs
uv run tools/migrate-to-0.3.0.py /var/lib/cozy/store \
  --tfs /path/to/tensorfs-0.3.0/tfs --apply
```

It stages and validates all new headers, Manifests, and repositories before changing a root;
repository documents remain under `migration-backup-0.3.0/`, and the prior SQLite catalog remains
as `tensorfs.sqlite.pre-0.3.0.bak`.

Everything below is what CI runs, in the same order.

```bash
export PYO3_PYTHON="$(uv python find 3.12)"
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
python3 scripts/fence.py
python3 scripts/version_contract.py source

cargo build --release --bin tfs
./target/release/tfs vectors verify
uv run --python 3.12 --project conformance/cbor/python \
  python scripts/cbor-conformance.py --tfs target/release/tfs conform

# the end-to-end test: wheel -> fresh venv -> real store -> lease/plan/ring/CAS
bash scripts/wheel-smoke.sh
```

The CBOR gate runs the production Rust codec and an independent Python codec over the same
green and red vectors.

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
authoritative path immediately; TensorFS retains no forensic copy. A writer holds an advisory lock
on its admission temp for the temp's whole life; `tfs reap` removes only the temps whose lock the
kernel hands out, never by age, so a slow live writer is never reaped from under.

## Repository publication

```text
tfs manifest build <entries.jsonl> --out <manifest.json>
tfs manifest inspect <manifest.json> --refs <refs.jsonl>
tfs manifest get <store> <manifest_id> --out <manifest.json>
tfs manifest admit <store> <manifest.json> --expect <sha256> [--expect-length <bytes>]
tfs repo inspect <repo.json> --rows <rows.jsonl>
tfs repo list <store> --rows <rows.jsonl>
tfs repo usage <store> --rows <rows.jsonl>
tfs repo get <store> <org> <name> --out <current.json>
tfs repo apply <current.json|-> <mutation.json> \
  --manifest <manifest.json> --header <header.cbor> --inventory <blobs.jsonl> \
  --out <replacement.json> --rows <rows.jsonl>
tfs repo commit <store> <current.json|-> <mutation.json>

tfs local resolve <store> <name>
tfs local replace <store> <name> <source-sha256> <manifest> <length> \
  --observed <repository-sha256|absent>
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

`repo list` emits one `kind: "release"` row per release lane and one `kind: "local"` row per local
alias (a checkpoint with a `source_selection` and no release). Repository rows carry only durable
snapshot identity and release state; historical ingest evidence is not part of the repository.

`repo usage` emits one `kind: "repo"` row per repository — `bytes_total`, the union of blobs its
retained checkpoints reach through the same closure GC keeps alive (manifest entries plus the
header's assets and tensor segments), and `bytes_unique`, the part no other repository
reaches, i.e. what deleting only it would free — then one `kind: "store"` row with the referenced
union (`bytes_total`), `bytes_unique_sum`, `bytes_unreferenced` (verified blobs nothing retained
reaches) and `repos`. Manifests are not counted; a blob counts once per repository however many
paths list it.

Rebuild always recovers the repository, Manifest, and complete reached blob set;
model headers additionally recover the diagnostic projection emitted by `store rebuild --rows`.

## Reviewed source planning and ingest

Foreign carrier paths never name components. A review act may bank one whole-source profile beside
the existing fingerprint entries; the profile supplies the exact required component set, target
encodings, and canonical construction traversal for each reviewed key-set variant. The registry may
bank several exact tensor-schema observations for one `(key set, component)`; the observed header still
has to match one of them exactly. This lets one SDXL profile cover reviewed FP16, BF16, and mixed
component tensor schemas without 16 source-specific code paths. An optional canonical provider
member/slot binds a component only when bytes cannot distinguish it, such as byte-identical H3 DiT
headers. A reviewed member prefix may admit interchangeable objects from one provider namespace;
the generic SDXL profile uses `civitai/files/`. Omitting both bindings remains local-only, so a
local profile cannot steal a provider candidate:

```text
tfs ingest bank <dialect> <converter> \
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

`tfs ingest run` spawns the conversion as an isolated worker under address-space and file-size
rlimits and no CPU-time limit: a CPU clock ends legitimate work at a fixed input size. The worker
announces every byte it moves on its stdout, and the run ends it only on observation — no byte
moved, no CPU tick, never seen running or in I/O across a window derived from the run's own
worst silence (floored at the catalog's declared lock wait), or CPU burning with no byte moved
past what the run's own rate says the whole input costs. Nothing about the run's duration is
consulted.
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
)
# prepared = {replayed, sources:[{slot, profile, manifest_digest, manifest_length}]}
store.release_model_source(operation_id)
```

Inputs are sorted exact profile and verified-file rows. TensorFS binds the supervisor-verified
ObjectRefs into operation identity and re-fences each regular non-symlink path by device, inode,
length, and change token before and after conversion. It deliberately does not perform a second
whole-file SHA pass: the existing streaming ingest pass hashes every admitted tensor byte into the
CAS. All profiles prepare under one operation; only a complete sorted result set becomes replayable.
Exact replay returns the same Manifest bytes; changed profile or file facts conflict, and explicit
release removes only operation holds. Access URLs, credentials, provenance, repository names, and
catalog release identity have no argument or durable field.

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
tfs gc <store> [--dry-run] [--json]
tfs gc headers <census> --out <header-requests.jsonl>
tfs gc holds <store> --out <holds.jsonl>
tfs gc plan <census> --holds <holds.jsonl> --out <plan.jsonl>
```

`tfs gc <store>` is the local reclamation act: under the store's exclusive lock it marks
`repositories -> manifests -> blobs` from the filesystem alone, keeps what an open ingest session
root (`tmp/ingest/<session>/session.json`) names, deletes every other blob and manifest, and
repairs the verified-blob rows so no row outlives its object. `tensorfs.sqlite` is never consulted
for liveness — the database is not the source of truth; the filesystem is self-describing. It
prints `reclaimed <bytes> B in <n> objects, kept <bytes> B in <m> held by <k> ingest sessions`
(`--json` for a program), is idempotent, and converges after a crash (an object is unlinked before
its row is dropped). A live writer or read lease refuses `STORE_BUSY` naming the holder; a session
whose writer is gone is abandoned only by the explicit `tfs ingest reap`. `--dry-run` plans without
deleting; `gc holds` + `gc plan` are the same plan as separate documents, which is how a cloud
caller (tensorhub) supplies its own holds against a staged census.

Inventory and hold JSONL rows contain only `{key,length}` and `{kind,key,length}`; the digest is
derived from the canonical domain-local key. Any malformed metadata, missing referenced byte,
length mismatch, incomplete/out-of-order inventory, or failed fence makes the pass delete nothing.

All temporary admission files, ingest candidates, writer markers, and read-lease markers live under
`tmp/`. Local writers and read leases share the recovery lock there; `gc` and `store rebuild` take
it exclusively, refuse while any holder is alive, and reap only markers whose lock the kernel
releases. Time never releases a hold.

Cloud storage uses the same Repository, Manifest, ObjectRef, projection, and reachability code.
Tensorhub supplies R2 conditional writes/listing and PostgreSQL coordination; TensorFS does not add
cloud-specific catalog-root, grant, or verification documents.
