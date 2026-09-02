# TensorFS

The current release is `0.1.3`.

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

Three schemas are STORED — written to a path, digested, and identified by their exact bytes:

1. Repository JSON, typed by `repos/`.
2. Manifest JSON, typed by `manifests/`.
3. `cozytensors/1`, the portable CBOR tensor header, stored as a blob.

A model Manifest selects one typed CozyTensors entry and may also carry ordinary files
such as a README, license, or sample image. `Store.walk_cozytensors` and
`Store.acquire_cozytensors` cover only the selected header and its transitive configs, model
assets, and tensor parts. The existing `Store.walk` and `Store.acquire_manifest` retain their
full-snapshot meaning, including every ordinary sibling.

Those three are the durable format contract. They are not the whole document inventory, and
this file used to claim they were. Nine further canonical grammars cross a process boundary
without being stored under their own path — `corpus.rs` `DOC_KINDS` enumerates them and the
frozen corpus verifies every one:

| grammar | where it lives |
| --- | --- |
| `EncodingSpec` | nested CBOR inside the CozyTensors header |
| `EncodingVectors` | compiled-in registry fixture |
| `IngestProfile`, `IngestSubject`, `IngestSession` | the ingest border's own documents |
| `IngestVerificationReceipt`, `Stamp` | ingest evidence, digested but not path-typed |
| `CodeTopology`, `CompatibilityReport` | `fit`'s code side, and the grade plane's walk record |

Four more are neither stored nor exchanged as documents but are a cross-repository contract
all the same: the `--json` output of `fit`, `compat check`, `grade` and `capability`, which
consumers shell out for and decode. They are frozen in `vectors/cases/machine/` and RE-EMITTED by
`tfs vectors verify`, so adding a key to any of them fails here, on the producer, naming the
consumer — instead of breaking it silently at runtime, which it did twice.

Plus one bounded operational DTO, `source-plan.json`, specified below. Naming these does not
make them stored formats; refusing to name them only made the count look smaller than the
system.

## Build and test

The native wheel is `cp312-abi3`, so CPython 3.12 is the floor and every later CPython loads
the same wheel.

`tensorfs.CodeTensor(component, key, dtype, shape, accepts, pinned=False)` and
`tensorfs.code_topology(contract_digest, tensors, constructed_components=None)` are the one
Python author seam for canonical CodeTopology bytes: the meta-device census of `factory(config)`,
which TensorFS validates and writes but never computes.

## Fit

`fit` is the ONE code <-> checkpoint matching function (tfs-042; design
`tracker/cross-cutting/model-code-fit.md` §3). Two inputs — a `CodeTopology` and a `cozytensors/1`
header — and exactly one of eight answers with the text a human reads:

```text
tfs fit <code-topology.json> <root> <header-id|header.cbor> [--custody canonical|local]
    [--device <class> --observations <records.json>] [--json]
```

| code | rule |
| --- | --- |
| `ok` | routes counted: `verbatim` (plain), `decoded_float`, `encoded_gemm` |
| `component_missing` | (1) every component the code constructs exists; extra components are ignored |
| `tensor_missing`, `shape_mismatch` | (2) every constructed key exists at the same logical shape |
| `dtype_pinned_mismatch` | (3) a `pinned` destination takes its dtype or a lossless widening; otherwise dtype is delivery's business |
| `extra_tensor` | (4) a stored key inside a constructed component the code does not consume — refused on `canonical` custody, listed and ignored on `local` |
| `encoding_unsupported` | (5) every stored encoding is admitted by the row's `accepts` |
| `encoding_unqualified` | (6) with `--device`, every encoded leaf has a qualification record for that class (plain storage has nothing to qualify) |

Exit 0 on `ok`, one distinct code per refusal (10..16). The same answer is
`tensorfs.fit(code_topology, header, *, custody, device=None, observations=None) -> dict`, and
`tensorfs.capability(encoding, device, observations=None)` answers the device axis alone. There is
no grade in a fit; `tfs compat check` / `tfs grade` are the search plane's ranking over the same
walk.

Every checkpoint row carries three digests, printed by `tfs checkpoint info` and `tfs repo list`:
`topology_digest` (identity), `shape_digest` (the fit index — dtype elided) and `config_digest` —
sha256 of the construction document: the config the header names, or the name-keyed map of
several, in RFC 8785 form (`jcs.rs`; the same bytes the hub's header route and cozy-runtime's
`construction_document` produce). It is absent when the header names no config. `tfs repo
migrate <store>` re-derives all three for rows written before tfs-043; `tfs repo apply` needs
`--config <path>` for every config the header names.

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
  --observed <repository-sha256|absent> --evidence <evidence.json> \
  [--expected-topology <sha256> --expected-encoding <sha256> ...]
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

`local replace` may require an exact producer output contract. The topology and one or more
repeated EncodingSpec digests are supplied together; encoding digests must already be sorted and
unique. TensorFS derives both sides from the resident Manifest, `cozytensors/1` header, and model
evidence before the Repository CAS. It accepts no encoding alias and leaves the prior alias intact
on any mismatch.

`repo list` emits one `kind: "release"` row per release lane and one `kind: "local"` row per local
alias (a checkpoint with a `source_selection` and no release). Each model row carries the
repository's exact canonical `evidence` object. Clients may preserve those bytes opaquely and
base64-wrap them for a later `put_checkpoint`; TensorFS remains the only parser and canonical
writer. Ordinary release rows omit evidence.

`repo usage` emits one `kind: "repo"` row per repository — `bytes_total`, the union of blobs its
retained checkpoints reach through the same closure GC keeps alive (manifest entries plus the
header's configs, assets and tensor segments), and `bytes_unique`, the part no other repository
reaches, i.e. what deleting only it would free — then one `kind: "store"` row with the referenced
union (`bytes_total`), `bytes_unique_sum`, `bytes_unreferenced` (verified blobs nothing retained
reaches) and `repos`. Manifests are not counted; a blob counts once per repository however many
paths list it.

A checkpoint row's `evidence` is ingest provenance — source dialect, converter build and recipe,
source digests, the verification verdict, and the three header-derived digests (`topology_digest`,
`shape_digest`, `config_digest`) — and it is informational: serving reads the manifest and the
header, never evidence. Reading is total: every evidence key is optional, unknown keys are ignored,
an absent digest answers "unknown". Only writers require more: `repo apply` refuses a CozyTensors
checkpoint row whose evidence lacks the digests or whose digests differ from the header's (the row
it authors is what the catalog serves `fit` from), and ingest provenance must name its verdict.
`repo migrate` re-derives the three digests for every row and rewrites in the current shape.
Rebuild always recovers the repository, Manifest, and complete reached blob set;
model headers additionally recover the diagnostic projection emitted by `store rebuild --rows`.

## Reviewed source planning and ingest

Foreign carrier paths never name components. A review act may bank one whole-source profile beside
the existing fingerprint entries; the profile supplies the exact required component set, target
encodings, and canonical construction traversal for each reviewed key-set variant. The registry may
bank several exact topology observations for one `(key set, component)`; the observed header still
has to match one of them exactly. This lets one SDXL profile cover reviewed FP16, BF16, and mixed
component topologies without 16 source-specific code paths. An optional canonical provider
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
