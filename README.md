# TensorFS

The [consumer interfaces](docs/interfaces.md) define Python, Rust and CLI ownership,
the full Python+CLI distribution, typed transaction results and their lifetimes.

TensorFS is a library used by both TensorD and Runtime, with no separate TensorFS daemon.
TensorD makes model weights available to Runtime: it coordinates machine-wide memory
policy, uses TensorFS to acquire serialized weight bytes and metadata, and fills and
retains shared CPU weight buffers. It can also retain exported GPU allocation handles.
Runtime constructs Python/PyTorch models and performs GPU allocation, copies, mapping and
unmapping through TensorFS's weight-plane/device APIs, within TensorD's grants. Stored
weight bytes, prepared CPU buffers and GPU-resident tensors are distinct representations;
retaining their bytes does not construct or execute a model.

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
local indexes and coordination. Open operation
holds protect in-flight writes from GC; derived transactions are resumable local work; verified
blob rows avoid rehashing unchanged files. SQL never makes repository content durable.

A writable local TensorFS root has exactly one host owner. SQLite/advisory locks coordinate
processes on that host; they are not a cross-host lock service and must not sit on NFS as shared
mutation authority. Multi-host publication and GC use Tensorhub's R2/PostgreSQL control plane.

Workers may additionally use a root (for example, `/workspace/cozy/repo-cache`) as an optional
datacenter cache. The example path is deployment policy, not part of the TensorFS format.

**The cache is a property of the Store, recorded once at setup.** `tfs store init <root>
--repo-cache <cache>` writes the mount into `<store-root>/repo-cache`; `--no-repo-cache` clears
it. Every later handle — `Store::open`, `Store::ensure`, and so every fetch, conversion, restore
and resume — reads it from there, so no request, protocol field, function argument or child
environment ever carries a mount path. The record is LOCAL to the Store and never on the cache,
which holds immutable objects only. Reading it can only change cost: an absent, unreadable or
malformed record is simply no cache, because a Store that would not open is an availability
failure and cache state may never cause one. Binding performs no I/O on the cache path, since a
stat on a hung mount does not return.

The cache is not a TensorFS Store and contains only immutable repository objects:

```text
blobs/<first-2-hex>/<next-2-hex>/<64-lowercase-hex>
manifests/<first-2-hex>/<next-2-hex>/<64-lowercase-hex>.json
```

There is no SQLite file, repository projection, lease, operation, verification authority, or GC
in that cache. Every hit is checked against an exact SHA-256 and length while TensorFS copies it
directly into the ordinary local Store. Missing, corrupt,
read-only, full, and unavailable cache states are soft misses; the authenticated cloud source
remains authoritative. Only objects a pull fetched from the origin are backfilled; a held object
is never offered. Backfill hashes while writing a hidden unique sibling, fsyncs, and links it
without clobbering, so concurrent writers converge and an interrupted write never has a final
name. That publication is the cache's integrity record: an exact-length blob at its name is
present without being read. A pull that reads an entry corrupt, and `tfs reclaim run --demote`, which
drops the local copy, re-verify instead and repair a corrupt regular entry by syncing its
replacement before an atomic rename. Publication pins the
configured cache root and opens each fanout directory without following symlinks; all temporary,
link and rename operations stay relative to those directory descriptors. Symlink and nonregular
entries remain unavailable. Repair grants no custody and adds no cache garbage collector.

Cache calls perform synchronous filesystem I/O and must run on a worker data path or subprocess,
never its control loop. A mounted filesystem which stops responding cannot provide a timely soft
miss from inside the blocked syscall, so subprocess progress/termination and live hung-mount proof
remain the supervisor's responsibility.

The production paths never name a cache directory: `tfs fetch`, `tfs ingest run`, `tfs repo-cache restore <store> ...` and
`tfs repo-cache resume <store> ...` use whatever the Store is bound to, and an unbound Store is a
run with no cache rather than a refusal — except a restore, where "bring these bytes back off the
cache" with no cache is a caller mistake and says so.

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

Those three are the durable format contract. Repository and Manifest JSON evolve additively:
a reader carries fields it does not know through every read and rewrite unchanged, so a
document keeps its bytes and identity across versions, and GC keeps any object such a field
names. A new entry kind, a changed field meaning or a new header format is a breaking change
and refuses. The frozen corpus also verifies the bounded
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

## Store root

TensorFS owns its Store location independently of Creator. User-facing entrypoints resolve one
default in this order: an explicit argument, `TENSORFS_HOME`, then `$HOME/.tensorfs`. Core Rust
Store operations remain explicit-root.

A Store is created only where someone NAMED it, and only by `init`/`ensure`: every `tfs` store verb
takes its root as an argument. Nor does a
writer marker create the root it locks, so no command lays down a skeleton Store in whatever
directory it happened to be run from.


## Disk budget

**A Store may be given a disk budget, and having one is the whole opt-in.** `tfs store
ensure <root> --disk-budget <bytes>` writes the number into `<store-root>/disk-budget`;
`--no-disk-budget` clears it. Every later handle reads it back, and `store info`/`store
ensure` print `disk-budget: <n> B, holding <m> B` so a supervisor can fence the readback
the way it fences `repo-cache:`.

**A Store with no budget behaves exactly as it did before the budget existed.** Nothing is
enforced, nothing is reclaimed, and nothing disappears — which is the point: a personal
`$HOME/.tensorfs` never gets one, and models never vanish from under their owner.

Where there IS one, `tfs fetch` refuses `CAPACITY_EXHAUSTED` **before it moves a byte**
when the pull would put the Store past it. The charge is what the pull ADDS
(`FetchPlan::wanted_bytes()`), not what its closure declares, so an object the Store
already holds is never bought twice — on a content-addressed store a second model sharing
an encoder with the first may add almost nothing.

The budget is measured against the Store's own occupancy — the blobs and manifests
TensorFS installed — and never against the filesystem. Filesystem headroom is `ensure`'s
admission (below), which the machine's one Store now owns (proto-063).

`CAPACITY_EXHAUSTED` is also what a genuinely full filesystem raises. It is a VERDICT and
never weather: retrying it fails at the same byte forever, which is exactly what the
generic `IO_FAILED` an out-of-space write used to raise bought on rented hardware.

## Fetch: the one downloader

`fetch::Fetcher` is the Store's one downloader; Runtime directs it, TensorD owns it.
[`Fetcher::want`] takes interest in objects of one origin (`hub`, `org/name@release`, lane,
manifest) at a priority and answers a ticket; `Ticket::wait` raises objects to `Now` and returns
once they are present; dropping the ticket releases the interest.

- **One fetch per object**: objects are deduplicated by digest across tickets; one closure per
  origin, resolved only when something is absent (a warm run asks the hub nothing); presigns
  are batched and minted ahead; the transport is the pull's (8 MiB ranged chunks, slow ones
  asked again against their peers).
- **Order**: `Now` (a reader is blocked), then `Next { run, rank }` by run acceptance and rank,
  then `Behind` (warm sets, hints), which runs only while nothing more urgent is out. `Now`
  takes streams from `Next` when none is free. Nothing is fetched without a live ticket.
- **Structure first**: a ticket also wants its origin's manifest, header and assets. Once
  they are here the origin is attested (`origins/<manifest>`, its objects indexed once): the
  checkpoint held in part is a cache root. GC keeps its present objects; at low disk they are
  evicted first (least recently used, structure kept) and ticketed or leased ones stay;
  `gc::collect_idle` drops it once unused for its TTL.
- **Executors** never fetch: TensorD serves each attempt `fetch::serve` over a socket, scoped
  to its selected manifests, and Runtime calls `tensorfs.fetch.Fetch` (`schedule`, `wait`,
  `present`). A derived writer or source channel reading a source held in part waits on a
  ticket with the origin its custody record names.
- **Failure**: a refused object (hub refusal, wrong bytes, no measured progress) fails the
  waits on it, named with its origin; a later `want` asks again. A short disk refuses
  `CAPACITY_EXHAUSTED` before a byte of the object moves.

## Ensure: model download

`ensure` makes a whole model's serialized bytes and metadata resident in the Store, through
the Store's fetcher: `tfs ensure` and `tensorfs.ensure`. This does not construct a
Python/PyTorch model or make its tensors GPU-resident. `tfs fetch` and `tensorfs.pull` remain
lower-level acquisition interfaces.

```bash
tfs capabilities                      # ensure/1, get-range/1, deliver/1, pressure/1
tfs ensure <root> org/name@1.0.0 --hub https://hub --lane fp8 --step "load" \
    [--ca-file hub-ca.pem] [--allow-hosts objects.example] [--keep sha256:<manifest>]...
tfs get <root> sha256:<blob> --range 0:1048576 --out -   # bytes to stdout, under a read lease
```

- **Stdout is JSON lines**: `ensure.progress` samples (`model`, `step`, `phase`,
  `bytes_done`, `bytes_total`, `rate`), then one `ensure.result` (exit 0) or
  `ensure.refused` with `code` and `resumable` (exit 1). The result's fetched and cached
  bytes are the Hub's cache observation.
- **One flight per manifest per Store**, across processes: a second `ensure` of the same
  manifest waits (`phase: waiting`, the leader's bytes) and moves nothing (`joined: true`).
- **Admission is disk only**: wanted bytes plus 16 + 3 inodes per object must fit under a
  reserve of 1/50 of the filesystem (clamped to 1–10 GiB) and 32 inodes, after what other live
  flights still owe. Short, the GC policy runs: unreferenced bytes, then idle checkpoints held
  in part, then delivered products (`Store.deliver_tree_root`, oldest first), then cached
  lanes; then `CAPACITY_EXHAUSTED`. `ensure::relieve` runs the same
  policy whenever no more than the reserve is free.
- **The Hub CA** (`--ca-file`, `ca_file=`) is added to process trust; an untrusted hub is
  `TLS_UNTRUSTED`, never retried. Nothing stalls on a timer: a download that lands nothing
  past its measured patience is `STALLED`.

## Durability

What an admission pays to be crash-safe is decided by the filesystem the Store lives on
(`statfs` of the root, Linux), never by a flag or an environment variable. `store info` prints
`durability: <class> (<filesystem>)`.

| class | filesystems | admission |
|---|---|---|
| ephemeral | tmpfs, ramfs, overlayfs (a container's disk) | no fsync at all: the disk dies with the pod |
| persistent | ext4, xfs, btrfs, zfs, anything unrecognised, every non-Linux OS | a pull: no per-object fsync; its own objects are fsynced together, in parallel, every 8 GiB and at its end; anything else: per-object fsync |
| network | NFS, SMB, FUSE, Ceph, 9p, AFS, Lustre, GPFS, ... | unchanged: each object is flushed before its name is published (close-to-open) |

A persistent pull first writes a durable marker, `tmp/unsynced/<boot-id>.<pid>.<n>.marker`,
naming when it began. After an unclean shutdown the first open of the Store finds a marker from an
earlier boot, rehashes every object not older than it, removes the torn ones (the next pull fetches
them again), and only then returns a handle. A marker a killed process left in the same boot needs
no rehash; the next pull to finish fsyncs the objects not older than it and retires it. Nothing
uses `syncfs`: on a shared filesystem it also waits for every other writer's pages. `tfs fetch`'s `pull.streams` event carries
`durability: {class, filesystem, syncs, sync_seconds, recovery}`.

## Reclamation

`gc` collects what nothing names. **`reclaim` chooses what to stop naming.** Two acts, two
verbs — and every one of them refuses on a Store with no recorded disk budget.

```text
tfs reclaim usage <root> --rows <rows.jsonl> [--keep <keep.jsonl>] [--candidates <cand.jsonl>]
tfs reclaim plan  <root> --need <bytes> [--keep <org>/<name>]... [--out <plan.json>]
tfs reclaim run   <root> --plan <plan.json> --assume-upstream [--demote]
```

**Reclaimable bytes are a property of the candidate SET, never of a model.** On a
content-addressed store two models can be backed by one object, so *"deleting a model might
reclaim almost no space because it was de-duped with one we want to keep"*. `reclaim usage`
answers both halves and neither is derivable from the other:

- per candidate, `bytes_marginal` — what dropping THAT ONE frees while the keep set and
  every other candidate stand;
- one `kind:"all"` row, `bytes_total` — what dropping every candidate frees.

Two models sharing a 200 KB base each exclude it from their own marginal, so the set frees
200 KB more than the sum. **A caller that adds the rows up under-estimates by the whole of
the sharing.** The `all` row also carries `occupancy_bytes`, `budget_bytes` and
`headroom_bytes`, so a caller asking how full the Store is has one answer rather than two
that can disagree; it is deliberately not a filesystem statistic.

Every number comes from the same walk the collection performs, with the deletions not yet
applied, so the estimate and the act cannot drift.

`reclaim plan` ranks by **marginal bytes per drop**, ties broken by the oldest-recorded
root — not by model size, which picks the wrong victim exactly when dedup is working, and
not by recency alone, which ranks access when the budget is about bytes. A root that would
free nothing never survives into a plan. If the whole droppable set frees less than
`--need` the plan refuses `RECLAIM_INSUFFICIENT` naming the shortfall, rather than emptying
the Store and failing anyway.

`reclaim run` requires `--assume-upstream`: the caller must first confirm that the
selected roots have durable upstream custody. A valid cache entry cannot supply that
promise because a cache owner may delete it at any time. Without the declaration, all
named local roots and bytes are preserved, even when `--demote` was requested.

With that declaration, `--demote` optionally warms the repo cache from verified local
bytes before repository roots are removed and ordinary GC runs. Failed cache writes do
not prevent reclaiming bytes whose upstream copy the caller has already confirmed.
Objects still named by a retained repository or hold remain protected by GC.

`reclaim run` takes the same store-wide exclusivity `gc` does and refuses `STORE_BUSY`
naming the live holder, so it cannot run beside a reader that holds a lease.


Python exposes the same resolver. Passing a root remains supported and always wins:

```python
import tensorfs

root = tensorfs.home()
store = tensorfs.Store.ensure()
store.prepare_readers()
held = store.complete_cozytensors_manifests()
other = tensorfs.Store.open("/srv/tensorfs")
```

`Store.init`/`tfs store init` accept only an absent or exactly empty real directory. Use
`Store.ensure` for normal startup: it initializes that same narrow case, but requires a nonempty
root to be a real current Store with a valid `tensorfs.sqlite`. Neither operation follows a root
or namespace symlink, merges foreign files into a Store, or recognizes an older layout.

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
| `ok`, with `ignored` | (4) a stored key inside a constructed component absent from the requirements is skipped, never loaded, on every custody: `ignored` lists it, `ignored_bytes` sums what it stores, and `warning` is the one line to show (count, bytes, first keys). `extra_tensor` is no longer answered |
| `encoding_unsupported` | (5) the package refuses encoded leaves |
| `encoding_unqualified` | (6) with `--device`, every encoded leaf has a qualification record for that class (plain storage has nothing to qualify) |

`tensorfs.capability(encoding, device, observations=None)` answers the device axis alone.
TensorFS has no compatibility grade or search plane.

Each named construction config is one bounded RFC 8785 JSON-object byte string stored directly in
the CozyTensors header. Header parsing refuses duplicate keys, trailing bytes, non-finite numbers,
non-object JSON, oversized values, and any noncanonical spelling. TensorFS stores no separate
config objects or aggregate config digest.

Everything below is what CI runs, in the same order.

```bash
export PYO3_PYTHON="$(uv python find 3.12)"
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
python3 scripts/fence.py
python3 scripts/version_contract.py source

cargo build --release --bin tfs
uv run --python 3.12 --project conformance/cbor/python \
  python scripts/cbor-conformance.py --tfs target/release/tfs conform

# the end-to-end test: wheel -> fresh venv -> real store -> lease/plan/ring/CAS
bash scripts/wheel-smoke.sh
```

One artifact, one version (tfs-051): the wheel ships the `tfs` binary beside the module, so
`uv pip install tensorfs==<v>` from that wheel yields both, and `tfs version` prints
`tfs <version> sha256:<digest of its own bytes>` — the line tensorhub and cozy-creator check
at startup. `scripts/build-wheel.sh` stages the binary and checks the wheel contract;
installed-wheel validation also requires the full Store API. The sdist carries sources
only, so source builders must stage the CLI with that script before building a wheel.
Ordinary installations use the wheel. Managed Runtime access depends on explicit owner
grants, not on installing a reduced Python variant.

The CBOR gate runs the production Rust codec and an independent Python codec over the same
green and red vectors.

## Store commands

```text
tfs store init <root>
tfs store rebuild <census> --rows <rows.jsonl>

tfs put <root> <file>
tfs get <root> <sha256>
tfs reap <root>
```

Blob admission hashes while streaming, installs with no-clobber semantics, and records its
rebuildable verification observation in SQLite. Each process loads those rows once per catalog
into memory; a trust check is an `fstat` compared with the row's `(dev, ino, length, mtime)`,
and a changed or unrecorded object is rehashed.
`tensorfs.stats()` returns this process's `catalog_opens`, `presence_passes` (closure walks that
prove every object resident) and `repo_cache_verified_reads`; diff two snapshots around a call.
Manifests use the same digest/length identity but
live under their typed namespace. A blob or Manifest that fails an exact rehash is removed from its
authoritative path immediately; TensorFS retains no forensic copy. A writer holds an advisory lock
on its admission temp for the temp's whole life; `tfs reap` removes only the temps whose lock the
kernel hands out, never by age, so a slow live writer is never reaped from under.

## Repository publication

```text
tfs manifest build <entries.jsonl> --out <manifest.json>
tfs manifest inspect <manifest.json> --refs <refs.jsonl>
tfs manifest get <store> <manifest_id> --out <manifest.json>
tfs repo inspect <repo.json> --rows <rows.jsonl>
tfs repo apply <current.json|-> <mutation.json> \
  --manifest <manifest.json> --header <header.cbor> --inventory <blobs.jsonl> \
  --out <replacement.json> --rows <rows.jsonl>

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

`storage::repository_release_lines` gives one `kind: "release"` row per release lane and one `kind: "local"` row per local
alias; `Census::usage` gives the bytes each repository reaches and the bytes unique to it.

Rebuild always recovers the repository, Manifest, and complete reached blob set;
model headers additionally recover the diagnostic projection emitted by `store rebuild --rows`.

## Reviewed source planning and ingest

Foreign carrier paths never name components. A review act may bank one whole-source profile beside
the existing fingerprint entries; the profile supplies the exact required component set, target
encodings, and canonical construction traversal for each reviewed key-set variant. Classification is by
`(key set, component)`: an entry banked from the exact tensor schema is preferred, and otherwise the
reviewed key set suffices while conversion validates the actual shapes and dtypes, so an FP16 upload of
a model banked in BF16 classifies. An optional canonical provider
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

**A carrier is read by its NAME, and the name decides where its shards are.** The reviewed member is
the name; the file name is the name only when no member is given. This is what lets a sharded
`*.safetensors.index.json` be read at all from a content-addressed location — a staged carrier lives
at `staging/<sha256>`, so its file name is a hex digest with no extension and its directory holds no
siblings. The same key answers the second question: a carrier named by MEMBER resolves its
`weight_map` through the carrier set, so every shard the index names must itself be a carrier, named
by member; a carrier named by FILE NAME resolves them beside itself, as a checkout on disk. So
`--carrier <path>` over a real tree needs only the index, while `--carrier <member>=<path>` must also
name the shards. Carriers no index names are ignored: a selection spans a whole model, and one
component's index has no business with another component's carriers.

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

The built-in reviewed registry ships with TensorFS and is extended, never replaced, by reviewed data
bound to a Store (`Store.bind_fingerprint_registry(bytes)`: a pinned file or the Hub's registry) and by
bytes a caller supplies to `prepare_model_source(registry=...)`. Newer entries replace older ones
under the same key and keep their own provenance, so an older TensorFS can ingest models banked after
it shipped. The CLI `--registry` flag remains for platform review and tests. A provider may present a
superset of candidate headers. TensorFS selects one exact reviewed subset and refuses if multiple
profile assignments survive, so adapters never choose a variant by filename. A reviewed
production may name its exact required source profile; this narrows only the registry profile, while
the same header/member proof still has to match it exactly.

**The registry is a hint, not a gate.** A source no reviewed profile matches, or one planned with
`--source-profile as-is/1`, is stored as it is: converter `identity/1` makes every safetensors key one
plain tensor, nothing folded, dropped or renamed. Its default selection keeps carriers in folders over
loose root files and, per folder or root stem, the non-variant file, else the lowest precision; shards
come with their index. One component is `model`; several are named by folder or stem. The header's
`normalization` config records `{"converter","dialect","state":"raw"}`; a normalizing converter records
`"state":"normalized"`, and reviewed converters that predate the record write none. Malformed
carriers still refuse.

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

Standalone Python Store owners can use the compiled source-preparation facade directly:

```python
prepared = store.prepare_model_source(
    operation_id,
    source_selection_digest,
    [("dits", "hf/minimax-h3/native-dual-bf16/1"),
     ("shared", "hf/minimax-h3/shared-bf16/1")],
    [(member, object_id, length, local_path, exact_header_bytes), ...],
    landed_members,
    checkpoints=[(slot, previous_head_id, previous_head_length), ...],
)
# prepared = {complete, replayed, sources, checkpoints, spent,
#             converted_roles, resumed_roles, deferred_ops, converted_bytes, largest_op_bytes}
# Only after the operation is adopted or explicitly abandoned:
store.release_model_source(operation_id)
```

TensorD performs managed source acquisition/preparation through the corresponding Rust
APIs. Runtime's package jobs use scoped source/output interfaces for their tensor work;
the direct Store example above does not describe a managed executor opening the Store.

The sorted roster contains all members from the outset. Headers are exact JSON index bytes or
the safetensors eight-byte length plus header JSON. Every body length is pinned; only members in
`landed_members` need exist. TensorFS plans from all headers, resolves each tensor to its physical
shard, converts available ops, and journals them before returning. One pass shares a 4 GiB
conversion budget across all profiles, stopping between complete ops. One oversized first op
may progress; `converted_bytes` and `largest_op_bytes` report the measured work. A pass with no
new work keeps its latest local head, including when the owner acknowledgment is delayed. A shared index object can still
name different shard sets through its two member names. Local paths and unrelated registry entries
do not change work identity; source ObjectRefs and the actual ordered tensor plan do.

`sources` contains final `{slot, profile, manifest_digest, manifest_length}` rows only when every
profile is complete. Partial work returns separate `checkpoints` rows:
`{slot, head, head_length, plan_digest, index, bytes}`. These are **local** progress, not a claim of
remote durability. The existing immutable Link format carries at most 128 new refs per link and
an opaque journal snapshot. `Store.checkpoint_page` validates operation/slot/plan and
returns bounded refs, predecessor and progress references. Fetch/PUT use `Store.checkpoint_fetch/push` over the existing transport.

Derived writers use the same checkpoint transport. `Store.begin_derived` requires
`work_fingerprint=` in addition to its existing transaction, epoch and declaration arguments.
Before opening a writer, `Store.derived_declaration(sources, targets, configs, order,
max_new_bytes, work_fingerprint=...)` produces the exact canonical bytes for its intent.
`Store.validate_derived_checkpoint(transaction_id, declaration, head_id, head_length,
operation_id=..., slot=..., plan_digest=...)` verifies restored work without opening a writer.
The caller owns that producer fingerprint and transaction ID; TensorFS locks them into one
canonical declaration. Higher writer epochs retain verified accepted work. `completed_parts()`
returns `(component, key, role)` keys and `completed_configs()` returns names, as detached views.
Re-adding identical bytes is idempotent; changing an accepted part or config refuses.

`writer.checkpoint(operation_id, slot, previous=...)` returns the same head/length/plan/index/bytes
facts as source preparation. It extends the newest local cursor even before acknowledgment.
After fetching the chain into an empty Store, pass `checkpoint=(head_id, length)` to
`begin_derived`; only accepted work is imported, never the previous writer's authority. An absent
or corrupt part is no longer completed and must be supplied again. Runtime decides which complete
role groups can be skipped and owns their numerical quality checks. Fencing revokes authority;
explicit abandonment or terminal disposition releases the private filesystem retention root.
Inherited source closures must also be restored through their existing acquisition path before
validating or opening the writer. A derived checkpoint carries added work, not a second copy of
its inputs. The writer's private root retains its admitted sources across process death and
release of their original source operation, until commit or abandonment.

The caller may delete `spent` raw carriers once canonical output and its journal are held locally.
The RecordOwner uploads each Link and its objects and retains an acknowledged head outside the pod.
Canonical output remains held until upload or an explicit terminal action. A replacement pod fetches the chain
through ordinary verified admission and passes its head back in `checkpoints`; TensorFS re-derives
the plan and verifies journalled parts before skipping work. No NFS mount is required or accessed
by source preparation. A completed preparation replays its persisted final heads after a lost
response, including when source bodies are gone or the caller still supplies an older ACK.
Explicit release removes the operation's holds and recovery journal.

`repo apply` validates the exact manifest, header, and complete blob inventory before the caller may
perform its repository CAS, then records only `(org,name,manifest_sha256)` in SQLite. Missing or
length-mismatched bytes refuse publication. Full repo/release/object rows remain diagnostic rebuild
output, not duplicated database tables.

`local` is the narrow Creator alias API over the same Repository bytes. Resolve emits one compact
canonical JSON row. Replace atomically swaps the complete `repos/local/<name>.json` document to
exactly one hidden `(source-selection, local) -> Manifest` release after verifying its closure;
stale observed repository digests refuse. Remove deletes the complete alias. Ordinary repository
mutations cannot append a second row under org `local`.

## GC and recovery

```text
tfs gc headers <census> --out <header-requests.jsonl>
tfs gc plan <census> --holds <holds.jsonl> --out <plan.jsonl>
```

`gc::collect` is the local reclamation act: under the store's exclusive lock it marks
`repositories -> manifests -> blobs` from the filesystem alone, keeps what an open ingest session
root (`tmp/ingest/<session>/session.json`) or private derived root (`roots/derived/<transaction>.json`)
names, deletes every other blob and manifest, and
repairs the verified-blob rows so no row outlives its object. `tensorfs.sqlite` is never consulted
for liveness — the database is not the source of truth; the filesystem is self-describing. It
prints `reclaimed <bytes> B in <n> objects, kept <bytes> B in <m> objects under retained roots`
(`--json` for a program), is idempotent, and converges after a crash (an object is unlinked before
its row is dropped). A live writer or read lease refuses `STORE_BUSY` naming the holder; a session
whose writer is gone is abandoned only by the explicit `tfs ingest reap`. `--dry-run` plans without
deleting; `gc holds` + `gc plan` are the same plan as separate documents, which is how a cloud
caller (tensorhub) supplies its own holds against a staged census.

A derived receipt names its declaration by `declaration_digest` and `declaration_length`, so it
stays constant however many tensors the declaration lists.

A derived commit creates a pending private root before releasing its writer. The owner's
`Store.derived_adopt(transaction_id, scratch_root_id)` names that root without creating a model
repository; the name is opaque to TensorFS. Same-name adoption replays, changing the name or
adopting released output refuses. `derived_lookup` reports disposition alongside its unchanged
receipt. `derived_dispose` durably releases the transaction before removing its filesystem root;
replay completes interrupted removal. An absent `derived_abandon` creates a tombstone that prevents
a late writer from opening that transaction. GC reads private roots directly, independent of the
transaction journal's readability.

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


`tensorfs.plan(header, traversal, components=...)` checks the complete declared component
census and its construction order. For bounded paging, cache that complete plan and call
`plan.select(["text_encoder/model.language_model.layers.0.self_attn.q_proj.weight#value", ...])`.
Selection uses exact existing part `what` identifiers, keeps every native window and source
ObjectRef for those parts in native order, and compacts destination offsets. Duplicate or
unknown identifiers refuse before I/O; an empty selection returns an empty plan. The original
plan is unchanged, and selected plans use the same lease, stream and caller-owned ring.

A committed derived result can acquire independent private retention without changing its
execution receipt: `Store.retain_derived_result(transaction_id, expected_receipt_digest,
retention_id)`. The caller supplies an exact SHA-256 retention identity; the result names
the original transaction, receipt, and manifest. It verifies native retained custody and
resident verification records without copying tensor payloads.

`Store.release_derived_retention` takes the same three arguments and releases only that
retention owner. Its durable tombstone rejects delayed reacquisition, including release
before acquire. The original transaction can be disposed independently. Native GC finds
these manifest roots directly from the filesystem, without relying on request or transaction
metadata. Eligibility and authorization belong to the caller; this API provides storage
retention and never schedules computation or publishes a model repository.
