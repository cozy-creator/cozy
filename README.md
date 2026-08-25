# tensorfs (v2)

Canonical tensor byte plane: CAS, cozytensors headers, snapshots, ingest. Design authority:
[tracker-v2](https://github.com/cozy-creator/tracker-v2) `tensorfs.md` / `cozytensors.md`; issues `tracker/tensorfs/`.
Rust workspace: `tensorfs-core` (torch/CUDA/HTTP/SQL-free) and `tensorfs-py` (marshal-only PyO3).
No automated tests (decisions.md #160): verification = live runs + benchmarks. Static analysis
IS required: `cargo clippy -D warnings`, `cargo fmt --check`, `mypy --strict`, `scripts/fence.py`.

## tfs (tfs-013)

```
cargo run --release -p tensorfs-core --bin tfs -- registry        # seed specs: alias -> digest
cargo run --release -p tensorfs-core --bin tfs -- vectors verify  # run the frozen corpus live
cargo run --release -p tensorfs-core --bin tfs -- vectors gen     # rewrite it (ids change: review the diff)
cargo run --release -p tensorfs-core --bin tfs -- bench 3699 50   # writer throughput + peak RSS
cargo run --release -p tensorfs-core --bin tfs -- flip            # identity-flip matrix
cargo run --release -p tensorfs-core --bin tfs -- id <file>       # object id + derived ids
```

## checkpoints (tfs-015)

The real end-to-end plane: a synthetic multi-component tree is objectized through the CAS,
declared by one canonical `model.cozytensors`, and reconstructed from the header alone.

```
tfs checkpoint write <root>            # ~382 MiB, 3 components, 4 encodings -> header_digest
tfs checkpoint read <root> <hex>       # every anchor equation, every segment, byte-identical
tfs checkpoint info <root> <hex>       # component/scope query — opens ONE object, the header
tfs checkpoint locate <root> <hex> <component> <key> <role> <off> <len>
tfs checkpoint admit <root> <hex>      # serving admission against the local capability pin
tfs checkpoint caps | size [n]         # whole-document total caps; uniform-form header cost
scripts/checkpoint-runs.sh all         # the whole thing plus five planted red arms
```

## snapshots (tfs-004)

```
tfs snapshot show <root> <hex>      # strict decode + decode->re-encode byte identity
tfs snapshot walk <root> <hex>      # transitive root; every-and-only vs the store census
tfs snapshot verify <root> <hex>    # full-checkpoint verify: every declared byte
tfs snapshot flip <root> <hex>      # what moves checkpoint_id / header_digest / topology
tfs snapshot fuzz <root> <hex>      # bounded mutation fuzz over canonical bytes
tfs snapshot blob put|read|repair <root> <file> [i]
scripts/snapshot-runs.sh all        # the whole plane plus its red arms
```

`checkpoint_id = snapshot_id`, unconditionally. Contained documents (EncodingSpec objects
and their vector sets) live at `closure/<digest>.json`, where the path terminal IS the
digest; the header attachment is TYPED and must be projected at exactly one path.

## reads and projection (tfs-005)

`read_into(range, caller_buffer, lease)` is the one byte door. The caller owns every buffer;
TensorFS owns range validation, verified identity, short reads and the lease. There is no
call that takes a bare digest and returns bytes, so `contains` cannot become a resume
predicate. Reader depth is load-bearing: 0.32 GB/s cold at queue depth 1 against 1.82 at 16.

```bash
tfs read lease <root> <snap>            # acquire, verify, release ONE ReadLease
tfs read plan|fill <root> <snap>        # destination order; the pooled fill into a ring
tfs read verify-fill|prove <root> <snap>  # the sequential path; delivered bytes vs the header
tfs read span <root> <snap> <c> <k> <role> <off> <len>
tfs read arms|tear <root> <snap>        # red arms; tear a verified object UNDER a live lease
tfs project <root> <snap> <dest>        # tier 2: relative symlinks, never a weight path
tfs materialize <root> <file> <dest>    # tier 3: stream a RETAINED carrier blob
scripts/read-runs.sh arms|real|bench
```

## roots, holds and GC (tfs-006)

`meta/state.json` under `meta/LOCK` is the SOLE root authority; `refs/` is a rendering with
none. A hold is reapable exactly when its owner is PROVABLY dead — the kernel drops the
owner's lock on SIGKILL, so there is no wall-clock grace. `exclusive` is a promise: deleting
a root reclaims exactly those bytes.

```bash
tfs root list|register|release|open|note|reconcile <root> ...
tfs hold take|reap|steal <root> ...     # owner-liveness holds
tfs gc mark|usage|sweep|reap|refs|pressure|arms <root>
scripts/gc-runs.sh full|arms
```

## ingest (tfs-003)

The only normal tensor-ingest door. `inspect` and `plan` read HEADERS only — every
structural refusal is decidable for a few hundred KiB, before a GPU is rented. `run` hands
the job to an isolated child; `install` is the coordinator's separate act; everything that
never reaches it reaps.

```
tfs ingest inspect [--markers] --source unet=<file> [--source vae=<file> ...]  # bounded decode
tfs ingest bank <dialect> <family> <variant> <converter> --source c=<file>   # the REVIEW act
tfs ingest plan  <component=alias,...> --source c=<file> [--keys]   # zero tensor bytes read
tfs ingest run   <store> <component=alias,...> --source c=<file> [--config c=<file>]
tfs ingest install <store> <session>        # the coordinator promotes ONE root
tfs ingest roots|reap <store> [--now]       # candidates are not checkpoints
tfs ingest reingest <store> <header> <target> [--rekey c=from:to]   # inherit by reference
tfs ingest converters | golden              # reviewed converters + their geometry, live
scripts/ingest-runs.sh all                  # the border plus 49 planted arms
```

A `--source` names its component; a FILENAME never authorizes anything. Classification is
a key-set fingerprint against a reviewed registry (`vectors/fingerprints/`), and both
absence and ambiguity refuse. Measured on a real 6.94 GB SDXL fp16 tree: **84 MB/s
end-to-end at a 13.2 MiB peak RSS** (streaming, never whole-file), planned in 30 ms from
336 KiB of headers. Re-ingesting an already-canonical component inherits every object by
reference: 167 MB carried, **zero bytes hashed**.

`h3.native/1` is the one PERMUTATION converter: every `attn.qkv_proj.weight` becomes
canonical `to_q|to_k|to_v` under the ratified head-interleaved seam, with the head dim read
off the module's own `q_norm` and never a constant. Over the pinned official native FL2VA
header the seam plans **12 permute ops, 672 runs, mean run 1,376,256 B, long-run** — and the
members stream out of the carrier one run at a time, so a permute costs the same constant
memory an identity stream does (6.3 MiB peak on both arms of a 352 MB pair). A fused
attention key with no banked seam, or one carrying encoding roles, refuses `SEAM_UNBANKED`
rather than carrying through.

## registry (tfs-008)

```
tfs structure list | match <root> <hex>       # banked fixtures, and what a checkpoint is
tfs structure bank <rows> <family> <variant>  # bank transcribed evidence
tfs conform [alias]                           # qualify implementations against vectors
tfs capability <digest> <device> [--alias a]  # fail-closed admission
scripts/registry-runs.sh all                  # fixtures, mining, conformance, red arms
```

Vectors are PRODUCER-MINED: `scripts/mine-vectors.py` runs v1's own quantizers (the quarry
under `~/cozy`) and transcribes raw bytes; `tfs vectors mine` canonicalises them. A
vectorless spec stores and inspects and can never qualify an implementation.

`vectors/` is frozen DATA: `closure/` holds the platform EncodingSpec objects, `fingerprints/`
the reviewed classification registry, `cases/<group>/`
holds each vector's exact stored bytes plus a `.verdict` (expected outcome, frozen ids),
`mined/` holds the producer-mined conformance vectors and `structure/` the banked
structure fixtures. `vectors gen` regenerates only `cases/` and `closure/` — the other two
are evidence with their own provenance and their own banking commands.
Caps in `limits.rs` are PROVISIONAL until proto-001's real H3+SDXL corpus ratifies them.

## the layout plane (tfs-009)

Two operations. **Stamp** runs inside isolated ingest and is the only thing that can mint an
`IngestVerificationReceipt`: it re-derives the subject/profile binding, source generations,
converter build/lock/recipe, the canonical header, the computed `topology_digest`, the
encoding part roles, the candidate object set (every-and-only against the snapshot walk),
the golden verdict and the bounded accounting — from bytes the store holds, never from what
the run reported. Its ledger is its own document (`tensorfs.stamp/1`) and the receipt binds
it. A receipt is still not a publication.

**`check_compatibility`** compares a bound model's code topology with a checkpoint's and
returns the ONE runtime vocabulary, `Satisfied | Mismatch(details)`, naming the first
disagreeing tensor with both sides plus the component-granular encoding facts. It plans
nothing and names no remedy: grading a mismatch into
`COMPATIBLE | CONVERTIBLE | PRODUCIBLE | INCOMPATIBLE` is the HUB's job, and a sixth fence
in `scripts/fence.py` keeps that vocabulary out of `compat.rs`.

```bash
tfs compat import <rows.json>          measured rows -> canonical header + code topology
tfs compat check <code> <header>       Satisfied | Mismatch(details) + the facts
tfs compat release-cut <report>        unclassified extras never cut a release
tfs grade <report>                     the four-verdict ladder, from headers alone
tfs grade register <name> <kind> <sp>  the registration ACT: it RUNS A->B->A
tfs verify-request | verify-compat | verify-consume    the hermetic seam
tfs ingest restamp <root> <session>    re-verify a candidate; --plant <what> for arms
scripts/compat-runs.sh all             real banked H3 pairs + 61 arms + benchmarks
```

A CONVERTIBLE grade names a REGISTERED lossless mapping, and registering one EXECUTES the
round trip here — over the complete 16-bit domain for a 16-bit cast, so the claim is proved
rather than sampled. A re-quantization cannot register: `f32 -> f16` refuses
`ROUND_TRIP_LOSSY`, and `bf16 -> f8_e4m3fn` is not expressible in the mapping vocabulary at
all. The verifier goes further than grading a report: it APPLIES the registration to the
code topology and re-checks, so CONVERTIBLE means the mapped topology satisfies, not that
one key happened to match.

The pairs are REAL. `scripts/h3-pairs.py` transcribes pinned upstream headers out of
`~/cozy_v2/h3-evidence` (33 artifacts, each pinned to a repo revision and a sha256 of the
exact bytes read) into neutral measured rows; `tfs compat import` turns those into canonical
documents and says on every run that the logical facts are real and the part object ids are
placeholders. No weight bytes transit this machine, and nothing below the topology layer is
read by the decision.

## the evidence bank, reconciled (tfs-010)

```bash
scripts/bank-reconcile.py              20 cases: banked verdict vs admission verdict
scripts/bank-reconcile.py --bank DIR   run against another copy of the bank (the red arm)
tfs compat import <rows> --survey      every measured role set and the spec it selects
```

Two independent routes to one answer. The BANKED column is computed from the evidence
bank's own recorded facts and re-reads no header byte; the ADMISSION column is observed by
running the binary over the same pinned bytes. The recorded relationship between them is
the fence — 14 agree, 6 disagree, and a disagreement is a finding with a code
(`REGISTRY_GAP`, `UNDER_INFORMED`, `READER_KILLED`, `LANE_TIERS_EXTRAS`), never something to
average away. It found that the bank's ninth key dialect does not exist: 450 nvfp4 encoding
ROLES had been counted as tensors, and folded correctly that artifact is the text encoder's
own 902 keys, 250 of them matching the registry's producer-mined `nvfp4-w4a4/1` on real
shipped bytes. 2.4 s, 34 MiB, whole table.

`--survey` exists because an artifact mixes encodings per tensor: the refusal names the
first gap, the survey names them all.

## the gguf-v1 planner (tfs-011)

```bash
tfs gguf plan <file> --probe [--twin <rows.json>] [--tensors N]
tfs gguf admit <file> --probe          the admission verdict — refuses, today
scripts/gguf-runs.sh all               the plan, the twins, the labels, 10 planted arms
```

A PLAN, not a second door: `carrier::sniff` still refuses `GGUF_REFUSED` on the normal
path, in the same run. From the metadata region alone (33–75 KB probes of 15–36 GB
carriers) it derives both views a lane gets — the retained carrier BLOB an engine streams
whole, and the decomposed per-tensor byte ranges — plus each tensor's ggml type, block
count, logical shape AND the source of that shape. 1.7 ms and 5 MiB for 902 tensors.

Mapping a quant family to a reviewed EncodingSpec is BLOCKED and says so: a spec is a
digest identity needing producer-mined vectors and a lane that names it, so `admit` refuses
`GGUF_ENCODING_NOT_REVIEWED` until se-007 or job-008 selects encodings.

Labels are never evidence, and the corpus proves it rather than the docs asserting it: the
plan digest is filename-independent, a DiT shipped as `-Q4_K_M` carries zero Q5_K/Q6_K, and
one publisher's H3 DiT declares `general.architecture = "wan"`. What identifies an artifact
is its tensor table — every logical shape in all five checkable carriers agrees exactly
with its pinned safetensors twin (535/535/535/532/902).

## cloud publish (tfs-017)

The invariant is no-corrupt/no-clobber, and HOW it is enforced was a qualification outcome
rather than an assumption. The gate ran first, against live R2, and came back **GREEN
(10/10)**: a SigV4-signed `x-amz-checksum-sha256` binds the body to its digest — a wrong
body is `400 BadDigest`, a header the client strips is `403 SignatureDoesNotMatch` — and
`If-None-Match: *` refuses replacement with `412 PreconditionFailed`, eight concurrent
same-key writers producing exactly one `200`. A 64 MiB segment goes in one conditional
request, so multipart, the one surface with no conditional form, is unreachable by
construction. The red arm is banked too: an UNCONDITIONAL same-key PUT really does replace
verified bytes, which is why no such grant is representable here.

So publish is `declare closure -> conditional checksummed PUTs at final keys ->
verify-resident-on-conflict -> VerifiedObject facts -> atomic catalog-root publication`.
**There is no staging namespace, staged-verifier pass or promotion copy** — that was the
fallback for a RED gate, and a seventh fence keeps its vocabulary out of the tree.

```bash
tfs cloud closure <root> <snap> <session>       declare first: the exact object set
tfs cloud grant <closure> <object-id>           both conditions, DERIVED, not optional
tfs cloud verified <grant> <version> <etag> <checksum-b64>
tfs cloud conflict <grant> <resident-checksum>  honest duplicate, or quarantine
tfs cloud publish <closure> <facts-dir>         the atomic root, over facts only
scripts/r2-qualify.py --env <f>                 the gate
scripts/r2-qualify.py --env <f> --publish <store> <snap>   the race matrix, live
```

Core links no HTTP, no cloud SDK and no credentials — the coordinator executes the
requests. What TensorFS owns is the SHAPE, and the shape is where the invariant is held:
`PublishGrant` has no field for the conditions because they are derived from the digest, so
an unconditional or unchecksummed final-key grant has no representation. The v1 ruling this
reverses did not fail because someone forgot a flag; it failed because a flag existed.

Verdicts and arms are frozen under `vectors/qualification/`.

## the Python facade (tfs-007)

One wheel, abi3-py311, `py.typed`. The compiled half owns every byte-level fact; the Python
package is types and names, and `scripts/fence.py` enforces it. **Torch is outside the
boundary**: a read fills a buffer you already own and hands back plain data; element sizes
come from `tensorfs.DTYPES`, which is the compiled dtype enum exported exactly once.

```python
import tensorfs
store = tensorfs.Store.open(root)
snap  = store.snapshot(snapshot_id)                    # manifest + header, canonical bytes
plan  = tensorfs.plan(snap["header"], traversal, window=1 << 22)

with store.acquire_snapshot(snapshot_id) as lease:     # verified, hold registered first
    def on_batch(b: tensorfs.Batch) -> None:
        issue_h2d(slots[b.slot][: b.nbytes])           # asynchronous
        b.release()                                    # after the completion event
    lease.stream(plan, slots, on_batch, readers=8)
```

`stream` is a CONTINUOUS slot ring, not a batch-at-a-time call: `execute` per batch drains
every reader at each boundary (tfs-005 measured 1.24 GB/s through a 64 MiB ring against
1.37 through a 1 GiB one), and a streaming consumer pays that per tile. The ring keeps the
pool running while the consumer holds a slot, and `release()` is a positive act — a slot
returns when the H2D event fires, never when a Python statement ends. Every slot held and
none released is `SLOT_STARVED`, not a hang.

Every refusal crosses TYPED: one Python exception class per `err::Code`, generated at import
from the Rust list, so `except tensorfs.errors.ShortRead` and `e.code` both work and neither
language can drift.

```bash
scripts/wheel-smoke.sh                  # build, install into a fresh venv, read a NAMED
                                        # tensor out of vectors/store, match the banked digest
scripts/facade-conform.py               # the 122 shared vectors, verdict-for-verdict
scripts/facade-runs.py arms|real|bench <store> <snap>
scripts/facade-h2d-smoke.py <store> <snap>   # cr-005's fill loop with the facade as reader
```

`vectors/store/` is a committed 176 KB snapshot (`tfs checkpoint write ... --blocks 1
--scale 32`) so every wheel in the matrix exercises a real read rather than an import.
