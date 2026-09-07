# Resident conversion profiling

`resident_conversion_profile` measures the existing native converter after parsing and
planning resident safe-carrier files. It reports elapsed conversion/finalization time,
process peak RSS and `/proc/self/io` deltas alongside the converter's logical counters.
It performs no download, quantization, publication or claim that a sampled artifact is a
complete H3 model. Its `span` command uses the existing carrier parser/range resolver.

```sh
cargo build --release --locked -p tensorfs-core --example resident_conversion_profile
# Inputs are real safetensors files. The caller explicitly names the component.
target/release/examples/resident_conversion_profile convert STORE OP all h3.native/2 ref2va_dit=INPUT
# Repeat with the same paths/operation to measure verified journal reuse.
```

`first` converts only the first landed carrier. `pause:BYTES:MARKER` is a profiling-only
fault trigger: the existing Store progress callback writes MARKER and parks after observing
that many moved bytes, allowing a harness to SIGKILL an actual unfinished write.

## 2026-09-07 evidence

Recorded outside the repository in `outputs/h3-conversion-profile-20260906/`:
`baseline.json`, `suite.json`, `comparison.json`, `durability-cost.json`,
`crash-proof.json`, and `qkv-real-header-match.json`.

The inputs are bounded real tensors from MiniMaxAI/MiniMax-H3 revision
42ed227ee7df40d41602854ae760620d6eb651fe. The source fidelity audit used the native
carrier parser/converter/range resolver and found exact HF/CAS equality. The full
231,211,264-byte fused-QKV+norm fixture reproduces the exact four records (including
all segment IDs) in released checkpoint3f6c224a...0b4df. All115,605,504 QKV BF16 values
are finite. Dataset creation/download is excluded from conversion timings.

| Resident file workload | Baseline | Optimized | Trials |
| --- | ---: | ---: | ---: |
| 4.30 MB /12 actual tensors, task-file cache evicted |491.4 ms|303.4 ms|3 each, medians|
|231.21 MB fused QKV, task-file cache evicted |1597.6 ms|1348.3 ms|3 each, medians|
|231.21 MB with independent bounded competing filesystem I/O |2098.0 ms|1660.5 ms|1 each|
|231.21 MB with warm OS page cache |639.3 ms|450.6 ms|1 each|

Cold QKV device reads were231.2 MB in every trial. Peak RSS stayed below6.6 MB.
Warm journal replays moved no payload, taking roughly8–15 ms for these fixtures.
A real SIGKILL after3,199,232 moved bytes left seven accepted roles; restart hashed
231,211,008 payload bytes versus233,361,664 for the clean run, and both manifests match.

The traced baseline small case performed87 SQLite fsyncs (264 ms),13 payload fsyncs
(41 ms),12 conversion-journal syncs (37 ms), and44 directory/other syncs (139 ms).
The patch groups the existing operation hold and verification record in one FULL SQL
transaction; it does not switch journal mode or relax filesystem synchronization.

The legacy `payload_hashed` counter is logical payload bytes, not total SHA work.
Baseline ingest also computed an unused whole-tensor digest in a Tee while the Store
hashed each segment. From that executed path, payload hash-input work was twice each
segmented byte plus each inline byte. Ingest now hashes only admitted segments; inline
bytes remain covered by the canonical header. Public writers retain whole_sha256, using
the same segment ID when one segment is the complete run and hashing the full run when
multiple segments require a distinct whole digest. Timing, logical counters and this
hash-work accounting are separate facts.

## Separate live cache attribution

The public0.3.12 H100 pull is not a resident-conversion measurement. Root observed all16
object workers blocked in FUSE cache backfill, zero sockets, and13.25 MB/s network RX
on a40 s sample. A supported cache unbind and replacement of only the fetch subprocess
preserved411 local CAS objects. The replacement had16 socket readers/no FUSE waits and
completed14.14 GB in52.057 s (about272 MB/s completed-object progress). Host/Runtime
processes and the Store remained. This identifies synchronous optional NFS backfill as
a separate acquisition bottleneck; this patch does not redesign cache policy or activation.
