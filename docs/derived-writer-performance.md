# Derived writer profiling (tfs-093 / tfs-114)

The first measured fix changes transaction scope, not tensor arithmetic: one writer now
reads and changes its own SQLite row. It no longer parses unrelated writers/history or
rewrites their rows when accepting a tensor. Read-only epoch checks no longer acquire
SQLite's writer lock. Per-part acceptance, checkpoint export and fencing are preserved.

## Evidence and reproduction

Baseline: TensorFS `560d9db` (0.3.39 source). Runtime broker: `71db0d6c`.
The patched native implementation is in TensorFS PR166. Profiles ran on the same shared
Linux workstation, in release mode with two CPUs available to the Python fixture and
`nice -n 19`; they use generated CPU bytes and no model download, GPU or rental.

`cargo test -p tensorfs-core --lib --release profile_derived_transaction_scaling -- --ignored --nocapture --test-threads=1`

The native profile varies declaration keys, unrelated transaction rows and checkpoint
count separately. The 219-key case contains 52,381 bytes of work metadata; the 3,968-key
case contains 940,895 bytes. These are synthetic declarations, not an exact H3 header.

`scripts/profile-runtime-writer.py --mode=broker --keys=3968 --writes=10 --fanout=4`

Run the second command using a Runtime Python environment, with the selected TensorFS
build and Runtime source on `PYTHONPATH`. `--mode=native` bypasses the broker only for
an explicitly labeled comparison. The broker uses the actual StoreProxy, WriterBroker,
bounded spool and native writer. Its exchange callback is in process, and supervisor
pressure admission is disabled in this fixture. Thus FD4 latency, process scheduling
and remote/NFS behavior are not measured. The 40 writes deliberately reuse the same
16 KiB body to isolate metadata overhead and payload deduplication.

Raw artifacts are in `outputs/derived-write-profile-20260913/` in the workspace. Python
profiles emit JSON including operation spans, counts, `/proc/self/io` deltas and RSS.
The two `*-untraced.json` samples were run sequentially; separate `*.strace` files are
syscall counts and must not be used as unperturbed latency measurements.

| Four-writer broker loop | Baseline | Scoped row |
| --- | ---: | ---: |
| Add-part time | 13.601 s | 4.482 s |
| Explicit checkpoint time | 15.465 s | 7.804 s |
| Combined | 29.066 s | 12.285 s |
| Bytes read through syscalls | 981,541,992 | 259,651,828 |
| Bytes written through syscalls | 352,941,756 | 120,967,552 |
| Peak process RSS | 118,072 KiB | 114,276 KiB |

The measured combined improvement is 2.37x, with 74% fewer bytes read and 66% fewer
bytes written. OS page caching means syscall bytes are not physical disk traffic.
The separate traced full processes used 1,595 versus 1,431 fsyncs, and 174,902 versus
44,602 pwrite64 calls. They include fixture setup and teardown; JSON I/O deltas cover
only the part/checkpoint loop.

For fixed 219-key declarations and 20 native epoch observations, the baseline took
33 ms with one transaction and 3,799 ms with 128. After scoping, observations stayed
approximately 35–61 ms across 1/4/32/128 rows. Some timing runs overlapped compilation
on the shared host; exact small-duration ratios should not be treated as benchmarks.
The history slope and byte/operation counts are the reliable result.

## What this does not solve

The successful H3 VAE cast took 1,103.262 s for 219 tensors (9.689 GB source, 4.845 GB
new data). The patch has not been qualified by repeating that remote conversion and
does not establish a new full-model conversion time.

The single-writer checkpoint path still reconstructs prior deltas and repeatedly walks
its durability chain. With equally tiny parts, 10 native checkpoints took 0.938 s and
219 took 63.540 s before the patch; the latter still took 59.874 s afterward. Therefore
on-disk delta metadata is linear while total replay work remains quadratic. Each own
transaction row also still contains its immutable declaration and cumulative parts,
so reading/re-encoding that row is additional repeated work. tfs-114 tracks this.

The greenfield target is one native accepted-work journal, an immutable work context,
a small durable epoch fence and a disposable active-writer index. Open/restart validates
and reconstructs once; normal writes and exports process only their new work. Existing
native journals/deltas should be reused, not supplemented with a parallel framework.
Any active-chain cache must bind transaction/epoch/head and preserve fork, stale writer,
corruption, custody and delayed-ack checks. Per-part local acceptance and remote recovery
export must remain explicit; batching cannot silently weaken acknowledged durability.

Runtime's executor spool adds a disk copy in each direction. Profile the real admission
and process boundary before replacing it with descriptor/stream transfer. Such a change
must preserve size bounds, no-follow regular-file checks, immutable consumption and the
native digest authority. The H3 math helper also materializes a whole tensor plus error
arrays; bounded chunk casting can reduce RAM, but it must accumulate equivalent fidelity
metrics and refuse a failing part before acceptance. Neither redesign is in this patch.

## Correctness checks

- 35 native derived, source-graft and recovery tests passed; the opt-in profile is ignored
  during ordinary tests. They cover interrupted work, epoch fencing, missing/corrupt
  sources, forks, delayed acknowledgement, GC custody and explicit abandonment.
- The new isolation regression accepts an own part despite malformed unrelated history,
  preserves the other writer's bytes, reads while another SQL writer owns the write lock,
  and resumes the correct accepted roles under a newer epoch.
- 10 Runtime tests passed for real broker recovery, exact source grafts, disk preflight,
  resumed role accounting and native receipt reuse. Pytest reported cleanup warnings
  about pre-existing temporary test directories; those directories were not altered.
- `cargo clippy -p tensorfs-core --release --lib -- -D warnings`, source fences and Ruff
  passed. No published wheel, daemon, worker, checkpoint or model release was modified.

An ordinary default-home CLI run with the reviewed native build remains a separate
integration qualification before release. This source/profile PR does not claim it.
