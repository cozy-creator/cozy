# tensorfs (v2)

Canonical tensor byte plane: CAS, cozytensors headers, snapshots, ingest. Design authority:
[tracker-v2](https://github.com/cozy-creator/tracker-v2) `tensorfs.md` / `cozytensors.md`; issues `tracker/tensorfs/`.
Rust workspace: `tensorfs-core` (torch/CUDA/HTTP/SQL-free), `tensorfs-layout`, `tensorfs-py` (marshal-only PyO3, later).
No automated tests (decisions.md #160): verification = live runs + benchmarks.

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

`vectors/` is frozen DATA: `closure/` holds the platform EncodingSpec objects, `cases/<group>/`
holds each vector's exact stored bytes plus a `.verdict` (expected outcome, frozen ids).
Caps in `limits.rs` are PROVISIONAL until proto-001's real H3+SDXL corpus ratifies them.
