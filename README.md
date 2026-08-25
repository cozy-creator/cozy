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

`vectors/` is frozen DATA: `closure/` holds the platform EncodingSpec objects, `cases/<group>/`
holds each vector's exact stored bytes plus a `.verdict` (expected outcome, frozen ids).
Caps in `limits.rs` are PROVISIONAL until proto-001's real H3+SDXL corpus ratifies them.
