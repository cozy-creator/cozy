# tensorfs (v2)

Canonical tensor byte plane: CAS, cozytensors headers, snapshots, ingest. Design authority:
[tracker-v2](https://github.com/cozy-creator/tracker-v2) `tensorfs.md` / `cozytensors.md`; issues `tracker/tensorfs/`.
Rust workspace: `tensorfs-core` (torch/CUDA/HTTP/SQL-free), `tensorfs-layout`, `tensorfs-py` (marshal-only PyO3, later).
No automated tests (decisions.md #160): verification = live runs + benchmarks.
