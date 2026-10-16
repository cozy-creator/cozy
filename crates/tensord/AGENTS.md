# Cozy machine

Module map and owners: `docs/ARCHITECTURE.md`.

TensorFS is this workspace's `crates/tensorfs`; never copy storage or model implementations.
Use uv for Python. Use CodeGraph first if an index exists; do not create one implicitly.

The CPU service never loads CUDA/NVML. GPU work or rentals require explicit session authority.
Records at boundaries are typed; peer versions never gate baseline operations. No time-based
process kills, hidden environment behavior switches, smaller substitute requests or mock
inference qualification.

Qualify vertical slices with actual process/socket/storage/inference paths and record their
limits. This repository is experimental until ordinary Cozy CLI, consumer and release gates pass.
