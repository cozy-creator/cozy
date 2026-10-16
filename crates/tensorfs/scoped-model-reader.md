# Scoped model reader for cr-174

Owner: Codex /root/tensorfs_native_resume.
Worktree: ~/cozy/.worktrees/tensorfs/cr174-scoped-model-reader.
Branch: feat/cr174-scoped-model-reader.
Base: ec4eb9022fd85be8737c22342bd6ec4a7e361c5f (TensorFS 0.3.41).

The existing exact-source descriptor now exposes native lease checks and bounded
semantic part reads. Source inspection exposes opaque encoded tensor equality
identities. Runtime commit 9b0263c8 consumes this API and deletes its parallel
model-reader geometry, metadata/part spool files, broker and WeightsSource classes.
Numerical float32 decoding remains a Runtime convenience over the same source handle.

Metadata inspection does not pin bytes. Check or read acquires the native lease;
closing the handle releases it. Resident reads require no disk-write admission.
The channel validates the accepted source identity before I/O and keeps the native
lease through read completion. No persisted header or journal format changes.

The implementation at 03244db7019f passed all six CI checks. Installed-wheel tests
cover 2 MiB reads, identities across renamed tensors, changed-byte inequality,
bounds/role/forged-source refusals, fencing, active-lease GC exclusion and post-close
collection. A real UID 65533 child reads geometry, identity and bytes through its
descriptor while denied access to the Store. Existing scoped writer/source/replay
and public facade/vector tests remain green. Eval's real numerical controls pass,
including unchanged-tensor zero reads and interrupted normalization replay.

The earlier source_prefixes CI failure was a fork-inherited recovery lock, reproduced
with a held child descriptor. The native guard remains strict. Process fixture
lifetimes are serialized, and the new test proves GC refuses until the child exits.

Version 0.3.42 is prepared as a genuine additive reader release. TensorFS 0.3.41
remains immutable. The release tag and published artifact verification are pending;
Runtime, ordinary CLI and consumer releases are separate delivery gates. The CLI
reader fixture uses content-addressed development wheels until those gates pass.
