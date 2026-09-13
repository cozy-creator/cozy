# Scoped model reader for cr-174

Owner: Codex /root/tensorfs_native_resume.
Worktree: ~/cozy/.worktrees/tensorfs/cr174-scoped-model-reader.
Branch: feat/cr174-scoped-model-reader.
Base: ec4eb9022fd85be8737c22342bd6ec4a7e361c5f (fetched origin/master).

Extend the existing exact-source FD with read-only native lease checks and bounded
semantic part reads. Native source inspection exposes opaque encoded tensor equality
identities. Runtime will consume this one API and delete its parallel model-reader
geometry, per-view metadata/part spool files, broker and WeightsSource classes.
Numerical float32 decoding remains a Runtime convenience over the same source handle.

Use content-addressed development wheels for the coordinated reader/Runtime/Eval proof.
TensorFS0.3.41 remains immutable; no release version or tag is assigned by this note.
