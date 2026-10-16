# Filesystem credential guard keeps the parent-death signal

Linux clears a thread's `PR_SET_PDEATHSIG` whenever its fsUID/fsGID change. A root
Runtime opening a Store owned by another user therefore lost its SIGKILL-on-parent-death
for the rest of its life (probe: 9 → 0). `FilesystemIdentity` now captures the thread's
configured signal and parent, re-arms it after entry and after restoration (including
error and unwind), and delivers it if the parent changed while it was cleared. A
disabled signal stays disabled. Store authority, permissions and I/O paths are unchanged.

## Qualification (2026-10-02, CPU only)

- Root gate, rootful `python:3.12-slim` container, release fixture `9ae305d2…`:
  `python3 crates/tensorfs-core/tests/filesystem_parent_death_gate.py target/release/examples/filesystem_parent_death_gate`
  → 5 passed, 0 skipped. Same driver against the unfixed 0.3.93 source: 2 failed
  (`guard 0 != 9` in the foreign-owner cases), so the gate discriminates.
- As root: `filesystem_identity` unit test, `store_boundaries` (6, WAL probe supplied
  via `TFS_TEST_CATALOG_WAL_PROBE`) and `trust_index` (2) pass. Nonroot: same pass.
- Clippy: only the existing `machine.rs:82` warning.

## Limits

`getppid` cannot see the death of the creating *thread* inside a live parent; spawners
must create executors from a persistent thread. A parent outside the PID namespace reads
as 0 and is not detected.
