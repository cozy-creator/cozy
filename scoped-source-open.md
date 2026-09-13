# Scoped source/open handoff for cr-174

Owner: Codex /root/tensorfs_scoped_finish.
Worktree: ~/cozy/.worktrees/tensorfs/cr174-scoped-source-open.
Branch: feat/cr174-scoped-source-open.
Base: 91ee78442a956a27a788c666a9e41de4f0c38287 (fetched origin/master).

This increment lets Runtime move declaration/begin/replay/adopt and source
inspection into its trusted worker, removing the executor's StoreProxy dependency.
DerivedTransaction.accept(fd) asks the bound native channel whether its owner
selected a live writer or retained receipt. serve_derived_replay records that
receipt before success. SourceCapability.from_fd(manifest,length,fd) reads only
that exact source's native metadata/configs via serve_source; it owns its connected
FD and supports close/context-manager lifetime. Config payloads use bounded 1 MiB
frames, not Runtime spool files or the 64 KiB executor control lane. Native parsing,
identity/custody validation and geometry stay in the existing TensorFS core.

No native code, format or journal changes. The scoped-native test (including a
2 MiB config and replay handoff) and strict facade typing pass with the unchanged
native extension from source f0893ff / wheel SHA2562859c1398e227ceadb31d592a5d7d658a58a35b0bb6172d457913aafa3918ee4.
The ignored python/tensorfs/_ext.abi3.so symlink is a local test binding only.
An exact source-derived wheel, source-inspector UID boundary proof, malformed/fenced
source controls and final Runtime integration remain required. Existing writer UID
proof is not claimed to cover this new source inspector.
