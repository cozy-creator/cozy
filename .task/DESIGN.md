# Root Runtime writes to a cache owned by the Host

Solitar downloaded the 33.118 GB Qwen checkpoint locally, but none of its new sampled files appeared on the mounted cache. The Runtime process was UID 0; the NFS directories were UID/GID 65532 with mode 0755. A read-only write-access check failed as root and succeeded as 65532, consistent with NFS root squashing. Host already initializes the local TensorFS Store as 65532, so that Store establishes the existing ownership contract.

The fix belongs to TensorFS's optional cache publication door. A privileged caller temporarily adopts the local Store owner's filesystem UID/GID for the current native thread's synchronous cache operation. It does not change real/effective process IDs, other threads, cache permissions, or the Runtime environment. Non-root callers retain their current identity. Reads and local admission continue through the existing TensorFS APIs.

The Linux-only nix dependency supplies safe setfsuid/setfsgid wrappers. Rustix currently has thread-specific effective-ID setters, but no filesystem-ID wrappers; using nix avoids adding unsafe code to TensorFS or changing process-visible effective IDs. Both syscalls silently return the prior ID even on failure, so admission and restoration explicitly read back both IDs. The guard cannot move to another thread. Drop restores IDs on ordinary returns, errors, and Rust unwinding. An impossible-to-recover restoration failure aborts rather than letting a reusable Runtime thread continue under the wrong identity.

No identity comes from remote cache metadata or an untrusted request. The configured cache path remains unchanged. Existing root-squashed NFS ownership is preserved. Failed cache writes remain optional and cannot discard verified local model data.

Required remote qualification: successful write as the Store owner, credential restoration on success/error/unwind, a concurrent thread retaining its identity, non-root behavior, and actual cached Qwen objects appearing on Solitar followed by an origin-free read on a replacement Store. Do not claim the current implementation is proven before these checks and a native wheel build succeed.

Sources: https://docs.rs/nix/0.31.3/nix/unistd/fn.setfsuid.html and https://man7.org/linux/man-pages/man2/setfsuid.2.html.

## Qualification attempt

Manual narrow GitHub run https://github.com/cozy-creator/tensorfs/actions/runs/36216925860 was rejected before starting any step on 2026-09-26: "The job was not started because an Actions budget is preventing further use." This provides no compilation or test evidence. The draft must remain unqualified until the job runs remotely or a coordinated CPU build slot is available on an already-active worker. No local CI fallback or live credential/permission changes were made.
