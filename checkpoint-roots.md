# Durable checkpoint input roots

Owner: Codex `/root/consumer_delivery_finish`.
Worktree: `~/cozy/.worktrees/tensorfs/checkpoint-root-retention`.
Branch: `feat/checkpoint-root-retention`; fetched base `8b3e2b2`.
Tracker: tfs-117; Runtime generic catalog Model input consumer follows cr-174.

An accepted standalone job must retain a catalog checkpoint independently of the
download/cache repository that supplied it. A read lease ends with its process;
`replace_local` requires optional snapshot siblings to be resident; ordinary tree
roots reject CozyTensors. None is the required durable runtime-closure pin.

The native Store API will retain one exact repository checkpoint under an opaque
owner digest, inspect that root, and permanently release it. First acquisition
checks native repository membership and verified CozyTensors runtime closure.
Retry of an existing held root does not depend on a movable repository pointer.
Release creates a tombstone even before acquisition. Root files, not a Runtime
SQLite row, supply GC authority. This creates no tensor, producer or derived receipt.

The execution owner authorizes the explicit catalog repository in an input grant,
journals its own input obligation, and retains before acknowledging submission.
Native Store access stays privileged; the executor receives its usual model/source
capability. Existing owner-derived Model input handling remains separate. Public
0.3.42 is immutable; the current 0.3.43.dev1 cohort includes upstream H3 LoRA work
and its public release is coordinated independently of development wheel builds.
