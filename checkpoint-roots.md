# Durable checkpoint input roots

Owner: Codex `/root/consumer_delivery_finish`.
Worktree: `~/cozy/.worktrees/tensorfs/checkpoint-root-retention`.
Branch: `feat/checkpoint-root-retention`; fetched base `8b3e2b2`.
Tracker: tfs-117; Runtime generic catalog Model input consumer follows cr-174.

An accepted standalone job must retain a catalog checkpoint independently of the
download/cache repository that supplied it. A read lease ends with its process;
`replace_local` requires optional snapshot siblings to be resident; ordinary tree
roots reject CozyTensors. None is the required durable runtime-closure pin.

The native Store API retains one exact repository checkpoint under an opaque
owner digest, inspect that root, and permanently release it. First acquisition
checks native repository membership and verified CozyTensors runtime closure.
Retry of an existing held root does not depend on a movable repository pointer.
Release creates a tombstone even before acquisition. Root files, not a Runtime
SQLite row, supply GC authority. This creates no tensor, producer or derived receipt.

The execution owner authorizes the explicit catalog repository in an input grant,
journals its own input obligation, and retains before acknowledging submission.
Native Store access stays privileged; the executor receives its usual model/source
capability. Existing owner-derived Model input handling remains separate. Public
0.3.42 is immutable. The 0.3.43 release candidate integrates merged H3 LoRA PR171/172
with this pin API; the dedicated tracker release ledger records its reservation.
The tag/public artifacts follow exact merged source and release validation.


Qualification: native pin source b5bf802 passed all six CI checks, eight native
controls, 210 native library cases and existing storage integration suites.
The ordinary Creator CLI on exact b5bf802/d051 development wheels selected the
real local/library-seed repository, read 512 values of 14, and verified the
released native checkpoint root without a fabricated derived input hold. Creator
source 662993b5 records that 122.31-second control. Runtime schema11 and public
TensorFS dependency-floor integration remain with the Runtime 0.18.1 owner.
