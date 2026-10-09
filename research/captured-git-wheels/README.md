Portable captured package dependencies
======================================

MiniMax H3 1.26.1 local capture exposed three failures before inference:

- Selected callable capture skipped declared pinned Git wheels, then called the missing artifact a registry platform mismatch. Source snapshots also left Git fetch/build work for workers without Git.
- Private StagePrepared used the publication registry-account policy with an empty publishing account and rejected an otherwise valid retained Hub dependency.
- Captured callable children synthesized exact client Runtime/TensorFS pins. Run 5024 failed pip check after the worker supplied its own compatible 0.19/0.4 pair; retained child metadata required 0.18.101/0.3.92. The failed environment was removed by installer cleanup, so its exact stderr was unavailable. Both working published H3 environments and the current kernel gate pass pip check.

Declared Git commits now become retained wheels inside the existing owned capture. The full fetched commit and built distribution/version are checked. Only the locked target selection is built; inactive extras stay unbuilt. Relative wheel sources and exact-version overrides replace transport in the copied metadata. A lock refresh may fetch missing index metadata, but comparison refuses changes to other locked identities or dependency edges. Original author files remain unchanged, and receivers need neither Git nor repository access.

Private wheel builds use their installed closure custody rules instead of publication account policy. Root and callable wheel metadata now share one SDK rule: preserve authored Runtime/TensorFS compatibility bounds, pin ordinary dependencies exactly, and let the receiving machine supply its SDK pair. Old invocation cache records trigger a fresh capture for new commands; accepted requests keep their retained inputs unchanged.

Validation
----------

- Base 8abcceee fails the selected Git capture and portable-source regressions; the SDK callable regression reproduces its exact equality pins.
- Real Git-server tests prove carried wheel bytes, a receiver with empty cache/no Git/offline uv sync, inactive-extra exclusion, a declared Git dependency through a local child, and changed locked commit refusal.
- Private Hub capture proves the old account-policy refusal and verifies retained private wheel custody. Public Git/refusal rules remain exercised.
- Callable SDK regression preserves authored extras and upper/lower ranges while removing synthesized equality pins; old capture cache lookup does not mutate retained records.
- Focused capture/publication/source tests pass (16.347s); final core tests pass (4.943s), and full build/vet pass.
- An ordinary candidate CLI describe of the original unchanged H3 Git/Hub source succeeds. It reproduces the exact published Diffusers wheel SHA 103ccb9e6f5a085dda13f2277da113ea29b2785c8aa051d195526b398ee70692; generated callable SDK metadata was inspected. Remote inference on the corrected capture remains root-owned qualification.

Evidence: outputs/captured-git-wheels-20261009. No live installation, Hub mutation or worker mutation is included in this task yet. The earlier manually vendored matrix snapshot is a diagnostic workaround, not the delivered workflow.
