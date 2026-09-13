# Captured dependency reuse

Owner: `/root/dependency_snapshot_fix`  
Branch: `fix/captured-dependency-selection-20260914`  
Base: `80e8ed42e8866f42cf7e2196b41ab2d0bc979419`

Selected root extras already travel in the captured wheel's flattened
Requires-Dist metadata. `CaptureUnpublishedClosure` and `CaptureWheel` both use
the existing active requirement selection and `PinDependencies`; no new resolver,
protocol field, or durable selection table is required.

Creator supplies the local Runtime with one cache path under the existing
`local-packages` owner: `dependency-objects`. The path travels as the controlled
`COZY_DEPENDENCY_CACHE` config value and is erased from device executors. Runtime
owns immutable payload validation and file reuse; Creator does not inspect or
decide which dependency bytes to link. Captured generations retain their own
hardlinks, so source and cache deletion cannot invalidate them.

Runtime implementation and qualification are in Runtime PR469. The new ordinary
CLI test selects a CUDA extra, executes two edited numerical scripts, verifies
shared immutable CPU/CUDA library inodes distinct from the SDK, then removes the
source/cache after stopping fixture processes and executes both retained captures.
This draft is not a shared CLI/Runtime installation or release.
