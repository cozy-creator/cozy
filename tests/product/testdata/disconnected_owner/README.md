# Disconnected execution qualification

Owner: memoization `/root/parent_native_finish`, thread01a07938-0dc9-7822-9723-d34b590e5c22.
Worktree: proto051-disconnected-owner; base4f437be3; branch test/proto051-disconnected-owner-red.

This fixture will establish the current failure boundary through the ordinary
Creator CLI: an independently running private Host keeps the Python parent alive,
but a later child call waits while its client-side execution coordinator is absent.
Restoring the coordinator must reconcile the same request and complete the call.
Watcher detach, suspended controller, process death and remote worker restart are
separate controls; no successful current offline continuation is assumed.

Use a new task-only local Host container, exact digest/SDK/boot readback, an isolated
Creator home and provider-catalog fixture. No paid rental or existing controller is
used. All source/function invocations go through cozy run. Direct container reads
are supplementary observation only. Test markers are written by the ordinary
nonmemoized script, while its memoized scalar operations remain deterministic.

The later green implementation will reuse Creator on the private machine as the
single execution owner from initial submission. This test does not implement that
mode or provide independent publication authority.
