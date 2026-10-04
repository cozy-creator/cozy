# Consume the actual running fact after a skipped live state

The machine #54 follow-up replays the actual journaled running revision/time in terminal
history. An observer may have last read queued/preparing before accepted work finished;
without the running fact, Creator still classifies subsequent progress as preparation.

Keep production classification unchanged: progress cannot authorize or infer a lifecycle
state. Add a focused records consumer proof starting queued, then consuming the real running
event shape before denoise 30/decoding and terminal. Check retained positions, terminal order,
reopened records and durable-cursor behavior. Source-only until CPU proof is available.

`TestQueuedConsumerUsesOnlyTheReplayedRunningFactForTerminalProgress` passes both arms:
missing running cannot infer lifecycle from telemetry, and the real running event shape
preserves denoise 30 before the terminal and after reopened records. The native machine #54
TLS/owned-process control independently proves that event is the actual journaled fact.

Combined CPU source e81f50e0 (integration df6 plus test-only #1028) passes records/API/machinev1/
output/canonical/cli packages. The actual authored SDK/managed-executor/ordinary CLI burst
and isolated controller restart passes in 5.93 s with machine 3e7c01d and explicit H370/f089 SDKs.
No production Creator behavior, durability or classification changed in this companion.

Tracker cozy-creator/tracker#322; companion machine PR #54. The existing ordinary authored SDK
burst/watch/controller-restart proof remains a separate regression gate. No model, VAE or
GPU qualification follows from records/TLS/CPU checks.
