Accepted-run observer recovery
==============================

After an accepted stream loses owner authorization, Creator probes Status before reopening the same run with a nil spec. The existing recovery returns silently on transient Status or rental-resolution failures, leaving manual `cozy run show` to recover the result.

This task adds capped-backoff recovery for those temporary read-only failures while retaining first-submission refusal, revoked-key backoff, machine identity checks and observer-only detach. No submission or control is authorized by retrying a probe. Real TLS tests must prove one submitted spec, the same run ID/cursor, nil-spec reattachment and zero controls.

Run5016 demonstrated observation stopping after an authorization error and later manual recovery. Its intermediate recovery-probe error was not retained, so that exact error is unknown; this change must be justified by its own baseline-failing regression. No live daemon or Hub changes while inference runs.
