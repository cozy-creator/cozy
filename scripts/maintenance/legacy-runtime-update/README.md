# Legacy rental Runtime maintenance

This operator command updates the Python Runtime/TensorFS pair on rentals whose
agent predates `machine-bootstrap/1`. The existing agent process stays in place.
Use ordinary `cozy rental update` for machines with the current bootstrap.

From the Creator checkout, inspect the attached rental without changing it:

```sh
go run ./scripts/maintenance/legacy-runtime-update \
  --machine motonari --runtime-version 0.18.89 --tensorfs-version 0.3.78
```

After the requested releases are public and the rental is idle, append `--apply`
and copy `agent_version` and `agent_started_unix_ms` from that reviewed plan into
`--expect-agent-version` and `--expect-agent-started-unix-ms`. These preconditions
also apply when resuming observation after a disconnect.

Verified public wheels can be uploaded before the rental becomes idle with
repeatable `--stage-wheel SHA256=PATH` arguments and the same reviewed agent
identity flags. This only uses the authenticated wheel-staging endpoint; it does
not submit an update, claim maintenance or alter the active Python environment.
The helper verifies the local digest/version and the machine's returned receipt.
Omit `--apply` to finish with `staged_only`. The later version-based update reuses
the staged wheel when its filename and digest match the public package index.

The command uses the existing rental key and pinned certificate. Its operation
ID is stable for the rental, boot, and requested pair. Rerunning an interrupted
command observes that operation; another pending operation refuses. Failed or
rolled-back operations remain failures. Signals detach observation without
canceling the machine's update. The machine's existing guarded restart owns
activation and rollback; the script never runs an installer or stops a process.

Successful verification requires the requested pair, authenticated Runtime
readiness, unchanged worker/boot identity, and the same running agent version
and start time. Package SDK generations refresh on their next preparation.
Full agent/bootstrap adoption remains a separate service or image update.
