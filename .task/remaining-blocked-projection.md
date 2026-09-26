# Remaining blocked producers and a narrow public-status follow-up

The permanent rental-loss producer is fixed by this PR. Other blocked producers remain unchanged:

| Producer | Current meaning | Proposed public projection |
| --- | --- | --- |
| records/events.go failQueuedRequest when retain_work is true; callers include machine_runs.go preparation failure and orchestrator/dispatch.go queued refusal | Preparation has stopped, retaining any usable local work. An updated script or explicit retry may be needed. | failed, with the existing typed reason and a separately derived retry_available flag. |
| orchestrator/dispatch.go requeue budget exhaustion | Automatic attempts have ended. Retained work does not mean another attempt is running. | failed, with retry budget reason and retained-work availability. |
| orchestrator/child_calls.go non-transient finalization/result-retention failure | Hard finalization failure. Unavailable errors already keep retrying in finalizing. | failed for this hard stop; waiting/queued for the existing active finalization-retry path. |
| Machine observer with a sent submission but no acceptance receipt | Acceptance remains ambiguous and automatic reconciliation is active. machineJobState already projects this particular blocked row as queued. | queued/waiting, with reconciling-acceptance reason; never imply safe manual re-submission. |

A blanket blocked-to-failed display replacement is insufficient. Public response status, list filters/counts and pagination, watch stopping/final verdict, and CLI grouping must share one projection. The raw historical event may remain request.blocked, but watch must still stop at the current manual failure and skip superseded old stop events. Public failed filters must also include internally blocked manual failures while excluding actively reconciling acceptance.

Implement this as a focused shared projection first, preserving existing retry admission, retained native custody, rental lifetime, and internal scheduling states. Separately audit any proposed durable state hard cut against those consumers. Keep paused as explicit user-paused work; it should not be relabeled failed. No broader projection or scheduling rewrite is included in the permanent-loss fix.
