# Explicit software update targets

Candidate 3 condition 13, following `D2/STATE.md` and the owner handoff.

A new CLI must never send an empty update request to an older controller: older code interprets it as the newest published software and can downgrade a rental. The CLI reads the configured Hub target once and names the selected versions. With no target, it sends no update. The current controller treats an empty selection as a no-op and does not discover a newest release implicitly.

Persist the resolved Runtime/TensorFS pair before dispatch and reuse that intent and operation ID after restart. An uncertain accepted update must be attached before any new selection is made. Keep the historical request_id database column so supported older controllers can still write their records.

Adopt the saved D2 checkpoint selectively on current master, preserving the native-only client, durable cancellation/observation and recent warm binding changes. Verify actual update/restart behavior and a new CLI paired with the saved a2200172 controller. New release naming (Calcifer and tensord) is tracked separately in tracker #336.

Integration requires independent review, a PR merge, and `R/mergecheck.py`; no direct master push.
