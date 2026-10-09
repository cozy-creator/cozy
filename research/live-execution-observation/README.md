Live execution observation
==========================

Run 5026 completed all seven segments and finished on its worker at 03:26:04.988 UTC, but retained progress stopped at Segment4/check_output at 03:21:49.699. A TCP read timeout stopped its observer. Its outcome was observed at 03:29:19.981; recipient collection completed at 03:30:31.707. This is an observation failure, not stalled inference.

Accepted direct-stream transport failures should use the same bounded-backoff, fresh authenticated Status and same-ID/cursor nil-spec reattachment path already used for expiring authority. Initial submission uncertainty must not authorize resubmission; detach, terminal rentals and genuine revocation keep their existing boundaries.

Add execution_elapsed_ms for the current running interval, based only on Runtime's retained started_unix_ms. This advancing display is separate from measured execution_ms and final benchmark duration. Unknown starts, queued/paused/terminal states remain distinct, and resumes use their own current interval.

Required validation: real TLS/separate-daemon recovery with one spec, nil-spec reattachment and zero controls; ordinary CLI list/show timer progression, source timestamp provenance, pause/resume and authoritative terminal timing. No live restart or rental/worker mutation is part of this change until coordinated by root.

Implemented behavior and proof
------------------------------

Direct transport errors now enter accepted-only observation recovery with the existing1–30s backoff. Fresh Status must identify the same worker before another nil-spec/cursor attach. First-submission uncertainty, permanent refusals, terminal rentals and explicit detach remain separate. Acceptance is reread after a stream ends so running work is not falsely described as waiting for a rental.

List and show render a separate `execution_elapsed_ms` wall clock for the current running interval. Only Runtime's explicit `started_unix_ms` starts it. Missing/future timestamps, queued/paused/terminal states do not invent elapsed execution; resumed intervals use their own start. Measured `execution_ms` remains authoritative and separate. The root-call report also uses the retained Runtime start rather than its slightly later client receipt timestamp.

The base code fails the real TLS disconnect/deadline recovery and live-timer regressions (49.366s). Corrected auth/transport/revocation/detach coverage passed43.128s; final TCP/timer/uncertain-first-reply coverage passed11.470s, with one spec, one nil-spec reattachment and zero controls. Internal tests, full build and vet pass. Independent peer timer and TCP recovery checks also pass.

Incident evidence is in outputs/run-5026-timing-20261009. Source start03:07:48.754 and finish03:26:04.988 give measured18m16.234s; observation followed at03:29:19.981 and collection at03:30:31.707. Flash-attn3's retained producer reports217366.5ms compilation; its broader observed wait was03:09:31.936 through cache publication03:14:26.509. Download progress labels span03:07:59.840–03:09:14.161. These windows are not independent additive costs. The later live Lancer job was preserved.
