# Renewing an expiring machine stream

The client signs a short machine capability for each RPC. A long read, upload or update
can outlive that capability while its signing key remains authorized. The machine ends
only that stream with UNAUTHENTICATED and `capability_expired`; it does not cancel work.

Only this specific expiry allows transparent reattachment. Key revocation and all other
authorization failures remain refusals. A new RPC obtains a newly signed capability.

An output read resumes at the number of bytes actually written, pins the first snapshot's
revision, and retains the original metadata. If the output changed, the revision check
refuses instead of splicing different revisions. The reader verifies the advertised length
and treats short bytes or a short destination write as an error.

An upload probes the actual staged length with a fresh RPC, reopens/seeks the source and
continues there. The source's content digest still names the whole object. This does not
replay inference. An accepted update reattaches to the same id and cursor without a spec.

The gRPC checks in `internal/machinev1/cap_renewal_test.go` exercise these transport endings,
exact byte continuation, revocation refusal and one-time update submission. They do not
qualify inference, the Rust server's authorization decision, or ordinary CLI deployment.
