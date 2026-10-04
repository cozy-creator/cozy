# Rental keepalive with an older controller daemon

A new CLI may be attached to a Rust rental while the default personal daemon still predates
`cozy.machine.v1`. The older daemon's keepalive request shape and worker RPC cannot carry
that operation. Installing a new daemon is not a prerequisite for a manual idle reset.

The CLI uses its existing `foregroundRental` capability check. When that daemon lacks v1,
it resets the named rental through `Status{keepalive:true}` using only the locally recorded
address, TLS pin, worker/boot identity and rental owner key. The same helper serves the
current daemon's keepalive path. The response must name the recorded worker and boot and
supply a future idle deadline before Creator records the acknowledgment.

This operation does not start the local machine, consult a Hub, create a second daemon,
retry an unacknowledged mutation, or release a rental. The live ordinary CLI check on the
root-owned rental is the product gate; compilation alone is insufficient.
