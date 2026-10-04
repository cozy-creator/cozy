# Direct ordinary run play

Playback reads this controller's existing v1 request/machine records, obtains the machine's
Status using the locally held owner key and TLS pin, and builds a run/output-scoped fragment
link. The run identifier remains its request id. Local machines are observation-only attaches;
rentals use their recorded endpoint/key and explicit endpoints retain their recorded selector.
No personal daemon upgrade or Hub read is required.

The Status supplies addresses and the fingerprint of that machine's media certificate. A
mismatched worker/boot/pin descriptor refuses only playback. URI fragment values are encoded,
so a run/output name cannot add another fragment parameter. Missing direct reachability is
an operation error; no relay is provided.

The paired machine endpoint change is cozy-machine PR #40. The product fixtures use actual
Rust machine-v1 jobs, owner keys, TLS pins, Pion ICE-TCP and Google Chrome. They exercise
completed and growing output delivery, the exact ordinary `cozy run play` link, reload and
reconnection from a cursor, range reads and seeking, replacement, malformed media, wrong
pins, expired capabilities and output scope. The network cut test uses a test-only TCP
forwarder to drop the connection; the product does not supply a relay.

CPU proof uses isolated authored homes and synthetic film bytes, an exact Runtime wheel
and its executor-plane TensorFS wheel, empty `CUDA_VISIBLE_DEVICES` and Chrome's
`--disable-gpu`. This establishes media transport and authority. Ordinary default-home
and rented inference/player qualification, observed provider port mappings, and actual
machine-v1 media throughput remain separate gates. `BenchmarkWebRTCGet` explicitly skips
because its numeric-run supervisor fixture no longer describes the current boundary.

The protobuf descriptor is advisory evolution: an older machine lacking it refuses only
playback. The client does not require a deployment SHA match.
