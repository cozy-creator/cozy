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

The paired machine endpoint change is codex/player-endpoint-20261004. Actual Pion, ordinary
local/rental CLI and browser proof is still required; generated bindings or unit link checks
alone do not qualify playback or inference.
