// Package mediawire is Cozy's expectation of Tensorhub's pod media HTTP
// contract. The server ships independently in the base worker image, so the
// client compares both service identity and revision before moving a byte.
package mediawire

// Service is what the pod's media plane calls itself in every answer that identifies it.
// It is the PLANE's name and not a binary's: cl-036 merged the plane into `pod-supervisor` and
// deliberately did not touch this string, because the owner compares it before a byte
// moves and a rename would refuse every pod that had not been rebuilt.
const Service = "cozy-media"

// ContractRev is the media plane's wire revision. BUMP IT whenever a route, an answer
// field, or a published bound changes incompatibly. Both ends compare it before
// any byte moves; a plane
// answering a different revision — or none, which is a pod older than this check — is
// refused at connect.
//
// Rev 2 narrowed the health answer to the two fields that have a reader. Rev 1
// also published `max_receipt_bytes`, `root`, `used_bytes`, `quota_bytes`,
// `max_object_bytes` and `plans`; nothing on either end ever read one of them. The ceiling
// crossed the wire because it once crossed a REPO boundary — `cozy-bootstrap` had to learn
// it without a credential — and cl-036 made the two ends one binary that reads
// MaxReceiptBytes below at compile time. The rest was operator telemetry no operator
// fetched. A field an authenticated route publishes and nobody reads is not free: it is
// pod state handed out on every pre-flight, and a shape both ends must keep agreeing on.
//
// Rev 3 added `GET /v1/triage/{subject}` (cl-101): the worker's triage bundle for one
// terminal attempt, read by the opaque subject the terminal named and bounded to the
// protocol's 1 MiB. Before it, a pod attempt's bundle stayed on the pod and the owner
// recorded `bundle_absent` for every remote failure.
const ContractRev = 3

// Health is the GET /v1/health contract. Scoped input support is negotiated
// separately so adding the safe upload route does not change the worker protocol
// or prevent an older client from connecting to a receiver without reclamation.
type Health struct {
	Service             string `json:"service"`
	ContractRev         *int   `json:"contract_rev,omitempty"`
	AttemptScopedInputs bool   `json:"attempt_scoped_inputs"`
}
