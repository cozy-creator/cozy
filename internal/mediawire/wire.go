// Package mediawire is Cozy's expectation of Tensorhub's pod media HTTP
// contract. The server ships independently in the base worker image, so the
// client checks service identity and a minimum revision before moving a byte.
package mediawire

// Service is what the pod's media plane calls itself in every answer that identifies it.
// It is the PLANE's name and not a binary's: cl-036 merged the plane into `pod-supervisor` and
// deliberately did not touch this string, because the owner compares it before a byte
// moves and a rename would refuse every pod that had not been rebuilt.
const Service = "cozy-media"

// ContractRev is the media plane's wire revision this host speaks, and the oldest plane it
// reads. BUMP IT whenever a route, an answer field, a required request parameter, or a
// published bound changes.
const ContractRev = 3

// Health is the GET /v1/health contract. `contract_rev` is the only negotiation on
// this plane; route and request shapes belong to it, never to capability flags.
type Health struct {
	Service     string `json:"service"`
	ContractRev *int   `json:"contract_rev,omitempty"`
}
