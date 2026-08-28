// Package mediawire is the ONE declaration of the pod media plane's contract: the name the
// service answers to, the integer revision both ends must agree on before a byte moves,
// and the bounds the plane PUBLISHES instead of each end restating.
//
// It exists because the two ends ship separately and always have. `cmd/cozy-media` is
// compiled into the pod image from a commit pin (`execution-substrates/versions.env`,
// CREATOR_COMMIT) while the owner's client in `internal/media` floats with master, and
// until now the only version signal between them was the literal `/v1/` in a URL path. A
// renumbered answer field or a moved route would have been MISPARSED rather than refused.
// The control leg one plane over already forecloses exactly this on `pb.WireSchemaRev`
// (`internal/orchestrator/owner.go`); this is that check for the byte plane.
//
// A REVISION IS NOT A DOCUMENT KIND. This plane authors no canonical document and gets no
// `cozy.<name>/<N>` format name: proto-007 (#616.a) demoted the one it briefly had —
// an ephemeral HTTP body wearing a format name that a sibling binary substring-matched —
// and the fence reddens on any such literal outside the document registry. A liveness
// answer may carry a plain service identity and a revision, and that is all this is.
package mediawire

// Service is what the pod's media server calls itself in every answer that identifies it.
const Service = "cozy-media"

// ContractRev is the media plane's wire revision. BUMP IT whenever a route, an answer
// field, or a published bound changes. Both ends compare it before any byte moves; a plane
// answering a different revision — or none, which is a pod older than this check — is
// refused at connect.
const ContractRev = 1

// MaxReceiptBytes is the ceiling on the pod readiness envelope that
// `GET /v1/bootstrap/receipt` serves. It is THIS repo's number because cozy-media is the
// process that refuses an oversized one; cozy-bootstrap writes that file and must not
// exceed it, and reads the bound from `cozy-media --bounds` rather than declaring a
// second, silently divergent copy of it.
const MaxReceiptBytes = 64 << 10

// Rev answers the revision as a document carries it. It is a pointer so that "answered a
// revision" and "answered none" are different facts on the wire rather than one zero.
func Rev() *int { v := ContractRev; return &v }

// Contract is what both ends compare: who is answering, at which revision, under which
// published bound. `cozy-media --bounds` prints exactly this document.
type Contract struct {
	Service         string `json:"service"`
	ContractRev     *int   `json:"contract_rev,omitempty"`
	MaxReceiptBytes int64  `json:"max_receipt_bytes"`
}

// Ours is this build's contract.
func Ours() Contract {
	return Contract{Service: Service, ContractRev: Rev(), MaxReceiptBytes: MaxReceiptBytes}
}

// Health is the exact document `GET /v1/health` answers: the contract, then what this
// server is holding. The owner's client checks the contract half before it uploads.
type Health struct {
	Contract
	Root           string `json:"root,omitempty"`
	UsedBytes      int64  `json:"used_bytes"`
	QuotaBytes     int64  `json:"quota_bytes"`
	MaxObjectBytes int64  `json:"max_object_bytes"`
	Plans          bool   `json:"plans"`
}
