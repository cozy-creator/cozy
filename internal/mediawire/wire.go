// Package mediawire is the ONE declaration of the pod media plane's contract: the name the
// service answers to, the integer revision both ends must agree on before a byte moves,
// and the bounds the plane PUBLISHES instead of each end restating.
//
// It exists because the two ends ship separately and always have. The plane
// (`internal/podmedia`, served inside `cmd/cozy-pod`) is compiled into the pod image from a
// commit pin (the image recipe's CREATOR_COMMIT) while the owner's client in
// `internal/media` floats with master, and until now the only version
// signal between them was the literal `/v1/` in a URL path. A
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

// Service is what the pod's media plane calls itself in every answer that identifies it.
// It is the PLANE's name and not a binary's: cl-036 merged the plane into `cozy-pod` and
// deliberately did not touch this string, because the owner compares it before a byte
// moves and a rename would refuse every pod that had not been rebuilt.
const Service = "cozy-media"

// ContractRev is the media plane's wire revision. BUMP IT whenever a route, an answer
// field, or a published bound changes. Both ends compare it before any byte moves; a plane
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
const ContractRev = 2

// MaxReceiptBytes is the ceiling on the pod readiness envelope that
// `GET /v1/bootstrap/receipt` serves. It is THIS repo's number because the plane is what
// refuses an oversized one; `cmd/cozy-pod` writes that file and must not exceed it, and
// IMPORTS this constant rather than declaring a second, silently divergent copy of it —
// which is exactly what it did while it lived in another repo. The two are one process
// since cl-036 and the rule did not relax: one declaration, imported at both ends.
const MaxReceiptBytes = 64 << 10

// Rev answers the revision as a document carries it. It is a pointer so that "answered a
// revision" and "answered none" are different facts on the wire rather than one zero.
func Rev() *int { v := ContractRev; return &v }

// Health is the exact document `GET /v1/health` answers, and it is ONLY the contract: who
// is answering and at which revision. That is the whole question the route exists to
// settle — the owner's client asks it once, before it uploads, and refuses the rental on
// either mismatch (`internal/media.Client.Health`). It publishes nothing about what this
// server is holding, because a bound is enforced where it is checked and this plane
// already refuses an over-quota or oversized write with a typed refusal that states the
// numbers. Telling every authenticated caller the pod's disk layout and fill level ahead
// of time added a reader-less field to a shape both ends must agree on.
type Health struct {
	Service     string `json:"service"`
	ContractRev *int   `json:"contract_rev,omitempty"`
}

// Ours is this build's answer.
func Ours() Health { return Health{Service: Service, ContractRev: Rev()} }
