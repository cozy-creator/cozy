package hub

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/rentalid"
)

// The RENTAL routes (cl-015, on th-042's product surface). The hub owns the pod: it
// provisions one, installs the byte-identical release this client installed, starts a
// worker hosting `WorkerControl` behind TLS, and answers with the address and the
// certificate to pin. This client provisions nothing and knows no provider; it asks, it
// polls, and it releases.
//
// THE THIRD PIECE OF THE DIAL TRIPLE IS NOT THE HUB'S TO GIVE (#495e). The RENTER mints
// the access token and sends only its sha256; the hub stores the hash, provisions the pod
// with the hash, and has no column, no route and no code path that could hand a token
// back. That is what makes the hub structurally incapable of authenticating to a pod it
// rented out. A lost token is not recovered — it is re-minted by rotating the hash set.
//
// The contract is the hub's and is consumed verbatim, exactly as the catalog's is:
//
//	POST   /v1/private-rentals       {endpoint_ref, accelerator_model,
//	                                 renter_token_sha256:[<64 hex>]}
//	                                 -> 202 {rental_id, state, ...}
//	GET    /v1/private-rentals/{id}  -> {state, worker_address, cert_pem, media_address,
//	                                     detail, renter_token_sha256:[...],
//	                                     control_snapshot:{digest,length,canonical_bytes}}
//	POST   /v1/private-rentals/{id}/worker-session-observations
//	                                     -> renter-authenticated deterministic WorkerFrames
//	DELETE /v1/private-rentals/{id}  -> 204
//
// `media_address` is now ALWAYS the hub's own word. The client used to derive it from the
// control address by a stated host:port+1 convention, because the pinned demo contract had
// no field for it; the product surface names it, so a guess about where somebody else's
// byte plane listens has no reason to exist and is gone.

// Rental states, the hub's own words — only the ones a reader here branches on. The
// hub's in-flight states (pending_acquisition, acquiring, materializing) reach this
// client as opaque strings it renders verbatim; naming them as constants nobody read
// was law-13 dead code (cl-028).
const (
	RentalConverging       = "converging"
	RentalReady            = "ready"
	RentalFailed           = "failed"
	RentalReleaseRequested = "release_requested"
	RentalReleased         = "released"
)

// Rental is one rented pod as the hub reports it. There is no token on it, and there is
// no route that adds one: this host has the token because this host minted it.
type Rental struct {
	ID          string
	State       string
	EndpointRef string
	Address     string
	CertPEM     string
	Detail      string
	// MediaAddress is the pod's BYTE PLANE (cl-014, ruled #506b): the co-resident media
	// server's own listener, which is where an owner uploads a payload and downloads an
	// output. The hub observed it and names it.
	MediaAddress string
	// TokenSHA256 is the pod's LIVE credential set, as hashes. It is here so this host can
	// see that the hash of the token it minted is one the pod was provisioned with —
	// a comparison neither end can make by saying the token.
	TokenSHA256 []string
	// ControlSnapshot is Tensorhub's exact acquisition-attempt control truth. Its
	// canonical bytes were persisted before provider Create; Creator verifies and
	// stores those same bytes before publishing a remote target.
	ControlSnapshot   *ExactControlDocument
	PlacementRevision uint64
}

// ExactControlDocument is one bounded exact-byte document transported by the
// existing ready rental view. CanonicalBytes is base64 on JSON; Digest and
// Length fence the decoded bytes.
type ExactControlDocument struct {
	CanonicalBytes []byte `json:"canonical_bytes"`
	Digest         string `json:"digest"`
	Length         int64  `json:"length"`
}

// Ready answers whether this rental carries the whole dial triple and an observed
// credential set. The caller still checks that set contains ITS token hash; Ready only
// prevents a partial ready projection from being mistaken for a usable pod.
func (r Rental) Ready() bool {
	return r.State == RentalReady && r.Address != "" && r.MediaAddress != "" &&
		r.CertPEM != "" && len(r.TokenSHA256) > 0 && r.ControlSnapshot != nil &&
		r.PlacementRevision > 0
}

// Attachable answers whether Tensorhub has published the complete, receipt-pinned
// private control projection. Converging is deliberately not ready: it exists so the
// renter's RecordOwner can claim the private WorkerControl service and return observed
// convergence evidence without giving Tensorhub the renter credential or a second live
// control owner.
func (r Rental) Attachable() bool {
	return (r.State == RentalConverging || r.State == RentalReady) &&
		r.Address != "" && r.MediaAddress != "" && r.CertPEM != "" &&
		len(r.TokenSHA256) > 0 && r.ControlSnapshot != nil && r.PlacementRevision > 0
}

// HoldsHash answers whether the hub's live set carries this hash — the renter's own
// check that the pod it is about to dial was provisioned with the token it holds. Both
// spellings are accepted because both exist: the wire's bare hex and the pod hash file's
// `sha256:` line are one fact, and a comparison that knew only one would read as a
// mismatch on the other.
func (r Rental) HoldsHash(hash string) bool {
	bare := strings.TrimPrefix(hash, "sha256:")
	for _, h := range r.TokenSHA256 {
		if strings.TrimPrefix(h, "sha256:") == bare {
			return true
		}
	}
	return false
}

// wireRental is the answer's own shape.
type wireRental struct {
	ID                string                `json:"rental_id"`
	State             string                `json:"state"`
	EndpointRef       string                `json:"endpoint_ref"`
	WorkerAddress     string                `json:"worker_address"`
	CertPEM           string                `json:"cert_pem"`
	Detail            string                `json:"detail"`
	MediaAddress      string                `json:"media_address"`
	TokenSHA256       []string              `json:"renter_token_sha256"`
	ControlSnapshot   *ExactControlDocument `json:"control_snapshot"`
	PlacementRevision uint64                `json:"placement_revision"`
}

var bareSHA256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

// A persisted control snapshot is bounded to 64 MiB decoded by Tensorhub. Its
// base64 members expand in the surrounding JSON, so the rental view gets one
// explicit 96 MiB transport cap rather than widening every hub response.
const maxRentalResponseBytes = 96 << 20

func validateRentalID(id string) *exit.Error {
	if !rentalid.Valid(id) {
		return exit.Named(exit.Validation, "hub.rental_id_invalid",
			"the hub returned an invalid opaque rental_id").
			WithRemedy("the hub must mint a portable opaque name, never path syntax or a reserved device name")
	}
	return nil
}

func (w wireRental) rental() Rental {
	return Rental{
		ID: w.ID, State: w.State, EndpointRef: w.EndpointRef,
		Address: w.WorkerAddress, CertPEM: w.CertPEM,
		Detail: w.Detail, MediaAddress: w.MediaAddress,
		TokenSHA256: w.TokenSHA256, ControlSnapshot: w.ControlSnapshot,
		PlacementRevision: w.PlacementRevision,
	}
}

// RentalRequest is the closed provider-neutral product intent. Provider,
// datacenter, offer, image, cache volume, disk, and ports do not have fields
// here: Tensorhub resolves and selects them.
type RentalRequest struct {
	EndpointRef       string   `json:"endpoint_ref"`
	AcceleratorModel  string   `json:"accelerator_model"`
	RenterTokenSHA256 []string `json:"renter_token_sha256"`
}

// RentalRequestBytes authors the exact bytes persisted before POST and replayed
// unchanged after response loss. There is one encoder, not a digest struct plus
// a separately marshaled transport map that can drift.
func RentalRequestBytes(endpointRef, acceleratorModel, tokenSHA256 string) ([]byte, *exit.Error) {
	req := RentalRequest{
		EndpointRef: strings.TrimSpace(endpointRef), AcceleratorModel: strings.TrimSpace(acceleratorModel),
		RenterTokenSHA256: []string{strings.TrimPrefix(strings.TrimSpace(tokenSHA256), "sha256:")},
	}
	if req.EndpointRef == "" || req.AcceleratorModel == "" ||
		!bareSHA256Pattern.MatchString(req.RenterTokenSHA256[0]) {
		return nil, exit.Named(exit.Validation, "rental.intent_incomplete",
			"endpoint_ref, accelerator_model, and one 64-character lowercase renter_token_sha256 are required")
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, exit.Internalf("cannot encode the closed rental request: %s", err)
	}
	return raw, nil
}

// Rent asks for a pod, presenting the HASH of a token the caller has already minted. It
// returns as soon as the hub has ACCEPTED the ask — provisioning is the hub's work and
// this client watches it through `Rental`, because a POST that blocked until a pod booted
// would be a request whose failure mode is a lost id.
//
// `tokenSHA256` is the BARE 64 lowercase hex — the spelling the hub's field takes; the
// `sha256:` prefix is the token-hash FILE's spelling and belongs to the pod, not the wire.
// The token itself is not an argument here, in this package, or anywhere on this wire: a
// hub that was sent a plaintext credential would be a hub that could use it.
// Rent retransmits the exact canonical bytes the caller persisted before the
// paid mutation. The hub authority is bound separately by the local operation
// digest; a retry against another hub therefore conflicts before this method.
func (c *Client) Rent(ctx context.Context, requestBody []byte, reason, operationKey string) (Rental, *exit.Error) {
	var out wireRental
	e := c.do(ctx, call{
		method: http.MethodPost, path: "/v1/private-rentals", admin: true, reason: reason,
		idempotency: operationKey, bodyBytes: requestBody, responseBytes: maxRentalResponseBytes,
	}, &out)
	if e != nil {
		return Rental{}, e
	}
	if e := out.named("accepted the rental"); e != nil {
		return Rental{}, e
	}
	return out.rental(), nil
}

func (w wireRental) named(what string) *exit.Error {
	if w.ID == "" {
		return exit.Named(exit.Internal, "hub.rental_unnamed",
			"the hub %s and named no rental_id", what).
			WithRemedy("this route may not exist on this hub build; `cozy endpoint search` names it and its version")
	}
	return validateRentalID(w.ID)
}

// Rental reads one rental's current state. Admin, like every first-party route.
func (c *Client) Rental(ctx context.Context, id string) (Rental, *exit.Error) {
	if e := validateRentalID(id); e != nil {
		return Rental{}, e
	}
	var out wireRental
	e := c.do(ctx, call{method: http.MethodGet, path: "/v1/private-rentals/" + url.PathEscape(id),
		admin: true, responseBytes: maxRentalResponseBytes}, &out)
	if e != nil {
		return Rental{}, e
	}
	if e := out.named("answered rental " + id); e != nil {
		return Rental{}, e
	}
	if out.ID != id {
		return Rental{}, exit.Named(exit.Conflict, "hub.rental_id_changed",
			"the hub answered rental %s with identity %s", id, out.ID).
			WithRemedy("preserve the original rental identity; never attach the response under another id")
	}
	return out.rental(), nil
}

// ArtifactGrant is one complete, expiring authorization for the immutable subjects
// Tensorhub selected for a rental. Identity remains in PlacementSet/2; this answer carries
// only access locations and can therefore be refreshed without changing desired state.
type ArtifactGrant struct {
	GrantID       string                  `json:"grant_id"`
	Subjects      []ArtifactGrantSubject  `json:"subjects"`
	Locations     []ArtifactGrantLocation `json:"locations"`
	ExpiresAtUnix uint64                  `json:"expires_at_unix"`
}

type ArtifactGrantSubject struct {
	Digest    string `json:"digest"`
	Kind      string `json:"kind"`
	Length    uint64 `json:"length"`
	SubjectID string `json:"subject_id"`
}

type ArtifactGrantLocation struct {
	Digest string `json:"digest"`
	URL    string `json:"url"`
}

type RentalArtifactGrant struct {
	GrantRevision uint64        `json:"grant_revision"`
	Grant         ArtifactGrant `json:"grant"`
}

// WorkerSessionObservation is the renter RecordOwner's exact evidence relay. The three
// normal frames are deterministic protobuf encodings of WorkerFrame and must be supplied
// together with the DesiredWorkerState revision that owner issued. BootFailure is the
// closed alternative: it is authenticated worker evidence before desired state exists.
// []byte is intentionally used here because encoding/json transports it as base64 without
// inventing a parallel textual protobuf representation.
type WorkerSessionObservation struct {
	ClaimAck        []byte `json:"claim_ack_base64,omitempty"`
	Snapshot        []byte `json:"snapshot_base64,omitempty"`
	ObservedState   []byte `json:"observed_state_base64,omitempty"`
	BootFailure     []byte `json:"boot_failure_base64,omitempty"`
	DesiredRevision uint64 `json:"desired_revision,omitempty"`
}

type WorkerSessionObservationResult struct {
	State             string `json:"state"`
	DesiredRevision   uint64 `json:"desired_revision"`
	AcceptedRevision  uint64 `json:"accepted_revision"`
	ConvergedRevision uint64 `json:"converged_revision"`
	Detail            string `json:"detail,omitempty"`
}

// ObserveWorkerSession relays evidence obtained by the renter's already-authenticated
// WorkerControl stream. WithToken supplies that same rental owner credential to this
// route; Tensorhub never receives the plaintext through rental creation or readback and
// never dials the private worker itself.
func (c *Client) ObserveWorkerSession(ctx context.Context, id string,
	observation WorkerSessionObservation) (WorkerSessionObservationResult, *exit.Error) {
	var out WorkerSessionObservationResult
	if e := validateRentalID(id); e != nil {
		return out, e
	}
	boot := len(observation.BootFailure) > 0
	normal := len(observation.ClaimAck) > 0 && len(observation.Snapshot) > 0 &&
		len(observation.ObservedState) > 0 && observation.DesiredRevision > 0
	if boot == normal || boot && (len(observation.ClaimAck) > 0 || len(observation.Snapshot) > 0 ||
		len(observation.ObservedState) > 0 || observation.DesiredRevision > 0) {
		return out, exit.Internalf("private worker observation requires exactly boot failure or the complete claim/snapshot/observed/revision set")
	}
	e := c.do(ctx, call{
		method: http.MethodPost,
		path:   "/v1/private-rentals/" + url.PathEscape(id) + "/worker-session-observations",
		admin:  true, reason: "cozy RecordOwner worker session observation",
		body: observation,
	}, &out)
	if e != nil {
		return out, e
	}
	if out.State != RentalConverging && out.State != RentalReady {
		return out, exit.Named(exit.Conflict, "rental.worker_observation_invalid",
			"Tensorhub accepted private worker evidence but answered state %q", out.State)
	}
	if boot && out.State != RentalConverging {
		return out, exit.Named(exit.Conflict, "rental.worker_observation_invalid",
			"Tensorhub answered %s to a typed worker boot failure", out.State)
	}
	if !boot && (out.DesiredRevision != observation.DesiredRevision ||
		out.AcceptedRevision > out.DesiredRevision || out.ConvergedRevision > out.AcceptedRevision ||
		out.State == RentalReady && (out.AcceptedRevision != out.DesiredRevision ||
			out.ConvergedRevision != out.DesiredRevision)) {
		return out, exit.Named(exit.Conflict, "rental.worker_observation_invalid",
			"Tensorhub answered impossible convergence revisions desired=%d accepted=%d converged=%d",
			out.DesiredRevision, out.AcceptedRevision, out.ConvergedRevision)
	}
	return out, nil
}

const maxArtifactGrantResponseBytes = 64 << 20

// ArtifactGrant asks Tensorhub for fresh, rental-scoped locations using the renter token
// that already authenticates Claim. The request is deliberately only revision plus TTL:
// callers cannot choose, omit, or substitute a subject selected by Tensorhub.
func (c *Client) ArtifactGrant(ctx context.Context, id string, revision, ttlSeconds uint64,
	reason string) (RentalArtifactGrant, *exit.Error) {
	var out RentalArtifactGrant
	if e := validateRentalID(id); e != nil {
		return out, e
	}
	if revision == 0 || ttlSeconds == 0 {
		return out, exit.Internalf("rental artifact grant requires non-zero revision and TTL")
	}
	e := c.do(ctx, call{
		method:        http.MethodPost,
		path:          "/v1/private-rentals/" + url.PathEscape(id) + "/artifact-grants",
		admin:         true,
		reason:        reason,
		responseBytes: maxArtifactGrantResponseBytes,
		body: struct {
			GrantRevision uint64 `json:"grant_revision"`
			TTLSeconds    uint64 `json:"ttl_seconds"`
		}{GrantRevision: revision, TTLSeconds: ttlSeconds},
	}, &out)
	if e != nil {
		return out, e
	}
	if out.GrantRevision != revision {
		return out, artifactGrantInvalid("answered revision %d for requested revision %d",
			out.GrantRevision, revision)
	}
	if e := validateArtifactGrant(out.Grant); e != nil {
		return out, e
	}
	return out, nil
}

func artifactGrantInvalid(format string, args ...any) *exit.Error {
	return exit.Named(exit.Conflict, "rental.artifact_grant_invalid", format, args...).
		WithRemedy("release this rental; Creator will not repair or widen Tensorhub's artifact authority")
}

func validateArtifactGrant(grant ArtifactGrant) *exit.Error {
	if grant.GrantID == "" || len(grant.Subjects) == 0 ||
		grant.ExpiresAtUnix <= uint64(time.Now().Unix()) {
		return artifactGrantInvalid("the grant header is incomplete or already expired")
	}
	locations := make(map[string]string, len(grant.Locations))
	prior := ""
	for i, location := range grant.Locations {
		if !validDigest(location.Digest) || location.URL == "" ||
			(i > 0 && location.Digest <= prior) {
			return artifactGrantInvalid("artifact location %d is malformed, duplicated, or unsorted", i)
		}
		locations[location.Digest] = location.URL
		prior = location.Digest
	}
	prior = ""
	for i, subject := range grant.Subjects {
		if !validDigest(subject.Digest) || subject.SubjectID == "" || subject.Kind == "" ||
			subject.Length == 0 || (i > 0 && subject.Digest <= prior) {
			return artifactGrantInvalid("artifact subject %d is malformed, duplicated, or unsorted", i)
		}
		if locations[subject.Digest] == "" {
			return artifactGrantInvalid("artifact subject %s has no acquisition location", subject.Digest)
		}
		prior = subject.Digest
	}
	if len(locations) != len(grant.Subjects) {
		return artifactGrantInvalid("the grant has %d subjects and %d locations",
			len(grant.Subjects), len(locations))
	}
	return nil
}

func validDigest(spelled string) bool {
	raw := strings.TrimPrefix(spelled, "sha256:")
	decoded, err := hex.DecodeString(raw)
	return strings.HasPrefix(spelled, "sha256:") && err == nil && len(decoded) == 32 &&
		raw == strings.ToLower(raw)
}

// Release asks the hub to tear the pod down. The hub answers 204 and nothing is decoded;
// a 404 reaches the caller as NotFound, and the caller decides what absence means.
func (c *Client) Release(ctx context.Context, id, reason string) *exit.Error {
	if e := validateRentalID(id); e != nil {
		return e
	}
	return c.do(ctx, call{
		method: http.MethodDelete, path: "/v1/private-rentals/" + url.PathEscape(id), admin: true, reason: reason,
	}, nil)
}
