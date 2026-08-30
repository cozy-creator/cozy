package hub

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/rentalid"
)

// The RENTAL routes (cl-015, on th-042's product surface). The hub owns the pod: it
// provisions one, installs the byte-identical release this client installed, starts a
// worker hosting `WorkerControl` behind TLS, and answers with the address and the
// certificate to pin. This client provisions nothing and knows no provider; it asks, it
// polls, and it releases.
//
// Creator mints one media bearer and one Ed25519 rental key before the paid ask. Only the
// bearer hash and public key cross this API. Tensorhub cannot open either pod door: the
// media bearer stays here and the Creator private key signs ClaimProof/ArtifactDelegation.
//
// The contract is the hub's and is consumed verbatim, exactly as the catalog's is:
//
//	GET    /v1/rental-skus           -> {skus:[{name, accelerator_model, vram_gb,
//	                                 price_usd_micros_per_hour}]}
//	POST   /v1/rentals               {package_ref, sku,
//	                                 media_token_sha256:<64 hex>, creator_public_key}
//	                                 -> 202 {rental_id, state, ...}
//	GET    /v1/rentals/{id}          -> {state, worker_address, cert_pem, media_address,
//	                                     detail, worker_id, worker_boot_id,
//	                                     media_token_sha256:[...],
//	                                     control_snapshot:{digest,length,canonical_bytes}}
//	POST   /v1/rentals/{id}/worker-observations
//	                                     -> renter-authenticated deterministic WorkerFrames
//	DELETE /v1/rentals/{id}          -> 204
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
	RentalDegraded         = "degraded"
	RentalFailed           = "failed"
	RentalReleaseRequested = "release_requested"
	RentalReleased         = "released"
)

// Rental is one rented pod as the hub reports it. No plaintext media bearer or private
// Creator key is part of this view.
type Rental struct {
	ID               string
	State            string
	PackageRef       string
	AcceleratorModel string
	Address          string
	CertPEM          string
	Detail           string
	// MediaAddress is the pod's BYTE PLANE (cl-014, ruled #506b): the co-resident media
	// server's own listener, which is where an owner uploads a payload and downloads an
	// output. The hub observed it and names it.
	MediaAddress     string
	WorkerID         string
	WorkerBootID     string
	CreatorPublicKey string
	// MediaTokenSHA256 is the pod media plane's LIVE credential set, as hashes. It is here so this host can
	// see that the hash of the token it minted is one the pod was provisioned with —
	// a comparison neither end can make by saying the token.
	MediaTokenSHA256 []string
	// Selection is the exact PlacementSet and its three small referenced documents.
	// It is the same DTO package install returns for a local venv.
	Selection         *PackageSelection
	PlacementRevision uint64
}

type ExactDocument struct {
	CanonicalBytes []byte `json:"canonical_bytes"`
	Digest         string `json:"digest"`
	Length         int64  `json:"length"`
}

type PackageSelection struct {
	Profile           string        `json:"profile"`
	PlacementSet      ExactDocument `json:"placement_set"`
	PackageRelease    ExactDocument `json:"package_release"`
	PackageDescriptor ExactDocument `json:"package_descriptor"`
	Qualification     ExactDocument `json:"qualification"`
}

// Ready answers whether this rental carries the whole dial triple and an observed
// media credential set. The caller still checks that set contains its bearer hash; Ready only
// prevents a partial ready projection from being mistaken for a usable pod.
func (r Rental) Ready() bool {
	return r.State == RentalReady && r.Address != "" && r.MediaAddress != "" &&
		r.CertPEM != "" && r.WorkerID != "" && r.WorkerBootID != "" && r.CreatorPublicKey != "" &&
		len(r.MediaTokenSHA256) > 0 && r.Selection != nil &&
		r.PlacementRevision > 0
}

// Attachable answers whether Tensorhub has published the complete, receipt-pinned
// private control projection. Converging is deliberately not ready: it exists so the
// renter's RecordOwner can claim the private WorkerControl service and return observed
// convergence evidence without giving Tensorhub the media bearer or a second live
// control owner.
func (r Rental) Attachable() bool {
	return (r.State == RentalConverging || r.State == RentalReady) &&
		r.Address != "" && r.MediaAddress != "" && r.CertPEM != "" &&
		r.WorkerID != "" && r.WorkerBootID != "" && r.CreatorPublicKey != "" && len(r.MediaTokenSHA256) > 0 &&
		r.Selection != nil && r.PlacementRevision > 0
}

// HoldsMediaHash answers whether the pod media plane's live set carries this hash.
// Both bare and sha256-prefixed spellings describe the same value.
func (r Rental) HoldsMediaHash(hash string) bool {
	bare := strings.TrimPrefix(hash, "sha256:")
	for _, h := range r.MediaTokenSHA256 {
		if strings.TrimPrefix(h, "sha256:") == bare {
			return true
		}
	}
	return false
}

// wireRental is the answer's own shape.
type wireRental struct {
	ID                string            `json:"rental_id"`
	State             string            `json:"state"`
	PackageRef        string            `json:"package_ref"`
	AcceleratorModel  string            `json:"requested_accelerator_model"`
	WorkerAddress     string            `json:"worker_address"`
	CertPEM           string            `json:"cert_pem"`
	Detail            string            `json:"detail"`
	MediaAddress      string            `json:"media_address"`
	WorkerID          string            `json:"worker_id"`
	WorkerBootID      string            `json:"worker_boot_id"`
	CreatorPublicKey  string            `json:"creator_public_key"`
	MediaTokenSHA256  []string          `json:"media_token_sha256"`
	Selection         *PackageSelection `json:"selection"`
	PlacementRevision uint64            `json:"desired_revision"`
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
		ID: w.ID, State: w.State, PackageRef: w.PackageRef,
		AcceleratorModel: w.AcceleratorModel,
		Address:          w.WorkerAddress, CertPEM: w.CertPEM,
		Detail: w.Detail, MediaAddress: w.MediaAddress,
		WorkerID: w.WorkerID, WorkerBootID: w.WorkerBootID,
		CreatorPublicKey: w.CreatorPublicKey,
		MediaTokenSHA256: w.MediaTokenSHA256, Selection: w.Selection,
		PlacementRevision: w.PlacementRevision,
	}
}

// RentalRequest is the closed provider-neutral product intent. Provider,
// datacenter, offer, image, cache volume, disk, and ports do not have fields
// here: Tensorhub resolves and selects them.
type RentalRequest struct {
	PackageRef       string           `json:"package_ref"`
	ModelSelections  []ModelSelection `json:"model_selections"`
	SKU              string           `json:"sku"`
	MediaTokenSHA256 string           `json:"media_token_sha256"`
	CreatorPublicKey string           `json:"creator_public_key"`
}

type ModelSelection struct {
	ID       string `json:"id"`
	ModelRef string `json:"model_ref"`
	Lane     string `json:"lane"`
}

// RentalRequestBytes authors the exact bytes persisted before POST and replayed
// unchanged after response loss. There is one encoder, not a digest struct plus
// a separately marshaled transport map that can drift.
func RentalRequestBytes(packageRef string, models []ModelSelection, sku, mediaTokenSHA256,
	creatorPublicKey string) ([]byte, *exit.Error) {
	req := RentalRequest{
		PackageRef: strings.TrimSpace(packageRef), SKU: strings.TrimSpace(sku),
		MediaTokenSHA256: strings.TrimPrefix(strings.TrimSpace(mediaTokenSHA256), "sha256:"),
		CreatorPublicKey: strings.TrimSpace(creatorPublicKey),
		ModelSelections:  append([]ModelSelection{}, models...),
	}
	if e := validateModelSelections(req.ModelSelections); e != nil {
		return nil, e
	}
	public, publicErr := base64.RawURLEncoding.DecodeString(req.CreatorPublicKey)
	if req.PackageRef == "" || req.SKU == "" ||
		!bareSHA256Pattern.MatchString(req.MediaTokenSHA256) || publicErr != nil || len(public) != 32 {
		return nil, exit.Named(exit.Validation, "rental.intent_incomplete",
			"package_ref, sku, media token hash, and one Ed25519 Creator public key are required")
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, exit.Internalf("cannot encode the closed rental request: %s", err)
	}
	return raw, nil
}

func PlacementRequestBytes(packageRef string, models []ModelSelection) ([]byte, *exit.Error) {
	request := struct {
		PackageRef      string           `json:"package_ref"`
		ModelSelections []ModelSelection `json:"model_selections"`
	}{PackageRef: strings.TrimSpace(packageRef), ModelSelections: append([]ModelSelection{}, models...)}
	if request.PackageRef == "" {
		return nil, exit.Named(exit.Validation, "rental.intent_incomplete", "package_ref is required")
	}
	if e := validateModelSelections(models); e != nil {
		return nil, e
	}
	raw, err := json.Marshal(request)
	if err != nil {
		return nil, exit.Internalf("cannot encode placement request: %s", err)
	}
	return raw, nil
}

func validateModelSelections(models []ModelSelection) *exit.Error {
	prior := ""
	for _, model := range models {
		if strings.TrimSpace(model.ID) == "" || strings.TrimSpace(model.ModelRef) == "" ||
			strings.TrimSpace(model.Lane) == "" || model.ID <= prior {
			return exit.Named(exit.Validation, "rental.model_selections_invalid",
				"model selections must be complete, unique, and sorted by id")
		}
		prior = model.ID
	}
	return nil
}

type RentalPlacementResult struct {
	RentalID        string           `json:"rental_id"`
	PackageRef      string           `json:"package_ref"`
	DesiredRevision uint64           `json:"desired_revision"`
	Selection       PackageSelection `json:"selection"`
}

func (c *Client) ReplaceRentalPlacement(ctx context.Context, id string, body []byte,
	operationKey string) (RentalPlacementResult, *exit.Error) {
	var out RentalPlacementResult
	if e := validateRentalID(id); e != nil {
		return out, e
	}
	e := c.do(ctx, call{method: http.MethodPut,
		path: "/v1/rentals/" + url.PathEscape(id) + "/placement", auth: true,
		reason: "cozy rental placement replacement", idempotency: operationKey,
		bodyBytes: body, strict: true, responseBytes: maxRentalResponseBytes}, &out)
	if e != nil {
		return out, e
	}
	if out.RentalID != id || out.PackageRef == "" || out.DesiredRevision < 2 || out.Selection.Profile == "" {
		return out, exit.Named(exit.Conflict, "rental.placement_response_invalid",
			"Tensorhub returned an invalid rental placement replacement")
	}
	return out, nil
}

// RentalSKU is one Cozy-priced product choice. Provider offer names and prices
// are deliberately absent: the caller rents from Tensorhub, not its adapter.
type RentalSKU struct {
	Name                  string `json:"name"`
	AcceleratorModel      string `json:"accelerator_model"`
	VRAMGB                int64  `json:"vram_gb"`
	PriceUSDMicrosPerHour int64  `json:"price_usd_micros_per_hour"`
}

// RentalSKUs reads Tensorhub's public product catalog.
func (c *Client) RentalSKUs(ctx context.Context) ([]RentalSKU, *exit.Error) {
	var out []RentalSKU
	if e := c.do(ctx, call{method: http.MethodGet, path: "/v1/rental-skus"}, &out); e != nil {
		return nil, e
	}
	seen := map[string]bool{}
	for _, sku := range out {
		if strings.TrimSpace(sku.Name) == "" || strings.TrimSpace(sku.AcceleratorModel) == "" ||
			sku.VRAMGB <= 0 || sku.PriceUSDMicrosPerHour <= 0 || seen[sku.Name] {
			return nil, exit.Named(exit.Conflict, "hub.rental_catalog_invalid",
				"the hub returned an invalid or duplicate rental SKU %q", sku.Name).
				WithRemedy("Tensorhub must publish unique names with model, positive VRAM, and positive Cozy price")
		}
		seen[sku.Name] = true
	}
	return out, nil
}

// Rent asks for a pod, presenting the media bearer HASH and Creator public key minted before the ask. It
// returns as soon as the hub has ACCEPTED the ask — provisioning is the hub's work and
// this client watches it through `Rental`, because a POST that blocked until a pod booted
// would be a request whose failure mode is a lost id.
//
// `mediaTokenSHA256` is the BARE 64 lowercase hex — the spelling the hub's field takes; the
// `sha256:` prefix is the token-hash FILE's spelling and belongs to the pod, not the wire.
// The bearer itself is not an argument here or anywhere on this wire.
// Rent retransmits the exact canonical bytes the caller persisted before the
// paid mutation. The hub authority is bound separately by the local operation
// digest; a retry against another hub therefore conflicts before this method.
func (c *Client) Rent(ctx context.Context, requestBody []byte, reason, operationKey string) (Rental, *exit.Error) {
	var out wireRental
	e := c.do(ctx, call{
		method: http.MethodPost, path: "/v1/rentals", auth: true, reason: reason,
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
			WithRemedy("this route may not exist on this hub build; `cozy package search` names it and its version")
	}
	return validateRentalID(w.ID)
}

// Rental reads one rental's current state using the renter's account authority.
func (c *Client) Rental(ctx context.Context, id string) (Rental, *exit.Error) {
	if e := validateRentalID(id); e != nil {
		return Rental{}, e
	}
	var out wireRental
	e := c.do(ctx, call{method: http.MethodGet, path: "/v1/rentals/" + url.PathEscape(id),
		auth: true, responseBytes: maxRentalResponseBytes}, &out)
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
	DesiredRevision uint64 `json:"-"`
}

type WorkerSessionObservationResult struct {
	State             string `json:"state"`
	DesiredRevision   uint64 `json:"desired_revision"`
	AcceptedRevision  uint64 `json:"accepted_revision"`
	ConvergedRevision uint64 `json:"converged_revision"`
	Detail            string `json:"detail,omitempty"`
}

// ObserveWorkerSession relays evidence obtained by the Creator-authenticated
// WorkerControl stream. Tensorhub never dials the private worker itself.
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
		path:   "/v1/rentals/" + url.PathEscape(id) + "/worker-observations",
		auth:   true, reason: "cozy RecordOwner worker session observation",
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

// Release asks the hub to tear the pod down. The hub answers 204 and nothing is decoded;
// a 404 reaches the caller as NotFound, and the caller decides what absence means.
func (c *Client) Release(ctx context.Context, id, reason string) *exit.Error {
	if e := validateRentalID(id); e != nil {
		return e
	}
	return c.do(ctx, call{
		method: http.MethodDelete, path: "/v1/rentals/" + url.PathEscape(id), auth: true, reason: reason,
	}, nil)
}
