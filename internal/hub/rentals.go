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
// provisions generic empty capacity, starts a worker hosting `WorkerControl` behind TLS,
// and answers with the address and certificate to pin. Creator sends desired package/model
// state directly to that worker after attachment.
//
// Creator mints one media bearer and one Ed25519 rental key before the paid ask. Only the
// bearer hash and public key cross this API. Tensorhub cannot open either pod door: the
// media bearer and rental private key stay here; the latter signs only ClaimProof.
//
// The contract is the hub's and is consumed verbatim, exactly as the catalog's is:
//
//	GET    /v1/rental-skus           -> [{name, accelerator_model, compute_capability,
//	                                 vram_gb, price_usd_micros_per_hour}]
//	POST   /v1/rentals               {sku, media_token_sha256:<64 hex>, creator_public_key}
//	                                 -> 202 {rental_id, state, ...}
//	GET    /v1/rentals/{id}          -> {state, worker_address, cert_pem, media_address,
//	                                     detail, worker_id, worker_boot_id,
//	                                     media_token_sha256:[...]}
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
	MediaTokenSHA256    []string
	HourlyRateUSDMicros int64
}

// ExactDocument and ModelSelection are package-download DTOs. Generic rental
// admission does not carry either type.
type ExactDocument struct {
	CanonicalBytes []byte `json:"canonical_bytes"`
	Digest         string `json:"digest"`
	Length         int64  `json:"length"`
}

type ModelSelection struct {
	ID       string `json:"id"`
	ModelRef string `json:"model_ref"`
	Lane     string `json:"lane"`
}

// Ready answers whether this rental carries the whole dial triple and an observed
// media credential set. The caller still checks that set contains its bearer hash; Ready only
// prevents a partial ready projection from being mistaken for a usable pod.
func (r Rental) Ready() bool {
	return r.State == RentalReady && r.Address != "" && r.MediaAddress != "" &&
		r.CertPEM != "" && r.WorkerID != "" && r.WorkerBootID != "" && r.CreatorPublicKey != "" &&
		len(r.MediaTokenSHA256) > 0
}

func (r Rental) Attachable() bool { return r.Ready() }

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
	ID                  string   `json:"rental_id"`
	State               string   `json:"state"`
	AcceleratorModel    string   `json:"requested_accelerator_model"`
	WorkerAddress       string   `json:"worker_address"`
	CertPEM             string   `json:"cert_pem"`
	Detail              string   `json:"detail"`
	MediaAddress        string   `json:"media_address"`
	WorkerID            string   `json:"worker_id"`
	WorkerBootID        string   `json:"worker_boot_id"`
	CreatorPublicKey    string   `json:"creator_public_key"`
	MediaTokenSHA256    []string `json:"media_token_sha256"`
	HourlyRateUSDMicros int64    `json:"hourly_rate_usd_micros"`
}

var bareSHA256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
var computeCapabilityPattern = regexp.MustCompile(`^[0-9]+\.[0-9]+$`)

const maxRentalResponseBytes = 1 << 20

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
		ID: w.ID, State: w.State,
		AcceleratorModel: w.AcceleratorModel,
		Address:          w.WorkerAddress, CertPEM: w.CertPEM,
		Detail: w.Detail, MediaAddress: w.MediaAddress,
		WorkerID: w.WorkerID, WorkerBootID: w.WorkerBootID,
		CreatorPublicKey:    w.CreatorPublicKey,
		MediaTokenSHA256:    w.MediaTokenSHA256,
		HourlyRateUSDMicros: w.HourlyRateUSDMicros,
	}
}

// RentalRequest is the closed provider-neutral product intent. Provider,
// datacenter, offer, image, cache volume, disk, and ports do not have fields
// here: Tensorhub resolves and selects them.
type RentalRequest struct {
	SKU              string `json:"sku"`
	MediaTokenSHA256 string `json:"media_token_sha256"`
	CreatorPublicKey string `json:"creator_public_key"`
}

// RentalRequestBytes authors the exact bytes persisted before POST and replayed
// unchanged after response loss. There is one encoder, not a digest struct plus
// a separately marshaled transport map that can drift.
func RentalRequestBytes(sku, mediaTokenSHA256, creatorPublicKey string) ([]byte, *exit.Error) {
	req := RentalRequest{
		SKU:              strings.TrimSpace(sku),
		MediaTokenSHA256: strings.TrimPrefix(strings.TrimSpace(mediaTokenSHA256), "sha256:"),
		CreatorPublicKey: strings.TrimSpace(creatorPublicKey),
	}
	public, publicErr := base64.RawURLEncoding.DecodeString(req.CreatorPublicKey)
	if req.SKU == "" ||
		!bareSHA256Pattern.MatchString(req.MediaTokenSHA256) || publicErr != nil || len(public) != 32 {
		return nil, exit.Named(exit.Validation, "rental.intent_incomplete",
			"sku, media token hash, and one Ed25519 Creator public key are required")
	}
	raw, err := json.Marshal(req)
	if err != nil {
		return nil, exit.Internalf("cannot encode the closed rental request: %s", err)
	}
	return raw, nil
}

// RentalSKU is one Cozy-priced product choice. Provider offer names and prices
// are deliberately absent: the caller rents from Tensorhub, not its adapter.
type RentalSKU struct {
	Name                  string `json:"name"`
	AcceleratorModel      string `json:"accelerator_model"`
	ComputeCapability     string `json:"compute_capability"`
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
			!computeCapabilityPattern.MatchString(sku.ComputeCapability) ||
			sku.VRAMGB <= 0 || sku.PriceUSDMicrosPerHour <= 0 || seen[sku.Name] {
			return nil, exit.Named(exit.Conflict, "hub.rental_catalog_invalid",
				"the hub returned an invalid or duplicate rental SKU %q", sku.Name).
				WithRemedy("Tensorhub must publish unique names with model, compute capability, positive VRAM, and positive Cozy price")
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
	if w.HourlyRateUSDMicros <= 0 {
		return exit.Named(exit.Conflict, "hub.rental_hourly_rate_missing",
			"the hub %s without a positive locked Cozy retail hourly rate", what).
			WithRemedy("upgrade Tensorhub before accepting a rental")
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
