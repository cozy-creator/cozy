package hub

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/rentalid"
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
//	GET    /v1/private-rentals/{id}  -> {state, address, cert_pem, media_address,
//	                                     detail, renter_token_sha256:[...]}
//	DELETE /v1/private-rentals/{id}  -> 204
//
// `media_address` is now ALWAYS the hub's own word. The client used to derive it from the
// control address by a stated host:port+1 convention, because the pinned demo contract had
// no field for it; the product surface names it, so a guess about where somebody else's
// byte plane listens has no reason to exist and is gone.

// Rental states, the hub's own words.
const (
	RentalPendingAcquisition = "pending_acquisition"
	RentalAcquiring          = "acquiring"
	RentalMaterializing      = "materializing"
	RentalReady              = "ready"
	RentalFailed             = "failed"
	RentalReleaseRequested   = "release_requested"
	RentalReleased           = "released"
)

// Rental is one rented pod as the hub reports it. There is no token on it, and there is
// no route that adds one: this host has the token because this host minted it.
type Rental struct {
	ID      string
	State   string
	Address string
	CertPEM string
	Detail  string
	// MediaAddress is the pod's BYTE PLANE (cl-014, ruled #506b): the co-resident media
	// server's own listener, which is where an owner uploads a payload and downloads an
	// output. The hub observed it and names it.
	MediaAddress string
	// TokenSHA256 is the pod's LIVE credential set, as hashes. It is here so this host can
	// see that the hash of the token it minted is one the pod was provisioned with —
	// a comparison neither end can make by saying the token.
	TokenSHA256 []string
}

// Ready answers whether this rental carries a usable dial pair. The credential is not
// checked here because it is not the hub's: it never left this host.
func (r Rental) Ready() bool {
	return r.State == RentalReady && r.Address != "" && r.CertPEM != ""
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
	ID           string   `json:"rental_id"`
	State        string   `json:"state"`
	Address      string   `json:"address"`
	CertPEM      string   `json:"cert_pem"`
	Detail       string   `json:"detail"`
	MediaAddress string   `json:"media_address"`
	TokenSHA256  []string `json:"renter_token_sha256"`
}

var bareSHA256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)

func validateRentalID(id string) *exit.Error {
	if !rentalid.Valid(id) {
		return exit.Named(exit.Validation, "hub.rental_id_invalid",
			"the hub returned an invalid opaque rental_id").
			WithRemedy("the hub must mint a portable opaque name, never path syntax or a reserved device name")
	}
	return nil
}

func (w wireRental) rental(id string) Rental {
	if w.ID != "" {
		id = w.ID
	}
	return Rental{
		ID: id, State: w.State, Address: w.Address, CertPEM: w.CertPEM,
		Detail: w.Detail, MediaAddress: w.MediaAddress,
		TokenSHA256: w.TokenSHA256,
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
		idempotency: operationKey, bodyBytes: requestBody,
	}, &out)
	if e != nil {
		return Rental{}, e
	}
	if out.ID == "" {
		return Rental{}, exit.Named(exit.Internal, "hub.rental_unnamed",
			"the hub accepted the rental and named no rental_id").
			WithRemedy("this route may not exist on this hub build; `cozy hub status` names it and its version")
	}
	if e := validateRentalID(out.ID); e != nil {
		return Rental{}, e
	}
	return out.rental(""), nil
}

// Rental reads one rental's current state. Admin, like every first-party route.
func (c *Client) Rental(ctx context.Context, id string) (Rental, *exit.Error) {
	if e := validateRentalID(id); e != nil {
		return Rental{}, e
	}
	var out wireRental
	e := c.do(ctx, call{method: http.MethodGet, path: "/v1/private-rentals/" + url.PathEscape(id), admin: true}, &out)
	if e != nil {
		return Rental{}, e
	}
	if out.ID != "" && out.ID != id {
		return Rental{}, exit.Named(exit.Conflict, "hub.rental_id_changed",
			"the hub answered rental %s with identity %s", id, out.ID).
			WithRemedy("preserve the original rental identity; never attach the response under another id")
	}
	return out.rental(id), nil
}

// Release tears the pod down. The hub answers 204, so there is nothing to decode — and a
// 404 reaches the caller as the hub's own refusal rather than being swallowed as "already
// gone": this client cannot tell a released rental from one that was never ours.
func (c *Client) Release(ctx context.Context, id, reason string) *exit.Error {
	if e := validateRentalID(id); e != nil {
		return e
	}
	return c.do(ctx, call{
		method: http.MethodDelete, path: "/v1/private-rentals/" + url.PathEscape(id), admin: true, reason: reason,
	}, nil)
}
