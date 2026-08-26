package hub

import (
	"context"
	"net/http"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
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
//	POST   /v1/private-rentals       {endpoint, card, renter_token_sha256:[<64 hex>]}
//	                                 -> 202 {rental_id, state, ...}
//	GET    /v1/private-rentals/{id}  -> {state, address, cert_pem, media_address, pod_id,
//	                                     detail, renter_token_sha256:[...]}
//	DELETE /v1/private-rentals/{id}  -> 204
//
// `media_address` is now ALWAYS the hub's own word. The client used to derive it from the
// control address by a stated host:port+1 convention, because the pinned demo contract had
// no field for it; the product surface names it, so a guess about where somebody else's
// byte plane listens has no reason to exist and is gone.

// Rental states, the hub's own words.
const (
	RentalProvisioning = "provisioning"
	RentalReady        = "ready"
	RentalFailed       = "failed"
	// The pod is being torn down, or is gone. Neither is a state a wait can outlast: a
	// rental that has left is not one that is still coming up.
	RentalReclaiming = "reclaiming"
	RentalDead       = "dead"
)

// Rental is one rented pod as the hub reports it. There is no token on it, and there is
// no route that adds one: this host has the token because this host minted it.
type Rental struct {
	ID      string
	State   string
	Address string
	CertPEM string
	PodID   string
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
	PodID        string   `json:"pod_id"`
	Detail       string   `json:"detail"`
	MediaAddress string   `json:"media_address"`
	TokenSHA256  []string `json:"renter_token_sha256"`
}

func (w wireRental) rental(id string) Rental {
	if w.ID != "" {
		id = w.ID
	}
	return Rental{
		ID: id, State: w.State, Address: w.Address, CertPEM: w.CertPEM,
		PodID: w.PodID, Detail: w.Detail, MediaAddress: w.MediaAddress,
		TokenSHA256: w.TokenSHA256,
	}
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
func (c *Client) Rent(ctx context.Context, endpoint, card, tokenSHA256, reason string) (Rental, *exit.Error) {
	body := map[string]any{
		"endpoint": endpoint, "card": card,
		"renter_token_sha256": []string{tokenSHA256},
	}
	var out wireRental
	e := c.do(ctx, call{
		method: http.MethodPost, path: "/v1/private-rentals", admin: true, reason: reason, body: body,
	}, &out)
	if e != nil {
		return Rental{}, e
	}
	if out.ID == "" {
		return Rental{}, exit.Named(exit.Internal, "hub.rental_unnamed",
			"the hub accepted the rental and named no rental_id").
			WithRemedy("this route may not exist on this hub build; `cozy hub status` names it and its version")
	}
	return out.rental(""), nil
}

// Rental reads one rental's current state. Admin, like every first-party route.
func (c *Client) Rental(ctx context.Context, id string) (Rental, *exit.Error) {
	var out wireRental
	e := c.do(ctx, call{method: http.MethodGet, path: "/v1/private-rentals/" + id, admin: true}, &out)
	if e != nil {
		return Rental{}, e
	}
	return out.rental(id), nil
}

// Release tears the pod down. The hub answers 204, so there is nothing to decode — and a
// 404 reaches the caller as the hub's own refusal rather than being swallowed as "already
// gone": this client cannot tell a released rental from one that was never ours.
func (c *Client) Release(ctx context.Context, id, reason string) *exit.Error {
	return c.do(ctx, call{
		method: http.MethodDelete, path: "/v1/private-rentals/" + id, admin: true, reason: reason,
	}, nil)
}
