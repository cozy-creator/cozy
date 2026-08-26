package hub

import (
	"context"
	"net/http"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/secret"
)

// The RENTAL routes (cl-015). The hub owns the pod: it provisions one, installs the
// byte-identical release this client installed, starts a worker hosting `WorkerControl`
// behind TLS, and answers with the DIAL TRIPLE — address, the certificate to pin, and the
// owner token the worker will check as `Claim.proof`. This client provisions nothing and
// knows no provider; it asks, it polls, and it releases.
//
// The contract is the hub's and is consumed verbatim, exactly as the catalog's is:
//
//	POST   /v1/rentals       {endpoint, card, duration_hint_s?} -> 202 {rental_id}
//	GET    /v1/rentals/{id}  -> {state, address, cert_pem, owner_token, pod_id, detail,
//	                             media_address?}
//	DELETE /v1/rentals/{id}  -> 204
//
// `media_address` is the ONE addition this lane makes to the consumed contract, and it is
// read as OPTIONAL: the pinned contract does not carry it yet, so a hub that omits it gets
// the stated convention in `MediaAddressOf` instead of a refusal.

// Rental states, the hub's own words. `ready` is the only one that carries a triple.
const (
	RentalProvisioning = "provisioning"
	RentalReady        = "ready"
	RentalFailed       = "failed"
)

// Rental is one rented pod as the hub reports it. The owner token arrives as a
// `secret.Value`, so a rendering of this struct — a log line, a `%v`, a `--json` — prints
// its digest and there is no verb that prints the value.
type Rental struct {
	ID      string
	State   string
	Address string
	CertPEM string
	Token   secret.Value
	PodID   string
	Detail  string
	// MediaAddress is the pod's BYTE PLANE (cl-014, ruled #506b): the co-resident media
	// server's own listener, which is where an owner uploads a payload and downloads an
	// output. See `MediaAddressOf` for what happens when the hub does not name one.
	MediaAddress string
}

// Ready answers whether this rental carries a usable dial triple.
func (r Rental) Ready() bool {
	return r.State == RentalReady && r.Address != "" && r.CertPEM != "" && r.Token.Present()
}

// wireRental is the answer's own shape. It exists so `owner_token` becomes a
// secret.Value at the boundary rather than living on as a string somebody could print.
type wireRental struct {
	ID           string `json:"rental_id"`
	State        string `json:"state"`
	Address      string `json:"address"`
	CertPEM      string `json:"cert_pem"`
	OwnerToken   string `json:"owner_token"`
	PodID        string `json:"pod_id"`
	Detail       string `json:"detail"`
	MediaAddress string `json:"media_address"`
}

func (w wireRental) rental(id string) Rental {
	if w.ID != "" {
		id = w.ID
	}
	r := Rental{
		ID: id, State: w.State, Address: w.Address, CertPEM: w.CertPEM,
		Token: secret.New(w.OwnerToken), PodID: w.PodID, Detail: w.Detail,
		MediaAddress: w.MediaAddress,
	}
	if r.MediaAddress == "" {
		r.MediaAddress = MediaAddressOf(r.Address)
	}
	return r
}

// MediaAddressOf DERIVES a pod's media address from its control address: the same host,
// the next port.
//
// It is a stated convention and not a discovery, and it is here because the hub's pinned
// rental contract has no `media_address` field yet. The alternative — refusing every
// rental the hub cannot describe fully — would make a byte plane the hub cannot yet
// announce impossible to reach at all, including on a pod that is running one. So: if the
// hub NAMES a media address this client uses it verbatim; if it does not, this convention
// applies and the pod either answers there or the first media call refuses typed, which is
// an observation rather than a guess that fails later as something else.
//
// The credential and the pinned certificate are deliberately the RENTAL's own — one
// provisioned identity per pod, two listeners. Splitting them is th-041's to do when the
// hub contract grows the field; nothing here mints anything.
func MediaAddressOf(control string) string {
	host, port, ok := strings.Cut(control, ":")
	if !ok {
		return ""
	}
	n, err := strconv.Atoi(port)
	if err != nil || n <= 0 || n >= 65535 {
		return ""
	}
	return host + ":" + strconv.Itoa(n+1)
}

// Rent asks for a pod. It returns as soon as the hub has ACCEPTED the ask — provisioning
// is the hub's work and this client watches it through `Rental`, because a POST that
// blocked until a pod booted would be a request whose failure mode is a lost id.
//
// The contract's optional `duration_hint_s` is NOT sent: nothing enforces it at either
// end, and a field that travels and decides nothing is the same bug as a flag that
// parses and is ignored. It comes back with the thing that acts on it.
func (c *Client) Rent(ctx context.Context, endpoint, card, reason string) (Rental, *exit.Error) {
	body := map[string]any{"endpoint": endpoint, "card": card}
	var out wireRental
	e := c.do(ctx, call{
		method: http.MethodPost, path: "/v1/rentals", admin: true, reason: reason, body: body,
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
	e := c.do(ctx, call{method: http.MethodGet, path: "/v1/rentals/" + id, admin: true}, &out)
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
		method: http.MethodDelete, path: "/v1/rentals/" + id, admin: true, reason: reason,
	}, nil)
}
