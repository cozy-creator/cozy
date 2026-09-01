package hub

// th-094. A private package revision is unpublished, editable code that only ever existed on
// this machine. It used to reach a rented pod by being relayed through the pod supervisor as
// 1 MiB control frames -- up to 1 GiB an operation. It now goes where every other package byte
// goes, and the pod is handed a read capability instead of the bytes.

import (
	"context"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy/internal/exit"
)

// MaxPrivateWheelGrants is the whole revision: one project wheel and its bounded dependency
// closure. It is the same bound the worker protocol and the pod ledger carry.
const MaxPrivateWheelGrants = 33

// PrivateWheelRequest names one wheel by the digest this machine measured.
type PrivateWheelRequest struct {
	Digest   string `json:"digest"`
	Filename string `json:"filename"`
	Kind     string `json:"kind"`
	Length   int64  `json:"length"`
}

type PrivateWheelUpload struct {
	URL             string            `json:"url"`
	RequiredHeaders map[string]string `json:"required_headers"`
}

// PrivateWheelGrant answers for exactly one wheel, and the two capabilities are mutually
// exclusive: the store either holds the object, in which case there is a read grant, or it does
// not, in which case there is a write grant. Asking again after uploading is what turns the
// first into the second.
type PrivateWheelGrant struct {
	PrivateWheelRequest
	Present  bool                `json:"present"`
	Upload   *PrivateWheelUpload `json:"upload,omitempty"`
	Download string              `json:"download_url,omitempty"`
}

type privateWheelGrants struct {
	Files []PrivateWheelGrant `json:"files"`
}

// GrantPrivateWheels is one round of the loop: ask, PUT what is missing, ask again. The call is
// stateless at the hub, so a re-grant after a download capability ages out is simply another
// round rather than a new operation.
func (c *Client) GrantPrivateWheels(ctx context.Context, org string,
	files []PrivateWheelRequest, reason string,
) ([]PrivateWheelGrant, *exit.Error) {
	if len(files) == 0 || len(files) > MaxPrivateWheelGrants {
		return nil, exit.Usagef("a private package revision names 1..%d wheels", MaxPrivateWheelGrants)
	}
	var out privateWheelGrants
	problem := c.do(ctx, call{method: http.MethodPost,
		path: "/v1/private-packages/" + url.PathEscape(org) + "/grants",
		auth: true, reason: reason, body: map[string]any{"files": files},
		byBytes: true, strict: true}, &out)
	if problem != nil {
		return nil, problem
	}
	if len(out.Files) != len(files) {
		return nil, exit.Named(exit.Internal, "private_package.grant_set_changed",
			"Tensorhub answered for %d of %d private wheels", len(out.Files), len(files))
	}
	for i, grant := range out.Files {
		if grant.Digest != files[i].Digest || grant.Length != files[i].Length {
			return nil, exit.Named(exit.Internal, "private_package.grant_identity_changed",
				"Tensorhub answered for a private wheel that was not asked for")
		}
		if grant.Present == (grant.Upload != nil) || (grant.Present != (grant.Download != "")) {
			return nil, exit.Named(exit.Internal, "private_package.grant_shape_invalid",
				"Tensorhub answered for private wheel %s with neither one capability nor the other",
				grant.Filename)
		}
	}
	return out.Files, nil
}
