package hub

// th-094. A local package revision is unpublished, editable code that only ever existed on
// this machine. It used to reach a rented pod by being relayed through the pod supervisor as
// 1 MiB control frames -- up to 1 GiB an operation. It now goes where every other package byte
// goes, and the pod is handed a read capability instead of the bytes.

import (
	"context"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy/internal/exit"
)

// MaxLocalWheelGrants is the whole revision: one project wheel and its bounded dependency
// closure. It is the same bound the worker protocol and the pod ledger carry.
const MaxLocalWheelGrants = 33

// LocalWheelRequest names one wheel by the digest this machine measured.
type LocalWheelRequest struct {
	Digest   string `json:"digest"`
	Filename string `json:"filename"`
	Kind     string `json:"kind"`
	Length   int64  `json:"length"`
}

type LocalWheelUpload struct {
	URL             string            `json:"url"`
	RequiredHeaders map[string]string `json:"required_headers"`
}

// LocalWheelGrant answers for exactly one wheel, and the two capabilities are mutually
// exclusive: the store either holds the object, in which case there is a read grant, or it does
// not, in which case there is a write grant. Asking again after uploading is what turns the
// first into the second.
type LocalWheelGrant struct {
	LocalWheelRequest
	Present  bool              `json:"present"`
	Upload   *LocalWheelUpload `json:"upload,omitempty"`
	Download string            `json:"download_url,omitempty"`
}

type localWheelGrants struct {
	Files []LocalWheelGrant `json:"files"`
}

// GrantLocalWheels is one round of the loop: ask, PUT what is missing, ask again. The call is
// stateless at the hub, so a re-grant after a download capability ages out is simply another
// round rather than a new operation.
func (c *Client) GrantLocalWheels(ctx context.Context, org string,
	files []LocalWheelRequest, reason string,
) ([]LocalWheelGrant, *exit.Error) {
	if len(files) == 0 || len(files) > MaxLocalWheelGrants {
		return nil, exit.Usagef("a local package revision names 1..%d wheels", MaxLocalWheelGrants)
	}
	var out localWheelGrants
	problem := c.do(ctx, call{method: http.MethodPost,
		path: "/v1/local-packages/" + url.PathEscape(org) + "/grants",
		auth: true, reason: reason, body: map[string]any{"files": files},
		byBytes: true, strict: true}, &out)
	if problem != nil {
		return nil, problem
	}
	if len(out.Files) != len(files) {
		return nil, exit.Named(exit.Internal, "local_package.grant_set_changed",
			"Tensorhub answered for %d of %d local wheels", len(out.Files), len(files))
	}
	for i, grant := range out.Files {
		if grant.Digest != files[i].Digest || grant.Length != files[i].Length {
			return nil, exit.Named(exit.Internal, "local_package.grant_identity_changed",
				"Tensorhub answered for a local wheel that was not asked for")
		}
		if grant.Present == (grant.Upload != nil) || (grant.Present != (grant.Download != "")) {
			return nil, exit.Named(exit.Internal, "local_package.grant_shape_invalid",
				"Tensorhub answered for local wheel %s with neither one capability nor the other",
				grant.Filename)
		}
	}
	return out.Files, nil
}
