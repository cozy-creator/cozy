package hub

// th-122 — the provider-volume routes, verbatim from tensorhub:
//
//	GET    /v1/volumes                                   (auth) -> {"volumes":[…]}
//	POST   /v1/volumes          {"provider"?, "datacenter"}  (auth) warm = ensure the standing volume exists
//	DELETE /v1/volumes/{id}                              (auth) drop
//
// A volume is entirely the hub's row: it has no minted credential, no pinned
// certificate, and nothing dispatches against it locally, so this client keeps
// NO local volume state — the hub is the one source.

import (
	"context"
	"net/http"
	"net/url"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Volume is one owner-scoped, disposable repo-object cache in one provider datacenter.
type Volume struct {
	ID               string `json:"id"`
	Provider         string `json:"provider"`
	Datacenter       string `json:"datacenter"`
	State            string `json:"state"`
	SizeGB           int64  `json:"size_gb"`
	USDMicrosPerHour int64  `json:"usd_micros_per_hour"`
	CreatedAt        string `json:"created_at"`
	LiveAt           string `json:"live_at,omitempty"`
	LastBoundAt      string `json:"last_bound_at,omitempty"`
	WarmBytes        int64  `json:"warm_bytes"`
	WarmObjects      int64  `json:"warm_objects"`
}

func (v Volume) valid() bool {
	return v.ID != "" && v.Provider != "" && v.Datacenter != "" && v.State != "" &&
		v.SizeGB > 0 && v.USDMicrosPerHour >= 0
}

// Volumes lists the caller's standing volumes.
func (c *Client) Volumes(ctx context.Context) ([]Volume, *exit.Error) {
	var out struct {
		Volumes []Volume `json:"volumes"`
	}
	if e := c.do(ctx, call{method: http.MethodGet, path: "/v1/volumes", auth: true}, &out); e != nil {
		return nil, e
	}
	for _, v := range out.Volumes {
		if !v.valid() {
			return nil, exit.Named(exit.Conflict, "hub.volume_invalid",
				"the hub returned an incomplete volume row %q", v.ID).
				WithRemedy("Tensorhub must list volumes with id, provider, datacenter, state, and size")
		}
	}
	return out.Volumes, nil
}

// WarmVolume ensures the optional cache volume exists in one datacenter. Pods
// populate it with immutable repo objects while acquiring model and dataset snapshots.
func (c *Client) WarmVolume(ctx context.Context, provider, datacenter, reason string) (Volume, *exit.Error) {
	if strings.TrimSpace(datacenter) == "" {
		return Volume{}, exit.Usagef("name the datacenter to warm")
	}
	body := map[string]string{"datacenter": datacenter}
	if provider != "" {
		body["provider"] = provider
	}
	var out Volume
	e := c.do(ctx, call{method: http.MethodPost, path: "/v1/volumes", auth: true,
		reason: reason, body: body}, &out)
	if e != nil {
		return Volume{}, e
	}
	if !out.valid() {
		return Volume{}, exit.Named(exit.Internal, "hub.volume_invalid",
			"the hub warmed a volume but answered an incomplete row").
			WithRemedy("read the hub log for the warm-volume request")
	}
	return out, nil
}

// DropVolume deletes one disposable cache volume and its cached copies.
func (c *Client) DropVolume(ctx context.Context, id, reason string) *exit.Error {
	if !strings.HasPrefix(id, "pvl-") {
		return exit.Usagef("volume ids begin with pvl-; %q is not one", id)
	}
	return c.do(ctx, call{method: http.MethodDelete, path: "/v1/volumes/" + url.PathEscape(id),
		auth: true, reason: reason}, nil)
}
