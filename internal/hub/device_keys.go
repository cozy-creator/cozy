package hub

import (
	"context"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy/internal/exit"
)

// RevokeDeviceKey confirms server-side revocation of one machine key.
func (c *Client) RevokeDeviceKey(ctx context.Context, id string) *exit.Error {
	var answer struct {
		OK bool `json:"ok"`
	}
	problem := c.do(ctx, call{
		method: http.MethodDelete,
		path:   "/v1/auth/device-keys/" + url.PathEscape(id),
		auth:   true,
	}, &answer)
	if problem == nil && !answer.OK {
		return exit.Internalf("Tensorhub did not confirm machine logout")
	}
	return problem
}

// RevokeOtherDeviceKeys keeps the token's current machine and revokes every
// other key on its account. AuthKit requires the token to carry email proof.
func (c *Client) RevokeOtherDeviceKeys(ctx context.Context) *exit.Error {
	var answer struct {
		OK bool `json:"ok"`
	}
	problem := c.do(ctx, call{
		method: http.MethodPost,
		path:   "/v1/auth/device-keys/revoke-others",
		body:   map[string]any{},
		auth:   true,
	}, &answer)
	if problem == nil && !answer.OK {
		return exit.Internalf("Tensorhub did not confirm revocation of the other machines")
	}
	return problem
}
