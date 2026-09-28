package hub

import (
	"context"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy/internal/exit"
)

// RevokeDeviceKey revokes one machine key server-side. Any 2xx answer, with or
// without a body, is the confirmation; a refusal arrives as a typed error.
func (c *Client) RevokeDeviceKey(ctx context.Context, id string) *exit.Error {
	return c.do(ctx, call{
		method: http.MethodDelete,
		path:   "/v1/auth/device-keys/" + url.PathEscape(id),
		auth:   true,
	}, nil)
}

// RevokeOtherDeviceKeys keeps the token's current machine and revokes every
// other key on its account. AuthKit requires the token to carry email proof.
func (c *Client) RevokeOtherDeviceKeys(ctx context.Context) *exit.Error {
	return c.do(ctx, call{
		method: http.MethodPost,
		path:   "/v1/auth/device-keys/revoke-others",
		body:   map[string]any{},
		auth:   true,
	}, nil)
}
