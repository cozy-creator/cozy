package hub

import (
	"context"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy/internal/exit"
)

// AuthAPI is where the Hub serves AuthKit's versioned JSON API.
const AuthAPI = "/v1/auth/v1"

// RevokeDeviceKey revokes one machine key server-side; AuthKit requires a
// recent sign-in. Any 2xx answer, with or without a body, is the
// confirmation; a refusal arrives as a typed error.
func (c *Client) RevokeDeviceKey(ctx context.Context, id string) *exit.Error {
	return c.do(ctx, call{
		method: http.MethodDelete,
		path:   AuthAPI + "/me/sign-in-keys/" + url.PathEscape(id),
		auth:   true,
	}, nil)
}

// RevokeOtherDeviceKeys keeps the token's current machine and revokes every
// other key on its account. AuthKit requires the token to carry email proof.
func (c *Client) RevokeOtherDeviceKeys(ctx context.Context) *exit.Error {
	return c.do(ctx, call{
		method: http.MethodDelete,
		path:   AuthAPI + "/device-keys",
		auth:   true,
	}, nil)
}
