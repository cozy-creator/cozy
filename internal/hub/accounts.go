package hub

import (
	"context"
	"net/http"
	"net/url"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Account is the one public Tensorhub namespace owned by the authenticated user.
type Account struct {
	Name string `json:"name"`
}

// CurrentAccount reads the authenticated user's immutable Tensorhub account name.
func (c *Client) CurrentAccount(ctx context.Context) (Account, *exit.Error) {
	var out Account
	problem := c.do(ctx, call{
		method: http.MethodGet, path: "/v1/accounts/current", auth: true, strict: true,
	}, &out)
	if problem == nil && !resourceSlug.MatchString(out.Name) {
		problem = exit.Named(exit.Internal, "account.current_invalid",
			"Tensorhub returned an invalid current account name")
	}
	return out, problem
}

// RegisterAccount gives the authenticated user its one immutable account name.
func (c *Client) RegisterAccount(ctx context.Context, name string) (Account, *exit.Error) {
	var out Account
	if !resourceSlug.MatchString(name) {
		return out, exit.Usagef("%q is not a Tensorhub account name", name).
			WithRemedy("use lowercase letters, digits, dots, underscores, or hyphens")
	}
	problem := c.do(ctx, call{
		method: http.MethodPut, path: "/v1/accounts/" + url.PathEscape(name),
		auth: true, body: struct{}{}, strict: true,
	}, &out)
	if problem == nil && out.Name != name {
		problem = exit.Named(exit.Internal, "account.registration_invalid",
			"Tensorhub registered account %q, not %q", out.Name, name)
	}
	return out, problem
}
