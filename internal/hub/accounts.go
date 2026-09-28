package hub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/url"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Account is the one public Tensorhub namespace owned by the authenticated user.
type Account struct {
	Name string `json:"name"`
}

// CurrentAccount is the authenticated user's immutable Tensorhub account name. It is read
// once per credential and kept beside it: a later command reads no Hub. Another login or
// operator token is another credential, and asks once.
func (c *Client) CurrentAccount(ctx context.Context) (Account, *exit.Error) {
	var out Account
	kept := c.accountPath()
	if loadRelease(kept, &out) && resourceSlug.MatchString(out.Name) {
		return out, nil
	}
	problem := c.do(ctx, call{
		method: http.MethodGet, path: "/v1/accounts/current", auth: true,
	}, &out)
	if problem == nil && !resourceSlug.MatchString(out.Name) {
		problem = exit.Named(exit.Internal, "account.current_invalid",
			"Tensorhub returned an invalid current account name")
	}
	if problem == nil {
		storeRelease(kept, out)
	}
	return out, problem
}

// accountPath keeps the account of this client's credential: its machine key, else its
// operator token. With neither there is no account to keep.
func (c *Client) accountPath() string {
	identity := ""
	if key, ok := c.tokens.(interface{ Identity() string }); ok {
		identity = key.Identity()
	}
	if identity == "" && c.token.Present() {
		identity = "token:" + c.token.Digest()
	}
	if c.releases == "" || identity == "" {
		return ""
	}
	name := sha256.Sum256([]byte(c.base + "\x00" + identity))
	return filepath.Join(filepath.Dir(c.releases), "auth", "accounts", hex.EncodeToString(name[:16])+".json")
}

// RegisterAccount gives the authenticated user its one immutable account name.
func (c *Client) RegisterAccount(ctx context.Context, name string) (Account, *exit.Error) {
	var out Account
	name = CanonicalName(name)
	if !resourceSlug.MatchString(name) {
		return out, exit.Usagef("%q is not a Tensorhub account name", name).
			WithRemedy("use letters, digits, dots, underscores, or hyphens")
	}
	problem := c.do(ctx, call{
		method: http.MethodPut, path: "/v1/accounts/" + url.PathEscape(name),
		auth: true, body: struct{}{},
	}, &out)
	if problem == nil && out.Name != name {
		problem = exit.Named(exit.Internal, "account.registration_invalid",
			"Tensorhub registered account %q, not %q", out.Name, name)
	}
	if problem == nil {
		storeRelease(c.accountPath(), out)
	}
	return out, problem
}
