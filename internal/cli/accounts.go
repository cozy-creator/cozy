package cli

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

// publicationAccount binds a publication to the logged-in user's one Tensorhub account.
func publicationAccount(ctx *Context) (*hub.Client, hub.Account, *exit.Error) {
	c := client(ctx)
	hctx, cancel := hub.Context()
	account, problem := c.CurrentAccount(hctx)
	cancel()
	return c, account, problem
}

func ownedPublication(ctx *Context, ref hub.Ref) (*hub.Client, *exit.Error) {
	// An explicitly configured token may be an operator or automation credential
	// with authority over more than one account. Tensorhub is the authority for
	// that scope; an end-user account lookup would incorrectly narrow it here.
	if ctx.Cfg.HubToken.Present() {
		return client(ctx), nil
	}
	c, account, problem := publicationAccount(ctx)
	if problem != nil {
		return nil, problem
	}
	if ref.Org != account.Name {
		return nil, exit.Usagef("cannot publish %s while logged in as Tensorhub account %s",
			ref.String(), account.Name).
			WithRemedy("publish as %s/%s", account.Name, ref.Name)
	}
	return c, nil
}
