package cli

import (
	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/publication"
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
	// An operator token is a fallback for unenrolled automation only. Once this
	// machine has a user key, that identity wins even if stale operator config
	// remains on disk.
	auth := ctx.AccountAuth
	if auth == nil {
		auth = accountauth.New(ctx.Cfg)
	}
	c := client(ctx)
	hctx, cancel := hub.Context()
	defer cancel()
	_, problem := publication.Authorize(hctx, c, ref, ctx.Cfg.HubToken.Present() && !auth.CredentialPresent())
	return c, problem
}
