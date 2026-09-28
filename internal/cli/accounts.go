package cli

import (
	"sync"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/packagepublish"
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

// commandNamespace answers the caller's account on this command's Tensorhub. It asks the
// Hub only when called: a project with no account dependencies or org-relative defaults
// never needs a signed-in caller.
func commandNamespace(ctx *Context) packagepublish.NamespaceSource {
	if ctx.namespace != nil {
		return ctx.namespace
	}
	var once sync.Once
	var namespace packagepublish.Namespace
	var problem *exit.Error
	ctx.namespace = func() (packagepublish.Namespace, *exit.Error) {
		once.Do(func() {
			var c *hub.Client
			var account hub.Account
			c, account, problem = publicationAccount(ctx)
			if problem != nil {
				problem = problem.WithRemedy("sign in on %s: this project names your account's packages or models", ctx.Cfg.HubURL)
				return
			}
			namespace = packagepublish.Namespace{Hub: c.Base(), Account: account.Name}
		})
		return namespace, problem
	}
	return ctx.namespace
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
