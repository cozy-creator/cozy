package publication

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
)

// Authorize checks current account authority immediately before a new effect.
// Operator mode is selected only when Creator has no enrolled machine key.
// Returned identity is a frozen namespace fact, never a bearer credential.
func Authorize(ctx context.Context, client *hub.Client, ref hub.Ref, operator bool) (string, *exit.Error) {
	if ref.Org == "local" {
		return "", exit.New(exit.Validation, "private local aliases cannot be published")
	}
	if operator {
		return ref.Org, nil
	}
	account, problem := client.CurrentAccount(ctx)
	if problem != nil {
		return "", problem
	}
	if ref.Org != account.Name {
		return "", exit.Usagef("cannot publish %s while logged in as Tensorhub account %s", ref.String(), account.Name)
	}
	return account.Name, nil
}
