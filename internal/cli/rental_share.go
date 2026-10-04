package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/rental"
)

func handleRentalShare(ctx *Context) *exit.Error   { return shareRental(ctx, true) }
func handleRentalUnshare(ctx *Context) *exit.Error { return shareRental(ctx, false) }

// shareRental asks the rental's Hub to add or remove a member. The machine admits the member's
// device keys from its next lease, with no restart, and drops them at the lease after removal.
func shareRental(ctx *Context, share bool) *exit.Error {
	subject, account := strings.TrimSpace(ctx.Inv.Args[0]), strings.TrimSpace(ctx.Inv.Args[1])
	_, st, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer st.Close()
	known, problem := rental.Resolve(st, subject)
	if problem != nil {
		return problem
	}
	if known.RentalID == "" {
		return exit.Named(exit.Conflict, "rental.identity_unknown", "rental %s has no Tensorhub id yet", subject)
	}
	ctx = ctx.forHub(known.Hub)
	hctx, cancel := hub.Context()
	members, problem := client(ctx).ShareRental(hctx, known.RentalID, account, share)
	cancel()
	if problem != nil {
		return problem
	}
	status := "shared"
	if !share {
		status = "unshared"
	}
	return emit(ctx, output.Record{Fields: []output.Field{
		{K: "rental", V: known.RentalID}, {K: "status", V: status}, {K: "account", V: account},
		{K: "users", V: members},
	}, Notes: []string{"the machine applies the change at its next lease of authorized keys"},
		Next: []string{"cozy rental show " + known.RentalID}})
}
