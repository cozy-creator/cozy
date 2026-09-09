package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func requestedRental(ctx *Context, target Target, function string) (string, *exit.Error) {
	name := strings.TrimSpace(ctx.Inv.Value("--rental"))
	if name == "" {
		return "", nil
	}
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return "", problem
	}
	defer store.Close()
	selected, problem := rental.Resolve(store, name)
	if problem != nil {
		return "", problem
	}
	if selected.Row == nil || records.RentalTerminalState(selected.Row.State) {
		return "", exit.Named(exit.NotFound, "rental.selection_unavailable", "--rental=%s does not name an existing usable rental", name)
	}
	constraints := releaseConstraints(ctx, records.Request{Package: target.Package, Release: target.Release, InstallID: target.InstallID, Entrypoint: function})
	if problem := rentalCompatibility(ctx, selected.RentalID, constraints); problem != nil {
		return "", problem
	}
	return selected.RentalID, nil
}
