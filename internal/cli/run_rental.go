package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func requestedRental(ctx *Context) (string, *exit.Error) {
	name := strings.TrimSpace(ctx.Inv.Value("--rental"))
	if name == "" {
		return "", nil
	}
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return "", problem
	}
	defer store.Close()
	if key := ctx.Inv.Value("--idempotency-key"); key != "" {
		prior, problem := store.RequestByIdempotencyKey(key)
		if problem != nil {
			return "", problem
		}
		if prior != nil && prior.RequestedRental != "" && (name == prior.RequestedRental || strings.EqualFold(name, prior.Machine)) {
			return prior.RequestedRental, nil
		}
	}
	selected, problem := rental.Resolve(store, name)
	if problem != nil {
		return "", problem
	}
	if selected.Row == nil || records.RentalTerminalState(selected.Row.State) {
		return "", exit.Named(exit.NotFound, "rental.selection_unavailable", "--rental=%s does not name an existing usable rental", name)
	}
	// A named rental is where the call goes. What it cannot run, its machine refuses.
	return selected.RentalID, nil
}

func rentalArgument(value *string) (string, *exit.Error) {
	if value == nil {
		return "", nil
	}
	name := strings.TrimSpace(*value)
	if name == "" {
		return "", exit.Usagef("--rental requires an existing rental name or id; use --rental-only for automatic allocation")
	}
	return name, nil
}
