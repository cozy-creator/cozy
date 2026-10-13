package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
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
	// Replaying a durable request observes its accepted identity; it is not a
	// fresh rental selection and must still work after that rental has ended.
	if key := ctx.Inv.Value("--idempotency-key"); key != "" {
		prior, problem := store.RequestByIdempotencyKey(key)
		if problem != nil {
			return "", problem
		}
		if prior != nil && prior.RequestedRental != "" && (name == prior.RequestedRental || strings.EqualFold(name, prior.Machine)) {
			return prior.RequestedRental, nil
		}
	}
	selected, problem := resolveLiveRental(ctx, store, name)
	if problem != nil {
		return "", problem
	}
	if hub.RentalAbsent(selected.Rental.State) || selected.Rental.State == hub.RentalReleaseRequested {
		return "", exit.Named(exit.NotFound, "rental.selection_unavailable", "--rental=%s does not name an existing usable rental", name)
	}
	// A named rental is where the call goes. What it cannot run, its machine refuses.
	return selected.Rental.ID, nil
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
