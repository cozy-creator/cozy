package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
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
	// One command resolves its rental once, however many of its steps ask.
	if id, ok := ctx.rentalSelections[name]; ok {
		return id, nil
	}
	id, problem := selectedRental(ctx, store, name)
	if problem != nil {
		return "", problem
	}
	if ctx.rentalSelections == nil {
		ctx.rentalSelections = map[string]string{}
	}
	ctx.rentalSelections[name] = id
	// A named rental is where the call goes. What it cannot run, its machine refuses.
	return id, nil
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
